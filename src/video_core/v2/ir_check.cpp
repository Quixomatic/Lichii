// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>

#include "common/logging.h"
#include "video_core/v2/ir_check.h"

namespace VideoCommon::V2::IrCheck {

namespace {

struct Expected {
    Ir::PredictedDraw draw;
    u64 record;
    u64 fingerprint;
    u32 method;
    bool has_fingerprint;
};

constexpr size_t QueueSize = 1024;
constexpr u64 MissingAge = 256;

struct State {
    std::array<Expected, QueueSize> queue{};
    size_t head{}; ///< Oldest
    size_t size{};
    u64 records{};
    // Dispatch window.
    bool window{};
    u64 window_records{};
    Ir::DispatchBody window_body{};
    u32 window_count{};
    Counters counters{};
    u32 draw_logs{};
    u32 dispatch_logs{};
};

State g_state;

const char* KindName(Ir::DrawKind kind) {
    const size_t index = static_cast<size_t>(kind);
    return index < std::size(Ir::draw_kind_names) ? Ir::draw_kind_names[index] : "?";
}

bool SameDraw(const Ir::PredictedDraw& a, const Ir::PredictedDraw& b) {
    if (a.kind != b.kind || a.topology != b.topology) {
        return false;
    }
    switch (a.kind) {
    case Ir::DrawKind::Indirect:
    case Ir::DrawKind::IndexedIndirect:
    case Ir::DrawKind::Texture:
        return true;
    case Ir::DrawKind::Clear:
        return a.first == b.first;
    default:
        return a.first == b.first && a.count == b.count && a.instances == b.instances &&
               a.base_vertex == b.base_vertex && a.base_instance == b.base_instance;
    }
}

void LogDraw(const char* what, const Ir::PredictedDraw& draw, u32 method) {
    if (g_state.draw_logs++ >= 32) {
        return;
    }
    LOG_WARNING(HW_GPU,
                "V2 IR draw check: {} {} topology {} first {} count {} instances {} base vertex "
                "{} base instance {} (method {:#x})",
                what, KindName(draw.kind), draw.topology, draw.first, draw.count, draw.instances,
                draw.base_vertex, draw.base_instance, method);
}

void DropMissing() {
    State& s = g_state;
    while (s.size != 0 && s.records - s.queue[s.head].record > MissingAge) {
        const Expected& expected = s.queue[s.head];
        LogDraw("predicted, never performed:", expected.draw, expected.method);
        ++s.counters.draws_missing;
        s.head = (s.head + 1) % QueueSize;
        --s.size;
    }
}

void CloseWindow() {
    State& s = g_state;
    if (!s.window) {
        return;
    }
    s.window = false;
    const Ir::DispatchBody& body = s.window_body;
    if (s.window_count < body.min_dispatches ||
        (body.max_dispatches != ~0U && s.window_count > body.max_dispatches)) {
        ++s.counters.dispatch_count_mismatches;
        if (s.dispatch_logs++ < 32) {
            LOG_WARNING(HW_GPU,
                        "V2 IR dispatch check: method {:#x} = {:#x} (QMD {:#x}) dispatched {} "
                        "times, the front predicted {}..{}",
                        body.method, body.argument, body.qmd_address, s.window_count,
                        body.min_dispatches, body.max_dispatches);
        }
    }
}

void Summary() {
    const Counters& c = g_state.counters;
    LOG_INFO(HW_GPU,
             "V2 IR draw check: {} draws matched, {} mismatched, {} unexpected, {} missing; "
             "fingerprints {} compared, {} differ; dispatches {} in {} windows, {} unexpected, "
             "{} count mismatches",
             c.draws_matched, c.draws_mismatched, c.draws_unexpected, c.draws_missing,
             c.fingerprints_compared, c.fingerprints_mismatched, c.dispatches,
             c.dispatch_windows, c.dispatches_unexpected, c.dispatch_count_mismatches);
}

} // Anonymous namespace

void Start() {
    g_state = State{};
    g_active = true;
}

void Stop() {
    if (g_active) {
        CloseWindow();
        Summary();
    }
    g_active = false;
}

void OnRecord(Ir::Kind kind) {
    State& s = g_state;
    if (kind == Ir::Kind::Engine) {
        return;
    }
    ++s.records;
    if (s.window && ++s.window_records >= 2) {
        CloseWindow();
    }
    if (s.size != 0) [[unlikely]] {
        DropMissing();
    }
}

void NoteDrawRecord(const Ir::DrawBody& body) {
    State& s = g_state;
    for (u32 i = 0; i < body.num_draws && i < Ir::MaxPredictedDraws; ++i) {
        if (s.size == QueueSize) {
            // Nothing took the oldest: it is missing.
            LogDraw("predicted, never performed:", s.queue[s.head].draw, s.queue[s.head].method);
            ++s.counters.draws_missing;
            s.head = (s.head + 1) % QueueSize;
            --s.size;
        }
        s.queue[(s.head + s.size) % QueueSize] = Expected{
            .draw = body.draws[i],
            .record = s.records,
            .fingerprint = body.fingerprint,
            .method = body.method,
            .has_fingerprint = (body.flags & Ir::DrawHasGroups) != 0,
        };
        ++s.size;
    }
}

void NoteDispatchRecord(const Ir::DispatchBody& body) {
    State& s = g_state;
    CloseWindow();
    s.window = true;
    s.window_records = 0;
    s.window_body = body;
    s.window_count = 0;
    ++s.counters.dispatch_windows;
}

void NoteDraw(const Ir::PredictedDraw& draw, const GroupTracker* groups) {
    State& s = g_state;
    if (s.size == 0) {
        ++s.counters.draws_unexpected;
        LogDraw("performed without a prediction:", draw, 0);
    } else {
        const Expected expected = s.queue[s.head];
        s.head = (s.head + 1) % QueueSize;
        --s.size;
        if (SameDraw(expected.draw, draw)) {
            ++s.counters.draws_matched;
        } else {
            ++s.counters.draws_mismatched;
            LogDraw("predicted", expected.draw, expected.method);
            LogDraw("  performed", draw, expected.method);
        }
        if (expected.has_fingerprint && groups != nullptr && groups->valid) {
            ++s.counters.fingerprints_compared;
            if (groups->Fingerprint(HashedGroups) != expected.fingerprint) {
                ++s.counters.fingerprints_mismatched;
                if (s.draw_logs++ < 32) {
                    LOG_WARNING(HW_GPU,
                                "V2 IR draw check: state fingerprint differs from the front's "
                                "(method {:#x})",
                                expected.method);
                }
            }
        }
    }
    const u64 checked = s.counters.draws_matched + s.counters.draws_mismatched +
                        s.counters.draws_unexpected;
    if ((checked & 0xffff) == 0) {
        Summary();
    }
}

void NoteDispatch(u64 qmd_address) {
    State& s = g_state;
    ++s.counters.dispatches;
    if (!s.window) {
        ++s.counters.dispatches_unexpected;
        if (s.dispatch_logs++ < 32) {
            LOG_WARNING(HW_GPU,
                        "V2 IR dispatch check: dispatch of QMD {:#x} without a front record",
                        qmd_address);
        }
        return;
    }
    ++s.window_count;
}

Counters GetCounters() {
    return g_state.counters;
}

size_t PendingDraws() {
    return g_state.size;
}

} // namespace VideoCommon::V2::IrCheck
