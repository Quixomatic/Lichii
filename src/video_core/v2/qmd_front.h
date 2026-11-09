// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>

#include "common/common_types.h"
#include "video_core/native_qmd.h"

// P5b (EDEN_NX2_V2_QMDSCHED): the split front's model of the back's native QMD scheduler (#805's
// FrontQmdView). It runs the same dependency logic (NativeQmdScheduler::SignalOn) on its own
// table, fed the same signals in the same order, so it knows on the front how many QMDs each
// signal launches (exact Dispatch record counts) and which QMDs they are (bounded hazard table
// writes). A QMD the front can't read for sure (bytes a queued write may change) makes the model
// unknown until the next drain, where it copies the back's table (the back is idle then).
namespace VideoCommon::V2::QmdFront {

using Scheduler = Tegra::NativeQmdScheduler;

struct SignalResult {
    bool valid = false;
    u32 dispatches = 0;
};

enum class ReadResult : u8 {
    Ok,
    Unmapped, ///< the back sees the same: an error that stops the chain there
    Unknown,  ///< the front can't know the words: the model becomes unknown
};

class Model {
public:
    /// Runs one QMD signal (method 0xae/0xaf/0xb0 and its argument, already filtered as the back
    /// filters it). `read(address, descriptor&) -> ReadResult`, `mapped(address, size) -> bool`,
    /// `on_dispatch(address, const Descriptor&)`, `on_release(address, payload, flags)`.
    template <typename Read, typename Mapped, typename OnDispatch, typename OnRelease>
    SignalResult Signal(u64 qmd_address, u32 argument, Read&& read, Mapped&& mapped,
                        OnDispatch&& on_dispatch, OnRelease&& on_release) {
        if (!valid) {
            return {};
        }
        bool unknown = false;
        u32 dispatches = 0;
        (void)runner.SignalOn(
            table, qmd_address, argument,
            [&](u64 address, Scheduler::Descriptor& descriptor) {
                switch (read(address, descriptor)) {
                case ReadResult::Ok:
                    return true;
                case ReadResult::Unmapped:
                    return false;
                case ReadResult::Unknown:
                    unknown = true;
                    return false;
                }
                return false;
            },
            [&](u64 address, u32 payload, u32 flags) {
                const bool four_words = (flags >> 30) == 0;
                if (!mapped(address, four_words ? 16 : 4)) {
                    return false;
                }
                on_release(address, payload, flags);
                return true;
            },
            [&](u64 address, const Scheduler::Descriptor& qmd) {
                ++dispatches;
                on_dispatch(address, qmd);
                return true;
            });
        if (unknown) {
            valid = false;
            ++invalidations;
            return {};
        }
        return {true, dispatches};
    }

    /// The front lost track (a read it can't trust): unknown until Resync.
    void Invalidate() noexcept {
        valid = false;
        ++invalidations;
    }

    /// Copies the back's table (call only while the back is idle: a drain).
    void Resync(const Scheduler::Table& back) {
        table = back;
        valid = true;
        ++resyncs;
    }

    [[nodiscard]] bool Valid() const noexcept {
        return valid;
    }
    [[nodiscard]] const Scheduler::Table& Table() const noexcept {
        return table;
    }
    [[nodiscard]] u64 Invalidations() const noexcept {
        return invalidations;
    }
    [[nodiscard]] u64 Resyncs() const noexcept {
        return resyncs;
    }

private:
    Scheduler runner;
    Scheduler::Table table;
    bool valid = true;
    u64 invalidations = 0;
    u64 resyncs = 0;
};

/// Inline QMDs (method 0x107) the front sent but the back may not have stored yet: the front
/// forwards their words to its own reads (store forwarding for 256-byte QMDs).
class InlineQmds {
public:
    static constexpr size_t Capacity = 64;

    void Note(u64 address, const Scheduler::Descriptor& words) {
        for (auto& entry : entries) {
            if (entry.used && entry.address == address) {
                entry.words = words;
                return;
            }
        }
        Entry& entry = entries[next];
        next = (next + 1) % Capacity;
        entry = {address, words, true};
    }

    [[nodiscard]] const Scheduler::Descriptor* Find(u64 address) const {
        for (const auto& entry : entries) {
            if (entry.used && entry.address == address) {
                return &entry.words;
            }
        }
        return nullptr;
    }

    void Clear() {
        entries = {};
        next = 0;
    }

private:
    struct Entry {
        u64 address = 0;
        Scheduler::Descriptor words{};
        bool used = false;
    };
    std::array<Entry, Capacity> entries{};
    size_t next = 0;
};

} // namespace VideoCommon::V2::QmdFront
