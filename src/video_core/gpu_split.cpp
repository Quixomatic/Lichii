// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstdlib>
#include <thread>

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <cstring>
#include <string>

#include "common/dkb_cpu_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/v2/resolve/split_resolve.h"
#include "video_core/v2/resolve/state_view.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/qmd_front.h"
#include "video_core/v2/ir_check.h"
#include "video_core/v2/ir_exec.h"
#include "video_core/v2/v2_stats.h"
#include "video_core/v2/versions.h"

namespace VideoCommon {

namespace {

void Pause() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#else
    std::this_thread::yield();
#endif
}

/// How long the back's old idle spin (8192 pauses) takes here, measured once. With
/// EDEN_NX2_V2_SPIN the back still runs its idle hook (the deferred submit) after this long, so
/// the spin window changes when it sleeps, not when it submits.
u64 OldIdleSpinNs() {
    static const u64 ns = [] {
        const u64 start = Nx2Profile::Now();
        for (u32 i = 0; i < 8192; ++i) {
            Pause();
        }
        return Nx2Profile::Now() - start;
    }();
    return ns;
}

[[maybe_unused]] std::atomic<u32> g_back_thread_id{};

#if defined(_MSC_VER) && !defined(__clang__)
#define SPLIT_FORCE_INLINE __forceinline
#else
#define SPLIT_FORCE_INLINE inline __attribute__((always_inline))
#endif

} // Anonymous namespace

void GpuSplit::CheckOwner(const char* where) {
#ifdef _WIN32
    // Also while the back is idle: its idle hook submits (Scheduler::FlushDeferred).
    if (!instance.back_running.load(std::memory_order_acquire)) {
        return;
    }
    const DWORD thread = GetCurrentThreadId();
    if (thread == g_back_thread_id.load(std::memory_order_relaxed)) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1) >= 12) {
        return;
    }
    std::string stack;
    void* frames[40];
    const USHORT count = RtlCaptureStackBackTrace(1, 40, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE module{};
        char name[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               static_cast<LPCSTR>(frames[i]), &module)) {
            GetModuleFileNameA(module, name, sizeof(name));
        }
        const char* base_name = std::strrchr(name, '\\');
        stack += fmt::format(" {}+{:#x}", base_name ? base_name + 1 : name,
                             reinterpret_cast<uintptr_t>(frames[i]) -
                                 reinterpret_cast<uintptr_t>(module));
    }
    LOG_ERROR(HW_GPU, "GPU split owner check: {} on thread {} while the back runs; stack:{}",
              where, thread, stack);
#else
    (void)where;
#endif
}

GpuSplit GpuSplit::instance;

void GpuSplit::MaybeStart(Core::System& system_) {
    static const char mode = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT");
        return value != nullptr ? value[0] : '2';
    }();
    const bool wanted = mode == '1' || mode == '2';
    if (!wanted || !Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        return;
    }
    // The front reads command lists straight from guest memory; a read that flushes the caches
    // first (safe DMA, accurate GPU) would touch back-owned state. The settings can change
    // mid-game: hand the engines back to the front (and start again when they allow it).
    const bool safe_reads = Settings::IsDMALevelDefault() ? Settings::IsGPULevelHigh()
                                                          : Settings::IsDMALevelSafe();
    const bool allowed = !safe_reads && !Settings::IsGPULevelHigh() &&
                         Settings::values.renderer_backend.GetValue() == Settings::RendererBackend::Vulkan;
    if (active && !allowed) {
        Drain();
        Stop();
        LOG_INFO(HW_GPU, "GPU thread split: stopped, the accuracy settings need the caches on the front");
        return;
    }
    if (active || !allowed) {
        return;
    }
    system = &system_;
    check_owner = std::getenv("EDEN_NX2_SPLIT_CHECK") != nullptr;
    ring = std::make_unique<u64[]>(RingWords);
    back_thread = std::jthread([this](std::stop_token stop_token) { BackLoop(stop_token); });
    active = true;
    ++starts;
    const char* jit = std::getenv("EDEN_NX2_MME_JIT");
    front_macros = mode == '2' && (jit == nullptr || jit[0] != '0');
    // Fixed until the next start: the back decodes both encodings, the front writes one.
    ir = V2::On(V2::Switch::Ir);
    ir_verify = V2::Verify(V2::Switch::Ir);
    if (ir) {
        AllocateIr();
    }
    draw_records = ir && front_macros && V2::On(V2::Switch::IrDraws);
    draw_verify = draw_records && V2::Verify(V2::Switch::IrDraws);
    full_draw_records = draw_records && (draw_verify || V2::On(V2::Switch::Parallel));
    if (draw_verify) {
        V2::IrCheck::Start();
    }
    StartParallel();
    V2::Resolve::MultiFront::SetActive(true);
    // P5a: with COMMIT=1 lock-free, the hook also serves work handed to the caches' owner.
    if (V2::On(V2::Switch::CpuLane) ||
        (V2::Get(V2::Switch::Commit) == V2::Mode::On && !V2::On(V2::Switch::CommitLocked))) {
        SetCpuLaneHook([](void* gpu) { static_cast<Tegra::GPU*>(gpu)->TickCpuLane(); },
                       &system->GPU());
    }
    acquire_mode = V2::Get(V2::Switch::Acquire);
    hazard_mode = V2::Get(V2::Switch::Hazard);
    tracks_writes = acquire_mode != V2::Mode::Off || hazard_mode != V2::Mode::Off;
    feed_idle_back = acquire_mode != V2::Mode::Off;
    hazards.Clear();
    unlanded.Clear();
    V2::Landing::landed.store(0, std::memory_order_relaxed);
    // AnyPS5's hand-off spin (HandoffSpin.hpp): spin before a kernel wait only with more than 8
    // logical CPUs, where the waiting thread does not take the other one's core.
    spin_window = V2::On(V2::Switch::Spin);
    if (spin_window) {
        u64 microseconds = 150;
        if (const char* value = std::getenv("EDEN_NX2_V2_SPIN_US")) {
            microseconds = std::strtoull(value, nullptr, 10);
        }
        spin_window_ns = std::thread::hardware_concurrency() > 8 ? microseconds * 1000 : 0;
        idle_hook_ns = OldIdleSpinNs();
        LOG_INFO(HW_GPU,
                 "GPU thread split: hand-off spin window {} us ({} logical CPUs), back idle "
                 "hook after {} us",
                 spin_window_ns / 1000, std::thread::hardware_concurrency(), idle_hook_ns / 1000);
    }
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}{}{}",
             front_macros ? ", macros on the first" : "",
             ir ? (ir_verify ? ", typed IR (verified)" : ", typed IR") : "",
             draw_records ? (draw_verify ? ", draw records (checked)" : ", draw records") : "");
}

void GpuSplit::Stop() {
    if (!back_thread.joinable()) {
        return;
    }
    back_thread.request_stop();
    wake.fetch_add(1, std::memory_order_seq_cst);
    wake.notify_one();
    back_thread.join();
    active = false;
    // The shards read the ring: they stop before it goes away.
    delete parallel;
    parallel = nullptr;
    V2::Resolve::MultiFront::SetActive(false);
    // The split is static and outlives the emulation session. The next session's back thread
    // starts reading at 0: with the old positions kept it skipped through the new, empty ring a
    // wrap at a time and never reached the old write position (the next game's first GPU work
    // never ran, it waited on its syncpoint forever), and the engines it held were freed.
    ring.reset();
    write_position = 0;
    last_writer_position = 0;
    last_sync_position = 0;
    published_local = 0;
    consumed_cache = 0;
    publish_every = PublishEvery;
    front_engine = nullptr;
    back_engine = nullptr;
    front_macros = false;
    ir = false;
    ir_verify = false;
    draw_records = false;
    draw_verify = false;
    full_draw_records = false;
    V2::IrCheck::Stop();
    ir_sequence = 0;
    ir_run = nullptr;
    ir_run_words = 0;
    ir_run_index = 0;
    unpublished_words = 0;
    sent_words = 0;
    arena.Release();
    objects.reset();
    tracks_writes = false;
    feed_idle_back = false;
    spin_window = false;
    acquire_mode = V2::Mode::Off;
    hazard_mode = V2::Mode::Off;
    hazards.Clear();
    unlanded.Clear();
    V2::Landing::landed.store(0, std::memory_order_relaxed);
    SetCpuLaneHook(nullptr, nullptr);
    cpu_lane_requested.store(false, std::memory_order_relaxed);
    back_idle_at.store(~u64{0}, std::memory_order_relaxed);
    back_sequence = 0;
    consumed_sequence.store(0, std::memory_order_relaxed);
    published.store(0, std::memory_order_relaxed);
    consumed.store(0, std::memory_order_relaxed);
    back_waiting.store(false, std::memory_order_relaxed);
    front_waiting.store(false, std::memory_order_relaxed);
}

u64* GpuSplit::ReserveSlow(size_t words) {
    size_t index = write_position & RingMask;
    if (index + words > RingWords) {
        // Entries never wrap: skip to the start of the ring.
        const size_t rest = RingWords - index;
        WaitForSpace(rest);
        ring[index] = static_cast<u64>(Kind::Wrap);
        write_position += rest;
        index = 0;
    }
    WaitForSpace(words);
    return &ring[index];
}

void GpuSplit::WaitForSpace(size_t words) {
    if (write_position + words - consumed_cache <= RingWords) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (write_position + words - consumed_cache <= RingWords) {
            return;
        }
        if (spins < 4096) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
}

void GpuSplit::Publish() {
    if (ir_run != nullptr) {
        IrCloseRun();
    }
    if (published_local == write_position) {
        return;
    }
    published_local = write_position;
    unpublished_words = 0;
    publish_every = std::min<size_t>(publish_every * 2, PublishEvery);
    published.store(write_position, std::memory_order_seq_cst);
    if (back_waiting.load(std::memory_order_seq_cst)) {
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
    if (parallel != nullptr) [[unlikely]] {
        ParallelPublish();
    }
}

void GpuSplit::StartParallel() {
    if (!V2::On(V2::Switch::Parallel)) {
        return;
    }
    // The shards replay the front's table-driven mirror and resolve the draw cache's keys.
    const char* missing = !draw_records ? "EDEN_NX2_V2_IR and EDEN_NX2_V2_IR_DRAWS with front macros"
                          : V2::Get(V2::Switch::RegSem) != V2::Mode::On ? "EDEN_NX2_V2_REGSEM=1"
                          : !V2::On(V2::Switch::DrawCache)              ? "EDEN_NX2_V2_DRAWCACHE"
                                                                       : nullptr;
    if (missing != nullptr) {
        LOG_WARNING(HW_GPU, "EDEN_NX2_V2_PARALLEL needs {}: off", missing);
        return;
    }
    parallel = new V2::Resolve::SplitResolve(*this, ring.get(), &arena);
}

void GpuSplit::ParallelRecord(V2::Ir::Kind kind) {
    parallel->OnRecord(kind);
}

void GpuSplit::ParallelBarrier(u32 op) {
    parallel->OnBarrier(op);
}

void GpuSplit::ParallelPublish() {
    parallel->OnPublish(published_local);
}

void GpuSplit::Drain(DrainSite site) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache == write_position) {
        return;
    }
    Publish();
    Nx2Profile::Scope scope{Nx2Profile::Zone::SplitDrain};
    const bool v2_stats = V2::Stats::Enabled();
    const u64 start = Nx2Profile::Enabled() || v2_stats ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache == write_position) {
            publish_every = 8;
            if (objects && objects->Size() > V2::Ir::ObjectTable::Capacity * 3 / 4) [[unlikely]] {
                // The back ran every record that names an object: start the table over.
                objects->Clear();
            }
            Nx2Profile::AddDrain(site, start);
            if (v2_stats) [[unlikely]] {
                V2::Stats::NoteDrain(site, start);
            }
            V2::Versions::HookEpoch(V2::Versions::EpochReason::Wait);
            if (V2::On(V2::Switch::QmdSched)) [[unlikely]] {
                const bool resync = !V2::QmdFront::FrontModel().Valid();
                V2::QmdFront::OnDrain();
                if (resync) {
                    V2::P5bStats::Add(V2::P5bStats::Counter::QmdResyncs);
                }
            }
            return;
        }
        Backoff(spins, write_position);
    }
}

bool GpuSplit::WaitForWriters() {
    return WaitFor(std::min(last_writer_position, write_position), DrainSite::ListWriters);
}

bool GpuSplit::WaitForSync() {
    return WaitFor(std::min(last_sync_position, write_position), DrainSite::ListSync);
}

bool GpuSplit::WaitFor(u64 target, DrainSite site) {
    consumed_cache = consumed.load(std::memory_order_acquire);
    if (consumed_cache >= target) {
        return false;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    wait_target.store(target, std::memory_order_seq_cst);
    Publish();
    const bool v2_stats = V2::Stats::Enabled();
    const u64 start = Nx2Profile::Enabled() || v2_stats ? Nx2Profile::Ticks() : 0;
    for (u32 spins = 0;; ++spins) {
        consumed_cache = consumed.load(std::memory_order_acquire);
        if (consumed_cache >= target) {
            wait_target.store(~u64{0}, std::memory_order_relaxed);
            Nx2Profile::AddDrain(site, start);
            if (v2_stats) [[unlikely]] {
                V2::Stats::NoteDrain(site, start);
            }
            V2::Versions::HookEpoch(V2::Versions::EpochReason::Wait);
            return true;
        }
        Backoff(spins, target);
    }
}

bool GpuSplit::SpinWindowOpen(u32 spins, u64& start) const {
    // Without spinning: a few hundred pauses for a back that is about to finish, then sleep.
    if (spin_window_ns == 0) {
        return spins < 256;
    }
    if (spins == 0) {
        start = Nx2Profile::Now();
        return true;
    }
    return (spins & 63) != 0 || Nx2Profile::Now() - start < spin_window_ns;
}

void GpuSplit::Backoff(u32 spins, u64 target) {
    if (spin_window) [[unlikely]] {
        if (SpinWindowOpen(spins, front_spin_start)) {
            Pause();
            return;
        }
    } else if (spins < 16384) {
        Pause();
        return;
    }
    if (!back_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
        return;
    }
    // A long wait sleeps until the back stores a new position (port of #161's front_wake): the
    // back checks front_waiting after each store, the front checks the position after setting it.
    front_waiting.store(true, std::memory_order_seq_cst);
    const u32 seen = front_wake.load(std::memory_order_seq_cst);
    if (consumed.load(std::memory_order_seq_cst) < target) {
        front_wake.wait(seen, std::memory_order_seq_cst);
    }
    front_waiting.store(false, std::memory_order_relaxed);
}

void GpuSplit::SwitchEngine(Tegra::Engines::EngineInterface* engine) {
    front_engine = engine;
    if (ir) {
        V2::Ir::ObjectRef ref = IrRef(engine);
        if (ref == V2::Ir::NoObject) [[unlikely]] {
            // The table is full: once the back ran every record, it starts over.
            ir_ref_failed = false;
            Drain();
            objects->Clear();
            ref = IrRef(engine);
        }
        constexpr size_t words = V2::Ir::HeaderWords + V2::Ir::BodyWordsOf<V2::Ir::EngineBody>;
        u64* const record = IrBegin(V2::Ir::Kind::Engine, 1, words, 0);
        const V2::Ir::EngineBody body{ref, 0};
        std::memcpy(record + V2::Ir::HeaderWords, &body, sizeof(body));
        Commit(words);
        return;
    }
    u64* const entry = Reserve(2);
    entry[0] = static_cast<u64>(Kind::Engine);
    entry[1] = reinterpret_cast<u64>(engine);
    Commit(2);
}

void GpuSplit::AllocateIr() {
    if (!arena.Allocated()) {
        arena.Allocate(ArenaWords, ArenaMarks);
    }
    arena.Clear();
    if (!objects) {
        objects = std::make_unique<V2::Ir::ObjectTable>();
    }
    objects->Clear();
}

V2::Ir::ObjectRef GpuSplit::IrRef(const void* object) {
    const u32 index = objects->Intern(object);
    if (index == V2::Ir::ObjectTable::None) [[unlikely]] {
        ir_ref_failed = true;
        return V2::Ir::NoObject;
    }
    return index;
}

u64* GpuSplit::ArenaBegin(size_t words, u64 sequence, u64 position) {
    if (u64* const span = arena.Begin(words, sequence, position)) [[likely]] {
        return span;
    }
    arena.Retire(ExecutedSequence());
    for (;;) {
        if (u64* const span = arena.Begin(words, sequence, position)) {
            return span;
        }
        // Full: wait for the back to run the oldest record with a payload.
        WaitFor(std::min(arena.Oldest().position, write_position), DrainSite::Other);
        arena.Retire(ExecutedSequence());
    }
}

u64* GpuSplit::IrPayloadBegin(V2::Ir::Kind kind, const void* body, size_t body_words,
                              size_t payload_words, u64 flags) {
    const size_t words = V2::Ir::HeaderWords + 1 + body_words;
    u64* const record = IrBegin(kind, body_words, words, flags | V2::Ir::HasPayload);
    u64* const span = ArenaBegin(payload_words, ir_sequence, write_position + words);
    record[V2::Ir::HeaderWords] =
        V2::Ir::PackRef(arena.IndexOf(span), static_cast<u32>(payload_words));
    if (body_words != 0) {
        std::memcpy(record + V2::Ir::HeaderWords + 1, body, body_words * sizeof(u64));
    }
    ir_pending_words = words;
    ir_pending_payload = payload_words;
    return span;
}

void GpuSplit::IrOpenRun(u64 word, const GPUVAddr* segment) {
    const size_t entry_words = segment != nullptr ? 2 : 1;
    constexpr size_t words = V2::Ir::HeaderWords + 1;
    u64* const record = IrBegin(V2::Ir::Kind::StateRun, 0, words, V2::Ir::HasPayload);
    u64* const entries = ArenaBegin(entry_words, ir_sequence, write_position + words);
    entries[0] = word;
    if (segment != nullptr) {
        entries[1] = *segment;
    }
    ir_run_index = arena.IndexOf(entries);
    record[V2::Ir::HeaderWords] = V2::Ir::PackRef(ir_run_index, 0);
    ir_run = record;
    ir_run_words = entry_words;
    Commit(words); // A publish here closes the run with its first entry.
    Advance(entry_words);
}

void GpuSplit::MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                           u32 amount, u32 methods_pending, const GPUVAddr* segment) {
    SelectEngine(engine);
    if (ir) {
        // MethodData records, in pieces as below; the words go to the payload arena.
        while (amount > 0) {
            const u32 piece = std::min(amount, MaxMultiWords);
            const size_t payload = (piece + 1) / 2;
            const V2::Ir::MethodDataBody data{
                .method = method & 0xffff,
                .amount = piece,
                .methods_pending = methods_pending,
                .reserved = 0,
                .segment = segment != nullptr ? *segment : 0,
            };
            u64* const span = IrPayloadBegin(
                V2::Ir::Kind::MethodData, &data, V2::Ir::BodyWordsOf<V2::Ir::MethodDataBody>,
                payload, segment != nullptr ? V2::Ir::HasSegment : 0);
            span[payload - 1] = 0;
            std::memcpy(span, words, size_t{piece} * sizeof(u32));
            IrPayloadCommit();
            words += piece;
            amount -= piece;
            methods_pending -= piece;
        }
        return;
    }
    // Long runs (constant buffer and inline uploads) go in pieces, as when a method's data
    // continues in the next GPFIFO entry.
    while (amount > 0) {
        const u32 piece = std::min(amount, MaxMultiWords);
        const size_t head = segment != nullptr ? 3 : 2;
        const size_t size = head + (piece + 1) / 2;
        u64* const entry = Reserve(size);
        entry[0] = static_cast<u64>(Kind::Multi) | (segment != nullptr ? HasSegment : 0) |
                   (u64{method & 0xffff} << 8) | (u64{piece} << 32);
        entry[1] = methods_pending;
        if (segment != nullptr) {
            entry[2] = *segment;
        }
        entry[size - 1] = 0;
        std::memcpy(entry + head, words, size_t{piece} * sizeof(u32));
        Commit(size);
        words += piece;
        amount -= piece;
        methods_pending -= piece;
    }
}

SPLIT_FORCE_INLINE size_t GpuSplit::ExecuteMethod(const u64* entry, size_t words) {
    const u64 head = entry[0];
    const u32 method = static_cast<u32>((head >> 8) & 0xffff);
    const u32 argument = static_cast<u32>(head >> 32);
    auto* const engine = back_engine;
    if (engine->fast_maxwell3d) {
        // Nx2Perf: as below without the profile zones (off with fast_maxwell3d) and the
        // virtual call.
        using Tegra::Engines::Maxwell3D;
        auto* const maxwell3d = static_cast<Maxwell3D*>(engine);
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        if (method - cb_data < 16 && (head & HasSegment) == 0) {
            // Single constant buffer words in a row (macros write them one by one) go up in
            // one write. They are executable: each entry would only consume the sink first.
            std::array<u32, 64> methods;
            std::array<u32, 64> arguments;
            methods[0] = method;
            arguments[0] = argument;
            u32 count = 1;
            while (count < methods.size() && count < words) {
                const u64 next = entry[count];
                const u32 next_method = static_cast<u32>((next >> 8) & 0xffff);
                if (static_cast<Kind>(next & 0xf) != Kind::Method || (next & HasSegment) != 0 ||
                    next_method - cb_data >= 16) {
                    break;
                }
                methods[count] = next_method;
                arguments[count] = static_cast<u32>(next >> 32);
                ++count;
            }
            maxwell3d->ConsumeSink(*system);
            if (count == 1) {
                maxwell3d->CallMethod(*system, method, argument, true);
            } else {
                maxwell3d->CallConstBufferWords(*system, methods.data(), arguments.data(),
                                                count);
            }
            return count;
        }
        if ((head & Direct) != 0) {
            maxwell3d->ConsumeSink(*system);
            maxwell3d->CallMethod(*system, method, argument, true);
        } else if (!maxwell3d->execution_mask[method]) {
            maxwell3d->method_sink.emplace_back(method, argument);
        } else {
            maxwell3d->ConsumeSink(*system);
            if ((head & HasSegment) != 0) {
                maxwell3d->current_dma_segment = entry[1];
            }
            maxwell3d->CallMethod(*system, method, argument, (head & IsLast) != 0);
        }
        return (head & HasSegment) != 0 ? 2 : 1;
    }
    if (!engine->reports_fast_draw_breaks) {
        Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
    }
    if ((head & Direct) != 0) {
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        engine->CallMethod(*system, method, argument, true);
    } else if (!engine->execution_mask[method]) {
        engine->method_sink.emplace_back(method, argument);
    } else {
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[1];
        }
        engine->CallMethod(*system, method, argument, (head & IsLast) != 0);
    }
    return (head & HasSegment) != 0 ? 2 : 1;
}

void GpuSplit::CheckIrRecord(const u64* entry) {
    // EDEN_NX2_V2_IR=verify: sequence numbers go up by one, kinds and sizes are sane.
    const V2::Ir::RecordView record = V2::Ir::Decode(entry);
    const size_t ref_words = record.has_payload ? 1 : 0;
    const bool bad =
        record.sequence != back_sequence + 1 || record.kind == V2::Ir::Kind::Invalid ||
        record.kind >= V2::Ir::Kind::Count || record.words < V2::Ir::HeaderWords + ref_words ||
        record.body_words + ref_words != record.words - V2::Ir::HeaderWords ||
        (record.has_payload &&
         u64{record.payload.index} + record.payload.words > arena.Capacity());
    if (bad && ir_bad_records++ < 16) {
        LOG_ERROR(HW_GPU, "V2 IR record check: kind {} sequence {} (expected {}) size {} body {}",
                  static_cast<u32>(record.kind), record.sequence, back_sequence + 1, record.words,
                  record.body_words);
    }
    if ((record.sequence & 0xfffff) == 0) {
        LOG_INFO(HW_GPU, "V2 IR record check: {} records, {} bad", record.sequence, ir_bad_records);
    }
}

size_t GpuSplit::ExecuteIr(const u64* entry) {
    const u64 head = entry[0];
    const size_t size = V2::Ir::RecordWords(head);
    if (ir_verify) [[unlikely]] {
        CheckIrRecord(entry);
    }
    if (draw_verify) [[unlikely]] {
        V2::IrCheck::OnRecord(V2::Ir::KindOf(head));
    }
    if (V2::Stats::Enabled()) [[unlikely]] {
        V2::Stats::NoteIrRecord(static_cast<u32>(V2::Ir::KindOf(head)), size,
                                (head & V2::Ir::HasPayload) != 0
                                    ? V2::Ir::UnpackRef(entry[V2::Ir::HeaderWords]).words
                                    : 0);
    }
    back_sequence = entry[1];
    const V2::Ir::PayloadRef payload = (head & V2::Ir::HasPayload) != 0
                                           ? V2::Ir::UnpackRef(entry[V2::Ir::HeaderWords])
                                           : V2::Ir::PayloadRef{};
    const u64* const body = entry + V2::Ir::HeaderWords + V2::Ir::RefWords(head);
    switch (V2::Ir::KindOf(head)) {
    case V2::Ir::Kind::Engine: {
        V2::Ir::EngineBody engine;
        std::memcpy(&engine, body, sizeof(engine));
        back_engine = objects->As<Tegra::Engines::EngineInterface>(engine.engine);
        break;
    }
    case V2::Ir::Kind::StateRun: {
        // Every method of the run ends at the record's end (the front notes its writes there;
        // releases and reports close the run, GpuSplit::CutForLanding).
        const u64* it = arena.At(payload.index);
        const u64* const end = it + payload.words;
        while (it < end) {
            it += ExecuteMethod(it, static_cast<size_t>(end - it));
        }
        break;
    }
    case V2::Ir::Kind::MethodData: {
        V2::Ir::MethodDataBody data;
        std::memcpy(&data, body, sizeof(data));
        auto* const engine = back_engine;
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & V2::Ir::HasSegment) != 0) {
            engine->current_dma_segment = data.segment;
        }
        engine->CallMultiMethod(*system, data.method,
                                reinterpret_cast<const u32*>(arena.At(payload.index)),
                                data.amount, data.methods_pending);
        break;
    }
    case V2::Ir::Kind::Segment: {
        // EDEN_NX2_V2_PARALLEL: the commit takes the next segment in order.
        V2::Ir::SegmentBody segment;
        std::memcpy(&segment, body, sizeof(segment));
        if (parallel != nullptr) {
            const auto* const maxwell3d =
                objects->As<Tegra::Engines::Maxwell3D>(segment.engine);
            const bool verify = maxwell3d != nullptr && parallel->Verify();
            V2::Resolve::CbufTable back_cbufs;
            if (verify) {
                back_cbufs = V2::Resolve::CbufTableOf(V2::Resolve::LiveView{*maxwell3d});
            }
            parallel->OnSegment(segment, verify ? maxwell3d->regs.reg_array.data() : nullptr,
                                verify ? &back_cbufs : nullptr);
        }
        break;
    }
    default: {
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
        }
        if ((head & V2::Ir::HasThunk) != 0) {
            // Callback: debug hooks only.
            const u64* const thunk = arena.At(payload.index);
            reinterpret_cast<void (*)(const u64*)>(thunk[0])(thunk + 1);
            break;
        }
        const V2::IrExec::Context context{*system, *objects, arena};
        V2::IrExec::Execute(V2::Ir::Decode(entry), context);
        break;
    }
    }
    return size;
}

size_t GpuSplit::EntryWords(const u64* entry) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        return 2;
    case Kind::Method:
        return (head & HasSegment) != 0 ? 2 : 1;
    case Kind::Multi:
        return ((head & HasSegment) != 0 ? 3 : 2) + ((head >> 32) + 1) / 2;
    case Kind::Call:
        return 2 + static_cast<size_t>((head >> 8) & 0xffffff);
    case Kind::Ir:
        return V2::Ir::RecordWords(head);
    }
    return 1;
}

size_t GpuSplit::Execute(const u64* entry, size_t words) {
    const u64 head = entry[0];
    switch (static_cast<Kind>(head & 0xf)) {
    case Kind::Wrap:
        return 0;
    case Kind::Engine:
        back_engine = reinterpret_cast<Tegra::Engines::EngineInterface*>(entry[1]);
        return 2;
    case Kind::Method:
        return ExecuteMethod(entry, words);
    case Kind::Ir:
        return ExecuteIr(entry);
    case Kind::Multi: {
        const u32 method = static_cast<u32>((head >> 8) & 0xffff);
        const u32 amount = static_cast<u32>(head >> 32);
        const size_t data = (head & HasSegment) != 0 ? 3 : 2;
        auto* const engine = back_engine;
        if (!engine->reports_fast_draw_breaks) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
        }
        Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
        engine->ConsumeSink(*system);
        if ((head & HasSegment) != 0) {
            engine->current_dma_segment = entry[2];
        }
        engine->CallMultiMethod(*system, method, reinterpret_cast<const u32*>(entry + data),
                                amount, static_cast<u32>(entry[1]));
        return data + (amount + 1) / 2;
    }
    case Kind::Call: {
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
        }
        const size_t payload = static_cast<size_t>((head >> 8) & 0xffffff);
        reinterpret_cast<void (*)(const u64*)>(entry[1])(entry + 2);
        return 2 + payload;
    }
    }
    return 1;
}

void GpuSplit::BackLoop(std::stop_token stop_token) {
    const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuBack};
    Common::SetCurrentThreadName("GPUBack");
    on_back_thread = true;
    back_running.store(true, std::memory_order_release);
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();
    if (Settings::values.pin_gpu_threads.GetValue()) {
        Common::PinCurrentThread(Common::PinnedRole::GpuBack);
    }
    system->RegisterHostThread();
#ifdef _WIN32
    g_back_thread_id.store(GetCurrentThreadId(), std::memory_order_relaxed);
#endif
    u64 read_position = 0;
    while (!stop_token.stop_requested()) {
        Nx2Profile::Report("GPU back thread");
        V2::Stats::Report("GPU back thread");
        u64 available = published.load(std::memory_order_acquire);
        if (available == read_position) {
            Nx2Profile::Scope scope{Nx2Profile::Zone::SplitWait};
            if (feed_idle_back) [[unlikely]] {
                // The front publishes what it holds after a few more words from now on.
                back_idle_at.store(read_position, std::memory_order_relaxed);
            }
            // The front usually sends more within microseconds: spin briefly before sleeping.
            // EDEN_NX2_V2_SPIN: the window decides when the back sleeps; the idle hook still
            // runs after the old spin's time (OldIdleSpinNs), so submits keep their cadence.
            u64 spin_start = 0;
            u64 idle_start = 0;
            bool hook_ran = false;
            for (u32 spins = 0;
                 available == read_position && !hook_requested.load(std::memory_order_relaxed);
                 ++spins) {
                if (!spin_window) {
                    if (spins >= 8192) {
                        break;
                    }
                } else {
                    const bool open = SpinWindowOpen(spins, spin_start);
                    if (spins == 0) {
                        idle_start = Nx2Profile::Now();
                    } else if (!hook_ran && (spins & 63) == 0 && idle_hook != nullptr &&
                               Nx2Profile::Now() - idle_start >= idle_hook_ns) {
                        hook_ran = true;
                        hook_requested.store(false, std::memory_order_seq_cst);
                        idle_hook_requested.store(false, std::memory_order_relaxed);
                        ServeCpuLane();
                        idle_hook(idle_argument);
                    }
                    if (!open && (hook_ran || idle_hook == nullptr)) {
                        break;
                    }
                }
                Pause();
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position && hook_requested.load(std::memory_order_relaxed)) {
                // A request ended the spin, not the idle timeout: serve it and keep spinning (a
                // lane request alone doesn't submit deferred work; going idle below does).
                ServeHookRequests();
                continue;
            }
            if (available == read_position && idle_hook != nullptr && !hook_ran) {
                hook_requested.store(false, std::memory_order_seq_cst);
                idle_hook_requested.store(false, std::memory_order_relaxed);
                ServeCpuLane();
                idle_hook(idle_argument);
                available = published.load(std::memory_order_acquire);
            }
            if (available == read_position) {
                // Publish bumps `wake` when it sees back_waiting: read it before the last look
                // at `published`, so a publish in between ends the wait at once.
                back_waiting.store(true, std::memory_order_seq_cst);
                const u32 seen = wake.load(std::memory_order_seq_cst);
                available = published.load(std::memory_order_seq_cst);
                // A request made after the hook ran (RequestIdleHook) runs it before sleeping.
                if (available == read_position && !stop_token.stop_requested() &&
                    !hook_requested.load(std::memory_order_seq_cst)) {
                    wake.wait(seen, std::memory_order_seq_cst);
                }
                back_waiting.store(false, std::memory_order_relaxed);
                continue;
            }
        }
        u64 last_store = read_position;
        while (read_position != available) {
            const size_t index = read_position & RingMask;
            if (tracks_writes) [[unlikely]] {
                back_entry_position = read_position;
                V2::Landing::tag = read_position + EntryWords(&ring[index]);
            }
            const size_t size =
                Execute(&ring[index], std::min<u64>(available - read_position, RingWords - index));
            read_position += size != 0 ? size : RingWords - index;
            if (hook_requested.load(std::memory_order_relaxed)) [[unlikely]] {
                // EDEN_NX2_V2_CPULANE: a guest CPU flush runs here, after the current entry.
                ServeHookRequests();
            }
            // Hand ring space back now and then while working through a long batch, and at once
            // when the front waits for this position (WaitForWriters).
            if ((read_position - last_store >= 4096 ||
                 read_position >= wait_target.load(std::memory_order_relaxed)) &&
                read_position != available) {
                consumed_sequence.store(back_sequence, std::memory_order_release);
                consumed.store(read_position, std::memory_order_seq_cst);
                last_store = read_position;
                NotifyFront();
            }
        }
        consumed_sequence.store(back_sequence, std::memory_order_release);
        consumed.store(read_position, std::memory_order_seq_cst);
        NotifyFront();
    }
    back_running.store(false, std::memory_order_seq_cst);

    front_wake.fetch_add(1, std::memory_order_seq_cst);
    front_wake.notify_all();
}

} // namespace VideoCommon
