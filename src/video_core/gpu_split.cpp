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
#include "video_core/engines/kepler_compute.h"
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
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/s1_deltas.h"
#include "video_core/v2/ir_exec.h"
#include "video_core/v2/front_writes.h"
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
    if (ir && !V2::IrTrace::Directory().empty()) {
        ir_trace = std::make_unique<V2::IrTrace::Recorder>();
        LOG_INFO(HW_GPU, "V2 IR trace: one file per frame in {}", V2::IrTrace::Directory());
    }
    // The trace is one of the per-record checks: without it ExecuteIr pays no extra branch.
    ir_checks = ir_verify || draw_verify || V2::Stats::Enabled() || ir_trace != nullptr;
    {
        // S1 single owner (Fix F1): the back applies the front's classified entries.
        const V2::Mode regsem = V2::Get(V2::Switch::RegSem);
        const bool s1_possible = ir && front_macros;
        s1_exec = s1_possible && draw_records && !draw_verify && regsem == V2::Mode::On;
        s1_check = s1_possible && !s1_exec && regsem != V2::Mode::Off &&
                   (regsem == V2::Mode::Verify || draw_verify);
    }
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
    pending_cb = {};
    cb_translations.fill({});
    unlanded.Clear();
    recent_releases.fill({});
    ticket_position = 0;
    ticket_sub = 0;
    last_ticket = 0;
    V2::Landing::landed.store(0, std::memory_order_relaxed);
    V2::Landing::issued.store(0, std::memory_order_relaxed);
    qmd_release_notes.Clear();
    V2::Landing::ResetUntracked();
    V2::Landing::verify = acquire_mode == V2::Mode::Verify || hazard_mode == V2::Mode::Verify;
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
    LOG_INFO(HW_GPU, "GPU thread split: engine methods run on a second thread{}{}{}{}",
             front_macros ? ", macros on the first" : "",
             ir ? (ir_verify ? ", typed IR (verified)" : ", typed IR") : "",
             draw_records ? (draw_verify ? ", draw records (checked)" : ", draw records") : "",
             s1_exec ? ", 3D register deltas" : (s1_check ? ", 3D register deltas (checked)" : ""));
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
    ir_checks = false;
    s1_exec = false;
    s1_check = false;
    ir_trace.reset();
    ir_run_classified = false;
    s1_open_words = 0;
    s1_cbuf = nullptr;
    s1_cursor = nullptr;
    s1_limit = nullptr;
    s1_unadvanced = 0;
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
    pending_cb = {};
    unlanded.Clear();
    ticket_position = 0;
    ticket_sub = 0;
    last_ticket = 0;
    if (V2::Landing::verify) {
        LOG_INFO(HW_GPU, "V2 landing verify: {} tickets landed out of order",
                 V2::Landing::order_violations.load(std::memory_order_relaxed));
        V2::Landing::verify = false;
    }
    V2::Landing::landed.store(0, std::memory_order_relaxed);
    V2::Landing::issued.store(0, std::memory_order_relaxed);
    qmd_release_notes.Clear();
    V2::Landing::ResetUntracked();
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

void GpuSplit::FlushCbWriteSlow() {
    const PendingCb pending = pending_cb;
    pending_cb = {};
    const u64 page = pending.begin >> 12;
    if (((pending.end - 1) >> 12) != page) {
        V2::NoteRangeWriteGpu(*this, *pending.memory, pending.begin, pending.end - pending.begin,
                              V2::Producer::CbData);
        return;
    }
    // One page: its GPU→device translation from a small cache, valid until any map runs.
    // Constant buffers are rewritten on the same few pages all frame.
    const u64 maps = maps_executed.load(std::memory_order_acquire);
    CbTranslation& cached = cb_translations[page % cb_translations.size()];
    if (cached.memory != pending.memory || cached.page != page || cached.maps != maps) {
        const auto device = pending.memory->GpuToCpuAddress(page << 12);
        cached = {pending.memory, page, maps, device ? *device : ~u64{0}};
    }
    if (cached.device != ~u64{0}) {
        NoteRangeWrite(cached.device + (pending.begin & 0xfff), pending.end - pending.begin,
                       V2::Producer::CbData);
    }
}

V2::HazardResult GpuSplit::FindWithLanding(u64 device_address, u64 size, u64 done,
                                            const V2::HazardFilter& filter, u8* out) {
    using Kind = V2::HazardResult::Kind;
    const u64 landed = V2::Landing::landed.load(std::memory_order_acquire);
    unlanded.Retire(landed);
    V2::HazardFilter landing_filter;
    landing_filter.ranged = filter.ranged & (V2::ProducerBit(V2::Producer::Release) |
                                             V2::ProducerBit(V2::Producer::Report));
    V2::HazardResult landing = unlanded.Find(device_address, size, landed, landing_filter, out);
    landing.landing = landing.kind != Kind::Clear;
    const V2::HazardResult queued = hazards.Find(device_address, size, done, filter, out);
    if (queued.kind == Kind::Wait || landing.kind == Kind::Clear) {
        return queued;
    }
    if (landing.kind == Kind::Wait || queued.kind == Kind::Clear) {
        return landing;
    }
    // Both forwarded: the queued stores went on top. Right only if they are newer than every
    // landing release applied; otherwise wait for them, after which only those remain.
    if (queued.position <= V2::RingPositionOf(landing)) {
        V2::HazardResult wait = queued;
        wait.kind = Kind::Wait;
        return wait;
    }
    V2::HazardResult merged = queued;
    merged.covered = queued.covered || landing.covered;
    merged.producers |= landing.producers;
    return merged;
}

V2::HazardResult GpuSplit::FindUntracked(u64 device_address, u64 size, u64 done,
                                          const V2::HazardFilter& filter,
                                          const V2::HazardResult& found, u64 floor) {
    constexpr u32 qmd = V2::ProducerBit(V2::Producer::QmdRelease);
    V2::HazardFilter untracked_filter;
    untracked_filter.ranged = filter.ranged & qmd;
    untracked_filter.unbounded = (filter.unbounded | filter.unbounded_synced) & qmd;
    if ((untracked_filter.ranged | untracked_filter.unbounded) == 0) {
        return found;
    }
    // A QMD release lands at its dispatch's fence (a four-word one writes its timestamp only
    // there, Landing::Untracked): once the back ran the entry, `hazards` no longer holds it,
    // but memory doesn't either until the operation ran.
    const u64 settled = V2::Landing::UntrackedSettled(done);
    qmd_release_notes.Retire(settled);
    const u64 live = std::max(settled, floor);
    // Ranged notes: their ranges decide.
    V2::HazardFilter ranged_filter;
    ranged_filter.ranged = untracked_filter.ranged;
    V2::HazardResult result =
        qmd_release_notes.Find(device_address, size, live, ranged_filter);
    // Fix F3h (F3-X1): an unbounded note (an unknown QMD) may write any word only while the
    // back hasn't run its entry. Once it has, the operations still to land are captured with
    // their device ranges (Landing::UntrackedTargetFor): only those that overlap the range
    // count, so an unknown QMD no longer holds every word for about a frame.
    if (untracked_filter.unbounded != 0) {
        V2::HazardFilter unbounded_filter;
        unbounded_filter.unbounded = filter.unbounded & qmd;
        unbounded_filter.unbounded_synced = filter.unbounded_synced & qmd;
        const V2::HazardResult unbounded =
            qmd_release_notes.Find(device_address, size, live, unbounded_filter);
        if (unbounded.kind != V2::HazardResult::Kind::Clear &&
            (unbounded.position > done || unbounded.overflow ||
             V2::Landing::UntrackedTargetFor(device_address, size) != 0)) {
            if (result.kind == V2::HazardResult::Kind::Clear ||
                unbounded.position > result.position) {
                result.position = unbounded.position;
            }
            result.kind = V2::HazardResult::Kind::Wait;
            result.producers |= unbounded.producers;
            result.overflow |= unbounded.overflow;
        }
    }
    if (result.kind == V2::HazardResult::Kind::Clear) {
        return found;
    }
    result.kind = V2::HazardResult::Kind::Wait;
    result.untracked = true;
    result.producers |= found.producers;
    return result;
}

bool GpuSplit::WaitUntrackedLanded(u64 ticket, DrainSite site, u64 timeout_us) {
    const auto landed = [ticket] {
        return V2::Landing::untracked_landed.load(std::memory_order_acquire) >= ticket;
    };
    if (landed()) {
        return true;
    }
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    const bool v2_stats = V2::Stats::Enabled();
    const u64 start = Nx2Profile::Enabled() || v2_stats ? Nx2Profile::Ticks() : 0;
    const auto begin = std::chrono::steady_clock::now();
    auto next_request = begin;
    for (u32 spins = 0; !landed(); ++spins) {
        const auto now = std::chrono::steady_clock::now();
        if (now - begin >= std::chrono::microseconds{timeout_us} || !BackRunning()) {
            return landed();
        }
        if (now >= next_request) {
            // Its fence may sit in a deferred submission (F4-02): ask for it, never drain.
            RequestIdleHook();
            next_request = now + std::chrono::milliseconds{1};
        }
        if (spins < 64) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
    Nx2Profile::AddDrain(site, start);
    if (v2_stats) [[unlikely]] {
        V2::Stats::NoteDrain(site, start);
    }
    return true;
}

bool GpuSplit::WaitLanded(u64 ticket, DrainSite site, u64 timeout_us) {
    if (V2::Landing::landed.load(std::memory_order_acquire) >= ticket) {
        return true;
    }
    // The back must have run the write before its operation exists.
    WaitForPosition(V2::Landing::RingPosition(ticket), site);
    Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
    const bool v2_stats = V2::Stats::Enabled();
    const u64 start = Nx2Profile::Enabled() || v2_stats ? Nx2Profile::Ticks() : 0;
    const auto begin = std::chrono::steady_clock::now();
    auto next_request = begin;
    for (u32 spins = 0; V2::Landing::landed.load(std::memory_order_acquire) < ticket; ++spins) {
        const auto now = std::chrono::steady_clock::now();
        if (now - begin >= std::chrono::microseconds{timeout_us} || !BackRunning()) {
            return V2::Landing::landed.load(std::memory_order_acquire) >= ticket;
        }
        if (now >= next_request) {
            // The operation's fence may sit in a deferred submission: ask for it (no drain).
            RequestIdleHook();
            next_request = now + std::chrono::milliseconds{1};
        }
        if (spins < 64) {
            Pause();
        } else {
            std::this_thread::yield();
        }
    }
    // Counted at its site, timeouts included.
    Nx2Profile::AddDrain(site, start);
    if (v2_stats) [[unlikely]] {
        V2::Stats::NoteDrain(site, start);
    }
    return V2::Landing::landed.load(std::memory_order_acquire) >= ticket;
}

bool GpuSplit::CompleteLanding(u64 ticket, DrainSite site, u64 (*complete)(void*),
                               void* argument, u64 untracked_position, u64 untracked_address,
                               u64 untracked_size, u64* settled) {
    const auto untracked_settled = [&] {
        const u64 position = V2::Landing::UntrackedSettled(ConsumedNow());
        if (position < untracked_position) {
            return false;
        }
        if (settled != nullptr) {
            *settled = position;
        }
        return true;
    };
    const auto landed = [&] {
        return V2::Landing::landed.load(std::memory_order_acquire) >= ticket &&
               (untracked_position == 0 || untracked_settled());
    };
    if (landed()) {
        return true;
    }
    // The back must have run the write before anything can complete it.
    WaitForPosition(std::max(V2::Landing::RingPosition(ticket), untracked_position), site);
    if (landed()) {
        return true;
    }
    u64 target = 0;
    u64 untracked_target = 0;
    u64 job_position = 0;
    {
        Nx2Profile::Scope scope{Nx2Profile::Zone::WriterWait};
        const bool v2_stats = V2::Stats::Enabled();
        const u64 start = Nx2Profile::Enabled() || v2_stats ? Nx2Profile::Ticks() : 0;
        landing_job.complete = complete;
        landing_job.argument = argument;
        landing_job.target = 0;
        landing_job.untracked_address = untracked_address;
        landing_job.untracked_size = untracked_position != 0 ? untracked_size : 0;
        landing_job.untracked_target = 0;
        landing_job.position = 0;
        landing_job.done.store(false, std::memory_order_relaxed);
        RequestBackJob(
            [](void* job_argument) {
                auto& job = *static_cast<LandingJob*>(job_argument);
                job.target = job.complete(job.argument);
                // Captured on this thread before now: every untracked operation of the entries
                // it ran (Landing::tag: the end of the last one) that writes the range.
                if (job.untracked_size != 0) {
                    job.untracked_target = V2::Landing::UntrackedTargetFor(job.untracked_address,
                                                                           job.untracked_size);
                }
                job.position = V2::Landing::tag;
                job.done.store(true, std::memory_order_release);
            },
            &landing_job);
        // The job reads `landing_job` until it stores `done`: wait for it, not only for the
        // landing.
        while (!landing_job.done.load(std::memory_order_acquire) && BackRunning()) {
            std::this_thread::yield();
        }
        if (!landing_job.done.load(std::memory_order_acquire)) {
            return false; // The back stopped
        }
        target = landing_job.target;
        untracked_target = landing_job.untracked_target;
        job_position = landing_job.position;
        Nx2Profile::AddDrain(site, start);
        if (v2_stats) [[unlikely]] {
            V2::Stats::NoteDrain(site, start);
        }
    }
    // Bytes only the operations write (a report's timestamp: the query cache's early write of
    // a final value leaves it out): the job signalled their fence; its operations run in order,
    // so once the back's newest ticket landed every one the back made before the job did.
    // WaitLanded asks for the deferred submit while it waits (SESSION_HANDOFF §5g).
    if (target != 0 && !WaitLanded(target, site, LandingBoundUs)) {
        return false;
    }
    // F3-X6: the QMD release operations queued by then that write the range, the same way
    // (untracked tickets land in capture order too). After them the range has nothing of the
    // entries the back ran before the job still to land.
    if (untracked_target != 0 && !WaitUntrackedLanded(untracked_target, site, LandingBoundUs)) {
        return false;
    }
    if (settled != nullptr && untracked_position != 0) {
        *settled = std::max(job_position, untracked_position);
    }
    return true;
}

void GpuSplit::Drain(DrainSite site) {
    if (site == DrainSite::Map) {
        FlushCbWrite(); // Translated with the mapping the words were sent under
        ++map_epoch;
    }
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

u64* GpuSplit::S1OpenRun(size_t words) {
    constexpr size_t header = V2::Ir::HeaderWords + 1;
    u64* const record = IrBegin(V2::Ir::Kind::StateRun, 0, header,
                                V2::Ir::HasPayload | (s1_exec ? V2::Ir::Classified : 0));
    u64* const entries = ArenaBegin(words, ir_sequence, write_position + header);
    ir_run_index = arena.IndexOf(entries);
    record[V2::Ir::HeaderWords] = V2::Ir::PackRef(ir_run_index, 0);
    ir_run = record;
    ir_run_words = words;
    ir_run_classified = s1_exec;
    s1_open_words = header;
    return entries;
}

u64* GpuSplit::S1BeginSlow(Tegra::Engines::EngineInterface* engine, size_t words) {
    SelectEngine(engine);
    s1_cbuf = nullptr;
    u64* entry = nullptr;
    if (ir_run != nullptr && ir_run_classified == s1_exec && ir_run_words + words <= MaxRunWords) {
        // The run's end is the arena's head, or the start of its reserved words: grow from there
        // (the payload stays contiguous).
        const size_t have = static_cast<size_t>(s1_limit - s1_cursor);
        u64* const start = s1_cursor != nullptr ? s1_cursor : nullptr;
        if (u64* const grown = arena.Extend(words - have)) {
            entry = start != nullptr ? start : grown;
            ir_run_words += words;
            s1_cursor = nullptr;
            s1_limit = nullptr;
        }
    }
    if (entry == nullptr) {
        entry = S1OpenRun(words);
    }
    if (s1_exec) {
        S1Reserve();
    }
    return entry;
}

void GpuSplit::S1Reserve() {
    if (ir_run == nullptr || !ir_run_classified ||
        ir_run_words + S1ReserveWords > MaxRunWords) {
        return;
    }
    if (u64* const reserved = arena.Extend(S1ReserveWords)) {
        s1_cursor = reserved;
        s1_limit = reserved + S1ReserveWords;
    }
}

void GpuSplit::S1Cbuf(Tegra::Engines::EngineInterface* engine, bool store, u32 slot, u64 buffer,
                      u32 offset, u32 word) {
    namespace S1 = V2::S1;
    if (u64* const head = s1_cbuf; head != nullptr && engine == front_engine) {
        const u64 h = head[0];
        const u32 count = S1::CbufCount(h);
        const u32 m0 = S1::CbufMethod0(h);
        const bool joins = ((h & S1::CbufStore) != 0) == store && head[1] == buffer &&
                           S1::ValueOf(h) + count * 4 == offset && count < S1::MaxCbufWords;
        bool inc = (h & S1::CbufInc) != 0;
        bool method_ok = !store;
        if (store) {
            if (count == 1 && !inc && slot == m0 + 1) {
                inc = true;
                method_ok = true;
            } else {
                method_ok = slot == m0 + (inc ? count : 0);
            }
        }
        if (joins && method_ok) {
            u32* data = nullptr;
            // Odd: the high half of the data word at `next`; even: a new word at `next`, right
            // after the entry (it is the run's last).
            u64* const next = head + 2 + count / 2;
            if (count % 2 != 0) {
                data = reinterpret_cast<u32*>(next) + 1;
            } else if (s1_cursor == next && s1_cursor != s1_limit) {
                *s1_cursor++ = 0;
                ++ir_run_words;
                data = reinterpret_cast<u32*>(next);
            } else if (s1_cursor == nullptr && ir_run_words + 1 <= MaxRunWords) {
                if (u64* const extra = arena.Extend(1)) {
                    ++ir_run_words;
                    *extra = 0;
                    data = reinterpret_cast<u32*>(extra);
                }
            }
            if (data != nullptr) {
                *data = word;
                head[0] = S1::MakeCbuf(store, inc, m0, count + 1, S1::ValueOf(h));
                if (count % 2 == 0) {
                    S1Commit(1); // A publish here closes the run with this word in it
                }
                return;
            }
        }
    }
    u64* const entry = S1Begin(engine, 3);
    entry[0] = S1::MakeCbuf(store, false, slot, 1, offset);
    entry[1] = buffer;
    entry[2] = word;
    s1_cbuf = entry;
    S1Commit(3);
}

void GpuSplit::S1CbufBlock(Tegra::Engines::EngineInterface* engine, u64 buffer, u32 offset,
                           const u32* words, u32 count) {
    namespace S1 = V2::S1;
    if (count == 0) {
        return;
    }
    const size_t size = 2 + (size_t{count} + 1) / 2;
    u64* const entry = S1Begin(engine, size);
    entry[0] = S1::MakeCbuf(false, false, 0, count, offset);
    entry[1] = buffer;
    entry[size - 1] = 0;
    std::memcpy(entry + 2, words, size_t{count} * sizeof(u32));
    S1Commit(size);
    s1_cbuf = nullptr;
}

void GpuSplit::S1Inline(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                        u32 count) {
    namespace S1 = V2::S1;
    while (count > 0) {
        const u32 piece = std::min(count, S1::MaxInlineWords);
        const size_t size = 1 + (size_t{piece} + 1) / 2;
        u64* const entry = S1Begin(engine, size);
        entry[0] = S1::MakeRegister(S1::Tag::Inline, method, piece);
        entry[size - 1] = 0;
        std::memcpy(entry + 1, words, size_t{piece} * sizeof(u32));
        S1Commit(size);
        words += piece;
        count -= piece;
    }
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

/// Nx2Perf: a method entry for a Maxwell3D with fast_maxwell3d, as ExecuteMethod below without
/// the profile zones (off with fast_maxwell3d) and the virtual call.
SPLIT_FORCE_INLINE size_t GpuSplit::ExecuteMethod3D(Tegra::Engines::EngineInterface* engine,
                                                    const u64* entry, size_t words) {
    const u64 head = entry[0];
    const u32 method = static_cast<u32>((head >> 8) & 0xffff);
    const u32 argument = static_cast<u32>(head >> 32);
    {
        using Tegra::Engines::Maxwell3D;
        auto* const maxwell3d = static_cast<Maxwell3D*>(engine);
        if (V2::S1::IsClassified(head)) [[unlikely]] {
            // S1Check: the entries the front would send instead, checked against this engine.
            return maxwell3d->S1CheckEntry(entry);
        }
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
}

SPLIT_FORCE_INLINE size_t GpuSplit::ExecuteMethod(const u64* entry, size_t words) {
    auto* const engine = back_engine;
    if (engine->fast_maxwell3d) {
        return ExecuteMethod3D(engine, entry, words);
    }
    const u64 head = entry[0];
    const u32 method = static_cast<u32>((head >> 8) & 0xffff);
    const u32 argument = static_cast<u32>(head >> 32);
    if (V2::S1::IsClassified(head)) [[unlikely]] {
        return static_cast<Tegra::Engines::Maxwell3D*>(engine)->S1CheckEntry(entry);
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

void GpuSplit::TraceIrRecord(const u64* entry) {
    const V2::Ir::RecordView record = V2::Ir::Decode(entry);
    std::span<const u64> payload;
    if (record.has_payload) {
        payload = {arena.At(record.payload.index), record.payload.words};
    }
    ir_trace->Add({entry, record.words}, payload);
    if (record.kind == V2::Ir::Kind::Present) {
        // One file per frame, cut at its present.
        const std::string path = fmt::format("{}/ir_frame_{:06}.irt",
                                             V2::IrTrace::Directory(), ir_trace_frame++);
        if (!ir_trace->Write(path) && ir_trace_frame < 4) {
            LOG_WARNING(HW_GPU, "V2 IR trace: could not write {}", path);
        }
        ir_trace->Clear();
    }
}

void GpuSplit::ReplayIrRecord(std::span<const u64> record, std::span<const u64> payload) {
    IrCut();
    const size_t words = record.size();
    const u64 head = record[0];
    if (payload.empty() || (head & V2::Ir::HasPayload) == 0) {
        u64* const out = Reserve(words);
        std::memcpy(out, record.data(), words * sizeof(u64));
        out[1] = ++ir_sequence;
        Commit(words);
        return;
    }
    u64* const span = ArenaBegin(payload.size(), ir_sequence + 1, write_position + words);
    std::memcpy(span, payload.data(), payload.size() * sizeof(u64));
    u64* const out = Reserve(words);
    std::memcpy(out, record.data(), words * sizeof(u64));
    out[1] = ++ir_sequence;
    out[V2::Ir::HeaderWords] =
        V2::Ir::PackRef(arena.IndexOf(span), static_cast<u32>(payload.size()));
    Commit(words);
    Advance(payload.size());
}

size_t GpuSplit::ExecuteIr(const u64* entry) {
    const u64 head = entry[0];
    const size_t size = V2::Ir::RecordWords(head);
    if (ir_checks) [[unlikely]] {
        if (ir_trace) {
            TraceIrRecord(entry);
        }
        if (ir_verify) {
            CheckIrRecord(entry);
        }
        if (draw_verify) {
            V2::IrCheck::OnRecord(V2::Ir::KindOf(head));
        }
        if (V2::Stats::Enabled()) {
            V2::Stats::NoteIrRecord(static_cast<u32>(V2::Ir::KindOf(head)), size,
                                    (head & V2::Ir::HasPayload) != 0
                                        ? V2::Ir::UnpackRef(entry[V2::Ir::HeaderWords]).words
                                        : 0);
        }
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
        // A run goes to one engine (the front opens a new one when it switches), so the
        // engine's path is chosen once per run, not per method.
        const u64* it = arena.At(payload.index);
        const u64* const end = it + payload.words;
        auto* const engine = back_engine;
        if ((head & V2::Ir::Classified) != 0) {
            // S1 single owner: register deltas and typed entries, no method execution. By engine
            // class, not fast path: profile builds run 3D without fast_maxwell3d (F1-X1).
            if (engine->is_maxwell3d) [[likely]] {
                static_cast<Tegra::Engines::Maxwell3D*>(engine)->ExecuteS1Run(it, end);
            } else {
                // Only the native compute class sends classified runs besides 3D (F1-05).
                if (!engine->reports_fast_draw_breaks) {
                    Nx2FastDraw::Break(Nx2FastDraw::Reason::OtherEngine);
                }
                Nx2Profile::Scope method_scope{Nx2Profile::Zone::Method};
                static_cast<Tegra::Engines::KeplerCompute*>(engine)->ExecuteS1Run(*system, it,
                                                                                  end);
            }
            break;
        }
        if (engine->fast_maxwell3d) {
            while (it < end) {
                it += ExecuteMethod3D(engine, it, static_cast<size_t>(end - it));
            }
        } else {
            while (it < end) {
                it += ExecuteMethod(it, static_cast<size_t>(end - it));
            }
        }
        break;
    }
    case V2::Ir::Kind::Compact: {
        // One per draw trigger and macro end: only a macro end runs anything (as
        // V2::IrExec::Execute, without decoding the record there).
        if ((head & FastDrawNeutral) == 0) {
            Nx2FastDraw::Break(Nx2FastDraw::Reason::Call);
        }
        V2::Ir::CompactBody compact;
        std::memcpy(&compact, body, sizeof(compact));
        if (compact.full_kind == V2::Ir::Kind::Draw &&
            compact.op == static_cast<u8>(V2::Ir::DrawOp::MacroEnd)) {
            objects->As<Tegra::Engines::Maxwell3D>(compact.engine)->FinishFrontMacro(*system);
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
            // The back's registers with the state writes still in its method sink (they count
            // as done; the next executable method applies them), and its shadow RAM.
            static thread_local std::array<u32, Tegra::Engines::Maxwell3D::Regs::NUM_REGS>
                back_regs;
            static thread_local std::array<u32, Tegra::Engines::Maxwell3D::Regs::NUM_REGS>
                back_shadow;
            if (verify) {
                back_cbufs = V2::Resolve::CbufTableOf(V2::Resolve::LiveView{*maxwell3d});
                back_regs = maxwell3d->regs.reg_array;
                back_shadow = maxwell3d->shadow_state.reg_array;
                const auto control = maxwell3d->shadow_state.shadow_ram_control;
                for (const auto& [method, value] : maxwell3d->method_sink) {
                    back_regs[method] =
                        V2::RegSem::ApplyShadow3D(control, back_shadow.data(), method, value);
                }
            }
            // With S1 the shadow RAM lives on the front only.
            parallel->OnSegment(segment, verify ? back_regs.data() : nullptr,
                                verify && !s1_exec ? back_shadow.data() : nullptr,
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
        back_batch_end = available;
        ++back_batches;
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
