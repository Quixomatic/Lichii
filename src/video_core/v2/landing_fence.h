// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"
#include "video_core/v2/landing.h"

// F4-X2 (R2-F4-01; MASTER_PLAN §5.5, §6; SESSION_HANDOFF §5: a landing wait causes its submit):
// which host tick a thread blocking on a landing needs submitted. An operation rides the fence
// queued next after its capture (FenceManager::SignalFence takes every uncommitted operation), so
// a capture at or below `issued` (`untracked`) needs at most host tick `tick` submitted. The
// waiter files an urgent request for that tick (RequestSubmit): the back's next idle hook submits
// it whether the GPU is busy or not, instead of after the GPU's queue drained.
namespace VideoCommon::V2::LandingFence {

/// The newest lazy fence the back queued: its host tick, and the newest tickets captured by then.
/// The back stores the tick before the tickets and readers load the tickets first, so the tick
/// they read is never older than the fence of the tickets they read. Not reset with the split's
/// tickets: a stale tick is a submitted one, whose request does nothing.
inline std::atomic<u64> tick{0};
inline std::atomic<u64> issued{0};
inline std::atomic<u64> untracked{0};

/// SYNC_SUBMIT_SPEC §7.3 fix 6: the open tick's state for the 2 s semaphore line (written by the
/// recording and fencing threads with relaxed stores, read only for that line).
/// Scheduler::CurrentTick after the last submission (the open tick; also the tick an idle-back
/// waiter asks for, 9c4a)
inline std::atomic<u64> diag_open_tick{1};
inline std::atomic<u64> diag_lazy_tick{0};       ///< The newest lazy fence's tick
inline std::atomic<bool> diag_submit_wanted{false};
inline std::atomic<u64> diag_fence_wait_tick{0}; ///< The tick the fencing thread blocks on (0: none)
/// §9.3: at the back's last idle hook, operations with no fence / uncommitted query flushes.
inline std::atomic<bool> diag_unfenced{false};
inline std::atomic<bool> diag_query_flushes{false};

/// The renderer's urgent request for a host tick (Scheduler::RequestLazyFlush) and the wake that
/// runs the back's idle hook (GpuSplit::RequestIdleHook). Set once before the back starts.
inline void (*request_hook)(void*, u64) = nullptr;
inline void* request_argument = nullptr;

/// Back: a lazy fence waiting for host tick `fence_tick` was queued: the captures made by now
/// ride it. SYNC_SUBMIT_SPEC §2.5: nothing is urgent at queue time (the 019d/95fd "urgent at
/// queue" made ~18 submits/f through FlushDeferred); a thread that blocks on one of them files
/// its own request for this tick (TickFor), served at the back's next entry boundary.
inline void NoteFence(u64 fence_tick) {
    const u64 now_issued = Landing::issued.load(std::memory_order_relaxed);
    const u64 now_untracked = Landing::untracked_issued.load(std::memory_order_relaxed);
    tick.store(fence_tick, std::memory_order_release);
    issued.store(now_issued, std::memory_order_release);
    untracked.store(now_untracked, std::memory_order_release);
}

/// Any thread: the host tick to submit for the capture `ticket` (or the untracked capture
/// `untracked_ticket`; 0 for neither) to land; 0 while its fence is not queued yet.
[[nodiscard]] inline u64 TickFor(u64 ticket, u64 untracked_ticket) {
    const bool tracked = ticket != 0 && issued.load(std::memory_order_acquire) >= ticket;
    const bool other =
        untracked_ticket != 0 && untracked.load(std::memory_order_acquire) >= untracked_ticket;
    return tracked || other ? tick.load(std::memory_order_acquire) : 0;
}

/// Any thread: the host tick to submit for every queued capture still to land; 0 when none.
[[nodiscard]] inline u64 TickForPending() {
    const bool tracked = issued.load(std::memory_order_acquire) >
                         Landing::landed.load(std::memory_order_acquire);
    const bool other = untracked.load(std::memory_order_acquire) >
                       Landing::untracked_landed.load(std::memory_order_acquire);
    return tracked || other ? tick.load(std::memory_order_acquire) : 0;
}

/// Any thread blocking on host tick `fence_tick` (0: none known yet): the urgent request. The
/// caller wakes the back (GpuSplit::RequestIdleHook).
inline void Request(u64 fence_tick) {
    if (fence_tick != 0 && request_hook != nullptr) {
        request_hook(request_argument, fence_tick);
    }
}

} // namespace VideoCommon::V2::LandingFence
