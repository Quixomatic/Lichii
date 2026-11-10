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

/// The renderer's urgent request for a host tick (Scheduler::RequestLazyFlush) and the wake that
/// runs the back's idle hook (GpuSplit::RequestIdleHook). Set once before the back starts.
inline void (*request_hook)(void*, u64) = nullptr;
inline void* request_argument = nullptr;

/// Front (a suspended acquire whose producer lands at a fence): the producer is at or before ring
/// position `position` (the front's write position when it asked). Its fence may not be queued
/// yet when the waiter asks; the back makes it urgent when it is (NoteFence).
inline std::atomic<u64> wanted_position{0};
inline void Want(u64 position) {
    Landing::RaiseTo(wanted_position, position);
}

/// Back: a lazy fence waiting for host tick `fence_tick` was queued. True when it carries a
/// capture a suspended waiter may need (made at or before `wanted_position`): the caller files
/// the urgent request for it (MASTER_PLAN §6: a submit is forced for an unsubmitted sync). The
/// test is by the newest capture of the previous fence, so it may include one entry too many,
/// never one too few.
inline bool NoteFence(u64 fence_tick) {
    const u64 previous_issued = issued.load(std::memory_order_relaxed);
    const u64 previous_untracked = untracked.load(std::memory_order_relaxed);
    const u64 now_issued = Landing::issued.load(std::memory_order_relaxed);
    const u64 now_untracked = Landing::untracked_issued.load(std::memory_order_relaxed);
    tick.store(fence_tick, std::memory_order_release);
    issued.store(now_issued, std::memory_order_release);
    untracked.store(now_untracked, std::memory_order_release);
    const u64 wanted = wanted_position.load(std::memory_order_acquire);
    if (wanted == 0) {
        return false;
    }
    const bool tracked =
        now_issued > previous_issued && Landing::RingPosition(previous_issued) <= wanted;
    const bool other = now_untracked > previous_untracked &&
                       Landing::UntrackedPosition(previous_untracked) <= wanted;
    return tracked || other;
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
