// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// F4-02/F4-04: requests for the recording thread to submit a tick that is still open. Any thread
// asks (Request) and wakes the recording thread through its idle hook; the recording thread
// submits at that hook (IdleSubmits), at a pass boundary or at a chunk. SESSION_HANDOFF §5g: a
// wait for a landing must cause its submit, so a waiter never blocks on an open tick without a
// request the idle hook serves. Ticks below `open_tick` (Scheduler::CurrentTick) are submitted.
namespace VideoCommon::V2 {

class SubmitRequests {
public:
    /// Any thread: `tick` must be submitted. `urgent`: a thread blocks on it now (a reader, a
    /// guest sync wait); not urgent: a bounded-latency request nobody blocks on yet (a lazy
    /// fence's window ran out), which the idle hook may leave for the GPU's idle (F4-X2).
    void Request(u64 tick, bool urgent = true) noexcept {
        Raise(requested, tick);
        if (urgent) {
            Raise(urgent_tick, tick);
        }
    }

    /// Whether a requested tick is still open.
    [[nodiscard]] bool Open(u64 open_tick) const noexcept {
        const u64 tick = requested.load(std::memory_order_relaxed);
        return tick != 0 && tick >= open_tick;
    }

    /// Whether an urgent request's tick is still open.
    [[nodiscard]] bool Urgent(u64 open_tick) const noexcept {
        const u64 tick = urgent_tick.load(std::memory_order_relaxed);
        return tick != 0 && tick >= open_tick;
    }

    enum class IdleAction : u8 {
        None,     ///< nothing to submit
        Deferred, ///< DeferFlush's work: submit
        Urgent,   ///< a thread blocks on the open tick: submit
        Leave,    ///< open work nobody blocks on: a pass boundary, a chunk or the sleep point
    };
    /// F4-X2 / SYNC_SUBMIT_SPEC §11 (MASTER_PLAN §5.5): the split's idle hook
    /// (Scheduler::ServeSplitIdleHook) serves DeferFlush's work and an urgent request (rule W).
    /// Other open work is left: rule I exists only as invariant S at the back's sleep point
    /// (FlushHeuristic::SleepSubmitDue), never at a wake or between entries. Waking the hook
    /// (RequestIdleHook) says nothing by itself (f1d05e9625: 54 idle-hook submits a frame).
    [[nodiscard]] IdleAction AtIdleHook(bool flush_pending, bool lazy_fences, bool wanted,
                                        u64 lazy_fence_tick, u64 open_tick) const noexcept {
        if (!IdleSubmits(flush_pending, lazy_fences, wanted, lazy_fence_tick, open_tick)) {
            return IdleAction::None;
        }
        if (flush_pending) {
            return IdleAction::Deferred;
        }
        if (Urgent(open_tick)) {
            return IdleAction::Urgent;
        }
        return IdleAction::Leave;
    }

    /// The idle hook's decision (the recording thread goes idle, or a waiter woke it). Submits
    /// deferred work (DeferFlush), a requested open tick, and with lazy fences any lazy fence's
    /// open tick or work marked as wanted: nothing else would submit them while it sleeps.
    [[nodiscard]] bool IdleSubmits(bool flush_pending, bool lazy_fences, bool wanted,
                                   u64 lazy_fence_tick, u64 open_tick) const noexcept {
        if (flush_pending || Open(open_tick)) {
            return true;
        }
        return lazy_fences && (wanted || (lazy_fence_tick != 0 && lazy_fence_tick >= open_tick));
    }

private:
    static void Raise(std::atomic<u64>& value, u64 tick) noexcept {
        u64 current = value.load(std::memory_order_relaxed);
        while (current < tick &&
               !value.compare_exchange_weak(current, tick, std::memory_order_relaxed)) {
        }
    }

    std::atomic<u64> requested{0};
    std::atomic<u64> urgent_tick{0};
};

} // namespace VideoCommon::V2
