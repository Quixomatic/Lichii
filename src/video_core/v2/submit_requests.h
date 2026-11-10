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
    /// Any thread: `tick` must be submitted.
    void Request(u64 tick) noexcept {
        u64 current = requested.load(std::memory_order_relaxed);
        while (current < tick &&
               !requested.compare_exchange_weak(current, tick, std::memory_order_relaxed)) {
        }
    }

    /// Whether a requested tick is still open.
    [[nodiscard]] bool Open(u64 open_tick) const noexcept {
        const u64 tick = requested.load(std::memory_order_relaxed);
        return tick != 0 && tick >= open_tick;
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
    std::atomic<u64> requested{0};
};

} // namespace VideoCommon::V2
