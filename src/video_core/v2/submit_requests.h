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
        None,      ///< nothing to submit
        Now,       ///< deferred work or a thread blocks on the open tick: submit
        AtGpuIdle, ///< open work nobody blocks on: submit while the GPU is idle, else at its idle
    };
    /// F4-X2: the split's idle hook (Scheduler::ServeSplitIdleHook). Only DeferFlush's work and
    /// an urgent request submit at once; waking the hook (RequestIdleHook) says nothing by itself.
    [[nodiscard]] IdleAction AtIdleHook(bool flush_pending, bool lazy_fences, bool wanted,
                                        u64 lazy_fence_tick, u64 open_tick) const noexcept {
        if (!IdleSubmits(flush_pending, lazy_fences, wanted, lazy_fence_tick, open_tick)) {
            return IdleAction::None;
        }
        return flush_pending || Urgent(open_tick) ? IdleAction::Now : IdleAction::AtGpuIdle;
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

/// F4-X2 (MASTER_PLAN §5.5 adaptive submission; DXVK: submit when the GPU is about to go idle):
/// open work that nobody waits for is left in the open submission while the GPU is busy, and the
/// finish thread wakes the recording thread's idle hook (as a request, which submits) once the
/// GPU has finished everything handed to it. Arm and OnProgress race: each side publishes its
/// state, fences, then reads the other's, so at least one of them sees the GPU idle with the
/// wake armed (the recording thread then submits itself, or the finish thread wakes it).
class GpuIdleWake {
public:
    /// Recording thread: open work waits for the GPU to go idle. False when it is idle already:
    /// the caller submits now.
    template <typename Busy>
    [[nodiscard]] bool Arm(Busy&& gpu_busy) noexcept {
        armed.store(true, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (gpu_busy()) {
            return true;
        }
        armed.store(false, std::memory_order_relaxed);
        return false;
    }

    /// Recording thread: a submission went out (nothing is left for the wake).
    void Disarm() noexcept {
        armed.store(false, std::memory_order_relaxed);
    }

    /// Finish thread, after it published a completed tick: whether to wake the recording thread
    /// now (once per Arm).
    template <typename Busy>
    [[nodiscard]] bool OnProgress(Busy&& gpu_busy) noexcept {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!armed.load(std::memory_order_relaxed) || gpu_busy()) {
            return false;
        }
        return armed.exchange(false, std::memory_order_relaxed);
    }

    [[nodiscard]] bool Armed() const noexcept {
        return armed.load(std::memory_order_relaxed);
    }

private:
    std::atomic<bool> armed{false};
};

} // namespace VideoCommon::V2
