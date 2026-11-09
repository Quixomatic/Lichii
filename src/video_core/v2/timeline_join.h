// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>

#include "common/common_types.h"

// P5b (EDEN_NX2_V2_XFERQ / _XREADBACK): bookkeeping for the joins between the graphics timeline
// and the transfer queue. Pure logic, owned by the recording thread (the scheduler's owner).
//
// Timelines. The graphics timeline counts scheduler ticks: submission T signals T. The transfer
// batch recorded alongside submission T signals the transfer timeline with T too, and submission
// T waits for it. Readback jobs run on their own per-thread timelines (BackgroundResources) and
// add an explicit wait to the next graphics submission.
//
// DXVK's rule decides what may run on the transfer queue: a resource not used by the open
// submission may go out of order (its transfer waits for the graphics tick that last used it);
// anything the open submission already used stays in order.
//
// The multi-queue caveat for class B (MASTER_PLAN §6): an acquire the front satisfied by a
// release ahead of it on the host queue is ordered only by that one in-order queue. Once the
// acquire's position executes on the back, transfer work recorded after it must wait for the
// graphics work before the release: the open submission can't be waited for by its own transfer
// batch (that submission waits for the batch), so hoisting stops until the next submission,
// whose batch then waits for the acquiring tick.
namespace VideoCommon::V2 {

class TimelineJoin {
public:
    struct Wait {
        u64 semaphore = 0; ///< opaque handle (VkSemaphore)
        u64 value = 0;
    };
    static constexpr size_t MaxExtraWaits = 8;
    struct PostJob {
        u64 cmdbuf = 0;
        u64 semaphore = 0;
        u64 value = 0;
    };
    static constexpr size_t MaxPostJobs = 4;

    /// What one submission needs, computed when it closes.
    struct Batch {
        u64 tick = 0;
        bool transfer = false;        ///< a transfer batch goes before the graphics submission
        u64 transfer_wait = 0;        ///< it waits for graphics >= this (0: no wait)
        u64 transfer_signal = 0;      ///< and signals the transfer timeline with this
        bool graphics_waits_transfer = false;
        u32 num_extra_waits = 0;      ///< readback jobs the graphics submission waits for
        std::array<Wait, MaxExtraWaits> extra_waits{};
        u32 num_post_jobs = 0;        ///< readback jobs submitted right after this submission
        std::array<PostJob, MaxPostJobs> post_jobs{};
    };

    /// Perf fix C: a readback job (opaque command buffer, its timeline and value) the submit
    /// path queues on the transfer queue right after the open submission, waiting for its tick.
    /// False when every slot is taken: the caller submits it itself.
    [[nodiscard]] bool AddPostJob(u64 cmdbuf, u64 semaphore, u64 value) noexcept {
        if (num_post_jobs == post_jobs.size()) {
            return false;
        }
        post_jobs[num_post_jobs++] = {cmdbuf, semaphore, value};
        return true;
    }

    /// Whether a transfer op on a resource last used by graphics tick `track` may join the open
    /// submission's transfer batch.
    [[nodiscard]] bool CanHoist(u64 track, u64 open_tick) const noexcept {
        return track < open_tick && acquire_floor < open_tick && !blocked;
    }

    /// Records a transfer op on such a resource: the batch waits for its last graphics use.
    void Hoist(u64 track, u64 open_tick) noexcept {
        has_transfer = true;
        transfer_wait = (std::max)({transfer_wait, track, acquire_floor});
        (void)open_tick;
    }

    /// Perf fix C: an op that reads no GPU-produced data and writes nothing queued work uses (a
    /// buffer upload into an unused region) joins the batch without any graphics wait, so the
    /// batch overlaps earlier submissions. Only a blocked batch (an ownership release recorded
    /// in it) refuses.
    [[nodiscard]] bool CanHoistUnordered() const noexcept {
        return !blocked;
    }
    void HoistUnordered() noexcept {
        has_transfer = true;
    }

    /// An acquire's position executed on the back during open tick `open_tick` (class A/B with
    /// the transfer queue on): later transfer work waits for every graphics tick up to it.
    void NoteAcquire(u64 open_tick) noexcept {
        acquire_floor = (std::max)(acquire_floor, open_tick);
    }

    /// Stops hoisting until the next submission (e.g. a queue-family release recorded in it).
    void BlockHoisting() noexcept {
        blocked = true;
    }

    /// The next graphics submission waits for `value` on `semaphore` (a readback job reading a
    /// resource graphics work after it may overwrite). Waits on one semaphore keep the newest.
    /// False when every slot holds another semaphore: the caller takes the in-order path.
    [[nodiscard]] bool AddGraphicsWait(u64 semaphore, u64 value) noexcept {
        for (u32 i = 0; i < num_waits; ++i) {
            if (waits[i].semaphore == semaphore) {
                waits[i].value = (std::max)(waits[i].value, value);
                return true;
            }
        }
        if (num_waits == waits.size()) {
            return false;
        }
        waits[num_waits++] = {semaphore, value};
        return true;
    }

    [[nodiscard]] bool HasTransferWork() const noexcept {
        return has_transfer;
    }
    [[nodiscard]] u64 AcquireFloor() const noexcept {
        return acquire_floor;
    }

    /// Closes the open submission `tick` and resets for the next one.
    Batch Close(u64 tick) noexcept {
        Batch batch{};
        batch.tick = tick;
        batch.transfer = has_transfer;
        batch.transfer_wait = has_transfer ? transfer_wait : 0;
        batch.transfer_signal = has_transfer ? tick : 0;
        batch.graphics_waits_transfer = has_transfer;
        batch.num_extra_waits = num_waits;
        batch.extra_waits = waits;
        batch.num_post_jobs = num_post_jobs;
        batch.post_jobs = post_jobs;
        num_post_jobs = 0;
        has_transfer = false;
        transfer_wait = 0;
        num_waits = 0;
        blocked = false;
        return batch;
    }

private:
    u64 acquire_floor = 0;
    u64 transfer_wait = 0;
    bool has_transfer = false;
    bool blocked = false;
    u32 num_waits = 0;
    std::array<Wait, MaxExtraWaits> waits{};
    u32 num_post_jobs = 0;
    std::array<PostJob, MaxPostJobs> post_jobs{};
};

/// A per-thread timeline's values for readback jobs (BackgroundResources): monotonic, and
/// completion seen without asking the driver when a known value already passed.
class JobTimeline {
public:
    [[nodiscard]] u64 Next() noexcept {
        return ++last_issued;
    }
    [[nodiscard]] u64 LastIssued() const noexcept {
        return last_issued;
    }
    void NoteCompleted(u64 value) noexcept {
        completed = (std::max)(completed, value);
    }
    [[nodiscard]] bool IsComplete(u64 value) const noexcept {
        return completed >= value;
    }

private:
    u64 last_issued = 0;
    u64 completed = 0;
};

} // namespace VideoCommon::V2
