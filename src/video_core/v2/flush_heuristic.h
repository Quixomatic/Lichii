// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>

#include "common/common_types.h"

// P5b (EDEN_NX2_V2_ADAPTSUBMIT): when to submit, in place of the fixed 128-draw native cadence.
// DXVK's GpuFlushTracker (util_flush.cpp: graded hints, "flush when the GPU is about to go idle",
// thresholds that grow with the pending work, a cost cap) plus Ryujinx's AutoFlushCounter
// (fast-flush mode while the guest waits long on syncs; attachment changes after enough draws and
// time). Pure logic: the rasterizer feeds it per draw/dispatch and acts on the answer.
namespace VideoCommon::V2 {

enum class FlushHint : u8 {
    None,
    Weak,     ///< e.g. a render target change: a good place to split
    Strong,   ///< right after work the guest will read back or wait on
    Explicit, ///< the guest or a reader needs it now (handled by the caller: always flushes)
};

enum class FlushAction : u8 {
    None,
    Dispatch, ///< hand recorded chunks to the worker, no submit
    Flush,
};

struct FlushParams {
    u32 chunk_draws = 32;       ///< draws per chunk (a Dispatch every chunk)
    u32 min_draws = 16;         ///< never submit fewer draws than this on a heuristic
    u32 max_draws = 1024;       ///< always submit at this many draws
    u32 weak_chunks = 6;        ///< DXVK: weak hints after 6 chunks
    u32 strong_chunks = 3;      ///< DXVK: strong hints after 3 chunks
    u32 idle_pending = 2;       ///< fewer submissions pending than this: the GPU is about to idle
    /// Fix pass 1 (B8): the idle rule waits this long since the last submission. With a light GPU
    /// load the GPU is always about to idle, and the rule alone submitted every chunk (OoT: 23 to
    /// 44 submissions a frame).
    u64 idle_min_interval_us = 1000;
    u32 max_chunk_threshold = 20;
    u32 cost_cap = 1500;        ///< DXVK: forced at a cost estimate of 1500
    u64 max_interval_us = 2000; ///< submit when this long passed with work recorded
    u64 fast_interval_us = 1500;///< Ryujinx fast-flush mode: every 1.5 ms of draws
    u64 fast_enter_wait_us = 1500; ///< average guest sync wait that enters fast-flush mode
    u64 fast_exit_wait_us = 100;   ///< and leaves it
};

class FlushHeuristic {
public:
    struct Decision {
        FlushAction action = FlushAction::None;
        u8 reason = 0; ///< FlushReason (stats)
    };
    enum Reason : u8 {
        ReasonNone,
        ReasonMaxDraws,
        ReasonCost,
        ReasonIdle,
        ReasonPending,
        ReasonStrong,
        ReasonWeak,
        ReasonInterval,
        ReasonFast,
        ReasonCount,
    };

    explicit FlushHeuristic(const FlushParams& params_ = {}) : params{params_} {}

    /// One draw (cost 1), dispatch (cost 4) or other work item was recorded. `pending` is the
    /// number of submissions the GPU has not finished, `now_us` a monotonic clock.
    Decision Advance(u32 cost, u32 pending, u64 now_us) noexcept {
        ++draws;
        cost_units += cost;
        if (last_submit_us == 0) {
            last_submit_us = now_us;
        }
        Decision decision{};
        if (draws >= params.max_draws) {
            decision = {FlushAction::Flush, ReasonMaxDraws};
        } else if (cost_units >= params.cost_cap) {
            decision = {FlushAction::Flush, ReasonCost};
        } else if (draws % params.chunk_draws == 0) {
            const u32 chunks = draws / params.chunk_draws;
            decision = DecideAtChunk(chunks, pending, now_us);
            if (decision.action == FlushAction::None) {
                decision.action = FlushAction::Dispatch;
            }
        } else if (fast_mode && draws >= params.min_draws &&
                   now_us - last_submit_us >= params.fast_interval_us) {
            decision = {FlushAction::Flush, ReasonFast};
        }
        if (decision.action == FlushAction::Flush) {
            NoteSubmit(now_us);
        }
        return decision;
    }

    /// A hint from the work just recorded; it is weighed at the next chunk boundary (or now for
    /// strong hints with enough chunks recorded).
    void Hint(FlushHint level) noexcept {
        hint = (std::max)(hint, level);
    }

    /// A strong hint decided right away (after a readback-producing op, not on a draw).
    [[nodiscard]] bool WantsFlushNow(u32 pending) const noexcept {
        if (draws < params.min_draws) {
            return false;
        }
        const u32 chunks = draws / params.chunk_draws;
        return hint >= FlushHint::Strong &&
               (chunks >= params.strong_chunks || pending < params.idle_pending);
    }

    /// Any submission (heuristic or not) resets the counters.
    void NoteSubmit(u64 now_us) noexcept {
        draws = 0;
        cost_units = 0;
        hint = FlushHint::None;
        last_submit_us = now_us;
    }

    /// Ryujinx's AutoFlushCounter: the guest waited `wait_us` on a sync this frame; the average
    /// over 20 frames enters or leaves fast-flush mode.
    void NoteGuestSyncWait(u64 wait_us) noexcept {
        frame_wait_us += wait_us;
    }
    void NoteFrame() noexcept {
        average_wait_us = (average_wait_us * 19 + frame_wait_us) / 20;
        frame_wait_us = 0;
        if (!fast_mode && average_wait_us >= params.fast_enter_wait_us) {
            fast_mode = true;
        } else if (fast_mode && average_wait_us <= params.fast_exit_wait_us) {
            fast_mode = false;
        }
    }

    [[nodiscard]] bool FastMode() const noexcept {
        return fast_mode;
    }
    [[nodiscard]] u32 Draws() const noexcept {
        return draws;
    }
    [[nodiscard]] const FlushParams& Params() const noexcept {
        return params;
    }

private:
    Decision DecideAtChunk(u32 chunks, u32 pending, u64 now_us) const noexcept {
        if (draws < params.min_draws) {
            return {};
        }
        // DXVK: the GPU is about to go idle: feed it now.
        if (pending < params.idle_pending &&
            now_us - last_submit_us >= params.idle_min_interval_us) {
            return {FlushAction::Flush, ReasonIdle};
        }
        if (hint >= FlushHint::Strong && chunks >= params.strong_chunks) {
            return {FlushAction::Flush, ReasonStrong};
        }
        if (hint >= FlushHint::Weak && chunks >= params.weak_chunks) {
            return {FlushAction::Flush, ReasonWeak};
        }
        // Thresholds grow with the queued work: min(20, pending * 3) chunks. An idle GPU is the
        // idle rule's (Fix pass 1, B8: with nothing pending the threshold was 0, and every chunk
        // submitted whatever the idle rule decided).
        const u32 threshold = (std::min)(params.max_chunk_threshold, pending * 3);
        if (pending >= params.idle_pending && chunks >= threshold) {
            return {FlushAction::Flush, ReasonPending};
        }
        if (now_us - last_submit_us >= params.max_interval_us) {
            return {FlushAction::Flush, ReasonInterval};
        }
        if (fast_mode && now_us - last_submit_us >= params.fast_interval_us) {
            return {FlushAction::Flush, ReasonFast};
        }
        return {};
    }

    FlushParams params;
    u32 draws = 0;
    u32 cost_units = 0;
    FlushHint hint = FlushHint::None;
    u64 last_submit_us = 0;
    u64 frame_wait_us = 0;
    u64 average_wait_us = 0;
    bool fast_mode = false;
};

} // namespace VideoCommon::V2
