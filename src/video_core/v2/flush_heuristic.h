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

/// Perf fix C: thresholds are in draws since the last submission of any kind, so heuristic
/// submits are at least as big as the old 128-draw cadence (DXVK's chunks are hundreds of
/// commands; at 32 draws, "3 chunks" made 16-96 draw submissions). Chunks only pace the worker.
struct FlushParams {
    u32 chunk_draws = 32;       ///< draws per chunk (a Dispatch every chunk)
    u32 min_draws = 128;        ///< never submit fewer draws than this on a no-hint heuristic
    u32 max_draws = 1024;       ///< always submit at this many draws
    u32 weak_chunks = 6;        ///< DXVK: weak hints after 6 chunks
    u32 strong_chunks = 3;      ///< DXVK: strong hints after 3 chunks (class C/D work only)
    u32 busy_pending = 2;       ///< DXVK: the pending threshold rule needs this many in flight
    /// The idle rule needs the GPU really idle (every handed-off submission complete, by its
    /// timeline) and the recording thread not behind; and this long since the last submission.
    u64 idle_min_interval_us = 1000;
    u32 max_chunk_threshold = 20;
    u32 cost_cap = 1500;        ///< DXVK: forced at a cost estimate of 1500
    u64 max_interval_us = 4000; ///< safety net: submit when this long passed with work recorded
    u64 fast_interval_us = 1500;///< Ryujinx fast-flush mode: every 1.5 ms of draws
    /// Average per-frame time the guest waited on work not yet submitted to the host GPU that
    /// enters fast-flush mode, and leaves it.
    u64 fast_enter_wait_us = 1500;
    u64 fast_exit_wait_us = 100;
};

/// What the heuristic knows about the GPU and the recording thread at a decision.
struct FlushState {
    u32 pending = 1;           ///< handed-off submissions the GPU has not finished
    bool back_behind = false;  ///< the recording thread has more work queued (CPU-bound)
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

    /// One draw (cost 1), dispatch (cost 4) or other work item was recorded. `now_us` is a
    /// monotonic clock; it and `state` are only read at chunk boundaries (callers may pass
    /// zeros elsewhere).
    Decision Advance(u32 cost, const FlushState& state, u64 now_us) noexcept {
        ++draws;
        cost_units += cost;
        Decision decision{};
        if (draws >= params.max_draws) {
            decision = {FlushAction::Flush, ReasonMaxDraws};
        } else if (cost_units >= params.cost_cap) {
            decision = {FlushAction::Flush, ReasonCost};
        } else if (draws % params.chunk_draws == 0) {
            if (last_submit_us == 0) {
                last_submit_us = now_us;
            }
            const u32 chunks = draws / params.chunk_draws;
            decision = DecideAtChunk(chunks, state, now_us);
            if (decision.action == FlushAction::None) {
                decision.action = FlushAction::Dispatch;
            }
        }
        if (decision.action == FlushAction::Flush) {
            NoteSubmit(now_us);
        }
        return decision;
    }
    Decision Advance(u32 cost, u32 pending, u64 now_us) noexcept {
        return Advance(cost, FlushState{pending, false}, now_us);
    }

    /// A hint from the work just recorded; it is weighed at the next chunk boundary (or now for
    /// strong hints with enough chunks recorded).
    void Hint(FlushHint level) noexcept {
        hint = (std::max)(hint, level);
    }

    /// A strong hint decided right away (after work the guest reads back, not on a draw). DXVK's
    /// ImplicitStrongHint: enough chunks recorded; no idle shortcut (that flushed after 16 draws).
    [[nodiscard]] bool WantsFlushNow() const noexcept {
        return hint >= FlushHint::Strong && draws >= params.strong_chunks * params.chunk_draws;
    }

    /// Any submission (heuristic or not) resets the counters.
    void NoteSubmit(u64 now_us) noexcept {
        draws = 0;
        cost_units = 0;
        hint = FlushHint::None;
        last_submit_us = now_us;
    }

    /// Ryujinx's AutoFlushCounter: the guest waited `wait_us` this frame on syncs whose work was
    /// not submitted yet (not the whole wait: a CPU-bound recorder makes every wait long); the
    /// average over 20 frames enters or leaves fast-flush mode.
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
    Decision DecideAtChunk(u32 chunks, const FlushState& state, u64 now_us) const noexcept {
        const u64 since = now_us - last_submit_us;
        if (hint >= FlushHint::Strong && chunks >= params.strong_chunks) {
            return {FlushAction::Flush, ReasonStrong};
        }
        // Ryujinx: the guest is waiting on unsubmitted work. Not while the recorder is behind:
        // then the wait is the recorder's speed, and more submits only slow it further.
        if (fast_mode && !state.back_behind && since >= params.fast_interval_us) {
            return {FlushAction::Flush, ReasonFast};
        }
        if (draws < params.min_draws) {
            return {};
        }
        if (hint >= FlushHint::Weak && chunks >= params.weak_chunks) {
            return {FlushAction::Flush, ReasonWeak};
        }
        // DXVK: thresholds grow with the queued work: min(20, pending * 3) chunks.
        if (state.pending >= params.busy_pending &&
            chunks >= (std::min)(params.max_chunk_threshold, state.pending * 3)) {
            return {FlushAction::Flush, ReasonPending};
        }
        // The GPU has finished everything handed to it and the recorder is keeping up: feed it.
        // A CPU-bound recorder always leaves the GPU idle; submitting more then only costs the
        // recorder (Perf fix C: this rule was ~25-45 submissions a frame).
        if (state.pending == 0 && !state.back_behind && since >= params.idle_min_interval_us) {
            return {FlushAction::Flush, ReasonIdle};
        }
        if (since >= params.max_interval_us) {
            return {FlushAction::Flush, ReasonInterval};
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
