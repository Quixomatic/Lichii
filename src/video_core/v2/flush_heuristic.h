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

    /// F4-03 (MASTER_PLAN §5.5, P5b-F1): decide at hint events, i.e. pass boundaries (render
    /// target changes, a dispatch ending a pass), instead of at every chunk. Chunk boundaries then
    /// keep only the safety nets: max draws, the cost cap, strong hints, a waiter's request and a
    /// long mid-pass interval.
    bool pass_boundaries = false;
    u32 pass_min_draws = 10;          ///< Ryujinx ShouldFlushAttachmentChange: at least 10 draws
    u64 pass_interval_us = 2000;      ///< ... and this long since the last submission
    u64 request_gap_us = 500;         ///< a waiter's request mid-pass, no sooner than this
    u64 mid_pass_interval_us = 8000;  ///< safety net for passes that never end
};

/// What the heuristic knows about the GPU and the recording thread at a decision.
struct FlushState {
    u32 pending = 1;           ///< handed-off submissions the GPU has not finished
    bool back_behind = false;  ///< the recording thread has more work queued (CPU-bound)
    /// F4-02: a thread waits for work recorded in the open submission (a lazy fence whose window
    /// ran out, a guest wait): the next pass boundary serves it, mid-pass after request_gap_us.
    bool requested = false;
    /// F4-X2: a thread blocks on the open tick (an urgent request). Only this cuts a pass
    /// mid-chunk; a lazy fence's bounded-time request waits for the pass boundary (019d:
    /// 18.7 chunk submits/f).
    bool urgent = false;
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
        ReasonPass,    ///< F4-03: a pass boundary after enough draws and time (Ryujinx)
        ReasonRequest, ///< F4-02: a waiter's request
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

    /// F4-03: the next item ends the active pass (`level`: Weak for a render target change,
    /// Strong for RT0, DXVK's D3D9 rule). Called before that item is recorded, so a submission here
    /// cuts no pass. Ryujinx: at least pass_min_draws and pass_interval_us since the last
    /// submission; DXVK: the pending rule and the idle rule; any waiter's request.
    Decision AtPassBoundary(FlushHint level, const FlushState& state, u64 now_us) noexcept {
        hint = (std::max)(hint, level);
        if (draws == 0) {
            return {};
        }
        if (last_submit_us == 0) {
            last_submit_us = now_us;
        }
        const u64 since = now_us - last_submit_us;
        const u32 chunks = draws / params.chunk_draws;
        Decision decision{};
        if (state.requested) {
            decision = {FlushAction::Flush, ReasonRequest};
        } else if (hint >= FlushHint::Strong && chunks >= params.strong_chunks) {
            decision = {FlushAction::Flush, ReasonStrong};
        } else if (fast_mode && !state.back_behind && since >= params.fast_interval_us) {
            decision = {FlushAction::Flush, ReasonFast};
        } else if (draws >= params.pass_min_draws && since >= params.pass_interval_us) {
            decision = {FlushAction::Flush, ReasonPass};
        } else if (draws >= params.min_draws) {
            if (hint >= FlushHint::Weak && chunks >= params.weak_chunks) {
                decision = {FlushAction::Flush, ReasonWeak};
            } else if (state.pending >= params.busy_pending &&
                       chunks >= (std::min)(params.max_chunk_threshold, state.pending * 3)) {
                decision = {FlushAction::Flush, ReasonPending};
            } else if (state.pending == 0 && !state.back_behind &&
                       since >= params.idle_min_interval_us) {
                decision = {FlushAction::Flush, ReasonIdle};
            }
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

    /// A strong hint decided right away (after work the guest reads back, not on a draw). DXVK's
    /// ImplicitStrongHint: enough chunks recorded, "aggressively to reduce readback latency",
    /// whether the GPU is busy or not; no idle shortcut (that flushed after 16 draws).
    [[nodiscard]] bool WantsFlushNow() const noexcept {
        return hint >= FlushHint::Strong && draws >= params.strong_chunks * params.chunk_draws;
    }

    /// F4-X2 (MASTER_PLAN §5.5; DXVK ImplicitSynchronization, util_flush.cpp): the recording
    /// thread's idle hook with open work no thread blocks on. While fewer than busy_pending
    /// submissions are in flight the GPU is about to go idle: submit once the work is worth one
    /// (Ryujinx's pass rule, pass_min_draws and pass_interval_us since the last submission, or
    /// min_draws). While it is busy: DXVK's threshold, min(20, pending * 3) chunks. Otherwise the
    /// work waits for the next pass boundary or chunk.
    [[nodiscard]] bool SyncPointDue(const FlushState& state, u64 now_us) const noexcept {
        if (draws == 0) {
            return false;
        }
        if (state.pending >= params.busy_pending) {
            return draws / params.chunk_draws >=
                   (std::min)(params.max_chunk_threshold, state.pending * 3);
        }
        const u64 since = last_submit_us == 0 ? 0 : now_us - last_submit_us;
        return draws >= params.min_draws ||
               (draws >= params.pass_min_draws && since >= params.pass_interval_us);
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
        if (params.pass_boundaries) {
            // SYNC_SUBMIT_SPEC §2.5 rule C (MP §5.5, DXVK GpuFlushTracker): mid-pass only the
            // pending rule. With busy_pending submissions in flight, min(20, pending × 3) chunks;
            // with fewer (the GPU is about to go idle) once the work is worth one (min_draws,
            // and Ryujinx's AutoFlushCounter interval since the last submission, so a pass is
            // cut at most once per pass_interval_us). No waiter's request (rule W is served at
            // the back's entry boundary), no hint (rule B weighs hints at the pass boundary).
            // Max draws and cost stay in Advance.
            if (state.pending >= params.busy_pending) {
                if (chunks >= (std::min)(params.max_chunk_threshold, state.pending * 3)) {
                    return {FlushAction::Flush, ReasonPending};
                }
                return {};
            }
            if (draws >= params.min_draws && since >= params.pass_interval_us) {
                return {FlushAction::Flush, ReasonIdle};
            }
            return {};
        }
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
