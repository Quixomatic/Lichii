// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>

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
    /// SYNC §14 #2 (MP §5.5 "chunk count, pending GPU work"; DXVK GpuFlushTracker), in pass
    /// mode at every chunk (rule C) and pass boundary (rule B): chunks recorded before a
    /// submission with the GPU idle (with a guest-visible fence open: idle_guest_chunks), with
    /// one submission pending, and min(max_chunk_threshold, 3 × pending) with more.
    u32 idle_chunks = 2;
    u32 idle_guest_chunks = 1;
    u32 one_pending_chunks = 3;
    /// SYNC §14.7a: rule B's nothing-pending minimum, in recorded cost units (draws; a dispatch
    /// counts 4). 64 = the table's 2 chunks. EDEN_NX2_SYNC_IDLE_MIN_DRAWS sets it for the
    /// 10/32/64/128 sweep (an experiment switch, removed once the cost-based rule is set).
    u32 idle_min_cost = 64;
    u64 request_gap_us = 500;         ///< a waiter's request mid-pass, no sooner than this
    u64 mid_pass_interval_us = 8000;  ///< safety net for passes that never end
};

/// SYNC §14 #5 / §14.4, corrected by §14.5 (diagnostic only): the GPU's idle gaps between
/// consecutive submissions of a sampled frame, by length (< 0.25, 0.25-1, 1-3, > 3 ms), split by
/// whether recorded but unsubmitted work existed during the gap. All times are host
/// steady-clock ns; GPU timestamps are mapped to it first (CalibrateGpuToHost).
/// - "GPU idle with recorded but unsubmitted work": |[idle start, idle end] ∩ [the next
///   submission's first record, its queue time]|, the idle time our submission rules create;
/// - hand-off latency: [queue time, GPU begin], the submit thread's and driver's share.
/// 95a344cc12 counted min(gap, queue − record): the back records continuously, so the next
/// tick's first record usually precedes the GPU going idle and every gap counted as "with work".
struct IdleGapHistogram {
    std::array<u32, 4> with_work{};
    std::array<u32, 4> without_work{};
    double idle_with_work_ms = 0.0;
    double handoff_ms = 0.0;

    /// One gap [idle_begin, idle_end] (the previous submission's GPU end to this one's GPU
    /// begin) before a submission recorded from `recorded` (0 unknown) and queued at `queued`.
    void Add(s64 idle_begin, s64 idle_end, s64 recorded, s64 queued) {
        if (idle_end <= idle_begin) {
            return; // back to back, or overlapping timestamps
        }
        const double gap_ms = static_cast<double>(idle_end - idle_begin) / 1e6;
        const size_t bucket = gap_ms < 0.25 ? 0 : gap_ms < 1.0 ? 1 : gap_ms < 3.0 ? 2 : 3;
        s64 overlap = 0;
        if (recorded != 0 && queued > recorded) {
            const s64 begin = (std::max)(idle_begin, recorded);
            const s64 end = (std::min)(idle_end, queued);
            overlap = end > begin ? end - begin : 0;
        }
        ++(overlap > 0 ? with_work : without_work)[bucket];
        idle_with_work_ms += static_cast<double>(overlap) / 1e6;
        if (queued != 0 && idle_end > queued) {
            handoff_ms += static_cast<double>(idle_end - (std::max)(queued, idle_begin)) / 1e6;
        }
    }
};

/// SYNC §14.5: the host-clock offset of GPU timestamps (in ns) without VK_EXT_calibrated
/// timestamps (not wired in this build): a submission can't start on the GPU before it was
/// queued, so offset ≥ queued − gpu_begin for each; the tightest bound is the largest, exact up
/// to the smallest hand-off latency of the frame (a submission queued to an idle GPU).
[[nodiscard]] inline s64 CalibrateGpuToHost(const s64* gpu_begin, const s64* queued,
                                            size_t count) {
    s64 offset = (std::numeric_limits<s64>::min)();
    for (size_t i = 0; i < count; ++i) {
        if (queued[i] != 0) {
            offset = (std::max)(offset, queued[i] - gpu_begin[i]);
        }
    }
    return offset == (std::numeric_limits<s64>::min)() ? 0 : offset;
}

/// SYNC §14 #1: handed-off submissions the GPU has not finished, from the open tick
/// (`current_tick`, the one being recorded) and the GPU's timeline value (the newest finished).
[[nodiscard]] constexpr u32 PendingSubmissions(u64 current_tick, u64 gpu_value) noexcept {
    return static_cast<u32>(current_tick > gpu_value + 1 ? current_tick - gpu_value - 1 : 0);
}

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
    /// SYNC §14 #2: the open submission holds a fence carrying a guest-visible result.
    bool guest_fence = false;
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
    /// cuts no pass. A waiter's request, a strong hint after strong_chunks, then SYNC §14 #2's
    /// DXVK table (TableChunks), then a weak hint after weak_chunks. No time terms.
    Decision AtPassBoundary(FlushHint level, const FlushState& state, u64 now_us) noexcept {
        hint = (std::max)(hint, level);
        if (draws == 0) {
            return {};
        }
        const u32 chunks = draws / params.chunk_draws;
        Decision decision{};
        if (state.requested) {
            decision = {FlushAction::Flush, ReasonRequest};
        } else if (hint >= FlushHint::Strong && chunks >= params.strong_chunks) {
            decision = {FlushAction::Flush, ReasonStrong};
        } else if (state.pending == 0 ? cost_units >= IdleMinCost(state)
                                      : RecordedChunks() >= TableChunks(state)) {
            decision = {FlushAction::Flush, state.pending == 0 ? ReasonIdle : ReasonPending};
        } else if (hint >= FlushHint::Weak && chunks >= params.weak_chunks) {
            decision = {FlushAction::Flush, ReasonWeak};
        }
        if (decision.action == FlushAction::Flush) {
            NoteSubmit(now_us);
        }
        return decision;
    }

    /// SYNC §14 #2 (DXVK GpuFlushTracker): the chunks a submission needs with `state.pending`
    /// handed-off submissions unfinished.
    [[nodiscard]] u32 TableChunks(const FlushState& state) const noexcept {
        if (state.pending == 0) {
            return state.guest_fence ? params.idle_guest_chunks : params.idle_chunks;
        }
        if (state.pending == 1) {
            return params.one_pending_chunks;
        }
        return (std::min)(params.max_chunk_threshold, state.pending * 3);
    }
    /// SYNC §14.7a: rule B's nothing-pending minimum (cost units): idle_min_cost, or the guest-read
    /// row's chunk if that is smaller. The final rule (record time ≥ k × measured submit cost)
    /// follows the sweep.
    [[nodiscard]] u32 IdleMinCost(const FlushState& state) const noexcept {
        const u32 guest = params.idle_guest_chunks * params.chunk_draws;
        return state.guest_fence ? (std::min)(params.idle_min_cost, guest) : params.idle_min_cost;
    }
    /// Chunks recorded since the last submission, in cost units (a dispatch counts 4 draws).
    [[nodiscard]] u32 RecordedChunks() const noexcept {
        return cost_units / params.chunk_draws;
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

    /// SYNC_SUBMIT_SPEC §11.2 invariant S (rule I only at the back's sleep point): the back may
    /// sleep on a non-empty open tick (`open_work`: a command or a lazy fence since the last
    /// submission) only with busy_pending submissions in flight and a completion wake guaranteed
    /// (`wake_guaranteed`: the finish thread publishes the GPU tick and wakes the back). Otherwise
    /// the open tick is due now.
    [[nodiscard]] bool SleepSubmitDue(u32 pending, bool open_work,
                                      bool wake_guaranteed) const noexcept {
        return open_work && (!wake_guaranteed || pending < params.busy_pending);
    }

    /// SYNC_SUBMIT_SPEC §14 #3 rule I′ (MP §5.5 "strong hints after work the guest will read",
    /// DXVK's "GPU idle → flush"): at the back's batch end (its ring ran empty) the open tick is
    /// due when it holds a guest-visible fence and nothing is pending. Bounded: after it, pending
    /// is 1 until that submission completes, so at most once per completed submission (§10's
    /// batch-end rule fired at every ring-empty moment, 27-29/f).
    [[nodiscard]] static constexpr bool BatchEndSubmitDue(u32 pending, bool open_work,
                                                          bool guest_fence) noexcept {
        return open_work && guest_fence && pending == 0;
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
            // SYNC_SUBMIT_SPEC §14.6 rule C (MP §5.5; DXVK evaluates GpuFlushTracker only at
            // hint events, its chunk count is a threshold there, not a per-chunk trigger): §14 #2
            // put the table at every chunk mid-pass, a submit every 2 chunks with the GPU idle
            // (95a3: C 41.7/f). Mid-pass only a pass that never ends with the GPU busy: pending
            // >= busy_pending and min(20, 3 × pending) chunks. The table's pending 0/1 rows run
            // at pass boundaries (rule B), strong hints and the batch end (I′). Max draws and
            // cost stay in Advance. No waiter's request (rule W is served at entry boundaries).
            (void)chunks;
            if (state.pending >= params.busy_pending && RecordedChunks() >= TableChunks(state)) {
                return {FlushAction::Flush, ReasonPending};
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
