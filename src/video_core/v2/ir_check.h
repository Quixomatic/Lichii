// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/v2/ir.h"
#include "video_core/v2/regsem.h"

// EDEN_NX2_V2_IR_DRAWS=verify (Pipeline v2 P1b): the split's back checks that every draw and
// dispatch it performs matches a Draw/Dispatch record the front emitted before it.
//   Draws: each record queues its predicted draws; each draw the back's draw manager performs
//   (before conditional rendering decides) takes the oldest one and compares kind, topology,
//   first, count, instances, base vertex and base instance (indirect draws: kind and topology),
//   and, when both sides track register groups, the state fingerprint. A draw with nothing
//   queued is unexpected; a prediction still queued 256 records later is missing.
//   Dispatches: a Dispatch record opens a window over the method after it; the dispatches in
//   the window must number within [min_dispatches, max_dispatches], and a dispatch outside any
//   window is unexpected.
// Mismatches log "V2 IR draw check" / "V2 IR dispatch check" warnings (first 32 of each), and a
// summary every 65536 checked draws. Only the back touches this state.
namespace VideoCommon::V2::IrCheck {

/// Set while the split runs with EDEN_NX2_V2_IR_DRAWS=verify (the back's hooks test it).
inline bool g_active{};

void Start();
void Stop();

/// The back starts an IR record (engine selections don't count).
void OnRecord(Ir::Kind kind);
/// The back runs a Draw record (any op).
void NoteDrawRecord(const Ir::DrawBody& body);
/// The back runs a Dispatch record.
void NoteDispatchRecord(const Ir::DispatchBody& body);

/// The back's draw manager performs a draw (`groups`: its register groups when valid).
void NoteDraw(const Ir::PredictedDraw& draw, const GroupTracker* groups);
/// The back's compute engine dispatches (`qmd_address`: 0 for the Kepler launch).
void NoteDispatch(u64 qmd_address);

struct Counters {
    u64 draws_matched;
    u64 draws_mismatched;
    u64 draws_unexpected;
    u64 draws_missing;
    u64 fingerprints_compared;
    u64 fingerprints_mismatched;
    u64 dispatch_windows;
    u64 dispatches;
    u64 dispatches_unexpected;
    u64 dispatch_count_mismatches;
};
/// For tests and the stats: what was checked since Start.
[[nodiscard]] Counters GetCounters();
/// Predicted draws not performed yet.
[[nodiscard]] size_t PendingDraws();

} // namespace VideoCommon::V2::IrCheck
