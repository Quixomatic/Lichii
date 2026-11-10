// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>

#include "common/common_types.h"

// Pipeline v2 switches (MASTER_PLAN.md). Each is EDEN_NX2_V2_<NAME> = 0 (default: the old
// behaviour), 1 (on) or verify (the new and the old path both run and are compared).
// EDEN_NX2_PIPELINE=v1 forces every one of them off. The values are read at static
// initialization and again by ReadSwitches (VideoCore::CreateGPU, once the frontend loaded
// nx2_switches.txt); the hot paths only read the cached array.
namespace VideoCommon::V2 {

enum class Mode : u8 {
    Off,
    On,
    Verify,
};

/// One entry per switch; add new ones before Count and their name to switch_names.
enum class Switch : u32 {
    Stats,  ///< EDEN_NX2_V2_STATS: P0 measurement counters (v2_stats.h)
    Ir,     ///< EDEN_NX2_V2_IR: typed IR records in the split ring (v2/ir.h, P1)
    RegSem, ///< EDEN_NX2_V2_REGSEM: table-driven register semantics (v2/regsem.h, P1)
    Acquire, ///< EDEN_NX2_V2_ACQUIRE: semaphore acquires from the front's view (v2/hazard.h, P2)
    Hazard,  ///< EDEN_NX2_V2_HAZARD: range hazards and targeted waits for front reads (P2)
    Spin,    ///< EDEN_NX2_V2_SPIN: hand-off spin window before front/back kernel waits (P2)
    CpuLane, ///< EDEN_NX2_V2_CPULANE: guest CPU flush requests served by the back early (P2)
    Readback, ///< EDEN_NX2_V2_READBACK: requester-side readbacks and learned early downloads
    MacroDump, ///< EDEN_NX2_V2_MACRO_DUMP: log NVN2 macros' hash and code (v2/macro_hle.h)
    MacroHle,  ///< EDEN_NX2_V2_MACROHLE: NVN2 macro HLE from nx2_macro_hle.txt entries
    IrDraws,   ///< EDEN_NX2_V2_IR_DRAWS: front Draw/Dispatch IR records, checked by the back
    PipeKey,   ///< EDEN_NX2_V2_PIPEKEY: incremental pipeline key, two-level lookup (P3)
    ContDraw,  ///< EDEN_NX2_V2_CONTDRAW: continued draws decided from the register groups (P3)
    DescCache, ///< EDEN_NX2_V2_DESCCACHE: content-keyed descriptor reuse (P3)
    Bindless,  ///< EDEN_NX2_V2_BINDLESS: TIC/TSC pools mirrored as a host descriptor heap (P3)
    CbufVer,   ///< EDEN_NX2_V2_CBUFVER: constant buffer versions instead of per-draw copies (P3)
    MarkDirty, ///< EDEN_NX2_V2_MARKDIRTY: a submit only marks command buffer state dirty (P3)
    LeanDraw,  ///< EDEN_NX2_V2_LEANDRAW: per-draw debug hooks behind one cached gate (P3)
    DrawCache, ///< EDEN_NX2_V2_DRAWCACHE: fingerprint -> recipe draw/dispatch cache (P4)
    Commit,    ///< EDEN_NX2_V2_COMMIT: pure cache lookups + serial commit stage, no global lock
    CommitLocked, ///< EDEN_NX2_V2_COMMIT_LOCKED: with COMMIT=1, keep the global cache locks
    Writeback, ///< EDEN_NX2_V2_WRITEBACK: deferred GPU->guest writebacks queued in commit (P5a)
    HeapTexel, ///< EDEN_NX2_V2_HEAP_TEXEL: texel buffers in the bindless heap (P5a)
    XferQ,     ///< EDEN_NX2_V2_XFERQ: dedicated transfer queue for uploads, timeline joins (P5b)
    XReadback, ///< EDEN_NX2_V2_XREADBACK: readbacks on the transfer queue, downloads in the
               ///< writer's submission (P5b)
    QmdSched,  ///< EDEN_NX2_V2_QMDSCHED: front QMD scheduler, exact scheduled dispatch counts
    GpuDecode, ///< EDEN_NX2_V2_GPUDECODE: compute untile as the default texture upload (P5b)
    DynRender, ///< EDEN_NX2_V2_DYNRENDER: dynamic rendering + guest-range barrier tracker (P5b)
    AdaptSubmit,  ///< EDEN_NX2_V2_ADAPTSUBMIT: adaptive flush heuristic instead of 128 draws
    SubmitThread, ///< EDEN_NX2_V2_SUBMITTHREAD: submit thread + finish thread (P5b)
    PresentPace,  ///< EDEN_NX2_V2_PRESENTPACE: bounded frames ahead with present wait (P5b)
    Parallel,     ///< EDEN_NX2_V2_PARALLEL: segmented parallel resolve, ordered merge (P6a)
    MultiFront,   ///< EDEN_NX2_V2_MULTIFRONT: front engines over channel groups (P6b)
    MacroMemo,    ///< EDEN_NX2_V2_MACROMEMO: front macro calls replayed (v2/macro_memo.h, F5)
    DrawRec,      ///< EDEN_NX2_V2_DRAWREC: ResolveDraw / ApplyDraw / EncodeDraw (P6 Step 0)
    CbufRing,     ///< EDEN_NX2_V2_CBUFRING: decode-time constant buffer versions (F2-02)
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(Switch::Count)> switch_names{
    "STATS",
    "IR",
    "REGSEM",
    "ACQUIRE",
    "HAZARD",
    "SPIN",
    "CPULANE",
    "READBACK",
    "MACRO_DUMP",
    "MACROHLE",
    "IR_DRAWS",
    "PIPEKEY",
    "CONTDRAW",
    "DESCCACHE",
    "BINDLESS",
    "CBUFVER",
    "MARKDIRTY",
    "LEANDRAW",
    "DRAWCACHE",
    "COMMIT",
    "COMMIT_LOCKED",
    "WRITEBACK",
    "HEAP_TEXEL",
    "XFERQ",
    "XREADBACK",
    "QMDSCHED",
    "GPUDECODE",
    "DYNRENDER",
    "ADAPTSUBMIT",
    "SUBMITTHREAD",
    "PRESENTPACE",
    "PARALLEL",
    "MULTIFRONT",
    "MACROMEMO",
    "DRAWREC",
    "CBUFRING",
};

namespace Detail {
std::array<Mode, static_cast<size_t>(Switch::Count)> ReadAll();
}

inline std::array<Mode, static_cast<size_t>(Switch::Count)> g_modes = Detail::ReadAll();

/// The switch's cached mode.
[[nodiscard]] inline Mode Get(Switch name) {
    return g_modes[static_cast<size_t>(name)];
}
/// On or Verify: one load and compare of the cached mode.
[[nodiscard]] inline bool On(Switch name) {
    return g_modes[static_cast<size_t>(name)] != Mode::Off;
}
[[nodiscard]] inline bool Verify(Switch name) {
    return g_modes[static_cast<size_t>(name)] == Mode::Verify;
}

/// Whether the executing Maxwell3D keeps its register group tracker (V2::GroupTracker):
/// EDEN_NX2_V2_REGSEM, or a P3/P4 consumer of the dirty groups and fingerprints (PIPEKEY,
/// CONTDRAW, DRAWCACHE): four bytes of the cached modes, tested per register write.
[[nodiscard]] inline bool TrackGroups() {
    return On(Switch::RegSem) || On(Switch::PipeKey) || On(Switch::ContDraw) ||
           On(Switch::DrawCache);
}

/// Reads every switch again and logs the ones set. Called by VideoCore::CreateGPU.
void ReadSwitches();

} // namespace VideoCommon::V2
