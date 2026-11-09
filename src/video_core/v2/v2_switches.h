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

/// Reads every switch again and logs the ones set. Called by VideoCore::CreateGPU.
void ReadSwitches();

} // namespace VideoCommon::V2
