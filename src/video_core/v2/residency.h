// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"

namespace VideoCommon::V2 {

/// Pipeline v2 F2-09 (MASTER_PLAN §5.3 textures, §8): image residency by epoch. A recipe or a
/// continued chain prepares its sampled views once under a stamp; while the texture cache's
/// current stamp equals it, preparing them again would only repeat the LRU touch.
struct ResidencyStamp {
    u64 epoch{};      ///< the texture cache's prepare epoch (0: never prepared)
    u64 frame{};      ///< LRU touches and frame marks are per frame
    u64 submission{}; ///< COMMIT's v2_track_id is per submission
    u64 barriers{};   ///< DYNRENDER: the barrier tracker's write serial (0: nothing pending)

    bool operator==(const ResidencyStamp&) const = default;
};

} // namespace VideoCommon::V2
