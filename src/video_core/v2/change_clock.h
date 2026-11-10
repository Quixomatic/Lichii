// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>

#include "common/common_types.h"

namespace VideoCommon::V2 {

/// Fix F3d (MASTER_PLAN §5.1, the memo's "relevant register fingerprint"): when a register file
/// last changed, by blocks of 64 registers. Each change of a register or of its shadow RAM copy
/// stamps its block with the next tick; Invalidate stands for a change of every register (a
/// copy or resync outside the write path). "No register in blocks `mask` changed after `tick`"
/// is then one compare per block, and exact: no hash can collide.
struct ChangeClock {
    static constexpr u32 Blocks = 64; ///< 4096 registers

    [[nodiscard]] static constexpr u64 Block(u32 index) {
        return u64{1} << ((index >> 6) & (Blocks - 1));
    }

    void Change(u32 index) {
        stamps[(index >> 6) & (Blocks - 1)] = ++now;
    }
    void Invalidate() {
        floor = ++now;
    }
    [[nodiscard]] u64 Now() const {
        return now;
    }
    /// Whether nothing in `mask`'s blocks changed after `tick` (a Now() taken earlier).
    [[nodiscard]] bool Unchanged(u64 mask, u64 tick) const {
        if (tick < floor) {
            return false;
        }
        for (u64 bits = mask; bits != 0; bits &= bits - 1) {
            if (stamps[static_cast<size_t>(std::countr_zero(bits))] > tick) {
                return false;
            }
        }
        return true;
    }

private:
    u64 now{1};
    u64 floor{1};
    std::array<u64, Blocks> stamps{};
};

} // namespace VideoCommon::V2
