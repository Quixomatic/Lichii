// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"

// P5b (EDEN_NX2_V2_PRESENTPACE): bounded host frames ahead (DXVK maxFrameLatency, Cemu's
// present_wait queue depth, PCSX2's VsyncQueueSize). A frame is "in flight" from the composite
// that records it until it is retired: shown on the display (present wait) or, without present
// wait, finished by the GPU. The renderer waits before compositing a new frame while `limit`
// frames are in flight. Pure logic.
//
// The limit follows the per-title frames-ahead setting of vi2's present (nx2_gsv.h): 0 frames
// ahead → 1 in flight, 1 → 2; automatic → 2. Host pacing only ever adds waits: vi2's own fence
// waits (the command memory snapshot rule, notes/01 §8.7) stay as they are, so the guest never
// runs further ahead of the GPU emulation than before.
namespace VideoCommon::V2 {

[[nodiscard]] constexpr u32 FramesInFlightLimit(int frames_ahead_setting,
                                                u32 max_frames = 3) noexcept {
    const u32 wanted = frames_ahead_setting <= 0 ? (frames_ahead_setting == 0 ? 1U : 2U)
                                                 : static_cast<u32>(frames_ahead_setting) + 1;
    return (std::clamp)(wanted, 1U, (std::max)(max_frames, 1U));
}

class FramePacer {
public:
    void SetLimit(u32 limit_) noexcept {
        limit = (std::max)(limit_, 1U);
    }
    [[nodiscard]] u32 Limit() const noexcept {
        return limit;
    }

    /// A new frame is about to be composited: whether the caller must first wait for frame
    /// `WaitTarget()` to retire.
    [[nodiscard]] bool MustWait() const noexcept {
        return composited - retired >= limit;
    }
    /// The frame (1-based) that must retire before the next composite may start.
    [[nodiscard]] u64 WaitTarget() const noexcept {
        return composited + 1 - limit;
    }

    /// Returns the new frame's number (1-based).
    u64 NoteComposite() noexcept {
        return ++composited;
    }
    void NoteRetired(u64 frame) noexcept {
        retired = (std::max)(retired, (std::min)(frame, composited));
    }

    [[nodiscard]] u64 InFlight() const noexcept {
        return composited - retired;
    }
    [[nodiscard]] u64 Composited() const noexcept {
        return composited;
    }
    [[nodiscard]] u64 Retired() const noexcept {
        return retired;
    }

private:
    u32 limit = 2;
    u64 composited = 0;
    u64 retired = 0;
};

} // namespace VideoCommon::V2
