// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// Pipeline v2 P2: when fence-flagged releases reach guest memory. Since #337 the query cache writes
// a fence-flagged payload release only when its fence runs on the fencing thread, after the split's
// back ran the release. The back tags each method it runs with the ring position after it (the
// position the front noted the write at, GpuSplit::NoteKnownWrite), the query cache carries the
// tag into the fence operation, and the operation reports it here once the bytes are in memory.
// Fence operations run in order, so `landed` only grows, and every fenced release with a position
// at or below it is in memory. The front keeps such releases in a second hazard table until then
// (GpuSplit::FindUnlanded), which lets it forward their payloads instead of draining.
namespace VideoCommon::V2::Landing {

/// The split's back, while the front records writes (GpuSplit::TracksWrites): the ring position
/// after the method or record it runs. 0 elsewhere.
inline thread_local u64 tag = 0;

/// The newest tag whose fence-flagged release is in memory.
inline std::atomic<u64> landed{0};

inline void Note(u64 position) {
    u64 current = landed.load(std::memory_order_relaxed);
    while (current < position &&
           !landed.compare_exchange_weak(current, position, std::memory_order_release,
                                         std::memory_order_relaxed)) {
    }
}

} // namespace VideoCommon::V2::Landing
