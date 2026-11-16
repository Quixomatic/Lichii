// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <optional>

#include "common/common_types.h"

// P5b (EDEN_NX2_V2_GPUDECODE): a fence-tracked ring allocator (Dolphin's VKStreamBuffer) that
// never waits. Each allocation remembers the tick that last uses it; space comes back once the
// GPU finished that tick. When the space isn't free yet, Allocate fails and the caller takes
// another path (it must not block under any lock). Pure logic; the buffer itself is the caller's.
namespace VideoCommon::V2 {

class TickRing {
public:
    explicit TickRing(u64 size_ = 0, u64 alignment_ = 256) : size{size_}, alignment{alignment_} {}

    void Reset(u64 size_, u64 alignment_) {
        size = size_;
        alignment = alignment_;
        head = 0;
        tail = 0;
        used = 0;
        fences.clear();
    }

    /// An offset of `bytes` free bytes used by `tick`, or nothing when the GPU still holds them.
    [[nodiscard]] std::optional<u64> Allocate(u64 bytes, u64 tick, u64 completed_tick) {
        Retire(completed_tick);
        bytes = AlignUp(bytes);
        if (bytes == 0 || bytes > size) {
            return std::nullopt;
        }
        u64 offset = head;
        if (fences.empty()) {
            head = 0;
            tail = 0;
            offset = 0;
        } else if (head == tail) {
            return std::nullopt; // Full: the newest allocation ends where the oldest begins
        } else if (head > tail) {
            // Free space: [head, size) and [0, tail).
            if (size - head < bytes) {
                if (tail < bytes) {
                    return std::nullopt;
                }
                // Wrap: the skipped end is held until this tick retires too.
                used += size - head;
                Mark(size, tick);
                offset = 0;
            }
        } else if (tail - head < bytes) {
            // head < tail: free space is [head, tail).
            return std::nullopt;
        }
        head = offset + bytes;
        used += bytes;
        Mark(head, tick);
        return offset;
    }

    /// Frees everything up to fences with tick <= completed_tick.
    void Retire(u64 completed_tick) {
        while (!fences.empty() && fences.front().tick <= completed_tick) {
            const Fence fence = fences.front();
            fences.pop_front();
            const u64 freed = fence.end >= tail ? fence.end - tail : size - tail + fence.end;
            used -= (std::min)(used, freed);
            tail = fence.end == size ? 0 : fence.end;
        }
        if (fences.empty()) {
            used = 0;
            head = 0;
            tail = 0;
        }
    }

    [[nodiscard]] u64 Used() const noexcept {
        return used;
    }
    [[nodiscard]] u64 Size() const noexcept {
        return size;
    }

private:
    struct Fence {
        u64 end;
        u64 tick;
    };

    [[nodiscard]] u64 AlignUp(u64 value) const noexcept {
        return (value + alignment - 1) / alignment * alignment;
    }

    void Mark(u64 end, u64 tick) {
        if (!fences.empty() && fences.back().tick == tick && fences.back().end != size) {
            fences.back().end = end;
            return;
        }
        fences.push_back({end, tick});
    }

    u64 size = 0;
    u64 alignment = 256;
    u64 head = 0; ///< next allocation
    u64 tail = 0; ///< oldest byte still in use
    u64 used = 0;
    std::deque<Fence> fences;
};

} // namespace VideoCommon::V2
