// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <deque>
#include <optional>

#include "common/common_types.h"

// Pipeline v2 F2-02 (b): the decode-time constant buffer version ring (MASTER_PLAN §2.1). S1
// (the front) allocates a version's bytes at decode; the back references them from the draws it
// records and gives the space back once the GPU is done with it. Positions are monotonic byte
// counts (the ring offset is position % capacity), so the two threads share only two counters:
//  - head (front writes, back reads): the end of the newest allocation;
//  - retired (back writes, front reads): every position below it is free again.
// Retirement is by timeline value, not scheduler tick: the back notes, per submit, the newest
// position its command buffer references (NoteSubmit) and retires up to it once that submit's
// timeline value completed (Collect). Versions survive submits until then.
namespace VideoCommon::V2::CbufRing {

class Ring {
public:
    explicit Ring(u64 capacity_) : capacity{capacity_} {}

    [[nodiscard]] u64 Capacity() const noexcept {
        return capacity;
    }

    /// Front: room for `size` bytes aligned to `align` (a power of two) that do not wrap; the
    /// allocation's position, or nothing when the unretired span would exceed the capacity (the
    /// caller falls back to the back's versions). A tail too short to hold it is skipped.
    [[nodiscard]] std::optional<u64> Allocate(u64 size, u64 align) {
        const u64 current = head.load(std::memory_order_relaxed);
        u64 begin = (current + align - 1) & ~(align - 1);
        const u64 offset = begin % capacity;
        if (offset + size > capacity) {
            begin += capacity - offset; // the next lap's start
        }
        const u64 end = begin + size;
        if (size > capacity || end - retired.load(std::memory_order_acquire) > capacity) {
            ++refusals;
            return std::nullopt;
        }
        head.store(end, std::memory_order_release);
        return begin;
    }

    /// Ring offset of a position.
    [[nodiscard]] u64 Offset(u64 position) const noexcept {
        return position % capacity;
    }

    /// Back: the submit with timeline value `timeline` references positions below `high` (the
    /// head read when its command buffer was closed).
    void NoteSubmit(u64 timeline, u64 high) {
        const u64 noted = pending.empty() ? retired.load(std::memory_order_relaxed)
                                          : pending.back().high;
        if (high <= noted) {
            return; // nothing new since the last noted submit
        }
        pending.push_back({timeline, high});
    }

    /// Back: the submits up to timeline value `completed` finished on the GPU.
    void Collect(u64 completed) {
        u64 free_below = retired.load(std::memory_order_relaxed);
        while (!pending.empty() && pending.front().timeline <= completed) {
            free_below = pending.front().high;
            pending.pop_front();
        }
        retired.store(free_below, std::memory_order_release);
    }

    [[nodiscard]] u64 Head() const noexcept {
        return head.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 Retired() const noexcept {
        return retired.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 Refusals() const noexcept {
        return refusals;
    }

private:
    struct Submit {
        u64 timeline;
        u64 high;
    };
    u64 capacity;
    std::atomic<u64> head{};
    std::atomic<u64> retired{};
    u64 refusals{}; ///< front only
    std::deque<Submit> pending; ///< back only
};

} // namespace VideoCommon::V2::CbufRing
