// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>

#include "common/common_types.h"

// P5b (EDEN_NX2_V2_DYNRENDER): pending GPU writes that no barrier has made visible yet, keyed by
// host resource and guest address range (DXVK's DxvkBarrierTracker, with Cemu's barrier-epoch
// idea). Render targets alias guest memory that later copies, samples and other images (aliases)
// read, so a later access hazards with a pending write when it touches the same resource or an
// overlapping guest range. Pure logic, owned by the recording thread.
//
// Stage and access masks are opaque bit sets (Vulkan VkPipelineStageFlags / VkAccessFlags); a
// hazard returns the union of the overlapping writes' masks, which is the barrier's source scope.
namespace VideoCommon::V2 {

class RangeBarrierTracker {
public:
    static constexpr size_t Capacity = 64;

    struct Write {
        u64 resource = 0; ///< host resource identity (VkImage); 0 = none
        u64 begin = 0;    ///< guest range [begin, end); empty when unknown
        u64 end = 0;
        u32 stages = 0;
        u32 access = 0;
    };

    struct Hazard {
        u32 stages = 0;
        u32 access = 0;
        u32 count = 0; ///< pending writes it covers
        [[nodiscard]] bool Any() const noexcept {
            return count != 0;
        }
    };

    /// Half-open overlap of [a_begin, a_end) and [b_begin, b_end); empty ranges never overlap.
    [[nodiscard]] static constexpr bool Overlaps(u64 a_begin, u64 a_end, u64 b_begin,
                                                 u64 b_end) noexcept {
        return a_begin < a_end && b_begin < b_end && a_begin < b_end && b_begin < a_end;
    }

    [[nodiscard]] static constexpr bool Matches(const Write& write, u64 resource, u64 begin,
                                                u64 end) noexcept {
        return (resource != 0 && write.resource == resource) ||
               Overlaps(write.begin, write.end, begin, end);
    }

    /// Records a write. Writes to the same resource and range merge their masks. When full, every
    /// pending write folds into one "everything" entry that any access hazards with.
    void NoteWrite(u64 resource, u64 begin, u64 end, u32 stages, u32 access) noexcept {
        ++write_serial;
        if (everything) {
            all_stages |= stages;
            all_access |= access;
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            Write& write = writes[i];
            if (write.resource == resource && write.begin == begin && write.end == end) {
                write.stages |= stages;
                write.access |= access;
                return;
            }
        }
        if (count == Capacity) {
            FoldAll();
            all_stages |= stages;
            all_access |= access;
            return;
        }
        writes[count++] = {resource, begin, end, stages, access};
    }

    /// The pending writes an access to (resource, [begin, end)) hazards with, without removing them.
    [[nodiscard]] Hazard Check(u64 resource, u64 begin, u64 end) const noexcept {
        Hazard hazard{};
        if (everything) {
            return {all_stages, all_access, 1};
        }
        for (size_t i = 0; i < count; ++i) {
            const Write& write = writes[i];
            if (Matches(write, resource, begin, end)) {
                hazard.stages |= write.stages;
                hazard.access |= write.access;
                ++hazard.count;
            }
        }
        return hazard;
    }

    /// As Check, and removes those writes: the caller records the barrier that covers them.
    Hazard Take(u64 resource, u64 begin, u64 end) noexcept {
        if (everything) {
            return TakeAll();
        }
        Hazard hazard{};
        size_t kept = 0;
        for (size_t i = 0; i < count; ++i) {
            const Write& write = writes[i];
            if (Matches(write, resource, begin, end)) {
                hazard.stages |= write.stages;
                hazard.access |= write.access;
                ++hazard.count;
            } else {
                writes[kept++] = write;
            }
        }
        count = kept;
        return hazard;
    }

    /// Every pending write (a full barrier covers them all).
    Hazard TakeAll() noexcept {
        Hazard hazard{};
        if (everything) {
            hazard = {all_stages, all_access, 1};
        }
        for (size_t i = 0; i < count; ++i) {
            hazard.stages |= writes[i].stages;
            hazard.access |= writes[i].access;
            ++hazard.count;
        }
        Clear();
        return hazard;
    }

    void Clear() noexcept {
        count = 0;
        everything = false;
        all_stages = 0;
        all_access = 0;
    }

    [[nodiscard]] bool Empty() const noexcept {
        return count == 0 && !everything;
    }
    [[nodiscard]] size_t Size() const noexcept {
        return everything ? Capacity + 1 : count;
    }
    [[nodiscard]] bool Folded() const noexcept {
        return everything;
    }
    /// Moves with every NoteWrite: while it holds, a Take of what was taken before finds nothing.
    [[nodiscard]] u64 WriteSerial() const noexcept {
        return write_serial;
    }

private:
    void FoldAll() noexcept {
        for (size_t i = 0; i < count; ++i) {
            all_stages |= writes[i].stages;
            all_access |= writes[i].access;
        }
        count = 0;
        everything = true;
    }

    std::array<Write, Capacity> writes{};
    size_t count = 0;
    bool everything = false;
    u32 all_stages = 0;
    u32 all_access = 0;
    u64 write_serial = 0;
};

} // namespace VideoCommon::V2
