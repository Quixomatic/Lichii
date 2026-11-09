// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>

#include "common/common_types.h"

// Pipeline v2 P5a: a lock-free address -> resource page table (vkd3d-proton's va_map.c idea).
// One writer (the commit owner) stores page ranges; any number of readers load with acquire
// semantics and never block. Leaves are allocated on first store and published with a release
// store of their pointer; they are only freed with the map, so a reader never sees freed memory.
// A value is a resource slot (0 = none). Readers that dereference the resource validate it with
// its generation (V2::Versions::Gens), since the writer may unmap and reuse a slot at any time.
namespace VideoCommon::V2 {

template <u32 PageBits, u32 AddressBits, u32 LeafBits = 12>
class VaMap {
    static_assert(AddressBits > PageBits + LeafBits);

public:
    static constexpr u64 NumPages = u64{1} << (AddressBits - PageBits);
    static constexpr u64 LeafPages = u64{1} << LeafBits;
    static constexpr u64 NumLeaves = NumPages >> LeafBits;
    static constexpr u64 PageSize = u64{1} << PageBits;

    VaMap() : root{std::make_unique<std::atomic<Leaf*>[]>(NumLeaves)} {}

    ~VaMap() {
        for (u64 index = 0; index < NumLeaves; ++index) {
            delete root[index].load(std::memory_order_relaxed);
        }
    }

    VaMap(const VaMap&) = delete;
    VaMap& operator=(const VaMap&) = delete;

    /// Value of the page holding `address` (0 when nothing is mapped). Any thread.
    [[nodiscard]] u32 Load(u64 address) const noexcept {
        const u64 page = address >> PageBits;
        if (page >= NumPages) {
            return 0;
        }
        const Leaf* const leaf = root[page >> LeafBits].load(std::memory_order_acquire);
        if (leaf == nullptr) {
            return 0;
        }
        return leaf->values[page & (LeafPages - 1)].load(std::memory_order_acquire);
    }

    /// The value every page of [address, address + size) holds, or 0 when they differ or one is
    /// unmapped. Any thread. Long ranges stop at the first mismatch.
    [[nodiscard]] u32 Uniform(u64 address, u64 size) const noexcept {
        if (size == 0) {
            return 0;
        }
        const u32 first = Load(address);
        if (first == 0) {
            return 0;
        }
        const u64 last_page = (address + size - 1) >> PageBits;
        for (u64 page = (address >> PageBits) + 1; page <= last_page; ++page) {
            if (Load(page << PageBits) != first) {
                return 0;
            }
        }
        return first;
    }

    /// Stores `value` into every page of [begin, end) (page aligned or not). Writer only.
    void Store(u64 begin, u64 end, u32 value) {
        if (end <= begin) {
            return;
        }
        const u64 first_page = begin >> PageBits;
        const u64 last_page = (std::min)((end - 1) >> PageBits, NumPages - 1);
        for (u64 page = first_page; page <= last_page; ++page) {
            std::atomic<Leaf*>& slot = root[page >> LeafBits];
            Leaf* leaf = slot.load(std::memory_order_relaxed);
            if (leaf == nullptr) {
                if (value == 0) {
                    // Clearing an unallocated leaf: skip to the next leaf.
                    page = ((page >> LeafBits) << LeafBits) + LeafPages - 1;
                    continue;
                }
                leaf = new Leaf{};
                slot.store(leaf, std::memory_order_release);
                ++allocated_leaves;
            }
            leaf->values[page & (LeafPages - 1)].store(value, std::memory_order_release);
        }
    }

    /// Leaves allocated so far (stats, tests).
    [[nodiscard]] u64 AllocatedLeaves() const noexcept {
        return allocated_leaves;
    }

private:
    struct Leaf {
        std::array<std::atomic<u32>, LeafPages> values{};
    };

    std::unique_ptr<std::atomic<Leaf*>[]> root;
    u64 allocated_leaves{};
};

} // namespace VideoCommon::V2
