// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>

#include "shader_recompiler/shader_info.h"
#include "video_core/page_filter.h"
#include "video_core/v2/shader_writes.h"
#include "video_core/v2/v2_switches.h"

namespace VideoCommon::V2::ShaderWrites {

namespace {

constexpr size_t NumSlots = 4096;

// word 0: bit 0 unbounded, bits 8-15 storage count. Storage words: index | offset << 8 |
// count << 40.
struct Slot {
    std::atomic<u32> sequence{};
    std::atomic<u64> key{};
    std::atomic<u64> head{};
    std::array<std::atomic<u64>, Summary::MaxStorage> storage{};
};

std::array<Slot, NumSlots> g_slots;

// The pages and the address range of every key ever recorded (never shrink). Invalidations run
// for every inline upload and pre-draw invalidation; almost none reach a program's page, so they
// return here instead of walking every slot.
std::atomic<u64> g_lowest{~u64{0}};
std::atomic<u64> g_highest{0};

PageFilter& RecordedPages() {
    static PageFilter filter;
    return filter;
}

size_t SlotOf(u64 device_address) {
    u64 value = (device_address >> 4) * 0x9e3779b97f4a7c15ULL;
    value ^= value >> 31;
    return static_cast<size_t>(value) & (NumSlots - 1);
}

Summary Summarise(const Shader::Info& info) {
    Summary summary;
    summary.unbounded = info.stores_global_memory || info.uses_global_increment ||
                        info.uses_global_decrement;
    for (const auto& image : info.image_descriptors) {
        summary.unbounded |= image.is_written;
    }
    for (const auto& image : info.image_buffer_descriptors) {
        summary.unbounded |= image.is_written;
    }
    for (const auto& storage : info.storage_buffers_descriptors) {
        if (!storage.is_written) {
            continue;
        }
        if (storage.indirect_offset != ~0U || summary.num_storage == Summary::MaxStorage ||
            storage.cbuf_offset > 0xffffffu || storage.count > 0xffu) {
            summary.unbounded = true;
            continue;
        }
        summary.storage[summary.num_storage++] = {storage.cbuf_index, storage.cbuf_offset,
                                                  storage.count};
    }
    return summary;
}

} // Anonymous namespace

bool Enabled() {
    return On(Switch::Hazard) || On(Switch::Acquire);
}

void Record(u64 device_address, const Shader::Info& info) {
    if (device_address == 0) {
        return;
    }
    const Summary summary = Summarise(info);
    Slot& slot = g_slots[SlotOf(device_address)];
    if (slot.key.load(std::memory_order_relaxed) == device_address) {
        // Unchanged bindings are the common case: skip the stores.
        const u64 head = slot.head.load(std::memory_order_relaxed);
        if ((head & 1) == (summary.unbounded ? 1u : 0u) &&
            ((head >> 8) & 0xff) == summary.num_storage) {
            bool same = true;
            for (u32 i = 0; i < summary.num_storage && same; ++i) {
                const StorageWrite& s = summary.storage[i];
                same = slot.storage[i].load(std::memory_order_relaxed) ==
                       (u64{s.cbuf_index} | (u64{s.cbuf_offset} << 8) | (u64{s.count} << 40));
            }
            if (same) {
                return;
            }
        }
    }
    if (slot.key.load(std::memory_order_relaxed) != device_address) {
        // Published before the key, so an invalidation that sees the key also sees these.
        RecordedPages().Mark(device_address, 1);
        if (device_address < g_lowest.load(std::memory_order_relaxed)) {
            g_lowest.store(device_address, std::memory_order_relaxed);
        }
        if (device_address > g_highest.load(std::memory_order_relaxed)) {
            g_highest.store(device_address, std::memory_order_relaxed);
        }
    }
    const u32 sequence = slot.sequence.load(std::memory_order_relaxed);
    slot.sequence.store(sequence + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    slot.key.store(device_address, std::memory_order_relaxed);
    slot.head.store((summary.unbounded ? 1u : 0u) | (u64{summary.num_storage} << 8),
                    std::memory_order_relaxed);
    for (u32 i = 0; i < summary.num_storage; ++i) {
        const StorageWrite& s = summary.storage[i];
        slot.storage[i].store(u64{s.cbuf_index} | (u64{s.cbuf_offset} << 8) | (u64{s.count} << 40),
                              std::memory_order_relaxed);
    }
    slot.sequence.store(sequence + 2, std::memory_order_release);
}

std::optional<Summary> Lookup(u64 device_address) {
    const Slot& slot = g_slots[SlotOf(device_address)];
    const u32 before = slot.sequence.load(std::memory_order_acquire);
    if ((before & 1) != 0 || slot.key.load(std::memory_order_relaxed) != device_address) {
        return std::nullopt;
    }
    Summary summary;
    const u64 head = slot.head.load(std::memory_order_relaxed);
    summary.unbounded = (head & 1) != 0;
    summary.num_storage = std::min<u32>(static_cast<u32>((head >> 8) & 0xff), Summary::MaxStorage);
    for (u32 i = 0; i < summary.num_storage; ++i) {
        const u64 word = slot.storage[i].load(std::memory_order_relaxed);
        summary.storage[i] = {static_cast<u32>(word & 0xff),
                              static_cast<u32>((word >> 8) & 0xffffffff),
                              static_cast<u32>(word >> 40)};
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (slot.sequence.load(std::memory_order_relaxed) != before ||
        slot.key.load(std::memory_order_relaxed) != device_address) {
        return std::nullopt;
    }
    return summary;
}

void Invalidate(u64 begin, u64 end) {
    if (begin >= end || end <= g_lowest.load(std::memory_order_relaxed) ||
        begin > g_highest.load(std::memory_order_relaxed) ||
        !RecordedPages().Any(begin, end - begin)) {
        return;
    }
    for (Slot& slot : g_slots) {
        const u64 key = slot.key.load(std::memory_order_relaxed);
        if (key >= begin && key < end) {
            slot.key.store(0, std::memory_order_release);
        }
    }
}

} // namespace VideoCommon::V2::ShaderWrites
