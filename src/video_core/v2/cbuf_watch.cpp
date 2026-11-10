// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdlib>
#include <string_view>

#include "video_core/v2/cbuf_watch.h"

namespace VideoCommon::V2::CbufWatch {

PageWords::~PageWords() {
    for (auto& chunk : chunks) {
        delete[] chunk.load(std::memory_order_acquire);
    }
}

std::atomic<u32>* PageWords::Slot(u64 page, bool allocate) {
    const u64 chunk = page >> ChunkBits;
    if (chunk >= MaxChunks) {
        return nullptr;
    }
    std::atomic<u32>* array = chunks[chunk].load(std::memory_order_acquire);
    if (array == nullptr) {
        if (!allocate) {
            return nullptr;
        }
        auto* const fresh = new std::atomic<u32>[ChunkSize]{};
        if (chunks[chunk].compare_exchange_strong(array, fresh, std::memory_order_acq_rel)) {
            array = fresh;
        } else {
            delete[] fresh; // another thread published it first
        }
    }
    return &array[page & (ChunkSize - 1)];
}

void PageWords::Clear() {
    for (auto& chunk : chunks) {
        if (std::atomic<u32>* const array = chunk.load(std::memory_order_acquire)) {
            for (u64 i = 0; i < ChunkSize; ++i) {
                array[i].store(0, std::memory_order_relaxed);
            }
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
}

void Watch::NoteForeign(u64 addr, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 last = (addr + size - 1) >> PageBits;
    bool bumped = false;
    for (u64 page = addr >> PageBits; page <= last; ++page) {
        std::atomic<u32>* const slot = words.Slot(page);
        if (slot == nullptr) {
            page |= PageWords::ChunkSize - 1;
            continue;
        }
        u32 word = slot->load(std::memory_order_relaxed);
        if (word == 0) {
            continue; // never requested nor used: no version snapshots it
        }
        if ((word & ArmedBit) == 0) {
            slot->fetch_add(SeqOne, std::memory_order_acq_rel);
            bumped = true;
            continue;
        }
        while ((word & ArmedBit) != 0) {
            if (slot->compare_exchange_weak(word, ((word & ~ArmedBit) | OwedBit) + SeqOne,
                                            std::memory_order_acq_rel)) {
                stats.foreign_disarms.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        if ((word & ArmedBit) == 0) {
            // A trap won the race: the page is unarmed now, move it as an unarmed one.
            slot->fetch_add(SeqOne, std::memory_order_acq_rel);
            bumped = true;
        }
    }
    if (bumped) {
        stats.foreign_bumps.fetch_add(1, std::memory_order_relaxed);
    }
}

void Watch::RequestArm(u64 addr, u32 size) {
    if (size == 0) {
        return;
    }
    const u64 last = (addr + size - 1) >> PageBits;
    for (u64 page = addr >> PageBits; page <= last; ++page) {
        std::atomic<u32>* const slot = words.Slot(page, true);
        if (slot == nullptr) {
            return;
        }
        const u32 word = slot->load(std::memory_order_relaxed);
        if ((word & (ArmedBit | QueuedBit | OwedBit)) != 0) {
            continue;
        }
        if ((slot->fetch_or(QueuedBit, std::memory_order_acq_rel) & QueuedBit) != 0) {
            continue; // another request won
        }
        stats.requests.fetch_add(1, std::memory_order_relaxed);
        std::scoped_lock lock{queue_mutex};
        queue.push_back(static_cast<u32>(page));
    }
}

void Watch::MarkUsed(u64 addr, u32 size) {
    if (size == 0) {
        return;
    }
    const u64 last = (addr + size - 1) >> PageBits;
    for (u64 page = addr >> PageBits; page <= last; ++page) {
        std::atomic<u32>* const slot = words.Slot(page, true);
        if (slot == nullptr) {
            return;
        }
        u32 expected = 0;
        slot->compare_exchange_strong(expected, SeqOne, std::memory_order_acq_rel);
    }
}

Watch& Get() {
    static Watch watch;
    return watch;
}

bool Enabled() {
    static const bool enabled = [] {
        const char* const value = std::getenv("EDEN_NX2_CBUFWATCH");
        return value == nullptr || value[0] != '0';
    }();
    return enabled && On(Switch::CbufRing);
}

bool OnlyHeldPages() {
    static const bool held = [] {
        const char* const value = std::getenv("EDEN_NX2_CBUFWATCH");
        return value != nullptr && std::string_view{value} == "held";
    }();
    return held;
}

} // namespace VideoCommon::V2::CbufWatch
