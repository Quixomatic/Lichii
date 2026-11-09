// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "video_core/v2/versions.h"

namespace VideoCommon::V2::Versions {

PageVersions::PageVersions()
    : pages{std::make_unique<std::atomic<u64>[]>(PageSlots)},
      regions{std::make_unique<std::atomic<u64>[]>(RegionSlots)} {
    Reset();
}

void PageVersions::Note(DAddr address, u64 size, Source source) {
    if (size == 0) {
        return;
    }
    counts[static_cast<size_t>(source)].fetch_add(1, std::memory_order_relaxed);
    const u64 value = clock.load(std::memory_order_acquire);
    const u64 last_byte = size > ~address ? ~u64{0} : address + size - 1;
    const u64 first_page = address >> PageBits;
    const u64 last_page = last_byte >> PageBits;
    if (last_page - first_page < MaxPageWrite) {
        for (u64 page = first_page; page <= last_page; ++page) {
            Raise(pages[PageSlot(page)], value);
        }
        return;
    }
    const u64 first_region = address >> RegionBits;
    const u64 last_region = last_byte >> RegionBits;
    if (last_region - first_region < MaxRegionWrite) {
        for (u64 region = first_region; region <= last_region; ++region) {
            Raise(regions[RegionSlot(region)], value);
        }
        return;
    }
    Raise(floor, value);
}

u64 PageVersions::Newest(DAddr address, u64 size) const {
    u64 newest = floor.load(std::memory_order_acquire);
    if (size == 0) {
        return newest;
    }
    const u64 last_byte = size > ~address ? ~u64{0} : address + size - 1;
    const u64 first_region = address >> RegionBits;
    const u64 last_region = last_byte >> RegionBits;
    if (last_region - first_region >= MaxRegionWrite) {
        // Ranges this long are not tracked page by page: treat them as just written.
        return ~u64{0};
    }
    for (u64 region = first_region; region <= last_region; ++region) {
        newest = (std::max)(newest, regions[RegionSlot(region)].load(std::memory_order_acquire));
    }
    const u64 last_page = last_byte >> PageBits;
    for (u64 page = address >> PageBits; page <= last_page; ++page) {
        newest = (std::max)(newest, pages[PageSlot(page)].load(std::memory_order_acquire));
    }
    return newest;
}

bool PageVersions::Unchanged(DAddr address, u64 size, u64 since) const {
    if (floor.load(std::memory_order_acquire) > since) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    const u64 last_byte = size > ~address ? ~u64{0} : address + size - 1;
    const u64 first_region = address >> RegionBits;
    const u64 last_region = last_byte >> RegionBits;
    if (last_region - first_region >= MaxRegionWrite) {
        return false;
    }
    for (u64 region = first_region; region <= last_region; ++region) {
        if (regions[RegionSlot(region)].load(std::memory_order_acquire) > since) {
            return false;
        }
    }
    const u64 last_page = last_byte >> PageBits;
    for (u64 page = address >> PageBits; page <= last_page; ++page) {
        if (pages[PageSlot(page)].load(std::memory_order_acquire) > since) {
            return false;
        }
    }
    return true;
}

std::array<u64, NumSources> PageVersions::TakeCounts() {
    std::array<u64, NumSources> result{};
    for (size_t i = 0; i < NumSources; ++i) {
        result[i] = counts[i].exchange(0, std::memory_order_relaxed);
    }
    return result;
}

void PageVersions::Reset() {
    for (size_t i = 0; i < PageSlots; ++i) {
        pages[i].store(0, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < RegionSlots; ++i) {
        regions[i].store(0, std::memory_order_relaxed);
    }
    floor.store(0, std::memory_order_relaxed);
    clock.store(1, std::memory_order_release);
    for (auto& count : counts) {
        count.store(0, std::memory_order_relaxed);
    }
}

void StreamLog::Note(DAddr begin, DAddr end) {
    const u64 position = head.fetch_add(1, std::memory_order_acq_rel);
    Slot& slot = slots[position % Size];
    // Mark the slot as being written first: a reader that sees this sequence treats it as an
    // overlap.
    slot.sequence.store(0, std::memory_order_release);
    slot.begin.store(begin, std::memory_order_relaxed);
    slot.end.store(end, std::memory_order_relaxed);
    slot.sequence.store(position + 1, std::memory_order_release);
}

bool StreamLog::WrittenSince(u64 since, DAddr begin, DAddr end) const {
    const u64 now = head.load(std::memory_order_acquire);
    if (now - since > Size) {
        return true;
    }
    for (u64 position = since; position < now; ++position) {
        const Slot& slot = slots[position % Size];
        if (slot.sequence.load(std::memory_order_acquire) != position + 1) {
            return true;
        }
        const DAddr write_begin = slot.begin.load(std::memory_order_relaxed);
        const DAddr write_end = slot.end.load(std::memory_order_relaxed);
        if (slot.sequence.load(std::memory_order_acquire) != position + 1) {
            return true;
        }
        if (write_begin < end && begin < write_end) {
            return true;
        }
    }
    return false;
}

void StreamLog::Reset() {
    head.store(0, std::memory_order_relaxed);
    for (Slot& slot : slots) {
        slot.sequence.store(0, std::memory_order_relaxed);
    }
}

std::array<u64, NumEpochReasons> CollectEpoch::TakeCounts() {
    std::array<u64, NumEpochReasons> result{};
    for (size_t i = 0; i < NumEpochReasons; ++i) {
        result[i] = counts[i].exchange(0, std::memory_order_relaxed);
    }
    return result;
}

bool EpochMemo::Unchanged(DAddr address, u64 size, u64 since) {
    ++counters.queries;
    const DAddr end = address + size;
    u64 key = (address >> PageVersions::PageBits) ^ (size << 37);
    key *= 0x9e3779b97f4a7c15ULL;
    Entry& entry = entries[static_cast<size_t>(key >> 56) % Entries];
    const u64 current_epoch = epoch->Current();
    if (entry.epoch == current_epoch && entry.begin == address && entry.end == end) {
        const u64 head = log->Head();
        if (entry.log_head == head || !log->WrittenSince(entry.log_head, address, end)) {
            entry.log_head = head;
            ++counters.hits;
            return entry.newest <= since;
        }
    }
    ++counters.walks;
    // The log position before the walk: a write logged during it is seen by the next query.
    const u64 head = log->Head();
    const u64 newest = versions->Newest(address, size);
    entry = Entry{
        .begin = address,
        .end = end,
        .epoch = current_epoch,
        .log_head = head,
        .newest = newest,
    };
    return newest <= since;
}

void EpochMemo::Clear() {
    entries.fill(Entry{});
}

void PageWatch::SetTracker(Tracker tracker_) {
    // A new owner (its memory is new): forget the pages without the old tracker.
    refs.clear();
    idle.clear();
    tracker = std::move(tracker_);
}

void PageWatch::Hold(DAddr address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 first = address >> PageVersions::PageBits;
    const u64 last = (address + size - 1) >> PageVersions::PageBits;
    for (u64 page = first; page <= last; ++page) {
        const auto [it, inserted] = refs.try_emplace(page, 0);
        if (inserted && tracker) {
            tracker(page << PageVersions::PageBits, u64{1} << PageVersions::PageBits, 1);
        }
        if (it->second++ == 0) {
            idle.erase(page);
        }
    }
}

void PageWatch::Drop(DAddr address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 first = address >> PageVersions::PageBits;
    const u64 last = (address + size - 1) >> PageVersions::PageBits;
    for (u64 page = first; page <= last; ++page) {
        const auto it = refs.find(page);
        if (it != refs.end() && it->second != 0 && --it->second == 0) {
            idle.insert(page);
        }
    }
    if (idle.size() > MaxIdle) {
        ReleaseIdle();
    }
}

void PageWatch::Clear() {
    for (const auto& [page, count] : refs) {
        if (tracker) {
            tracker(page << PageVersions::PageBits, u64{1} << PageVersions::PageBits, -1);
        }
    }
    refs.clear();
    idle.clear();
}

void PageWatch::ReleaseIdle() {
    for (const u64 page : idle) {
        refs.erase(page);
        if (tracker) {
            tracker(page << PageVersions::PageBits, u64{1} << PageVersions::PageBits, -1);
        }
    }
    idle.clear();
}

void Generations::Bump(Kind kind, u32 id) {
    id &= 0x7fffffffU;
    const size_t chunk = id >> ChunkBits;
    if (chunk >= MaxChunks) {
        return;
    }
    auto& slot = chunks[static_cast<size_t>(kind)][chunk];
    std::atomic<u32>* array = slot.load(std::memory_order_acquire);
    if (array == nullptr) {
        auto* const fresh = new std::atomic<u32>[ChunkSize];
        for (size_t i = 0; i < ChunkSize; ++i) {
            fresh[i].store(0, std::memory_order_relaxed);
        }
        if (slot.compare_exchange_strong(array, fresh, std::memory_order_acq_rel)) {
            array = fresh;
        } else {
            delete[] fresh;
        }
    }
    array[id & (ChunkSize - 1)].fetch_add(1, std::memory_order_acq_rel);
    bumps.fetch_add(1, std::memory_order_relaxed);
}

Generations::~Generations() {
    for (auto& kind : chunks) {
        for (auto& chunk : kind) {
            delete[] chunk.load(std::memory_order_relaxed);
        }
    }
}

PageVersions& Pages() {
    static PageVersions versions;
    return versions;
}

StreamLog& Stream() {
    static StreamLog log;
    return log;
}

CollectEpoch& Epochs() {
    static CollectEpoch epoch;
    return epoch;
}

Generations& Gens() {
    static Generations generations;
    return generations;
}

} // namespace VideoCommon::V2::Versions
