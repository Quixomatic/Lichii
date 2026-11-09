// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "video_core/v2/versions.h"

namespace VideoCommon::V2::Versions {

PageVersions::PageVersions()
    : pages{std::make_unique<std::atomic<u64>[]>(PageSlots)},
      regions{std::make_unique<std::atomic<u64>[]>(RegionSlots)},
      fine{std::make_unique<std::atomic<u64>[]>(FineSlots)},
      coarse_all{std::make_unique<std::atomic<u64>[]>(CoarseSlots)},
      coarse_long{std::make_unique<std::atomic<u64>[]>(CoarseSlots)} {
    Reset();
}

void PageVersions::LogWrite(u64 first_byte, u64 last_byte) {
    // After the version raises (seq_cst): a walk that missed them read the sequence before this.
    const u64 value = sequence.fetch_add(1, std::memory_order_seq_cst) + 1;
    if ((last_byte >> RegionBits) - (first_byte >> RegionBits) >= MaxRegionWrite) {
        Raise(all_written, value);
        return;
    }
    const u64 first_coarse = first_byte >> CoarseBits;
    const u64 last_coarse = last_byte >> CoarseBits;
    const u64 first_fine = first_byte >> FineBits;
    const u64 last_fine = last_byte >> FineBits;
    if (last_fine - first_fine < MaxFineBuckets) {
        for (u64 bucket = first_fine; bucket <= last_fine; ++bucket) {
            Raise(fine[FineSlot(bucket)], value);
        }
        for (u64 bucket = first_coarse; bucket <= last_coarse; ++bucket) {
            Raise(coarse_all[CoarseSlot(bucket)], value);
        }
        return;
    }
    for (u64 bucket = first_coarse; bucket <= last_coarse; ++bucket) {
        Raise(coarse_all[CoarseSlot(bucket)], value);
        Raise(coarse_long[CoarseSlot(bucket)], value);
    }
}

bool PageVersions::WrittenSince(u64 since, DAddr begin, DAddr end) const {
    if (all_written.load(std::memory_order_seq_cst) > since) {
        return true;
    }
    if (end <= begin) {
        return false;
    }
    const u64 last_byte = end - 1;
    if ((last_byte >> RegionBits) - (begin >> RegionBits) >= MaxRegionWrite) {
        return true;
    }
    const u64 first_coarse = begin >> CoarseBits;
    const u64 last_coarse = last_byte >> CoarseBits;
    const u64 first_fine = begin >> FineBits;
    const u64 last_fine = last_byte >> FineBits;
    if (last_fine - first_fine < MaxFineBuckets) {
        for (u64 bucket = first_fine; bucket <= last_fine; ++bucket) {
            if (fine[FineSlot(bucket)].load(std::memory_order_seq_cst) > since) {
                return true;
            }
        }
        for (u64 bucket = first_coarse; bucket <= last_coarse; ++bucket) {
            if (coarse_long[CoarseSlot(bucket)].load(std::memory_order_seq_cst) > since) {
                return true;
            }
        }
        return false;
    }
    for (u64 bucket = first_coarse; bucket <= last_coarse; ++bucket) {
        if (coarse_all[CoarseSlot(bucket)].load(std::memory_order_seq_cst) > since) {
            return true;
        }
    }
    return false;
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
    const u64 first_region = address >> RegionBits;
    const u64 last_region = last_byte >> RegionBits;
    if (last_page - first_page < MaxPageWrite) {
        for (u64 page = first_page; page <= last_page; ++page) {
            Raise(pages[PageSlot(page)], value);
        }
        LogWrite(address, last_byte);
    } else if (last_region - first_region < MaxRegionWrite) {
        for (u64 region = first_region; region <= last_region; ++region) {
            Raise(regions[RegionSlot(region)], value);
        }
        // The region slots move for every range in those regions.
        LogWrite(first_region << RegionBits, ((last_region + 1) << RegionBits) - 1);
    } else {
        Raise(floor, value);
        LogWrite(address, last_byte);
    }
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
    for (size_t i = 0; i < FineSlots; ++i) {
        fine[i].store(0, std::memory_order_relaxed);
    }
    for (size_t i = 0; i < CoarseSlots; ++i) {
        coarse_all[i].store(0, std::memory_order_relaxed);
        coarse_long[i].store(0, std::memory_order_relaxed);
    }
    all_written.store(0, std::memory_order_relaxed);
    clock.store(1, std::memory_order_release);
    for (auto& count : counts) {
        count.store(0, std::memory_order_relaxed);
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
    // No write reached the range's buckets since the walk: its newest version still holds. The
    // entry keeps its walk's sequence (a write numbered before a later query may still be
    // raising its buckets).
    if (entry.valid && entry.begin == address && entry.end == end &&
        !versions->WrittenSince(entry.sequence, address, end)) {
        ++counters.hits;
        return entry.newest <= since;
    }
    ++counters.walks;
    // The sequence before the walk (seq_cst, as LogWrite): a write the walk missed is numbered
    // after it, so the next query sees it in its buckets.
    const u64 sequence = versions->Sequence();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    const u64 newest = versions->Newest(address, size);
    entry = Entry{
        .begin = address,
        .end = end,
        .sequence = sequence,
        .newest = newest,
        .valid = true,
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

CollectEpoch& Epochs() {
    static CollectEpoch epoch;
    return epoch;
}

Generations& Gens() {
    static Generations generations;
    return generations;
}

} // namespace VideoCommon::V2::Versions
