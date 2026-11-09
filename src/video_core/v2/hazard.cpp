// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstring>

#include "video_core/v2/hazard.h"

namespace VideoCommon::V2 {

namespace {

constexpr u64 PageBits = 12;
constexpr size_t NumProducers = static_cast<size_t>(Producer::Count);

} // Anonymous namespace

HazardTable::HazardTable() = default;
HazardTable::~HazardTable() = default;

void HazardTable::Clear() {
    head = 0;
    count = 0;
    if (report_pages) {
        report_pages->fill(0);
    }
    if (slots) {
        slots->fill(Link{});
    }
    wide.fill(0);
    wide_entries.fill(Link{});
    wide_head = 0;
    wide_count = 0;
    target_sets.fill(TargetSet{});
    target_current = 0;
    targets_overflow = 0;
    unbounded.fill(0);
    unbounded_synced.fill(0);
    overflow_position = 0;
    overflows = 0;
}

size_t HazardTable::Slot(u64 page) {
    u64 value = page * 0x9e3779b97f4a7c15ULL;
    value ^= value >> 29;
    return static_cast<size_t>(value) & (PageSlots - 1);
}

bool HazardTable::Live(const Link& link, u64 consumed) const {
    // Positions are unique: a reused index holds a newer entry. Retired and folded entries are
    // older than everything still chained before them.
    return link.position > consumed && entries[link.index].position == link.position;
}

HazardTable::Entry& HazardTable::Push(u64 address, u64 end, u64 position, Producer producer) {
    if (!entries) [[unlikely]] {
        entries = std::make_unique<Entry[]>(Capacity);
        slots = std::make_unique<std::array<Link, PageSlots>>();
        slots->fill(Link{});
        report_pages = std::make_unique<std::array<u64, ReportPageBits / 64>>();
        report_pages->fill(0);
    }
    if (count == Capacity) {
        // The oldest entry may still be queued: whatever it wrote now waits for its position.
        const Entry& oldest = entries[head];
        overflow_position = std::max(overflow_position, oldest.position);
        ++overflows;
        head = (head + 1) & (Capacity - 1);
        --count;
    }
    const u32 index = static_cast<u32>((head + count) & (Capacity - 1));
    Entry& entry = entries[index];
    ++count;
    entry.begin = address;
    entry.end = end;
    entry.position = position;
    entry.producer = producer;
    entry.known = 0;
    Insert(entry, index);
    return entry;
}

size_t HazardTable::ReportBit(u64 page) {
    return static_cast<size_t>((page * 0x9e3779b97f4a7c15ULL) >> 40) & (ReportPageBits - 1);
}

bool HazardTable::MayHoldReports(u64 address, u64 size) const {
    if (!report_pages || size == 0) {
        return false;
    }
    const u64 last = (address + size - 1) >> PageBits;
    for (u64 page = address >> PageBits; page <= last; ++page) {
        const size_t bit = ReportBit(page);
        if (((*report_pages)[bit / 64] >> (bit % 64)) & 1) {
            return true;
        }
    }
    return false;
}

void HazardTable::Insert(Entry& entry, u32 index) {
    const u64 first = entry.begin >> PageBits;
    const u64 last = (entry.end - 1) >> PageBits;
    if (first != last) {
        entry.chained = false;
        entry.previous = {};
        if (wide_count == MaxWide) {
            // The oldest wide write leaves the list: lookups for its producer scan everything
            // while it may still be queued.
            const Link& oldest = wide_entries[wide_head];
            if (entries[oldest.index].position == oldest.position) {
                const size_t p = static_cast<size_t>(entries[oldest.index].producer);
                wide[p] = std::max(wide[p], oldest.position);
            }
            wide_head = (wide_head + 1) % MaxWide;
            --wide_count;
        }
        wide_entries[(wide_head + wide_count) % MaxWide] = {index, entry.position};
        ++wide_count;
        return;
    }
    Link& slot = (*slots)[Slot(first)];
    entry.chained = true;
    entry.previous = slot;
    slot = {index, entry.position};
}

void HazardTable::NoteKnown(u64 address, const void* bytes, u32 size, u64 position,
                            Producer producer) {
    if (size == 0) {
        return;
    }
    if (size > MaxKnownBytes) {
        NoteRange(address, size, position, producer);
        return;
    }
    Entry& entry = Push(address, address + size, position, producer);
    entry.known = static_cast<u8>(size);
    std::memcpy(entry.bytes.data(), bytes, size);
}

void HazardTable::NoteRange(u64 address, u64 size, u64 position, Producer producer) {
    if (size == 0) {
        return;
    }
    const u64 end = address + size;
    if (producer == Producer::CbData && count != 0) {
        // Constant buffer words arrive one by one: grow the newest write while it stays on its
        // page (it is the newest on its slot's chain, whose head takes the new position).
        const u32 index = static_cast<u32>((head + count - 1) & (Capacity - 1));
        Entry& last = entries[index];
        if (last.producer == Producer::CbData && last.known == 0 && last.chained &&
            address >= last.begin && address <= last.end &&
            ((end - 1) >> PageBits) == (last.begin >> PageBits)) {
            Link& slot = (*slots)[Slot(last.begin >> PageBits)];
            if (slot.index == index && slot.position == last.position) {
                last.end = std::max(last.end, end);
                last.position = position;
                slot.position = position;
                return;
            }
        }
    }
    Push(address, end, position, producer);
    if (producer == Producer::Release || producer == Producer::Report) {
        // A counter report or reduction: its value reaches memory at a point the front can't see.
        for (u64 page = address >> PageBits; page <= (end - 1) >> PageBits; ++page) {
            const size_t bit = ReportBit(page);
            (*report_pages)[bit / 64] |= u64{1} << (bit % 64);
        }
    }
}

void HazardTable::NoteUnbounded(Producer producer, u64 position) {
    unbounded[static_cast<size_t>(producer)] = position;
}

void HazardTable::NoteSync(u64 position) {
    (void)position;
    unbounded_synced = unbounded;
    for (TargetSet& set : target_sets) {
        set.synced = set.position;
    }
}

void HazardTable::NoteDrawTargets(std::span<const std::pair<u64, u64>> ranges, u64 position) {
    const size_t num = std::min(ranges.size(), MaxTargets);
    TargetSet& current = target_sets[target_current];
    if (current.count == num &&
        std::equal(ranges.begin(), ranges.begin() + num, current.ranges.begin())) {
        current.position = position;
        return;
    }
    target_current = (target_current + 1) % TargetSets;
    TargetSet& next = target_sets[target_current];
    targets_overflow = std::max(targets_overflow, next.synced);
    next.count = static_cast<u32>(num);
    std::copy(ranges.begin(), ranges.begin() + num, next.ranges.begin());
    next.position = position;
    next.synced = 0;
}

void HazardTable::Retire(u64 consumed) {
    while (count != 0 && entries[head].position <= consumed) {
        head = (head + 1) & (Capacity - 1);
        --count;
    }
}

void HazardTable::Linear(u64 address, u64 end, u64 consumed, u32 ranged, HazardResult& result,
                         u64& wait, std::array<u32, 64>& found, size_t& num_found,
                         u64& covered) const {
    const u64 size = end - address;
    const u64 all_bits = size >= 64 ? ~u64{0} : (u64{1} << size) - 1;
    // Every pending entry, newest first: long reads and wide writes.
    size_t visited = 0;
    for (size_t i = count; i-- > 0;) {
        const u32 index = static_cast<u32>((head + i) & (Capacity - 1));
        const Entry& entry = entries[index];
        if (entry.position <= consumed) {
            return; // Older entries ran too
        }
        if (++visited > MaxScan) {
            wait = std::max(wait, entries[(head + count - 1) & (Capacity - 1)].position);
            result.overflow = true;
            return;
        }
        const u32 bit = ProducerBit(entry.producer);
        if ((ranged & bit) == 0 || entry.end <= address || entry.begin >= end) {
            continue;
        }
        result.producers |= bit;
        if (entry.known == 0) {
            wait = std::max(wait, entry.position);
            return;
        }
        if (num_found == found.size()) {
            wait = std::max(wait, entry.position);
            result.overflow = true;
            return;
        }
        found[num_found++] = index;
        if (size <= 64) {
            // Newer known stores cover the read: older entries don't matter.
            const u64 begin = std::max(entry.begin, address) - address;
            const u64 length = std::min(entry.end, end) - address - begin;
            covered |= (length >= 64 ? ~u64{0} : (u64{1} << length) - 1) << begin;
            if (covered == all_bits) {
                return;
            }
        }
    }
}

HazardResult HazardTable::Find(u64 address, u64 size, u64 consumed, const HazardFilter& filter,
                               u8* out) {
    HazardResult result;
    if (size == 0) {
        return result;
    }
    u64 wait = 0;
    for (size_t p = 0; p < NumProducers; ++p) {
        const u32 bit = 1U << p;
        u64 position = 0;
        if ((filter.unbounded & bit) != 0) {
            position = unbounded[p];
        } else if ((filter.unbounded_synced & bit) != 0) {
            position = unbounded_synced[p];
        }
        if (position > consumed) {
            wait = std::max(wait, position);
            result.producers |= bit;
        }
    }
    if (overflow_position > consumed) {
        wait = std::max(wait, overflow_position);
        result.overflow = true;
    }
    const u64 end = address + size;
    std::array<u32, 64> found;
    size_t num_found = 0;
    const bool track_cover = size <= 64;
    const auto bits_of = [](u64 begin, u64 stop) {
        const u64 length = stop - begin;
        return (length >= 64 ? ~u64{0} : (u64{1} << length) - 1) << begin;
    };
    u64 covered = 0;
    const auto cover = [&](const Entry& entry) {
        if (track_cover) {
            covered |= bits_of(std::max(entry.begin, address) - address,
                               std::min(entry.end, end) - address);
        }
    };
    if (count != 0 && filter.ranged != 0 &&
        entries[(head + count - 1) & (Capacity - 1)].position > consumed) {
        const u64 first_page = address >> PageBits;
        const u64 last_page = (end - 1) >> PageBits;
        bool linear = last_page - first_page >= MaxLookupPages;
        for (size_t p = 0; p < NumProducers && !linear; ++p) {
            linear = (filter.ranged & (1U << p)) != 0 && wide[p] > consumed;
        }
        if (linear) {
            Linear(address, end, consumed, filter.ranged, result, wait, found, num_found,
                   covered);
        } else {
            for (size_t i = 0; i < wide_count; ++i) {
                const Link link = wide_entries[(wide_head + i) % MaxWide];
                if (!Live(link, consumed)) {
                    continue;
                }
                const Entry& entry = entries[link.index];
                const u32 bit = ProducerBit(entry.producer);
                if ((filter.ranged & bit) == 0 || entry.end <= address || entry.begin >= end) {
                    continue;
                }
                // Wide writes are not forwarded (their order against the chains' is unknown).
                result.producers |= bit;
                wait = std::max(wait, entry.position);
            }
            for (u64 page = first_page; page <= last_page; ++page) {
                // The read's bytes on this page: once newer known stores cover them, older
                // entries on the page don't matter.
                const u64 page_bits =
                    track_cover ? bits_of(std::max(address, page << PageBits) - address,
                                          std::min(end, (page + 1) << PageBits) - address)
                                : 0;
                Link link = (*slots)[Slot(page)];
                size_t visited = 0;
                while (Live(link, consumed)) {
                    const Entry& entry = entries[link.index];
                    if (++visited > MaxScan) {
                        wait = std::max(wait, (*slots)[Slot(page)].position);
                        result.overflow = true;
                        break;
                    }
                    const u32 bit = ProducerBit(entry.producer);
                    if ((filter.ranged & bit) != 0 && entry.end > address && entry.begin < end) {
                        result.producers |= bit;
                        if (entry.known == 0) {
                            wait = std::max(wait, entry.position);
                            break; // Older ones on this page are behind the wait
                        }
                        if (num_found == found.size()) {
                            wait = std::max(wait, entry.position);
                            result.overflow = true;
                            break;
                        }
                        found[num_found++] = link.index;
                        cover(entry);
                        if (track_cover && (covered & page_bits) == page_bits) {
                            break;
                        }
                    }
                    link = entry.previous;
                }
            }
        }
    }
    if (filter.synced_draw_targets) {
        if (targets_overflow > consumed) {
            wait = std::max(wait, targets_overflow);
            result.overflow = true;
        }
        for (const TargetSet& set : target_sets) {
            if (set.synced <= consumed) {
                continue;
            }
            for (u32 i = 0; i < set.count; ++i) {
                if (set.ranges[i].first < end && set.ranges[i].second > address) {
                    wait = std::max(wait, set.synced);
                    result.producers |= ProducerBit(Producer::Draw);
                    break;
                }
            }
        }
    }
    if (wait > consumed) {
        result.kind = HazardResult::Kind::Wait;
        result.position = wait;
        return result;
    }
    if (num_found == 0) {
        return result;
    }
    // Oldest first, so newer stores win.
    std::sort(found.begin(), found.begin() + num_found,
              [this](u32 a, u32 b) { return entries[a].position < entries[b].position; });
    covered = 0;
    for (size_t i = 0; i < num_found; ++i) {
        const Entry& entry = entries[found[i]];
        const u64 begin = std::max(entry.begin, address);
        const u64 stop = std::min(entry.end, end);
        if (out != nullptr) {
            std::memcpy(out + (begin - address), entry.bytes.data() + (begin - entry.begin),
                        static_cast<size_t>(stop - begin));
        }
        cover(entry);
    }
    result.kind = HazardResult::Kind::Forward;
    result.position = entries[found[num_found - 1]].position;
    result.covered = track_cover && covered == bits_of(0, size);
    return result;
}

AcquireDecision DecideAcquire(SemaphoreOp op, u32 value, const HazardResult& word, u32 forwarded,
                              u32 memory_word, u64 pending_dispatch) {
    switch (word.kind) {
    case HazardResult::Kind::Wait:
        // A queued counter report or reduction: the front can't tell what it leaves.
        return {AcquireVerdict::Fallback, word.position};
    case HazardResult::Kind::Forward:
        if (!word.covered) {
            return {AcquireVerdict::Fallback, word.position};
        }
        if (!SemaphoreMet(op, forwarded, value)) {
            return {AcquireVerdict::Fallback, word.position};
        }
        // The host runs the release before the acquire's channel goes on. A dispatch queued
        // after the release may release the word again (QMD releases the front can't see).
        if (!SemaphoreWaitsToReach(op) && pending_dispatch > word.position) {
            return {AcquireVerdict::WaitFor, pending_dispatch};
        }
        return {AcquireVerdict::MetByQueued, word.position};
    case HazardResult::Kind::Clear:
        break;
    }
    if (!SemaphoreWaitsToReach(op) && pending_dispatch != 0) {
        return {AcquireVerdict::WaitFor, pending_dispatch};
    }
    return {SemaphoreMet(op, memory_word, value) ? AcquireVerdict::Met : AcquireVerdict::NotMet, 0};
}

} // namespace VideoCommon::V2
