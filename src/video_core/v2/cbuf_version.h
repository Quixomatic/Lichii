// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common_types.h"

// Pipeline v2 P3 (EDEN_NX2_V2_CBUFVER): constant buffer versions (MASTER_PLAN §5.3, notes/04
// §10.3). A small constant buffer the shaders read is copied into the stream buffer once per
// version instead of once per draw and stage: the copy is a version of {device address, size},
// and every later bind of the same range (any stage, any slot) in the same command buffer
// references it while nothing can have written the range since.
//
// What can write a range, and how a version learns of it:
// - inline constant buffer data and inline-to-memory uploads (the GPU's cached writes, flushed
//   by MemoryManager::FlushCaching before a draw resolves): exact device ranges, in order, in a
//   write log (NoteInline);
// - guest CPU writes: the pages of every version stay in the rasterizer's write tracking (cached
//   count) while the table holds them, so every CPU write to them takes the trapped path, which
//   bumps the pages' counters (NoteCpuWrite, from Core::Memory's rasterizer write handler);
// - other host writes the rasterizer is told about from any thread (invalidations): the same
//   per-page counters (NoteHost);
// - GPU writes (shaders, DMA): the caller only takes this path for ranges the buffer cache does
//   not hold GPU-modified.
// Versions live for one scheduler tick, as the stream buffer allocation they reference.
namespace VideoCommon::V2::CbufVer {

class WriteLog {
public:
    static constexpr size_t LogSize = 128;
    static constexpr size_t PageBuckets = 4096;
    static constexpr u64 PageBits = 12;
    static constexpr u64 PageSize = u64{1} << PageBits;
    /// Host writes longer than this many pages end every version instead.
    static constexpr u64 MaxHostPages = 64;

    /// The GPU's own cached write of [begin, end) (drawing thread).
    void NoteInline(DAddr begin, DAddr end);
    /// A host write of [address, address + size) (any thread).
    void NoteHost(DAddr address, u64 size);
    /// A guest CPU write to a rasterizer-tracked page (any thread): counted for watched pages.
    void NoteCpuWrite(DAddr address, u64 size) {
        const u64 page = address >> PageBits;
        if (watch[page % PageBuckets].load(std::memory_order_relaxed) == 0 &&
            ((address + size - 1) >> PageBits) == page) [[likely]] {
            return;
        }
        NoteHost(address, size);
    }
    /// Everything before is unknown.
    void NoteBoundary() {
        boundary.fetch_add(1, std::memory_order_release);
    }
    /// A version's table starts or stops watching `page` (a page index).
    void Watch(u64 page, s32 delta) {
        watch[page % PageBuckets].fetch_add(static_cast<u32>(delta), std::memory_order_acq_rel);
    }

    [[nodiscard]] u64 Sequence() const {
        return sequence.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 Boundary() const {
        return boundary.load(std::memory_order_acquire);
    }
    /// Sum of the host write counters of the pages of [address, address + size).
    [[nodiscard]] u64 PageSum(DAddr address, u32 size) const;
    /// Whether an inline write logged after `since` (a Sequence) may overlap [begin, end): true
    /// when one does or when the log no longer holds them all.
    [[nodiscard]] bool WrittenSince(u64 since, DAddr begin, DAddr end) const;

    void Reset();

private:
    struct Range {
        DAddr begin;
        DAddr end;
    };
    std::array<Range, LogSize> ranges{};
    std::atomic<u64> sequence{};
    std::atomic<u64> boundary{};
    std::array<std::atomic<u32>, PageBuckets> pages{};
    std::array<std::atomic<u32>, PageBuckets> watch{};
};

/// The process-wide write log (one GPU, one drawing thread).
[[nodiscard]] WriteLog& Log();

/// Versions by {device address, size}, direct mapped. Every entry keeps its pages in the
/// rasterizer's write tracking (through the tracker, the device memory's cached page counts) so
/// CPU writes to them always trap; pages no entry holds any more are let go once more than
/// MaxIdlePages pile up.
template <typename Ref>
class VersionTable {
public:
    static constexpr size_t NumEntries = 1024;
    static constexpr size_t MaxIdlePages = 1024;
    /// Changes the cached count of the device pages of [address, address + size) by delta.
    using Tracker = std::function<void(DAddr address, u64 size, s32 delta)>;

    explicit VersionTable(WriteLog& log_) : log{&log_} {}

    /// A new owner (buffer cache): forgets every entry and page without the old tracker (its
    /// memory is gone) and uses `tracker` from now on.
    void Reset(Tracker tracker_) {
        for (const auto& [page, refs] : page_refs) {
            log->Watch(page, -1);
        }
        page_refs.clear();
        idle.clear();
        for (Entry& entry : entries) {
            entry = Entry{};
        }
        tracker = std::move(tracker_);
    }

    /// Starts watching the pages of [address, address + size) for a version about to be made, and
    /// returns their write counters' sum: take it before reading the bytes.
    [[nodiscard]] u64 Track(DAddr address, u32 size) {
        ForEachPage(address, size, [&](u64 page) {
            if (page_refs.try_emplace(page, 0).second) {
                // Into the write tracking before the sum is read: a later write traps.
                if (tracker) {
                    tracker(page << WriteLog::PageBits, WriteLog::PageSize, 1);
                }
                log->Watch(page, 1);
                idle.insert(page);
            }
        });
        return log->PageSum(address, size);
    }

    /// The version of [address, address + size) made in `tick`, if nothing wrote it since.
    [[nodiscard]] const Ref* Find(DAddr address, u32 size, u64 tick) {
        Entry& entry = entries[Slot(address, size)];
        if (!entry.valid || entry.address != address || entry.size != size ||
            entry.tick != tick || entry.boundary != log->Boundary()) {
            return nullptr;
        }
        const u64 sequence = log->Sequence();
        if (entry.sequence != sequence) {
            if (log->WrittenSince(entry.sequence, address, address + size)) {
                entry.valid = false;
                return nullptr;
            }
            entry.sequence = sequence;
        }
        if (entry.page_sum != log->PageSum(address, size)) {
            entry.valid = false;
            return nullptr;
        }
        return &entry.ref;
    }

    /// Records a new version whose pages Track took (`page_sum` its result); `bytes` (verify
    /// only) are what it holds.
    void Insert(DAddr address, u32 size, u64 tick, const Ref& ref, u64 page_sum,
                const u8* bytes = nullptr) {
        Entry& entry = entries[Slot(address, size)];
        Hold(address, size);
        if (entry.holds) {
            Drop(entry.address, entry.size);
        }
        entry.valid = true;
        entry.holds = true;
        entry.address = address;
        entry.size = size;
        entry.tick = tick;
        entry.boundary = log->Boundary();
        entry.sequence = log->Sequence();
        entry.page_sum = page_sum;
        entry.ref = ref;
        if (bytes != nullptr) {
            entry.bytes.assign(bytes, bytes + size);
        } else {
            entry.bytes.clear();
        }
    }

    /// The bytes Insert was given for the version Find returned (verify).
    [[nodiscard]] const std::vector<u8>& Bytes(DAddr address, u32 size) const {
        return entries[Slot(address, size)].bytes;
    }

    /// Drops every entry and stops watching every page (through the tracker).
    void Clear() {
        for (Entry& entry : entries) {
            if (entry.holds) {
                Drop(entry.address, entry.size);
            }
            entry = Entry{};
        }
        ReleaseIdle();
    }

    /// Pages kept in the write tracking now (tests, stats).
    [[nodiscard]] size_t TrackedPages() const {
        return page_refs.size();
    }

private:
    struct Entry {
        bool valid{};
        bool holds{}; ///< its pages count in page_refs
        DAddr address{};
        u32 size{};
        u64 tick{};
        u64 boundary{};
        u64 sequence{};
        u64 page_sum{};
        Ref ref{};
        std::vector<u8> bytes;
    };

    template <typename Func>
    static void ForEachPage(DAddr address, u32 size, Func&& func) {
        if (size == 0) {
            return;
        }
        const u64 first = address >> WriteLog::PageBits;
        const u64 last = (address + size - 1) >> WriteLog::PageBits;
        for (u64 page = first; page <= last; ++page) {
            func(page);
        }
    }

    void Hold(DAddr address, u32 size) {
        ForEachPage(address, size, [&](u64 page) {
            u32& refs = page_refs[page];
            if (refs++ == 0) {
                idle.erase(page);
            }
        });
    }

    void Drop(DAddr address, u32 size) {
        ForEachPage(address, size, [&](u64 page) {
            const auto it = page_refs.find(page);
            if (it != page_refs.end() && it->second != 0 && --it->second == 0) {
                idle.insert(page);
            }
        });
        if (idle.size() > MaxIdlePages) {
            ReleaseIdle();
        }
    }

    void ReleaseIdle() {
        for (const u64 page : idle) {
            page_refs.erase(page);
            if (tracker) {
                tracker(page << WriteLog::PageBits, WriteLog::PageSize, -1);
            }
            log->Watch(page, -1);
        }
        idle.clear();
    }

    static size_t Slot(DAddr address, u32 size) {
        u64 key = (address >> 4) ^ (static_cast<u64>(size) << 40);
        key ^= key >> 29;
        key *= 0xbf58476d1ce4e5b9ULL;
        key ^= key >> 32;
        return static_cast<size_t>(key) & (NumEntries - 1);
    }

    WriteLog* log;
    Tracker tracker;
    std::array<Entry, NumEntries> entries{};
    std::unordered_map<u64, u32> page_refs; ///< tracked page -> entries holding it
    std::unordered_set<u64> idle;           ///< tracked pages no entry holds
};

/// The table of a buffer cache runtime's stream references (one buffer cache at a time).
template <typename Ref>
[[nodiscard]] VersionTable<Ref>& Table() {
    static VersionTable<Ref> table{Log()};
    return table;
}

} // namespace VideoCommon::V2::CbufVer
