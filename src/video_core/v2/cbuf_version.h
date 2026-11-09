// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
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
// - other host writes the rasterizer is told about from any thread (invalidations): per-page
//   counters (NoteHost);
// - guest CPU writes: every version keeps a copy of its bytes, and a lookup compares the range's
//   bytes in guest memory with it. A CPU write that changed any byte ends the version at its next
//   lookup, with no page kept in the write tracking: no protection changes, no trapped guest
//   reads or writes (the copy is what the shader reads, so equal bytes are the same version);
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
    /// Everything before is unknown.
    void NoteBoundary() {
        boundary.fetch_add(1, std::memory_order_release);
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
};

/// The process-wide write log (one GPU, one drawing thread).
[[nodiscard]] WriteLog& Log();

/// Versions by {device address, size}, direct mapped. Each entry keeps the bytes it was made
/// from; Find compares them with what guest memory holds now.
template <typename Ref>
class VersionTable {
public:
    static constexpr size_t NumEntries = 1024;

    explicit VersionTable(WriteLog& log_) : log{&log_} {}

    /// A new owner (buffer cache): forgets every entry.
    void Reset() {
        for (Entry& entry : entries) {
            entry = Entry{};
        }
    }

    /// The host write counters' sum of [address, address + size) for a version about to be made:
    /// take it before reading the bytes.
    [[nodiscard]] u64 Begin(DAddr address, u32 size) const {
        return log->PageSum(address, size);
    }

    /// The version of [address, address + size) made in `tick`, if nothing wrote it since:
    /// no inline or host write logged over it, and `current` (what guest memory holds now, `size`
    /// bytes) equal to its bytes.
    [[nodiscard]] const Ref* Find(DAddr address, u32 size, u64 tick, const u8* current) {
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
        if (current == nullptr || entry.bytes.size() != size ||
            std::memcmp(entry.bytes.data(), current, size) != 0) {
            // Written behind (a guest CPU write): a new version replaces it.
            return nullptr;
        }
        return &entry.ref;
    }

    /// Records a new version made from `bytes` (`size` of them) after Begin returned `page_sum`.
    void Insert(DAddr address, u32 size, u64 tick, const Ref& ref, u64 page_sum,
                const u8* bytes) {
        Entry& entry = entries[Slot(address, size)];
        entry.valid = true;
        entry.address = address;
        entry.size = size;
        entry.tick = tick;
        entry.boundary = log->Boundary();
        entry.sequence = log->Sequence();
        entry.page_sum = page_sum;
        entry.ref = ref;
        entry.bytes.assign(bytes, bytes + size);
    }

    /// The bytes of the version of [address, address + size) (verify).
    [[nodiscard]] const std::vector<u8>& Bytes(DAddr address, u32 size) const {
        return entries[Slot(address, size)].bytes;
    }

    /// Drops every entry.
    void Clear() {
        for (Entry& entry : entries) {
            entry.valid = false;
        }
    }

private:
    struct Entry {
        bool valid{};
        DAddr address{};
        u32 size{};
        u64 tick{};
        u64 boundary{};
        u64 sequence{};
        u64 page_sum{};
        Ref ref{};
        std::vector<u8> bytes;
    };

    static size_t Slot(DAddr address, u32 size) {
        u64 key = (address >> 4) ^ (static_cast<u64>(size) << 40);
        key ^= key >> 29;
        key *= 0xbf58476d1ce4e5b9ULL;
        key ^= key >> 32;
        return static_cast<size_t>(key) & (NumEntries - 1);
    }

    WriteLog* log;
    std::array<Entry, NumEntries> entries{};
};

/// The table of a buffer cache runtime's stream references (one buffer cache at a time).
template <typename Ref>
[[nodiscard]] VersionTable<Ref>& Table() {
    static VersionTable<Ref> table{Log()};
    return table;
}

} // namespace VideoCommon::V2::CbufVer
