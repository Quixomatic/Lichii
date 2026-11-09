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

    /// The checks of Find that read no guest memory (Perf fix D: a lookup reads and compares
    /// the range's bytes only when these pass): whether the version of [address, address +
    /// size) made in `tick` may still hold. No inline or host write logged over it since, and
    /// `page_sum` (the range's PageSum now, Begin) equal to its own.
    [[nodiscard]] bool MayHold(DAddr address, u32 size, u64 tick, u64 page_sum) {
        Entry& entry = entries[Slot(address, size)];
        if (!entry.valid || entry.address != address || entry.size != size ||
            entry.tick != tick || entry.boundary != log->Boundary()) {
            return false;
        }
        const u64 sequence = log->Sequence();
        if (entry.sequence != sequence) {
            if (log->WrittenSince(entry.sequence, address, address + size)) {
                entry.valid = false;
                return false;
            }
            entry.sequence = sequence;
        }
        if (entry.page_sum != page_sum) {
            entry.valid = false;
            return false;
        }
        return true;
    }

    /// After MayHold returned true: the version, if `current` (what guest memory holds now,
    /// `size` bytes) equals its bytes. A guest CPU write that changed any byte makes it miss.
    [[nodiscard]] const Ref* Matching(DAddr address, u32 size, const u8* current) const {
        const Entry& entry = entries[Slot(address, size)];
        if (current == nullptr || !entry.valid || entry.address != address ||
            entry.bytes.size() != size || std::memcmp(entry.bytes.data(), current, size) != 0) {
            return nullptr;
        }
        return &entry.ref;
    }

    /// The version of [address, address + size) made in `tick`, if nothing wrote it since:
    /// no inline or host write logged over it, and `current` (what guest memory holds now, `size`
    /// bytes) equal to its bytes.
    [[nodiscard]] const Ref* Find(DAddr address, u32 size, u64 tick, const u8* current) {
        if (!MayHold(address, size, tick, log->PageSum(address, size))) {
            return nullptr;
        }
        return Matching(address, size, current);
    }

    /// As Find, reading guest memory (`read()` returns its `size` bytes) only when the version
    /// may still hold.
    template <typename Read>
    [[nodiscard]] const Ref* FindWith(DAddr address, u32 size, u64 tick, Read&& read) {
        if (!MayHold(address, size, tick, log->PageSum(address, size))) {
            return nullptr;
        }
        return Matching(address, size, read());
    }

    /// Records a new version made from `bytes` (`size` of them) after Begin returned `page_sum`.
    void Insert(DAddr address, u32 size, u64 tick, const Ref& ref, u64 page_sum,
                const u8* bytes) {
        u8* const storage = Prepare(address, size, tick, page_sum);
        std::memcpy(storage, bytes, size);
        Commit(address, size, ref);
    }

    /// A new version of [address, address + size) in `tick` after Begin returned `page_sum`:
    /// returns where its `size` bytes go. The caller reads guest memory into it once, copies
    /// it to the stream (so the two can't differ), then Commit makes the version findable.
    [[nodiscard]] u8* Prepare(DAddr address, u32 size, u64 tick, u64 page_sum) {
        Entry& entry = entries[Slot(address, size)];
        entry.valid = false;
        entry.address = address;
        entry.size = size;
        entry.tick = tick;
        entry.boundary = log->Boundary();
        entry.sequence = log->Sequence();
        entry.page_sum = page_sum;
        entry.bytes.resize(size);
        return entry.bytes.data();
    }
    void Commit(DAddr address, u32 size, const Ref& ref) {
        Entry& entry = entries[Slot(address, size)];
        if (entry.address != address || entry.size != size || entry.bytes.size() != size) {
            return;
        }
        entry.ref = ref;
        entry.valid = true;
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
