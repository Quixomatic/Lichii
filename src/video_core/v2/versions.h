// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common/common_types.h"
#include "video_core/v2/v2_switches.h"

// Pipeline v2 P4: per-resource versions, collect epochs and generations (MASTER_PLAN §5.3 step 2,
// §8; notes/05 §3.1 "collect epochs", §3.9 items 3 and 8; notes/06 §4.3; notes/03 §4).
//
// Page versions. Every guest device page has a version: the version clock's value at its last
// write, kept in a hashed table of atomics (two pages sharing a slot share a version, which only
// makes a check fail more often). Writes over many pages bump coarse region slots instead, so a
// render target's first GPU write costs a few stores, not thousands. The clock moves only when a
// reader takes a stamp, so writers never contend on it: a write stores the clock's value; a stamp
// returns the value and moves the clock past it, so every write after the stamp is > V and every
// write before it is <= V.
//   - "Range unchanged since V" = every page and region slot of the range is <= V.
//   - Write sources: guest CPU writes to tracked pages (Core::Memory's rasterizer write handler),
//     host invalidations (any thread), DMA engine writes, GPU writes (render targets on their
//     first write since a sync, storage buffers, copies) and inline uploads (cbuf data, I2M).
//
// Collect epochs. The epoch moves at ordering points: submits, semaphore acquires, the split's
// drains and targeted waits, and GP entry boundaries. A guest CPU write is only ordered before
// GPU work by one of those, so "unchanged since V" answers for a range are memoised per epoch
// (EpochMemo): a repeat check in the same epoch costs one probe. Writes made by the GPU stream
// itself (inline uploads, DMA, GPU writes, host invalidations) are not deferred: they also go to
// the stream log, and a memo entry is only reused while no logged write since it overlaps it.
//
// Generations. Resources referenced by a draw recipe are weak handles {slot id, generation}: the
// owning cache bumps a slot's generation when it frees, recreates or rescales the object, so a
// dead handle is a recipe miss, never a dangling reference. Cache-wide epochs (pipelines, image
// registration, resolution scale) cover what has no slot.
//
// Thread safety: every table here is atomics and can be read from any thread (P5's lock-free
// lookups, P6's resolve shards). EpochMemo and PageWatch belong to one thread (the resolver).
namespace VideoCommon::V2::Versions {

enum class Source : u8 {
    Cpu,    ///< guest CPU write to a tracked page
    Host,   ///< host invalidation (OnCPUWrite, InvalidateRegion, cache invalidations)
    Dma,    ///< DMA engine write (copy engine, accelerated buffer copies and clears)
    Gpu,    ///< GPU write (render target first write, storage buffer, image copy)
    Inline, ///< inline upload (constant buffer data, inline-to-memory)
    Count,
};
inline constexpr size_t NumSources = static_cast<size_t>(Source::Count);
inline constexpr std::array<const char*, NumSources> source_names{"cpu", "host", "dma", "gpu",
                                                                 "inline"};

enum class EpochReason : u8 {
    Submit,
    Acquire,
    Wait,  ///< a drain or a targeted wait of the split
    Entry, ///< a GP entry boundary
    Count,
};
inline constexpr size_t NumEpochReasons = static_cast<size_t>(EpochReason::Count);
inline constexpr std::array<const char*, NumEpochReasons> epoch_reason_names{"submit", "acquire",
                                                                             "wait", "entry"};

class PageVersions {
public:
    static constexpr u64 PageBits = 12;
    static constexpr u64 RegionBits = 18; ///< 64 pages
    static constexpr size_t PageSlots = size_t{1} << 18;
    static constexpr size_t RegionSlots = size_t{1} << 14;
    /// Writes over more pages than this bump region slots.
    static constexpr u64 MaxPageWrite = 16;
    /// Writes over more regions than this move the floor (every range changes).
    static constexpr u64 MaxRegionWrite = 1024;

    PageVersions();

    /// A stamp V: every write noted after this call is > V. Take it before reading the inputs.
    [[nodiscard]] u64 Stamp() {
        return clock.fetch_add(1, std::memory_order_acq_rel);
    }
    /// The clock without moving it (stats).
    [[nodiscard]] u64 Now() const {
        return clock.load(std::memory_order_acquire);
    }

    /// A write of [address, address + size) from `source` (any thread).
    void Note(DAddr address, u64 size, Source source);

    /// A guest CPU write: only a load and a compare when the page already shows a version newer
    /// than every stamp taken (the usual case for repeated writes).
    void NoteCpuWrite(DAddr address, u64 size) {
        const u64 page = address >> PageBits;
        if (((address + size - 1) >> PageBits) == page) [[likely]] {
            const u64 value = clock.load(std::memory_order_relaxed);
            std::atomic<u64>& slot = pages[PageSlot(page)];
            if (slot.load(std::memory_order_relaxed) >= value) {
                return;
            }
            Raise(slot, value);
            counts[static_cast<size_t>(Source::Cpu)].fetch_add(1, std::memory_order_relaxed);
            return;
        }
        Note(address, size, Source::Cpu);
    }

    /// Whether nothing wrote [address, address + size) after stamp `since`.
    [[nodiscard]] bool Unchanged(DAddr address, u64 size, u64 since) const;
    /// The newest version of [address, address + size) (0 when never written).
    [[nodiscard]] u64 Newest(DAddr address, u64 size) const;

    /// Writes noted per source since the last call (stats).
    [[nodiscard]] std::array<u64, NumSources> TakeCounts();

    /// Forgets every version (tests).
    void Reset();

    [[nodiscard]] static size_t PageSlot(u64 page) {
        u64 key = page * 0x9e3779b97f4a7c15ULL;
        return static_cast<size_t>(key >> (64 - 18));
    }
    [[nodiscard]] static size_t RegionSlot(u64 region) {
        u64 key = region * 0xbf58476d1ce4e5b9ULL;
        return static_cast<size_t>(key >> (64 - 14));
    }

private:
    static void Raise(std::atomic<u64>& slot, u64 value) {
        u64 current = slot.load(std::memory_order_relaxed);
        while (current < value &&
               !slot.compare_exchange_weak(current, value, std::memory_order_release,
                                           std::memory_order_relaxed)) {
        }
    }

    std::atomic<u64> clock{1};
    std::atomic<u64> floor{0};
    std::unique_ptr<std::atomic<u64>[]> pages;
    std::unique_ptr<std::atomic<u64>[]> regions;
    std::array<std::atomic<u64>, NumSources> counts{};
};

/// Ring of the stream's own writes (every source but guest CPU writes), for EpochMemo. Lock free
/// for many writers; a slot being rewritten reads as an overlap.
class StreamLog {
public:
    static constexpr size_t Size = 256;

    void Note(DAddr begin, DAddr end);
    [[nodiscard]] u64 Head() const {
        return head.load(std::memory_order_acquire);
    }
    /// Whether a write logged at or after position `since` may overlap [begin, end): true when one
    /// does, when a slot is being written, or when the ring no longer holds them all.
    [[nodiscard]] bool WrittenSince(u64 since, DAddr begin, DAddr end) const;

    void Reset();

private:
    struct Slot {
        std::atomic<u64> sequence{}; ///< position + 1 once written
        std::atomic<DAddr> begin{};
        std::atomic<DAddr> end{};
    };
    std::atomic<u64> head{};
    std::array<Slot, Size> slots{};
};

/// The collect epoch: moved at ordering points (any thread).
class CollectEpoch {
public:
    void Advance(EpochReason reason) {
        epoch.fetch_add(1, std::memory_order_acq_rel);
        counts[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    }
    [[nodiscard]] u64 Current() const {
        return epoch.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::array<u64, NumEpochReasons> TakeCounts();

private:
    std::atomic<u64> epoch{1};
    std::array<std::atomic<u64>, NumEpochReasons> counts{};
};

/// "Unchanged since V" answers memoised per collect epoch (owned by one thread). An entry holds the
/// newest version a walk of its range found; within its epoch, and while no stream write since
/// overlaps it, it answers without walking.
class EpochMemo {
public:
    static constexpr size_t Entries = 256;

    struct Counters {
        u64 queries{};
        u64 hits{};
        u64 walks{};
    };

    EpochMemo(PageVersions& versions_, StreamLog& log_, CollectEpoch& epoch_)
        : versions{&versions_}, log{&log_}, epoch{&epoch_} {}

    [[nodiscard]] bool Unchanged(DAddr address, u64 size, u64 since);
    /// The same without the memo (verify).
    [[nodiscard]] bool UnchangedNow(DAddr address, u64 size, u64 since) const {
        return versions->Unchanged(address, size, since);
    }

    void Clear();
    [[nodiscard]] Counters TakeCounters() {
        return std::exchange(counters, Counters{});
    }

private:
    struct Entry {
        DAddr begin{};
        DAddr end{};
        u64 epoch{};
        u64 log_head{};
        u64 newest{};
    };

    PageVersions* versions;
    StreamLog* log;
    CollectEpoch* epoch;
    std::array<Entry, Entries> entries{};
    Counters counters{};
};

/// Pages a resolver keeps in the rasterizer's write tracking, so guest CPU writes to them trap and
/// bump their versions (constant buffer words holding handles, texture and sampler pool entries:
/// memory no cache holds). Reference counted; pages nobody holds are released once more than
/// MaxIdle pile up. Owned by one thread.
class PageWatch {
public:
    static constexpr size_t MaxIdle = 1024;
    /// Changes the cached count of the device pages of [address, address + size) by delta.
    using Tracker = std::function<void(DAddr address, u64 size, s32 delta)>;

    void SetTracker(Tracker tracker_);
    void Hold(DAddr address, u64 size);
    void Drop(DAddr address, u64 size);
    /// Releases every page through the tracker.
    void Clear();
    [[nodiscard]] size_t Pages() const {
        return refs.size();
    }
    [[nodiscard]] bool HasTracker() const {
        return static_cast<bool>(tracker);
    }

private:
    void ReleaseIdle();

    Tracker tracker;
    std::unordered_map<u64, u32> refs; ///< tracked page -> holders
    std::unordered_set<u64> idle;      ///< tracked pages nobody holds
};

/// Resource kinds with per-slot generations.
enum class Kind : u8 {
    ImageView,
    Image,
    Sampler,
    Buffer,
    Count,
};
inline constexpr size_t NumKinds = static_cast<size_t>(Kind::Count);

/// Cache-wide epochs for what has no slot.
enum class Epoch : u8 {
    Pipelines, ///< a pipeline evicted, swapped or its shader invalidated
    Rescale,   ///< an image rescaled (its views' generations move too)
    Count,
};
inline constexpr size_t NumEpochs = static_cast<size_t>(Epoch::Count);

/// Per-slot generations: chunked arrays of atomics, allocated on first use of a chunk.
class Generations {
public:
    static constexpr size_t ChunkBits = 12;
    static constexpr size_t ChunkSize = size_t{1} << ChunkBits;
    static constexpr size_t MaxChunks = 256; ///< 1M slots per kind

    /// The generation of slot `id` (a SlotId index; its tag bit is ignored). Ids past the table
    /// (the invalid id) read as ~0, which no bump changes.
    [[nodiscard]] u32 Get(Kind kind, u32 id) const {
        id &= 0x7fffffffU;
        const size_t chunk = id >> ChunkBits;
        if (chunk >= MaxChunks) {
            return ~0U;
        }
        const auto* const array = chunks[static_cast<size_t>(kind)][chunk].load(
            std::memory_order_acquire);
        return array == nullptr ? 0 : array[id & (ChunkSize - 1)].load(std::memory_order_acquire);
    }
    void Bump(Kind kind, u32 id);

    [[nodiscard]] u64 GetEpoch(Epoch epoch) const {
        return epochs[static_cast<size_t>(epoch)].load(std::memory_order_acquire);
    }
    void BumpEpoch(Epoch epoch) {
        epochs[static_cast<size_t>(epoch)].fetch_add(1, std::memory_order_acq_rel);
        bumps.fetch_add(1, std::memory_order_relaxed);
    }
    [[nodiscard]] u64 TakeBumps() {
        return bumps.exchange(0, std::memory_order_relaxed);
    }

    ~Generations();

private:
    std::array<std::array<std::atomic<std::atomic<u32>*>, MaxChunks>, NumKinds> chunks{};
    std::array<std::atomic<u64>, NumEpochs> epochs{};
    std::atomic<u64> bumps{};
};

/// A weak handle: a slot id and the generation it had when taken.
struct WeakRef {
    u32 id{};
    u32 generation{};

    bool operator==(const WeakRef&) const = default;
};

/// Whether the draw cache (and so the version hooks) is on: one load of the cached modes.
[[nodiscard]] inline bool Active() {
    return On(Switch::DrawCache);
}

/// The process-wide tables (one GPU).
[[nodiscard]] PageVersions& Pages();
[[nodiscard]] StreamLog& Stream();
[[nodiscard]] CollectEpoch& Epochs();
[[nodiscard]] Generations& Gens();

/// A write from any source but guest CPU writes: page versions and the stream log.
inline void NoteWrite(DAddr address, u64 size, Source source) {
    if (size == 0) {
        return;
    }
    Pages().Note(address, size, source);
    Stream().Note(address, address + size);
}

/// Hook helpers: one cached-flag branch when the draw cache is off.
inline void HookWrite(DAddr address, u64 size, Source source) {
    if (Active()) [[unlikely]] {
        NoteWrite(address, size, source);
    }
}
inline void HookCpuWrite(DAddr address, u64 size) {
    if (Active()) [[unlikely]] {
        Pages().NoteCpuWrite(address, size);
    }
}
inline void HookEpoch(EpochReason reason) {
    if (Active()) [[unlikely]] {
        Epochs().Advance(reason);
    }
}
inline void HookBump(Kind kind, u32 id) {
    if (Active()) [[unlikely]] {
        Gens().Bump(kind, id);
    }
}
inline void HookBumpEpoch(Epoch epoch) {
    if (Active()) [[unlikely]] {
        Gens().BumpEpoch(epoch);
    }
}

} // namespace VideoCommon::V2::Versions
