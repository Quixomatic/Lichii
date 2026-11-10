// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <span>
#include <utility>

#include "common/common_types.h"

// Pipeline v2 range hazards (MASTER_PLAN.md §5.1, §6; AnyPS5's DrawPipeline::StoredBytes). The
// split front records every guest memory write it sends to the back, keyed by the ring position
// after the record that makes it (a GpuSplit ticket's position): its device address range and,
// for small writes whose value the front knows (semaphore releases with literal payloads), the
// bytes. A front read of guest memory then asks the table first:
// - Clear: nothing still queued for the back writes the range; read memory.
// - Forward: only writes of known bytes overlap; they are applied over what memory holds.
// - Wait: a queued write of unknown bytes, or a write the front can't bound (a compute dispatch,
//   a copy into block-linear memory), may touch the range: wait until the back consumed
//   `position` (GpuSplit::WaitFor) and look again.
//
// The table belongs to the front thread. It reads the back's consumed position (an atomic load
// the caller passes in) and retires entries the back ran; it has no lock and no shared state.
// Memory is bounded: past Capacity entries the oldest pending ones fold into one marker that
// makes every lookup wait for them.
namespace VideoCommon::V2 {

/// What makes a write. Lookups choose the producers they must order against.
enum class Producer : u8 {
    Release,  ///< A semaphore release (puller, copy engine): literal payload
    Report,   ///< A 3D or compute report semaphore: payload literal or a GPU counter
    CbData,   ///< 3D constant buffer data
    Upload,   ///< Inline-to-memory, 3D and compute uploads
    Copy,     ///< Copy engine and 2D transfers
    Dispatch, ///< Compute dispatches: their shaders' stores
    Draw,     ///< 3D draws: shader stores, transform feedback (render targets are not noted)
    QmdRelease, ///< The semaphore releases compute QMDs make (the front doesn't parse them)
    Count,
};

[[nodiscard]] constexpr u32 ProducerBit(Producer producer) {
    return 1U << static_cast<u32>(producer);
}
inline constexpr u32 AllProducers = (1U << static_cast<u32>(Producer::Count)) - 1;

inline constexpr const char* producer_names[] = {
    "release", "report", "cbuf data", "upload", "copy", "dispatch", "draw", "qmd release",
};

/// Which pending writes a lookup orders against.
struct HazardFilter {
    u32 ranged = AllProducers;   ///< Ranged writes of these producers that overlap
    u32 unbounded = 0;           ///< Unbounded writes of these producers, any still queued
    u32 unbounded_synced = 0;    ///< ...only those sent before the last host-level sync
};

struct HazardResult {
    enum class Kind : u8 {
        Clear,
        Forward,
        Wait,
    };
    Kind kind = Kind::Clear;
    /// Wait: the ring position to wait for. Forward: the position of the newest store applied.
    u64 position = 0;
    /// Forward: every byte of the range came from known stores (memory's bytes are not needed).
    bool covered = false;
    /// The producers that made the result (ProducerBit).
    u32 producers = 0;
    /// Wait: the table overflowed (or a scan was too long) rather than a known write overlapping.
    bool overflow = false;
    /// GpuSplit::FindWrites: `position` is a landing ticket (v2/landing.h), not a ring position.
    /// Wait: a write only the query cache's operation makes (a counter, a timestamp) hasn't
    /// landed; waiting for the back to consume doesn't settle it. Forward: the newest store
    /// applied is a release still to land.
    bool landing = false;
    /// GpuSplit::FindWrites, Wait: a QMD release (Producer::QmdRelease, no landing ticket) the
    /// back consumed may still be to land at its operation; `position` is the ring position of
    /// its entry. Waiting for the back to consume doesn't settle it (GpuSplit::CompleteLanding).
    bool untracked = false;
};

/// Landing tickets (v2/landing.h): ring position << LandingSubBits | index within the entry.
inline constexpr u64 LandingSubBits = 8;

/// The ring position of a result's `position` (landing tickets are converted).
[[nodiscard]] constexpr u64 RingPositionOf(const HazardResult& result) {
    return result.landing ? result.position >> LandingSubBits : result.position;
}

class HazardTable {
public:
    static constexpr size_t Capacity = 4096;
    static constexpr size_t MaxKnownBytes = 16;
    /// Reads over more pages than this scan every pending entry.
    static constexpr u64 MaxLookupPages = 16;
    /// Entries a lookup visits before it answers Wait for the newest it would visit.
    static constexpr size_t MaxScan = 512;

    HazardTable();
    ~HazardTable();

    /// Forgets everything (the split stopped: ring positions start at 0 again).
    void Clear();

    /// A write of `size` (<= MaxKnownBytes) known bytes, made by the ring up to `position`.
    void NoteKnown(u64 address, const void* bytes, u32 size, u64 position, Producer producer);
    /// A write of [address, address + size) whose bytes the front doesn't know. A CbData write
    /// that continues the previous one extends it.
    void NoteRange(u64 address, u64 size, u64 position, Producer producer);
    /// A write the front can't bound, sent by the current channel (SetChannel).
    void NoteUnbounded(Producer producer, u64 position);
    /// A host-level wait was sent at `position`: the unbounded writes before it are complete
    /// before what the front fetches after it (HazardFilter::unbounded_synced). `position` may
    /// be older than the newest write noted (an acquire met by a queued release orders only the
    /// work before that release): the newest unbounded write at or before it counts, or, past
    /// what UnboundedHistory keeps, the oldest one kept (later, so the wait covers it).
    void NoteSync(u64 position);
    /// Fix F3f (W-01a): a wait that orders only the current channel's own work (a GP entry's
    /// sync bit, the puller's wait for idle): the unbounded writes that channel sent before
    /// `position` count for its later lookups; other channels' don't (they are ordered against
    /// it only through a semaphore, NoteSync). Lookups take the current channel's.
    void NoteChannelSync(u64 position);
    /// The channel the front runs (Scheduler::BindChannel).
    void SetChannel(u32 channel) {
        current_channel = channel;
    }
    static constexpr size_t UnboundedHistory = 8;
    /// Channels' own syncs, by bind id modulo this (channels sharing a slot share their syncs,
    /// which only makes their waits longer).
    static constexpr size_t ChannelSlots = 16;
    /// Drops entries the back consumed.
    void Retire(u64 consumed);

    /// The pending writes to [address, address + size) that `filter` orders against, with the
    /// back at `consumed`. For Forward, `out` (size bytes, filled with memory's bytes by the
    /// caller, or nullptr) gets the known stores applied in order.
    [[nodiscard]] HazardResult Find(u64 address, u64 size, u64 consumed, const HazardFilter& filter,
                                    u8* out = nullptr);

    /// The newest unbounded write of `producer` still queued, or 0.
    [[nodiscard]] u64 PendingUnbounded(Producer producer, u64 consumed) const {
        const u64 position = unbounded[static_cast<size_t>(producer)];
        return position > consumed ? position : 0;
    }

    /// Entries held (for tests and stats).
    [[nodiscard]] size_t Size() const {
        return count;
    }
    /// Entries folded into the overflow marker so far.
    [[nodiscard]] u64 Overflows() const {
        return overflows;
    }

private:
    /// An entry by index and position: stale once the slot holds a newer entry.
    struct Link {
        u32 index = 0;
        u64 position = 0;
    };
    struct Entry {
        u64 begin;
        u64 end;
        u64 position;
        Link previous; ///< The next older entry on a page with the same slot
        Producer producer;
        u8 known; ///< Bytes known (0: unknown bytes)
        bool chained; ///< Within one page: on its slot's chain (else a wide write)
        std::array<u8, MaxKnownBytes> bytes;
    };

    static constexpr size_t PageSlots = 4096;
    static constexpr size_t MaxWide = 64;

    Entry& Push(u64 address, u64 end, u64 position, Producer producer);
    /// Links a new entry into its page slot's chain, or marks it wide.
    void Insert(Entry& entry, u32 index);
    [[nodiscard]] bool Live(const Link& link, u64 consumed) const;
    static size_t Slot(u64 page);
    void Linear(u64 address, u64 end, u64 consumed, u32 ranged, HazardResult& result, u64& wait,
                std::array<u32, 64>& found, size_t& num_found, u64& covered) const;

    std::unique_ptr<Entry[]> entries;
    size_t head = 0; ///< Oldest entry
    size_t count = 0;
    /// Per page slot (a hash of the 4 KiB page): the newest entry on such a page.
    std::unique_ptr<std::array<Link, PageSlots>> slots;
    /// Writes over more than one page, newest last (validated like chain links).
    std::array<Link, MaxWide> wide_entries{};
    size_t wide_head = 0;
    size_t wide_count = 0;
    /// Per producer: the newest position of a wide write that fell out of wide_entries; lookups
    /// for it scan every pending entry.
    std::array<u64, static_cast<size_t>(Producer::Count)> wide{};
    std::array<u64, static_cast<size_t>(Producer::Count)> unbounded{};
    std::array<u64, static_cast<size_t>(Producer::Count)> unbounded_synced{};
    /// Per producer: the last UnboundedHistory distinct positions of its unbounded writes.
    std::array<std::array<u64, UnboundedHistory>, static_cast<size_t>(Producer::Count)>
        unbounded_history{};
    /// The channel of each unbounded_history entry.
    std::array<std::array<u32, UnboundedHistory>, static_cast<size_t>(Producer::Count)>
        unbounded_history_channel{};
    std::array<u64, static_cast<size_t>(Producer::Count)> unbounded_noted{};
    /// NoteChannelSync, per channel slot and producer; and each slot's newest unbounded write.
    std::array<std::array<u64, static_cast<size_t>(Producer::Count)>, ChannelSlots>
        channel_synced{};
    std::array<std::array<u64, static_cast<size_t>(Producer::Count)>, ChannelSlots>
        channel_unbounded{};
    u32 current_channel = 0;
    /// The newest unbounded write at or before `position` (of `channel` only, unless ~0U).
    [[nodiscard]] u64 SyncedUnbounded(size_t producer, u64 position, u32 channel) const;
    u64 overflow_position = 0; ///< Pending entries that no longer fit
    u64 overflows = 0;
};

// Semaphore acquires against the table (MASTER_PLAN.md §6 classes A and B).

/// An acquire's comparison.
enum class SemaphoreOp : u8 {
    Equal,          ///< word == value
    Gequal,         ///< word >= value
    CircularGequal, ///< (s32)(word - value) >= 0
    Mask,           ///< (word & value) != 0
    MaskZero,       ///< (word & value) == 0
};

[[nodiscard]] constexpr bool SemaphoreMet(SemaphoreOp op, u32 word, u32 value) {
    switch (op) {
    case SemaphoreOp::Equal:
        return word == value;
    case SemaphoreOp::Gequal:
        return word >= value;
    case SemaphoreOp::CircularGequal:
        return static_cast<s32>(word - value) >= 0;
    case SemaphoreOp::Mask:
        return (word & value) != 0;
    case SemaphoreOp::MaskZero:
        return (word & value) == 0;
    }
    return false;
}

/// Whether the acquire waits for a word that only advances (NVN2's syncs): a value that meets
/// it now still meets it after any later release.
[[nodiscard]] constexpr bool SemaphoreWaitsToReach(SemaphoreOp op) {
    return op == SemaphoreOp::Gequal || op == SemaphoreOp::CircularGequal;
}

enum class AcquireVerdict : u8 {
    Met,         ///< Class A: memory meets it and nothing queued writes the word
    MetByQueued, ///< Class B: the last write queued to the word meets it (host order)
    NotMet,      ///< Nothing queued writes the word and memory doesn't meet it
    WaitFor,     ///< Wait for `position` (the write the word depends on), then memory decides
    Fallback,    ///< Take the old path (drain, then read the word)
};

struct AcquireDecision {
    AcquireVerdict verdict = AcquireVerdict::Fallback;
    u64 position = 0;
};

/// Decides an acquire from the front's view of its word (GpuSplit::FindWrites over the queued
/// writes and the releases still to land; `forwarded`: the word's value if Forward), memory's
/// word now, and the QMD release the acquire orders behind (0 if none or it settled; Fix F3h:
/// fixed when the acquire was parsed, GpuSplit::QmdTargetFor), whose bytes the front can't see
/// (MASTER_PLAN §6):
///   Pending unknown write (counter report, reduction) or partly known: Fallback (class D).
///   Pending known write, queued or run but not landed, fence-flagged or not: the newest one's
///     value decides. MetByQueued (class B): the host queue runs it before the acquire's channel
///     goes on, and the landing rule (v2/landing.h) keeps later CPU-visible writes behind it.
///     NotMet when it doesn't meet the acquire: only a later release can. Equal and mask
///     acquires first wait for a dispatch queued after it.
///   Nothing pending: memory decides (class A); equal and mask acquires first wait for queued
///     dispatches (a word that only advances can't be moved back by them).
[[nodiscard]] AcquireDecision DecideAcquire(SemaphoreOp op, u32 value, const HazardResult& word,
                                            u32 forwarded, u32 memory_word,
                                            u64 pending_dispatch);

/// EDEN_NX2_V2_ACQUIRE=1: whether an acquire the view doesn't pass suspends the channel without a
/// drain. `old_path_only`: the view can't read the word at all (settings it doesn't model, an
/// unmapped word): the old path decides at once, as it reads memory as the guest would.
[[nodiscard]] constexpr bool ViewSuspends(AcquireVerdict verdict, bool old_path_only) {
    return verdict == AcquireVerdict::NotMet ||
           (verdict == AcquireVerdict::Fallback && !old_path_only);
}

/// What a poll of a channel suspended by the view does while the acquire stays unmet.
enum class SuspendedPoll : u8 {
    Wait,      ///< Its wake classes bring the next check (consumed, landing, release, CPU, map)
    ReadBack,  ///< Read the GPU-written word back (class C)
    Request,   ///< File one submit request for the producer's fence tick (rule W)
};

/// SYNC_SUBMIT_SPEC §2.1/§2.2 (MP §6: class D "land on a host fence", "an interrupt forces a
/// submit if the sync is not yet submitted"; SESSION_HANDOFF §5: a wait for a landing causes its
/// submit, an idle back is woken): what a suspended poll does besides waiting for its wake
/// classes. Only the at-fence row (`request`: a producer that lands at a fence operation) asks
/// for anything:
/// - its fence is queued (`tick` != 0): one request per tick change (no timed re-ask);
/// - no fence carries it yet: wait (the back signals the fence that carries it; the open tick
///   holding it is submitted by rule I at the back's batch end or by the blocked fencing
///   thread's rule W, §7.3 fixes 1-4).
/// No forced landing (the back never blocks on the GPU for an acquire, §2.7), no drain.
[[nodiscard]] constexpr SuspendedPoll PlanSuspendedPoll(bool gpu_written, bool request, u64 tick,
                                                        u64 requested_tick) {
    if (gpu_written) {
        return SuspendedPoll::ReadBack;
    }
    if (!request || tick == 0) {
        return SuspendedPoll::Wait;
    }
    return tick != requested_tick ? SuspendedPoll::Request : SuspendedPoll::Wait;
}

} // namespace VideoCommon::V2
