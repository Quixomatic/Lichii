// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <memory>

#include "common/common_types.h"
#include "video_core/v2/hazard.h"

// SYNC_SUBMIT_SPEC §2.1 (MASTER_PLAN §5.1 "owns semaphore releases with known payloads; syncpoint
// increments; QMD releases", §6 classes A/B, DIAGRAMS 2.1 "OWNS guest-visible results"): S1's
// semaphore view. Per semaphore word, the newest writer the front sent that may not be in memory
// yet, with the value the in-order host queue leaves there once it ran, when S1 can tell it. An
// acquire decides from it (DecideFromView), never from when a fence lands. Front thread only.
namespace VideoCommon::V2 {

/// The back's progress as the front reads it once per lookup.
struct SemProgress {
    u64 consumed = 0;          ///< GpuSplit::ConsumedNow
    u64 landed = 0;            ///< Landing::landed
    u64 untracked_settled = 0; ///< Landing::UntrackedSettled(consumed)
};

class SemView {
public:
    static constexpr size_t Slots = 8192;
    static constexpr size_t Probe = 16;

    /// When a writer's bytes are in memory.
    enum class Retire : u8 {
        Consumed,  ///< Written when the back runs it: consumed >= position
        Landing,   ///< At a fence operation with a landing ticket: landed >= ticket
        Untracked, ///< A QMD write (its entry's untracked operations): settled >= position
        All,       ///< Overflow: every one of the above
    };
    enum class Kind : u8 {
        Release,
        Report,
        QmdPayload,
        QmdReduction,
        QmdMax,
        QmdUnknown,
        Counter, ///< A report counter or timestamp (class D)
    };
    struct Writer {
        u64 address = 0; ///< Device address of the word (0: free slot)
        u64 position = 0;
        u64 ticket = 0;
        u32 value = 0;
        u32 prior = 0;          ///< The value of the pending known writer it replaced
        bool known = false;
        bool bound = false;     ///< `value` is a lower bound (for Gequal only)
        bool has_prior = false;
        Retire retire = Retire::Consumed;
        Kind kind = Kind::Release;
    };
    struct Answer {
        bool pending = false;
        bool overflow = false; ///< Not found while evicted writers may still be pending
        Writer writer{};
    };

    SemView() : slots{std::make_unique<std::array<Writer, Slots>>()} {}

    void Clear() {
        slots->fill({});
        unbounded_unknown = 0;
        overflow_position = 0;
        overflow_ticket = 0;
        notes = 0;
    }

    [[nodiscard]] static bool Retired(Retire retire, u64 position, u64 ticket,
                                      const SemProgress& progress) {
        switch (retire) {
        case Retire::Consumed:
            return progress.consumed >= position;
        case Retire::Landing:
            return progress.landed >= ticket;
        case Retire::Untracked:
            return progress.consumed >= position && progress.untracked_settled >= position;
        case Retire::All:
            return progress.consumed >= position && progress.untracked_settled >= position &&
                   progress.landed >= ticket;
        }
        return true;
    }

    /// The newest writer of the word at `address` that may not be in memory.
    [[nodiscard]] Answer Find(u64 address, const SemProgress& progress) const {
        const size_t home = Home(address);
        for (size_t i = 0; i < Probe; ++i) {
            const Writer& slot = (*slots)[(home + i) & (Slots - 1)];
            if (slot.address == address) {
                if (Retired(slot.retire, slot.position, slot.ticket, progress)) {
                    return {};
                }
                return {.pending = true, .overflow = false, .writer = slot};
            }
        }
        if (overflow_position != 0 &&
            !Retired(Retire::All, overflow_position, overflow_ticket, progress)) {
            // A writer of this word may have been evicted: unknown, conservatively the newest.
            Writer writer{};
            writer.address = address;
            writer.position = overflow_position;
            writer.ticket = overflow_ticket;
            writer.retire = Retire::All;
            writer.kind = Kind::QmdUnknown;
            return {.pending = true, .overflow = true, .writer = writer};
        }
        return {};
    }

    /// A write of `value` (when `known`) to the word, sent at ring `position`.
    void Note(u64 address, u32 value, bool known, u64 position, Retire retire, u64 ticket,
              Kind kind, const SemProgress& progress, bool bound = false) {
        ++notes;
        const size_t home = Home(address);
        Writer* free_slot = nullptr;
        Writer* oldest = nullptr;
        for (size_t i = 0; i < Probe; ++i) {
            Writer& slot = (*slots)[(home + i) & (Slots - 1)];
            if (slot.address == address) {
                const bool pending =
                    !Retired(slot.retire, slot.position, slot.ticket, progress);
                const bool has_prior = pending && slot.known && !slot.bound;
                const u32 prior = slot.value;
                Store(slot, address, value, known, position, retire, ticket, kind, bound);
                slot.has_prior = has_prior;
                slot.prior = prior;
                return;
            }
            if (free_slot == nullptr &&
                (slot.address == 0 ||
                 Retired(slot.retire, slot.position, slot.ticket, progress))) {
                free_slot = &slot;
            }
            if (oldest == nullptr || slot.position < oldest->position) {
                oldest = &slot;
            }
        }
        if (free_slot == nullptr) {
            // Every slot of the window holds a pending writer: fold the oldest into the
            // overflow marker (lookups that miss answer unknown until it retires).
            ++evictions;
            overflow_position = std::max(overflow_position, oldest->position);
            if (oldest->retire == Retire::Landing || oldest->retire == Retire::All) {
                overflow_ticket = std::max(overflow_ticket, oldest->ticket);
            }
            free_slot = oldest;
        }
        Store(*free_slot, address, value, known, position, retire, ticket, kind, bound);
    }

    /// A reduction (ReduceSemaphore `op`, or a max release when `max`) of `operand`: its value
    /// over the pending known value, or over memory (`read()`) when nothing is pending, which then
    /// holds every earlier GPU write. `memory_complete`: false when a writer S1 can't see may
    /// still change the word (the result is then only a lower bound, kept for monotonic ops).
    template <typename Reduce, typename ReadMemory>
    void NoteReduction(u64 address, bool max, u32 op, u32 operand, u64 position, Retire retire,
                       u64 ticket, Kind kind, const SemProgress& progress, bool memory_complete,
                       const Reduce& reduce, const ReadMemory& read) {
        const Answer answer = Find(address, progress);
        bool known = true;
        bool bound = false;
        u32 base = 0;
        if (answer.pending) {
            known = answer.writer.known;
            bound = answer.writer.bound;
            base = answer.writer.value;
        } else {
            base = read();
            bound = !memory_complete;
        }
        const u32 value = max ? std::max(base, operand) : reduce(op, base, operand);
        if (known && bound) {
            // A lower bound stays one only through ops that never lower the word.
            const bool monotonic = max || (op == 2) || (op == 0 && value >= base);
            known = monotonic;
        }
        Note(address, value, known, position, retire, ticket, kind, progress, known && bound);
    }

    /// An unknown QMD (a write S1 can't bound) sent at `position`.
    void NoteUnboundedUnknown(u64 position) {
        unbounded_unknown = position;
    }
    /// The newest unbounded unknown writer still queued, or 0.
    [[nodiscard]] u64 UnboundedUnknown(u64 consumed) const {
        return unbounded_unknown > consumed ? unbounded_unknown : 0;
    }
    /// Writes noted so far (the "release parsed" wake epoch).
    [[nodiscard]] u64 Notes() const {
        return notes;
    }
    /// Evictions since the last call (stats).
    [[nodiscard]] u64 TakeEvictions() {
        const u64 taken = evictions;
        evictions = 0;
        return taken;
    }

private:
    static size_t Home(u64 address) {
        return static_cast<size_t>(((address >> 2) * 0x9E3779B97F4A7C15ULL) >> 51) & (Slots - 1);
    }
    static void Store(Writer& slot, u64 address, u32 value, bool known, u64 position,
                      Retire retire, u64 ticket, Kind kind, bool bound) {
        slot.address = address;
        slot.value = known ? value : 0;
        slot.known = known;
        slot.bound = bound;
        slot.position = position;
        slot.retire = retire;
        slot.ticket = ticket;
        slot.kind = kind;
        slot.has_prior = false;
        slot.prior = 0;
    }

    std::unique_ptr<std::array<Writer, Slots>> slots;
    u64 unbounded_unknown = 0;
    u64 overflow_position = 0;
    u64 overflow_ticket = 0;
    u64 notes = 0;
    u64 evictions = 0;
};

/// What can let a suspended acquire go on (SYNC_SUBMIT_SPEC §2.2), as bits.
namespace SemWake {
inline constexpr u8 ReleaseParsed = 1 << 0;
inline constexpr u8 Consumed = 1 << 1;
inline constexpr u8 Landing = 1 << 2;
inline constexpr u8 Map = 1 << 3;
inline constexpr u8 Readback = 1 << 4;
inline constexpr u8 Cpu = 1 << 5;
} // namespace SemWake

/// The §2.1 table's rows (counted per decision).
enum class SemRow : u8 {
    MapQueued,
    Unmapped,
    GpuWritten,
    NoWriter,
    PendingKnown,
    UnknownAtExecute,
    UnknownAtFence,
    Count,
};

struct SemInputs {
    bool mapped = true;
    bool map_queued = false;
    bool gpu_written = false;
    u32 memory_word = 0;
    u64 unbounded_unknown = 0; ///< SemView::UnboundedUnknown (0: none queued)
};

struct SemDecision {
    /// Met, MetByQueued (class B, NoteSyncAt(position)), NotMet (suspend with `wake`), Fallback
    /// (class C: read the GPU-written word back behind its writers).
    AcquireVerdict verdict = AcquireVerdict::NotMet;
    SemRow row = SemRow::NoWriter;
    u8 wake = 0;
    u64 position = 0; ///< MetByQueued: the writer's position; Consumed wake: the target
    u64 ticket = 0;   ///< Landing wake: the writer's ticket (Untracked: its position)
    SemView::Retire retire = SemView::Retire::Consumed;
    bool request = false; ///< Class D / at-fence row: one submit request for the ticket's tick
};

/// SYNC_SUBMIT_SPEC §2.1 decision table.
[[nodiscard]] inline SemDecision DecideFromView(SemaphoreOp op, u32 value,
                                                const SemView::Answer& answer,
                                                const SemInputs& in) {
    const bool reach = SemaphoreWaitsToReach(op);
    SemDecision d{};
    if (!in.mapped) {
        // §5.2 Map records: a queued map may map it; otherwise memory as the guest reads it.
        if (in.map_queued) {
            d.row = SemRow::MapQueued;
            d.wake = SemWake::Map;
            return d;
        }
        d.row = SemRow::Unmapped;
        d.verdict = SemaphoreMet(op, in.memory_word, value) ? AcquireVerdict::Met
                                                            : AcquireVerdict::NotMet;
        d.wake = SemWake::Map | SemWake::Cpu;
        return d;
    }
    if (in.gpu_written) {
        // MP §6 class C: a targeted readback behind its writers.
        d.row = SemRow::GpuWritten;
        d.verdict = AcquireVerdict::Fallback;
        d.wake = SemWake::Readback;
        return d;
    }
    // An unknown QMD sent after the newest writer S1 knows may change the word: Equal and mask
    // acquires wait until the back ran it (Gequal: the §10 monotonic rule needs no wait).
    const u64 newest = answer.pending ? answer.writer.position : 0;
    if (!reach && in.unbounded_unknown > newest) {
        d.row = answer.pending ? SemRow::UnknownAtExecute : SemRow::NoWriter;
        d.wake = SemWake::Consumed;
        d.position = in.unbounded_unknown;
        return d;
    }
    if (!answer.pending) {
        d.row = SemRow::NoWriter;
        if (SemaphoreMet(op, in.memory_word, value)) {
            d.verdict = AcquireVerdict::Met;
            return d;
        }
        d.wake = SemWake::ReleaseParsed | SemWake::Cpu;
        return d;
    }
    const SemView::Writer& w = answer.writer;
    if (w.known && (!w.bound || reach)) {
        d.row = SemRow::PendingKnown;
        if (SemaphoreMet(op, w.value, value)) {
            d.verdict = AcquireVerdict::MetByQueued;
            d.position = w.position;
            d.ticket = w.ticket;
            d.retire = w.retire;
            return d;
        }
        if (!w.bound) {
            d.wake = SemWake::ReleaseParsed;
            return d;
        }
        // A lower bound that doesn't meet it: decided like an unknown writer below.
    }
    const bool at_fence = w.retire != SemView::Retire::Consumed;
    d.row = at_fence ? SemRow::UnknownAtFence : SemRow::UnknownAtExecute;
    if (reach) {
        // MP §10: a word that only advances meets it once memory (or an older known value still
        // queued) does.
        if (SemaphoreMet(op, in.memory_word, value)) {
            d.verdict = AcquireVerdict::Met;
            return d;
        }
        if (w.has_prior && SemaphoreMet(op, w.prior, value)) {
            d.verdict = AcquireVerdict::MetByQueued;
            d.position = w.position;
            d.ticket = w.ticket;
            d.retire = w.retire;
            return d;
        }
    }
    d.position = w.position;
    d.ticket = w.ticket;
    d.retire = w.retire;
    if (at_fence) {
        // MP §6 class D / a write the landing rule deferred: wait for its landing, and file one
        // submit request for its tick (§6 "an interrupt forces a submit").
        d.wake = SemWake::Landing | SemWake::ReleaseParsed;
        d.request = true;
    } else {
        d.wake = SemWake::Consumed | SemWake::ReleaseParsed;
    }
    return d;
}

} // namespace VideoCommon::V2
