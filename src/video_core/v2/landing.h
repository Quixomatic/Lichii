// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>

#include "common/common_types.h"
#include "video_core/v2/front_wake.h"
#include "video_core/v2/hazard.h"

// Pipeline v2 P2: when releases reach guest memory. Since #337 the query cache writes a
// fence-flagged release (and every counter report) only when its operation runs on the fencing
// thread, after the split's back ran it. Each such write gets a landing ticket: the ring position
// after the entry that sent it, shifted by LandingSubBits, plus its index among the landing
// writes of that entry (a StateRun can carry several). The front numbers the writes it notes
// (GpuSplit::NoteUnlandedWrite), the back numbers the operations it creates the same way
// (Capture), the operation reports its ticket here once the bytes are in memory. Operations run
// in order on one thread, so `landed` only grows and every ticket at or below it is in memory.
// The back may number fewer writes than the front (a write whose operation is dropped), never
// more: writes the front can't see (QMD releases) take no ticket (Untracked).
//
// MASTER_PLAN §6 class B landing rule (Fix F3-02): a guest-CPU-visible write the back would make
// at once while an earlier ticket is still above `landed` (MustDefer) becomes an operation of
// the same in-order list instead, so the guest CPU never sees it before the earlier release.
namespace VideoCommon::V2::Landing {

inline constexpr u64 SubMask = (u64{1} << LandingSubBits) - 1;

/// The split's back, while the front records writes (GpuSplit::TracksWrites): the ring position
/// after the method or record it runs. 0 elsewhere.
inline thread_local u64 tag = 0;
/// Back: the entry and index of the last ticket taken, and the newest ticket taken.
inline thread_local u64 captured_tag = 0;
inline thread_local u64 captured_sub = 0;
inline thread_local u64 pending = 0;

/// The newest ticket whose write is in memory.
inline std::atomic<u64> landed{0};
/// The newest ticket the back captured (its `pending`, for other threads): while it is above
/// `landed` an operation the back queued is still to land, whatever the front noted for it.
inline std::atomic<u64> issued{0};

/// Landing-order verify (EDEN_NX2_V2_ACQUIRE or _HAZARD = verify): tickets of CPU-visible
/// writes must land in order; one reported below a ticket already landed is counted.
inline bool verify = false;
inline std::atomic<u64> order_violations{0};

[[nodiscard]] constexpr u64 Ticket(u64 position, u64 sub) {
    return (position << LandingSubBits) | sub;
}
[[nodiscard]] constexpr u64 RingPosition(u64 ticket) {
    return ticket >> LandingSubBits;
}

/// Back: the ticket of a landing write about to be made (0 when the front doesn't record).
inline u64 Capture() {
    if (tag == 0) {
        return 0;
    }
    if (tag != captured_tag) {
        captured_tag = tag;
        captured_sub = 0;
    } else if (captured_sub < SubMask) {
        ++captured_sub;
    }
    pending = Ticket(tag, captured_sub);
    issued.store(pending, std::memory_order_release);
    return pending;
}

/// Back: whether a write made at once now would pass an earlier write still to land.
[[nodiscard]] inline bool MustDefer() {
    return tag != 0 && pending > landed.load(std::memory_order_acquire);
}

inline void Note(u64 ticket) {
    u64 current = landed.load(std::memory_order_relaxed);
    if (verify && ticket < current) [[unlikely]] {
        order_violations.fetch_add(1, std::memory_order_relaxed);
    }
    while (current < ticket &&
           !landed.compare_exchange_weak(current, ticket, std::memory_order_release,
                                         std::memory_order_relaxed)) {
    }
}

/// Back: inside an Untracked scope, the ring position its writes would have been tagged with.
inline thread_local u64 untracked_tag = 0;
inline thread_local u64 untracked_captured_tag = 0;
inline thread_local u64 untracked_captured_sub = 0;
/// Writes the front never notes by ticket (QMD releases) that land at a delayed operation
/// (F3F5b-5): the newest untracked ticket queued, and the newest whose operation ran. The front
/// can't tell their words, but while one is queued a suspended acquire's release may be it: the
/// view asks for the submit and lets the latch resume it rather than draining for it
/// (GpuSplit::UntrackedReleasePending). Fix F3f (F3-X6): a ticket per operation (entry end <<
/// UntrackedSubBits | index), so the operations of one entry are told apart: a read the front
/// noted a QMD release for waits until the operations of that entry ran (UntrackedSettled).
inline constexpr u64 UntrackedSubBits = 16;
inline constexpr u64 UntrackedSubMask = (u64{1} << UntrackedSubBits) - 1;
inline std::atomic<u64> untracked_issued{0};
inline std::atomic<u64> untracked_landed{0};

[[nodiscard]] constexpr u64 UntrackedTicket(u64 position, u64 sub) {
    return (position << UntrackedSubBits) | sub;
}
[[nodiscard]] constexpr u64 UntrackedPosition(u64 ticket) {
    return ticket >> UntrackedSubBits;
}

/// Front: every untracked operation of the entries ending at or before the returned ring
/// position ran. `consumed`: the back's consumed position, loaded before this call. Operations
/// run in the order they were captured: one of a later entry ran, so did every earlier one; and
/// once the newest ticket captured ran, so did every operation of the entries consumed.
[[nodiscard]] inline u64 UntrackedSettled(u64 consumed) {
    const u64 newest_issued = untracked_issued.load(std::memory_order_acquire);
    const u64 newest_landed = untracked_landed.load(std::memory_order_acquire);
    const u64 position = UntrackedPosition(newest_landed);
    const u64 before = position != 0 ? position - 1 : 0;
    return newest_landed >= newest_issued ? std::max(consumed, before) : before;
}

inline void RaiseTo(std::atomic<u64>& value, u64 position) {
    u64 current = value.load(std::memory_order_relaxed);
    while (current < position &&
           !value.compare_exchange_weak(current, position, std::memory_order_release,
                                        std::memory_order_relaxed)) {
    }
}

/// The device ranges of the untracked operations captured last (F3-X6), so a front read waits
/// only for the ones that write its words (UntrackedTargetFor). Written by the back only; Fix
/// F3h (F3-X1) reads them from the front too: once the back ran an unknown QMD's entry, its
/// operations' ranges are known, so it no longer counts as writing every word. A slot's ticket
/// is cleared while its range changes (a reader that sees it change answers conservatively).
struct UntrackedWrite {
    std::atomic<u64> address{0};
    std::atomic<u64> size{0};
    std::atomic<u64> ticket{0};
};
inline constexpr size_t UntrackedWriteSlots = 256;
inline std::array<UntrackedWrite, UntrackedWriteSlots> untracked_writes{};
inline size_t untracked_write_head = 0; ///< Back only
/// The newest ticket whose range fell out of the slots (raised before the slot is reused).
inline std::atomic<u64> untracked_evicted{0};

/// The split (re)starts: no untracked operation is queued.
inline void ResetUntracked() {
    untracked_issued.store(0, std::memory_order_relaxed);
    untracked_landed.store(0, std::memory_order_relaxed);
    for (UntrackedWrite& write : untracked_writes) {
        write.ticket.store(0, std::memory_order_relaxed);
        write.address.store(0, std::memory_order_relaxed);
        write.size.store(0, std::memory_order_relaxed);
    }
    untracked_write_head = 0;
    untracked_evicted.store(0, std::memory_order_relaxed);
}

/// Back: an untracked write of [address, address + size) (device) about to land at a delayed
/// operation: its ticket (0 outside an Untracked scope, or when the front doesn't record). Only
/// operations that wait for their fence take one: one run at once is in memory before the back
/// stores `consumed`.
inline u64 CaptureUntracked(u64 address, u64 size) {
    if (untracked_tag == 0) {
        return 0;
    }
    if (untracked_tag != untracked_captured_tag) {
        untracked_captured_tag = untracked_tag;
        untracked_captured_sub = 0;
    } else if (untracked_captured_sub < UntrackedSubMask) {
        ++untracked_captured_sub;
    }
    const u64 ticket = UntrackedTicket(untracked_tag, untracked_captured_sub);
    UntrackedWrite& slot = untracked_writes[untracked_write_head++ % UntrackedWriteSlots];
    RaiseTo(untracked_evicted, slot.ticket.load(std::memory_order_relaxed));
    slot.ticket.store(0, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    slot.address.store(address, std::memory_order_relaxed);
    slot.size.store(size, std::memory_order_relaxed);
    slot.ticket.store(ticket, std::memory_order_release);
    RaiseTo(untracked_issued, ticket);
    return ticket;
}

/// Any thread: the newest untracked ticket still to land whose write overlaps [address, address
/// + size), 0 when none (conservative once a range still to land fell out of the slots, or a
/// slot changed while it was read). The front sees every operation of the entries the back
/// consumed: they were captured before the back stored `consumed`.
[[nodiscard]] inline u64 UntrackedTargetFor(u64 address, u64 size) {
    const u64 newest_landed = untracked_landed.load(std::memory_order_acquire);
    const u64 newest_issued = untracked_issued.load(std::memory_order_acquire);
    if (newest_issued <= newest_landed) {
        return 0;
    }
    u64 target = untracked_evicted.load(std::memory_order_acquire) > newest_landed
                     ? newest_issued
                     : 0;
    for (const UntrackedWrite& write : untracked_writes) {
        const u64 ticket = write.ticket.load(std::memory_order_acquire);
        if (ticket <= newest_landed) {
            continue;
        }
        const u64 begin = write.address.load(std::memory_order_relaxed);
        const u64 length = write.size.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (write.ticket.load(std::memory_order_relaxed) != ticket) {
            return newest_issued; // Rewritten while read
        }
        if (begin < address + size && address < begin + length) {
            target = std::max(target, ticket);
        }
    }
    return target;
}

/// Its operation ran (operations run in order). A suspended acquire may wait for it (Fix F3h,
/// F3-X1: an Equal/Mask acquire waits for its QMD producer's operations): end the front's sleep.
inline void NoteUntracked(u64 ticket) {
    RaiseTo(untracked_landed, ticket);
    FrontWake::Ring();
}

/// Back: writes made inside take no ticket (the front never noted them).
class Untracked {
public:
    Untracked() : saved{tag}, saved_untracked{untracked_tag} {
        untracked_tag = tag;
        tag = 0;
    }
    ~Untracked() {
        tag = saved;
        untracked_tag = saved_untracked;
    }
    Untracked(const Untracked&) = delete;
    Untracked& operator=(const Untracked&) = delete;

private:
    u64 saved;
    u64 saved_untracked;
};

} // namespace VideoCommon::V2::Landing
