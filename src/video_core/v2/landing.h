// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"
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
/// Writes the front never notes (QMD releases) that land at an operation (F3F5b-5): the newest
/// entry that queued one, and the newest whose operation ran. The front can't tell their words,
/// but while one is queued a suspended acquire's release may be it: the view asks for the submit
/// and lets the latch resume it rather than draining for it (GpuSplit::UntrackedReleasePending).
inline std::atomic<u64> untracked_issued{0};
inline std::atomic<u64> untracked_landed{0};

inline void RaiseTo(std::atomic<u64>& value, u64 position) {
    u64 current = value.load(std::memory_order_relaxed);
    while (current < position &&
           !value.compare_exchange_weak(current, position, std::memory_order_release,
                                        std::memory_order_relaxed)) {
    }
}

/// Back: an untracked write about to land at an operation: its entry (0 outside an Untracked
/// scope, or when the front doesn't record).
inline u64 CaptureUntracked() {
    if (untracked_tag == 0) {
        return 0;
    }
    RaiseTo(untracked_issued, untracked_tag);
    return untracked_tag;
}

/// Its operation ran (operations run in order).
inline void NoteUntracked(u64 position) {
    RaiseTo(untracked_landed, position);
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
