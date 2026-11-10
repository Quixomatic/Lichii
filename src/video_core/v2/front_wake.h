// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

#include "common/common_types.h"

// Fix F3f (W-01b): while every channel is suspended and no command is queued, the GPU front
// thread sleeps in one-timer-tick steps (it also waits for semaphores the guest CPU releases,
// which nothing announces). Events that can let a channel go on ring it awake at once instead
// of after the tick: a release latching a waiter, the back moving `consumed`, a word readback
// landing, a new command. Only EDEN_NX2_V2_ACQUIRE's sleep listens (v1 keeps its sleep).
namespace VideoCommon::V2::FrontWake {

namespace Detail {
inline std::atomic<bool> sleeping{false};
/// SYNC_SUBMIT_SPEC Step 5: rings so far (a front that saw no new ring since its last check of
/// the suspended channels may sleep until the next one).
inline std::atomic<u64> rings{0};
/// Front: a suspended acquire waits on a guest CPU write (no writer pending), which nothing rings.
inline bool tick_waiter = false;
void Wake();
} // namespace Detail

/// The front thread, once at its start: the thread Ring wakes.
void RegisterFrontThread();

/// Front: sleeps for at most one timer tick (50 us elsewhere); a Ring ends it early.
void Sleep();

/// Front: the ring count, taken before it checks the suspended channels.
[[nodiscard]] inline u64 Epoch() {
    return Detail::rings.load(std::memory_order_seq_cst);
}

/// How a front sleep ended (the STATS line).
enum class SleepEnd : u8 {
    Ring,    ///< An event rang it (or one had rung since `seen`)
    Tick,    ///< One timer tick passed (a guest CPU waiter polls memory)
    Timeout, ///< SleepUntilRing's liveness bound passed with no ring (expected ~0)
};

/// Front (Step 5): with no guest CPU waiter, sleeps until a Ring after `seen` (Epoch), bounded
/// by a liveness backstop (a missing ring source would show as Timeout, never as a hang).
SleepEnd SleepUntilRing(u64 seen);

/// Front: the suspended channels' polls of this pass found a guest CPU waiter (re-armed per pass).
inline void NoteTickWaiter() {
    Detail::tick_waiter = true;
}
[[nodiscard]] inline bool TakeTickWaiter() {
    const bool waiter = Detail::tick_waiter;
    Detail::tick_waiter = false;
    return waiter;
}

/// Any thread: ends a Sleep in progress (nothing when the front doesn't sleep).
inline void Ring() {
    Detail::rings.fetch_add(1, std::memory_order_seq_cst);
    if (Detail::sleeping.load(std::memory_order_seq_cst) &&
        Detail::sleeping.exchange(false, std::memory_order_acq_rel)) [[unlikely]] {
        Detail::Wake();
    }
}

} // namespace VideoCommon::V2::FrontWake
