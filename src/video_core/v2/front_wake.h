// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

// Fix F3f (W-01b): while every channel is suspended and no command is queued, the GPU front
// thread sleeps in one-timer-tick steps (it also waits for semaphores the guest CPU releases,
// which nothing announces). Events that can let a channel go on ring it awake at once instead
// of after the tick: a release latching a waiter, the back moving `consumed`, a word readback
// landing, a new command. Only EDEN_NX2_V2_ACQUIRE's sleep listens (v1 keeps its sleep).
namespace VideoCommon::V2::FrontWake {

namespace Detail {
inline std::atomic<bool> sleeping{false};
void Wake();
} // namespace Detail

/// The front thread, once at its start: the thread Ring wakes.
void RegisterFrontThread();

/// Front: sleeps for at most one timer tick (50 us elsewhere); a Ring ends it early.
void Sleep();

/// Any thread: ends a Sleep in progress (nothing when the front doesn't sleep).
inline void Ring() {
    if (Detail::sleeping.load(std::memory_order_acquire) &&
        Detail::sleeping.exchange(false, std::memory_order_acq_rel)) [[unlikely]] {
        Detail::Wake();
    }
}

} // namespace VideoCommon::V2::FrontWake
