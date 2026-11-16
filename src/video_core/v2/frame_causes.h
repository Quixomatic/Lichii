// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>

#include "common/common_types.h"
#include "common/dkb_frame_trace.h"

// SYNC_SUBMIT_SPEC §12 R5: name every long frame. With EDEN_DKB_FRAME_TRACE=1 the latency
// paths a frame can stall on note their worst case since the last present, and the present
// logs the frame's values when it took longer than 80 ms ("DKB long frame"), plus the window
// maxima on the once-a-second DKB renderer line. Evidence only: no behaviour change.
namespace VideoCommon::V2::FrameCauses {

enum class Cause : u8 {
    OpenTickAge,      ///< the open tick's age at its submit (Scheduler::NoteOpenTickSubmitted)
    LandingLatency,   ///< a deferred QMD reduction's capture-to-landing time
    Finish,           ///< Scheduler::Finish time (summed: the back's GPU waits)
    ListWriterWait,   ///< the front's wait for a command list's writers (WaitForListWriters)
    SuspendedAcquire, ///< the longest suspended acquire, resumed this frame
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(Cause::Count)> names{
    "open_tick_age_ms", "landing_ms", "finish_ms", "list_writer_wait_ms", "suspended_acquire_ms",
};
/// V2::Stats::WakeClass names, for the longest suspended acquire.
inline constexpr std::array<const char*, 7> wake_names{
    "release_parsed", "consumed", "landing", "map", "readback", "cpu", "none",
};

[[nodiscard]] inline bool Enabled() {
    return Common::DkbFrameTrace::Enabled();
}

struct Frame {
    std::array<u64, static_cast<size_t>(Cause::Count)> ns{};
    u8 wake = 6; ///< the longest suspended acquire's wake class (6: none)
};

inline std::array<std::atomic<u64>, static_cast<size_t>(Cause::Count)> worst{};
inline std::atomic<u8> worst_wake{6};

/// Any thread: `ns` of `cause` (the largest of the frame is kept; Finish adds up).
inline void Note(Cause cause, u64 ns) {
    auto& slot = worst[static_cast<size_t>(cause)];
    if (cause == Cause::Finish) {
        slot.fetch_add(ns, std::memory_order_relaxed);
        return;
    }
    u64 current = slot.load(std::memory_order_relaxed);
    while (current < ns && !slot.compare_exchange_weak(current, ns, std::memory_order_relaxed)) {
    }
}

/// Front: a suspended acquire resumed after `ns`, last waiting for `wake` (V2::Stats::WakeClass).
inline void NoteAcquire(u64 ns, u8 wake) {
    auto& slot = worst[static_cast<size_t>(Cause::SuspendedAcquire)];
    u64 current = slot.load(std::memory_order_relaxed);
    while (current < ns) {
        if (slot.compare_exchange_weak(current, ns, std::memory_order_relaxed)) {
            worst_wake.store(wake, std::memory_order_relaxed);
            return;
        }
    }
}

/// The present: this frame's values, reset for the next.
[[nodiscard]] inline Frame Take() {
    Frame frame;
    for (size_t i = 0; i < frame.ns.size(); ++i) {
        frame.ns[i] = worst[i].exchange(0, std::memory_order_relaxed);
    }
    frame.wake = worst_wake.exchange(6, std::memory_order_relaxed);
    return frame;
}

/// The largest of each over several frames (the window line).
inline void Fold(Frame& into, const Frame& frame) {
    for (size_t i = 0; i < frame.ns.size(); ++i) {
        if (frame.ns[i] > into.ns[i]) {
            into.ns[i] = frame.ns[i];
            if (i == static_cast<size_t>(Cause::SuspendedAcquire)) {
                into.wake = frame.wake;
            }
        }
    }
}

} // namespace VideoCommon::V2::FrameCauses
