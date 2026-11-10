// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>

#include "common/windows/timer_resolution.h"
#endif

#include "video_core/v2/front_wake.h"

namespace VideoCommon::V2::FrontWake {

#ifdef _WIN32
namespace {
std::atomic<HANDLE> front_thread{nullptr};

void CALLBACK NoOp(ULONG_PTR) {}
} // Anonymous namespace

void RegisterFrontThread() {
    HANDLE handle = nullptr;
    if (DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
                        THREAD_SET_CONTEXT, FALSE, 0)) {
        if (HANDLE old = front_thread.exchange(handle, std::memory_order_acq_rel)) {
            CloseHandle(old);
        }
    }
}

void Sleep() {
    Detail::sleeping.store(true, std::memory_order_seq_cst);
    // Alertable: the APC a Ring queues ends the tick early (or this sleep at once, if it was
    // queued just before).
    Common::Windows::SleepForOneTickAlertable();
    Detail::sleeping.store(false, std::memory_order_relaxed);
}

void Detail::Wake() {
    if (HANDLE handle = front_thread.load(std::memory_order_acquire)) {
        QueueUserAPC(NoOp, handle, 0);
    }
}
#else
void RegisterFrontThread() {}

void Sleep() {
    std::this_thread::sleep_for(std::chrono::microseconds{50});
}

void Detail::Wake() {}
#endif

} // namespace VideoCommon::V2::FrontWake
