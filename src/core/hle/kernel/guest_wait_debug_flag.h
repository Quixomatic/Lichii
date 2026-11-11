// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>

// SYNC_SUBMIT_SPEC §14.8 H-14.8: whether guest waits are registered for the GPU front's 2 s
// waiter dump (Kernel::GuestWaitDebug). Set by the split while its acquire view runs
// (EDEN_NX2_V2_ACQUIRE) or STATS is on; otherwise each blocking SVC pays one relaxed load and
// takes no lock (v1 unchanged).
namespace Kernel::GuestWaitDebug {

inline std::atomic<bool> enabled{false};

} // namespace Kernel::GuestWaitDebug
