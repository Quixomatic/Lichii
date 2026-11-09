// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"

// P5b counters (EDEN_NX2_V2_STATS=1). Process-wide relaxed atomics: the recording thread, the
// Vulkan worker, the submit and finish threads, readers and the present thread all count here.
// Logged every 10 s by whichever thread reports first (V2 stats ... P5b lines).
namespace VideoCommon::V2::P5bStats {

enum class Counter : u32 {
    Submits,
    SubmitDraws,     ///< draws + dispatches recorded, for draws per submit
    Frames,
    Passes,          ///< render passes begun (classic + dynamic)
    DynamicPasses,
    PassDraws,       ///< draws inside passes, for pass length
    PassEndBarriers, ///< old all-attachment pass-end barriers recorded
    TargetedBarriers,///< barriers from the range tracker (pass begin, draw/dispatch reads)
    FullBarriers,    ///< tracker flushes at untracked out-of-pass ops
    HazardBreaks,    ///< passes ended because a draw read a pending write
    SkippedBarriers, ///< pass ends that recorded pending writes instead of a barrier
    HoistedUploads,  ///< image uploads moved to the submission's upload stream
    TransferBatches,
    TransferCopies,
    TransferBytes,
    TransferFallbacks, ///< transfer-eligible ops kept in order (used by the open submission...)
    TransferImageUploads,
    AcquireJoins,
    Readbacks,       ///< readback jobs on the transfer queue
    ReadbackLatencyUs,
    ReadbackLanded,
    ReadbackFallbacks,
    TextureReadbacks,
    MappedReadbacks,
    EarlyDownloads,  ///< learned downloads recorded right after their writer
    GpuDecodes,
    GpuDecodeBytes,
    CpuDecodes,
    CpuDecodeBytes,
    GpuDecodeBatches,
    GpuDecodeFallbacks,
    FlushIdle,
    FlushPending,
    FlushStrong,
    FlushWeak,
    FlushInterval,
    FlushFast,
    FlushMax,
    FlushCost,
    FlushOther,
    SubmitThreadSubmits,
    SubmitLatencyUs, ///< worker hand-off → vkQueueSubmit2 returned
    FinishCompletions,
    FinishLatencyUs, ///< submit → finish thread saw completion
    RecycledContexts,
    FramesInFlight,  ///< sum at each composite
    PaceWaits,
    PaceWaitUs,
    PresentWaits,    ///< frames retired through present wait
    QmdSignals,
    QmdExact,
    QmdUnknown,
    QmdDispatches,
    QmdInvalidations,
    QmdResyncs,
    QmdBoundedWrites,
    Count,
};

struct Data {
    std::array<std::atomic<u64>, static_cast<size_t>(Counter::Count)> values{};
};

[[nodiscard]] Data& Get();

/// Cached EDEN_NX2_V2_STATS (one load).
[[nodiscard]] bool Enabled();

inline void Add(Counter counter, u64 amount = 1) {
    if (Enabled()) [[unlikely]] {
        Get().values[static_cast<size_t>(counter)].fetch_add(amount, std::memory_order_relaxed);
    }
}

/// Logs and resets once per window (any thread; the first to arrive in a new window reports).
void Report(const char* thread_name, u64 window_ns);

} // namespace VideoCommon::V2::P5bStats
