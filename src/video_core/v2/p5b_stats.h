// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

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
    ReadbacksDirect, ///< F4-04: producer already on the queue: no flush
    ReadbacksRiding, ///< F4-04: rode the open submission without a submit of its own
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
    FlushDeferred,   ///< F4-01: a deferred fence/list flush (DeferFlush) served by the idle hook
    FlushLazy,       ///< F4-01: a lazy fence's or a waiter's submit request served
    FlushPass,       ///< F4-03: a heuristic flush decided at a pass boundary
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
    StagingCreates,     ///< P-01: staging buffers allocated (StagingBufferPool)
    StagingCreateBytes,
    StagingFrees,       ///< staging buffers released to the driver
    BufferCacheDeletes, ///< buffer cache buffers deleted (joins, GC)
    ImageCacheDeletes,  ///< R2-F2-08: texture cache images deleted (GC, aliases, rebuilds)
    WfiBarriers,        ///< F4-X3: wait-for-idle barriers (pending tracked writes folded in)
    WfiDeferred,        ///< F4-X3: wait-for-idles inside a pass, barrier at its end
    WfiPassEnds,        ///< F4-X3: wait-for-idles that ended a pass (stores or XFB in it)
    IdleUrgent,         ///< F4-X2: idle-hook submits for a thread blocking on the open tick
    IdleUrgentCoalesced,///< F4-X2: urgent requests left to the next boundary (GPU busy, back busy)
    FlushRequestChunk,  ///< F4-X2: mid-pass chunk submits for an urgent request
    FlushRequestPass,   ///< F4-X2: pass-boundary submits for a waiter's request
    IdleDeferredWork,   ///< F4-X2: idle-hook submits of DeferFlush's work
    IdleSync,           ///< F4-X2: idle-hook submits by DXVK's synchronization rule
    IdleLeft,           ///< F4-X2: idle hooks that left open work for the next boundary
    ListCopiesRiding,   ///< F4-X2: command list copies left in the open submission
    Clears,             ///< R2-F4-06: RasterizerVulkan::Clear calls that clear something
    ClearsDeferred,     ///< ... that became the next pass's load ops
    ClearsInPass,       ///< ... recorded as vkCmdClearAttachments in a pass
    ClearsBlit,         ///< ... drawn by the blit helper (partial channels or stencil masks)
    ClearPassEnds,      ///< passes a clear ended
    ClearPassBegins,    ///< passes a clear began
    ClearBarriers,      ///< targeted barriers recorded inside a clear
    ClearsKeptPass,     ///< chunk B2: clears recorded in the open pass of another framebuffer key
    ClearPassEndsFull,  ///< passes ended by a full clear of attachments the pass does not hold
    ClearPassEndsPartial, ///< passes ended by a partial or blit clear
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

/// F4-01: submissions counted by the call site Scheduler::Flush/Finish was given (its
/// source_location). Sites are few (tens), so a short list under a lock is enough for STATS.
class SubmitSites {
public:
    struct Site {
        const char* file;
        u32 line;
        u64 count;
    };

    void Note(const char* file, u32 line) {
        std::scoped_lock lock{mutex};
        for (Site& site : sites) {
            if (site.line == line &&
                (site.file == file || std::strcmp(site.file, file) == 0)) {
                ++site.count;
                return;
            }
        }
        sites.push_back({file, line, 1});
    }

    /// The counts so far, most frequent first; resets them.
    [[nodiscard]] std::vector<Site> Take() {
        std::vector<Site> taken;
        {
            std::scoped_lock lock{mutex};
            taken.swap(sites);
        }
        std::ranges::stable_sort(taken, [](const Site& a, const Site& b) {
            return a.count > b.count;
        });
        return taken;
    }

    /// "file.cpp:123" without the directories.
    [[nodiscard]] static std::string Name(const char* file, u32 line) {
        std::string_view name{file};
        if (const auto slash = name.find_last_of("/\\"); slash != std::string_view::npos) {
            name.remove_prefix(slash + 1);
        }
        return std::string{name} + ':' + std::to_string(line);
    }

private:
    std::mutex mutex;
    std::vector<Site> sites;
};

/// Counts one submission at file:line (callers check Enabled() first).
void NoteSubmitSite(const char* file, u32 line);
/// F4-06: one full barrier (an operation the range tracker doesn't follow) asked for at file:line.
void NoteFullBarrierSite(const char* file, u32 line);

/// Logs and resets once per window (any thread; the first to arrive in a new window reports).
void Report(const char* thread_name, u64 window_ns);

} // namespace VideoCommon::V2::P5bStats
