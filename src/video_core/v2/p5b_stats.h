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
    PassColorRestarts,  ///< chunk B3: same attachments, pass restarted for the colour count
    ClearsDeferredEnded,///< chunk B3: full clears that ended another pass and became load ops
    // SYNC_SUBMIT_SPEC §2.5 / Step 0: submissions by rule (the rest are counted as "other").
    SubmitsPassBoundary, ///< rule B: the flush heuristic at a pass boundary
    SubmitsChunk,        ///< rule C: the flush heuristic at a chunk
    SubmitsIdle,         ///< rule I: the back's idle hook (DXVK's synchronization rule)
    SubmitsWaiter,       ///< rule W: a thread blocks on the open tick (a request served)
    SubmitsDeferred,     ///< DeferFlush's work (FlushDeferred, idle hook)
    SubmitsPresent,      ///< the present's own submission
    QmdReductionsAtExecute, ///< SYNC Step 3: QMD reduction/max releases run at execute time
    QmdReductionsDeferred,  ///< ... at their fence (delayed fences, the landing rule)
    RequestsAcquire,  ///< SYNC Step 4: submit requests filed by suspended acquires (at-fence row)
    RequestsLanding,  ///< ... filed by front landing waits (WaitLanded/WaitUntrackedLanded)
    DmaReadsSemForwarded, ///< SYNC Step 6: macro DMA reads forwarded from the semaphore view
    DmaReadsSemWaited,    ///< ... after waiting for an unknown at-execute writer's position
    DmaReadsSemClear,     ///< ... nothing pending: memory
    HoistedCopies,        ///< SYNC Step 9: hazard-free buffer copies in the upload command buffer
    QueryWfiAfterPass,    ///< SYNC Step 9: query result copies deferred to the pass end
    SubmitsIdleBatchEnd,  ///< SYNC §7.3: rule I submits from the idle hook at a batch end
    SubmitsIdleNonDraw,   ///< ... rule I for guest-visible work without draws
    FencingRequests,      ///< ... rule W requests from the blocked fencing thread
    OpenTickAgeMaxUs,     ///< ... the oldest open tick (with a lazy fence) at its submit (max)
    ResumePendingPasses,  ///< SYNC §8.2: suspended-channel passes (one per ResumePending call)
    ResumeWithoutProgress,///< ... a poll passed but the channel didn't resume (expected 0)
    IdleUnfencedFences,   ///< SYNC §9.3: idle fences made for operations with no fence after them
    SubmitsSleep,         ///< SYNC §11.2 rule S: the back submits the open tick before it sleeps
    BackSleeps,           ///< ... the back's sleeps (S's trigger frequency, §11.6 gate)
    SleepWakeSubmits,     ///< ... S submits after a completion came between the check and the sleep
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

/// The largest value seen this window (a max counter).
inline void Max(Counter counter, u64 value) {
    if (Enabled()) [[unlikely]] {
        auto& slot = Get().values[static_cast<size_t>(counter)];
        u64 current = slot.load(std::memory_order_relaxed);
        while (current < value &&
               !slot.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
        }
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
        u64 sum; ///< Note's `amount` (a wait's ns), summed
    };

    void Note(const char* file, u32 line, u64 amount = 0) {
        std::scoped_lock lock{mutex};
        for (Site& site : sites) {
            if (site.line == line &&
                (site.file == file || std::strcmp(site.file, file) == 0)) {
                ++site.count;
                site.sum += amount;
                return;
            }
        }
        sites.push_back({file, line, 1, amount});
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
/// SYNC Step 0: one Scheduler::Finish at file:line that waited `ns` for the GPU.
void NoteFinishSite(const char* file, u32 line, u64 ns);

/// Logs and resets once per window (any thread; the first to arrive in a new window reports).
void Report(const char* thread_name, u64 window_ns);

} // namespace VideoCommon::V2::P5bStats
