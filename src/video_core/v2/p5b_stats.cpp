// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <chrono>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/v2_switches.h"

namespace VideoCommon::V2::P5bStats {
namespace {
std::atomic<u64> g_last_report_ns{};

u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

double PerSecond(u64 value, double seconds) {
    return seconds > 0.0 ? static_cast<double>(value) / seconds : 0.0;
}

double Ratio(u64 value, u64 total) {
    return total != 0 ? static_cast<double>(value) / static_cast<double>(total) : 0.0;
}
} // Anonymous namespace

Data& Get() {
    static Data data;
    return data;
}

bool Enabled() {
    return On(Switch::Stats);
}

namespace {
SubmitSites& Sites() {
    static SubmitSites sites;
    return sites;
}
} // Anonymous namespace

void NoteSubmitSite(const char* file, u32 line) {
    Sites().Note(file, line);
}

namespace {
SubmitSites& BarrierSites() {
    static SubmitSites sites;
    return sites;
}
} // Anonymous namespace

void NoteFullBarrierSite(const char* file, u32 line) {
    BarrierSites().Note(file, line);
}

namespace {
SubmitSites& FinishSites() {
    static SubmitSites sites;
    return sites;
}
} // Anonymous namespace

void NoteFinishSite(const char* file, u32 line, u64 ns) {
    FinishSites().Note(file, line, ns);
}

namespace {
/// SYNC §14.7 (3): guest CPU blocking waits by what they wait on (event name, object type).
class GuestWaitTable {
public:
    struct Entry {
        std::string name;
        u64 count;
        u64 ns;
    };
    void Note(std::string_view name, u64 ns) {
        std::scoped_lock lock{mutex};
        for (Entry& entry : entries) {
            if (entry.name == name) {
                ++entry.count;
                entry.ns += ns;
                return;
            }
        }
        if (entries.size() < 64) {
            entries.push_back({std::string{name}, 1, ns});
        }
    }
    [[nodiscard]] std::vector<Entry> Take() {
        std::vector<Entry> taken;
        {
            std::scoped_lock lock{mutex};
            taken.swap(entries);
        }
        std::ranges::stable_sort(taken, [](const Entry& a, const Entry& b) { return a.ns > b.ns; });
        return taken;
    }

private:
    std::mutex mutex;
    std::vector<Entry> entries;
};
GuestWaitTable& GuestWaitSites() {
    static GuestWaitTable sites;
    return sites;
}
} // Anonymous namespace

void NoteGuestWait(std::string_view name, u64 ns) {
    GuestWaitSites().Note(name, ns);
}

void Report(const char* thread_name, u64 window_ns) {
    if (!Enabled()) {
        return;
    }
    const u64 now = NowNs();
    u64 last = g_last_report_ns.load(std::memory_order_relaxed);
    if (last != 0 && now - last < window_ns / 2) {
        return; // Another thread reported this window
    }
    if (!g_last_report_ns.compare_exchange_strong(last, now)) {
        return;
    }
    if (last == 0) {
        last = now - window_ns;
    }
    const double seconds = static_cast<double>(now - last) / 1e9;
    std::array<u64, static_cast<size_t>(Counter::Count)> v{};
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = Get().values[i].exchange(0, std::memory_order_relaxed);
    }
    const auto at = [&v](Counter counter) { return v[static_cast<size_t>(counter)]; };
    const u64 frames = at(Counter::Frames);
    const u64 submits = at(Counter::Submits);
    const u64 passes = at(Counter::Passes);
    const auto per_frame = [frames](u64 value) { return Ratio(value, frames); };
    LOG_INFO(HW_GPU,
             "V2 stats {} P5b submission | frames {:.1f}/s, submits {:.1f}/frame ({:.0f}/s), "
             "draws per submit {:.1f} | flush reasons (per s): idle {:.0f}, pending {:.0f}, "
             "strong {:.0f}, weak {:.0f}, interval {:.0f}, fast {:.0f}, max draws {:.0f}, cost "
             "{:.0f}, other {:.0f}, pass {:.0f}, deferred {:.0f}, lazy/requested {:.0f} | idle "
             "hook (per s): urgent {:.0f}, urgent coalesced {:.0f}, deferred work {:.0f}, sync "
             "rule {:.0f}, left {:.0f}; requests served at a chunk {:.0f}, at a pass boundary "
             "{:.0f}; list copies riding {:.0f} | submit "
             "thread {:.0f}/s, hand-off {:.1f} us, finish {:.0f}/s ({:.1f} us after submit), "
             "recycled contexts {:.0f}/s",
             thread_name, PerSecond(frames, seconds), per_frame(submits),
             PerSecond(submits, seconds), Ratio(at(Counter::SubmitDraws), submits),
             PerSecond(at(Counter::FlushIdle), seconds),
             PerSecond(at(Counter::FlushPending), seconds),
             PerSecond(at(Counter::FlushStrong), seconds), PerSecond(at(Counter::FlushWeak), seconds),
             PerSecond(at(Counter::FlushInterval), seconds),
             PerSecond(at(Counter::FlushFast), seconds), PerSecond(at(Counter::FlushMax), seconds),
             PerSecond(at(Counter::FlushCost), seconds), PerSecond(at(Counter::FlushOther), seconds),
             PerSecond(at(Counter::FlushPass), seconds),
             PerSecond(at(Counter::FlushDeferred), seconds),
             PerSecond(at(Counter::FlushLazy), seconds),
             PerSecond(at(Counter::IdleUrgent), seconds),
             PerSecond(at(Counter::IdleUrgentCoalesced), seconds),
             PerSecond(at(Counter::IdleDeferredWork), seconds),
             PerSecond(at(Counter::IdleSync), seconds),
             PerSecond(at(Counter::IdleLeft), seconds),
             PerSecond(at(Counter::FlushRequestChunk), seconds),
             PerSecond(at(Counter::FlushRequestPass), seconds),
             PerSecond(at(Counter::ListCopiesRiding), seconds),
             PerSecond(at(Counter::SubmitThreadSubmits), seconds),
             Ratio(at(Counter::SubmitLatencyUs), at(Counter::SubmitThreadSubmits)),
             PerSecond(at(Counter::FinishCompletions), seconds),
             Ratio(at(Counter::FinishLatencyUs), at(Counter::FinishCompletions)),
             PerSecond(at(Counter::RecycledContexts), seconds));
    const auto site_list = [&per_frame](SubmitSites& table) {
        const auto sites = table.Take();
        std::string list;
        for (size_t i = 0; i < sites.size() && i < 16; ++i) {
            list += fmt::format("{}{} {:.2f}", i == 0 ? "" : ", ",
                                SubmitSites::Name(sites[i].file, sites[i].line),
                                per_frame(sites[i].count));
        }
        return list.empty() ? std::string{"none"} : list;
    };
    // F4-01: where the submissions came from (Scheduler::Flush/Finish call sites).
    LOG_INFO(HW_GPU, "V2 stats {} P5b submits by site (per frame) | {}", thread_name,
             site_list(Sites()));
    // SYNC_SUBMIT_SPEC §2.5 (Step 0): submissions by rule; Finish calls by site with their wait.
    {
        u64 by_rule = 0;
        for (const Counter rule : SubmitRuleCounters) {
            by_rule += at(rule);
        }
        const auto finishes = FinishSites().Take();
        std::string finish_list;
        for (size_t i = 0; i < finishes.size() && i < 16; ++i) {
            finish_list += fmt::format("{}{} {:.1f}/s {:.1f} ms/s", i == 0 ? "" : ", ",
                                       SubmitSites::Name(finishes[i].file, finishes[i].line),
                                       PerSecond(finishes[i].count, seconds),
                                       PerSecond(finishes[i].sum, seconds) / 1e6);
        }
        LOG_INFO(HW_GPU,
                 "V2 stats {} SYNC submits by rule (per frame, additive) | B pass boundary "
                 "{:.2f}, strong hint {:.2f}, C chunk {:.2f}, W waiter {:.2f}, S sleep point "
                 "{:.2f}, I′ batch end {:.2f}, present "
                 "{:.2f}, readback {:.2f}, deferred {:.2f}, idle (non-split) {:.2f}, other "
                 "{:.2f} | QMD reductions at execute {:.0f}/s, at their fence {:.0f}/s | "
                 "requests by origin (per s): acquires {:.0f}, landing waits {:.0f}, fencing "
                 "thread {:.0f} | open tick age at submit max {:.1f} ms | resume passes {:.0f}/s, resumed without "
                 "progress {:.0f}/s | idle fences for unfenced operations {:.0f}/s | macro DMA "
                 "reads by the semaphore view (per s): forwarded {:.0f}, waited "
                 "{:.0f}, clear {:.0f} | S: racing completions caught {:.2f}/frame, back sleeps "
                 "{:.1f}/frame | finishes by site: {}",
                 thread_name, per_frame(at(Counter::SubmitsPassBoundary)),
                 per_frame(at(Counter::SubmitsStrong)), per_frame(at(Counter::SubmitsChunk)),
                 per_frame(at(Counter::SubmitsWaiter)), per_frame(at(Counter::SubmitsSleep)),
                 per_frame(at(Counter::SubmitsBatchEnd)),
                 per_frame(at(Counter::SubmitsPresent)), per_frame(at(Counter::SubmitsReadback)),
                 per_frame(at(Counter::SubmitsDeferred)), per_frame(at(Counter::SubmitsIdle)),
                 per_frame(submits > by_rule ? submits - by_rule : 0),
                 PerSecond(at(Counter::QmdReductionsAtExecute), seconds),
                 PerSecond(at(Counter::QmdReductionsDeferred), seconds),
                 PerSecond(at(Counter::RequestsAcquire), seconds),
                 PerSecond(at(Counter::RequestsLanding), seconds),
                 PerSecond(at(Counter::FencingRequests), seconds),
                 static_cast<double>(at(Counter::OpenTickAgeMaxUs)) / 1000.0,
                 PerSecond(at(Counter::ResumePendingPasses), seconds),
                 PerSecond(at(Counter::ResumeWithoutProgress), seconds),
                 PerSecond(at(Counter::IdleUnfencedFences), seconds),
                 PerSecond(at(Counter::DmaReadsSemForwarded), seconds),
                 PerSecond(at(Counter::DmaReadsSemWaited), seconds),
                 PerSecond(at(Counter::DmaReadsSemClear), seconds),
                 per_frame(at(Counter::SleepWakeSubmits)), per_frame(at(Counter::BackSleeps)),
                 finish_list.empty() ? std::string{"none"} : finish_list);
        LOG_INFO(HW_GPU,
                 "V2 stats {} SYNC §12 | CPU-written list fetches (per s): no wait {:.0f}, "
                 "ranged writers {:.0f}, GPU pages (unbounded before the sync) {:.0f}, residual "
                 "(unbounded skipped on CPU pages) {:.0f} | R3 host-waited syncpoint fences "
                 "(rule W requests) {:.0f}/s",
                 thread_name, PerSecond(at(Counter::ListFetchNoWait), seconds),
                 PerSecond(at(Counter::ListFetchRanged), seconds),
                 PerSecond(at(Counter::ListFetchGpuPages), seconds),
                 PerSecond(at(Counter::ListFetchResidual), seconds),
                 PerSecond(at(Counter::HostSyncpointRequests), seconds));
        const auto causes = [&](Counter first) {
            std::string text;
            static constexpr std::array<const char*, 6> names{
                "other", "ranged", "unbounded", "scan cap", "found cap", "overflow marker"};
            for (size_t i = 0; i < names.size(); ++i) {
                text += fmt::format("{}{} {:.0f}", i == 0 ? "" : ", ", names[i],
                                    PerSecond(at(static_cast<Counter>(
                                                  static_cast<u32>(first) + i)),
                                              seconds));
            }
            return text;
        };
        LOG_INFO(HW_GPU,
                 "V2 stats {} SYNC §13 list-fetch waits by cause (per s) | CPU-written: {} | "
                 "GPU-written: {} | hazard table overflows {:.0f}/s, list writer table {:.0f}/s "
                 "| GPU-written fetches (per s): no wait {:.0f}, waited for a ranged writer "
                 "{:.0f}, for an unbounded one {:.0f}",
                 thread_name, causes(Counter::ListWaitCpuOther), causes(Counter::ListWaitGpuOther),
                 PerSecond(at(Counter::HazardOverflows), seconds),
                 PerSecond(at(Counter::WriterTableOverflows), seconds),
                 PerSecond(at(Counter::GpuListNoWait), seconds),
                 PerSecond(at(Counter::GpuListPageProducer), seconds),
                 PerSecond(at(Counter::GpuListUnboundedNewer), seconds));
        LOG_INFO(HW_GPU,
                 "V2 stats {} SYNC §14 | decisions by measured pending (per frame): 0 {:.1f}, "
                 "1 {:.1f}, 2+ {:.1f} | I′ batch-end submits {:.2f}/frame | acquire passes by "
                 "cause (per s, without a tick / at a tick): latch {:.0f} / {:.0f}, parse "
                 "{:.0f} / {:.0f}, memory {:.0f} / {:.0f}",
                 thread_name, per_frame(at(Counter::DecisionsPending0)),
                 per_frame(at(Counter::DecisionsPending1)),
                 per_frame(at(Counter::DecisionsPending2)),
                 per_frame(at(Counter::SubmitsBatchEnd)),
                 PerSecond(at(Counter::ResumeLatchRing), seconds),
                 PerSecond(at(Counter::ResumeLatchTick), seconds),
                 PerSecond(at(Counter::ResumeParseRing), seconds),
                 PerSecond(at(Counter::ResumeParseTick), seconds),
                 PerSecond(at(Counter::ResumeMemoryRing), seconds),
                 PerSecond(at(Counter::ResumeMemoryTick), seconds));
        const u64 handoffs = at(Counter::SubmitThreadSubmits);
        const auto guest_waits = GuestWaitSites().Take();
        std::string guest_list;
        for (size_t i = 0; i < guest_waits.size() && i < 8; ++i) {
            guest_list += fmt::format("{}{} {:.0f}/s {:.0f} ms/s", i == 0 ? "" : ", ",
                                      guest_waits[i].name,
                                      PerSecond(guest_waits[i].count, seconds),
                                      PerSecond(guest_waits[i].ns, seconds) / 1e6);
        }
        LOG_INFO(HW_GPU,
                 "V2 stats {} SYNC §14.7 | strong-hint submits by source (per frame): query "
                 "{:.2f}, buffer download {:.2f}, texture download {:.2f}, both {:.2f} | submit "
                 "hand-off (mean us): pickup {:.0f}, queue lock {:.0f} (max {:.0f}), submit call "
                 "{:.0f}; behind a present (lock > 1 ms) {:.2f}/frame | list writer table notes "
                 "(per s): copy {:.0f}, dispatch {:.0f}, upload {:.0f} | guest CPU waits while "
                 "the front sleeps on pending channels: {}",
                 thread_name, per_frame(at(Counter::StrongFromQuery)),
                 per_frame(at(Counter::StrongFromBuffer)),
                 per_frame(at(Counter::StrongFromTexture)),
                 per_frame(at(Counter::StrongFromBoth)),
                 Ratio(at(Counter::HandoffPickupUs), handoffs),
                 Ratio(at(Counter::HandoffLockUs), handoffs),
                 static_cast<double>(at(Counter::HandoffLockMaxUs)),
                 Ratio(at(Counter::HandoffSubmitUs), handoffs),
                 per_frame(at(Counter::HandoffBehindPresent)),
                 PerSecond(at(Counter::WriterNotesCopy), seconds),
                 PerSecond(at(Counter::WriterNotesDispatch), seconds),
                 PerSecond(at(Counter::WriterNotesUpload), seconds),
                 guest_list.empty() ? std::string{"none"} : guest_list);
        // §14.7 (5): guest-read fence age at submit, p50/p99 as the bucket's upper bound.
        u64 fences = 0;
        for (size_t i = 0; i < GuestFenceAgeBoundsUs.size(); ++i) {
            fences += at(static_cast<Counter>(static_cast<u32>(Counter::GuestFenceAge0) + i));
        }
        const auto percentile = [&](double fraction) -> std::string {
            if (fences == 0) {
                return "-";
            }
            const u64 rank = static_cast<u64>(fraction * static_cast<double>(fences - 1)) + 1;
            u64 seen = 0;
            for (size_t i = 0; i < GuestFenceAgeBoundsUs.size(); ++i) {
                seen += at(static_cast<Counter>(static_cast<u32>(Counter::GuestFenceAge0) + i));
                if (seen >= rank) {
                    return i + 1 < GuestFenceAgeBoundsUs.size()
                               ? fmt::format("< {} us", GuestFenceAgeBoundsUs[i])
                               : fmt::format(">= {} us", GuestFenceAgeBoundsUs[i - 1]);
                }
            }
            return "-";
        };
        LOG_INFO(HW_GPU,
                 "V2 stats {} SYNC §14.7 | guest-read fence age at submit: {:.0f}/s, p50 {}, "
                 "p99 {} | HAZARD=verify command lists compared {:.0f}/s, mismatched {}",
                 thread_name, PerSecond(fences, seconds), percentile(0.5), percentile(0.99),
                 PerSecond(at(Counter::ListVerifyCompared), seconds),
                 at(Counter::ListVerifyMismatched));
    }
    // F4-06: which untracked operations still flush every pending write.
    LOG_INFO(HW_GPU, "V2 stats {} P5b full barriers by site (per frame) | {}", thread_name,
             site_list(BarrierSites()));
    LOG_INFO(HW_GPU,
             "V2 stats {} P5b passes | {:.1f}/frame ({:.0f}% dynamic), {:.1f} draws each | "
             "barriers per frame: pass-end {:.1f}, targeted {:.1f}, full {:.1f}; pass ends "
             "without a barrier {:.1f}/frame, hazard breaks {:.1f}/frame | wait-for-idle: "
             "barriers {:.1f}/frame, in a pass deferred {:.1f}/frame, ended it {:.1f}/frame | "
             "uploads hoisted out of passes {:.0f}/s | clears {:.1f}/frame: deferred {:.1f}, in a "
             "pass {:.1f} (kept the open pass {:.1f}), blit {:.1f}; passes ended by clears {:.1f} "
             "(full clears of other attachments {:.1f}, partial {:.1f}; then deferred as load ops "
             "{:.1f}), begun {:.1f}, barriers {:.1f} | passes restarted for a pipeline's colour "
             "count {:.1f}/frame | SYNC step 9: buffer copies in the upload stream {:.1f}/frame, "
             "query result copies after the pass {:.1f}/frame",
             thread_name, per_frame(passes), 100.0 * Ratio(at(Counter::DynamicPasses), passes),
             Ratio(at(Counter::PassDraws), passes), per_frame(at(Counter::PassEndBarriers)),
             per_frame(at(Counter::TargetedBarriers)), per_frame(at(Counter::FullBarriers)),
             per_frame(at(Counter::SkippedBarriers)), per_frame(at(Counter::HazardBreaks)),
             per_frame(at(Counter::WfiBarriers)), per_frame(at(Counter::WfiDeferred)),
             per_frame(at(Counter::WfiPassEnds)),
             PerSecond(at(Counter::HoistedUploads), seconds), per_frame(at(Counter::Clears)),
             per_frame(at(Counter::ClearsDeferred)), per_frame(at(Counter::ClearsInPass)),
             per_frame(at(Counter::ClearsKeptPass)), per_frame(at(Counter::ClearsBlit)),
             per_frame(at(Counter::ClearPassEnds)), per_frame(at(Counter::ClearPassEndsFull)),
             per_frame(at(Counter::ClearPassEndsPartial)),
             per_frame(at(Counter::ClearsDeferredEnded)), per_frame(at(Counter::ClearPassBegins)),
             per_frame(at(Counter::ClearBarriers)), per_frame(at(Counter::PassColorRestarts)),
             per_frame(at(Counter::HoistedCopies)), per_frame(at(Counter::QueryWfiAfterPass)));
    LOG_INFO(HW_GPU,
             "V2 stats {} P5b transfer queue | batches {:.0f}/s, buffer copies {:.0f}/s ({:.0f} "
             "KiB/s), image uploads {:.0f}/s, kept in order {:.0f}/s, acquire joins {:.0f}/s | "
             "readbacks {:.0f}/s (textures {:.0f}, mapped buffers {:.0f}; direct {:.0f}, riding "
             "{:.0f}), latency {:.1f} us, "
             "fallbacks {:.0f}/s, early downloads in the writer's submission {:.0f}/s",
             thread_name, PerSecond(at(Counter::TransferBatches), seconds),
             PerSecond(at(Counter::TransferCopies), seconds),
             PerSecond(at(Counter::TransferBytes), seconds) / 1024.0,
             PerSecond(at(Counter::TransferImageUploads), seconds),
             PerSecond(at(Counter::TransferFallbacks), seconds),
             PerSecond(at(Counter::AcquireJoins), seconds),
             PerSecond(at(Counter::Readbacks), seconds),
             PerSecond(at(Counter::TextureReadbacks), seconds),
             PerSecond(at(Counter::MappedReadbacks), seconds),
             PerSecond(at(Counter::ReadbacksDirect), seconds),
             PerSecond(at(Counter::ReadbacksRiding), seconds),
             Ratio(at(Counter::ReadbackLatencyUs), at(Counter::ReadbackLanded)),
             PerSecond(at(Counter::ReadbackFallbacks), seconds),
             PerSecond(at(Counter::EarlyDownloads), seconds));
    const u64 gpu_bytes = at(Counter::GpuDecodeBytes);
    const u64 cpu_bytes = at(Counter::CpuDecodeBytes);
    LOG_INFO(HW_GPU,
             "V2 stats {} P5b texture decode | GPU {:.0f} images/s ({:.0f}% of bytes), CPU {:.0f} "
             "images/s, batches {:.0f}/s, GPU fallbacks to CPU {:.0f}/s | frames in flight "
             "{:.2f} at composite, pace waits {:.1f}/s ({:.2f} ms each), retired by present wait "
             "{:.0f}/s | QMD front model: signals {:.0f}/s, exact {:.1f}%, unknown {:.0f}/s, "
             "dispatches {:.0f}/s, bounded writes {:.0f}/s, invalidations {:.0f}/s, resyncs "
             "{:.0f}/s | GPU allocations: staging buffers created {:.1f}/s ({:.0f} KiB/s), freed "
             "{:.1f}/s, buffer cache buffers deleted {:.1f}/s, images deleted {:.1f}/s",
             thread_name, PerSecond(at(Counter::GpuDecodes), seconds),
             100.0 * Ratio(gpu_bytes, gpu_bytes + cpu_bytes),
             PerSecond(at(Counter::CpuDecodes), seconds),
             PerSecond(at(Counter::GpuDecodeBatches), seconds),
             PerSecond(at(Counter::GpuDecodeFallbacks), seconds),
             per_frame(at(Counter::FramesInFlight)), PerSecond(at(Counter::PaceWaits), seconds),
             Ratio(at(Counter::PaceWaitUs), at(Counter::PaceWaits)) / 1000.0,
             PerSecond(at(Counter::PresentWaits), seconds),
             PerSecond(at(Counter::QmdSignals), seconds),
             100.0 * Ratio(at(Counter::QmdExact), at(Counter::QmdSignals)),
             PerSecond(at(Counter::QmdUnknown), seconds),
             PerSecond(at(Counter::QmdDispatches), seconds),
             PerSecond(at(Counter::QmdBoundedWrites), seconds),
             PerSecond(at(Counter::QmdInvalidations), seconds),
             PerSecond(at(Counter::QmdResyncs), seconds),
             PerSecond(at(Counter::StagingCreates), seconds),
             PerSecond(at(Counter::StagingCreateBytes), seconds) / 1024.0,
             PerSecond(at(Counter::StagingFrees), seconds),
             PerSecond(at(Counter::BufferCacheDeletes), seconds),
             PerSecond(at(Counter::ImageCacheDeletes), seconds));
}

} // namespace VideoCommon::V2::P5bStats
