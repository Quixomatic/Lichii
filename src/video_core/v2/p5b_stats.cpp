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
             "{:.0f}, other {:.0f}, pass {:.0f}, deferred {:.0f}, lazy/requested {:.0f} | submit "
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
             PerSecond(at(Counter::SubmitThreadSubmits), seconds),
             Ratio(at(Counter::SubmitLatencyUs), at(Counter::SubmitThreadSubmits)),
             PerSecond(at(Counter::FinishCompletions), seconds),
             Ratio(at(Counter::FinishLatencyUs), at(Counter::FinishCompletions)),
             PerSecond(at(Counter::RecycledContexts), seconds));
    {
        // F4-01: where the submissions came from (Scheduler::Flush/Finish call sites).
        const auto sites = Sites().Take();
        std::string list;
        for (size_t i = 0; i < sites.size() && i < 16; ++i) {
            list += fmt::format("{}{} {:.2f}", i == 0 ? "" : ", ",
                                SubmitSites::Name(sites[i].file, sites[i].line),
                                per_frame(sites[i].count));
        }
        LOG_INFO(HW_GPU, "V2 stats {} P5b submits by site (per frame) | {}", thread_name,
                 list.empty() ? std::string{"none"} : list);
    }
    LOG_INFO(HW_GPU,
             "V2 stats {} P5b passes | {:.1f}/frame ({:.0f}% dynamic), {:.1f} draws each | "
             "barriers per frame: pass-end {:.1f}, targeted {:.1f}, full {:.1f}; pass ends "
             "without a barrier {:.1f}/frame, hazard breaks {:.1f}/frame | uploads hoisted out "
             "of passes {:.0f}/s",
             thread_name, per_frame(passes), 100.0 * Ratio(at(Counter::DynamicPasses), passes),
             Ratio(at(Counter::PassDraws), passes), per_frame(at(Counter::PassEndBarriers)),
             per_frame(at(Counter::TargetedBarriers)), per_frame(at(Counter::FullBarriers)),
             per_frame(at(Counter::SkippedBarriers)), per_frame(at(Counter::HazardBreaks)),
             PerSecond(at(Counter::HoistedUploads), seconds));
    LOG_INFO(HW_GPU,
             "V2 stats {} P5b transfer queue | batches {:.0f}/s, buffer copies {:.0f}/s ({:.0f} "
             "KiB/s), image uploads {:.0f}/s, kept in order {:.0f}/s, acquire joins {:.0f}/s | "
             "readbacks {:.0f}/s (textures {:.0f}, mapped buffers {:.0f}), latency {:.1f} us, "
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
             "{:.0f}/s",
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
             PerSecond(at(Counter::QmdResyncs), seconds));
}

} // namespace VideoCommon::V2::P5bStats
