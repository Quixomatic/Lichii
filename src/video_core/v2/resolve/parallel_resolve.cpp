// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <string>
#include <thread>

#include <fmt/format.h>

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/v2/resolve/parallel_resolve.h"

namespace VideoCommon::V2::Resolve {

Stats& GetStats() {
    static Stats stats;
    return stats;
}

u32 ShardsFromEnv() {
    const char* const value = std::getenv("EDEN_NX2_V2_PARALLEL_SHARDS");
    if (value != nullptr) {
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed > 0) {
            return static_cast<u32>(std::min<long>(parsed, Stats::MaxShards));
        }
    }
    const u32 hardware = std::thread::hardware_concurrency();
    const u32 free = hardware > ReservedThreads ? hardware - ReservedThreads : 1;
    return std::clamp<u32>(free, 1, 4);
}

SegmentPolicy PolicyFromEnv() {
    return SegmentPolicy::Parse(std::getenv("EDEN_NX2_V2_PARALLEL_SEGMENT"));
}

u32 DepthFromEnv() {
    const char* const value = std::getenv("EDEN_NX2_V2_PARALLEL_DEPTH");
    long parsed = value != nullptr ? std::strtol(value, nullptr, 10) : 8;
    parsed = std::clamp<long>(parsed <= 0 ? 8 : parsed, 2, 64);
    return std::bit_ceil(static_cast<u32>(parsed));
}

bool ShouldLogMismatch(MismatchLog kind) {
    static std::array<std::atomic<u32>, static_cast<size_t>(MismatchLog::Count)> logged{};
    return logged[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed) < 32;
}

void LogSegmentMismatch(u64 segment, size_t record, const char* field) {
    LOG_WARNING(HW_GPU,
                "V2 parallel verify: segment {} record {}: the shard's resolve differs from the "
                "serial resolve in {}",
                segment, record, field);
}

void NameThread(u32 index) {
    Common::SetCurrentThreadName(fmt::format("V2Resolve{}", index).c_str());
}

namespace {

u64 Take(std::atomic<u64>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}

} // Anonymous namespace

void ReportStats(const char* thread_name, u64 window_ns) {
    Stats& stats = GetStats();
    const double seconds = window_ns > 0 ? static_cast<double>(window_ns) / 1e9 : 1.0;
    const u64 segments = Take(stats.segments);
    const u64 by_shard = Take(stats.by_shard);
    const u64 records = Take(stats.records);
    std::string serial;
    u64 serial_total = 0;
    for (size_t reason = 1; reason < NumSerialReasons; ++reason) {
        const u64 count = Take(stats.serial[reason]);
        serial_total += count;
        if (count != 0) {
            serial += fmt::format("{}{} {:.1f}%", serial.empty() ? "" : ", ",
                                  serial_reason_names[reason],
                                  100.0 * static_cast<double>(count) /
                                      static_cast<double>(segments != 0 ? segments : 1));
        }
    }
    std::string cuts;
    for (size_t reason = 0; reason < NumCutReasons; ++reason) {
        const u64 count = Take(stats.cuts[reason]);
        if (count != 0) {
            cuts += fmt::format("{}{} {:.0f}/s", cuts.empty() ? "" : ", ", cut_reason_names[reason],
                                static_cast<double>(count) / seconds);
        }
    }
    const u32 shards = stats.shards.load(std::memory_order_relaxed);
    std::string shard_line;
    u64 busy_total = 0;
    for (u32 shard = 0; shard < shards && shard < Stats::MaxShards; ++shard) {
        const u64 busy = Take(stats.shard_busy_ns[shard]);
        const u64 count = Take(stats.shard_segments[shard]);
        busy_total += busy;
        shard_line += fmt::format("{}{}: {:.1f}% {:.0f}/s", shard_line.empty() ? "" : ", ", shard,
                                  100.0 * static_cast<double>(busy) / static_cast<double>(window_ns),
                                  static_cast<double>(count) / seconds);
    }
    const u64 commit_ns = Take(stats.commit_ns);
    const u64 wait_ns = Take(stats.merge_wait_ns);
    const u64 waits = Take(stats.merge_waits);
    const u64 stolen_ns = Take(stats.stolen_resolve_ns);
    const u64 copied = Take(stats.snapshot_blocks_copied);
    const u64 shared = Take(stats.snapshot_blocks_shared);
    const u64 pre = Take(stats.draws_pre);
    const u64 unmatched = Take(stats.draws_unmatched);
    const u64 drift = Take(stats.draws_drift);
    const u64 no_shape = Take(stats.draws_no_shape);
    std::string steps;
    for (size_t step = 0; step < NumSteps; ++step) {
        const u64 shard = Take(stats.step_shard[step]);
        const u64 commit = Take(stats.step_commit[step]);
        steps += fmt::format("{}{} {:.1f}% of {:.0f}/s", steps.empty() ? "" : ", ",
                             step_names[step],
                             shard + commit != 0 ? 100.0 * static_cast<double>(shard) /
                                                       static_cast<double>(shard + commit)
                                                 : 0.0,
                             static_cast<double>(shard + commit) / seconds);
    }
    const u64 commit_draws = Take(stats.commit_draws);
    const u64 commit_draw_ns = Take(stats.commit_draw_ns);
    const u64 front_draws = Take(stats.front_draws);
    const u64 predicated_off = Take(stats.draws_predicated_off);
    const u64 misfit = Take(stats.draws_misfit);
    // F1-36 (R2-F6-03): draws per segment in the segmenter's unit, the front's draws (records
    // and ordinals, predicated-off ones included); the commit's draws and the predicated-off
    // ones separately. The share of draws a shard pre-resolved is of the commit's draws.
    const double draws_each =
        segments != 0 ? static_cast<double>(front_draws) / static_cast<double>(segments) : 0.0;
    const double commit_draws_each =
        segments != 0 ? static_cast<double>(commit_draws) / static_cast<double>(segments) : 0.0;
    const double draws_by_shards =
        commit_draws != 0 ? 100.0 * static_cast<double>(pre) / static_cast<double>(commit_draws)
                          : 0.0;
    LOG_INFO(
        HW_GPU,
        "V2 stats {} P6 parallel resolve | segments {:.0f}/s ({:.1f} front draws each, {:.1f} "
        "commit draws, {:.1f} records; predicated off {:.0f}/s; draws by shards {:.1f}%), by "
        "shards "
        "{:.1f}%, serial {:.1f}% ({}) | cuts: {} | size {} | shards {} ({:.1f}% busy overall): {} | "
        "commit {:.2f} ms/s, merge wait {:.2f} ms/s ({} waits), serial resolve on commit {:.2f} "
        "ms/s | refused {}, stolen before submit {}, while queued {}, early submits {} (streamed "
        "{}, stream end wait {:.2f} ms/s, draws not ready {:.0f}/s) | front snapshot {:.2f} ms/s, "
        "blocks copied {:.0f}/s, shared {:.1f}% | pre-resolved draws "
        "{:.0f}/s, unmatched {:.0f}/s, misfit {:.0f}/s, drift {}, no shape {:.0f}/s | verify: {} "
        "segments, {} "
        "mismatches ({} dependent); {} draws, {} mismatches; {} snapshots, {} mismatches | determinism: {} "
        "in order, {} out of order | steps on shards: {} | commit per draw {:.2f} us ({:.0f} "
        "draws/s)",
        thread_name, static_cast<double>(segments) / seconds, draws_each, commit_draws_each,
        segments != 0 ? static_cast<double>(records) / static_cast<double>(segments) : 0.0,
        static_cast<double>(predicated_off) / seconds, draws_by_shards,
        segments != 0 ? 100.0 * static_cast<double>(by_shard) / static_cast<double>(segments) : 0.0,
        segments != 0 ? 100.0 * static_cast<double>(serial_total) / static_cast<double>(segments)
                      : 0.0,
        serial.empty() ? "none" : serial, cuts.empty() ? "none" : cuts,
        stats.segment_size.load(std::memory_order_relaxed), shards,
        shards != 0 ? 100.0 * static_cast<double>(busy_total) /
                          (static_cast<double>(window_ns) * shards)
                    : 0.0,
        shard_line, static_cast<double>(commit_ns) / 1e6 / seconds,
        static_cast<double>(wait_ns) / 1e6 / seconds, waits,
        static_cast<double>(stolen_ns) / 1e6 / seconds, Take(stats.refused),
        Take(stats.stolen_unsubmitted), Take(stats.stolen_queued), Take(stats.early_submits),
        Take(stats.early_streamed), static_cast<double>(Take(stats.stream_finish_ns)) / 1e6 / seconds,
        static_cast<double>(Take(stats.draws_not_ready)) / seconds,
        static_cast<double>(Take(stats.front_snapshot_ns)) / 1e6 / seconds, static_cast<double>(copied) / seconds,
        copied + shared != 0 ? 100.0 * static_cast<double>(shared) /
                                   static_cast<double>(copied + shared)
                             : 0.0,
        static_cast<double>(pre) / seconds, static_cast<double>(unmatched) / seconds,
        static_cast<double>(misfit) / seconds, drift,
        static_cast<double>(no_shape) / seconds, Take(stats.verify_segments),
        Take(stats.verify_mismatches), Take(stats.verify_dependent), Take(stats.verify_draws),
        Take(stats.verify_draw_mismatches), Take(stats.snapshot_checks),
        Take(stats.snapshot_mismatches), Take(stats.determinism_checks),
        stats.determinism_failures.load(std::memory_order_relaxed), steps,
        commit_draws != 0
            ? static_cast<double>(commit_draw_ns) / 1e3 / static_cast<double>(commit_draws)
            : 0.0,
        static_cast<double>(commit_draws) / seconds);
}

} // namespace VideoCommon::V2::Resolve
