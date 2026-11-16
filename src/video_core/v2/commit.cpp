// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/v2_stats.h"

namespace VideoCommon::V2::Commit {

CompareResult Compare(std::span<const Request> predicted, std::span<const LogEntry> actual) {
    CompareResult result{};
    bool dependent = false;
    const auto note = [&](size_t index, CompareResult::Field field) {
        if (dependent) {
            ++result.dependent;
            return;
        }
        ++result.mismatches;
        if (result.first_mismatch == ~size_t{0}) {
            result.first_mismatch = index;
            result.field = field;
        }
    };
    constexpr u32 OrderingEffects =
        EffectCreate | EffectDelete | EffectAlias | EffectRescale | EffectModify;
    const size_t count = (std::min)(predicted.size(), actual.size());
    for (size_t index = 0; index < count; ++index) {
        const Request& want = predicted[index];
        const LogEntry& got = actual[index];
        ++result.compared;
        // A proxy, or no slot at all (a creation's result), matches any slot.
        const bool known = !IsProxy(want.slot) && want.slot != InvalidSlot;
        if (want.kind != got.kind) {
            note(index, CompareResult::Field::Kind);
        } else if (known && want.slot != got.slot) {
            note(index, CompareResult::Field::Slot);
        } else if (known &&
                   (want.effects & PredictedEffects) != (got.effects & PredictedEffects)) {
            note(index, CompareResult::Field::Effects);
        }
        if ((got.effects & OrderingEffects) != 0) {
            dependent = true;
        }
    }
    if (predicted.size() != actual.size()) {
        note(count, CompareResult::Field::Count);
    }
    return result;
}

EpochDomain& Epochs() {
    static EpochDomain domain;
    return domain;
}

void CountEffect(u32 effect) {
    const u32 index = static_cast<u32>(std::countr_zero(effect));
    if (index < NumEffects) {
        Stats::Count(Stats::Get().effects[index]);
    }
}

namespace Stats {

namespace {
Data g_data;

double PerSecond(u64 value, u64 window_ns) {
    return window_ns == 0 ? 0.0 : static_cast<double>(value) * 1e9 / static_cast<double>(window_ns);
}

u64 Take(std::atomic<u64>& counter) {
    return counter.exchange(0, std::memory_order_relaxed);
}
} // Anonymous namespace

Data& Get() {
    return g_data;
}

void Report(const char* thread_name, u64 window_ns) {
    Data& d = g_data;
    const u64 commits = Take(d.commits);
    const u64 commit_ns = Take(d.commit_ns);
    std::string kinds;
    u64 total = 0;
    for (size_t kind = 0; kind < NumKinds; ++kind) {
        const u64 count = Take(d.requests[kind]);
        total += count;
        if (count != 0) {
            kinds += fmt::format(" {} {:.0f},", kind_names[kind], PerSecond(count, window_ns));
        }
    }
    std::string effects;
    for (size_t effect = 0; effect < NumEffects; ++effect) {
        if (const u64 count = Take(d.effects[effect]); count != 0) {
            effects += fmt::format(" {} {:.0f},", effect_names[effect],
                                   PerSecond(count, window_ns));
        }
    }
    std::string external;
    for (size_t kind = 0; kind < NumExternalKinds; ++kind) {
        if (const u64 count = Take(d.external[kind]); count != 0) {
            external += fmt::format(" {} {:.0f},", external_names[kind],
                                    PerSecond(count, window_ns));
        }
    }
    if (commits == 0 && total == 0 && external.empty()) {
        return;
    }
    const auto trim = [](std::string& text) {
        if (!text.empty()) {
            text.pop_back();
        } else {
            text = " none";
        }
    };
    trim(kinds);
    trim(effects);
    trim(external);
    const u64 hits = Take(d.lookup_hits);
    const u64 misses = Take(d.lookup_misses);
    LOG_INFO(HW_GPU,
             "V2 stats {} P5a commit | requests (per s):{} | effects (per s):{} | commits {:.0f}/s, "
             "{:.2f} us each | lock-free lookups {:.0f}/s, hit {:.1f}% | proxies patched {:.0f}/s "
             "| external ops (per s):{}; before an owner {}, floods {}, owner calls {:.0f}/s, "
             "fence batches to the owner {:.0f}/s | hoistable uploads {:.0f}/s | writebacks "
             "queued {:.0f}/s, flushed {:.0f}/s, dropped {:.0f}/s | replay render targets: no "
             "request {:.0f}/s, requests {:.0f}/s | verify {} checks, {} mismatches, {} dependent "
             "(compare the request rates with the P0 'cache mutations' line)",
             thread_name, kinds, effects, PerSecond(commits, window_ns),
             commits == 0 ? 0.0 : static_cast<double>(commit_ns) / 1e3 / static_cast<double>(commits),
             PerSecond(hits + misses, window_ns),
             hits + misses == 0 ? 0.0
                                : static_cast<double>(hits) * 100.0 /
                                      static_cast<double>(hits + misses),
             PerSecond(Take(d.proxies_patched), window_ns), external, Take(d.external_direct),
             Take(d.floods), PerSecond(Take(d.owner_calls), window_ns),
             PerSecond(Take(d.fence_backlog), window_ns), PerSecond(Take(d.hoistable), window_ns),
             PerSecond(Take(d.writeback_queued), window_ns),
             PerSecond(Take(d.writeback_flushed), window_ns),
             PerSecond(Take(d.writeback_dropped), window_ns),
             PerSecond(Take(d.rt_skipped), window_ns), PerSecond(Take(d.rt_requests), window_ns),
             Take(d.verify_draws), Take(d.verify_mismatches), Take(d.verify_dependent));
}

} // namespace Stats

} // namespace VideoCommon::V2::Commit
