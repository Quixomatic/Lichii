// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/v2/draw_cache.h"
#include "video_core/v2/v2_stats.h"

namespace VideoCommon::V2::DrawCache {

void CapturedInputs::SealPages() {
    pages.clear();
    for (const Capture& capture : words) {
        if (capture.size == 0) {
            continue;
        }
        const u64 first = capture.device >> Versions::PageVersions::PageBits;
        const u64 last = (capture.device + capture.size - 1) >> Versions::PageVersions::PageBits;
        for (u64 page = first; page <= last; ++page) {
            pages.push_back(page);
        }
    }
    std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
}

Miss ValidateInputs(CapturedInputs& inputs, Versions::EpochMemo& memo,
                    Versions::PageVersions& versions, bool use_memo, bool* restamped,
                    bool* buffers_changed) {
    constexpr u64 page_size = u64{1} << Versions::PageVersions::PageBits;
    const auto unchanged = [&](const ContentRange& range) {
        return use_memo ? memo.Unchanged(range.device, range.size, inputs.stamp)
                        : versions.Unchanged(range.device, range.size, inputs.stamp);
    };
    // Content first: a written image page is a miss whatever the words say. Buffer pages are not
    // inputs of anything a replay skips (the buffer cache synchronizes every bound range on the
    // replay as on a resolve): they are only reported.
    for (const ContentRange& range : inputs.content) {
        if (range.image) {
            if (!unchanged(range)) {
                return Miss::ImagePages;
            }
        } else if (buffers_changed != nullptr && !*buffers_changed && !unchanged(range)) {
            *buffers_changed = true;
        }
    }
    bool vouched = true;
    for (const u64 page : inputs.pages) {
        // One page each: never memoised (a probe would cost what the check does).
        if (!versions.Unchanged(page * page_size, page_size, inputs.stamp)) {
            vouched = false;
            break;
        }
    }
    if (vouched) {
        return Miss::Count;
    }
    // Written behind: compare the words themselves. The new stamp is taken first, so a write
    // during the compare is newer than it.
    const u64 stamp = versions.Stamp();
    for (const Capture& capture : inputs.words) {
        if (capture.host == nullptr ||
            std::memcmp(capture.host, capture.bytes.data(), capture.size) != 0) {
            return MissFor(capture.what);
        }
    }
    // The content again, without the memo: a write between the first check and the new stamp
    // (or one the memo deferred to the next epoch) would otherwise hide behind the new stamp.
    for (const ContentRange& range : inputs.content) {
        if (range.image && !versions.Unchanged(range.device, range.size, inputs.stamp)) {
            return Miss::ImagePages;
        }
    }
    inputs.stamp = stamp;
    if (restamped != nullptr) {
        *restamped = true;
    }
    return Miss::Count;
}

bool HandlesAlive(const Recipe& recipe, const Versions::Generations& generations) {
    using Versions::Kind;
    for (const RecipeView& view : recipe.views) {
        if (generations.Get(Kind::ImageView, view.id) != view.generation) {
            return false;
        }
    }
    for (const Versions::WeakRef& sampler : recipe.samplers) {
        if (generations.Get(Kind::Sampler, sampler.id) != sampler.generation) {
            return false;
        }
    }
    for (const Versions::WeakRef& target : recipe.targets) {
        if (generations.Get(Kind::ImageView, target.id) != target.generation) {
            return false;
        }
    }
    return true;
}

bool CompareRecipes(const Recipe& hit, const Recipe& fresh, std::string& reasons) {
    const size_t before = reasons.size();
    const auto differ = [&](const char* field) {
        if (!reasons.empty()) {
            reasons += ", ";
        }
        reasons += field;
    };
    if (hit.pipeline != fresh.pipeline) {
        differ("pipeline");
    }
    if (hit.views.size() != fresh.views.size() ||
        !std::equal(hit.views.begin(), hit.views.end(), fresh.views.begin(),
                    [](const RecipeView& lhs, const RecipeView& rhs) {
                        return lhs.index == rhs.index && lhs.id == rhs.id;
                    })) {
        differ("image views");
    }
    if (hit.samplers.size() != fresh.samplers.size() ||
        !std::equal(hit.samplers.begin(), hit.samplers.end(), fresh.samplers.begin(),
                    [](const Versions::WeakRef& lhs, const Versions::WeakRef& rhs) {
                        return lhs.id == rhs.id;
                    })) {
        differ("samplers");
    }
    if (hit.storage != fresh.storage) {
        differ("storage buffers");
    }
    bool targets_equal = true;
    for (size_t i = 0; i < NumRecipeTargets; ++i) {
        targets_equal &= hit.targets[i].id == fresh.targets[i].id;
    }
    if (!targets_equal) {
        differ("render targets");
    }
    if (hit.render_pass != fresh.render_pass) {
        differ("render pass");
    }
    if (hit.push_words != fresh.push_words ||
        !std::equal(hit.push.begin(), hit.push.begin() + hit.push_words, fresh.push.begin())) {
        differ("push data");
    }
    if (hit.heap_words != fresh.heap_words) {
        differ("heap words");
    }
    if (hit.native_key != fresh.native_key) {
        differ("native key");
    }
    return reasons.size() == before;
}

bool CompareBufferRecipes(const BufferRecipe& hit, const BufferRecipe& fresh,
                          std::string& reasons) {
    if (!hit.valid || !fresh.valid) {
        return true;
    }
    std::string differ;
    const auto check = [&](const char* what, const std::vector<BufferBind>& hits,
                           const std::vector<BufferBind>& fresh_binds) {
        for (const BufferBind& bind : fresh_binds) {
            const BufferBind* const kept = BufferRecipe::Find(hits, bind.stage, bind.index);
            if (kept == nullptr || !kept->Contains(bind.device_addr, bind.size)) {
                continue; // the replay takes the sync path for it
            }
            if (kept->buffer_id != bind.buffer_id ||
                (kept->kind == BufferBind::Kind::Cached) != (bind.kind == BufferBind::Kind::Cached)) {
                differ += fmt::format("{}{} {}:{}", differ.empty() ? "" : " ", what, bind.stage,
                                      bind.index);
            }
        }
    };
    if (hit.has_index && fresh.has_index && hit.index.Contains(fresh.index.device_addr,
                                                               fresh.index.size) &&
        hit.index.buffer_id != fresh.index.buffer_id) {
        differ += "index";
    }
    check("vertex", hit.vertex, fresh.vertex);
    check("uniform", hit.uniform, fresh.uniform);
    check("storage", hit.storage, fresh.storage);
    check("texture", hit.texture, fresh.texture);
    if (differ.empty()) {
        return true;
    }
    if (!reasons.empty()) {
        reasons += ", ";
    }
    reasons += "buffers (" + differ + ")";
    return false;
}

Recipe* RecipeTable::Find(u64 key) {
    Entry* const set = &entries[SetOf(key) * Ways];
    for (size_t way = 0; way < Ways; ++way) {
        if (set[way].live && set[way].key == key) {
            set[way].touched = ++clock;
            return &set[way].recipe;
        }
    }
    return nullptr;
}

u32 VerifyN() {
    static const u32 n = [] {
        const char* const value = std::getenv("EDEN_NX2_V2_DRAWCACHE_VERIFY_N");
        if (value == nullptr) {
            return u32{64};
        }
        const long parsed = std::strtol(value, nullptr, 10);
        return parsed <= 0 ? u32{64} : static_cast<u32>(parsed);
    }();
    return n;
}

ThreadData& ThreadStats() {
    thread_local std::unique_ptr<ThreadData> stats;
    if (!stats) {
        stats = std::make_unique<ThreadData>();
    }
    return *stats;
}

namespace {

std::string FormatCounters(const char* name, const char* unit, const Counters& c, size_t occupied,
                           size_t capacity, double seconds) {
    const u64 decided = c.hits + c.continued +
                        [&] {
                            u64 total = 0;
                            for (const u64 miss : c.misses) {
                                total += miss;
                            }
                            return total;
                        }();
    const auto pct = [&](u64 value) {
        return decided == 0 ? 0.0 : 100.0 * static_cast<double>(value) / static_cast<double>(decided);
    };
    std::string misses;
    for (size_t i = 0; i < NumMisses; ++i) {
        if (c.misses[i] != 0) {
            misses += fmt::format(" {} {:.1f}%", miss_names[i], pct(c.misses[i]));
        }
    }
    const auto micros = [](u64 ns, u64 draws) {
        return draws == 0 ? 0.0 : static_cast<double>(ns) / 1e3 / static_cast<double>(draws);
    };
    return fmt::format(
        "{}: {:.0f}/s, continued {:.1f}%, hit {:.1f}%, invalidated {:.1f}% | misses:{} | replay "
        "{:.2f} us vs resolve {:.2f} us per {} | replays sent back to resolve {:.0f}/s, restamped "
        "{:.0f}/s, hits with storage pages written (synchronized) {:.0f}/s | buffer replay: all "
        "{:.0f}/s, partial {:.0f}/s, existing path {:.0f}/s, bindings from the recipe {:.0f}/s, "
        "synced {:.0f}/s, verify stale ranges {} | table {}/{} "
        "({:.1f}%), inserts {:.0f}/s, evictions {:.0f}/s, first sightings (no recipe) {:.0f}/s | "
        "verify {} samples, {} mismatches, {} "
        "deferred by the epoch",
        name, static_cast<double>(decided) / seconds, pct(c.continued), pct(c.hits),
        pct(c.invalidated), misses.empty() ? std::string{" none"} : misses,
        micros(c.replay_ns, c.replay_draws), micros(c.resolve_ns, c.resolve_draws),
        unit, static_cast<double>(c.replays_failed) / seconds,
        static_cast<double>(c.restamped) / seconds,
        static_cast<double>(c.buffers_written) / seconds,
        static_cast<double>(c.buffer_replays) / seconds,
        static_cast<double>(c.buffer_partial) / seconds,
        static_cast<double>(c.buffer_normal) / seconds,
        static_cast<double>(c.buffer_binds_replayed) / seconds,
        static_cast<double>(c.buffer_binds_synced) / seconds, c.buffer_stale, occupied, capacity,
        capacity == 0 ? 0.0 : 100.0 * static_cast<double>(occupied) / static_cast<double>(capacity),
        static_cast<double>(c.inserts) / seconds, static_cast<double>(c.evictions) / seconds,
        static_cast<double>(c.probation) / seconds, c.verify_samples, c.verify_mismatches, c.verify_deferred);
}

} // Anonymous namespace

void Report(const char* thread, Versions::EpochMemo* memo) {
    if (!::VideoCommon::V2::Stats::Enabled()) {
        return;
    }
    ThreadData& stats = ThreadStats();
    using Clock = std::chrono::steady_clock;
    const u64 now = static_cast<u64>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
    if (stats.window_start == 0) {
        stats.window_start = now;
        return;
    }
    if (now - stats.window_start < 10'000'000'000ULL) {
        return;
    }
    const double seconds = static_cast<double>(now - stats.window_start) / 1e9;
    const auto sources = Versions::Pages().TakeCounts();
    const auto epochs = Versions::Epochs().TakeCounts();
    std::string writes;
    for (size_t i = 0; i < Versions::NumSources; ++i) {
        writes += fmt::format(" {} {:.0f}", Versions::source_names[i],
                              static_cast<double>(sources[i]) / seconds);
    }
    std::string epoch_text;
    for (size_t i = 0; i < Versions::NumEpochReasons; ++i) {
        epoch_text += fmt::format(" {} {:.0f}", Versions::epoch_reason_names[i],
                                  static_cast<double>(epochs[i]) / seconds);
    }
    std::string memo_text = "no memo";
    if (memo != nullptr) {
        const auto counters = memo->TakeCounters();
        memo_text = fmt::format(
            "memo {:.0f} queries/s, {:.1f}% answered without a walk",
            static_cast<double>(counters.queries) / seconds,
            counters.queries == 0 ? 0.0
                                  : 100.0 * static_cast<double>(counters.hits) /
                                        static_cast<double>(counters.queries));
    }
    LOG_INFO(HW_GPU,
             "V2 stats {} P4 draw cache | {} | {} | versions: writes/s{} | collect epochs/s{} | "
             "{} | generation bumps {:.0f}/s, watched pages {}",
             thread,
             FormatCounters("draws", "draw", stats.graphics, stats.graphics_occupied,
                            stats.graphics_capacity, seconds),
             FormatCounters("dispatches", "dispatch", stats.compute, stats.compute_occupied,
                            stats.compute_capacity, seconds),
             writes, epoch_text, memo_text,
             static_cast<double>(Versions::Gens().TakeBumps()) / seconds,
             stats.graphics.watched_pages);
    const size_t graphics_occupied = stats.graphics_occupied;
    const size_t graphics_capacity = stats.graphics_capacity;
    const size_t compute_occupied = stats.compute_occupied;
    const size_t compute_capacity = stats.compute_capacity;
    stats = ThreadData{};
    stats.window_start = now;
    stats.graphics_occupied = graphics_occupied;
    stats.graphics_capacity = graphics_capacity;
    stats.compute_occupied = compute_occupied;
    stats.compute_capacity = compute_capacity;
}

void LogMismatch(bool compute, u64 key, const std::string& reasons) {
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
        LOG_WARNING(HW_GPU, "V2 draw cache verify: {} key {:016x}: the full resolve differs in {}",
                    compute ? "dispatch" : "draw", key, reasons);
    }
}

} // namespace VideoCommon::V2::DrawCache
