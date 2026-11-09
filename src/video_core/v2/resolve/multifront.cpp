// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <deque>
#include <map>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/v2/v2_stats.h"
#include "video_core/v2/v2_switches.h"

namespace VideoCommon::V2::Resolve::MultiFront {

// ---- Grouper ------------------------------------------------------------------------------------

u32 Grouper::IndexOf(u32 channel) {
    const auto [it, inserted] = index_of.try_emplace(channel, static_cast<u32>(channels.size()));
    if (inserted) {
        channels.push_back(channel);
        parent.push_back(it->second);
        work.push_back(0);
    }
    return it->second;
}

u32 Grouper::Find(u32 index) const {
    while (parent[index] != index) {
        parent[index] = parent[parent[index]];
        index = parent[index];
    }
    return index;
}

void Grouper::Join(u32 a, u32 b) {
    const u32 root_a = Find(a);
    const u32 root_b = Find(b);
    if (root_a != root_b) {
        parent[(std::max)(root_a, root_b)] = (std::min)(root_a, root_b);
    }
}

void Grouper::Note(const Event& event) {
    const u32 index = IndexOf(event.channel);
    ++events;
    ++work[index];
    switch (event.kind) {
    case EventKind::Work:
        break;
    case EventKind::Release:
        last_release[event.address] = index;
        break;
    case EventKind::Acquire: {
        const auto it = last_release.find(event.address);
        if (it != last_release.end() && it->second != index && semaphore_edges.size() < 65536) {
            semaphore_edges.emplace_back(it->second, index);
        }
        break;
    }
    case EventKind::MacroUpload:
        last_uploader = index;
        break;
    case EventKind::MacroCall:
        if (last_uploader && *last_uploader != index && macro_edges.size() < 65536) {
            macro_edges.emplace_back(*last_uploader, index);
            last_uploader.reset(); // one dependency per upload burst
        }
        break;
    case EventKind::QmdUse:
        if (qmd_user && Find(*qmd_user) != Find(index)) {
            Join(*qmd_user, index);
            ++joins_qmd;
        }
        qmd_user = index;
        break;
    case EventKind::Count:
        break;
    }
}

Plan Grouper::Decide(u32 max_groups) const {
    Plan plan;
    plan.events = events;
    plan.joins_qmd = joins_qmd;
    u32 active = 0;
    for (const u64 count : work) {
        active += count != 0 ? 1 : 0;
    }
    if (active <= 1 || max_groups <= 1) {
        plan.fallback = active <= 1 ? Fallback::SingleChannel : Fallback::OneGroup;
        for (const u32 channel : channels) {
            plan.group_of[channel] = 0;
        }
        return plan;
    }
    // Groups in order of first appearance of their root, folded onto max_groups engines.
    std::unordered_map<u32, u32> group_of_root;
    for (u32 index = 0; index < channels.size(); ++index) {
        const u32 root = Find(index);
        const auto [it, inserted] =
            group_of_root.try_emplace(root, static_cast<u32>(group_of_root.size()));
        plan.group_of[channels[index]] = it->second % max_groups;
    }
    plan.groups = (std::min)(static_cast<u32>(group_of_root.size()), max_groups);
    const auto group = [&](u32 index) { return plan.group_of[channels[index]]; };
    for (const auto& [from, to] : semaphore_edges) {
        plan.cross_edges += group(from) != group(to) ? 1 : 0;
    }
    for (const auto& [from, to] : macro_edges) {
        plan.macro_edges += group(from) != group(to) ? 1 : 0;
    }
    if (plan.groups <= 1) {
        plan.fallback = Fallback::OneGroup;
    } else if (((plan.cross_edges + plan.macro_edges) << MaxEdgeShift) > events) {
        plan.fallback = Fallback::SemaphoreRate;
    }
    if (plan.fallback != Fallback::None) {
        plan.groups = 1;
        for (auto& [channel, g] : plan.group_of) {
            g = 0;
        }
    }
    return plan;
}

void Grouper::Clear() {
    *this = Grouper{};
}

// ---- Orders -------------------------------------------------------------------------------------

namespace {

/// Acquires no release in the window satisfies were met by memory (a CPU release, or before the
/// window): they never block.
std::vector<bool> MemorySatisfied(std::span<const Event> events) {
    std::vector<bool> result(events.size(), false);
    for (size_t i = 0; i < events.size(); ++i) {
        if (events[i].kind != EventKind::Acquire) {
            continue;
        }
        bool any = false;
        for (const Event& other : events) {
            if (other.kind == EventKind::Release && other.address == events[i].address &&
                Satisfies(events[i].op, other.value, events[i].value)) {
                any = true;
                break;
            }
        }
        result[i] = !any;
    }
    return result;
}

/// Semaphore words as published: per address, every release value by ticket (input index).
using Words = std::unordered_map<u64, std::map<u64, u64>>;

/// What an engine may ask about the other engines (or itself): whether a release to `address`, or
/// a macro upload, below `ticket` is still unprocessed.
struct Pending {
    virtual ~Pending() = default;
    [[nodiscard]] virtual bool Release(u64 address, u64 ticket) const = 0;
    [[nodiscard]] virtual bool Upload(u64 ticket) const = 0;
};

/// One front engine over a set of channels (the single front is the engine over all of them):
/// input order, a channel suspended at an unmet acquire, resumed when it is met. An acquire at
/// ticket T reads the semaphore as of T: once every release below T is published, the newest one
/// decides; if it doesn't satisfy, the first later release that does (with every release below it
/// published) resumes the channel. A macro call runs once every upload below it is published.
class Engine {
public:
    Engine(std::span<const Event> events_, const std::vector<bool>& memory_,
           std::vector<u64> order_)
        : events{events_}, memory{&memory_}, order{std::move(order_)} {}

    /// Processes one event if it can. Returns the processed index, or ~0.
    u64 Step(Words& words, const Pending& pending, u64* depends_on) {
        // Suspended channels first, in suspension order (ResumePending).
        for (auto it = suspended.begin(); it != suspended.end(); ++it) {
            std::deque<u64>& queue = it->second;
            if (Ready(queue.front(), words, pending, depends_on)) {
                const u64 index = queue.front();
                queue.pop_front();
                if (queue.empty()) {
                    suspended.erase(it);
                }
                Publish(index, words);
                return index;
            }
        }
        while (cursor < order.size()) {
            const u64 index = order[cursor];
            const u32 channel = events[index].channel;
            const auto held = std::ranges::find_if(
                suspended, [channel](const auto& entry) { return entry.first == channel; });
            ++cursor;
            if (held != suspended.end()) {
                held->second.push_back(index);
                continue;
            }
            if (!Ready(index, words, pending, depends_on)) {
                suspended.emplace_back(channel, std::deque<u64>{index});
                continue;
            }
            Publish(index, words);
            return index;
        }
        return ~u64{0};
    }

    /// Whether an event of `kind` (to `address` for releases) below `ticket` is unprocessed here.
    [[nodiscard]] bool Holds(EventKind kind, u64 address, u64 ticket) const {
        const auto matches = [&](u64 index) {
            return index < ticket && events[index].kind == kind &&
                   (kind != EventKind::Release || events[index].address == address);
        };
        for (size_t i = cursor; i < order.size() && order[i] < ticket; ++i) {
            if (matches(order[i])) {
                return true;
            }
        }
        for (const auto& [channel, queue] : suspended) {
            if (std::ranges::any_of(queue, matches)) {
                return true;
            }
        }
        return false;
    }

private:
    bool Ready(u64 index, const Words& words, const Pending& pending, u64* depends_on) const {
        const Event& event = events[index];
        *depends_on = ~u64{0};
        if (event.kind == EventKind::MacroCall) {
            return !pending.Upload(index);
        }
        if (event.kind != EventKind::Acquire || (*memory)[index]) {
            return true;
        }
        if (pending.Release(event.address, index)) {
            return false; // the value as of this ticket is not known yet
        }
        const auto found = words.find(event.address);
        if (found == words.end()) {
            return false;
        }
        const std::map<u64, u64>& versions = found->second;
        auto it = versions.lower_bound(index);
        if (it != versions.begin()) {
            const auto newest = std::prev(it);
            if (Satisfies(event.op, newest->second, event.value)) {
                *depends_on = newest->first;
                return true;
            }
        }
        for (; it != versions.end(); ++it) {
            if (pending.Release(event.address, it->first)) {
                return false;
            }
            if (Satisfies(event.op, it->second, event.value)) {
                *depends_on = it->first;
                return true;
            }
        }
        return false;
    }

    void Publish(u64 index, Words& words) const {
        const Event& event = events[index];
        if (event.kind == EventKind::Release) {
            words[event.address][index] = event.value;
        }
    }

    std::span<const Event> events;
    const std::vector<bool>* memory;
    std::vector<u64> order;
    size_t cursor = 0;
    std::vector<std::pair<u32, std::deque<u64>>> suspended;
};

struct EnginesPending final : Pending {
    const std::vector<Engine>* engines = nullptr;
    [[nodiscard]] bool Release(u64 address, u64 ticket) const override {
        return std::ranges::any_of(*engines, [&](const Engine& engine) {
            return engine.Holds(EventKind::Release, address, ticket);
        });
    }
    [[nodiscard]] bool Upload(u64 ticket) const override {
        return std::ranges::any_of(*engines, [&](const Engine& engine) {
            return engine.Holds(EventKind::MacroUpload, 0, ticket);
        });
    }
};

} // Anonymous namespace

std::vector<u64> SingleFrontOrder(std::span<const Event> events) {
    const Plan plan; // one group
    return MultiFrontOrder(events, plan, 0);
}

std::vector<u64> MultiFrontOrder(std::span<const Event> events, const Plan& plan, u64 seed,
                                 std::vector<Processed>* engine_log) {
    const std::vector<bool> memory = MemorySatisfied(events);
    const u32 groups = (std::max)(plan.groups, 1U);
    std::vector<std::vector<u64>> orders(groups);
    for (u64 i = 0; i < events.size(); ++i) {
        orders[plan.GroupOf(events[i].channel) % groups].push_back(i);
    }
    std::vector<Engine> engines;
    engines.reserve(groups);
    for (u32 g = 0; g < groups; ++g) {
        engines.emplace_back(events, memory, std::move(orders[g]));
    }
    // The engines run asynchronously: which one moves next is the seed's choice.
    Words words;
    EnginesPending pending;
    pending.engines = &engines;
    std::vector<std::vector<Processed>> outputs(groups);
    u64 state = seed * 0x9e3779b97f4a7c15ULL + 1;
    size_t remaining = events.size();
    while (remaining != 0) {
        bool moved = false;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const u32 first = static_cast<u32>(state >> 33) % groups;
        for (u32 attempt = 0; attempt < groups && !moved; ++attempt) {
            const u32 g = (first + attempt) % groups;
            u64 depends_on = ~u64{0};
            const u64 index = engines[g].Step(words, pending, &depends_on);
            if (index != ~u64{0}) {
                outputs[g].push_back(Processed{index, g, depends_on});
                if (engine_log != nullptr) {
                    engine_log->push_back(outputs[g].back());
                }
                --remaining;
                moved = true;
            }
        }
        if (!moved) {
            return {}; // a dependency cycle: no engine can move
        }
    }
    // The merge rule: the smallest ticket among the engines' heads whose dependency is merged
    // (an acquire after its release, a macro call after every earlier upload).
    std::vector<size_t> heads(groups, 0);
    std::vector<bool> merged(events.size(), false);
    std::vector<u64> uploads;
    for (u64 i = 0; i < events.size(); ++i) {
        if (events[i].kind == EventKind::MacroUpload) {
            uploads.push_back(i);
        }
    }
    size_t uploads_merged_prefix = 0;
    std::vector<u64> result;
    result.reserve(events.size());
    while (result.size() < events.size()) {
        u64 best = ~u64{0};
        u32 best_group = 0;
        for (u32 g = 0; g < groups; ++g) {
            if (heads[g] >= outputs[g].size()) {
                continue;
            }
            const Processed& head = outputs[g][heads[g]];
            if (head.depends_on != ~u64{0} && !merged[head.depends_on]) {
                continue;
            }
            if (events[head.index].kind == EventKind::MacroCall) {
                while (uploads_merged_prefix < uploads.size() &&
                       merged[uploads[uploads_merged_prefix]]) {
                    ++uploads_merged_prefix;
                }
                if (uploads_merged_prefix < uploads.size() &&
                    uploads[uploads_merged_prefix] < head.index) {
                    continue;
                }
            }
            if (head.index < best) {
                best = head.index;
                best_group = g;
            }
        }
        if (best == ~u64{0}) {
            return {};
        }
        merged[best] = true;
        ++heads[best_group];
        result.push_back(best);
    }
    return result;
}

namespace {

std::vector<u64> ObservedReleases(std::span<const Event> events, std::span<const u64> order) {
    std::vector<u64> observed(events.size(), ~u64{0});
    std::unordered_map<u64, u64> last;
    for (const u64 index : order) {
        const Event& event = events[index];
        if (event.kind == EventKind::Release) {
            last[event.address] = index;
        } else if (event.kind == EventKind::Acquire) {
            const auto it = last.find(event.address);
            observed[index] = it != last.end() ? it->second : ~u64{0};
        }
    }
    return observed;
}

std::vector<u64> MacroVersions(std::span<const Event> events, std::span<const u64> order) {
    std::vector<u64> versions(events.size(), 0);
    u64 uploads = 0;
    for (const u64 index : order) {
        if (events[index].kind == EventKind::MacroUpload) {
            ++uploads;
        } else if (events[index].kind == EventKind::MacroCall) {
            versions[index] = uploads;
        }
    }
    return versions;
}

} // Anonymous namespace

CompareResult Compare(std::span<const Event> events, std::span<const u64> reference,
                      std::span<const u64> merged) {
    CompareResult result;
    const auto mismatch = [&](const char* field, u64 index) {
        if (result.mismatches++ == 0) {
            result.first_field = field;
            result.first_index = index;
        }
    };
    if (reference.size() != merged.size()) {
        mismatch("event count", 0);
        return result;
    }
    // Per-channel order.
    std::unordered_map<u32, std::vector<u64>> by_channel_ref;
    std::unordered_map<u32, std::vector<u64>> by_channel_merged;
    for (const u64 index : reference) {
        by_channel_ref[events[index].channel].push_back(index);
    }
    for (const u64 index : merged) {
        by_channel_merged[events[index].channel].push_back(index);
    }
    for (const auto& [channel, list] : by_channel_ref) {
        ++result.compared;
        if (by_channel_merged[channel] != list) {
            mismatch("channel order", channel);
        }
    }
    // The release each acquire observed, and the macro RAM version each call read.
    const std::vector<u64> observed_ref = ObservedReleases(events, reference);
    const std::vector<u64> observed_merged = ObservedReleases(events, merged);
    const std::vector<u64> versions_ref = MacroVersions(events, reference);
    const std::vector<u64> versions_merged = MacroVersions(events, merged);
    for (u64 i = 0; i < events.size(); ++i) {
        if (events[i].kind == EventKind::Acquire) {
            ++result.compared;
            if (observed_ref[i] != observed_merged[i]) {
                mismatch("acquire release", i);
            }
        } else if (events[i].kind == EventKind::MacroCall) {
            ++result.compared;
            if (versions_ref[i] != versions_merged[i]) {
                mismatch("macro version", i);
            }
        }
    }
    // QMD table accesses keep their order.
    std::vector<u64> qmd_ref;
    std::vector<u64> qmd_merged;
    for (const u64 index : reference) {
        if (events[index].kind == EventKind::QmdUse) {
            qmd_ref.push_back(index);
        }
    }
    for (const u64 index : merged) {
        if (events[index].kind == EventKind::QmdUse) {
            qmd_merged.push_back(index);
        }
    }
    ++result.compared;
    if (qmd_ref != qmd_merged) {
        mismatch("qmd order", 0);
    }
    return result;
}

bool MergeInvariantHolds(std::span<const Event> events, std::span<const u64> merged) {
    const std::vector<bool> memory = MemorySatisfied(events);
    std::unordered_map<u32, u64> last_of_channel;
    std::unordered_map<u64, u64> words;
    for (const u64 index : merged) {
        const Event& event = events[index];
        const auto it = last_of_channel.find(event.channel);
        if (it != last_of_channel.end() && it->second > index) {
            return false;
        }
        last_of_channel[event.channel] = index;
        if (event.kind == EventKind::Release) {
            words[event.address] = event.value;
        } else if (event.kind == EventKind::Acquire && !memory[index]) {
            const auto word = words.find(event.address);
            if (word == words.end() || !Satisfies(event.op, word->second, event.value)) {
                return false;
            }
        }
    }
    return true;
}

// ---- The real front -----------------------------------------------------------------------------

namespace {

constexpr size_t Window = 4096;
constexpr u32 MaxEngines = 4;

struct Recorder {
    std::vector<Event> window;
    u64 seed = 1;
    u64 window_start_ns = 0;
};

Recorder& GetRecorder() {
    static Recorder recorder;
    return recorder;
}

struct Counters {
    std::array<std::atomic<u64>, static_cast<size_t>(EventKind::Count)> events{};
    std::array<std::atomic<u64>, static_cast<size_t>(Fallback::Count)> plans{};
    std::atomic<u64> windows{};
    std::atomic<u64> groups_sum{};
    std::atomic<u64> cross_edges{};
    std::atomic<u64> macro_edges{};
    std::atomic<u64> qmd_joins{};
    std::atomic<u64> verify_windows{};
    std::atomic<u64> verify_compared{};
    std::atomic<u64> verify_mismatches{};
    std::atomic<u64> invariant_failures{};
    std::atomic<u64> analysis_ns{};
};

Counters& GetCounters() {
    static Counters counters;
    return counters;
}

void Analyze(std::span<const Event> window, u64 seed) {
    Counters& counters = GetCounters();
    const auto start = std::chrono::steady_clock::now();
    Grouper grouper;
    for (const Event& event : window) {
        grouper.Note(event);
    }
    Plan plan = grouper.Decide(MaxEngines);
    if (Verify(Switch::MultiFront) && plan.fallback == Fallback::None) {
        const std::vector<u64> merged = MultiFrontOrder(window, plan, seed);
        if (merged.empty()) {
            plan.fallback = Fallback::Cycle;
        } else {
            // The real front's order is the single-front reference.
            std::vector<u64> reference(window.size());
            for (u64 i = 0; i < reference.size(); ++i) {
                reference[i] = i;
            }
            const CompareResult result = Compare(window, reference, merged);
            counters.verify_windows.fetch_add(1, std::memory_order_relaxed);
            counters.verify_compared.fetch_add(result.compared, std::memory_order_relaxed);
            counters.verify_mismatches.fetch_add(result.mismatches, std::memory_order_relaxed);
            if (!MergeInvariantHolds(window, merged)) {
                counters.invariant_failures.fetch_add(1, std::memory_order_relaxed);
            }
            static std::atomic<u32> logged{};
            if (result.mismatches != 0 && logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                LOG_WARNING(HW_GPU,
                            "V2 multi-front verify: {} of {} comparisons differ from the single "
                            "front (first: {} at event {}, {} groups)",
                            result.mismatches, result.compared, result.first_field,
                            result.first_index, plan.groups);
            }
        }
    }
    counters.windows.fetch_add(1, std::memory_order_relaxed);
    counters.plans[static_cast<size_t>(plan.fallback)].fetch_add(1, std::memory_order_relaxed);
    counters.groups_sum.fetch_add(plan.fallback == Fallback::None ? plan.groups : 1,
                                  std::memory_order_relaxed);
    counters.cross_edges.fetch_add(plan.cross_edges, std::memory_order_relaxed);
    counters.macro_edges.fetch_add(plan.macro_edges, std::memory_order_relaxed);
    counters.qmd_joins.fetch_add(plan.joins_qmd, std::memory_order_relaxed);
    counters.analysis_ns.fetch_add(
        static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count()),
        std::memory_order_relaxed);
}

} // Anonymous namespace

void Record(EventKind kind, u64 address, u64 value, AcquireOp op) {
    if (t_channel == ~0U) {
        return; // only the front thread records (the back runs macros too in split mode 1)
    }
    Recorder& recorder = GetRecorder();
    GetCounters().events[static_cast<size_t>(kind)].fetch_add(1, std::memory_order_relaxed);
    recorder.window.push_back(Event{address, value, t_channel, kind, op});
    if (recorder.window.size() >= Window) {
        Analyze(recorder.window, recorder.seed++);
        recorder.window.clear();
        if (V2::Stats::Enabled()) {
            const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 std::chrono::steady_clock::now().time_since_epoch())
                                                 .count());
            if (now - recorder.window_start_ns > 10'000'000'000ULL) {
                if (recorder.window_start_ns != 0) {
                    ReportStats("GPU thread");
                }
                recorder.window_start_ns = now;
            }
        }
    }
}

void SetActive(bool active) {
    g_recording = active && On(Switch::MultiFront);
    if (g_recording) {
        GetRecorder().window.reserve(Window);
    } else {
        GetRecorder().window.clear();
    }
}

void ReportStats(const char* thread_name) {
    Counters& c = GetCounters();
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const u64 windows = take(c.windows);
    if (windows == 0) {
        return;
    }
    std::string events;
    for (size_t kind = 0; kind < c.events.size(); ++kind) {
        events += fmt::format("{}{} {}", events.empty() ? "" : ", ", event_names[kind],
                              take(c.events[kind]));
    }
    std::string plans;
    for (size_t reason = 0; reason < c.plans.size(); ++reason) {
        const u64 count = take(c.plans[reason]);
        if (count != 0) {
            plans += fmt::format("{}{} {}", plans.empty() ? "" : ", ",
                                 reason == 0 ? "multi" : fallback_names[reason], count);
        }
    }
    LOG_INFO(HW_GPU,
             "V2 stats {} P6b multi-front | events: {} | windows {}: {} | mean engines {:.2f} | "
             "cross-group semaphore edges {}, macro edges {}, QMD joins {} | verify: {} windows, "
             "{} comparisons, {} mismatches, merge invariant failures {} | analysis {:.2f} ms",
             thread_name, events, windows, plans,
             static_cast<double>(take(c.groups_sum)) / static_cast<double>(windows),
             take(c.cross_edges), take(c.macro_edges), take(c.qmd_joins),
             take(c.verify_windows), take(c.verify_compared), take(c.verify_mismatches),
             take(c.invariant_failures), static_cast<double>(take(c.analysis_ns)) / 1e6);
}

} // namespace VideoCommon::V2::Resolve::MultiFront
