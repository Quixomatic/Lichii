// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"

// Pipeline v2 P6b: several S1 front engines over disjoint channel groups (MASTER_PLAN §5.1
// "Scaling the front", notes/01 §7.1.5 and §7.4).
//
// Channels are independent except for three process-wide coupling points: macro RAM (uploaded by
// one channel, executed by all), the native QMD dependency table, and semaphores released by one
// channel and acquired by another. This header holds the pure parts:
//  - VersionedStore: the coupling points behind a small store whose values carry the global
//    ticket of the write that made them, so an engine reads "the value as of my position";
//  - Grouper: which channels must share an engine (QMD table, macro RAM writes) and whether a
//    title is too coupled for more than one (fallback reasons);
//  - the merge rule: each engine's output stream is merged into one global order by ticket, an
//    acquire after the release that satisfied it and a macro call after every earlier upload
//    (sequence-number dependencies between streams);
//  - the verify model: the multi-front order of the same inputs against the single-front order.
namespace VideoCommon::V2::Resolve::MultiFront {

enum class EventKind : u8 {
    Work,        ///< a command list entry of the channel (no coupling)
    Release,     ///< semaphore release (address, value)
    Acquire,     ///< semaphore acquire (address, value, op)
    MacroUpload, ///< a macro RAM write (address = macro RAM position)
    MacroCall,   ///< a macro execution (reads macro RAM)
    QmdUse,      ///< a native QMD dependency table access (signal or scheduled launch)
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(EventKind::Count)> event_names{
    "work", "release", "acquire", "macro upload", "macro call", "qmd",
};

enum class AcquireOp : u8 {
    Equal,
    GreaterEqual,
};

struct Event {
    u64 address = 0;
    u64 value = 0;
    u32 channel = 0;
    EventKind kind = EventKind::Work;
    AcquireOp op = AcquireOp::Equal;
};

[[nodiscard]] constexpr bool Satisfies(AcquireOp op, u64 have, u64 want) {
    return op == AcquireOp::Equal ? have == want : have >= want;
}

/// A process-wide value behind versions. Writers publish (key, ticket, value); a reader at ticket
/// T reads the newest version written before T. Not on any per-draw path: one small mutex.
template <typename Value>
class VersionedStore {
public:
    struct Version {
        u64 ticket;
        Value value;
    };

    void Publish(u64 key, u64 ticket, const Value& value) {
        std::scoped_lock lock{mutex};
        auto& versions = table[key];
        const auto it = std::upper_bound(
            versions.begin(), versions.end(), ticket,
            [](u64 t, const Version& version) { return t < version.ticket; });
        versions.insert(it, Version{ticket, value});
        ++version_count;
    }

    /// The newest version with ticket < `before`.
    [[nodiscard]] std::optional<Version> Read(u64 key, u64 before) const {
        std::scoped_lock lock{mutex};
        const auto found = table.find(key);
        if (found == table.end()) {
            return std::nullopt;
        }
        const auto& versions = found->second;
        const auto it = std::lower_bound(
            versions.begin(), versions.end(), before,
            [](const Version& version, u64 t) { return version.ticket < t; });
        if (it == versions.begin()) {
            return std::nullopt;
        }
        return *std::prev(it);
    }

    /// The first version (by ticket) whose value satisfies `pred`, or none.
    template <typename Pred>
    [[nodiscard]] std::optional<Version> FirstWhere(u64 key, Pred&& pred) const {
        std::scoped_lock lock{mutex};
        const auto found = table.find(key);
        if (found == table.end()) {
            return std::nullopt;
        }
        for (const Version& version : found->second) {
            if (pred(version.value)) {
                return version;
            }
        }
        return std::nullopt;
    }

    /// Drops versions no reader below `floor` can need (keeps the newest one before it).
    void Retire(u64 floor) {
        std::scoped_lock lock{mutex};
        for (auto& [key, versions] : table) {
            const auto it = std::lower_bound(
                versions.begin(), versions.end(), floor,
                [](const Version& version, u64 t) { return version.ticket < t; });
            if (it != versions.begin()) {
                versions.erase(versions.begin(), std::prev(it));
            }
        }
    }

    [[nodiscard]] u64 Versions() const {
        std::scoped_lock lock{mutex};
        return version_count;
    }

private:
    mutable std::mutex mutex;
    std::unordered_map<u64, std::vector<Version>> table;
    u64 version_count = 0;
};

/// The three coupling points as stores (macro RAM words, QMD table counters, semaphore words).
struct Stores {
    VersionedStore<u32> macro_ram;
    VersionedStore<u32> qmd_table;
    VersionedStore<u64> semaphores;
};

/// Why a title runs with one front.
enum class Fallback : u8 {
    None,
    SingleChannel, ///< only one channel did work
    OneGroup,      ///< the QMD table / macro RAM couplings joined every channel
    SemaphoreRate, ///< too many cross-group semaphore dependencies per event
    Cycle,         ///< the merge found no ready stream (a dependency cycle)
    Count,
};
inline constexpr std::array<const char*, static_cast<size_t>(Fallback::Count)> fallback_names{
    "none", "single channel", "one group", "semaphore rate", "cycle",
};

struct Plan {
    std::unordered_map<u32, u32> group_of; ///< channel -> group
    u32 groups = 1;
    Fallback fallback = Fallback::None;
    u64 cross_edges = 0;   ///< semaphore dependencies between groups
    u64 joins_qmd = 0;     ///< channel pairs joined by the QMD table
    u64 macro_edges = 0;   ///< macro calls after another group's upload in the window
    u64 events = 0;

    [[nodiscard]] u32 GroupOf(u32 channel) const {
        const auto it = group_of.find(channel);
        return it != group_of.end() ? it->second : 0;
    }
};

/// Channel grouping. Channels that share the QMD table go to one engine: its dependency counters are
/// read-modify-write state with no guest-defined order between channels, so splitting them would
/// need a lock-step protocol. Macro RAM is a versioned store: a call reads the version as of its
/// ticket, ordered after every earlier upload by a sequence dependency (a cross edge). Semaphores
/// stay cross-engine dependencies too (they define their own order).
class Grouper {
public:
    /// Max cross-group semaphore dependencies per event before falling back (1 in 32).
    static constexpr u64 MaxEdgeShift = 5;

    void Note(const Event& event);
    [[nodiscard]] Plan Decide(u32 max_groups) const;
    void Clear();

private:
    u32 IndexOf(u32 channel);
    u32 Find(u32 index) const;
    void Join(u32 a, u32 b);

    std::unordered_map<u32, u32> index_of;
    std::vector<u32> channels;
    mutable std::vector<u32> parent;
    std::vector<u64> work;
    std::unordered_map<u64, u32> last_release; ///< address -> channel index
    std::vector<std::pair<u32, u32>> semaphore_edges;
    std::vector<std::pair<u32, u32>> macro_edges; ///< (uploader, caller) after an upload
    std::optional<u32> qmd_user;
    std::optional<u32> last_uploader;
    u64 joins_qmd = 0;
    u64 events = 0;
};

/// An event as an engine processed it: its index in the input, its ticket (the single-front
/// position of the submission it belongs to; here the input index), the engine (group), and the
/// release it waited for (the input index of the satisfying release, or ~0).
struct Processed {
    u64 index = 0;
    u32 group = 0;
    u64 depends_on = ~u64{0};
};

/// The single-front reference: events in input order, a channel suspended at an unmet acquire and
/// resumed once a release meets it (Scheduler::ResumePending). Returns input indices.
[[nodiscard]] std::vector<u64> SingleFrontOrder(std::span<const Event> events);

/// Multi-front: one engine per plan group, each processing its channels like the single front
/// against the shared semaphore store as published so far, advancing in an order `seed` chooses
/// (asynchronous engines); then the merge rule. Returns input indices, or empty on a cycle.
[[nodiscard]] std::vector<u64> MultiFrontOrder(std::span<const Event> events, const Plan& plan,
                                               u64 seed, std::vector<Processed>* engine_log =
                                                             nullptr);

struct CompareResult {
    u64 compared = 0;
    u64 mismatches = 0;
    const char* first_field = nullptr;
    u64 first_index = ~u64{0};
};

/// Compares a merged order with the single-front order where it is defined: per-channel order,
/// the release each acquire observed, the macro RAM version each call read, and the order of QMD
/// table accesses.
[[nodiscard]] CompareResult Compare(std::span<const Event> events, std::span<const u64> reference,
                                    std::span<const u64> merged);

/// Checks the merge invariant itself: every acquire comes after a release that satisfies it (or
/// was satisfied by memory before the window), per-channel order holds.
[[nodiscard]] bool MergeInvariantHolds(std::span<const Event> events, std::span<const u64> merged);

// ---- The real front (EDEN_NX2_V2_MULTIFRONT): event recording and analysis ---------------------

/// Set when the switch is on and the split runs (one load per hook).
inline bool g_recording = false;
/// The channel the front is dispatching (Scheduler::BindChannel).
inline thread_local u32 t_channel = 0;

/// Front thread: records one coupling event of the current channel. Every Window events the
/// window is analysed (grouping plan; with verify, the multi-front order is compared with the real
/// single-front order) and the counters are updated.
void Record(EventKind kind, u64 address = 0, u64 value = 0, AcquireOp op = AcquireOp::Equal);
inline void Note(EventKind kind, u64 address = 0, u64 value = 0,
                 AcquireOp op = AcquireOp::Equal) {
    if (g_recording) [[unlikely]] {
        Record(kind, address, value, op);
    }
}
void SetActive(bool active);
void ReportStats(const char* thread_name);

} // namespace VideoCommon::V2::Resolve::MultiFront
