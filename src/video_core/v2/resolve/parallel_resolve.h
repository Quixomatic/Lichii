// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/resolve/segment.h"

#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
#include <immintrin.h>
#endif

// Pipeline v2 P6a: shard workers and the ordered merge (MASTER_PLAN §5.6.2-§5.6.6).
//
// A producer (the segmenter's owner) submits closed segments with consecutive ids. N shard threads
// claim them through one shared cursor (dynamic round robin: an idle shard takes the oldest
// unclaimed segment) and resolve each into relocatable records plus a Commit::RequestList, using
// only pure lookups. The commit thread takes segments strictly in id order: it applies a segment's
// requests first (proxies patched), then patches the records, then hands them to encode. Encode
// therefore sees the serial order by construction, whatever order shards finish in.
//
// The commit never waits for a later segment. For the segment it needs next it either takes the
// finished result, waits for the shard that is running it (bounded by one segment's resolve), or
// steals it: a segment no shard has started is resolved on the commit thread at once. Segments in
// flight are bounded by `depth`; a submission that finds no free slot is refused (back pressure)
// and the commit resolves that segment serially.
//
// Traits:
//   using Input;  using Record;  using Shard;      // Shard: per-thread resolver state
//   static SerialReason Resolve(Shard&, const Input&, std::vector<Record>&, Commit::RequestList&);
//   static void Patch(Record&, const Commit::RequestList&);   // proxies -> committed slots
//   static const char* Compare(const Record& shard, const Record& serial); // null when equal;
//       records keep their lookup keys, so a proxy (shard) and a slot (serial) for the same key
//       compare equal (verify)
namespace VideoCommon::V2::Resolve {

/// EDEN_NX2_V2_STATS counters of the parallel resolve (process-wide, relaxed atomics).
/// P6c: the resolve steps a shard can perform for the commit thread (stats per step).
enum class Step : u8 {
    Shape,       ///< P4 draw-cache shape
    DrawKey,     ///< P4 draw-cache key with the constant buffer bindings
    PipelineKey, ///< P3 pipeline key (FixedPipelineState) and its hash
    Pipeline,    ///< R3: the pipeline lookup (shader hashes + variant table)
    Dynamic,     ///< R5: the record's dynamic state (EDEN_NX2_V2_DRAWREC full resolves)
    Count,
};
inline constexpr size_t NumSteps = static_cast<size_t>(Step::Count);
inline constexpr std::array<const char*, NumSteps> step_names{"shape", "draw key", "pipeline key",
                                                              "pipeline", "dynamic state"};

struct Stats {
    static constexpr size_t MaxShards = 16;
    std::atomic<u64> segments{};      ///< committed
    std::atomic<u64> by_shard{};      ///< committed with a shard's result
    std::array<std::atomic<u64>, NumSerialReasons> serial{}; ///< committed serially, by reason
    std::array<std::atomic<u64>, NumCutReasons> cuts{};
    std::atomic<u64> records{};
    std::atomic<u64> commit_ns{};     ///< commit thread: take + apply + patch + encode hand-off
    std::atomic<u64> merge_wait_ns{}; ///< commit thread waiting for the shard running its segment
    std::atomic<u64> merge_waits{};
    std::atomic<u64> stolen_resolve_ns{}; ///< commit thread resolving serial/stolen segments
    std::array<std::atomic<u64>, MaxShards> shard_busy_ns{};
    std::array<std::atomic<u64>, MaxShards> shard_segments{};
    std::atomic<u64> refused{};       ///< submissions refused by back pressure
    std::atomic<u64> stolen_unsubmitted{}; ///< the commit reached a segment not submitted yet
    std::atomic<u64> stolen_queued{};      ///< the commit took a submitted segment no shard began
    std::atomic<u64> early_submits{};   ///< segments submitted before their end was published
    std::atomic<u64> front_snapshot_ns{}; ///< the front's snapshot time at cuts (R0 gate)
    std::atomic<u64> early_streamed{}; ///< ...whose records the commit took as they came (R0-d)
    std::atomic<u64> stream_finish_ns{}; ///< commit waiting for a streamed segment's shard to stop
    std::atomic<u64> draws_not_ready{}; ///< streamed draws whose record was not ready yet (live)
    std::atomic<u64> verify_segments{};
    std::atomic<u64> verify_mismatches{};
    std::atomic<u64> verify_dependent{}; ///< request differences an earlier commit explains
    std::atomic<u64> determinism_checks{};   ///< segments committed in id order (by construction)
    std::atomic<u64> determinism_failures{}; ///< a segment committed out of order (must stay 0)
    std::atomic<u64> snapshot_blocks_copied{};
    std::atomic<u64> snapshot_blocks_shared{};
    std::atomic<u32> segment_size{};  ///< current adaptive size (last set)
    std::atomic<u32> shards{};
    // Real-path consumption (split_resolve): pre-resolved draws.
    std::atomic<u64> draws_pre{};       ///< draws the commit took from a shard record
    std::atomic<u64> draws_unmatched{}; ///< draws with no matching record (other engine, skipped)
    std::atomic<u64> draws_drift{};     ///< shard state disagreed with the record's fingerprint
    std::atomic<u64> draws_no_shape{};  ///< shard had no published shape (key not computed)
    std::atomic<u64> verify_draws{};
    std::atomic<u64> verify_draw_mismatches{};
    std::atomic<u64> snapshot_checks{};
    std::atomic<u64> snapshot_mismatches{};
    // P6c: who performed each resolve step of a draw on the commit thread's path.
    std::array<std::atomic<u64>, NumSteps> step_shard{};
    std::array<std::atomic<u64>, NumSteps> step_commit{};
    std::atomic<u64> commit_draws{};   ///< draws resolved and recorded on the commit thread
    std::atomic<u64> commit_draw_ns{}; ///< their time in PrepareDraw
    /// F1-36: the front's draws the commit reached in segments (ordinals, predicated-off ones
    /// included: the unit the segmenter cuts in), and those the back never drew.
    std::atomic<u64> front_draws{};
    std::atomic<u64> draws_predicated_off{};
    std::atomic<u64> draws_misfit{}; ///< the record at the draw's ordinal was for another draw
};
Stats& GetStats();

inline void Count(std::atomic<u64>& counter, u64 amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

/// EDEN_NX2_V2_PARALLEL_SHARDS (default min(hardware threads - ReservedThreads, 4), at least 1).
u32 ShardsFromEnv();
/// Threads already busy that shards should not compete with: four guest CPU cores, the split's
/// front, its back (the commit thread).
inline constexpr u32 ReservedThreads = 6;
/// EDEN_NX2_V2_PARALLEL_SEGMENT, see SegmentPolicy.
SegmentPolicy PolicyFromEnv();
/// Segments in flight (EDEN_NX2_V2_PARALLEL_DEPTH, default 8, a power of two in [2, 64]).
u32 DepthFromEnv();

/// Logs the P6 line and resets the window (EDEN_NX2_V2_STATS; the commit thread).
void ReportStats(const char* thread_name, u64 window_ns);

/// What a verify log line is about: each kind has its own budget (Fix pass 1: the segment lines
/// took all 32 and no snapshot or draw detail was ever logged).
enum class MismatchLog : u8 { Segment, Snapshot, Draw, Count };
/// Rate-limited verify log (first 32 mismatches of each kind).
bool ShouldLogMismatch(MismatchLog kind = MismatchLog::Segment);

[[nodiscard]] inline u64 NowNs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch())
                                .count());
}

inline void CpuRelax() {
#if defined(_M_X64) || defined(__x86_64__) || defined(_M_IX86) || defined(__i386__)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

struct RequestCompare {
    const char* field = nullptr; ///< first real difference, or null
    u32 dependent = 0;           ///< differences an earlier segment's commit explains
};

/// Field-by-field comparison of a shard's request list with the serial resolve of the same
/// segment at commit time (verify). A create the shard made (it has a proxy) that the serial
/// resolve doesn't make, or a request on a proxy where the serial one names the real slot, is
/// dependent: an earlier segment's commit made the resource after the shard looked, and the
/// commit's appliers re-check (ensure semantics, P5a I7). Everything else must be equal.
[[nodiscard]] inline RequestCompare CompareRequests(std::span<const Commit::Request> shard,
                                                    std::span<const Commit::Request> serial) {
    RequestCompare result;
    const auto same_kind = [](const Commit::Request& x, const Commit::Request& y) {
        return x.kind == y.kind && x.resource == y.resource && x.address == y.address &&
               x.size == y.size && x.arg == y.arg;
    };
    size_t i = 0;
    size_t j = 0;
    while (i < shard.size() && j < serial.size()) {
        const Commit::Request& x = shard[i];
        const Commit::Request& y = serial[j];
        const bool slots_match =
            x.slot == y.slot || (Commit::IsProxy(x.slot) && Commit::IsProxy(y.slot));
        const bool proxies_match = Commit::IsProxy(x.proxy) == Commit::IsProxy(y.proxy);
        if (same_kind(x, y) && slots_match && proxies_match) {
            if (x.effects != y.effects) {
                result.field = "request effects";
                return result;
            }
            ++i;
            ++j;
        } else if (Commit::IsProxy(x.proxy) && !Commit::IsProxy(y.proxy) && same_kind(x, y)) {
            ++result.dependent; // the shard asked to create what the serial resolve found
            ++i;
            ++j;
        } else if (Commit::IsProxy(x.proxy)) {
            ++result.dependent; // a create the serial resolve didn't need at all
            ++i;
        } else if (Commit::IsProxy(x.slot) && !Commit::IsProxy(y.slot) && same_kind(x, y)) {
            ++result.dependent; // on a proxy where the serial resolve names the slot
            ++i;
            ++j;
        } else {
            result.field = x.kind != y.kind             ? "request kind"
                           : x.resource != y.resource   ? "request resource"
                           : x.address != y.address || x.size != y.size ? "request range"
                           : x.arg != y.arg             ? "request arg"
                                                        : "request slot";
            return result;
        }
    }
    for (; i < shard.size(); ++i) {
        if (!Commit::IsProxy(shard[i].proxy)) {
            result.field = "request count";
            return result;
        }
        ++result.dependent;
    }
    if (j < serial.size()) {
        result.field = "request count";
    }
    return result;
}

/// Fixed-size chunks whose elements never move: a record may point at an element (§5.2: records
/// are plain data with their payloads in an arena) while the resolver adds more. Clear keeps the
/// chunks for the next segment.
template <typename T, size_t ChunkSize = 8>
class StableArena {
public:
    T& Push() {
        const size_t chunk = used / ChunkSize;
        if (chunk == chunks.size()) {
            chunks.push_back(std::make_unique<std::array<T, ChunkSize>>());
        }
        return (*chunks[chunk])[used++ % ChunkSize];
    }
    void Clear() {
        used = 0;
    }
    [[nodiscard]] size_t Size() const {
        return used;
    }

private:
    std::vector<std::unique_ptr<std::array<T, ChunkSize>>> chunks;
    size_t used = 0;
};

/// Traits::Arena (optional): payloads a segment's records point into, kept with its result.
template <typename Traits>
struct ArenaOf {
    struct type {
        void Clear() {}
    };
};
template <typename Traits>
    requires requires { typename Traits::Arena; }
struct ArenaOf<Traits> {
    using type = typename Traits::Arena;
};

template <typename Traits>
class ParallelResolver {
public:
    using Input = typename Traits::Input;
    using Record = typename Traits::Record;
    using Shard = typename Traits::Shard;
    using Arena = typename ArenaOf<Traits>::type;

    struct Config {
        u32 shards = 2;
        u32 depth = 8;              ///< segments in flight, a power of two
        bool verify = false;        ///< resolve shard segments serially too and compare
        bool resolve_stolen = true; ///< resolve serial/stolen segments here (else the caller does)
        bool stats = false;
    };

    struct Result {
        u64 id = 0;
        std::vector<Record> records;
        Commit::RequestList requests;
        Arena arena; ///< what `records` point into (valid until the segment is released)
        SerialReason serial = SerialReason::None;
        bool by_shard = false; ///< the records came from a shard
        bool resolved = false; ///< records and requests are filled (else the caller resolves)
        bool waited = false;   ///< the commit waited for the running shard
        u64 wait_ns = 0;       ///< how long (F1-36: the adaptive size weighs it)
        bool stolen = false;   ///< no shard had started it
        bool has_input = false;
    };

    template <typename MakeShard>
    ParallelResolver(const Config& config_, MakeShard&& make_shard)
        : config{config_}, mask{std::bit_ceil((std::max)(config_.depth, 2U)) - 1}, slots(mask + 1) {
        config.shards = std::clamp<u32>(config.shards, 1, Stats::MaxShards);
        for (u64 index = 0; index < slots.size(); ++index) {
            slots[index].word.store(Pack(index, State::Free), std::memory_order_relaxed);
        }
        commit_shard = make_shard(config.shards);
        shard_states.reserve(config.shards);
        for (u32 index = 0; index < config.shards; ++index) {
            shard_states.push_back(make_shard(index));
        }
        for (u32 index = 0; index < config.shards; ++index) {
            threads.emplace_back([this, index](std::stop_token stop) { ShardLoop(stop, index); });
        }
        GetStats().shards.store(config.shards, std::memory_order_relaxed);
    }

    ~ParallelResolver() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        work.fetch_add(1, std::memory_order_release);
        work.notify_all();
        threads.clear();
    }

    ParallelResolver(const ParallelResolver&) = delete;
    ParallelResolver& operator=(const ParallelResolver&) = delete;

    /// Producer: hands segment `id` (consecutive ids) to the shards. `reason` != None makes it
    /// serial up front (no shard touches it). False when refused: no free slot (back pressure) or
    /// the commit already took it; the commit then resolves it serially. `retry` (optional): set
    /// when refused only because the slot still holds an older segment; `input` is then left
    /// as it was and the producer submits it again later (AP6 T0: back pressure is not
    /// serialisation).
    bool Submit(u64 id, Input&& input, SerialReason reason = SerialReason::None,
                bool* retry = nullptr) {
        Slot& slot = slots[id & mask];
        const u64 expected_word = Pack(id, State::Free);
        if (retry != nullptr) {
            const u64 word = slot.word.load(std::memory_order_acquire);
            *retry = IdOf(word) < id;
            if (*retry) {
                Count(GetStats().refused);
                return false;
            }
        }
        bool accepted = false;
        if (slot.word.load(std::memory_order_acquire) == expected_word) {
            slot.input = std::move(input);
            slot.reason = reason;
            u64 expected = expected_word;
            const State state = reason == SerialReason::None ? State::Queued : State::SerialQueued;
            accepted = slot.word.compare_exchange_strong(expected, Pack(id, state),
                                                         std::memory_order_acq_rel);
        }
        u64 seen = attempted.load(std::memory_order_relaxed);
        while (seen < id + 1 &&
               !attempted.compare_exchange_weak(seen, id + 1, std::memory_order_release)) {
        }
        if (!accepted) {
            Count(GetStats().refused);
            return false;
        }
        work.fetch_add(1, std::memory_order_release);
        work.notify_one();
        return true;
    }

    /// Commit thread: segment `id` (the next in order). Waits only for a shard already running it.
    Result& Take(u64 id) {
        Slot& slot = slots[id & mask];
        Result& result = slot.result;
        for (u32 spins = 0;; ++spins) {
            u64 word = slot.word.load(std::memory_order_acquire);
            if (IdOf(word) != id) {
                // The slot still holds an older segment: the producer never reached this one.
                Count(GetStats().stolen_unsubmitted);
                StealEmpty(slot, id, SerialReason::Stolen);
                return result;
            }
            switch (StateOf(word)) {
            case State::Free:
                if (slot.word.compare_exchange_strong(word, Pack(id, State::Stolen),
                                                      std::memory_order_acq_rel)) {
                    Count(GetStats().stolen_unsubmitted);
                    const bool refused = attempted.load(std::memory_order_acquire) > id;
                    StealEmpty(slot, id,
                               refused ? SerialReason::BackPressure : SerialReason::Stolen);
                    return result;
                }
                continue;
            case State::Queued:
            case State::SerialQueued: {
                if (!slot.word.compare_exchange_strong(word, Pack(id, State::Stolen),
                                                       std::memory_order_acq_rel)) {
                    continue;
                }
                const bool serial_up_front = StateOf(word) == State::SerialQueued;
                if (!serial_up_front) {
                    Count(GetStats().stolen_queued);
                }
                ResetResult(result, id);
                result.stolen = !serial_up_front;
                result.serial = serial_up_front ? slot.reason : SerialReason::Stolen;
                result.has_input = true;
                ResolveHere(slot);
                return result;
            }
            case State::Running: {
                if constexpr (requires { slot.input.progress->ready; }) {
                    if (slot.input.progress != nullptr) {
                        // F1-36 (R0-d): an early-submitted segment's shard publishes its records
                        // one at a time; the commit takes each draw's record once it is ready
                        // and resolves the draw live otherwise, never waiting here. (This
                        // replaces abandoning a shard that waits for the front: that valve kept
                        // the back from waiting on a shard waiting on the front, and the commit
                        // now waits on no early segment until FinishStream, when the segment's
                        // end is published.) The result belongs to the shard until then.
                        streaming_id = id;
                        Count(GetStats().early_streamed);
                        return result;
                    }
                }
                const u64 start = NowNs();
                while (StateOf(slot.word.load(std::memory_order_acquire)) == State::Running) {
                    if (++spins < 4096) {
                        CpuRelax();
                    } else {
                        std::this_thread::yield();
                    }
                }
                const u64 waited = NowNs() - start;
                result.waited = true;
                result.wait_ns += waited;
                if (config.stats) {
                    Count(GetStats().merge_wait_ns, waited);
                    Count(GetStats().merge_waits);
                }
                continue;
            }
            case State::Done:
                result.has_input = true;
                if (result.serial != SerialReason::None) {
                    // The shard found a reason to go serial while resolving.
                    ResolveHere(slot);
                } else if (config.verify) {
                    VerifyHere(slot);
                }
                return result;
            case State::Stolen:
                return result;
            }
        }
    }

    /// Commit thread: the whole ordered merge of segment `id`. `fallback(result)` fills an
    /// unresolved result (a segment the producer never submitted) from the caller's own data;
    /// `apply(request)` is the commit applier (returns the slot for the request's proxy);
    /// `encode(records)` receives the patched records in order. Returns the result's serial reason.
    template <typename Fallback, typename Apply, typename Encode>
    SerialReason Commit(u64 id, Fallback&& fallback, Apply&& apply, Encode&& encode) {
        const u64 start = config.stats ? NowNs() : 0;
        Result& result = Merge(id, fallback, apply);
        encode(std::span<const Record>(result.records));
        NoteCommitted(result, start);
        Release(id);
        return result.serial;
    }

    /// Commit thread: the ordered merge of segment `id` up to the encode (§5.6.3): take it in
    /// order, fill an unresolved result with `fallback`, apply its requests in order, patch the
    /// records' proxies. A caller that encodes the records one at a time later (the split: one
    /// per draw method) consumes the result until it calls Release(id).
    template <typename Fallback, typename Apply>
    Result& Merge(u64 id, Fallback&& fallback, Apply&& apply) {
        Result& result = Take(id);
        NoteOrder(id);
        if (streaming_id == id) {
            // The shard still writes the result: the caller reads its records through the
            // segment's progress until FinishStream (they carry no requests before R4).
            return result;
        }
        if (!result.resolved) {
            fallback(result);
            result.resolved = true;
        }
        stage.Apply(result.requests, apply);
        for (Record& record : result.records) {
            Traits::Patch(record, result.requests);
        }
        return result;
    }

    /// Commit thread: whether Take streams segment `id` (its shard still running).
    [[nodiscard]] bool Streaming(u64 id) const {
        return streaming_id == id;
    }

    /// Commit thread, before a streamed segment's result is read or released: stops its shard
    /// and waits until it has left (its end is published by then: the commit is past it, so the
    /// shard waits for no front). Verify compares what the shard resolved with a serial resolve.
    /// Returns the time waited (0 when `id` is not streamed).
    u64 FinishStream(u64 id) {
        if (streaming_id != id) {
            return 0;
        }
        streaming_id = 0;
        Slot& slot = slots[id & mask];
        const u64 start = NowNs();
        if constexpr (requires { slot.input.progress->abandon; }) {
            slot.input.progress->abandon.store(true, std::memory_order_release);
        }
        for (u32 spins = 0;
             StateOf(slot.word.load(std::memory_order_acquire)) == State::Running; ++spins) {
            if (spins < 4096) {
                CpuRelax();
            } else {
                std::this_thread::yield();
            }
        }
        const u64 waited = NowNs() - start;
        Result& result = slot.result;
        result.has_input = true;
        result.wait_ns += waited;
        if (config.stats) {
            Count(GetStats().stream_finish_ns, waited);
        }
        if (config.verify && result.serial == SerialReason::None) {
            VerifyHere(slot);
        }
        return waited;
    }

    /// Commit thread: the segment is consumed; its slot goes to segment id + depth.
    void Release(u64 id) {
        FinishStream(id);
        Slot& slot = slots[id & mask];
        slot.input = Input{};
        slot.word.store(Pack(id + slots.size(), State::Free), std::memory_order_release);
        work.fetch_add(1, std::memory_order_release);
        work.notify_one();
    }

    /// Commit thread, for callers that consume Take() themselves: the determinism check (segments
    /// must be committed in id order) and the stats of one committed segment.
    void NoteOrder(u64 id) {
        if (config.stats) {
            Count(GetStats().determinism_checks);
        }
        // Fix pass 1: the first segment sets the order. next_commit began at 0 while the
        // segmenter's ids start at 1, which counted one out-of-order commit per run.
        if (committed_any && id != next_commit) {
            Count(GetStats().determinism_failures);
        }
        committed_any = true;
        next_commit = id + 1;
    }
    void NoteCommitted(const Result& result, u64 start_ns) {
        if (!config.stats) {
            return;
        }
        Stats& stats = GetStats();
        Count(stats.segments);
        Count(stats.records, result.records.size());
        if (result.by_shard && result.serial == SerialReason::None) {
            Count(stats.by_shard);
        } else {
            Count(stats.serial[static_cast<size_t>(result.serial == SerialReason::None
                                                       ? SerialReason::Stolen
                                                       : result.serial)]);
        }
        if (start_ns != 0) {
            Count(stats.commit_ns, NowNs() - start_ns);
        }
    }

    [[nodiscard]] Shard& CommitShard() {
        return commit_shard;
    }
    [[nodiscard]] const Config& GetConfig() const {
        return config;
    }
    [[nodiscard]] Commit::Stage& CommitStage() {
        return stage;
    }
    /// The input of a segment the commit took (valid until Release).
    [[nodiscard]] const Input& InputOf(u64 id) const {
        return slots[id & mask].input;
    }

private:
    enum class State : u64 { Free, Queued, SerialQueued, Running, Done, Stolen };

    struct alignas(64) Slot {
        std::atomic<u64> word{};
        Input input{};
        SerialReason reason = SerialReason::None;
        Result result;
    };

    static constexpr u64 Pack(u64 id, State state) {
        return (id << 3) | static_cast<u64>(state);
    }
    static constexpr u64 IdOf(u64 word) {
        return word >> 3;
    }
    static constexpr State StateOf(u64 word) {
        return static_cast<State>(word & 7);
    }

    static void ResetResult(Result& result, u64 id) {
        result.id = id;
        result.records.clear();
        result.requests.Reset(id);
        result.arena.Clear();
        result.serial = SerialReason::None;
        result.by_shard = false;
        result.resolved = false;
        result.waited = false;
        result.wait_ns = 0;
        result.stolen = false;
        result.has_input = false;
    }

    void StealEmpty(Slot& slot, u64 id, SerialReason reason) {
        Result& result = slot.result;
        ResetResult(result, id);
        result.serial = reason;
        result.stolen = true;
    }

    static SerialReason RunResolve(Shard& shard, const Input& input, std::vector<Record>& records,
                                   Commit::RequestList& requests, Arena& arena) {
        if constexpr (requires { typename Traits::Arena; }) {
            return Traits::Resolve(shard, input, records, requests, arena);
        } else {
            static_cast<void>(arena);
            return Traits::Resolve(shard, input, records, requests);
        }
    }

    void ResolveHere(Slot& slot) {
        Result& result = slot.result;
        result.records.clear();
        result.requests.Reset(result.id);
        result.arena.Clear();
        result.by_shard = false;
        if (!config.resolve_stolen) {
            result.resolved = false;
            return;
        }
        const u64 start = config.stats ? NowNs() : 0;
        const SerialReason found = RunResolve(commit_shard, slot.input, result.records,
                                              result.requests, result.arena);
        static_cast<void>(found); // serial on the commit thread is the fallback itself
        result.resolved = true;
        if (config.stats) {
            Count(GetStats().stolen_resolve_ns, NowNs() - start);
        }
    }

    void VerifyHere(Slot& slot) {
        Result& result = slot.result;
        verify_records.clear();
        verify_requests.Reset(result.id);
        verify_arena.Clear();
        RunResolve(commit_shard, slot.input, verify_records, verify_requests, verify_arena);
        Count(GetStats().verify_segments);
        const RequestCompare requests =
            CompareRequests(result.requests.Requests(), verify_requests.Requests());
        Count(GetStats().verify_dependent, requests.dependent);
        const char* field = requests.field;
        size_t index = 0;
        // F1-36: an abandoned early segment holds the prefix its shard resolved before the
        // commit reached it (the serial resolve stops at what is published by now): only the
        // common prefix is compared.
        bool partial = false;
        if constexpr (requires { slot.input.progress->abandon; }) {
            partial = slot.input.progress != nullptr &&
                      slot.input.progress->abandon.load(std::memory_order_acquire);
        }
        if (partial) {
            const auto common = static_cast<std::ptrdiff_t>(
                (std::min)(result.records.size(), verify_records.size()));
            result.records.erase(result.records.begin() + common, result.records.end());
            verify_records.erase(verify_records.begin() + common, verify_records.end());
        }
        if constexpr (requires(const Record& r) { r.ordinal; }) {
            // Records pair up by their draw ordinal: a draw only one side kept is reported as
            // that, not as a difference in every draw after it.
            size_t other = 0;
            while (field == nullptr && index < result.records.size() &&
                   other < verify_records.size()) {
                const auto lhs = result.records[index].ordinal;
                const auto rhs = verify_records[other].ordinal;
                if (lhs != rhs) {
                    field = "draw kept by one side only";
                    break;
                }
                field = Traits::Compare(result.records[index], verify_records[other]);
                if (field == nullptr) {
                    ++index;
                    ++other;
                }
            }
            if (field == nullptr &&
                (index != result.records.size() || other != verify_records.size())) {
                field = "record count";
            }
        } else {
            if (field == nullptr && verify_records.size() != result.records.size()) {
                field = "record count";
            }
            for (; field == nullptr && index < verify_records.size(); ++index) {
                field = Traits::Compare(result.records[index], verify_records[index]);
                if (field != nullptr) {
                    break;
                }
            }
        }
        if (field != nullptr) {
            Count(GetStats().verify_mismatches);
            if (ShouldLogMismatch()) {
                LogMismatch(result.id, index, field);
            }
            // The serial resolve decides (as every v2 verify mode). The arenas go with their
            // records (elements never move, so the records' pointers stay valid).
            std::swap(result.records, verify_records);
            std::swap(result.requests, verify_requests);
            std::swap(result.arena, verify_arena);
        }
    }

    static void LogMismatch(u64 segment, size_t record, const char* field);

    void ShardLoop(std::stop_token stop, u32 index) {
        NameShardThread(index);
        Shard& shard = shard_states[index];
        while (!stop.stop_requested()) {
            // Read before looking for work, so a submission after the look ends the wait.
            const u32 seen = work.load(std::memory_order_acquire);
            u64 cursor = claim.load(std::memory_order_acquire);
            Slot& slot = slots[cursor & mask];
            u64 word = slot.word.load(std::memory_order_acquire);
            if (word == Pack(cursor, State::Queued)) {
                if (!claim.compare_exchange_weak(cursor, cursor + 1, std::memory_order_acq_rel)) {
                    continue;
                }
                if (!slot.word.compare_exchange_strong(word, Pack(cursor, State::Running),
                                                       std::memory_order_acq_rel)) {
                    continue; // the commit stole it
                }
                const u64 start = NowNs();
                u64 waited_before = 0;
                if constexpr (requires { shard.wait_ns; }) {
                    waited_before = shard.wait_ns;
                }
                Result& result = slot.result;
                ResetResult(result, cursor);
                result.serial = RunResolve(shard, slot.input, result.records, result.requests,
                                           result.arena);
                result.by_shard = result.serial == SerialReason::None;
                result.resolved = result.by_shard;
                if (config.stats) {
                    // F6-01: resolve time only; a shard waiting for the front is idle.
                    u64 waited = 0;
                    if constexpr (requires { shard.wait_ns; }) {
                        waited = shard.wait_ns - waited_before;
                    }
                    const u64 elapsed = NowNs() - start;
                    Count(GetStats().shard_busy_ns[index], elapsed - (std::min)(elapsed, waited));
                    Count(GetStats().shard_segments[index]);
                }
                slot.word.store(Pack(cursor, State::Done), std::memory_order_release);
                continue;
            }
            const u64 id = IdOf(word);
            bool skip = id > cursor || (id == cursor && StateOf(word) != State::Free);
            if (!skip && id == cursor && attempted.load(std::memory_order_acquire) > cursor) {
                // The producer is done with this id; a submission it made is visible now.
                if (slot.word.load(std::memory_order_acquire) != word) {
                    continue;
                }
                skip = true; // refused
            }
            if (skip) {
                // Serial, stolen, taken by another shard, refused, or long gone.
                claim.compare_exchange_weak(cursor, cursor + 1, std::memory_order_acq_rel);
                continue;
            }
            // Not submitted yet (or the window is full): wait for the producer or the commit.
            bool changed = false;
            for (u32 spin = 0; spin < 256; ++spin) {
                CpuRelax();
                if (work.load(std::memory_order_acquire) != seen) {
                    changed = true;
                    break;
                }
            }
            if (!changed && !stop.stop_requested()) {
                work.wait(seen, std::memory_order_acquire);
            }
        }
    }

    static void NameShardThread(u32 index);

    Config config;
    u64 mask;
    std::vector<Slot> slots;
    alignas(64) std::atomic<u64> claim{0};
    alignas(64) std::atomic<u64> attempted{0};
    alignas(64) std::atomic<u32> work{0};
    Shard commit_shard{};
    std::vector<Shard> shard_states;
    std::vector<std::jthread> threads;
    Commit::Stage stage;
    u64 next_commit = 0;
    bool committed_any = false; ///< The first commit starts the order (segment ids start at 1)
    u64 streaming_id = 0;       ///< F1-36: the segment Take streams (0: none; ids start at 1)
    std::vector<Record> verify_records;
    Commit::RequestList verify_requests;
    Arena verify_arena;
};

void LogSegmentMismatch(u64 segment, size_t record, const char* field);
void NameThread(u32 index);

template <typename Traits>
void ParallelResolver<Traits>::LogMismatch(u64 segment, size_t record, const char* field) {
    LogSegmentMismatch(segment, record, field);
}
template <typename Traits>
void ParallelResolver<Traits>::NameShardThread(u32 index) {
    NameThread(index);
}

} // namespace VideoCommon::V2::Resolve
