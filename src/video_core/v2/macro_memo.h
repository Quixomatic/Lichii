// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/macro_turing.h"

// Pipeline v2 F5: NVN2 macro call statistics and memoisation on the split front (MASTER_PLAN
// §5.1: "memoisation keyed by (macro, args, relevant register fingerprint)").
//
// A call is keyed by its start and parameter words; the memo is cleared when the macro code
// changes (its generation). A miss runs the interpreter, which records what the call read and
// did: the registers it read (MME STATE), the data RAM words it read and wrote, its method
// writes in order and the words its MME_DMA_READ_FIFOED reads returned. A later call with the
// same key whose register and data RAM reads still hold replays the writes instead of running
// the JIT. A DMA read is made again at its place in the replay and must return the recorded
// words; when it doesn't, the interpreter finishes the call from that write (the writes before
// it are the same: the macro is deterministic in its inputs, and those it read are served from
// the record). Calls are not stored when a register they read may have been changed by their
// own writes, when they read data RAM they wrote, or when they DMA into data RAM.
// EDEN_NX2_V2_MACROMEMO=verify runs the JIT on every hit instead and compares its writes with
// the record.
namespace VideoCommon::V2::MacroMemo {

using Result = Tegra::TuringMacroInterpreter::Result;
using Pairs = std::vector<std::pair<u32, u32>>;

/// What a recording run saw.
struct Trace {
    struct Read {
        u32 address;
        u32 value;
        u32 writes_before; ///< Method writes the call made before this read
    };
    struct Dma {
        u32 write_index; ///< The write that started the read
        u32 words;
    };
    std::vector<Read> reads;
    Pairs writes;
    std::vector<Dma> dma;
    std::vector<u32> dma_words;
    bool memoisable = true;

    void Clear() {
        reads.clear();
        writes.clear();
        dma.clear();
        dma_words.clear();
        memoisable = true;
    }
};

/// A stored call; the spans point into the memo and stay valid until the next Store.
struct Record {
    std::span<const u32> reads;       ///< (address, value, writes_before)
    std::span<const u32> dram_reads;  ///< (index, value)
    std::span<const u32> dram_writes; ///< (index, value) in order
    std::span<const u32> writes;      ///< (method, value) in order
    std::span<const u32> dma;         ///< (write index, words)
    std::span<const u32> dma_words;
};

struct Statistics {
    u64 calls{};
    u64 hits{};           ///< Replayed (or, verify, checked) calls
    u64 slot_hits{};      ///< Records found as the macro's last one (no hash probe)
    u64 misses{};         ///< No record for the key
    u64 guard_failures{}; ///< A record whose register or data RAM reads no longer hold
    u64 unmemoisable{};
    u64 bypassed{}; ///< Calls of macros the memo gave up on (JIT only)
    u64 stores{};
    u64 refetches{}; ///< Hits whose DMA read returned other words
    u64 verified{};
    u64 verify_mismatches{};
    u64 clears{};
};

/// The key's hash (also the exact-repeat statistic's).
[[nodiscard]] u64 Hash(u32 start, std::span<const u32> parameters);

class Memo {
public:
    Memo();

    /// Forgets every call when the macro code is no longer the code they ran.
    void Synchronize(u64 code_generation);
    /// Whether calls of the macro at `start` go through the memo: not when it keeps missing or
    /// its calls can't be stored (the interpreter's recording costs more than the JIT).
    [[nodiscard]] bool Wanted(u32 start) const;
    /// An earlier call with this key whose reads still hold (`regs[address]`, or 0 past
    /// `num_regs`, as the engines' MME STATE reads; `dram` words).
    [[nodiscard]] bool Find(u32 start, std::span<const u32> parameters, const u32* regs,
                            size_t num_regs, std::span<const u32> dram, Record& record);
    void Store(u32 start, std::span<const u32> parameters, const Trace& recorded,
               std::span<const std::pair<u32, u32>> dram_reads,
               std::span<const std::pair<u32, u32>> dram_writes);
    /// After each call through the memo: decides Wanted.
    void NoteOutcome(u32 start, bool hit, bool stored);

    [[nodiscard]] size_t Size() const {
        return entries.size();
    }

    Statistics statistics;
    Trace trace;                     ///< The recording run's (reused)
    std::vector<u32> appended;       ///< Words each replayed DMA read appended (reused)
    Pairs captured;                  ///< Verify: the JIT's writes (reused)
    std::vector<u32> captured_words; ///< Verify: its DMA words (reused)

private:
    struct Identity {
        using is_avalanching = std::true_type;
        size_t operator()(u64 key) const noexcept {
            return static_cast<size_t>(key);
        }
    };
    static constexpr u32 NoRecord = ~0U;
    struct StartInfo {
        u32 start{~0U};
        u32 last{NoRecord}; ///< Offset of the last record found or stored
        u32 hits{};
        u32 other{};    ///< Calls that were no hit
        u32 unstored{}; ///< Unmemoisable calls in a row
        bool bypass{};
    };
    static constexpr size_t HeaderWords = 8;
    static constexpr size_t MaxArenaWords = size_t{1} << 22;

    [[nodiscard]] bool Match(u32 offset, u32 start, std::span<const u32> parameters,
                             const u32* regs, size_t num_regs, std::span<const u32> dram,
                             Record& record) const;
    [[nodiscard]] static size_t RecordWords(const u32* header);
    void Clear();
    void Compact();

    std::unordered_map<u64, u32, Identity> entries;
    std::vector<u32> arena;
    std::vector<u32> compacted;
    std::array<StartInfo, 256> starts{};
    u64 generation{~u64{0}};
};

/// One call through the memo.
struct CallInfo {
    u32 start;
    std::vector<u32>* fifo;    ///< The parameters; DMA reads append to it
    std::span<u32> dram;       ///< The interpreter's active data RAM
    std::span<const u32> code; ///< The macro code (three words per instruction)
    const u64* generation;     ///< The engine's code generation (may change under a call)
    const u32* regs;           ///< What MME STATE reads
    size_t num_regs;
    /// Registers a method write can change besides its own (constant buffer offset, ...).
    std::span<const u32> side_effects;
    u32 dma_fifoed; ///< MME_DMA_READ_FIFOED's method (~0: none)
    u32 dma_ram;    ///< MME_DMA_READ's method (~0: none)
    bool verify;
};

namespace Detail {
void LogVerifyMismatch(u32 start, std::span<const u32> record_writes,
                       const Pairs& captured_writes);
void LogRecordFailure(u32 start, std::string_view error);
} // namespace Detail

/// Runs the interpreter for a call, recording it, with the first `prefix` writes already made
/// (their reads served from `served`, their DMA words already in the FIFO, `prefix_words`
/// each). Stores the call if it can be.
template <typename Read, typename Write, typename Dma>
Result RecordCall(Memo& memo, Tegra::TuringMacroInterpreter& interpreter, const CallInfo& call,
                  size_t num_parameters, size_t prefix, std::span<const u32> served,
                  std::span<const u32> prefix_words, Read& read, Write& write, Dma& dma) {
    Trace& trace = memo.trace;
    trace.Clear();
    std::vector<u32>& fifo = *call.fifo;
    const u64 generation = *call.generation;
    u32 writes_seen = 0;
    size_t read_index = 0;
    size_t prefix_dma = 0;
    const auto read_fn = [&](u32 address) -> u32 {
        u32 value;
        const size_t served_at = read_index * 3;
        if (served_at + 2 < served.size() && served[served_at + 2] < prefix &&
            served[served_at] == address) {
            value = served[served_at + 1]; // as it was before the made writes
        } else {
            value = read(address);
        }
        ++read_index;
        if (writes_seen != 0 && trace.memoisable) {
            // Read after the call's own writes: its value is not the one before the call when
            // they wrote it.
            const bool written =
                std::ranges::find(trace.writes, address, &std::pair<u32, u32>::first) !=
                    trace.writes.end() ||
                std::ranges::find(call.side_effects, address) != call.side_effects.end();
            trace.memoisable = !written;
        }
        trace.reads.push_back({address, value, writes_seen});
        return value;
    };
    const auto emit_fn = [&](u32 method, u32 value) -> bool {
        const u32 index = writes_seen++;
        const bool made = index < prefix;
        if (!made && !write(method, value)) {
            return false;
        }
        trace.writes.emplace_back(method, value);
        if (method == call.dma_fifoed || method == call.dma_ram) {
            trace.memoisable &= method == call.dma_fifoed;
            if (made) {
                // Its words follow the parameters and the earlier reads' words in the FIFO.
                const u32 words = prefix_dma < prefix_words.size() ? prefix_words[prefix_dma] : 0;
                size_t first = num_parameters;
                for (size_t i = 0; i < prefix_dma; ++i) {
                    first += prefix_words[i];
                }
                ++prefix_dma;
                const size_t end = std::min(first + words, fifo.size());
                trace.dma_words.insert(trace.dma_words.end(), fifo.begin() + first,
                                       fifo.begin() + end);
                trace.dma.push_back({index, words});
            } else {
                const size_t before = fifo.size();
                dma(method, value);
                trace.dma_words.insert(trace.dma_words.end(), fifo.begin() + before, fifo.end());
                trace.dma.push_back({index, static_cast<u32>(fifo.size() - before)});
            }
        }
        return true;
    };
    // The JIT reads never-written MME registers as 0; so does the recording.
    interpreter.allow_uninitialized = true;
    const Result result = interpreter.Execute(call.code, call.start, fifo, read_fn, emit_fn);
    interpreter.allow_uninitialized = false;
    const bool stored = result.success && trace.memoisable && !interpreter.reread_data_ram &&
                        *call.generation == generation;
    if (stored) {
        memo.Store(call.start, std::span<const u32>{fifo.data(), num_parameters}, trace,
                   interpreter.data_ram_reads, interpreter.data_ram_writes);
    } else if (result.success) {
        ++memo.statistics.unmemoisable;
    } else {
        Detail::LogRecordFailure(call.start, result.error);
    }
    memo.NoteOutcome(call.start, false, stored);
    return result;
}

/// A macro call through the memo. `live(emit)` runs it the old way (the JIT) with `emit`;
/// `read(address)` is the MME STATE read; `write(method, value)` the front's write of a macro
/// method (false: not one); `dma(method, value)` the MME DMA read that write starts.
template <typename Live, typename Read, typename Write, typename Dma>
Result Call(Memo& memo, Tegra::TuringMacroInterpreter& interpreter, const CallInfo& call,
            Live&& live, Read&& read, Write&& write, Dma&& dma) {
    std::vector<u32>& fifo = *call.fifo;
    ++memo.statistics.calls;
    const auto live_emit = [&](u32 method, u32 value) {
        if (!write(method, value)) {
            return false;
        }
        if (method == call.dma_fifoed || method == call.dma_ram) {
            dma(method, value);
        }
        return true;
    };
    if (!memo.Wanted(call.start)) {
        ++memo.statistics.bypassed;
        return live(live_emit);
    }
    const size_t num_parameters = fifo.size();
    Record record;
    if (!memo.Find(call.start, fifo, call.regs, call.num_regs, call.dram, record)) {
        return RecordCall(memo, interpreter, call, num_parameters, 0, {}, {}, read, write, dma);
    }
    if (call.verify) {
        // The JIT makes the writes; the record must have predicted them.
        memo.captured.clear();
        memo.captured_words.clear();
        const auto capture = [&](u32 method, u32 value) {
            if (!write(method, value)) {
                return false;
            }
            memo.captured.emplace_back(method, value);
            if (method == call.dma_fifoed || method == call.dma_ram) {
                const size_t before = fifo.size();
                dma(method, value);
                memo.captured_words.insert(memo.captured_words.end(), fifo.begin() + before,
                                           fifo.end());
            }
            return true;
        };
        const Result result = live(capture);
        ++memo.statistics.hits;
        ++memo.statistics.verified;
        const bool same_words = std::ranges::equal(memo.captured_words, record.dma_words);
        bool same = result.success && memo.captured.size() * 2 == record.writes.size();
        for (size_t i = 0; same && i < memo.captured.size(); ++i) {
            same = memo.captured[i].first == record.writes[i * 2] &&
                   memo.captured[i].second == record.writes[i * 2 + 1];
        }
        // Data RAM: the last value the record wrote at each word.
        for (size_t i = 0; same && i < record.dram_writes.size(); i += 2) {
            bool last = true;
            for (size_t j = i + 2; j < record.dram_writes.size() && last; j += 2) {
                last = record.dram_writes[j] != record.dram_writes[i];
            }
            same = !last || call.dram[record.dram_writes[i]] == record.dram_writes[i + 1];
        }
        if (!same_words) {
            ++memo.statistics.refetches; // other inputs: the replay would have left the record
        } else if (!same) {
            ++memo.statistics.verify_mismatches;
            Detail::LogVerifyMismatch(call.start, record.writes, memo.captured);
        }
        memo.NoteOutcome(call.start, true, false);
        return result;
    }
    // Replay. A DMA read is made where the call made it and must return the recorded words.
    memo.appended.clear();
    const size_t num_writes = record.writes.size() / 2;
    size_t next_dma = 0;
    size_t word_offset = 0;
    for (size_t i = 0; i < num_writes; ++i) {
        const u32 method = record.writes[i * 2];
        const u32 value = record.writes[i * 2 + 1];
        write(method, value);
        if (next_dma * 2 >= record.dma.size() || record.dma[next_dma * 2] != i) {
            continue;
        }
        const u32 words = record.dma[next_dma * 2 + 1];
        const size_t before = fifo.size();
        dma(method, value);
        const size_t got = fifo.size() - before;
        memo.appended.push_back(static_cast<u32>(got));
        const bool same =
            got == words &&
            std::equal(fifo.begin() + before, fifo.end(), record.dma_words.begin() + word_offset);
        if (!same) {
            // The rest depends on the new words: the interpreter finishes the call (the record's
            // reads are copied first, Store may move them).
            ++memo.statistics.refetches;
            const std::vector<u32> served{record.reads.begin(), record.reads.end()};
            const std::vector<u32> prefix_words = memo.appended;
            return RecordCall(memo, interpreter, call, num_parameters, i + 1, served,
                              prefix_words, read, write, dma);
        }
        word_offset += words;
        ++next_dma;
    }
    for (size_t i = 0; i < record.dram_writes.size(); i += 2) {
        call.dram[record.dram_writes[i]] = record.dram_writes[i + 1];
    }
    ++memo.statistics.hits;
    memo.NoteOutcome(call.start, true, false);
    return Result{true, call.start, num_parameters, {}};
}

/// EDEN_NX2_V2_STATS: calls per macro and how many repeat a recent call exactly (same start
/// and parameters), logged every 10 s with the memo's counters.
class CallStats {
public:
    /// `engine` names the log line; `code` identifies new starts.
    void Note(const char* engine, u32 start, std::span<const u32> parameters,
              std::span<const u32> code, u64 code_generation, const Statistics* memo);

private:
    struct Counts {
        u64 hash{};
        u64 calls{};
        u64 repeats{};
    };
    void Fold();
    void Report(const char* engine, const Statistics* memo, double seconds);

    std::unordered_map<u32, Counts> by_start;
    std::unordered_map<u64, Counts> by_hash; ///< Folded at code changes and reports
    std::array<u64, 4096> recent{};
    u64 generation{~u64{0}};
    u64 calls{};
    u64 repeats{};
    u64 window_start_ns{};
    Statistics memo_start{};
};

} // namespace VideoCommon::V2::MacroMemo
