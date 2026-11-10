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
#include "video_core/v2/change_clock.h"

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
    u32 offset; ///< In the memo's arena
    u32 info;   ///< RecordInfo bits
    std::span<const u32> guards;      ///< (address, value): each register read once, in range
    std::span<const u32> reads;       ///< (address, value, writes_before); calls with DMA only
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
    u64 jit_records{}; ///< Recordings the JIT made (code that can't touch data RAM)
    u64 refetches{}; ///< Hits whose DMA read returned other words
    u64 verified{};
    u64 verify_mismatches{};
    u64 clears{};
    u64 fingerprint_hits{}; ///< Fix F3d: guards taken from the change clock, not compared
    u64 noop_hits{};        ///< Fix F3d: hits that made only their executable writes
    u64 skipped_writes{};   ///< Writes those hits left out (their registers held them)
    u64 cost_bypasses{};    ///< Windows whose estimated cost exceeded the JIT's
};

/// CallInfo::context when no no-op replay is possible.
inline constexpr u32 NoContext = ~0U;

/// The key's hash (also the exact-repeat statistic's).
[[nodiscard]] u64 Hash(u32 start, std::span<const u32> parameters);

struct CallInfo;

/// How a call through the memo went (NoteOutcome's cost estimate).
enum class Outcome : u8 {
    Hit,        ///< Replayed (or verified)
    Recorded,   ///< Miss or guard failure, recorded and stored
    Unstorable, ///< Recorded but not storable
};

/// Fix F3F5b-3 (report 7833a32: Find 5.5% + Match 1.5% of the front, 41k guard failures/s):
/// the key (start, parameters) is hashed once per call; records sit in a direct-mapped table of
/// two-way sets indexed by it (no map, no scan), each way holding the full key as its
/// fingerprint, so a probe compares two keys and then only the record whose key matched; Match
/// compares the parameters and the deduplicated registers the call read (its guards), not the
/// read trace. Two ways keep two register contexts of one key (a macro alternating between two
/// states no longer re-records on every call). Misses of code that can't touch data RAM are
/// recorded by the JIT; macros whose records keep failing are left to the JIT for a while.
class Memo {
public:
    Memo();

    /// Forgets every call when the macro code is no longer the code they ran.
    void Synchronize(u64 code_generation);
    /// Whether calls of the macro at `start` (in `code`) go through the memo: not when its calls
    /// can't be stored, or while its hit rate doesn't pay for the recordings.
    [[nodiscard]] bool Wanted(u32 start, std::span<const u32> code);
    /// Whether misses of the macro at `start` may be recorded by the JIT (its code can't read or
    /// write data RAM: the JIT doesn't report those). Valid after Wanted.
    [[nodiscard]] bool JitRecordable(u32 start) const {
        const StartInfo& info = starts[start % starts.size()];
        return info.start == start && info.jit_recordable;
    }
    /// An earlier call with this key whose reads still hold (`call.regs[address]`, or 0 past
    /// `num_regs`, as the engines' MME STATE reads; data RAM words). `key`: Hash(start,
    /// parameters). Fix F3d: with a change clock, guards whose blocks didn't change since they
    /// were last known to hold are not compared (MACROMEMO=verify compares them anyway).
    [[nodiscard]] bool Find(u64 key, const CallInfo& call, Record& record);
    /// `fresh`: a whole recording (no prefix made before it).
    void Store(u64 key, const CallInfo& call, std::span<const u32> parameters,
               const Trace& recorded, std::span<const std::pair<u32, u32>> dram_reads,
               std::span<const std::pair<u32, u32>> dram_writes, bool fresh);
    /// After each call through the memo: decides Wanted from the estimated cost of its calls
    /// against the JIT's. `skipped`: writes a no-op hit left out.
    void NoteOutcome(u32 start, Outcome outcome, u32 skipped = 0, bool interpreter = false);
    /// Fix F3d: whether a hit of `record` may make only its executable writes (its other
    /// writes are those its registers and shadow copies hold since it was last applied).
    [[nodiscard]] bool NoopReplay(const Record& record, const CallInfo& call) const;
    /// Fix F3d: `record`'s writes were just applied (replayed or run): its state writes hold now,
    /// and so do its guards when its writes can't change them.
    void NoteApplied(const Record& record, const CallInfo& call);

    /// Records in the table.
    [[nodiscard]] size_t Size() const;

    /// Whether the macro at `start` may read or write data RAM (DREAD/DWRITE in any lane of
    /// the code it can reach; true when its branches leave the code it was scanned in).
    [[nodiscard]] static bool MayUseDataRam(std::span<const u32> code, u32 start);
    /// Instructions from `start` through the one after the first end bit (0: unknown).
    [[nodiscard]] static u32 StaticLength(std::span<const u32> code, u32 start);

    Statistics statistics;
    Trace trace;                     ///< The recording run's (reused)
    std::vector<u32> appended;       ///< Words each replayed DMA read appended (reused)
    Pairs captured;                  ///< Verify: the JIT's writes (reused)
    std::vector<u32> captured_words; ///< Verify: its DMA words (reused)
    std::vector<std::pair<u32, u32>> guard_scratch; ///< Store's deduplicated reads (reused)
    /// JIT recordings of code that touches data RAM: its reads and writes (reused).
    std::vector<std::pair<u32, u32>> dram_read_scratch;
    std::vector<std::pair<u32, u32>> dram_write_scratch;
    u32 last_skipped{}; ///< Writes the last call's no-op hit left out

    /// Record header words (Fix F3d after the counts): block masks and ticks of the change clock
    /// as two words each, then the RecordInfo bits and the context.
    static constexpr size_t GuardMask = 9;
    static constexpr size_t GuardTick = 11;
    static constexpr size_t WriteMask = 13;
    static constexpr size_t WriteTick = 15;
    static constexpr size_t InfoWord = 17;
    static constexpr size_t ContextWord = 18;
    static constexpr size_t HeaderWords = 19;
    /// RecordInfo bits.
    static constexpr u32 Idempotent = 1; ///< Each state register written once-valued, no barrier,
                                         ///< no side-effect register, no DMA
    static constexpr u32 GuardsKeep = 2; ///< No write can change a register the call read

    // Cost estimate per call (ns, MSVC Castle Town, F3d breakdown) [est]: a hit's lookup, a JIT
    // run (entry plus per static instruction), a write a no-op hit leaves out (the write
    // callback and the table compare), a recording by the JIT or the interpreter.
    static constexpr s64 HitNs = 30;
    static constexpr s64 JitBaseNs = 25;
    static constexpr s64 JitInstructionNs = 2;
    static constexpr s64 SkippedWriteNs = 12;
    static constexpr s64 JitRecordingNs = 90;
    static constexpr s64 InterpreterRecordingNs = 400;

private:
    static constexpr u32 NoRecord = ~0U;
    struct StartInfo {
        u32 start{~0U};
        u32 last{NoRecord}; ///< Offset of the last record found or stored
        u32 calls{};        ///< Calls through the memo in the current window
        u32 unstored{};     ///< Unmemoisable calls in a row
        u32 lifetime_hits{};
        u32 bypass_left{}; ///< Calls left to the JIT before the memo tries again
        u32 instructions{}; ///< Static length of the macro (JIT cost estimate)
        s64 saved_ns{};     ///< Estimated time the window's calls saved against the JIT
        bool never{};      ///< Gave up until the code changes (calls can't be stored)
        bool jit_recordable{};
    };
    struct Way {
        u64 key{};
        u32 offset{NoRecord};
    };
    static constexpr size_t Sets = 4096;
    static constexpr size_t Window = 2048;   ///< Calls per hit-rate decision
    static constexpr u32 BypassCalls = 32768; ///< Calls left to the JIT after a poor window
    // Header: start, parameters, guards, reads, data RAM reads, data RAM writes, writes,
    // DMA reads, DMA words; then the F3d words above.
    static constexpr size_t MaxArenaWords = size_t{1} << 22;

    [[nodiscard]] bool Match(u32 offset, const CallInfo& call, Record& record);
    [[nodiscard]] static u64 Get64(const u32* words) {
        return u64{words[0]} | (u64{words[1]} << 32);
    }
    static void Set64(u32* words, u64 value) {
        words[0] = static_cast<u32>(value);
        words[1] = static_cast<u32>(value >> 32);
    }
    [[nodiscard]] static size_t RecordWords(const u32* header);
    [[nodiscard]] StartInfo& Info(u32 start);
    void Clear();
    void Compact();

    std::vector<Way> ways; ///< Sets x 2; way 0 the most recently used of its set
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
    /// A register whose value in `regs` may be stale (~0: none): a record that read it is a
    /// guard failure, and the recording's `read` gets the current value.
    u32 stale_register = ~0U;
    /// Hash(start, parameters) when the caller has it (0: Call computes it).
    u64 key = 0;
    /// Fix F3d (null: the caller keeps none): when each block of `regs` and their shadow copies
    /// last changed. A record's guards that held at a tick still hold while their blocks are
    /// unchanged (the §5.1 register fingerprint, exact), and a hit whose state writes are still
    /// the last ones applied to their registers makes only its executable writes.
    const ChangeClock* clock = nullptr;
    /// With `clock`: a write is executable when `flags[method] & exec_mask` (it acts even with
    /// the value its register holds); a record writing a method with `flags & barrier_mask`
    /// (the shadow RAM mode, which changes what later writes do) is never a no-op replay.
    const u16* flags = nullptr;
    u16 exec_mask = 0;
    u16 barrier_mask = 0;
    /// What a state write needs besides its register to change nothing (the shadow RAM mode);
    /// NoContext: no no-op replay now.
    u32 context = NoContext;
    /// No-op replays: asked before the first and after each executable write; false sends
    /// the remaining writes through `write` (a deferred draw makes every write act).
    bool (*writes_skippable)(const void*) = nullptr;
    const void* skippable_arg = nullptr;
    /// MACROMEMO=verify: the shadow copies a skipped write must also hold (null: none).
    const u32* shadow = nullptr;
    /// `live(read, emit, dram_read, dram_write)` runs the JIT with data RAM hooks, so misses of
    /// code that touches data RAM are recorded by the JIT too.
    bool jit_data_ram = false;
};

namespace Detail {
void LogVerifyMismatch(u32 start, std::span<const u32> record_writes,
                       const Pairs& captured_writes);
void LogRecordFailure(u32 start, std::string_view error);
/// MACROMEMO=verify, Fix F3d: the change clock passed a guard that no longer holds, or a no-op
/// replay would have left out a write that changes its register.
void LogFastPathMismatch(u32 start, const char* what, u32 address, u32 expected, u32 found);
} // namespace Detail

/// Runs a call recording it, with the first `prefix` writes already made (their reads served
/// from `served`, their DMA words already in the FIFO, `prefix_words` each): the interpreter,
/// or `live(read, emit)` (the JIT) for a whole call of code that can't touch data RAM. Stores
/// the call if it can be.
template <typename Live, typename Read, typename Write, typename Dma>
Result RecordCall(Memo& memo, Tegra::TuringMacroInterpreter& interpreter, const CallInfo& call,
                  u64 key, size_t num_parameters, size_t prefix, std::span<const u32> served,
                  std::span<const u32> prefix_words, Live& live, Read& read, Write& write,
                  Dma& dma) {
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
    Result result;
    bool stored;
    bool by_interpreter = false;
    const std::span<const u32> parameters{fifo.data(), num_parameters};
    if (prefix == 0 && memo.JitRecordable(call.start)) {
        // No data RAM access in the code: the JIT's register reads and writes are all of it.
        ++memo.statistics.jit_records;
        result = live(read_fn, emit_fn);
        stored = result.success && trace.memoisable && *call.generation == generation;
        if (stored) {
            memo.Store(key, call, parameters, trace, {}, {}, true);
        }
    } else if (prefix == 0 && call.jit_data_ram) {
        // Fix F3d: code that may touch data RAM, recorded by the JIT through data RAM hooks
        // (the interpreter's lists: every read, in order, and a re-read of a written word).
        ++memo.statistics.jit_records;
        auto& dram_reads = memo.dram_read_scratch;
        auto& dram_writes = memo.dram_write_scratch;
        dram_reads.clear();
        dram_writes.clear();
        bool reread = false;
        const auto dram_read = [&](u32 index) -> u32 {
            const u32 value = call.dram[index];
            if (std::ranges::find(dram_writes, index, &std::pair<u32, u32>::first) !=
                dram_writes.end()) {
                reread = true;
            }
            dram_reads.emplace_back(index, value);
            return value;
        };
        const auto dram_write = [&](u32 index, u32 value) {
            call.dram[index] = value;
            dram_writes.emplace_back(index, value);
        };
        result = live(read_fn, emit_fn, dram_read, dram_write);
        stored = result.success && trace.memoisable && !reread && *call.generation == generation;
        if (stored) {
            memo.Store(key, call, parameters, trace, dram_reads, dram_writes, true);
        }
    } else {
        // The JIT reads never-written MME registers as 0; so does the recording.
        by_interpreter = true;
        interpreter.allow_uninitialized = true;
        result = interpreter.Execute(call.code, call.start, fifo, read_fn, emit_fn);
        interpreter.allow_uninitialized = false;
        stored = result.success && trace.memoisable && !interpreter.reread_data_ram &&
                 *call.generation == generation;
        if (stored) {
            memo.Store(key, call, parameters, trace, interpreter.data_ram_reads,
                       interpreter.data_ram_writes, prefix == 0);
        }
    }
    if (!stored && result.success) {
        ++memo.statistics.unmemoisable;
    } else if (!result.success) {
        Detail::LogRecordFailure(call.start, result.error);
    }
    memo.NoteOutcome(call.start, stored ? Outcome::Recorded : Outcome::Unstorable, 0,
                     by_interpreter);
    return result;
}

/// A macro call through the memo. `live(read, emit)` runs it the old way (the JIT) with those
/// callbacks; `read(address)` is the MME STATE read; `write(method, value)` the front's write of
/// a macro method (false: not one); `dma(method, value)` the MME DMA read that write starts.
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
    if (!memo.Wanted(call.start, call.code)) {
        ++memo.statistics.bypassed;
        return live(read, live_emit);
    }
    const size_t num_parameters = fifo.size();
    const u64 key = call.key != 0 ? call.key : Hash(call.start, fifo);
    Record record;
    memo.last_skipped = 0;
    if (!memo.Find(key, call, record)) {
        return RecordCall(memo, interpreter, call, key, num_parameters, 0, {}, {}, live, read,
                          write, dma);
    }
    if (call.verify) {
        if (memo.NoopReplay(record, call)) {
            // The writes a no-op hit would leave out must change nothing.
            for (size_t i = 0; i < record.writes.size(); i += 2) {
                const u32 method = record.writes[i];
                const u32 value = record.writes[i + 1];
                if ((call.flags[method] & call.exec_mask) != 0) {
                    continue;
                }
                const u32 held = call.regs[method];
                const bool kept = call.shadow == nullptr || call.shadow[method] == value;
                if (held != value || !kept) {
                    ++memo.statistics.verify_mismatches;
                    Detail::LogFastPathMismatch(call.start, "no-op replay", method, value,
                                                held != value ? held : call.shadow[method]);
                    break;
                }
            }
        }
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
        const Result result = live(read, capture);
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
        if (result.success && same_words) {
            memo.NoteApplied(record, call);
        }
        memo.NoteOutcome(call.start, Outcome::Hit);
        return result;
    }
    if (memo.NoopReplay(record, call)) {
        // Fix F3d: every state write of the record is what its register (and shadow copy) holds
        // since the record was last applied: Write3D would drop each one. Only the executable
        // writes act, until something (a deferred draw) makes every write count.
        ++memo.statistics.noop_hits;
        bool skip = call.writes_skippable == nullptr || call.writes_skippable(call.skippable_arg);
        u32 skipped = 0;
        for (size_t i = 0; i < record.writes.size(); i += 2) {
            const u32 method = record.writes[i];
            const u32 value = record.writes[i + 1];
            if ((call.flags[method] & call.exec_mask) != 0) {
                write(method, value);
                skip = skip && (call.writes_skippable == nullptr ||
                                call.writes_skippable(call.skippable_arg));
            } else if (skip) {
                ++skipped;
            } else {
                write(method, value);
            }
        }
        for (size_t i = 0; i < record.dram_writes.size(); i += 2) {
            call.dram[record.dram_writes[i]] = record.dram_writes[i + 1];
        }
        memo.last_skipped = skipped;
        memo.statistics.skipped_writes += skipped;
        ++memo.statistics.hits;
        memo.NoteApplied(record, call);
        memo.NoteOutcome(call.start, Outcome::Hit, skipped);
        return Result{true, call.start, num_parameters, {}};
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
            return RecordCall(memo, interpreter, call, key, num_parameters, i + 1, served,
                              prefix_words, live, read, write, dma);
        }
        word_offset += words;
        ++next_dma;
    }
    for (size_t i = 0; i < record.dram_writes.size(); i += 2) {
        call.dram[record.dram_writes[i]] = record.dram_writes[i + 1];
    }
    ++memo.statistics.hits;
    memo.NoteApplied(record, call);
    memo.NoteOutcome(call.start, Outcome::Hit);
    return Result{true, call.start, num_parameters, {}};
}

/// EDEN_NX2_V2_STATS: calls per macro and how many repeat a recent call exactly (same start
/// and parameters), logged every 10 s with the memo's counters.
class CallStats {
public:
    /// `engine` names the log line; `code` identifies new starts. `key`: Hash(start,
    /// parameters) when the caller has it (0: computed here).
    void Note(const char* engine, u32 start, std::span<const u32> parameters,
              std::span<const u32> code, u64 code_generation, const Statistics* memo,
              u64 key = 0);

private:
    struct Counts {
        u64 hash{};
        u64 calls{};
        u64 repeats{};
    };
    struct StartCounts {
        u32 start{~0U};
        Counts counts;
    };
    void Fold();
    void Report(const char* engine, const Statistics* memo, double seconds);

    /// Counts by start, direct-mapped (F3F5b-3: no map lookup per call); a start that takes
    /// another's slot folds the old one's counts first.
    std::array<StartCounts, 256> by_start{};
    std::unordered_map<u64, Counts> by_hash; ///< Folded at code changes and reports
    std::array<u64, 4096> recent{};
    u64 generation{~u64{0}};
    u64 calls{};
    u64 repeats{};
    u64 window_start_ns{};
    Statistics memo_start{};
};

} // namespace VideoCommon::V2::MacroMemo
