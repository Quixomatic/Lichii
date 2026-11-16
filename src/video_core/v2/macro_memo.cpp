// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <atomic>
#include <chrono>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/v2/macro_hle.h"
#include "video_core/v2/macro_memo.h"

namespace VideoCommon::V2::MacroMemo {

u64 Hash(u32 start, std::span<const u32> parameters) {
    u64 hash = 0x9e3779b97f4a7c15ULL ^ (u64{start} << 32) ^ parameters.size();
    for (const u32 word : parameters) {
        hash = (hash ^ word) * 0xff51afd7ed558ccdULL;
        hash ^= hash >> 32;
    }
    return hash;
}

Memo::Memo() : ways(Sets * 2) {}

void Memo::Synchronize(u64 code_generation) {
    if (code_generation != generation) {
        generation = code_generation;
        Clear();
        starts.fill({});
    }
}

Memo::StartInfo& Memo::Info(u32 start) {
    return starts[start % starts.size()];
}

u32 Memo::StaticLength(std::span<const u32> code, u32 start) {
    constexpr u32 MaxInstructions = 4096;
    bool end_next = false;
    for (u32 pc = start; pc < start + MaxInstructions && size_t{pc} * 3 + 2 < code.size(); ++pc) {
        if (end_next) {
            return pc - start + 1;
        }
        end_next = (code[size_t{pc} * 3 + 2] & 1) != 0;
    }
    return 0;
}

bool Memo::MayUseDataRam(std::span<const u32> code, u32 start) {
    // The instructions from `start` through the one after the first end bit (as Identify),
    // with every branch and loop target inside them; else the code it reaches is unknown.
    constexpr u32 MaxInstructions = 4096;
    u32 last = start;
    bool end_next = false;
    for (u32 pc = start;; ++pc) {
        if (pc >= start + MaxInstructions || size_t{pc} * 3 + 2 >= code.size()) {
            return true;
        }
        last = pc;
        if (end_next) {
            break;
        }
        end_next = (code[size_t{pc} * 3 + 2] & 1) != 0;
    }
    for (u32 pc = start; pc <= last; ++pc) {
        const std::array<u32, 3> bits{code[size_t{pc} * 3 + 2], code[size_t{pc} * 3 + 1],
                                      code[size_t{pc} * 3]};
        const auto d = Tegra::TuringMacroInterpreter::Decode(bits);
        for (u32 lane = 0; lane < 2; ++lane) {
            const u32 op = d.op[lane];
            const u32 imm = d.imm[lane];
            if (op == 30 || op == 31) {
                return true; // DREAD / DWRITE
            }
            s64 target = pc;
            if (op == 23) {
                target = s64{pc} + imm - 1; // loop end
            } else if (op == 24) {
                target = s64{pc} + (imm & 0x7fff);
            } else if (op >= 25 && op <= 29) {
                const s32 offset = static_cast<s32>(imm << 19) >> 19;
                target = offset == -4096 ? pc : s64{pc} + offset;
            }
            if (target < start || target > last) {
                return true;
            }
        }
    }
    return false;
}

bool Memo::Wanted(u32 start, std::span<const u32> code) {
    StartInfo& info = Info(start);
    if (info.start != start) [[unlikely]] {
        const u32 length = StaticLength(code, start);
        info = StartInfo{.start = start,
                         .instructions = length != 0 ? length : 64,
                         .jit_recordable = !MayUseDataRam(code, start)};
    }
    if (info.never) {
        return false;
    }
    if (info.bypass_left != 0) {
        --info.bypass_left;
        return false;
    }
    return true;
}

size_t Memo::RecordWords(const u32* header) {
    // start, parameters, guards, reads, data RAM reads, data RAM writes, writes, DMA reads, DMA
    // words.
    return HeaderWords + header[1] + size_t{2} * header[2] + size_t{3} * header[3] +
           size_t{2} * (header[4] + header[5] + header[6] + header[7]) + header[8];
}

bool Memo::Match(u32 offset, const CallInfo& call, Record& record) {
    const std::span<const u32> parameters = *call.fifo;
    const u32* const header = arena.data() + offset;
    if (header[0] != call.start || header[1] != parameters.size()) {
        return false;
    }
    // Fix F3e: one to four words in OoT; a loop, not a memcmp call.
    const u32* const stored = header + HeaderWords;
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (stored[i] != parameters[i]) {
            return false;
        }
    }
    const u32* it = stored + header[1];
    // Each register the call read, once (Store kept those below num_regs: the rest read 0):
    // the §5.1 register fingerprint, compared exactly.
    const u32* const guards_end = it + size_t{2} * header[2];
    for (const u32* guard = it; guard != guards_end; guard += 2) {
        if (call.regs[guard[0]] != guard[1] || guard[0] == call.stale_register) {
            return false;
        }
    }
    it = guards_end;
    if (header[4] != 0) [[unlikely]] {
        const u32* const dram_reads = it + size_t{3} * header[3];
        for (size_t i = 0; i < size_t{2} * header[4]; i += 2) {
            const u32 index = dram_reads[i];
            if (index >= call.dram.size() || call.dram[index] != dram_reads[i + 1]) {
                return false;
            }
        }
    }
    record.offset = offset;
    record.guards = {header + HeaderWords + header[1], size_t{2} * header[2]};
    record.reads = {it, size_t{3} * header[3]};
    it += record.reads.size();
    record.dram_reads = {it, size_t{2} * header[4]};
    it += record.dram_reads.size();
    record.dram_writes = {it, size_t{2} * header[5]};
    it += record.dram_writes.size();
    record.writes = {it, size_t{2} * header[6]};
    it += record.writes.size();
    record.dma = {it, size_t{2} * header[7]};
    it += record.dma.size();
    record.dma_words = {it, header[8]};
    return true;
}

bool Memo::Find(u64 key, const CallInfo& call, Record& record) {
    StartInfo& info = Info(call.start);
    // Many calls repeat the macro's previous call: try its record before the table (Fix F3e:
    // only for its key, so another key's call reads no arena words here).
    if (info.last != NoRecord && info.last_key == key && Match(info.last, call, record)) {
        ++statistics.slot_hits;
        return true;
    }
    Way* const set = ways.data() + (key & (Sets - 1)) * 2;
    bool keyed = false;
    for (size_t way = 0; way < 2; ++way) {
        if (set[way].key != key || set[way].offset == NoRecord) {
            continue;
        }
        keyed = true;
        if ((set[way].offset != info.last || info.last_key != key) &&
            Match(set[way].offset, call, record)) {
            info.last = set[way].offset;
            info.last_key = key;
            if (way != 0) {
                std::swap(set[0], set[1]); // way 0: the most recently used
            }
            return true;
        }
    }
    ++(keyed ? statistics.guard_failures : statistics.misses);
    return false;
}

void Memo::Store(u64 key, const CallInfo& call, std::span<const u32> parameters,
                 const Trace& recorded, std::span<const std::pair<u32, u32>> dram_reads,
                 std::span<const std::pair<u32, u32>> dram_writes) {
    const u32 start = call.start;
    const size_t num_regs = call.num_regs;
    ++statistics.stores;
    // Guards: each register read before any write of the call, once (a read after its own
    // write made the call unmemoisable), within the register file (the rest read as 0).
    guard_scratch.clear();
    for (const Trace::Read& read : recorded.reads) {
        if (read.address < num_regs &&
            std::ranges::find(guard_scratch, read.address, &std::pair<u32, u32>::first) ==
                guard_scratch.end()) {
            guard_scratch.emplace_back(read.address, read.value);
        }
    }
    // The full read trace only serves an interpreter finish after a refetched DMA read.
    const size_t reads = recorded.dma.empty() ? 0 : recorded.reads.size();
    const std::array<u32, HeaderWords> header{
        start,
        static_cast<u32>(parameters.size()),
        static_cast<u32>(guard_scratch.size()),
        static_cast<u32>(reads),
        static_cast<u32>(dram_reads.size()),
        static_cast<u32>(dram_writes.size()),
        static_cast<u32>(recorded.writes.size()),
        static_cast<u32>(recorded.dma.size()),
        static_cast<u32>(recorded.dma_words.size()),
    };
    const size_t words = RecordWords(header.data());
    // The least recently used way takes it (way 0 is the most recently used): a key whose
    // context just failed keeps that one too, so a macro alternating between two register
    // contexts hits both.
    Way* set = ways.data() + (key & (Sets - 1)) * 2;
    const size_t way = set[0].offset == NoRecord ? 0 : 1;
    u32 offset;
    // A record replaced by one of the same size is rewritten in place (no stale copies).
    if (set[way].offset != NoRecord && RecordWords(arena.data() + set[way].offset) == words) {
        offset = set[way].offset;
    } else {
        if (arena.size() + words > MaxArenaWords) {
            Compact();
            set = ways.data() + (key & (Sets - 1)) * 2;
        }
        offset = static_cast<u32>(arena.size());
        arena.resize(arena.size() + words);
    }
    // A StartInfo::last left on a replaced record stays harmless: Match checks its start and
    // parameters, and a record left in the arena still holds for the code it was made with.
    set[way] = Way{key, offset};
    if (way != 0) {
        std::swap(set[0], set[1]);
    }
    u32* out = std::copy(header.begin(), header.end(), arena.data() + offset);
    out = std::copy(parameters.begin(), parameters.end(), out);
    for (const auto& [address, value] : guard_scratch) {
        *out++ = address;
        *out++ = value;
    }
    for (size_t i = 0; i < reads; ++i) {
        *out++ = recorded.reads[i].address;
        *out++ = recorded.reads[i].value;
        *out++ = recorded.reads[i].writes_before;
    }
    for (const auto pairs : {dram_reads, dram_writes}) {
        for (const auto& [index, value] : pairs) {
            *out++ = index;
            *out++ = value;
        }
    }
    for (const auto& [method, value] : recorded.writes) {
        *out++ = method;
        *out++ = value;
    }
    for (const Trace::Dma& dma : recorded.dma) {
        *out++ = dma.write_index;
        *out++ = dma.words;
    }
    std::copy(recorded.dma_words.begin(), recorded.dma_words.end(), out);
    StartInfo& start_info = Info(start);
    if (start_info.start == start) {
        start_info.last = offset;
        start_info.last_key = key;
    }
}

void Memo::NoteOutcome(u32 start, Outcome outcome, u32 skipped, bool interpreter) {
    StartInfo& info = Info(start);
    if (info.start != start) {
        return; // Wanted sets it up first
    }
    ++info.calls;
    if (outcome == Outcome::Hit) {
        ++info.lifetime_hits;
        info.unstored = 0;
        // A hit saves the JIT run and, as a no-op replay, the writes it left out.
        info.saved_ns += JitBaseNs + JitInstructionNs * info.instructions +
                         SkippedWriteNs * skipped - HitNs;
    } else {
        info.unstored = outcome == Outcome::Recorded ? 0 : info.unstored + 1;
        // Calls that can't be stored never pay for their recordings.
        if (info.unstored >= 16 && info.lifetime_hits == 0) {
            info.never = true;
            return;
        }
        // A miss or guard failure costs the lookup and a recording (the JIT with callbacks, or
        // the interpreter) on top of the run.
        info.saved_ns -= HitNs + (interpreter ? InterpreterRecordingNs : JitRecordingNs);
    }
    if (info.calls < Window) {
        return;
    }
    // Fix F3d (F3d breakdown: the hit-rate break-even let 1 ms/f of recordings through): by
    // estimated cost, not hit rate. A macro whose calls cost more through the memo than on
    // the JIT runs on the JIT for a while, then the memo tries again.
    if (info.saved_ns < 0) {
        info.bypass_left = BypassCalls;
        ++statistics.cost_bypasses;
    }
    info.calls = 0;
    info.saved_ns = 0;
}

size_t Memo::Size() const {
    return static_cast<size_t>(std::ranges::count_if(
        ways, [](const Way& way) { return way.offset != NoRecord; }));
}

void Memo::Clear() {
    std::ranges::fill(ways, Way{});
    arena.clear();
    for (StartInfo& info : starts) {
        info.last = NoRecord;
    }
    ++statistics.clears;
}

void Memo::Compact() {
    // Copies the records still in the table: a record replaced by one of another size was left
    // behind. Clears only when the live records alone would fill most of it.
    compacted.clear();
    for (Way& way : ways) {
        if (way.offset == NoRecord) {
            continue;
        }
        const u32* const record = arena.data() + way.offset;
        const size_t words = RecordWords(record);
        way.offset = static_cast<u32>(compacted.size());
        compacted.insert(compacted.end(), record, record + words);
    }
    arena.swap(compacted);
    for (StartInfo& info : starts) {
        info.last = NoRecord;
    }
    if (arena.size() >= MaxArenaWords / 4 * 3) {
        Clear();
    }
}

namespace Detail {

void LogVerifyMismatch(u32 start, std::span<const u32> record_writes,
                       const Pairs& captured_writes) {
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1) >= 32) {
        return;
    }
    std::string recorded, ran;
    for (size_t i = 0; i + 1 < record_writes.size() && i < 32; i += 2) {
        recorded += fmt::format("{:x}={:x},", record_writes[i], record_writes[i + 1]);
    }
    for (size_t i = 0; i < captured_writes.size() && i < 16; ++i) {
        ran += fmt::format("{:x}={:x},", captured_writes[i].first, captured_writes[i].second);
    }
    LOG_WARNING(HW_GPU, "V2 macro memo verify: start {:#x} replays {} but the JIT wrote {}", start,
                recorded, ran);
}

void LogRecordFailure(u32 start, std::string_view error) {
    LOG_ERROR(HW_GPU, "Native MME (split front, memo recording) failed: start={:#x}, error={}",
              start, error);
}

void LogSkipMismatch(u32 start, u32 method, u32 value) {
    static std::atomic<u32> logged{};
    if (logged.fetch_add(1) >= 32) {
        return;
    }
    LOG_WARNING(HW_GPU,
                "V2 macro memo verify: start {:#x}: the replay would leave out {:#x}={:#x}, but "
                "the write path did not drop it",
                start, method, value);
}

} // namespace Detail

void CallStats::Note(const char* engine, u32 start, std::span<const u32> parameters,
                     std::span<const u32> code, u64 code_generation, const Statistics* memo,
                     u64 key) {
    if (code_generation != generation) {
        Fold();
        by_start.fill({}); // the identities belong to the old code
        generation = code_generation;
    }
    StartCounts& counts = by_start[start % by_start.size()];
    if (counts.start != start) [[unlikely]] {
        if (counts.start != ~0U && counts.counts.calls != 0) {
            // Another start's counts so far, folded under its hash.
            Counts& folded = by_hash[counts.counts.hash];
            folded.hash = counts.counts.hash;
            folded.calls += counts.counts.calls;
            folded.repeats += counts.counts.repeats;
        }
        counts.start = start;
        counts.counts = Counts{.hash = MacroHle::Identify(code, start).hash};
    }
    if (key == 0) {
        key = Hash(start, parameters);
    }
    u64& slot = recent[key % recent.size()];
    const bool repeat = slot == key;
    slot = key;
    ++counts.counts.calls;
    ++calls;
    if (repeat) {
        ++counts.counts.repeats;
        ++repeats;
    }
    if ((calls & 4095) != 0) {
        return;
    }
    const u64 now = static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    if (window_start_ns == 0) {
        window_start_ns = now;
        memo_start = memo != nullptr ? *memo : Statistics{};
        return;
    }
    if (now - window_start_ns >= 10'000'000'000ULL) {
        Report(engine, memo, static_cast<double>(now - window_start_ns) / 1e9);
        window_start_ns = now;
        memo_start = memo != nullptr ? *memo : Statistics{};
    }
}

void CallStats::Fold() {
    for (StartCounts& slot : by_start) {
        if (slot.start == ~0U || slot.counts.calls == 0) {
            continue;
        }
        Counts& folded = by_hash[slot.counts.hash];
        folded.hash = slot.counts.hash;
        folded.calls += slot.counts.calls;
        folded.repeats += slot.counts.repeats;
        slot.counts.calls = 0;
        slot.counts.repeats = 0;
    }
}

void CallStats::Report(const char* engine, const Statistics* memo, double seconds) {
    Fold();
    std::vector<Counts> top;
    top.reserve(by_hash.size());
    for (const auto& [hash, counts] : by_hash) {
        top.push_back(counts);
    }
    std::ranges::sort(top, [](const Counts& a, const Counts& b) { return a.calls > b.calls; });
    const auto percent = [](u64 part, u64 whole) {
        return whole != 0 ? 100.0 * static_cast<double>(part) / static_cast<double>(whole) : 0.0;
    };
    std::string text = fmt::format(
        "V2 stats {} macros (per s): calls {:.0f}, exact repeats {:.1f}%, distinct {}", engine,
        static_cast<double>(calls) / seconds, percent(repeats, calls), top.size());
    if (memo != nullptr && memo->calls != memo_start.calls) {
        const auto delta = [&](u64 Statistics::*field) {
            return static_cast<double>(memo->*field - memo_start.*field) / seconds;
        };
        text += fmt::format(
            " | memo: hits {:.0f} ({:.1f}% of calls, {:.0f} on the last record), misses {:.0f}, "
            "guard failures {:.0f}, unmemoisable {:.0f}, JIT only {:.0f}, DMA refetches {:.0f}, "
            "stores {:.0f}, JIT recordings {:.0f}, verified {:.0f} ({:.0f} mismatches), "
            "clears {:.0f}, hits leaving writes out {:.0f} ({:.0f} writes left out), cost "
            "bypasses {:.0f}",
            delta(&Statistics::hits),
            percent(memo->hits - memo_start.hits, memo->calls - memo_start.calls),
            delta(&Statistics::slot_hits), delta(&Statistics::misses),
            delta(&Statistics::guard_failures), delta(&Statistics::unmemoisable),
            delta(&Statistics::bypassed), delta(&Statistics::refetches),
            delta(&Statistics::stores), delta(&Statistics::jit_records),
            delta(&Statistics::verified),
            delta(&Statistics::verify_mismatches), delta(&Statistics::clears),
            delta(&Statistics::noop_hits),
            delta(&Statistics::skipped_writes), delta(&Statistics::cost_bypasses));
    }
    text += " | top:";
    for (size_t i = 0; i < std::min<size_t>(top.size(), 10); ++i) {
        text += fmt::format(" {:016x} {:.0f}/s ({:.0f}% repeats),", top[i].hash,
                            static_cast<double>(top[i].calls) / seconds,
                            percent(top[i].repeats, top[i].calls));
    }
    text.pop_back();
    LOG_INFO(HW_GPU, "{}", text);
    by_hash.clear();
    calls = 0;
    repeats = 0;
}

} // namespace VideoCommon::V2::MacroMemo
