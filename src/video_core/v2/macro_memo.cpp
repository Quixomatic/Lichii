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

Memo::Memo() = default;

void Memo::Synchronize(u64 code_generation) {
    if (code_generation != generation) {
        generation = code_generation;
        Clear();
        starts.fill({});
    }
}

bool Memo::Wanted(u32 start) const {
    const StartInfo& info = starts[start % starts.size()];
    return info.start != start || !info.bypass;
}

size_t Memo::RecordWords(const u32* header) {
    // start, parameters, reads, data RAM reads, data RAM writes, writes, DMA reads, DMA words.
    return HeaderWords + header[1] + size_t{3} * header[2] +
           size_t{2} * (header[3] + header[4] + header[5] + header[6]) + header[7];
}

bool Memo::Match(u32 offset, u32 start, std::span<const u32> parameters, const u32* regs,
                 size_t num_regs, std::span<const u32> dram, Record& record) const {
    const u32* const header = arena.data() + offset;
    if (header[0] != start || header[1] != parameters.size() ||
        !std::equal(parameters.begin(), parameters.end(), header + HeaderWords)) {
        return false;
    }
    const u32* it = header + HeaderWords + header[1];
    record.reads = {it, size_t{3} * header[2]};
    for (size_t i = 0; i < record.reads.size(); i += 3) {
        const u32 address = record.reads[i];
        if ((address < num_regs ? regs[address] : 0U) != record.reads[i + 1]) {
            return false;
        }
    }
    it += record.reads.size();
    record.dram_reads = {it, size_t{2} * header[3]};
    for (size_t i = 0; i < record.dram_reads.size(); i += 2) {
        const u32 index = record.dram_reads[i];
        if (index >= dram.size() || dram[index] != record.dram_reads[i + 1]) {
            return false;
        }
    }
    it += record.dram_reads.size();
    record.dram_writes = {it, size_t{2} * header[4]};
    it += record.dram_writes.size();
    record.writes = {it, size_t{2} * header[5]};
    it += record.writes.size();
    record.dma = {it, size_t{2} * header[6]};
    it += record.dma.size();
    record.dma_words = {it, header[7]};
    return true;
}

bool Memo::Find(u32 start, std::span<const u32> parameters, const u32* regs, size_t num_regs,
                std::span<const u32> dram, Record& record) {
    StartInfo& info = starts[start % starts.size()];
    // Many calls repeat the macro's previous call: try its record before hashing.
    if (info.start == start && info.last != NoRecord &&
        Match(info.last, start, parameters, regs, num_regs, dram, record)) {
        ++statistics.slot_hits;
        return true;
    }
    const auto it = entries.find(Hash(start, parameters));
    if (it == entries.end()) {
        ++statistics.misses;
        return false;
    }
    if (!Match(it->second, start, parameters, regs, num_regs, dram, record)) {
        ++statistics.guard_failures; // or another call with the same hash
        return false;
    }
    if (info.start == start) {
        info.last = it->second;
    }
    return true;
}

void Memo::Store(u32 start, std::span<const u32> parameters, const Trace& recorded,
                 std::span<const std::pair<u32, u32>> dram_reads,
                 std::span<const std::pair<u32, u32>> dram_writes) {
    ++statistics.stores;
    const u64 hash = Hash(start, parameters);
    const std::array<u32, HeaderWords> header{
        start,
        static_cast<u32>(parameters.size()),
        static_cast<u32>(recorded.reads.size()),
        static_cast<u32>(dram_reads.size()),
        static_cast<u32>(dram_writes.size()),
        static_cast<u32>(recorded.writes.size()),
        static_cast<u32>(recorded.dma.size()),
        static_cast<u32>(recorded.dma_words.size()),
    };
    const size_t words = RecordWords(header.data());
    u32 offset;
    // A call stored again (its reads no longer held) usually has the same shape: rewrite its
    // record in place, or the arena fills with stale copies.
    if (const auto it = entries.find(hash); it != entries.end() &&
                                            arena[it->second] == start &&
                                            RecordWords(arena.data() + it->second) == words) {
        offset = it->second;
    } else {
        if (arena.size() + words > MaxArenaWords) {
            Compact();
        }
        offset = static_cast<u32>(arena.size());
        arena.resize(arena.size() + words);
        entries.insert_or_assign(hash, offset);
    }
    u32* out = std::copy(header.begin(), header.end(), arena.data() + offset);
    out = std::copy(parameters.begin(), parameters.end(), out);
    for (const Trace::Read& read : recorded.reads) {
        *out++ = read.address;
        *out++ = read.value;
        *out++ = read.writes_before;
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
    StartInfo& info = starts[start % starts.size()];
    if (info.start != start) {
        info = StartInfo{.start = start};
    }
    info.last = offset;
}

void Memo::NoteOutcome(u32 start, bool hit, bool stored) {
    StartInfo& info = starts[start % starts.size()];
    if (info.start != start) {
        info = StartInfo{.start = start};
    }
    if (hit) {
        ++info.hits;
        info.unstored = 0;
        return;
    }
    ++info.other;
    info.unstored = stored ? 0 : info.unstored + 1;
    // Recording runs the interpreter: give up on macros whose calls can't be stored or rarely
    // repeat (until the code changes).
    info.bypass = (info.unstored >= 16 && info.hits == 0) ||
                  (info.other >= 1024 && u64{info.hits} * 8 < info.other);
}

void Memo::Clear() {
    entries.clear();
    arena.clear();
    for (StartInfo& info : starts) {
        info.last = NoRecord;
    }
    ++statistics.clears;
}

void Memo::Compact() {
    // Copies the records still in the map: a call stored again with another shape left its old
    // record behind. Clears only when the live records alone would fill most of it.
    compacted.clear();
    for (auto& entry : entries) {
        const u32* const record = arena.data() + entry.second;
        const size_t words = RecordWords(record);
        entry.second = static_cast<u32>(compacted.size());
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

} // namespace Detail

void CallStats::Note(const char* engine, u32 start, std::span<const u32> parameters,
                     std::span<const u32> code, u64 code_generation, const Statistics* memo) {
    if (code_generation != generation) {
        Fold();
        generation = code_generation;
    }
    auto [it, inserted] = by_start.try_emplace(start);
    if (inserted) {
        it->second.hash = MacroHle::Identify(code, start).hash;
    }
    const u64 key = Hash(start, parameters);
    u64& slot = recent[key % recent.size()];
    const bool repeat = slot == key;
    slot = key;
    ++it->second.calls;
    ++calls;
    if (repeat) {
        ++it->second.repeats;
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
    for (const auto& [start, counts] : by_start) {
        Counts& folded = by_hash[counts.hash];
        folded.hash = counts.hash;
        folded.calls += counts.calls;
        folded.repeats += counts.repeats;
    }
    by_start.clear();
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
            "stores {:.0f}, verified {:.0f} ({:.0f} mismatches), clears {:.0f}",
            delta(&Statistics::hits),
            percent(memo->hits - memo_start.hits, memo->calls - memo_start.calls),
            delta(&Statistics::slot_hits), delta(&Statistics::misses),
            delta(&Statistics::guard_failures), delta(&Statistics::unmemoisable),
            delta(&Statistics::bypassed), delta(&Statistics::refetches),
            delta(&Statistics::stores), delta(&Statistics::verified),
            delta(&Statistics::verify_mismatches), delta(&Statistics::clears));
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
