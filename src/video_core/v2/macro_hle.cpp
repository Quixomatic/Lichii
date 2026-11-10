// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

#include "common/fs/path_util.h"
#include "common/logging.h"
#include "video_core/v2/macro_hle.h"
#include "video_core/v2/regsem.h"

namespace VideoCommon::V2::MacroHle {

namespace {

constexpr u32 MaxInstructions = 4096;

// Built in, from EDEN_NX2_V2_MACRO_DUMP captures (IMPLEMENTATION_LOG Fix F5).
constexpr std::array<Entry, 3> built_in{
    // Fix F3h (R2-F5-03, notes/conformance-r2-p1-s1.md d3): the two hot 3D macros the memo
    // can't help (Ocarina of Time, 4488 dump; RE with lichii-refs/tools/mme_symbolic.py).
    // 96d36d962a7ff692 (start 0x21, ~207k calls/s): see Evaluate.
    Entry{.hash = 0x96d36d962a7ff692ULL, .kind = Kind::ConstantBufferBind},
    // 5b6e8dc0f6ab7d8e (start 0x8, ~56k calls/s): see Evaluate.
    Entry{.hash = 0x5b6e8dc0f6ab7d8eULL, .kind = Kind::SurfaceBind},
    // 3D 966aeb0e43e7a13c (Ocarina of Time, ~600 calls a frame): p1/p2 into the MME DMA source
    // registers, a 6-word MME_DMA_READ_FIFOED of a ReportSemaphore::Compare there, then
    // render_enable.mode = p0 ^ ((w0 ^ w4) | (w1 ^ w5) != 0): IfNotEqual (p0 = 0) or IfEqual
    // (p0 = 1) evaluated by the macro.
    Entry{
        .hash = 0x966aeb0e43e7a13cULL,
        .kind = Kind::RenderEnableCompare,
        .writes = {{
            {0x154, {true, 1}},
            {0x155, {true, 2}},
            {0x158, {false, 6}},
        }},
        .num_writes = 3,
        .invert = {true, 0},
        .words = 6,
    },
};

std::optional<Operand> ParseOperand(const std::string& text) {
    if (text.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    if (text[0] == 'p') {
        const unsigned long value = std::strtoul(text.c_str() + 1, &end, 10);
        return *end == '\0' ? std::optional<Operand>{Operand{true, static_cast<u32>(value)}}
                            : std::nullopt;
    }
    const unsigned long value = std::strtoul(text.c_str(), &end, 0);
    return *end == '\0' ? std::optional<Operand>{Operand{false, static_cast<u32>(value)}}
                        : std::nullopt;
}

struct Table {
    std::unordered_map<u64, Entry> entries;

    Table() {
        for (const Entry& entry : built_in) {
            entries.emplace(entry.hash, entry);
        }
        const auto path =
            Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir) / "nx2_macro_hle.txt";
        std::ifstream file{path};
        std::string line;
        u32 loaded = 0;
        while (std::getline(file, line)) {
            if (const auto entry = ParseLine(line)) {
                entries.insert_or_assign(entry->hash, *entry);
                ++loaded;
            } else if (!line.empty() && line[0] != '#') {
                LOG_WARNING(HW_GPU, "V2 macro HLE: can't parse '{}' in nx2_macro_hle.txt", line);
            }
        }
        if (loaded != 0) {
            LOG_INFO(HW_GPU, "V2 macro HLE: {} entries from nx2_macro_hle.txt", loaded);
        }
    }
};

const Table& GetTable() {
    static const Table table;
    return table;
}

} // Anonymous namespace

Identity Identify(std::span<const u32> code, u32 start) {
    Identity identity;
    u64 hash = Mix64(0x4e564e32ULL); // "NVN2": the code alone, wherever it was uploaded
    bool end_next = false;
    for (u32 pc = start; pc < start + MaxInstructions && size_t{pc} * 3 + 2 < code.size(); ++pc) {
        for (u32 word = 0; word < 3; ++word) {
            hash = Mix64(hash ^ code[size_t{pc} * 3 + word]);
        }
        ++identity.instructions;
        if (end_next) {
            break;
        }
        // Bit 0 of the instruction's last word ends the macro after the next one.
        end_next = (code[size_t{pc} * 3 + 2] & 1) != 0;
    }
    identity.hash = hash;
    return identity;
}

std::optional<Entry> ParseLine(const std::string& line) {
    std::istringstream stream{line};
    std::string hash_text, kind_text;
    if (!(stream >> hash_text >> kind_text) || hash_text[0] == '#') {
        return std::nullopt;
    }
    Entry entry;
    entry.hash = std::strtoull(hash_text.c_str(), nullptr, 16);
    if (kind_text == "draw_arrays_indirect") {
        entry.kind = Kind::DrawArraysIndirect;
    } else if (kind_text == "draw_indexed_indirect") {
        entry.kind = Kind::DrawIndexedIndirect;
    } else if (kind_text == "multi_draw_indexed_indirect_count") {
        entry.kind = Kind::MultiDrawIndexedIndirectCount;
    } else if (kind_text == "render_enable_compare") {
        entry.kind = Kind::RenderEnableCompare;
    } else {
        return std::nullopt;
    }
    bool has_topology = false, has_address = false, has_invert = false;
    std::string field;
    while (stream >> field) {
        if (field == "base_vertex_attribute") {
            entry.base_vertex_attribute = true;
            continue;
        }
        const auto equals = field.find('=');
        if (equals == std::string::npos) {
            return std::nullopt;
        }
        const std::string key = field.substr(0, equals);
        const std::string value = field.substr(equals + 1);
        const auto comma = value.find(',');
        if (key == "topology") {
            const auto operand = ParseOperand(value);
            if (!operand) {
                return std::nullopt;
            }
            entry.topology = *operand;
            has_topology = true;
        } else if ((key == "address" || key == "count_address") && comma != std::string::npos) {
            const auto high = ParseOperand(value.substr(0, comma));
            const auto low = ParseOperand(value.substr(comma + 1));
            if (!high || !low) {
                return std::nullopt;
            }
            if (key == "address") {
                entry.address_high = *high;
                entry.address_low = *low;
                has_address = true;
            } else {
                entry.count_address_high = *high;
                entry.count_address_low = *low;
                entry.has_count_address = true;
            }
        } else if (key == "count") {
            const auto operand = ParseOperand(value);
            if (!operand) {
                return std::nullopt;
            }
            entry.count = *operand;
        } else if (key == "stride") {
            entry.stride = static_cast<u32>(std::strtoul(value.c_str(), nullptr, 0));
        } else if (key == "invert") {
            const auto operand = ParseOperand(value);
            if (!operand) {
                return std::nullopt;
            }
            entry.invert = *operand;
            has_invert = true;
        } else if (key == "words") {
            entry.words = static_cast<u32>(std::strtoul(value.c_str(), nullptr, 0));
        } else if (key == "writes") {
            // <method>:<operand>,...
            std::istringstream list{value};
            std::string item;
            while (std::getline(list, item, ',')) {
                const auto colon = item.find(':');
                if (colon == std::string::npos || entry.num_writes == entry.writes.size()) {
                    return std::nullopt;
                }
                const auto operand = ParseOperand(item.substr(colon + 1));
                if (!operand) {
                    return std::nullopt;
                }
                entry.writes[entry.num_writes++] = WriteOp{
                    static_cast<u32>(std::strtoul(item.substr(0, colon).c_str(), nullptr, 0)),
                    *operand};
            }
        } else {
            return std::nullopt;
        }
    }
    if (entry.kind == Kind::RenderEnableCompare) {
        // The report pair's six words, read where the writes put the DMA source.
        if (!has_invert || entry.words != 6 || entry.num_writes == 0) {
            return std::nullopt;
        }
        return entry;
    }
    if (!has_topology || !has_address ||
        (entry.kind == Kind::MultiDrawIndexedIndirectCount && !entry.has_count_address)) {
        return std::nullopt;
    }
    return entry;
}

const Entry* Find(u64 hash) {
    const auto& entries = GetTable().entries;
    const auto it = entries.find(hash);
    return it != entries.end() ? &it->second : nullptr;
}

Draw Resolve(const Entry& entry, std::span<const u32> parameters) {
    Draw draw;
    draw.kind = entry.kind;
    draw.topology = entry.topology.Get(parameters);
    draw.address = (GPUVAddr{entry.address_high.Get(parameters) & 0xff} << 32) |
                   entry.address_low.Get(parameters);
    if (entry.has_count_address) {
        draw.count_address = (GPUVAddr{entry.count_address_high.Get(parameters) & 0xff} << 32) |
                             entry.count_address_low.Get(parameters);
    }
    draw.max_draws = std::max<u32>(entry.count.Get(parameters), 1);
    draw.stride = entry.stride;
    draw.base_vertex_attribute = entry.base_vertex_attribute;
    return draw;
}

namespace {

// TuringMacroInterpreter's data RAM bounds (a read or write past it fails the macro).
constexpr u32 DataRamWords = 0xc00;

// The MERGE operation: `count` bits of y from bit `source` into x at bit `dest`.
constexpr u32 Merge(u32 x, u32 y, u32 dest, u32 count, u32 source) {
    const u32 mask = (1U << count) - 1;
    return (x & ~(mask << dest)) | (((y >> source) & mask) << dest);
}

void AddWrite(Effects& out, u32 method, u32 value) {
    out.writes[out.num_writes++] = {method, value};
}

void AddDramWrite(Effects& out, u32 index, u32 value) {
    out.dram_writes[out.num_dram_writes++] = {index, value};
}

// 3D 96d36d962a7ff692 (15 instructions at 0x21), p0 stage or binding index, p1 slot, p2
// address low, p3 size << 12 | address high:
//   if !(s32(p0) < 8): r1 = merge(merge(0x480, p0 bits 0-1 at 5), p1 bits 0-3 at 1)
//                      stage = dram[0x520 + p0]; dram[r1] = p2; dram[r1 + 1] = p3
//   else stage = p0
//   if !(s32(stage) < 15): end (nothing written)
//   0x8e0 = p3 >> 12 & 0xfffff; 0x8e1 = p3 & 0xff; 0x8e2 = p2
//   [0x904 + (stage & 7) * 8] = 1 | ((p1 + 3) & 0xfffffff) << 4
bool EvaluateConstantBufferBind(std::span<const u32> p, std::span<const u32> dram,
                                Effects& out) {
    if (p.size() != 4) {
        return false;
    }
    u32 stage = p[0];
    if (!(static_cast<s32>(p[0]) < 8)) {
        const u32 index = Merge(Merge(0x480, p[0], 5, 2, 0), p[1], 1, 4, 0);
        const u32 stage_index = 0x520 + p[0];
        if (stage_index >= DataRamWords || index + 1 >= DataRamWords ||
            stage_index >= dram.size()) {
            return false;
        }
        stage = dram[stage_index]; // read before the writes (same instruction order)
        AddDramWrite(out, index, p[2]);
        AddDramWrite(out, index + 1, p[3]);
    }
    if (!(static_cast<s32>(stage) < 15)) {
        return true;
    }
    AddWrite(out, 0x8e0, Merge(0, p[3], 0, 20, 12));
    AddWrite(out, 0x8e1, p[3] & 0xff);
    AddWrite(out, 0x8e2, p[2]);
    AddWrite(out, Merge(0x904, stage, 3, 3, 0) & 0xfff, Merge(1, p[1] + 3, 4, 28, 0));
    return true;
}

// 3D 5b6e8dc0f6ab7d8e (23 instructions at 0x8), p0 index, p1..p4 the binding (p4: slot in
// bits 24-26, cb_bind stage in bits 28-30):
//   base = 0x500 + p0 * 4 (signed multiply); if p1 == dram[base]: end
//   mask = dram[0x52c] (read before the writes); dram[base..base+3] = p1, p2, p3, p4
//   if (mask & 1 << p0) == 0: end
//   s = p4 >> 24 & 7: [0x805 + 16s] = p2 >> 24; [0x806 + 16s] = p2 << 8; [0x807 + 16s] =
//   p4 >> 9 & 0x7f; 0x8e0 = 0x10000; 0x8e1 = p3 >> 24; 0x8e2 = p3 << 8;
//   [0x904 + (p4 >> 28 & 7) * 8] = 0x11; [0x803 + 16s] = p4 & 0x1ff; [0x800 + 16s] = 1 | s << 4
bool EvaluateSurfaceBind(std::span<const u32> p, std::span<const u32> dram, Effects& out) {
    if (p.size() != 5) {
        return false;
    }
    const u32 base = static_cast<u32>(static_cast<s64>(static_cast<s32>(p[0])) * 4) + 0x500;
    if (base >= DataRamWords || base >= dram.size()) {
        return false;
    }
    if (p[1] == dram[base]) {
        return true;
    }
    if (0x52c >= dram.size() || base + 3 >= DataRamWords) {
        return false;
    }
    const u32 mask = dram[0x52c];
    AddDramWrite(out, base, p[1]);
    AddDramWrite(out, base + 1, p[2]);
    AddDramWrite(out, base + 2, p[3]);
    AddDramWrite(out, base + 3, p[4]);
    if ((mask & (1U << (p[0] & 31))) == 0) {
        return true;
    }
    const u32 slot = (p[4] >> 24) & 7;
    const u32 rt = Merge(0x805, p[4], 4, 3, 24); // 0x805 + 16s
    AddWrite(out, rt, (p[2] >> 24) & 0xff);
    AddWrite(out, rt + 1, (p[2] & 0xffffff) << 8);
    AddWrite(out, rt + 2, (p[4] >> 9) & 0x7f);
    AddWrite(out, 0x8e0, 0x10000);
    AddWrite(out, 0x8e1, (p[3] >> 24) & 0xff);
    AddWrite(out, 0x8e2, (p[3] & 0xffffff) << 8);
    AddWrite(out, Merge(0x904, p[4], 3, 3, 28), 0x11);
    AddWrite(out, Merge(0x803, p[4], 4, 3, 24), p[4] & 0x1ff);
    AddWrite(out, Merge(0x800, p[4], 4, 3, 24), Merge(1, slot, 4, 3, 0));
    return true;
}

} // Anonymous namespace

bool Evaluate(const Entry& entry, std::span<const u32> parameters, std::span<const u32> dram,
              Effects& out) {
    out = {};
    switch (entry.kind) {
    case Kind::ConstantBufferBind:
        return EvaluateConstantBufferBind(parameters, dram, out);
    case Kind::SurfaceBind:
        return EvaluateSurfaceBind(parameters, dram, out);
    default:
        return false;
    }
}

std::optional<std::string> CheckEffects(const Effects& predicted,
                                        std::span<const std::pair<u32, u32>> writes,
                                        std::span<const u32> dram) {
    std::string differences;
    if (writes.size() != predicted.num_writes) {
        differences += fmt::format("{} writes, predicted {}; ", writes.size(),
                                   predicted.num_writes);
    }
    for (u32 i = 0; i < std::min<size_t>(writes.size(), predicted.num_writes); ++i) {
        if (writes[i] != predicted.writes[i]) {
            differences += fmt::format("write {} {:#x}={:#x}, predicted {:#x}={:#x}; ", i,
                                       writes[i].first, writes[i].second,
                                       predicted.writes[i].first, predicted.writes[i].second);
        }
    }
    for (u32 i = 0; i < predicted.num_dram_writes; ++i) {
        const auto [index, value] = predicted.dram_writes[i];
        if (index < dram.size() && dram[index] != value) {
            differences += fmt::format("data RAM {:#x}={:#x}, predicted {:#x}; ", index,
                                       dram[index], value);
        }
    }
    if (differences.empty()) {
        return std::nullopt;
    }
    return differences;
}

bool Applies(const Entry& entry, std::span<const u32> parameters) {
    if (entry.kind != Kind::RenderEnableCompare) {
        return true;
    }
    const auto covered = [&](const Operand& operand) {
        return !operand.parameter || operand.value < parameters.size();
    };
    for (u32 i = 0; i < entry.num_writes; ++i) {
        if (!covered(entry.writes[i].value)) {
            return false;
        }
    }
    // The macro XORs the parameter into the mode: only 0 and 1 give IfNotEqual / IfEqual.
    return covered(entry.invert) && entry.invert.Get(parameters) <= 1;
}

u32 RenderEnableValue(bool invert, std::span<const u32> words) {
    const auto word = [&](size_t i) { return i < words.size() ? words[i] : 0U; };
    // ReportSemaphore::Compare: initial sequence and mode, two words, current sequence and mode.
    const bool differ = ((word(0) ^ word(4)) | (word(1) ^ word(5))) != 0;
    return (differ ? 1U : 0U) ^ (invert ? 1U : 0U);
}

std::optional<std::string> CheckRenderEnable(const Entry& entry, std::span<const u32> parameters,
                                             std::span<const std::pair<u32, u32>> writes,
                                             std::span<const u32> words) {
    std::vector<std::pair<u32, u32>> expected;
    for (u32 i = 0; i < entry.num_writes; ++i) {
        expected.emplace_back(entry.writes[i].method, entry.writes[i].value.Get(parameters));
    }
    expected.emplace_back(0x556, RenderEnableValue(entry.invert.Get(parameters) != 0, words));
    const bool same = words.size() == entry.words && writes.size() == expected.size() &&
                      std::equal(writes.begin(), writes.end(), expected.begin());
    if (same) {
        return std::nullopt;
    }
    std::string text = "macro wrote ";
    for (const auto& [method, value] : writes) {
        text += fmt::format("{:x}={:x},", method, value);
    }
    text += " after reading ";
    for (const u32 word : words) {
        text += fmt::format("{:x},", word);
    }
    text += " the entry writes ";
    for (const auto& [method, value] : expected) {
        text += fmt::format("{:x}={:x},", method, value);
    }
    return text;
}

bool ShouldDump(const char* engine, u64 hash) {
    static std::mutex mutex;
    static std::unordered_set<u64> dumped;
    std::scoped_lock lock{mutex};
    return dumped.insert(hash ^ (engine[0] == 'c' ? 0x9e3779b97f4a7c15ULL : 0)).second;
}

void Dump(const char* engine, u32 method, u32 start, const Identity& identity,
          std::span<const u32> code, std::span<const u32> parameters,
          const Observation& observation) {
    std::string text = fmt::format(
        "engine={} hash={:016x} method={:#x} start={:#x} instructions={} parameters={}", engine,
        identity.hash, method, start, identity.instructions, parameters.size());
    text += " params=";
    for (const u32 parameter : parameters.first(std::min<size_t>(parameters.size(), 32))) {
        text += fmt::format("{:x},", parameter);
    }
    text += " dma=";
    for (const auto& [source, words] : observation.dma_reads) {
        text += fmt::format("{:#x}x{},", source, words);
    }
    text += " dma_words=";
    for (const u32 word : observation.dma_words) {
        text += fmt::format("{:x},", word);
    }
    text += " writes=";
    for (const auto& [written, value] : observation.writes) {
        text += fmt::format("{:x}={:x},", written, value);
    }
    text += " code=";
    for (u32 i = 0; i < identity.instructions * 3 && size_t{start} * 3 + i < code.size(); ++i) {
        text += fmt::format("{:08x}", code[size_t{start} * 3 + i]);
        text += (i % 3 == 2) ? ' ' : ':';
    }
    LOG_INFO(HW_GPU, "V2 macro dump: {}", text);
    static std::mutex file_mutex;
    std::scoped_lock lock{file_mutex};
    static std::ofstream file{Common::FS::GetEdenPath(Common::FS::EdenPath::LogDir) /
                                  "nx2_macro_dump.txt",
                              std::ios::app};
    file << text << '\n';
    file.flush();
}

} // namespace VideoCommon::V2::MacroHle
