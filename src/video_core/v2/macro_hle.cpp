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

// Built in: none yet. No NVN2 macro has been identified from captured code; entries come from
// nx2_macro_hle.txt until dumps (EDEN_NX2_V2_MACRO_DUMP) identify them (IMPLEMENTATION_LOG P2).
constexpr std::array<Entry, 0> built_in{};

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
    } else {
        return std::nullopt;
    }
    bool has_topology = false, has_address = false;
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
        } else {
            return std::nullopt;
        }
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
