// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"

// Pipeline v2 P2: NVN2 (Turing macro ISA) macro identification and HLE (MASTER_PLAN.md §5.1;
// Ryujinx MacroHLE, notes/03 §3, §10.4). Switch 1's macros are identified by a hash of their
// code (macro.cpp HLE table); NVN2 macros live in one shared code RAM and are bound by start
// position, so a macro here is the code from its start position through its end instruction.
//
// - EDEN_NX2_V2_MACRO_DUMP=1: every distinct macro the split front runs is written once to the
//   log ("V2 macro dump") and to nx2_macro_dump.txt in the log directory: engine, hash, start,
//   code, parameters, the memory it read (MME DMA) and the methods it wrote on its first run.
// - EDEN_NX2_V2_MACROHLE=1: a macro whose hash has an entry (built in, or a line in
//   nx2_macro_hle.txt in the config directory) is not run: the back issues a host indirect draw
//   that reads its arguments from the GPU buffer (draw_manager indirect state, as Switch 1's
//   HLE_DrawArraysIndirect / HLE_DrawIndexedIndirect / HLE_MultiDrawIndexedIndirectCount), so the
//   front never waits to read them. verify runs the macro as before and logs what the entry
//   would have drawn.
//
// nx2_macro_hle.txt, one entry per line ('#' comments):
//   <hash hex> <kind> topology=<p> address=<hi>,<lo> [count=<p>|<n>] [count_address=<hi>,<lo>]
//              [stride=<n>] [base_vertex_attribute]
// kind: draw_arrays_indirect | draw_indexed_indirect | multi_draw_indexed_indirect_count.
// <p>: a macro parameter index ("p3") or a literal number; <hi>,<lo>: parameter indices of a
// GPU address's high and low words.
namespace VideoCommon::V2::MacroHle {

struct Identity {
    u64 hash = 0;
    u32 instructions = 0;
};

/// The macro starting at instruction `start` of a Turing code RAM (three words per
/// instruction): its code through the instruction after the first one with the end bit.
[[nodiscard]] Identity Identify(std::span<const u32> code, u32 start);

enum class Kind : u8 {
    DrawArraysIndirect,
    DrawIndexedIndirect,
    MultiDrawIndexedIndirectCount,
};

/// A value from a macro parameter or a literal.
struct Operand {
    bool parameter = false;
    u32 value = 0;

    [[nodiscard]] u32 Get(std::span<const u32> parameters) const {
        return !parameter ? value : value < parameters.size() ? parameters[value] : 0;
    }
};

struct Entry {
    u64 hash = 0;
    Kind kind{};
    Operand topology;
    Operand address_high;
    Operand address_low;
    Operand count{false, 1};
    Operand count_address_high;
    Operand count_address_low;
    bool has_count_address = false;
    u32 stride = 0;
    bool base_vertex_attribute = false;
};

/// What an entry draws for these parameters (built on the front, run on the back).
struct Draw {
    Kind kind{};
    u32 topology = 0;
    GPUVAddr address = 0;
    GPUVAddr count_address = 0;
    u32 max_draws = 1;
    u32 stride = 0;
    bool base_vertex_attribute = false;
};

/// The entry for a macro hash, if any (built-in table, then nx2_macro_hle.txt).
[[nodiscard]] const Entry* Find(u64 hash);
[[nodiscard]] Draw Resolve(const Entry& entry, std::span<const u32> parameters);
/// Parses one line of nx2_macro_hle.txt (also for tests).
[[nodiscard]] std::optional<Entry> ParseLine(const std::string& line);

/// What a macro did on the run that is dumped.
struct Observation {
    std::vector<std::pair<u32, u32>> writes;           ///< method, value (first 64)
    std::vector<std::pair<GPUVAddr, u32>> dma_reads;   ///< source, words
    std::vector<u32> dma_words;                        ///< the words they read (first 64)
};

/// EDEN_NX2_V2_MACRO_DUMP: whether `hash` of `engine` was not dumped yet (claims it).
[[nodiscard]] bool ShouldDump(const char* engine, u64 hash);
void Dump(const char* engine, u32 method, u32 start, const Identity& identity,
          std::span<const u32> code, std::span<const u32> parameters,
          const Observation& observation);

} // namespace VideoCommon::V2::MacroHle
