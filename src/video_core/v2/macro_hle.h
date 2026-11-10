// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
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
//   Conditional render macros (Kind::RenderEnableCompare, Fix F5-03/04) make their register
//   writes on the front and leave the report pair read to the back, which writes the render
//   enable mode the macro computes; verify compares the macro's writes with the entry's.
//
// Entries are built in (keyed by code hash) and nx2_macro_hle.txt lines override them, one
// entry per line ('#' comments):
//   <hash hex> <kind> topology=<p> address=<hi>,<lo> [count=<p>|<n>] [count_address=<hi>,<lo>]
//              [stride=<n>] [base_vertex_attribute]
//   <hash hex> render_enable_compare invert=<p> writes=<method>:<p>,... words=<n>
// kind: draw_arrays_indirect | draw_indexed_indirect | multi_draw_indexed_indirect_count |
// render_enable_compare.
// <p>: a macro parameter index ("p3") or a literal number; <hi>,<lo>: parameter indices of a
// GPU address's high and low words. render_enable_compare reads its report pair at the address
// in the MME DMA source registers (0x154/0x155) after its writes.
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
    /// render_enable.mode = invert ^ (report pair's two (sequence, mode) words differ): the
    /// macro's MME_DMA_READ_FIFOED of a ReportSemaphore::Compare at the DMA source registers.
    RenderEnableCompare,
};

/// A value from a macro parameter or a literal.
struct Operand {
    bool parameter = false;
    u32 value = 0;

    [[nodiscard]] u32 Get(std::span<const u32> parameters) const {
        return !parameter ? value : value < parameters.size() ? parameters[value] : 0;
    }
};

/// A register write an entry makes as its macro does.
struct WriteOp {
    u32 method = 0;
    Operand value;
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
    // RenderEnableCompare:
    std::array<WriteOp, 4> writes{}; ///< Made on the front in order
    u32 num_writes = 0;
    Operand invert; ///< 0 or 1 (other values: the macro runs)
    u32 words = 0;  ///< Words of the DMA read (6: a ReportSemaphore::Compare)
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
/// Whether the entry stands for a call with these parameters (else the macro runs).
[[nodiscard]] bool Applies(const Entry& entry, std::span<const u32> parameters);
/// RenderEnableCompare's mode from the report pair's six words.
[[nodiscard]] u32 RenderEnableValue(bool invert, std::span<const u32> words);
/// MACROHLE=verify for RenderEnableCompare: what differs between the macro's writes (and the
/// words its DMA read returned) and what the entry makes; nothing when they agree.
[[nodiscard]] std::optional<std::string> CheckRenderEnable(
    const Entry& entry, std::span<const u32> parameters,
    std::span<const std::pair<u32, u32>> writes, std::span<const u32> words);
/// Parses one line of nx2_macro_hle.txt (also for tests).
[[nodiscard]] std::optional<Entry> ParseLine(const std::string& line);

/// What a macro did on the run that is dumped.
struct Observation {
    std::vector<std::pair<u32, u32>> writes;           ///< method, value (first 64)
    std::vector<std::pair<GPUVAddr, u32>> dma_reads;   ///< source, words
    std::vector<u32> dma_words;                        ///< the words they read (first 64)
    std::vector<u32> read_words; ///< the words MME_DMA_READ_FIFOED gave the macro
};

/// EDEN_NX2_V2_MACRO_DUMP: whether `hash` of `engine` was not dumped yet (claims it).
[[nodiscard]] bool ShouldDump(const char* engine, u64 hash);
void Dump(const char* engine, u32 method, u32 start, const Identity& identity,
          std::span<const u32> code, std::span<const u32> parameters,
          const Observation& observation);

} // namespace VideoCommon::V2::MacroHle
