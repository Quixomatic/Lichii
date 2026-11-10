// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <bit>
#include <cstring>

#include "common/common_types.h"
#include "video_core/v2/ir.h"
#include "video_core/v2/regsem.h"

// Pipeline v2 S1 single owner (MASTER_PLAN §5.1/§5.2, Fix F1): the split front is the only place
// Maxwell3D register semantics run (shadow RAM, compare-on-write, auto-increment, macro pointers,
// falcon[4], instance loops, draw triggers). What reaches the back is a StateRun of classified
// entries (an Ir::Classified run): post-shadow register deltas of changed registers, and typed
// entries for the writes with side effects. The back stores deltas (regs[i] = v plus the v1
// dirty flags), and executes only the typed entries: no shadow RAM, no compare, no action switch
// on state writes, no method sink, no draw manager state machine, no group hashing.
//
// Entries are u64 words; bits 0-3 are the tag (the old Method entry, tag 2, never appears in a
// Classified run, so the shards and checks can tell the two apart in mixed runs):
//   Delta  [tag][Quiet 4][index 8-23][value 32-63]         regs[index] = value (Quiet: no
//                                                           dirty flag)
//   Exec   [tag][IsLast 4][index 8-23][value 32-63]        store, then the register's action
//                                                           (reports, barriers, binds, uploads,
//                                                           root tables; Action3D)
//   Cbuf   [tag][Store 4][Inc 5][m0 8-11][count 12-31][offset 32-63], [buffer GPU address],
//          then `count` data words packed 2 per u64        a constant buffer upload at decode time:
//                                                           buffer + offset; with Store the words
//                                                           also land in const_buffer.buffer[m0 +
//                                                           Inc * k]; const_buffer.offset ends at
//                                                           offset + 4 * count
//   Draw   [tag][variant 8-15], then an Ir::PredictedDraw  a draw the front's DrawModel predicted,
//                                                           on registers already current
//   Inline [tag][index 8-23][count 32-63], words packed     inline indices appended
//   Groups [tag][RegBreak 4][MethodBreak 5][dirty 8-23][method 32-63], then one hash per hashed
//          group in dirty                                   the front's dirty groups and hashes at
//                                                           the next draw, and the fast draw breaks
namespace VideoCommon::V2::S1 {

enum class Tag : u8 {
    Delta = 8,
    Exec = 9,
    Cbuf = 10,
    Draw = 11,
    Inline = 12,
    Groups = 13,
    /// KeplerCompute (native class): [tag][dirty 8-23], then one hash per hashed compute group
    /// in dirty; the front's compute groups before a write that may launch QMDs.
    ComputeGroups = 14,
    /// EDEN_NX2_V2_CBUFRING (v2/cbuf_ring.h): [tag][size 32-63], [window GPU address], [ring
    /// position, ~0 for none]; the window's decode-time version, sent before the next draw.
    CbufVersion = 15,
};
inline constexpr size_t CbufVersionWords = 3;

inline constexpr u64 TagMask = 0xf;
inline constexpr u64 IsLast = 1 << 4;
/// Delta: a side effect's store (macro bind pointer, falcon[4] scratch) that marks no dirty
/// flag, as the engine's handler stores it.
inline constexpr u64 DeltaQuiet = 1 << 4;
inline constexpr u64 CbufStore = 1 << 4;
inline constexpr u64 CbufInc = 1 << 5;
inline constexpr u64 GroupsRegisterBreak = 1 << 4;
inline constexpr u64 GroupsMethodBreak = 1 << 5;
/// Compute Exec of the inline QMD (0x107): the upload only; the launch follows as entries.
inline constexpr u64 ComputeNoLaunch = 1 << 5;
/// Compute Exec of the inline QMD's launch: skipped when its upload found no mapped memory.
inline constexpr u64 ComputeInlineLaunch = 1 << 6;
inline constexpr u32 MaxCbufWords = 256;
inline constexpr u32 MaxInlineWords = 512;
inline constexpr size_t DrawWords = 1 + sizeof(Ir::PredictedDraw) / sizeof(u64);
static_assert(sizeof(Ir::PredictedDraw) % sizeof(u64) == 0);

/// DrawModel variants the back's draw needs beyond the PredictedDraw (DrawOutput::variants).
enum DrawVariant : u8 {
    DrawInline = 1 << 0,     ///< Inline index draw: index format u32, the collected bytes cleared
    DrawIndexDirty = 1 << 1, ///< The index buffer dirty flag (index_buffer*_first/subsequent)
};

/// Groups whose changes don't break fast draw reuse: draw parameters, the index buffer, constant
/// buffer binds and data (the rasterizer checks those against its handles) and state no draw
/// reads. Any other changed group is a register break (Nx2FastDraw::Reason::Register), which
/// EDEN_NX2_V2_CONTDRAW resolves from the groups.
inline constexpr u32 LightGroups = GroupBit(RegGroup::DrawParams) |
                                   GroupBit(RegGroup::IndexBuffer) |
                                   GroupBit(RegGroup::CbufBind) | GroupBit(RegGroup::CbufData) |
                                   GroupBit(RegGroup::Misc);

[[nodiscard]] constexpr Tag TagOf(u64 word) {
    return static_cast<Tag>(word & TagMask);
}
[[nodiscard]] constexpr bool IsClassified(u64 word) {
    return (word & TagMask) >= static_cast<u64>(Tag::Delta);
}
[[nodiscard]] constexpr u64 MakeRegister(Tag tag, u32 index, u32 value, u64 flags = 0) {
    return static_cast<u64>(tag) | flags | (u64{index & 0xffff} << 8) | (u64{value} << 32);
}
[[nodiscard]] constexpr u32 IndexOf(u64 word) {
    return static_cast<u32>((word >> 8) & 0xffff);
}
[[nodiscard]] constexpr u32 ValueOf(u64 word) {
    return static_cast<u32>(word >> 32);
}
[[nodiscard]] constexpr u64 MakeCbuf(bool store, bool inc, u32 m0, u32 count, u32 offset) {
    return static_cast<u64>(Tag::Cbuf) | (store ? CbufStore : 0) | (inc ? CbufInc : 0) |
           (u64{m0 & 0xf} << 8) | (u64{count & 0xfffff} << 12) | (u64{offset} << 32);
}
[[nodiscard]] constexpr u32 CbufCount(u64 word) {
    return static_cast<u32>((word >> 12) & 0xfffff);
}
[[nodiscard]] constexpr u32 CbufMethod0(u64 word) {
    return static_cast<u32>((word >> 8) & 0xf);
}
[[nodiscard]] constexpr u64 MakeGroups(u32 dirty, u64 breaks, u32 method) {
    return static_cast<u64>(Tag::Groups) | breaks | (u64{dirty & 0xffff} << 8) |
           (u64{method} << 32);
}
[[nodiscard]] constexpr u32 GroupsDirty(u64 word) {
    return static_cast<u32>((word >> 8) & 0xffff);
}

/// The size of the classified entry at `entry`, in words.
[[nodiscard]] inline size_t EntryWords(const u64* entry) {
    const u64 word = entry[0];
    switch (TagOf(word)) {
    case Tag::Cbuf:
        return 2 + (size_t{CbufCount(word)} + 1) / 2;
    case Tag::Draw:
        return DrawWords;
    case Tag::Inline:
        return 1 + (size_t{ValueOf(word)} + 1) / 2;
    case Tag::Groups:
        return 1 + static_cast<size_t>(std::popcount(GroupsDirty(word) & HashedGroups));
    case Tag::ComputeGroups:
        return 1 + static_cast<size_t>(std::popcount(GroupsDirty(word) & HashedComputeGroups));
    case Tag::CbufVersion:
        return CbufVersionWords;
    default:
        return 1;
    }
}

/// The register a Cbuf entry's word k lands in (with CbufStore).
[[nodiscard]] constexpr u32 CbufMethod(u64 word, u32 cb_data, u32 k) {
    return cb_data + CbufMethod0(word) + ((word & CbufInc) != 0 ? k : 0);
}

[[nodiscard]] inline Ir::PredictedDraw DrawOf(const u64* entry) {
    Ir::PredictedDraw draw;
    std::memcpy(&draw, entry + 1, sizeof(draw));
    return draw;
}

/// What the back stores for a classified run (the register part of every entry), on any
/// register file: the verify mirror, the shards and the tests use it; the executing Maxwell3D
/// does the same plus the side effects. `store(index, value)` is called for every register write.
template <typename Store>
inline void ApplyRegisters(const u64* entry, u32 cb_data, u32 cb_offset, Store&& store) {
    const u64 word = entry[0];
    switch (TagOf(word)) {
    case Tag::Delta:
    case Tag::Exec:
        store(IndexOf(word), ValueOf(word));
        break;
    case Tag::Cbuf: {
        const u32 count = CbufCount(word);
        if ((word & CbufStore) != 0) {
            const u32* const words = reinterpret_cast<const u32*>(entry + 2);
            for (u32 k = 0; k < count; ++k) {
                store(CbufMethod(word, cb_data, k), words[k]);
            }
        }
        store(cb_offset, ValueOf(word) + count * static_cast<u32>(sizeof(u32)));
        break;
    }
    default:
        break;
    }
}

/// Front: how one register write the semantics engine (RegSem::Write3D) accepted becomes
/// classified entries. `regs` are the registers after the write; `old_value` and `old_offset`
/// the register and const_buffer.offset before it; `touch[i]`: the register maps to a v1 dirty
/// flag, so an executable write with an unchanged value still marks it (v1 runs
/// ProcessDirtyRegisters for every executable method). The sink takes Delta(index, value),
/// Quiet(index, value) (DeltaQuiet), Exec(index, value, is_last), Cbuf(store, method, buffer, offset, word) and
/// Inline(method, words, count).
template <typename Sink>
inline void Classify3D(const u32* regs, const RegSem::Table3D& table, const bool* touch,
                       bool back_skip_same, u32 method, u32 old_value, u32 old_offset,
                       bool is_last, Sink& sink) {
    using RegSem::Action3D;
    const u32 value = regs[method];
    const bool changed = value != old_value;
    switch (table.actions[method]) {
    case Action3D::Store:
        if (changed || !back_skip_same) {
            sink.Delta(method, value);
        }
        return;
    case Action3D::CbData: {
        const u32 offset_index = table.const_buffer_offset;
        // const_buffer: size, address_high, address_low, offset.
        const u64 buffer = (u64{regs[offset_index - 2]} << 32) | regs[offset_index - 1];
        sink.Cbuf(true, method, buffer, old_offset, value);
        return;
    }
    case Action3D::StoreExec:
    case Action3D::TopologyOverride:
    case Action3D::DrawBegin:
    case Action3D::DrawEnd:
    case Action3D::ShadowControl:
    case Action3D::MacroCodePointer:
    case Action3D::MacroCode:
        if (changed || touch[method]) {
            sink.Delta(method, value);
        }
        return;
    case Action3D::MacroBind:
        if (changed || touch[method]) {
            sink.Delta(method, value);
        }
        sink.Quiet(table.start_address_ptr, regs[table.start_address_ptr]);
        return;
    case Action3D::Falcon4:
        if (changed || touch[method]) {
            sink.Delta(method, value);
        }
        sink.Quiet(table.shadow_scratch, regs[table.shadow_scratch]);
        return;
    case Action3D::DrawManager:
        if (changed || touch[method]) {
            sink.Delta(method, value);
        }
        if ((table.flags[method] & RegSem::InlineIndex) != 0) {
            sink.Inline(method, &value, 1);
        }
        return;
    default:
        // Binds, reports, barriers, uploads, root tables, render enable, sync points.
        sink.Exec(method, value, is_last);
        return;
    }
}

} // namespace VideoCommon::V2::S1
