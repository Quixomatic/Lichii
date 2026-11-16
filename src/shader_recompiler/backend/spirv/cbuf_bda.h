// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <bit>

#include "common/common_types.h"
#include "shader_recompiler/backend/spirv/bindless_heap.h"
#include "shader_recompiler/shader_info.h"

// Pipeline v2 F2-02 (d) (EDEN_NX2_V2_CBUFBDA): constant buffers read through buffer device
// addresses (MASTER_PLAN §5.3 "draws reference {VA, size} through push data or BDA"). A constant
// buffer marked in Info::cbuf_bda_mask declares no uniform buffer binding; its loads take a 64-bit
// address and read the PhysicalStorageBuffer there. Addresses are numbered across the pipeline's
// stages in stage order, each stage's in cbuf index order from Info::cbuf_bda_base. When they fit
// the push constants after the rescaling, render area and heap words (Info::cbuf_bda_pushed) they
// are pushed at Info::cbuf_bda_offset; otherwise one pushed address there points at the draw's
// address table (one 64-bit address per index). Only constant buffers read at immediate offsets
// take an address (their reads stay below the size the host backs; a dynamically indexed one
// keeps its binding), so a pipeline can mix both kinds per slot. The host and the shader take
// every number below from these functions.
namespace Shader::Backend::SPIRV {

inline constexpr u32 CBUF_BDA_PUSH_END = 128;
inline constexpr u32 CBUF_BDA_ADDRESS_BYTES = 8;

/// Where a pipeline's addresses start: after the rescaling and render area layouts (48 bytes),
/// after the pushed heap words, or after the heap word table's pushed base (8-byte aligned).
[[nodiscard]] constexpr u32 CbufBdaPushOffset(bool heap, bool heap_words_pushed) {
    return !heap ? HEAP_PUSH_OFFSET
                 : (heap_words_pushed ? HEAP_PUSH_RANGE_END : HEAP_PUSH_OFFSET + 8);
}

/// The addresses that fit the push constants from `offset`.
[[nodiscard]] constexpr u32 CbufBdaPushCapacity(u32 offset) {
    return offset >= CBUF_BDA_PUSH_END ? 0 : (CBUF_BDA_PUSH_END - offset) / CBUF_BDA_ADDRESS_BYTES;
}

/// Whether constant buffer `index` of `info` can be read through an address: read, and only at
/// immediate offsets (a dynamic offset marks the whole 64 KiB bank as used).
[[nodiscard]] inline bool CbufBdaEligible(const Info& info, u32 index) {
    return index < Info::MAX_CBUFS && ((info.constant_buffer_mask >> index) & 1) != 0 &&
           info.constant_buffer_used_sizes[index] != 0 &&
           info.constant_buffer_used_sizes[index] < 0x10000;
}

[[nodiscard]] constexpr bool UsesCbufBda(const Info& info) {
    return info.cbuf_bda_mask != 0;
}

/// Whether constant buffer `index` of `info` is read through an address.
[[nodiscard]] constexpr bool UsesCbufBda(const Info& info, u32 index) {
    return index < 32 && ((info.cbuf_bda_mask >> index) & 1) != 0;
}

/// The addresses a stage reads.
[[nodiscard]] inline u32 CbufBdaCount(const Info& info) {
    return static_cast<u32>(std::popcount(info.cbuf_bda_mask));
}

/// The address index of constant buffer `index` (marked in the mask).
[[nodiscard]] inline u32 CbufBdaIndex(const Info& info, u32 index) {
    const u32 below = index >= 32 ? info.cbuf_bda_mask : info.cbuf_bda_mask & ((1U << index) - 1);
    return info.cbuf_bda_base + static_cast<u32>(std::popcount(below));
}

/// The uniform buffer bindings a stage declares: its constant buffer descriptors' elements, less
/// the ones read through addresses unless they keep their bindings.
[[nodiscard]] inline u32 CbufBoundCount(const Info& info) {
    u32 count = 0;
    for (const ConstantBufferDescriptor& desc : info.constant_buffer_descriptors) {
        if (info.cbuf_bda_keep_bindings || !UsesCbufBda(info, desc.index)) {
            count += desc.count;
        }
    }
    return count;
}

/// The constant buffer descriptors of a stage that take no binding (no uniform buffer entry in
/// the layout or the payload).
[[nodiscard]] inline u32 CbufUnboundCount(const Info& info) {
    return info.cbuf_bda_keep_bindings ? 0 : CbufBdaCount(info);
}

/// Marks a pipeline's stages (in stage order, after their heap marks): every eligible constant
/// buffer reads through an address, numbered across the stages; the addresses are pushed when
/// they fit after the heap words, else the table's pointer is. Returns the pipeline's addresses.
template <typename Infos>
u32 AssignCbufBda(Infos&& infos, bool keep_bindings) {
    bool heap = false;
    bool heap_pushed = false;
    for (const Info* info : infos) {
        heap |= UsesAnyHeap(*info);
        heap_pushed |= info->heap_words_pushed;
    }
    u32 count = 0;
    for (Info* info : infos) {
        info->cbuf_bda_mask = 0;
        info->cbuf_bda_base = count;
        for (const ConstantBufferDescriptor& desc : info->constant_buffer_descriptors) {
            if (CbufBdaEligible(*info, desc.index)) {
                info->cbuf_bda_mask |= 1U << desc.index;
            }
        }
        count += CbufBdaCount(*info);
    }
    const u32 offset = CbufBdaPushOffset(heap, heap_pushed);
    const bool pushed = count <= CbufBdaPushCapacity(offset);
    for (Info* info : infos) {
        info->cbuf_bda_offset = offset;
        info->cbuf_bda_pushed = pushed;
        info->cbuf_bda_keep_bindings = keep_bindings;
    }
    return count;
}

} // namespace Shader::Backend::SPIRV
