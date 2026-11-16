// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <bit>

#include "common/common_types.h"

// P5b (EDEN_NX2_V2_GPUDECODE): the parameters and the per-byte address math of the generic
// block-linear → linear compute untile (host_shaders/block_linear_unswizzle_linear.comp). One
// segment is one (level, layer) of UnswizzleImage's loop: the shader produces exactly the bytes
// Tegra::Texture::UnswizzleTexture writes for it, into the same buffer layout, so the image is
// filled by the same BufferImageCopy list as the CPU path. This header is the C++ twin of the
// shader (the unit tests compare it with UnswizzleTexture).
namespace VideoCommon::V2::GpuUnswizzle {

/// Push constants of one segment (16 words; the shader declares the same order).
struct Segment {
    u32 src_offset;    ///< bytes into the swizzled input
    u32 dst_offset;    ///< bytes into the linear output (multiple of 4)
    u32 num_bytes;     ///< linear bytes of the segment (multiple of 4)
    u32 pitch;         ///< linear bytes per row (tiles across × bytes per block)
    u32 height;        ///< rows (tiles down)
    u32 block_height;  ///< log2 GOBs
    u32 block_depth;   ///< log2 GOBs
    u32 block_size;    ///< bytes of a block column of GOBs
    u32 slice_size;    ///< bytes of a slice of blocks
    u32 x_shift;
    u32 native_tiling;
    u32 pad[5];
};
static_assert(sizeof(Segment) == 64);

constexpr u32 GobSizeShift = 9;
constexpr u32 SwizzleXBits = 0b100101111;
constexpr u32 SwizzleYBits = 0b011010000;

[[nodiscard]] constexpr u32 Pdep(u32 value, u32 mask) noexcept {
    u32 result = 0;
    for (u32 bit = 1; mask != 0; bit += bit) {
        if (value & bit) {
            result |= mask & (~mask + 1);
        }
        mask &= mask - 1;
    }
    return result;
}

[[nodiscard]] constexpr u32 DivCeilLog2(u32 value, u32 log2) noexcept {
    return (value + (1U << log2) - 1) >> log2;
}

[[nodiscard]] constexpr u32 AlignUpLog2(u32 value, u32 log2) noexcept {
    return (value + (1U << log2) - 1) >> log2 << log2;
}

/// The segment for UnswizzleTexture(output, input, bytes_per_block, width, height, depth,
/// block_height, block_depth, stride_alignment, native) (width/height/depth in tiles).
[[nodiscard]] constexpr Segment MakeSegment(u32 src_offset, u32 dst_offset, u32 bytes_per_block,
                                            u32 width, u32 height, u32 depth, u32 block_height,
                                            u32 block_depth, u32 stride_alignment,
                                            bool native) noexcept {
    const u32 stride = AlignUpLog2(width, stride_alignment) * bytes_per_block;
    const u32 gobs_in_x = DivCeilLog2(stride, 6);
    const u32 block_size = gobs_in_x << (GobSizeShift + block_height + block_depth);
    Segment segment{};
    segment.src_offset = src_offset;
    segment.dst_offset = dst_offset;
    segment.pitch = width * bytes_per_block;
    segment.height = height;
    segment.num_bytes = segment.pitch * height * depth;
    segment.block_height = block_height;
    segment.block_depth = block_depth;
    segment.block_size = block_size;
    segment.slice_size = DivCeilLog2(height, block_height + 3) * block_size;
    segment.x_shift = GobSizeShift + block_height + block_depth;
    segment.native_tiling = native ? 1 : 0;
    return segment;
}

/// The swizzled byte (relative to src_offset) that linear byte `b` of the segment comes from.
[[nodiscard]] constexpr u32 SourceByte(const Segment& s, u32 b) noexcept {
    const u32 slice_bytes = s.pitch * s.height;
    const u32 slice = b / slice_bytes;
    const u32 rest = b - slice * slice_bytes;
    const u32 line = rest / s.pitch;
    const u32 x = rest - line * s.pitch;
    const u32 offset_z = (slice >> s.block_depth) * s.slice_size +
                         ((slice & ((1U << s.block_depth) - 1)) << (GobSizeShift + s.block_height));
    const u32 block_y = line >> 3;
    const u32 offset_y = (block_y >> s.block_height) * s.block_size +
                         ((block_y & ((1U << s.block_height) - 1)) << GobSizeShift);
    const u32 offset_x = (x >> 6) << s.x_shift;
    u32 offset = offset_z + offset_y + offset_x + (Pdep(x & 63, SwizzleXBits) | Pdep(line & 7, SwizzleYBits));
    if (s.native_tiling != 0) {
        const u32 difference = ((offset >> 5) ^ (offset >> 6)) & 1;
        offset ^= (difference << 5) | (difference << 6);
    }
    return offset;
}

/// Images below this many guest bytes stay on the CPU (dispatch and barrier cost).
constexpr u64 MinGpuBytes = 16 * 1024;

} // namespace VideoCommon::V2::GpuUnswizzle
