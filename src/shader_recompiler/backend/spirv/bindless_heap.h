// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <bit>

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

// Pipeline v2 P3 (EDEN_NX2_V2_BINDLESS): sampled textures read from a host descriptor heap. The
// heap is descriptor set 1: binding 0 is an array of samplers, the other bindings arrays of
// sampled images, one per image class (dimensions, arrayed, multisampled) and component type.
// A texture descriptor marked in Info::heap_texture_mask declares no binding of its own: its
// image and sampler are elements of those arrays, selected by a word the draw pushes (image
// element in the low 16 bits, sampler element in the high 16 bits) at HEAP_PUSH_OFFSET.
// The host and the shader take every index below from these functions, so they agree.
namespace Shader::Backend::SPIRV {

inline constexpr u32 HEAP_SET = 1;
inline constexpr u32 HEAP_SAMPLER_BINDING = 0;
/// Classes: 1D, 1D array, 2D (and 2D rect), 2D array, 3D, cube, cube array, 2D MS, 2D array MS.
inline constexpr u32 NUM_HEAP_IMAGE_CLASSES = 9;
inline constexpr u32 NUM_HEAP_BINDINGS = 1 + NUM_HEAP_IMAGE_CLASSES * 2;
/// The heap words follow RescalingLayout and RenderAreaLayout in the push constants.
inline constexpr u32 HEAP_PUSH_OFFSET = 48;
/// Heap textures per pipeline: 16 words keep the push constants within the 128 bytes every
/// Vulkan device offers.
inline constexpr u32 MAX_HEAP_TEXTURES = 16;
inline constexpr u32 HEAP_PUSH_SIZE = MAX_HEAP_TEXTURES * sizeof(u32);
/// Elements of every heap array, images and samplers (the shader's array type and the host's
/// descriptor counts must be equal).
inline constexpr u32 MAX_HEAP_ARRAY_LENGTH = 4096;

/// The image class of a texture descriptor's type.
[[nodiscard]] constexpr u32 HeapImageClass(TextureType type, bool is_multisample) {
    switch (type) {
    case TextureType::Color1D:
        return 0;
    case TextureType::ColorArray1D:
        return 1;
    case TextureType::Color2D:
    case TextureType::Color2DRect:
        return is_multisample ? 7 : 2;
    case TextureType::ColorArray2D:
        return is_multisample ? 8 : 3;
    case TextureType::Color3D:
        return 4;
    case TextureType::ColorCube:
        return 5;
    case TextureType::ColorArrayCube:
        return 6;
    case TextureType::Buffer:
        break;
    }
    return 2;
}

/// The heap binding of a texture descriptor's images.
[[nodiscard]] constexpr u32 HeapImageBinding(TextureType type, bool is_multisample,
                                             bool is_integer) {
    return 1 + HeapImageClass(type, is_multisample) * 2 + (is_integer ? 1 : 0);
}

/// Whether texture descriptor `index` of `info` reads the heap.
[[nodiscard]] constexpr bool UsesHeap(const Info& info, size_t index) {
    return index < 64 && ((info.heap_texture_mask >> index) & 1) != 0;
}

/// The push word of heap texture descriptor `index` of `info`.
[[nodiscard]] constexpr u32 HeapPushIndex(const Info& info, size_t index) {
    const u64 below = index == 0 ? 0 : (info.heap_texture_mask & ((u64{1} << index) - 1));
    return info.heap_push_base + static_cast<u32>(std::popcount(below));
}

} // namespace Shader::Backend::SPIRV
