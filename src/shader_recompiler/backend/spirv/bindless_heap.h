// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

// Pipeline v2 P3 (EDEN_NX2_V2_BINDLESS): sampled textures and storage images read from a host
// descriptor heap. The heap is descriptor set 1: binding 0 is an array of samplers, then arrays of
// sampled images (one per image class and component type), then arrays of storage images (one per
// dimension and component type). A descriptor marked in Info::heap_texture_mask or
// Info::heap_image_mask declares no binding of its own: each of its elements reads a word from
// the draw's heap word table, a storage buffer at set 0, binding HEAP_WORDS_BINDING, and takes
// its image (low 16 bits) and, for textures, its sampler (high 16 bits) from the heap.
// The host and the shader take every index below from these functions, so they agree.
namespace Shader::Backend::SPIRV {

inline constexpr u32 HEAP_SET = 1;
inline constexpr u32 HEAP_SAMPLER_BINDING = 0;
/// Sampled image classes: 1D, 1D array, 2D (and 2D rect), 2D array, 3D, cube, cube array, 2D MS,
/// 2D array MS.
inline constexpr u32 NUM_HEAP_IMAGE_CLASSES = 9;
/// Storage image classes: 1D, 1D array, 2D, 2D array, 3D.
inline constexpr u32 NUM_HEAP_STORAGE_CLASSES = 5;
inline constexpr u32 HEAP_FIRST_STORAGE_BINDING = 1 + NUM_HEAP_IMAGE_CLASSES * 2;
inline constexpr u32 NUM_HEAP_BINDINGS = HEAP_FIRST_STORAGE_BINDING + NUM_HEAP_STORAGE_CLASSES * 2;
/// Elements of every heap array (the shader's array type and the host's descriptor counts must
/// be equal).
inline constexpr u32 MAX_HEAP_ARRAY_LENGTH = 4096;
/// The draw's heap word table: a storage buffer of set 0 at a binding above any a pipeline's
/// descriptors take (pipelines whose own bindings would reach it don't use the heap).
inline constexpr u32 HEAP_WORDS_BINDING = 1023;

/// The image class of a sampled texture descriptor's type.
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

/// The heap binding of a sampled texture descriptor's images.
[[nodiscard]] constexpr u32 HeapImageBinding(TextureType type, bool is_multisample,
                                             bool is_integer) {
    return 1 + HeapImageClass(type, is_multisample) * 2 + (is_integer ? 1 : 0);
}

/// Whether a storage image of `type` can live in the heap (the classes above).
[[nodiscard]] constexpr bool HeapStorageType(TextureType type) {
    switch (type) {
    case TextureType::Color1D:
    case TextureType::ColorArray1D:
    case TextureType::Color2D:
    case TextureType::ColorArray2D:
    case TextureType::Color3D:
        return true;
    default:
        return false;
    }
}

/// The heap binding of a storage image descriptor's images.
[[nodiscard]] constexpr u32 HeapStorageBinding(TextureType type, bool is_integer) {
    u32 storage_class = 2;
    switch (type) {
    case TextureType::Color1D:
        storage_class = 0;
        break;
    case TextureType::ColorArray1D:
        storage_class = 1;
        break;
    case TextureType::ColorArray2D:
        storage_class = 3;
        break;
    case TextureType::Color3D:
        storage_class = 4;
        break;
    default:
        break;
    }
    return HEAP_FIRST_STORAGE_BINDING + storage_class * 2 + (is_integer ? 1 : 0);
}

/// Whether texture descriptor `index` of `info` reads the heap.
[[nodiscard]] constexpr bool UsesHeap(const Info& info, size_t index) {
    return index < 64 && ((info.heap_texture_mask >> index) & 1) != 0;
}

/// Whether storage image descriptor `index` of `info` reads the heap.
[[nodiscard]] constexpr bool UsesHeapImage(const Info& info, size_t index) {
    return index < 64 && ((info.heap_image_mask >> index) & 1) != 0;
}

/// The words a stage reads: its heap textures' elements, then its heap images' elements.
[[nodiscard]] inline u32 HeapWordCount(const Info& info) {
    u32 words = 0;
    for (size_t i = 0; i < info.texture_descriptors.size(); ++i) {
        words += UsesHeap(info, i) ? info.texture_descriptors[i].count : 0;
    }
    for (size_t i = 0; i < info.image_descriptors.size(); ++i) {
        words += UsesHeapImage(info, i) ? info.image_descriptors[i].count : 0;
    }
    return words;
}

/// The word of the first element of heap texture descriptor `index`.
[[nodiscard]] inline u32 HeapTextureWord(const Info& info, size_t index) {
    u32 word = info.heap_word_base;
    for (size_t i = 0; i < index; ++i) {
        word += UsesHeap(info, i) ? info.texture_descriptors[i].count : 0;
    }
    return word;
}

/// The word of the first element of heap storage image descriptor `index`.
[[nodiscard]] inline u32 HeapImageWord(const Info& info, size_t index) {
    u32 word = info.heap_word_base;
    for (size_t i = 0; i < info.texture_descriptors.size(); ++i) {
        word += UsesHeap(info, i) ? info.texture_descriptors[i].count : 0;
    }
    for (size_t i = 0; i < index; ++i) {
        word += UsesHeapImage(info, i) ? info.image_descriptors[i].count : 0;
    }
    return word;
}

} // namespace Shader::Backend::SPIRV
