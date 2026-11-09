// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "shader_recompiler/shader_info.h"

// Pipeline v2 P3 (EDEN_NX2_V2_BINDLESS): sampled textures and storage images read from a host
// descriptor heap. The heap is descriptor set 1: binding 0 is an array of samplers, then arrays of
// sampled images (one per image class and component type), then arrays of storage images (one per
// dimension and component type), then (P5a, EDEN_NX2_V2_HEAP_TEXEL) an array of uniform texel
// buffers and two of storage texel buffers (float, integer). A descriptor marked in
// Info::heap_texture_mask, heap_image_mask, heap_texel_mask or heap_image_buffer_mask declares no
// binding of its own: each of its elements reads a word, and takes its image or texel buffer (low
// 16 bits) and, for textures, its sampler (high 16 bits) from the heap. The words are push
// constants at HEAP_PUSH_OFFSET when the pipeline's words fit (Info::heap_words_pushed: no load
// before the descriptor fetch, a stable descriptor payload); otherwise they live in the heap word
// table, a storage buffer at set 0, binding HEAP_WORDS_BINDING, bound once per table chunk, and
// the push constant at HEAP_PUSH_OFFSET is the draw's first word in it. Per stage the words are
// the heap texture buffers' elements, the heap image buffers', the heap textures', then the heap
// storage images' (the order the host binds them in). The host and the shader take every index
// below from these functions, so they agree.
namespace Shader::Backend::SPIRV {

inline constexpr u32 HEAP_SET = 1;
inline constexpr u32 HEAP_SAMPLER_BINDING = 0;
/// Sampled image classes: 1D, 1D array, 2D (and 2D rect), 2D array, 3D, cube, cube array, 2D MS,
/// 2D array MS.
inline constexpr u32 NUM_HEAP_IMAGE_CLASSES = 9;
/// Storage image classes: 1D, 1D array, 2D, 2D array, 3D.
inline constexpr u32 NUM_HEAP_STORAGE_CLASSES = 5;
inline constexpr u32 HEAP_FIRST_STORAGE_BINDING = 1 + NUM_HEAP_IMAGE_CLASSES * 2;
/// P5a: texel buffers (uniform; then storage float, storage integer).
inline constexpr u32 HEAP_UNIFORM_TEXEL_BINDING =
    HEAP_FIRST_STORAGE_BINDING + NUM_HEAP_STORAGE_CLASSES * 2;
inline constexpr u32 HEAP_FIRST_STORAGE_TEXEL_BINDING = HEAP_UNIFORM_TEXEL_BINDING + 1;
inline constexpr u32 NUM_HEAP_BINDINGS = HEAP_FIRST_STORAGE_TEXEL_BINDING + 2;
/// Elements of every heap array (the shader's array type and the host's descriptor counts must
/// be equal).
inline constexpr u32 MAX_HEAP_ARRAY_LENGTH = 4096;
/// The draw's heap word table: a storage buffer of set 0 at a binding above any a pipeline's
/// descriptors take (pipelines whose own bindings would reach it don't use the heap).
inline constexpr u32 HEAP_WORDS_BINDING = 1023;
/// The heap push constants: after the rescaling and render area layouts (48 bytes), up to
/// HEAP_PUSH_WORDS words (or one, the table base), within Vulkan's guaranteed 128 bytes.
inline constexpr u32 HEAP_PUSH_OFFSET = 48;
inline constexpr u32 HEAP_PUSH_WORDS = 16;
/// The push constant range a heap pipeline layout needs.
inline constexpr u32 HEAP_PUSH_RANGE_END = HEAP_PUSH_OFFSET + HEAP_PUSH_WORDS * 4;
static_assert(HEAP_PUSH_RANGE_END <= 128);

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

/// The heap binding of a storage texel buffer (image buffer) descriptor's elements.
[[nodiscard]] constexpr u32 HeapStorageTexelBinding(bool is_integer) {
    return HEAP_FIRST_STORAGE_TEXEL_BINDING + (is_integer ? 1 : 0);
}

/// Whether any descriptor of `info` reads the heap.
[[nodiscard]] constexpr bool UsesAnyHeap(const Info& info) {
    return info.heap_texture_mask != 0 || info.heap_image_mask != 0 ||
           info.heap_texel_mask != 0 || info.heap_image_buffer_mask != 0;
}

/// Whether texture descriptor `index` of `info` reads the heap.
[[nodiscard]] constexpr bool UsesHeap(const Info& info, size_t index) {
    return index < 64 && ((info.heap_texture_mask >> index) & 1) != 0;
}

/// Whether storage image descriptor `index` of `info` reads the heap.
[[nodiscard]] constexpr bool UsesHeapImage(const Info& info, size_t index) {
    return index < 64 && ((info.heap_image_mask >> index) & 1) != 0;
}

/// Whether texture buffer descriptor `index` of `info` reads the heap.
[[nodiscard]] constexpr bool UsesHeapTexel(const Info& info, size_t index) {
    return index < 64 && ((info.heap_texel_mask >> index) & 1) != 0;
}

/// Whether image buffer descriptor `index` of `info` reads the heap.
[[nodiscard]] constexpr bool UsesHeapImageBuffer(const Info& info, size_t index) {
    return index < 64 && ((info.heap_image_buffer_mask >> index) & 1) != 0;
}

/// The words a stage's heap texel buffers take (they come first).
[[nodiscard]] inline u32 HeapTexelWordCount(const Info& info) {
    u32 words = 0;
    for (size_t i = 0; i < info.texture_buffer_descriptors.size(); ++i) {
        words += UsesHeapTexel(info, i) ? info.texture_buffer_descriptors[i].count : 0;
    }
    for (size_t i = 0; i < info.image_buffer_descriptors.size(); ++i) {
        words += UsesHeapImageBuffer(info, i) ? info.image_buffer_descriptors[i].count : 0;
    }
    return words;
}

/// The word of the first element of heap texture buffer descriptor `index`.
[[nodiscard]] inline u32 HeapTexelWord(const Info& info, size_t index) {
    u32 word = info.heap_word_base;
    for (size_t i = 0; i < index; ++i) {
        word += UsesHeapTexel(info, i) ? info.texture_buffer_descriptors[i].count : 0;
    }
    return word;
}

/// The word of the first element of heap image buffer descriptor `index`.
[[nodiscard]] inline u32 HeapImageBufferWord(const Info& info, size_t index) {
    u32 word = info.heap_word_base;
    for (size_t i = 0; i < info.texture_buffer_descriptors.size(); ++i) {
        word += UsesHeapTexel(info, i) ? info.texture_buffer_descriptors[i].count : 0;
    }
    for (size_t i = 0; i < index; ++i) {
        word += UsesHeapImageBuffer(info, i) ? info.image_buffer_descriptors[i].count : 0;
    }
    return word;
}

/// The words a stage reads: its heap texel buffers', its heap textures', then its heap images'
/// elements.
[[nodiscard]] inline u32 HeapWordCount(const Info& info) {
    u32 words = HeapTexelWordCount(info);
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
    u32 word = info.heap_word_base + HeapTexelWordCount(info);
    for (size_t i = 0; i < index; ++i) {
        word += UsesHeap(info, i) ? info.texture_descriptors[i].count : 0;
    }
    return word;
}

/// The word of the first element of heap storage image descriptor `index`.
[[nodiscard]] inline u32 HeapImageWord(const Info& info, size_t index) {
    u32 word = info.heap_word_base + HeapTexelWordCount(info);
    for (size_t i = 0; i < info.texture_descriptors.size(); ++i) {
        word += UsesHeap(info, i) ? info.texture_descriptors[i].count : 0;
    }
    for (size_t i = 0; i < index; ++i) {
        word += UsesHeapImage(info, i) ? info.image_descriptors[i].count : 0;
    }
    return word;
}

} // namespace Shader::Backend::SPIRV
