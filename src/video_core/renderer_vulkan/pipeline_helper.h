// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "shader_recompiler/backend/spirv/bindless_heap.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/vk_bindless_heap.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/surface.h"
#include "video_core/v2/p3_stats.h"
#include "video_core/texture_cache/types.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

using Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS;

[[nodiscard]] inline std::optional<PixelFormat> PixelFormatFromImageFormat(
    Shader::ImageFormat format) {
    switch (format) {
    case Shader::ImageFormat::Typeless:
        return std::nullopt;
    case Shader::ImageFormat::R8_UINT:
        return PixelFormat::R8_UINT;
    case Shader::ImageFormat::R8_SINT:
        return PixelFormat::R8_SINT;
    case Shader::ImageFormat::R16_UINT:
        return PixelFormat::R16_UINT;
    case Shader::ImageFormat::R16_SINT:
        return PixelFormat::R16_SINT;
    case Shader::ImageFormat::R32_UINT:
        return PixelFormat::R32_UINT;
    case Shader::ImageFormat::R32G32_UINT:
        return PixelFormat::R32G32_UINT;
    case Shader::ImageFormat::R32G32B32A32_UINT:
        return PixelFormat::R32G32B32A32_UINT;
    }
    return std::nullopt;
}

[[nodiscard]] inline VkDeviceSize DescriptorSizeForType(const Device& device,
                                                        VkDescriptorType type) {
    const auto& props = device.DescriptorBufferProperties();
    const bool robust = device.IsRobustBufferAccessEnabled();
    switch (type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        return robust ? props.robustUniformBufferDescriptorSize : props.uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        return robust ? props.robustStorageBufferDescriptorSize : props.storageBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        return robust ? props.robustUniformTexelBufferDescriptorSize
                      : props.uniformTexelBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        return robust ? props.robustStorageTexelBufferDescriptorSize
                      : props.storageTexelBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        return props.combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        return props.storageImageDescriptorSize;
    default:
        return 0;
    }
}

struct DescriptorBufferBinding {
    VkDescriptorType type;
    u32 count;
    VkDeviceSize offset;
    VkDeviceSize stride;
};

struct DescriptorBufferLayout {
    VkDeviceSize size{};
    boost::container::small_vector<DescriptorBufferBinding, 32> bindings;

    [[nodiscard]] bool Empty() const noexcept {
        return bindings.empty();
    }
};

inline void WriteDescriptorBuffer(const Device& device, const DescriptorBufferLayout& layout,
                                  const DescriptorUpdateEntry* payload, u8* host) {
    const vk::Device& dev = device.GetLogical();
    for (const DescriptorBufferBinding& binding : layout.bindings) {
        for (u32 index = 0; index < binding.count; ++index) {
            const DescriptorUpdateEntry& entry = *(payload++);
            const VkDescriptorAddressInfoEXT address_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                .pNext = nullptr,
                .address = entry.address.address,
                .range = entry.address.range,
                .format = entry.address.format,
            };
            VkDescriptorGetInfoEXT get_info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                .pNext = nullptr,
                .type = binding.type,
                .data{},
            };
            switch (binding.type) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                get_info.data.pUniformBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                get_info.data.pStorageBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
                get_info.data.pUniformTexelBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                get_info.data.pStorageTexelBuffer = &address_info;
                break;
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
                get_info.data.pCombinedImageSampler = &entry.image;
                break;
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                get_info.data.pStorageImage = &entry.image;
                break;
            default:
                continue;
            }
            dev.GetDescriptorEXT(get_info, binding.stride,
                                 host + binding.offset + index * binding.stride);
        }
    }
}

[[nodiscard]] inline u32 NumDescriptorEntries(const Shader::Info& info) {
    // Heap textures and images (EDEN_NX2_V2_BINDLESS) have no entry of their own.
    return Shader::NumDescriptors(info.constant_buffer_descriptors) +
           Shader::NumDescriptors(info.storage_buffers_descriptors) +
           Shader::NumDescriptors(info.texture_buffer_descriptors) +
           Shader::NumDescriptors(info.image_buffer_descriptors) +
           Shader::NumDescriptors(info.texture_descriptors) +
           Shader::NumDescriptors(info.image_descriptors) -
           Shader::Backend::SPIRV::HeapWordCount(info);
}

/// EDEN_NX2_V2_BINDLESS: why descriptors stay out of the heap (Info::heap_block_reasons).
enum HeapReason : u32 {
    HeapReasonMeshEmulation = 1 << 0,  ///< mesh programs run as compute: their own layout
    HeapReasonTexelBuffers = 1 << 1,   ///< texel buffers keep descriptors
    HeapReasonByteAddressed = 1 << 2,  ///< byte-addressed storage views (compute shadow copies)
    HeapReasonNoStorage = 1 << 3,      ///< device lacks update-after-bind storage images
    HeapReasonStorageType = 1 << 4,    ///< cube or buffer storage images
    HeapReasonBindings = 1 << 5,       ///< past 64 descriptors of a kind, or set 0 too large
    NumHeapReasons = 6,
};

/// EDEN_NX2_V2_BINDLESS stats: a draw or dispatch with images, of a pipeline that reads `words`
/// heap words and left descriptors out for `reasons`.
inline void CountHeapCoverage(bool heap_pipeline, u32 reasons, u32 words, bool compute) {
    auto& data = VideoCommon::V2::P3Stats::Get();
    if (heap_pipeline && reasons == 0) {
        ++(compute ? data.heap_dispatches : data.heap_draws);
    } else if (heap_pipeline) {
        ++(compute ? data.heap_partial_dispatches : data.heap_partial_draws);
    } else {
        ++(compute ? data.heap_fallback_dispatches : data.heap_fallback_draws);
    }
    data.heap_words += words;
    for (u32 reason = 0; reason < NumHeapReasons; ++reason) {
        data.heap_reasons[reason] += (reasons >> reason) & 1;
    }
}

/// EDEN_NX2_V2_BINDLESS: the heap words of a draw or dispatch, in Info word order.
struct HeapPushConstant {
    std::vector<u32> words;
    BindlessHeap* heap{};
    bool verify{};
};

class DescriptorLayoutBuilder {
public:
    DescriptorLayoutBuilder(const Device& device_) : device{&device_} {}

    bool CanUsePushDescriptor() const noexcept {
        if (!device->IsKhrPushDescriptorSupported() ||
            num_descriptors > device->MaxPushDescriptors()) {
            return false;
        }
        return !device->IsExtDescriptorBufferSupported() ||
               device->DescriptorBufferProperties().bufferlessPushDescriptors;
    }

    bool CanUseDescriptorBuffer() const noexcept {
        const auto& props = device->DescriptorBufferProperties();
        if (!device->IsExtDescriptorBufferSupported() || bindings.empty() ||
            !props.combinedImageSamplerDescriptorSingleArray) {
            return false;
        }
        return !props.bufferlessPushDescriptors || !CanUsePushDescriptor();
    }

    DescriptorBufferLayout MakeDescriptorBufferLayout(VkDescriptorSetLayout layout) const {
        DescriptorBufferLayout result;
        if (!layout) {
            return result;
        }
        const vk::Device& dev = device->GetLogical();
        result.size = dev.GetDescriptorSetLayoutSizeEXT(layout);
        result.bindings.reserve(bindings.size());
        for (const VkDescriptorSetLayoutBinding& entry : bindings) {
            result.bindings.push_back(DescriptorBufferBinding{
                .type = entry.descriptorType,
                .count = entry.descriptorCount,
                .offset = dev.GetDescriptorSetLayoutBindingOffsetEXT(layout, entry.binding),
                .stride = DescriptorSizeForType(*device, entry.descriptorType),
            });
        }
        return result;
    }

    vk::DescriptorSetLayout CreateDescriptorSetLayout(bool use_push_descriptor,
                                                      bool use_descriptor_buffer = false) const {
        if (bindings.empty()) {
            return nullptr;
        }
        VkDescriptorSetLayoutCreateFlags flags = 0;
        if (use_push_descriptor) {
            flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        }
        if (use_descriptor_buffer) {
            flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
        }
        boost::container::small_vector<VkDescriptorBindingFlags, 32> binding_flags;
        VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{};
        const void* pnext = nullptr;
        if (!use_push_descriptor && device->IsDescriptorBindingPartiallyBoundSupported()) {
            binding_flags.assign(bindings.size(), VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
            binding_flags_ci = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                .pNext = nullptr,
                .bindingCount = static_cast<u32>(binding_flags.size()),
                .pBindingFlags = binding_flags.data(),
            };
            pnext = &binding_flags_ci;
        }
        return device->GetLogical().CreateDescriptorSetLayout({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = pnext,
            .flags = flags,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        });
    }

    vk::DescriptorUpdateTemplate CreateTemplate(VkDescriptorSetLayout descriptor_set_layout,
                                                VkPipelineLayout pipeline_layout,
                                                bool use_push_descriptor) const {
        if (entries.empty()) {
            return nullptr;
        }
        const VkDescriptorUpdateTemplateType type =
            use_push_descriptor ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
                                : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        return device->GetLogical().CreateDescriptorUpdateTemplate({
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .descriptorUpdateEntryCount = static_cast<u32>(entries.size()),
            .pDescriptorUpdateEntries = entries.data(),
            .templateType = type,
            .descriptorSetLayout = descriptor_set_layout,
            .pipelineBindPoint =
                is_compute ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
            .pipelineLayout = pipeline_layout,
            .set = 0,
        });
    }

    /// With `heap` (EDEN_NX2_V2_BINDLESS), set 1 is the heap.
    vk::PipelineLayout CreatePipelineLayout(VkDescriptorSetLayout descriptor_set_layout,
                                            const BindlessHeap* heap = nullptr) const {
        using Shader::Backend::SPIRV::RenderAreaLayout;
        using Shader::Backend::SPIRV::RescalingLayout;
        const u32 size_offset = is_compute ? sizeof(RescalingLayout::down_factor) : 0u;
        if (heap != nullptr) [[unlikely]] {
            const std::array set_layouts{descriptor_set_layout ? descriptor_set_layout
                                                               : heap->EmptyLayout(),
                                         heap->Layout()};
            const VkPushConstantRange heap_range{
                .stageFlags = static_cast<VkShaderStageFlags>(
                    is_compute ? VK_SHADER_STAGE_COMPUTE_BIT
                               : (uses_mesh ? VK_SHADER_STAGE_ALL_GRAPHICS |
                                                  VK_SHADER_STAGE_TASK_BIT_EXT |
                                                  VK_SHADER_STAGE_MESH_BIT_EXT
                                            : VK_SHADER_STAGE_ALL_GRAPHICS)),
                .offset = 0,
                .size = static_cast<u32>(sizeof(RescalingLayout)) - size_offset +
                        static_cast<u32>(sizeof(RenderAreaLayout)),
            };
            return device->GetLogical().CreatePipelineLayout({
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .setLayoutCount = static_cast<u32>(set_layouts.size()),
                .pSetLayouts = set_layouts.data(),
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &heap_range,
            });
        }
        const VkPushConstantRange range{
            .stageFlags = static_cast<VkShaderStageFlags>(
                is_compute ? VK_SHADER_STAGE_COMPUTE_BIT
                : mesh_emulation
                    ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT
                    : (uses_mesh ? VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_EXT |
                                       VK_SHADER_STAGE_MESH_BIT_EXT
                                 : VK_SHADER_STAGE_ALL_GRAPHICS)),
            .offset = 0,
            .size = static_cast<u32>(sizeof(RescalingLayout)) - size_offset +
                    static_cast<u32>(sizeof(RenderAreaLayout)),
        };
        return device->GetLogical().CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = descriptor_set_layout ? 1U : 0U,
            .pSetLayouts = bindings.empty() ? nullptr : &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
    }

    /// Mesh emulation: a graphics pipeline layout whose mesh program runs as a compute shader
    /// (its bindings carry VK_SHADER_STAGE_COMPUTE_BIT and so does the push constant range).
    void EnableMeshEmulation() noexcept {
        mesh_emulation = true;
    }

    /// Mesh emulation: the output storage buffer the compute shader writes and the vertex
    /// shader reads, at an explicit binding (after every program's bindings).
    void AddMeshEmulationOutput(u32 output_binding) {
        bindings.push_back({
            .binding = output_binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        });
        entries.push_back({
            .dstBinding = output_binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .offset = offset,
            .stride = sizeof(DescriptorUpdateEntry),
        });
        binding = (std::max)(binding, output_binding + 1);
        ++num_descriptors;
        offset += sizeof(DescriptorUpdateEntry);
    }

    void Add(const Shader::Info& info, VkShaderStageFlags stage) {
        is_compute |= (stage & VK_SHADER_STAGE_COMPUTE_BIT) != 0 && !mesh_emulation;
        uses_mesh |= (stage & (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) != 0;

        Add(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, stage, info.constant_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, stage, info.storage_buffers_descriptors);
        Add(VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, stage, info.texture_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, stage, info.image_buffer_descriptors);
        Add(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, stage, info.texture_descriptors,
            info.heap_texture_mask);
        Add(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, stage, info.image_descriptors,
            info.heap_image_mask);
        heap_stages |= stage;
    }

    /// EDEN_NX2_V2_BINDLESS: the draw's heap word table (a storage buffer at
    /// HEAP_WORDS_BINDING, last in the payload). False when the pipeline's own bindings reach it.
    bool AddHeapWords() {
        using Shader::Backend::SPIRV::HEAP_WORDS_BINDING;
        if (binding >= HEAP_WORDS_BINDING) {
            return false;
        }
        bindings.push_back({
            .binding = HEAP_WORDS_BINDING,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = heap_stages,
            .pImmutableSamplers = nullptr,
        });
        entries.push_back({
            .dstBinding = HEAP_WORDS_BINDING,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .offset = offset,
            .stride = sizeof(DescriptorUpdateEntry),
        });
        binding = HEAP_WORDS_BINDING + 1;
        ++num_descriptors;
        offset += sizeof(DescriptorUpdateEntry);
        return true;
    }

private:
    VkShaderStageFlags heap_stages{};
    /// `skip`: descriptors (by index) that read the heap instead (EDEN_NX2_V2_BINDLESS).
    template <typename Descriptors>
    void Add(VkDescriptorType type, VkShaderStageFlags stage, const Descriptors& descriptors,
             u64 skip = 0) {
        const size_t num{descriptors.size()};
        for (size_t i = 0; i < num; ++i) {
            if (i < 64 && ((skip >> i) & 1) != 0) [[unlikely]] {
                continue;
            }
            bindings.push_back({
                .binding = binding,
                .descriptorType = type,
                .descriptorCount = descriptors[i].count,
                .stageFlags = stage,
                .pImmutableSamplers = nullptr,
            });
            entries.push_back({
                .dstBinding = binding,
                .dstArrayElement = 0,
                .descriptorCount = descriptors[i].count,
                .descriptorType = type,
                .offset = offset,
                .stride = sizeof(DescriptorUpdateEntry),
            });
            ++binding;
            num_descriptors += descriptors[i].count;
            offset += sizeof(DescriptorUpdateEntry) * descriptors[i].count;
        }
    }

    const Device* device{};
    bool is_compute{};
    bool uses_mesh{};
    bool mesh_emulation{};
    boost::container::small_vector<VkDescriptorSetLayoutBinding, 32> bindings;
    boost::container::small_vector<VkDescriptorUpdateTemplateEntry, 32> entries;
    u32 binding{};
    u32 num_descriptors{};
    size_t offset{};
};

class RescalingPushConstant {
public:
    explicit RescalingPushConstant() noexcept {}

    void PushTexture(bool is_rescaled) noexcept {
        *texture_ptr |= is_rescaled ? texture_bit : 0u;
        texture_bit <<= 1u;
        if (texture_bit == 0u) {
            texture_bit = 1u;
            ++texture_ptr;
        }
    }

    void PushImage(bool is_rescaled) noexcept {
        *image_ptr |= is_rescaled ? image_bit : 0u;
        image_bit <<= 1u;
        if (image_bit == 0u) {
            image_bit = 1u;
            ++image_ptr;
        }
    }

    const std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS>& Data() const noexcept {
        return words;
    }

private:
    std::array<u32, NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> words{};
    u32* texture_ptr{words.data()};
    u32* image_ptr{words.data() + Shader::Backend::SPIRV::NUM_TEXTURE_SCALING_WORDS};
    u32 texture_bit{1u};
    u32 image_bit{1u};
};

class RenderAreaPushConstant {
public:
    bool uses_render_area{};
    std::array<f32, 4> words{};
};

inline void PushImageDescriptors(TextureCache& texture_cache,
                                 GuestDescriptorQueue& guest_descriptor_queue,
                                 const Shader::Info& info, RescalingPushConstant& rescaling,
                                 const VideoCommon::SamplerId*& samplers,
                                 const VideoCommon::ImageViewInOut*& views,
                                 HeapPushConstant* heap = nullptr) {
    const u32 num_texture_buffers = Shader::NumDescriptors(info.texture_buffer_descriptors);
    const u32 num_image_buffers = Shader::NumDescriptors(info.image_buffer_descriptors);
    views += num_texture_buffers;
    views += num_image_buffers;
    namespace SPIRV = Shader::Backend::SPIRV;
    for (size_t desc_index = 0; desc_index < info.texture_descriptors.size(); ++desc_index) {
        const auto& desc{info.texture_descriptors[desc_index]};
        const bool in_heap{heap != nullptr && SPIRV::UsesHeap(info, desc_index)};
        bool is_rescaled{};
        for (u32 index = 0; index < desc.count; ++index) {
            const VideoCommon::ImageViewId image_view_id{(views++)->id};
            const VideoCommon::SamplerId sampler_id{*(samplers++)};
            ImageView& image_view{texture_cache.GetImageView(image_view_id)};
            VkImageView vk_image_view{image_view.Handle(desc.type)};
            if (vk_image_view == VK_NULL_HANDLE) {
                const VkImageView null_image_view{texture_cache.GetImageView(VideoCommon::NULL_IMAGE_VIEW_ID).Handle(desc.type)};
                if (null_image_view != VK_NULL_HANDLE) vk_image_view = null_image_view;
            }
            Sampler& sampler{texture_cache.GetSampler(sampler_id)};
            const VkSampler vk_sampler{sampler.HandleFor(image_view, desc.is_depth)};
            if (in_heap) [[unlikely]] {
                // EDEN_NX2_V2_BINDLESS: the heap elements of this view and sampler (written when
                // first seen), as the element's word in the table instead of a descriptor.
                const u32 binding{
                    SPIRV::HeapImageBinding(desc.type, desc.is_multisample, desc.is_integer)};
                const u32 image_slot{heap->heap->ImageSlot(binding, vk_image_view)};
                const u32 sampler_slot{heap->heap->SamplerSlot(vk_sampler)};
                heap->words.push_back(image_slot | (sampler_slot << 16));
                if (heap->verify) {
                    // What the old path would bind against what the heap holds there.
                    auto& data = VideoCommon::V2::P3Stats::Get();
                    ++data.heap_verify;
                    if (heap->heap->ImageAt(binding, image_slot) != vk_image_view ||
                        heap->heap->SamplerAt(sampler_slot) != vk_sampler) {
                        ++data.heap_mismatches;
                    }
                }
            } else {
                guest_descriptor_queue.AddSampledImage(vk_image_view, vk_sampler);
            }
            const bool element_rescaled{texture_cache.IsRescaling(image_view)};
            is_rescaled |= element_rescaled;
        }
        rescaling.PushTexture(is_rescaled);
    }
    for (size_t desc_index = 0; desc_index < info.image_descriptors.size(); ++desc_index) {
        const auto& desc{info.image_descriptors[desc_index]};
        const bool in_heap{heap != nullptr && SPIRV::UsesHeapImage(info, desc_index)};
        bool is_rescaled{};
        for (u32 index = 0; index < desc.count; ++index) {
            ImageView& image_view{texture_cache.GetImageView((views++)->id)};
            const VkImageView vk_image_view{
                image_view.StorageView(desc.type, desc.format, desc.is_byte_addressed)};
            if (desc.is_written) {
                texture_cache.MarkModification(image_view.image_id);
            }
            if (in_heap) [[unlikely]] {
                const u32 binding{SPIRV::HeapStorageBinding(desc.type, desc.is_integer)};
                const u32 slot{heap->heap->ImageSlot(binding, vk_image_view)};
                heap->words.push_back(slot);
                if (heap->verify) {
                    auto& data = VideoCommon::V2::P3Stats::Get();
                    ++data.heap_verify;
                    data.heap_mismatches += heap->heap->ImageAt(binding, slot) != vk_image_view;
                }
            } else {
                guest_descriptor_queue.AddImage(vk_image_view);
            }
            const bool element_rescaled{texture_cache.IsRescaling(image_view)};
            is_rescaled |= element_rescaled;
        }
        rescaling.PushImage(is_rescaled);
    }
}

} // namespace Vulkan
