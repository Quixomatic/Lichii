// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "shader_recompiler/backend/spirv/bindless_heap.h"
#include "video_core/v2/commit.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;

/// Pipeline v2 P3 (EDEN_NX2_V2_BINDLESS): the host descriptor heap that sampled textures of heap
/// pipelines read (layout in shader_recompiler/backend/spirv/bindless_heap.h). One descriptor set
/// with update-after-bind arrays: a slot is written once, when a view or sampler is first seen,
/// never per draw, and a slot is only rewritten for another view or sampler once the GPU is done
/// with every draw that used it (the update-unused-while-pending rule). Each Vulkan image view
/// and sampler has its own slot, so texture cache aliasing, rescaling and format reinterpretation
/// (which make new views) reach the heap by themselves; the texture cache forgets a view's slots
/// when it destroys the view.
class BindlessHeap {
public:
    static constexpr u32 Capacity = Shader::Backend::SPIRV::MAX_HEAP_ARRAY_LENGTH;
    static constexpr u32 NumBindings = Shader::Backend::SPIRV::NUM_HEAP_BINDINGS;
    static constexpr u32 SampledBindings = Shader::Backend::SPIRV::NUM_HEAP_IMAGE_CLASSES * 2;
    static constexpr u32 StorageBindings = Shader::Backend::SPIRV::NUM_HEAP_STORAGE_CLASSES * 2;
    static constexpr u32 TexelBindings =
        Shader::Backend::SPIRV::NUM_HEAP_BINDINGS - Shader::Backend::SPIRV::HEAP_UNIFORM_TEXEL_BINDING;

    /// Whether a heap binding holds texel buffers (P5a).
    [[nodiscard]] static constexpr bool IsTexelBinding(u32 binding) {
        return binding >= Shader::Backend::SPIRV::HEAP_UNIFORM_TEXEL_BINDING;
    }
    /// Whether a heap binding holds storage images.
    [[nodiscard]] static constexpr bool IsStorageImageBinding(u32 binding) {
        return binding >= Shader::Backend::SPIRV::HEAP_FIRST_STORAGE_BINDING &&
               !IsTexelBinding(binding);
    }

    /// The descriptor type of a heap binding.
    [[nodiscard]] static constexpr VkDescriptorType DescriptorType(u32 binding) {
        if (binding == Shader::Backend::SPIRV::HEAP_SAMPLER_BINDING) {
            return VK_DESCRIPTOR_TYPE_SAMPLER;
        }
        if (binding == Shader::Backend::SPIRV::HEAP_UNIFORM_TEXEL_BINDING) {
            return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        }
        if (IsTexelBinding(binding)) {
            return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        }
        return IsStorageImageBinding(binding) ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                              : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    }

    /// Whether storage images can live in the heap on this device.
    [[nodiscard]] bool StorageImages() const noexcept {
        return storage_images;
    }
    /// Whether texel buffers can live in the heap (EDEN_NX2_V2_HEAP_TEXEL and device support).
    [[nodiscard]] bool TexelBuffers() const noexcept {
        return texel_buffers;
    }

    /// The heap, or null when the device lacks update-after-bind sampled images, partially bound
    /// and update-unused-while-pending bindings, or descriptor limits for the arrays.
    [[nodiscard]] static std::unique_ptr<BindlessHeap> Create(const Device& device,
                                                              Scheduler& scheduler);
    /// The heap alive now (pipelines are created without a reference to it), or null.
    [[nodiscard]] static BindlessHeap* Active() noexcept;

    ~BindlessHeap();
    BindlessHeap(const BindlessHeap&) = delete;
    BindlessHeap& operator=(const BindlessHeap&) = delete;

    [[nodiscard]] VkDescriptorSetLayout Layout() const noexcept {
        return *layout;
    }
    /// A set layout without bindings, for set 0 of heap pipelines with no other descriptor.
    [[nodiscard]] VkDescriptorSetLayout EmptyLayout() const noexcept {
        return *empty_layout;
    }
    [[nodiscard]] VkDescriptorSet Set() const noexcept {
        return set;
    }

    /// The element of `view` in image binding `binding`, written when new. Marks it used by the
    /// current tick.
    [[nodiscard]] u32 ImageSlot(u32 binding, VkImageView view);
    /// The element of `sampler` in the sampler binding, written when new.
    [[nodiscard]] u32 SamplerSlot(VkSampler sampler);
    /// P5a: the element of texel buffer `view` in texel binding `binding`, written when new.
    [[nodiscard]] u32 TexelSlot(u32 binding, VkBufferView view);

    /// What element `slot` of `binding` holds (verify).
    [[nodiscard]] VkImageView ImageAt(u32 binding, u32 slot) const;
    [[nodiscard]] VkSampler SamplerAt(u32 slot) const;
    [[nodiscard]] VkBufferView TexelAt(u32 binding, u32 slot) const;

    /// The view or sampler is destroyed: its slots are free (the GPU is done with it).
    void ForgetImageView(VkImageView view);
    void ForgetSampler(VkSampler sampler);
    /// P5a: the buffer cache destroyed a texel buffer view (its buffer died, the GPU is done).
    void ForgetBufferView(VkBufferView view);

    /// Slot writes since the last call.
    [[nodiscard]] u64 TakeWrites() noexcept;

    /// F2-04: moves whenever a slot may stop holding what it held (recycled or freed): while it
    /// stays, heap words taken before still name the same descriptors.
    [[nodiscard]] u64 Releases() const noexcept {
        return releases.load(std::memory_order_acquire);
    }
    /// F2-04: marks `slot` of `binding` used by the current tick (what Slot does on a lookup).
    void Touch(u32 binding, u32 slot);

    /// P5a: writes the descriptors of slots reserved since the last call (lookups reserve slot
    /// indices lock-free and only queue the write). Called before the draw or dispatch that uses
    /// them is recorded, on the caches' owner.
    void Commit();

private:
    BindlessHeap(const Device& device, Scheduler& scheduler, bool storage_images,
                 bool texel_buffers);

    /// P5a: a binding's slots. Lookups read the published handle -> slot map and reserve new
    /// slots lock-free (free ring, then a bump index); only the commit (owner) writes the map and
    /// the descriptors. A handle reserved twice before a commit keeps both slots (extra).
    struct Space {
        std::unique_ptr<VideoCommon::V2::PublishedMap<u64>> slots;
        std::unique_ptr<std::atomic<u64>[]> handles;  ///< slot -> handle, 0 when free
        std::unique_ptr<std::atomic<u64>[]> last_use; ///< slot -> tick of its last draw
        VideoCommon::V2::Commit::BoundedMpmc<u32, Capacity> free;
        std::atomic<u32> next{};
        std::vector<std::pair<u64, u32>> extra; ///< duplicate reservations (owner)
    };

    /// The slot of `handle` in `binding`: published, or reserved now (its write queued).
    u32 Slot(u32 binding, u64 handle);
    /// A full binding: the least recently used slot no pending draw reads (under the mutex).
    u32 Recycle(Space& space);
    void Write(u32 binding, u32 slot, u64 handle);
    void Forget(Space& space, u64 handle);

    const Device& device;
    Scheduler& scheduler;
    vk::DescriptorSetLayout layout;
    vk::DescriptorSetLayout empty_layout;
    vk::DescriptorPool pool;
    vk::DescriptorSets sets;
    VkDescriptorSet set{};
    std::array<Space, NumBindings> spaces;
    std::mutex mutex; ///< owner-side changes (commit, forget, recycle); lookups take none
    VideoCommon::V2::Commit::ExternalQueue pending; ///< reserved slots to write (address = handle)
    std::atomic<u64> generation{1};                 ///< moves at each Commit
    std::atomic<u64> releases{0};                   ///< see Releases
    u64 writes{};
    bool storage_images{};
    bool texel_buffers{};
};

} // namespace Vulkan
