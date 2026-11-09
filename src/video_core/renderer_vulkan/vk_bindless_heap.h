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

    /// What element `slot` of `binding` holds (verify).
    [[nodiscard]] VkImageView ImageAt(u32 binding, u32 slot) const;
    [[nodiscard]] VkSampler SamplerAt(u32 slot) const;

    /// The view or sampler is destroyed: its slots are free (the GPU is done with it).
    void ForgetImageView(VkImageView view);
    void ForgetSampler(VkSampler sampler);

    /// Slot writes since the last call.
    [[nodiscard]] u64 TakeWrites() noexcept;

private:
    BindlessHeap(const Device& device, Scheduler& scheduler);

    struct Space {
        std::unordered_map<u64, u32> slots; ///< handle -> slot
        std::vector<u64> handles;           ///< slot -> handle, 0 when free
        std::vector<u64> last_use;          ///< slot -> tick of its last draw
        std::vector<u32> free;
        u32 next{};
    };

    u32 Acquire(Space& space, u64 handle);
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
    std::mutex mutex;
    u64 writes{};
};

} // namespace Vulkan
