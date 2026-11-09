// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <type_traits>
#include <utility>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_bindless_heap.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/v2/p3_stats.h"
#include "video_core/v2/v2_switches.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {
std::atomic<BindlessHeap*> g_active{};

namespace SPIRV = Shader::Backend::SPIRV;

template <typename T>
u64 ToHandle(T handle) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<u64>(handle);
    } else {
        return static_cast<u64>(handle);
    }
}

template <typename T>
T FromHandle(u64 handle) {
    if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<T>(handle);
    } else {
        return static_cast<T>(handle);
    }
}
} // Anonymous namespace

std::unique_ptr<BindlessHeap> BindlessHeap::Create(const Device& device, Scheduler& scheduler) {
    if (!device.IsSampledImageUpdateAfterBindSupported() ||
        !device.IsDescriptorBindingPartiallyBoundSupported()) {
        LOG_WARNING(Render_Vulkan, "EDEN_NX2_V2_BINDLESS: the device lacks update-after-bind "
                                   "sampled images, partially bound or update-unused-while-pending "
                                   "bindings; heap off");
        return nullptr;
    }
    VkPhysicalDeviceDescriptorIndexingProperties indexing{};
    indexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &indexing;
    device.GetPhysical().GetProperties2(properties);
    const u64 images = u64{Capacity} * SampledBindings;
    const u64 storage = u64{Capacity} * StorageBindings;
    if (indexing.maxPerStageDescriptorUpdateAfterBindSampledImages < images ||
        indexing.maxDescriptorSetUpdateAfterBindSampledImages < images ||
        indexing.maxPerStageDescriptorUpdateAfterBindSamplers < Capacity ||
        indexing.maxDescriptorSetUpdateAfterBindSamplers < Capacity ||
        indexing.maxUpdateAfterBindDescriptorsInAllPools < images + Capacity) {
        LOG_WARNING(Render_Vulkan,
                    "EDEN_NX2_V2_BINDLESS: descriptor limits too small for the heap ({} images "
                    "per stage, {} samplers); heap off",
                    indexing.maxPerStageDescriptorUpdateAfterBindSampledImages,
                    indexing.maxPerStageDescriptorUpdateAfterBindSamplers);
        return nullptr;
    }
    // Storage images need their own update-after-bind feature and limits; without them the
    // storage arrays are empty bindings and storage images keep descriptors.
    const bool storage_images =
        device.IsStorageImageUpdateAfterBindSupported() &&
        indexing.maxPerStageDescriptorUpdateAfterBindStorageImages >= storage &&
        indexing.maxDescriptorSetUpdateAfterBindStorageImages >= storage &&
        indexing.maxUpdateAfterBindDescriptorsInAllPools >= images + storage + Capacity;
    // P5a (EDEN_NX2_V2_HEAP_TEXEL): uniform texel buffers count as sampled images and storage
    // texel buffers as storage images in the per-stage and per-set limits.
    const u64 texel_sampled = images + Capacity;
    const u64 texel_storage = storage + u64{Capacity} * 2;
    const bool texel_buffers =
        VideoCommon::V2::On(VideoCommon::V2::Switch::HeapTexel) && storage_images &&
        device.IsTexelBufferHeapSupported() &&
        indexing.maxPerStageDescriptorUpdateAfterBindSampledImages >= texel_sampled &&
        indexing.maxDescriptorSetUpdateAfterBindSampledImages >= texel_sampled &&
        indexing.maxPerStageDescriptorUpdateAfterBindStorageImages >= texel_storage &&
        indexing.maxDescriptorSetUpdateAfterBindStorageImages >= texel_storage &&
        indexing.maxUpdateAfterBindDescriptorsInAllPools >=
            texel_sampled + texel_storage + Capacity;
    if (VideoCommon::V2::On(VideoCommon::V2::Switch::HeapTexel) && !texel_buffers) {
        LOG_WARNING(Render_Vulkan, "EDEN_NX2_V2_HEAP_TEXEL: the device lacks update-after-bind or "
                                   "dynamically indexed texel buffer arrays (or their limits); "
                                   "texel buffers keep descriptors");
    }
    std::unique_ptr<BindlessHeap> heap{
        new BindlessHeap(device, scheduler, storage_images, texel_buffers)};
    if (heap->set == VK_NULL_HANDLE) {
        return nullptr;
    }
    g_active.store(heap.get(), std::memory_order_release);
    LOG_INFO(Render_Vulkan,
             "EDEN_NX2_V2_BINDLESS: heap of {} elements x {} sampled image arrays, {} storage "
             "image arrays{}, samplers",
             Capacity, SampledBindings, StorageBindings,
             storage_images ? (texel_buffers ? ", texel buffer arrays" : "")
                            : " (storage images unsupported: kept as descriptors)");
    return heap;
}

BindlessHeap* BindlessHeap::Active() noexcept {
    return g_active.load(std::memory_order_acquire);
}

BindlessHeap::BindlessHeap(const Device& device_, Scheduler& scheduler_, bool storage_images_,
                           bool texel_buffers_)
    : device{device_}, scheduler{scheduler_}, storage_images{storage_images_},
      texel_buffers{texel_buffers_} {
    VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT;
    if (device.IsExtMeshShaderSupported()) {
        stages |= VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    }
    std::array<VkDescriptorSetLayoutBinding, NumBindings> bindings{};
    std::array<VkDescriptorBindingFlags, NumBindings> flags{};
    for (u32 binding = 0; binding < NumBindings; ++binding) {
        const bool unused = (IsStorageImageBinding(binding) && !storage_images) ||
                            (IsTexelBinding(binding) && !texel_buffers);
        bindings[binding] = VkDescriptorSetLayoutBinding{
            .binding = binding,
            .descriptorType = DescriptorType(binding),
            .descriptorCount = unused ? 0 : Capacity,
            .stageFlags = stages,
            .pImmutableSamplers = nullptr,
        };
        flags[binding] = unused
                             ? VkDescriptorBindingFlags{}
                             : VkDescriptorBindingFlags{
                                   VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                   VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                   VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT};
    }
    const VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = NumBindings,
        .pBindingFlags = flags.data(),
    };
    const vk::Device& dev = device.GetLogical();
    layout = dev.CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &flags_ci,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = NumBindings,
        .pBindings = bindings.data(),
    });
    empty_layout = dev.CreateDescriptorSetLayout({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 0,
        .pBindings = nullptr,
    });
    const std::array sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, Capacity},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, Capacity * SampledBindings},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, Capacity * StorageBindings},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, Capacity},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, Capacity * 2},
    };
    pool = dev.CreateDescriptorPool({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<u32>(texel_buffers ? sizes.size() : storage_images ? 3 : 2),
        .pPoolSizes = sizes.data(),
    });
    const VkDescriptorSetLayout set_layout = *layout;
    sets = pool.Allocate({
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    });
    if (sets.IsOutOfPoolMemory() || sets.size() != 1) {
        LOG_ERROR(Render_Vulkan, "EDEN_NX2_V2_BINDLESS: heap set allocation failed");
        return;
    }
    set = sets[0];
    for (u32 binding = 0; binding < NumBindings; ++binding) {
        Space& space = spaces[binding];
        space.handles.assign(Capacity, 0);
        space.last_use.assign(Capacity, 0);
        space.slots.reserve(1024);
        if (binding != SPIRV::HEAP_SAMPLER_BINDING) {
            // A null view (devices with null descriptors) is element 0 of every image array.
            space.next = 1;
            const bool written = (!IsStorageImageBinding(binding) || storage_images) &&
                                 (!IsTexelBinding(binding) || texel_buffers);
            if (written && device.HasNullDescriptor()) {
                Write(binding, 0, 0);
            }
        }
    }
}

BindlessHeap::~BindlessHeap() {
    BindlessHeap* expected = this;
    g_active.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

u32 BindlessHeap::Acquire(Space& space, u64 handle) {
    // Image slot 0 is the null view's (VK_NULL_HANDLE never reaches here).
    const u64 tick = scheduler.CurrentTick();
    if (const auto it = space.slots.find(handle); it != space.slots.end()) {
        space.last_use[it->second] = tick;
        return it->second;
    }
    u32 slot = Capacity;
    if (!space.free.empty()) {
        slot = space.free.back();
        space.free.pop_back();
    } else if (space.next < Capacity) {
        slot = space.next++;
    } else {
        // Full: the least recently used slot no pending draw reads any more.
        u64 oldest = ~u64{};
        for (u32 candidate = 0; candidate < Capacity; ++candidate) {
            const u64 used = space.last_use[candidate];
            if (used < oldest && used != tick && scheduler.IsFree(used)) {
                oldest = used;
                slot = candidate;
            }
        }
        if (slot == Capacity) {
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_ERROR(Render_Vulkan,
                          "EDEN_NX2_V2_BINDLESS: every heap slot is used by pending draws");
            }
            slot = 0;
        }
        space.slots.erase(space.handles[slot]);
    }
    space.slots[handle] = slot;
    space.handles[slot] = handle;
    space.last_use[slot] = tick;
    return slot;
}

void BindlessHeap::Write(u32 binding, u32 slot, u64 handle) {
    VkDescriptorImageInfo image_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = VK_NULL_HANDLE,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const bool is_sampler = binding == SPIRV::HEAP_SAMPLER_BINDING;
    const bool is_texel = IsTexelBinding(binding);
    const VkBufferView texel_view = is_texel ? FromHandle<VkBufferView>(handle) : VK_NULL_HANDLE;
    if (is_sampler) {
        image_info.sampler = FromHandle<VkSampler>(handle);
    } else if (!is_texel) {
        image_info.imageView = FromHandle<VkImageView>(handle);
    }
    const VkDescriptorType type = DescriptorType(binding);
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = set,
        .dstBinding = binding,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType = type,
        .pImageInfo = is_texel ? nullptr : &image_info,
        .pBufferInfo = nullptr,
        .pTexelBufferView = is_texel ? &texel_view : nullptr,
    };
    device.GetLogical().UpdateDescriptorSets(write, {});
    ++writes;
    if (VideoCommon::V2::P3Stats::Enabled()) {
        ++VideoCommon::V2::P3Stats::Get().heap_slot_writes;
    }
}

u32 BindlessHeap::ImageSlot(u32 binding, VkImageView view) {
    const u64 handle = ToHandle(view);
    if (handle == 0) {
        return 0;
    }
    std::scoped_lock lock{mutex};
    Space& space = spaces[binding];
    const auto it = space.slots.find(handle);
    if (it != space.slots.end()) {
        space.last_use[it->second] = scheduler.CurrentTick();
        return it->second;
    }
    const u32 slot = Acquire(space, handle);
    Write(binding, slot, handle);
    return slot;
}

u32 BindlessHeap::SamplerSlot(VkSampler sampler) {
    const u64 handle = ToHandle(sampler);
    std::scoped_lock lock{mutex};
    Space& space = spaces[SPIRV::HEAP_SAMPLER_BINDING];
    const auto it = space.slots.find(handle);
    if (it != space.slots.end()) {
        space.last_use[it->second] = scheduler.CurrentTick();
        return it->second;
    }
    const u32 slot = Acquire(space, handle);
    Write(SPIRV::HEAP_SAMPLER_BINDING, slot, handle);
    return slot;
}

u32 BindlessHeap::TexelSlot(u32 binding, VkBufferView view) {
    const u64 handle = ToHandle(view);
    if (handle == 0) {
        return 0;
    }
    std::scoped_lock lock{mutex};
    Space& space = spaces[binding];
    const auto it = space.slots.find(handle);
    if (it != space.slots.end()) {
        space.last_use[it->second] = scheduler.CurrentTick();
        return it->second;
    }
    const u32 slot = Acquire(space, handle);
    Write(binding, slot, handle);
    return slot;
}

VkBufferView BindlessHeap::TexelAt(u32 binding, u32 slot) const {
    return FromHandle<VkBufferView>(spaces[binding].handles[slot]);
}

void BindlessHeap::ForgetBufferView(VkBufferView view) {
    if (view == VK_NULL_HANDLE) {
        return;
    }
    const u64 handle = ToHandle(view);
    std::scoped_lock lock{mutex};
    for (u32 binding = SPIRV::HEAP_UNIFORM_TEXEL_BINDING; binding < NumBindings; ++binding) {
        Forget(spaces[binding], handle);
    }
}

VkImageView BindlessHeap::ImageAt(u32 binding, u32 slot) const {
    return FromHandle<VkImageView>(spaces[binding].handles[slot]);
}

VkSampler BindlessHeap::SamplerAt(u32 slot) const {
    return FromHandle<VkSampler>(spaces[SPIRV::HEAP_SAMPLER_BINDING].handles[slot]);
}

void BindlessHeap::Forget(Space& space, u64 handle) {
    const auto it = space.slots.find(handle);
    if (it == space.slots.end()) {
        return;
    }
    const u32 slot = it->second;
    space.slots.erase(it);
    space.handles[slot] = 0;
    space.free.push_back(slot);
}

void BindlessHeap::ForgetImageView(VkImageView view) {
    if (view == VK_NULL_HANDLE) {
        return;
    }
    const u64 handle = ToHandle(view);
    std::scoped_lock lock{mutex};
    for (u32 binding = 0; binding < NumBindings; ++binding) {
        if (binding != SPIRV::HEAP_SAMPLER_BINDING && !IsTexelBinding(binding)) {
            Forget(spaces[binding], handle);
        }
    }
}

void BindlessHeap::ForgetSampler(VkSampler sampler) {
    if (sampler == VK_NULL_HANDLE) {
        return;
    }
    std::scoped_lock lock{mutex};
    Forget(spaces[SPIRV::HEAP_SAMPLER_BINDING], ToHandle(sampler));
}

u64 BindlessHeap::TakeWrites() noexcept {
    std::scoped_lock lock{mutex};
    return std::exchange(writes, 0);
}

} // namespace Vulkan
