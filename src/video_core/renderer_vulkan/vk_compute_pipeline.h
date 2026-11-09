// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <span>

#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/v2/draw_cache.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

class Device;
class PipelineStatistics;
class Scheduler;

class ComputePipeline {
public:
    explicit ComputePipeline(const Device& device, Scheduler& scheduler, vk::PipelineCache& pipeline_cache,
                             DescriptorPool& descriptor_pool,
                             GuestDescriptorQueue& guest_descriptor_queue,
                             DescriptorBufferRing& descriptor_buffer_ring,
                             Common::ThreadWorker* thread_worker,
                             PipelineStatistics* pipeline_statistics,
                             VideoCore::ShaderNotify* shader_notify, const Shader::Info& info,
                             vk::ShaderModule spv_module, u64 shader_hash);

    ComputePipeline& operator=(ComputePipeline&&) noexcept = delete;
    ComputePipeline(ComputePipeline&&) noexcept = delete;

    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(const ComputePipeline&) = delete;

    using BindCommand = std::function<void(vk::CommandBuffer)>;

    [[nodiscard]] bool Configure(Tegra::Engines::KeplerCompute& kepler_compute,
                                 Tegra::MemoryManager& gpu_memory, Scheduler& scheduler,
                                 BufferCache& buffer_cache, TextureCache& texture_cache,
                                 TextureCacheRuntime& texture_runtime,
                                 std::optional<ByteAddressedShadow>& shadow,
                                 BindCommand* deferred_bind = nullptr);

    static constexpr u64 Ocarina1440pShader = 0x6bd913e1f34e83ebULL;
    [[nodiscard]] bool CanReplaceOcarinaUpscale() const;

    bool IsBound() const noexcept {
        return static_cast<bool>(pipeline);
    }

    [[nodiscard]] u64 ShaderHash() const noexcept {
        return shader_hash;
    }

    /// Storage images written by the last configured dispatch
    [[nodiscard]] std::span<const VideoCommon::ImageViewId> WrittenImages() const noexcept {
        return written_images;
    }

    struct SampledImage {
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The textures sampled by the last configured dispatch, kept while a sampled frame dump runs.
    [[nodiscard]] std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    /// Advances the frame window of EDEN_NX2_UPSCALER_TRACE
    static void TickUpscalerTrace();

    /// Whether no build of this pipeline is left queued on a worker thread, which holds it
    [[nodiscard]] bool IsBuildTaskDone() const noexcept {
        return build_task_done.load(std::memory_order::acquire);
    }

    [[nodiscard]] const Shader::Info& ShaderInfo() const noexcept {
        return info;
    }

    /// EDEN_NX2_V2_DRAWCACHE: the next Configure records what a dispatch recipe needs.
    void SetRecipeCapture(VideoCommon::V2::DrawCache::RecipeCapture* capture) noexcept {
        recipe_capture = capture;
    }
    /// EDEN_NX2_V2_DRAWCACHE: the next Configure takes its image views and samplers from
    /// `recipe` (no handle or TIC/TSC read, no view lookup; the views are prepared as a texture
    /// memo hit prepares them). Storage buffers are bound from their descriptors as always: their
    /// GPU-written extent depends on each dispatch's grid. RecipeReplayFailed tells a false
    /// Configure caused by the replay (resolve in full then).
    void SetRecipeReplay(const VideoCommon::V2::DrawCache::Recipe* recipe) noexcept {
        recipe_replay = recipe;
        recipe_replay_failed = false;
    }
    [[nodiscard]] bool RecipeReplayFailed() const noexcept {
        return recipe_replay_failed;
    }
    /// Whether dispatches of this pipeline can be replayed from a recipe (no handles computed at
    /// dispatch time, no byte-addressed images).
    [[nodiscard]] bool TakesRecipes() const noexcept {
        return recipe_eligible;
    }
    /// Constant buffer slots the dispatch reads texture and image handles from.
    [[nodiscard]] u32 HandleConstantBuffers() const noexcept {
        return handle_constant_buffers;
    }

private:
    template <bool Recipes>
    [[nodiscard]] bool ConfigureImpl(Tegra::Engines::KeplerCompute& kepler_compute,
                                     Tegra::MemoryManager& gpu_memory, Scheduler& scheduler,
                                     BufferCache& buffer_cache, TextureCache& texture_cache,
                                     TextureCacheRuntime& texture_runtime,
                                     std::optional<ByteAddressedShadow>& shadow,
                                     BindCommand* deferred_bind);

    /// Builds the pipeline once, on whichever thread gets here first.
    void RunBuild();

    const Device& device;
    vk::PipelineCache& pipeline_cache;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;
    Shader::Info info;
    u64 shader_hash{};
    u32 num_descriptor_entries{};
    boost::container::small_vector<VideoCommon::ImageViewId, 8> written_images;
    std::vector<SampledImage> sampled_images;

    VideoCommon::ComputeUniformBufferSizes uniform_buffer_sizes{};

    vk::ShaderModule spv_module;
    vk::DescriptorSetLayout descriptor_set_layout;
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};
    DescriptorBufferLayout descriptor_buffer_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    // EDEN_NX2_V2_DRAWCACHE (v2/draw_cache.h).
    VideoCommon::V2::DrawCache::RecipeCapture* recipe_capture{};
    const VideoCommon::V2::DrawCache::Recipe* recipe_replay{};
    bool recipe_replay_failed{};
    bool recipe_eligible{};
    u32 handle_constant_buffers{};

    /// EDEN_NX2_V2_BINDLESS: textures or storage images read the heap (set 1, word table).
    bool heap_pipeline{false};
    HeapPushConstant heap_push{};
    HeapTexelMasks heap_texels{}; ///< P5a: texel buffer bindings that read the heap
    u32 heap_reasons{};
    std::vector<u32> last_heap_words;
    VkBuffer last_heap_buffer{};
    VkDeviceSize last_heap_offset{};
    u64 last_heap_tick{~u64{}};

    std::function<void()> build_task;
    std::atomic_bool build_claimed{false};
    std::atomic_bool build_task_done{true};
    std::condition_variable build_condvar;
    std::mutex build_mutex;
    std::atomic_bool is_built{false};
};

} // namespace Vulkan
