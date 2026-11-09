// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/v2/draw_cache.h"

namespace Tegra {
class MemoryManager;
}

namespace Vulkan {

class ComputePipeline;
class GraphicsPipeline;

/// EDEN_NX2_V2_DRAWCACHE: the Vulkan side of the P4 draw cache (v2/draw_cache.h holds the pure
/// parts: keys, recipes, the table, validation and the ladder). Owned by the rasterizer and used by
/// the thread that resolves draws, with no lock of its own (see the P6 notes in the log).
class DrawCacheGlue {
public:
    using Decision = VideoCommon::V2::DrawCache::Decision;
    using Recipe = VideoCommon::V2::DrawCache::Recipe;
    using RecipeCapture = VideoCommon::V2::DrawCache::RecipeCapture;

    explicit DrawCacheGlue(Tegra::MaxwellDeviceMemoryManager& device_memory);
    ~DrawCacheGlue();

    DrawCacheGlue(const DrawCacheGlue&) = delete;
    DrawCacheGlue& operator=(const DrawCacheGlue&) = delete;

    // --- Draws ---------------------------------------------------------------------------------
    /// The ladder for a draw: continued (P3 took a delta draw), else a validated recipe, else a
    /// resolve. No cache mutation here and no lock needed: versions and generations are atomics,
    /// the rest belongs to the resolving thread.
    [[nodiscard]] Decision DecideDraw(bool continued, bool is_indexed, bool is_indirect,
                                      const Tegra::Engines::Maxwell3D& maxwell3d,
                                      const Tegra::MemoryManager& memory);
    /// Before a full Configure of `pipeline`: the capture to arm (null when the draw can't be
    /// cached). Takes the stamp the recipe's inputs are checked against.
    [[nodiscard]] RecipeCapture* BeginDrawCapture(const GraphicsPipeline& pipeline, bool is_indexed,
                                                  bool is_indirect, bool mesh,
                                                  bool frame_dump,
                                                  const Tegra::Engines::Maxwell3D& maxwell3d);
    /// After the capturing Configure succeeded (caches locked): builds the recipe, or for a verify
    /// sample builds it aside and compares it with the hit's recipe field by field.
    void FinishDrawCapture(GraphicsPipeline& pipeline, const Decision& decision, bool is_indexed,
                           const Tegra::Engines::Maxwell3D& maxwell3d, Tegra::MemoryManager& memory,
                           TextureCache& texture_cache, BufferCache& buffer_cache);
    /// The replay recorded the draw.
    void NoteReplayed(Recipe& recipe);
    /// The replay found a reason to resolve after all (Rebuild).
    void NoteReplayRebuilt(Recipe& recipe);
    /// Prepare time per path (EDEN_NX2_V2_STATS).
    void NoteDrawTime(bool replay, u64 ns);

    // --- Dispatches ----------------------------------------------------------------------------
    [[nodiscard]] Decision DecideDispatch(const Tegra::Engines::KeplerCompute& kepler,
                                          const Tegra::MemoryManager& memory, bool eligible);
    [[nodiscard]] RecipeCapture* BeginDispatchCapture(const ComputePipeline& pipeline,
                                                      bool eligible);
    void FinishDispatchCapture(ComputePipeline& pipeline, u64 native_key, const Decision& decision,
                               const Tegra::Engines::KeplerCompute& kepler,
                               Tegra::MemoryManager& memory, TextureCache& texture_cache);
    void NoteDispatchReplayed(Recipe& recipe);
    void NoteDispatchRebuilt(Recipe& recipe);
    void NoteDispatchTime(bool replay, u64 ns);

    /// Logs the stats every 10 s (EDEN_NX2_V2_STATS).
    void Report();

private:
    struct Pending {
        u64 shape{};
        bool valid{};
        bool capturing{};
    };

    [[nodiscard]] u64 GraphicsShape(bool is_indexed, const Tegra::Engines::Maxwell3D& maxwell3d,
                                    const Tegra::MemoryManager& memory) const;
    [[nodiscard]] static u64 GraphicsKey(u64 shape, const VideoCommon::V2::DrawCache::ShapeInfo& info,
                                         const Tegra::Engines::Maxwell3D& maxwell3d);
    [[nodiscard]] static u64 ComputeShape(const Tegra::Engines::KeplerCompute& kepler,
                                          const Tegra::MemoryManager& memory);
    [[nodiscard]] static u64 ComputeKey(u64 shape, const VideoCommon::V2::DrawCache::ShapeInfo& info,
                                        const Tegra::Engines::KeplerCompute& kepler);
    /// Generations, epochs and inputs of a recipe (Count when it holds).
    [[nodiscard]] VideoCommon::V2::DrawCache::Miss Validate(Recipe& recipe, const void* engine,
                                                            const Tegra::MemoryManager& memory,
                                                            bool compute);
    /// Fills `recipe` from the capture (inputs, views, samplers). False when an input can't be
    /// captured (crosses a page, memory moved under it).
    [[nodiscard]] bool FillFromCapture(Recipe& recipe, bool compute, Tegra::MemoryManager& memory,
                                       TextureCache& texture_cache);
    void Drop(const Recipe& recipe);
    /// Builds into the table, or aside and compares (verify).
    void Store(VideoCommon::V2::DrawCache::RecipeTable& table, u64 key, Recipe& built,
               const Decision& decision, bool compute);

    Tegra::MaxwellDeviceMemoryManager& device_memory;
    VideoCommon::V2::DrawCache::RecipeTable graphics_table;
    VideoCommon::V2::DrawCache::RecipeTable compute_table;
    VideoCommon::V2::DrawCache::ShapeTable graphics_shapes;
    VideoCommon::V2::DrawCache::ShapeTable compute_shapes;
    VideoCommon::V2::Versions::EpochMemo memo;
    VideoCommon::V2::Versions::PageWatch watch;
    VideoCommon::V2::DrawCache::VerifySampler graphics_sampler;
    VideoCommon::V2::DrawCache::VerifySampler compute_sampler;
    Pending pending_draw;
    Pending pending_dispatch;
    RecipeCapture capture;
    Recipe built;
};

} // namespace Vulkan
