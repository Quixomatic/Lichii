// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "common/thread_worker.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/pipeline_build.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_descriptor_buffer.h"
#include "video_core/renderer_vulkan/vk_descriptor_pool.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/renderer_vulkan/vk_update_descriptor.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/desc_cache.h"
#include "video_core/v2/draw_cache.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCore {
class ShaderNotify;
}

namespace Vulkan {

struct GraphicsPipelineCacheKey {
    std::array<u64, 6> unique_hashes;
    FixedPipelineState state;

    size_t Hash() const noexcept;

    bool operator==(const GraphicsPipelineCacheKey& rhs) const noexcept;

    bool operator!=(const GraphicsPipelineCacheKey& rhs) const noexcept {
        return !operator==(rhs);
    }

    size_t Size() const noexcept {
        return sizeof(unique_hashes) + state.Size();
    }
};
static_assert(std::has_unique_object_representations_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_copyable_v<GraphicsPipelineCacheKey>);
static_assert(std::is_trivially_constructible_v<GraphicsPipelineCacheKey>);

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::GraphicsPipelineCacheKey> {
    size_t operator()(const Vulkan::GraphicsPipelineCacheKey& k) const noexcept {
        return k.Hash();
    }
};
} // namespace std

namespace Vulkan {

class Device;
class PipelineStatistics;
class RenderPassCache;
class RescalingPushConstant;
class RenderAreaPushConstant;
class Scheduler;

/// Mesh emulation on a device without VK_EXT_mesh_shader: the mesh program (stage 0) is built as
/// a compute shader that writes its output to a storage buffer, and this vertex shader draws it
/// (see Shader::Backend::SPIRV::EmitMeshEmulationVertex).
struct MeshEmulationSetup {
    std::vector<u32> vertex_spirv;
    u32 output_binding{};
    u32 block_words{};    ///< output words per mesh workgroup
    u32 max_primitives{}; ///< triangles per mesh workgroup
    // With a task program (stage 0, the mesh program at mesh_stage): its payload buffer and the
    // distribution pass's offsets and arguments (Shader::RuntimeInfo), and that pass's shader.
    bool has_task{};
    u32 mesh_stage{};
    u32 payload_stride{};
    u32 payload_binding{};
    u32 offsets_binding{};
    u32 args_binding{};
    std::vector<u32> distribution_spirv;
    // Output layout, for the readback diagnostics.
    u32 max_vertices{};
    u32 vertex_lines{};
    std::array<u32, 4> position_lines{~0U, ~0U, ~0U, ~0U}; ///< line of PositionX..W
};

class GraphicsPipeline {
    static constexpr size_t NUM_STAGES = Tegra::Engines::Maxwell3D::Regs::MaxShaderStage;

public:
    explicit GraphicsPipeline(
        Scheduler& scheduler, BufferCache& buffer_cache, TextureCache& texture_cache,
        vk::PipelineCache& pipeline_cache, VideoCore::ShaderNotify* shader_notify,
        const Device& device, DescriptorPool& descriptor_pool,
        GuestDescriptorQueue& guest_descriptor_queue,
        DescriptorBufferRing& descriptor_buffer_ring, Common::ThreadWorker* worker_thread,
        PipelineStatistics* pipeline_statistics, RenderPassCache& render_pass_cache,
        const GraphicsPipelineCacheKey& key, std::array<vk::ShaderModule, NUM_STAGES> stages,
        const std::array<const Shader::Info*, NUM_STAGES>& infos,
        std::function<std::array<vk::ShaderModule, NUM_STAGES>()> emit_stages = {},
        std::optional<MeshEmulationSetup> mesh_emulation = std::nullopt);

    static std::chrono::nanoseconds TakeHoldTime() noexcept;

    bool HasDynamicVertexInput() const noexcept { return key.state.dynamic_vertex_input; }
    /// The pipeline's cache key (the pipeline cache's v2 lookup tables point into it).
    const GraphicsPipelineCacheKey& Key() const noexcept {
        return key;
    }
    const std::array<u64, 6>& ShaderHashes() const noexcept {
        return key.unique_hashes;
    }

    struct SampledImage {
        u32 stage;
        u32 tic;
        VideoCommon::ImageViewId id;
        VideoCommon::SamplerId sampler;
    };

    /// The sampled textures of the last configured draw, kept while a sampled frame dump runs.
    std::span<const SampledImage> SampledImages() const noexcept {
        return sampled_images;
    }

    bool SupportsAlphaToCoverage() const noexcept {
        return fragment_has_color0_output;
    }

    bool SupportsAlphaToOne() const noexcept {
        return fragment_has_color0_output;
    }

    /// Color components of a render target the fragment shader writes, as VkColorComponentFlags
    VkColorComponentFlags ColorOutputMask(size_t render_target) const noexcept {
        return (color_output_mask >> (render_target * 4)) & 0xF;
    }

    /// Color components the fragment shader writes, 4 bits per render target
    u32 ColorOutputMask() const noexcept {
        return color_output_mask;
    }

    bool UsesExtendedDynamicState() const noexcept {
        return key.state.extended_dynamic_state != 0;
    }

    /// Switch 2 task/mesh pipeline: drawn with vkCmdDrawMeshTasksEXT.
    [[nodiscard]] bool IsMeshPipeline() const noexcept {
        return stage_infos[0].is_task || stage_infos[0].is_mesh;
    }

    /// Mesh pipeline drawn through the compute emulation (MeshEmulationSetup).
    [[nodiscard]] bool IsEmulatedMesh() const noexcept {
        return mesh_emulation.has_value();
    }

    /// Output bytes the emulation needs for one mesh workgroup.
    [[nodiscard]] u64 MeshEmulationBlockBytes() const noexcept {
        return mesh_emulation ? u64{mesh_emulation->block_words} * 4 : 0;
    }

    /// Vertices of the emulation's triangle list for one mesh workgroup.
    [[nodiscard]] u32 MeshEmulationVerticesPerGroup() const noexcept {
        return mesh_emulation ? mesh_emulation->max_primitives * 3 : 0;
    }

    /// The emulation's setup (output layout), or null.
    [[nodiscard]] const MeshEmulationSetup* MeshEmulation() const noexcept {
        return mesh_emulation ? &*mesh_emulation : nullptr;
    }

    /// The emulation's output buffer and workgroup count for the next Configure.
    void SetMeshEmulationOutput(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                u32 groups) noexcept {
        mesh_output_buffer = buffer;
        mesh_output_offset = offset;
        mesh_output_size = size;
        mesh_groups = groups;
    }

    struct MeshEmulationBuffer {
        VkBuffer buffer{};
        VkDeviceSize offset{};
        VkDeviceSize size{};
    };
    /// With a task program: the task payload, offset and argument buffers, the arena counter,
    /// and the words of the output arena (the draw's group count is the task's).
    void SetMeshEmulationTaskBuffers(const MeshEmulationBuffer& payload,
                                     const MeshEmulationBuffer& offsets,
                                     const MeshEmulationBuffer& args,
                                     const MeshEmulationBuffer& counter,
                                     u32 capacity_words) noexcept {
        mesh_task_buffers = {payload, offsets, args, counter};
        mesh_capacity = capacity_words;
    }

    [[nodiscard]] bool MeshEmulationHasTask() const noexcept {
        return mesh_emulation && mesh_emulation->has_task;
    }

    [[nodiscard]] u32 MeshEmulationPayloadStride() const noexcept {
        return mesh_emulation ? mesh_emulation->payload_stride : 0;
    }
    GraphicsPipeline& operator=(GraphicsPipeline&&) noexcept = delete;
    GraphicsPipeline(GraphicsPipeline&&) noexcept = delete;

    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(const GraphicsPipeline&) = delete;

    void AddTransition(GraphicsPipeline* transition);

    bool Configure(bool is_indexed) {
        return configure_func(this, is_indexed);
    }

    [[nodiscard]] GraphicsPipeline* Next(const GraphicsPipelineCacheKey& current_key) noexcept {
        if (key == current_key) {
            return this;
        }
        const auto it{std::find(transition_keys.begin(), transition_keys.end(), current_key)};
        return it != transition_keys.end() ? transitions[std::distance(transition_keys.begin(), it)]
                                           : nullptr;
    }

    /// The frame the pipeline cache last handed this pipeline out (evicts long unused ones).
    void MarkUsed(u64 frame) noexcept {
        last_use_frame = frame;
        used = true;
    }

    [[nodiscard]] u64 LastUse() const noexcept {
        return last_use_frame;
    }

    /// Whether the cache has handed this pipeline out at all (frame 0 included), as opposed to a
    /// pipeline prebuilt from the disk cache that no draw has used yet.
    [[nodiscard]] bool WasUsed() const noexcept {
        return used;
    }

    /// Forgets the transitions, which may point to pipelines the cache evicted.
    void ClearTransitions() noexcept {
        transition_keys.clear();
        transitions.clear();
    }

    [[nodiscard]] bool WritesStorage() const noexcept {
        return writes_storage;
    }

    /// Writes storage images or image buffers (their images are marked modified when resolved).
    [[nodiscard]] bool WritesImages() const noexcept {
        return writes_images;
    }

    [[nodiscard]] bool IsBuilt() const noexcept {
        return build->IsBuilt();
    }

    /// Waits up to `timeout` for the build; true when built.
    [[nodiscard]] bool WaitBuilt(std::chrono::microseconds timeout) const {
        return build->WaitFor(timeout);
    }

    [[nodiscard]] bool Failed() const noexcept {
        return build->Failed();
    }

    /// Fast draw (nx2_fast_draw.h): whether draws can keep the image views and samplers the last
    /// captured Configure resolved. Handles computed at draw time are resolved on every draw.
    [[nodiscard]] bool TakesDeltaDraws() const noexcept {
        return delta_blocker == DeltaBlocker::None && delta_views_valid;
    }

    /// Why TakesDeltaDraws is false (diagnostics).
    enum class DeltaBlocker : u8 {
        None,
        WritesImages,
        Mesh,
        ComputedHandles,
        NotConfigured,
        Count,
    };
    [[nodiscard]] DeltaBlocker DeltaDrawBlocker() const noexcept {
        return delta_blocker != DeltaBlocker::None ? delta_blocker
               : delta_views_valid                 ? DeltaBlocker::None
                                                   : DeltaBlocker::NotConfigured;
    }

    /// Constant buffer slots per stage the pipeline reads texture and image handles from.
    [[nodiscard]] const std::array<u32, NUM_STAGES>& HandleConstantBuffers() const noexcept {
        return handle_constant_buffers;
    }

    /// Constant buffer slots per stage the pipeline reads, and how many bytes of each.
    [[nodiscard]] const std::array<u32, 5>& UniformBufferMasks() const noexcept {
        return enabled_uniform_buffer_masks;
    }
    [[nodiscard]] const VideoCommon::UniformBufferSizes& UniformSizes() const noexcept {
        return uniform_buffer_sizes;
    }

    /// Whether any stage samples a texture or reads an image or texel buffer (the pools matter).
    [[nodiscard]] bool UsesTexturePools() const noexcept {
        return num_image_elements != 0 || num_textures != 0;
    }

    /// Whether the vertex stage reads any generic attribute (the vertex streams matter).
    [[nodiscard]] bool ReadsVertexAttributes() const noexcept {
        for (size_t index = 0; index < 32; ++index) {
            if (stage_infos[0].loads.Generic(index)) {
                return true;
            }
        }
        return false;
    }

    /// Resolves and binds every buffer as Configure does (storage buffer addresses are read from
    /// the constant buffers again) and records the draw's descriptors with the image views,
    /// samplers and texture buffers the last captured Configure resolved. False when the draw
    /// can't be recorded, as for Configure.
    bool ConfigureDelta(bool is_indexed);

    /// The next Configure keeps the views and samplers it resolves for delta draws. With
    /// `verify`, it first compares them with the ones kept before; DeltaMismatch reports it.
    void CaptureNextDelta(bool verify) noexcept {
        capture_delta = true;
        verify_delta = verify;
    }
    [[nodiscard]] bool DeltaMismatch() const noexcept {
        return delta_mismatch;
    }

    /// Whether an image the kept views sample has an upload pending or a newer alias to copy in:
    /// ConfigureDelta prepares none, Configure does.
    [[nodiscard]] bool DeltaImagesChange() const noexcept;

    /// EDEN_NX2_V2_DRAWCACHE: the next Configure records into `capture` what a draw recipe needs
    /// (the handle words, storage buffer descriptors, views, samplers and push data it resolves;
    /// a texture memo hit gives the handle words it compared). Null: none.
    void SetRecipeCapture(VideoCommon::V2::DrawCache::RecipeCapture* capture) noexcept {
        recipe_capture = capture;
    }
    /// Configure recording into the capture SetRecipeCapture set (the plain Configure has no
    /// capture code at all).
    bool ConfigureCapture(bool is_indexed);
    /// Whether draws with this pipeline can be replayed from a recipe: no storage writes, no
    /// handles computed at draw time, no storage buffers through table pointers, no mesh.
    [[nodiscard]] bool TakesRecipes() const noexcept {
        return recipe_eligible;
    }
    /// FixedPipelineState::Hash of the key (the pipeline part of a draw-cache shape).
    [[nodiscard]] u64 StateHash() const noexcept {
        return state_hash;
    }
    enum class RecipeResult : u8 {
        Recorded, ///< the draw's state is recorded
        Rebuild,  ///< the recipe no longer matches (images deleted, other render targets):
                  ///< resolve in full
        Failed,   ///< the draw can't be recorded (as Configure returning false)
    };
    /// Records the draw from a recipe: its image views and samplers (prepared as a texture memo
    /// hit prepares them: uploads and alias copies still happen), its storage buffer bindings and
    /// texture buffers, with no handle, TIC/TSC or storage descriptor read and no view lookup.
    /// Vertex, index and uniform buffers are resolved and synchronized by the buffer cache as
    /// always. `push_out` (verify) receives the push data and heap words recorded.
    /// With `memo`, buffers are replayed too (BufferCache::ReplayGraphicsBuffers): bindings whose
    /// buffer and guest range are unchanged since the recipe's buffer stamp are bound from it, the
    /// rest through the lookup and sync; the recipe's buffers are recorded again when any binding
    /// took the sync path. `buffers` (stats) receives what happened.
    struct BufferReplay {
        bool replayed{}; ///< the buffer replay ran (else the existing calls)
        u32 binds_replayed{};
        u32 binds_synced{};
    };
    RecipeResult ConfigureRecipe(bool is_indexed, VideoCommon::V2::DrawCache::Recipe& recipe,
                                 VideoCommon::V2::DrawCache::RecipeCapture* push_out,
                                 VideoCommon::V2::Versions::EpochMemo* memo,
                                 BufferReplay* buffers);
    /// Constant buffer slots per stage holding the storage buffer descriptors the pipeline reads.
    [[nodiscard]] const std::array<u32, NUM_STAGES>& StorageConstantBuffers() const noexcept {
        return storage_constant_buffers;
    }

    template <typename Spec>
    static auto MakeConfigureSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) { return pl->ConfigureImpl<Spec>(is_indexed); };
    }
    template <typename Spec>
    static auto MakeCapturingSpecFunc() {
        return [](GraphicsPipeline* pl, bool is_indexed) {
            return pl->ConfigureImpl<Spec, true>(is_indexed);
        };
    }

    void SetEngine(Tegra::Engines::Maxwell3D* maxwell3d_, Tegra::MemoryManager* gpu_memory_) {
        maxwell3d = maxwell3d_;
        gpu_memory = gpu_memory_;
    }

    /// The translated program of `stage` (VertexB, TessControl, TessEval, Geometry, Fragment).
    [[nodiscard]] const Shader::Info& StageInfo(size_t stage) const noexcept {
        return stage_infos[stage];
    }

    /// Pipeline v2 P6 Step 0 (EDEN_NX2_V2_DRAWREC, vk_draw_resolve.cpp): the draw's record from
    /// the view alone (pure). Its cache mutations go to `textures` (applied first) and `buffers`
    /// as requests. Stale: the descriptor tables must be synchronized first (once).
    /// CommitOnly(reason): the draw takes the old Configure.
    template <class View>
    [[nodiscard]] VideoCommon::V2::Resolve::ResolveStatus ResolveDraw(
        const View& view, const VideoCommon::V2::Resolve::ResolveContext& ctx,
        VideoCommon::V2::Resolve::ResolvedDraw& out,
        VideoCommon::V2::Commit::RequestList& textures,
        VideoCommon::V2::Commit::RequestList& buffers) const;

    /// The serial mutator of a record (spec §3): the texture then buffer requests (with the
    /// deleted-images/buffers loops), the record's proxies patched, the image descriptors derived
    /// (heap slots reserved), the heap committed, the render targets updated, the feedback-loop
    /// check. Template: a recipe hit (heap words replayed while valid). DeriveOnly (verify):
    /// nothing applied, proxies looked up again, slots peeked. False: the draw can't be recorded.
    enum class ApplyMode : u8 { Normal, Template, DeriveOnly };
    [[nodiscard]] bool ApplyDraw(VideoCommon::V2::Resolve::ResolvedDraw& record,
                                 VideoCommon::V2::Commit::RequestList& textures,
                                 VideoCommon::V2::Commit::RequestList& buffers, DrawApplyOut& out,
                                 ApplyMode mode, VideoCommon::V2::Commit::Stage& stage,
                                 VideoCommon::V2::DrawCache::Recipe* recipe = nullptr);

    /// DRAWREC=verify (step 11): the next configure (full, delta or recipe) builds the record it
    /// leaves behind and compares it with `resolved` before recording the draw. Null in v1.
    struct DrawRecObserver {
        const VideoCommon::V2::Resolve::ResolvedDraw* resolved{};
        const VideoCommon::V2::Resolve::DynamicFeatures* features{};
        VideoCommon::V2::Resolve::ResolvedDraw observed;
        size_t views_before{};
        size_t samplers_before{};
        size_t buffers_before{};
        bool done{};
        // Step 12: what the configure derived (heap and rescaling words).
        std::vector<u32> heap_words;
        std::array<u32, Shader::Backend::SPIRV::NUM_TEXTURE_AND_IMAGE_SCALING_WORDS> rescaling{};
        DrawApplyOut derived;
        bool derived_ok{};
        // Step 13: the configure's descriptor payload and render area; whether the record's
        // buffer groups matched (a second encode only rewrites equal bindings then).
        std::vector<DescriptorUpdateEntry> payload;
        std::array<f32, 4> render_area{};
        bool uses_render_area{};
        bool buffers_match{};
    };
    /// The encode of a record (spec §3): channel state bindings, host geometry binds, the
    /// descriptor payload (stage buffers and image descriptors), heap words, the deferred bind
    /// commit, the fragment barrier decision, ConfigureDraw. PayloadOnly (verify) stops before
    /// ConfigureDraw and compares the payload with the observed configure's.
    enum class EncodeMode : u8 { Normal, Replay, PayloadOnly };
    [[nodiscard]] bool EncodeDraw(const VideoCommon::V2::Resolve::ResolvedDraw& record,
                                  const DrawApplyOut& out,
                                  const VideoCommon::V2::Resolve::ResolvedDraw* previous,
                                  EncodeMode mode,
                                  VideoCommon::V2::DrawCache::Recipe* recipe = nullptr);
    /// DRAWREC=verify (step 13): EncodeDraw(PayloadOnly) after the observed configure.
    void VerifyDrawEncode(const VideoCommon::V2::Resolve::ResolvedDraw& record);
    /// DRAWREC=1 (step 14): what ConfigureImpl keeps for delta draws after a resolve, from a
    /// record's applied views and samplers (CaptureNextDelta), and the record kept as the
    /// pipeline's last one.
    void NoteRecordDelta(const DrawApplyOut& out);
    void StoreLastRecord(const VideoCommon::V2::Resolve::ResolvedDraw& record) {
        last_record.h = record.h;
        last_record.arena.assign(record.arena.begin(), record.arena.end());
    }
    [[nodiscard]] const VideoCommon::V2::Resolve::ResolvedDraw& LastRecord() const noexcept {
        return last_record;
    }
    /// DRAWREC=verify (step 12): ApplyDraw(DeriveOnly) on the observed draw's record, its heap and
    /// rescaling words compared with the ones the configure used.
    void VerifyDrawApply(VideoCommon::V2::Resolve::ResolvedDraw& record,
                         VideoCommon::V2::Commit::Stage& stage);
    void SetDrawRecObserver(DrawRecObserver* observer) noexcept {
        drawrec_observer = observer;
    }

private:
    /// The observer's compare (vk_draw_resolve.cpp): `views`/`samplers` are what the configure
    /// bound, in ConfigureImpl's order.
    void ObserveDrawRecord(std::span<const VideoCommon::ImageViewInOut> views,
                           std::span<const VideoCommon::SamplerId> samplers,
                           const RescalingPushConstant& rescaling,
                           const RenderAreaPushConstant& render_area);
    DrawRecObserver* drawrec_observer{};
    /// DRAWREC: the record of the pipeline's last draw (the base of a continued draw, step 17).
    VideoCommon::V2::Resolve::ResolvedDraw last_record;
    /// ApplyDraw's pieces: record proxies patched from a list (or looked up again), bindings
    /// peeked again after a deletion, texture buffers from the views again.
    void PatchRecordViews(VideoCommon::V2::Resolve::ResolvedDraw& record,
                          const VideoCommon::V2::Commit::RequestList& textures) const;
    void PatchRecordBuffers(VideoCommon::V2::Resolve::ResolvedDraw& record,
                            const VideoCommon::V2::Commit::RequestList& buffers) const;
    void RepeekRecordBuffers(VideoCommon::V2::Resolve::ResolvedDraw& record,
                             VideoCommon::V2::Commit::RequestList& buffers) const;
    void RelookupRecordProxies(VideoCommon::V2::Resolve::ResolvedDraw& record) const;
    [[nodiscard]] bool RefreshRecordTextureBuffers(
        VideoCommon::V2::Resolve::ResolvedDraw& record) const;

    // Pipeline v2 P6 Step 0 (EDEN_NX2_V2_DRAWREC, vk_draw_resolve.cpp): the pure resolve of a
    // draw record. Reads only the view, guest memory and const cache lookups.
    /// The draw's texture handles, views and samplers into `out` (views, samplers, reads) and
    /// their requests into `textures`: sampler requests first, then view requests (each with
    /// its Rescale for blacklisted images), as CommitSamplerLookups and FillImageViewsCommit
    /// apply them.
    template <class View>
    void ResolveTextures(const View& view, VideoCommon::V2::Resolve::ResolvedDraw& out,
                         VideoCommon::V2::Commit::RequestList& textures) const;
    /// The view lookups of `views` (TIC indices) and their requests (PredictImageViews).
    void PushViewRequests(std::span<VideoCommon::V2::Resolve::ResolvedView> views,
                          bool blacklist, VideoCommon::V2::Commit::RequestList& textures) const;
    /// The draw's index, vertex, uniform, storage and texture buffer bindings into `out`, a
    /// missing buffer as a BufferCreate request with a proxy into `buffers` (in the update
    /// passes' order). CommitOnly for what the record path doesn't take yet (transform feedback,
    /// indirect, inline indices, multi-range storage, a texture buffer view not created yet).
    /// After ResolveTextures (texture buffers come from its views).
    template <class View>
    [[nodiscard]] VideoCommon::V2::Resolve::ResolveStatus ResolveBuffers(
        const View& view, const VideoCommon::V2::Resolve::ResolveContext& ctx,
        VideoCommon::V2::Resolve::ResolvedDraw& out,
        VideoCommon::V2::Commit::RequestList& buffers) const;

    template <typename Spec, bool Capture = false>
    bool ConfigureImpl(bool is_indexed);

    bool ConfigureDraw(const RescalingPushConstant& rescaling,
                       const RenderAreaPushConstant& render_are);

    void MakePipeline(VkRenderPass render_pass);

    void Validate();

    const GraphicsPipelineCacheKey key;
    const u64 state_hash;
    Tegra::Engines::Maxwell3D* maxwell3d;
    Tegra::MemoryManager* gpu_memory;
    const Device& device;
    TextureCache& texture_cache;
    BufferCache& buffer_cache;
    vk::PipelineCache& pipeline_cache;
    Scheduler& scheduler;
    GuestDescriptorQueue& guest_descriptor_queue;
    DescriptorBufferRing& descriptor_buffer_ring;

    bool (*configure_func)(GraphicsPipeline*, bool){};

    std::vector<GraphicsPipelineCacheKey> transition_keys;
    std::vector<GraphicsPipeline*> transitions;
    u64 last_use_frame{};
    bool used{};

    std::array<vk::ShaderModule, NUM_STAGES> spv_modules;

    std::array<Shader::Info, NUM_STAGES> stage_infos;
    std::array<u32, 5> enabled_uniform_buffer_masks{};
    VideoCommon::UniformBufferSizes uniform_buffer_sizes{};
    u32 num_descriptor_entries{};
    size_t num_image_elements{};
    std::vector<SampledImage> sampled_images;
    u32 num_textures{};
    bool fragment_has_color0_output{};
    u32 color_output_mask{};
    bool writes_storage{};
    /// Every texture is sampled with a handle read from a constant buffer: ConfigureImpl can
    /// resolve them from texture_memo.
    bool texture_memo_eligible{};
    VideoCommon::SampledTextureMemo texture_memo;
    bool writes_images{};

    DeltaBlocker delta_blocker{};
    bool delta_views_valid{};
    bool capture_delta{};
    bool verify_delta{};
    bool delta_mismatch{};
    std::array<bool, NUM_STAGES> configured_stages{};
    std::array<u32, NUM_STAGES> handle_constant_buffers{};
    std::vector<VideoCommon::ImageViewInOut> delta_views;
    std::vector<VideoCommon::SamplerId> delta_samplers;
    /// F2-14: the recipe build (slot, stamp) the delta views came from (null: a resolve).
    const void* delta_recipe{};
    u64 delta_recipe_stamp{};

    // EDEN_NX2_V2_DRAWCACHE (v2/draw_cache.h).
    VideoCommon::V2::DrawCache::RecipeCapture* recipe_capture{};
    bool recipe_eligible{};
    std::array<bool, NUM_STAGES> recipe_stages{};
    std::array<u32, NUM_STAGES> storage_constant_buffers{};
    /// Records a storage buffer descriptor read for the recipe capture (before the bind reads it).
    void CaptureStorageDescriptor(size_t stage, u32 cbuf_index, u32 cbuf_offset,
                                  u32 indirect_offset);

    vk::DescriptorSetLayout descriptor_set_layout;
    DescriptorAllocator descriptor_allocator;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorUpdateTemplate descriptor_update_template;
    vk::Pipeline pipeline;

    // EDEN_NX2_V2_DESCCACHE (v2/desc_cache.h): descriptor buffer ranges by payload and ring
    // generation; the payload last pushed and the recording epoch it was pushed in; descriptor
    // sets by payload and tick, touched only by the VulkanWorker closures.
    struct DescriptorBufferRange {
        VkDeviceSize offset{};
        u32 chunk{};
    };
    VideoCommon::V2::DescriptorContentCache<DescriptorBufferRange> descriptor_buffer_cache;
    std::vector<u8> last_pushed_payload;
    u64 last_pushed_epoch{~u64{}};
    u64 last_pushed_hash{};
    VideoCommon::V2::DescriptorContentCache<VkDescriptorSet> worker_set_cache;

    DescriptorBufferLayout descriptor_buffer_layout;
    std::vector<DescriptorUpdateEntry> last_descriptor_payload;
    VkDeviceSize last_descriptor_buffer_offset{};
    u32 last_descriptor_buffer_chunk{};
    u64 last_descriptor_buffer_generation{};

    const bool worker_only_build;
    bool uses_push_descriptor{false};
    bool uses_descriptor_buffer{false};
    /// EDEN_NX2_V2_BINDLESS: some textures or storage images read the heap (set 1, word table).
    bool heap_pipeline{false};
    HeapPushConstant heap_push{};
    /// P5a (EDEN_NX2_V2_HEAP_TEXEL): per stage, the texel buffer bindings that read the heap.
    std::array<HeapTexelMasks, NUM_STAGES> heap_texels{};
    /// Runs a stage's buffer binds; heap texel buffers push heap words instead of descriptors.
    template <typename Func>
    void WithHeapTexels(size_t stage, Func&& func) {
        const HeapTexelMasks& masks = heap_texels[stage];
        if (masks.heap == 0) [[likely]] {
            func();
            return;
        }
        buffer_cache.runtime.BeginHeapTexels(heap_push.words, masks.heap, masks.storage,
                                             masks.integer, heap_push.verify);
        func();
        buffer_cache.runtime.EndHeapTexels();
    }
    u32 heap_reasons{}; ///< HeapReason bits of descriptors left out
    u32 heap_words{};
    /// F2-04: every sampled texture and storage image of the pipeline reads the heap and no
    /// texel buffer does (its heap words are its image words only). -1: not computed yet.
    s8 heap_images_only{-1};
    [[nodiscard]] bool HeapImagesOnly();
    /// Whether a recipe hit can replay the recipe's heap and rescaling words / learn them now.
    [[nodiscard]] bool HeapWordsReplayable(const VideoCommon::V2::DrawCache::Recipe& recipe);
    [[nodiscard]] bool HeapWordsLearnable(const VideoCommon::V2::DrawCache::Recipe& recipe);
    /// Marks the heap slots of `words` used by the current submission.
    void TouchHeapWords(std::span<const u32> words);
    /// The heap words are push constants (they fit HEAP_PUSH_WORDS); else the word table.
    bool heap_words_pushed{false};
    HeapWordTable heap_table;
    HeapPushData heap_push_data; ///< what the configured draw pushes at HEAP_PUSH_OFFSET
    void AddHeapWords();

    // Mesh emulation (MeshEmulationSetup).
    std::optional<MeshEmulationSetup> mesh_emulation;
    vk::ShaderModule mesh_vertex_module;
    vk::Pipeline mesh_compute_pipeline;
    VkDescriptorSet mesh_descriptor_set{}; ///< shared by the dispatch and the draw it feeds
    VkBuffer mesh_output_buffer{};
    VkDeviceSize mesh_output_offset{};
    VkDeviceSize mesh_output_size{};
    u32 mesh_groups{};
    // With a task program: its compute pipeline, the distribution pass between it and the mesh
    // dispatch, and the per-draw buffers (payload, offsets, arguments, arena counter).
    vk::Pipeline task_compute_pipeline;
    vk::ShaderModule distribution_module;
    vk::DescriptorSetLayout distribution_set_layout;
    vk::PipelineLayout distribution_layout;
    vk::Pipeline distribution_pipeline;
    DescriptorAllocator distribution_allocator;
    std::array<MeshEmulationBuffer, 4> mesh_task_buffers{};
    u32 mesh_capacity{};

    std::optional<PipelineBuild> build;
};

} // namespace Vulkan
