// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "common/scope_exit.h"

#include "core/memory.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_draw_cache.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/v2_stats.h"

namespace Vulkan {

namespace V2 = VideoCommon::V2;
namespace DC = VideoCommon::V2::DrawCache;
namespace Versions = VideoCommon::V2::Versions;
using V2::GroupBit;
using V2::RegGroup;

namespace {

/// The register groups the shape holds whole: pipeline-key state, the programs and the render
/// targets. Of the viewport group only the registers the pipeline key reads (swizzles and the
/// like, KeyViewportRegisters) are in it, so scissor and viewport changes (dynamic state) keep
/// the recipe; of the vertex streams only their layout (GraphicsKey).
constexpr u32 ShapeGroups = GroupBit(RegGroup::Pipeline) | GroupBit(RegGroup::Shaders) |
                            GroupBit(RegGroup::RenderTargets);

/// The viewport-group registers some pipeline key block reads (P3's KeyBlockRegisters through
/// the register semantics table), built once.
const std::vector<u32>& KeyViewportRegisters() {
    static const std::vector<u32> registers = [] {
        using Regs = Tegra::Engines::Maxwell3D::Regs;
        // Of the viewport transforms the key reads only the swizzle word (RefreshSwizzles); the
        // scale and translation are dynamic state.
        constexpr u32 transforms =
            static_cast<u32>(offsetof(Regs, viewport_transform) / sizeof(u32));
        constexpr u32 transform_words = sizeof(Regs::ViewportTransform) / sizeof(u32);
        constexpr u32 transforms_end =
            transforms + static_cast<u32>(sizeof(Regs::viewport_transform) / sizeof(u32));
        constexpr u32 swizzle_word =
            static_cast<u32>(offsetof(Regs::ViewportTransform, swizzle) / sizeof(u32));
        std::vector<u32> result;
        const auto& groups = V2::RegSem::GetTable3D().groups;
        for (size_t block = 0; block < NumKeyBlocks; ++block) {
            for (const auto& [first, count] : KeyBlockRegisters(static_cast<KeyBlock>(block))) {
                for (u32 index = first; index < first + count && index < groups.size(); ++index) {
                    if (groups[index] != RegGroup::Viewport) {
                        continue;
                    }
                    if (index >= transforms && index < transforms_end &&
                        (index - transforms) % transform_words != swizzle_word) {
                        continue;
                    }
                    result.push_back(index);
                }
            }
        }
        std::ranges::sort(result);
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }();
    return registers;
}

constexpr size_t GraphicsSets = 1024; ///< x4 ways: 4096 recipes
constexpr size_t ComputeSets = 256;   ///< x4 ways: 1024 recipes

/// The launch description words that make a dispatch's pipeline and bindings: all but the grid
/// dimensions (words 0xc, 0xd) and the constant buffer addresses (0x1d-0x2c, in the key only for
/// the slots that hold handles).
constexpr bool ShapeWord(size_t word) {
    return word != 0xc && word != 0xd && (word < 0x1d || word > 0x2c);
}

DC::Counters& GraphicsCounters() {
    return DC::ThreadStats().graphics;
}
DC::Counters& ComputeCounters() {
    return DC::ThreadStats().compute;
}

} // Anonymous namespace

DrawCacheGlue::DrawCacheGlue(Tegra::MaxwellDeviceMemoryManager& device_memory_)
    : device_memory{device_memory_}, graphics_table{GraphicsSets}, compute_table{ComputeSets},
      memo{Versions::Pages(), Versions::Stream(), Versions::Epochs()},
      graphics_sampler{DC::VerifyN()}, compute_sampler{DC::VerifyN()} {
    // Guest CPU writes to the words recipes captured must trap (and move their pages' versions):
    // their pages stay in the device memory's write tracking while a recipe holds them.
    watch.SetTracker([this](DAddr address, u64 size, s32 delta) {
        device_memory.UpdatePagesCachedCount(address, size, delta);
    });
}

DrawCacheGlue::~DrawCacheGlue() {
    graphics_table.Clear();
    compute_table.Clear();
    watch.Clear();
}

u64 DrawCacheGlue::GraphicsShape(bool is_indexed, const Tegra::Engines::Maxwell3D& maxwell3d,
                                 const Tegra::MemoryManager& memory) const {
    u64 shape = maxwell3d.v2_groups.Fingerprint(ShapeGroups);
    for (const u32 index : KeyViewportRegisters()) {
        shape = DC::Mix(shape, (u64{index} << 32) | maxwell3d.regs.reg_array[index]);
    }
    const KeyInputs inputs = FixedPipelineState::CurrentInputs(maxwell3d);
    shape = DC::Mix(shape, static_cast<u64>(inputs.topology));
    shape = DC::Mix(shape, static_cast<u64>(inputs.engine_state));
    shape = DC::Mix(shape, inputs.squashed_iterated_blend ? 1 : 0);
    shape = DC::Mix(shape, is_indexed ? 1 : 0);
    shape = DC::Mix(shape, static_cast<u64>(maxwell3d.regs.sampler_binding));
    shape = DC::Mix(shape,
                    const_cast<Tegra::Engines::Maxwell3D&>(maxwell3d).GetNativeMacroEngine()
                            .native_turing
                        ? 1
                        : 0);
    shape = DC::Mix(shape, reinterpret_cast<uintptr_t>(&maxwell3d));
    shape = DC::Mix(shape, reinterpret_cast<uintptr_t>(&memory));
    return shape;
}

u64 DrawCacheGlue::GraphicsKey(u64 shape, const DC::ShapeInfo& info,
                               const Tegra::Engines::Maxwell3D& maxwell3d) {
    // Relocation (AnyPS5's DrawRelocation, by construction): pointer-valued state the replay
    // resolves anyway stays out of the key, so draws that differ only in it share a recipe. That is
    // every uniform-only constant buffer address (bound per draw by the buffer cache), the index
    // buffer, and the vertex buffer addresses and limits: of the vertex streams only the layout the
    // pipeline reads (stride, enable, divisor, instancing, bound or not) is in the key.
    u64 key = DC::Mix(
        shape, maxwell3d.v2_groups.Fingerprint(info.relevant_groups &
                                               ~GroupBit(RegGroup::VertexStreams)));
    if ((info.relevant_groups & GroupBit(RegGroup::VertexStreams)) != 0) {
        const auto& regs = maxwell3d.regs;
        for (size_t index = 0; index < regs.vertex_streams.size(); ++index) {
            const auto& stream = regs.vertex_streams[index];
            if (stream.enable == 0) {
                continue;
            }
            key = DC::Mix(key, (u64{static_cast<u32>(index)} << 48) |
                                   (u64{stream.Address() != 0 ? 1U : 0U} << 40) |
                                   (u64{regs.vertex_stream_instances.is_instanced[index] != 0 ? 1U : 0U}
                                    << 32) |
                                   stream.stride.Value());
            key = DC::Mix(key, stream.frequency);
        }
    }
    for (size_t stage = 0; stage < info.key_cbufs.size(); ++stage) {
        const auto& cbufs = maxwell3d.state.shader_stages[stage].const_buffers;
        for (u32 mask = info.key_cbufs[stage]; mask != 0; mask &= mask - 1) {
            const auto& cbuf = cbufs[std::countr_zero(mask)];
            key = DC::Mix(key, cbuf.address);
            key = DC::Mix(key, (u64{cbuf.size} << 1) | (cbuf.enabled ? 1 : 0));
        }
    }
    return key;
}

u64 DrawCacheGlue::ComputeShape(const Tegra::Engines::KeplerCompute& kepler,
                                const Tegra::MemoryManager& memory) {
    const auto& qmd = kepler.launch_description;
    static_assert(sizeof(qmd) == Tegra::Engines::KeplerCompute::LaunchParams::NUM_LAUNCH_PARAMETERS *
                                     sizeof(u32));
    std::array<u32, Tegra::Engines::KeplerCompute::LaunchParams::NUM_LAUNCH_PARAMETERS> words{};
    std::memcpy(words.data(), &qmd, sizeof(words));
    u64 shape = 0x6b43a9b5f1e2d3c4ULL;
    for (size_t word = 0; word < words.size(); ++word) {
        if (ShapeWord(word)) {
            shape = DC::Mix(shape, (u64{static_cast<u32>(word)} << 32) | words[word]);
        }
    }
    const auto& regs = kepler.regs;
    shape = DC::Mix(shape, regs.tic.Address());
    shape = DC::Mix(shape, regs.tic.limit);
    shape = DC::Mix(shape, regs.tsc.Address());
    shape = DC::Mix(shape, regs.tsc.limit);
    shape = DC::Mix(shape, regs.code_loc.Address());
    shape = DC::Mix(shape, kepler.IsNativeMode() ? 1 : 0);
    shape = DC::Mix(shape, reinterpret_cast<uintptr_t>(&kepler));
    shape = DC::Mix(shape, reinterpret_cast<uintptr_t>(&memory));
    return shape;
}

u64 DrawCacheGlue::ComputeKey(u64 shape, const DC::ShapeInfo& info,
                              const Tegra::Engines::KeplerCompute& kepler) {
    const auto& qmd = kepler.launch_description;
    u64 key = DC::Mix(shape, qmd.const_buffer_enable_mask.Value());
    for (u32 mask = info.key_cbufs[0]; mask != 0; mask &= mask - 1) {
        const auto& cbuf = qmd.const_buffer_config[std::countr_zero(mask)];
        key = DC::Mix(key, cbuf.Address());
        key = DC::Mix(key, cbuf.size.Value());
    }
    return key;
}

DC::Miss DrawCacheGlue::Validate(Recipe& recipe, const void* engine,
                                 const Tegra::MemoryManager& memory, bool compute) {
    const Versions::Generations& gens = Versions::Gens();
    if (recipe.engine != engine || recipe.memory != &memory ||
        recipe.map_serial != memory.MapSerial() ||
        recipe.host_serial != memory.HostPointerSerial() ||
        recipe.rescale_epoch != gens.GetEpoch(Versions::Epoch::Rescale)) {
        return DC::Miss::Epoch;
    }
    if (recipe.pipeline_epoch != gens.GetEpoch(Versions::Epoch::Pipelines)) {
        return DC::Miss::Pipeline;
    }
    if (!DC::HandlesAlive(recipe, gens)) {
        return DC::Miss::Generation;
    }
    bool restamped = false;
    bool buffers_changed = false;
    const bool stats = V2::Stats::Enabled();
    const DC::Miss miss = DC::ValidateInputs(recipe.inputs, memo, Versions::Pages(), true,
                                             &restamped, stats ? &buffers_changed : nullptr);
    DC::Counters& counters = compute ? ComputeCounters() : GraphicsCounters();
    counters.restamped += restamped ? 1 : 0;
    counters.buffers_written += buffers_changed && miss == DC::Miss::Count ? 1 : 0;
    return miss;
}

DrawCacheGlue::Decision DrawCacheGlue::DecideDraw(bool continued, bool is_indexed,
                                                  bool is_indirect,
                                                  const Tegra::Engines::Maxwell3D& maxwell3d,
                                                  const Tegra::MemoryManager& memory) {
    DC::Counters& counters = GraphicsCounters();
    ++counters.lookups;
    pending_draw = {};
    const bool cache_on = !is_indirect && maxwell3d.draw_manager.draw_state
                                              .inline_index_draw_indexes.empty();
    const Decision decision = DC::Decide(
        continued, cache_on,
        [&](DC::Miss& miss) -> Recipe* {
            const u64 shape = GraphicsShape(is_indexed, maxwell3d, memory);
            pending_draw.shape = shape;
            pending_draw.valid = true;
            const DC::ShapeInfo* const info = graphics_shapes.Find(shape);
            if (info == nullptr) {
                miss = DC::Miss::NoShape;
                return nullptr;
            }
            Recipe* const recipe = graphics_table.Find(GraphicsKey(shape, *info, maxwell3d));
            if (recipe == nullptr) {
                miss = DC::Miss::NoEntry;
            }
            return recipe;
        },
        [&](Recipe& recipe) { return Validate(recipe, &maxwell3d, memory, false); },
        [this] { return V2::Verify(V2::Switch::DrawCache) && graphics_sampler.Next(); });
    if (decision.path == DC::Path::Continued) {
        ++counters.continued;
    } else if (decision.path == DC::Path::Replay) {
        // Counted as a hit once the replay recorded the draw (NoteReplayed).
        if (decision.verify) {
            ++counters.verify_samples;
        }
    } else {
        ++counters.misses[static_cast<size_t>(decision.miss)];
        counters.invalidated += decision.invalidated ? 1 : 0;
    }
    return decision;
}

DrawCacheGlue::RecipeCapture* DrawCacheGlue::BeginDrawCapture(
    const GraphicsPipeline& pipeline, bool is_indexed, bool is_indirect, bool mesh,
    bool frame_dump, const Tegra::Engines::Maxwell3D& maxwell3d) {
    static_cast<void>(is_indexed);
    if (!pending_draw.valid || is_indirect || mesh || frame_dump || !pipeline.TakesRecipes() ||
        !pipeline.IsBuilt() || maxwell3d.regs.transform_feedback_enabled != 0 ||
        !maxwell3d.draw_manager.draw_state.inline_index_draw_indexes.empty()) {
        return nullptr;
    }
    capture.Clear();
    built.Clear();
    // Before the resolve reads anything: a write after this is newer than the recipe.
    built.inputs.stamp = Versions::Pages().Stamp();
    pending_draw.capturing = true;
    return &capture;
}

bool DrawCacheGlue::FillFromCapture(Recipe& recipe, bool compute, Tegra::MemoryManager& memory,
                                    TextureCache& texture_cache) {
    if (capture.ineligible) {
        return false;
    }
    // Every word's page goes into the write tracking before its bytes are checked against what
    // the resolve read: a guest CPU write after the check traps (and moves the page's version),
    // one before it shows in the bytes. On failure the pages held so far are let go.
    bool ok = false;
    SCOPE_EXIT {
        if (!ok) {
            Drop(recipe);
            recipe.inputs.words.clear();
        }
    };
    const Versions::Generations& gens = Versions::Gens();
    const auto add_word = [&](const DC::InputRead& read) {
        if ((read.address & Core::Memory::YUZU_PAGEMASK) + read.size > Core::Memory::YUZU_PAGESIZE) {
            return false;
        }
        const u8* const host = memory.GetPointer(read.address);
        const std::optional<DAddr> device = memory.GpuToCpuAddress(read.address);
        if (host == nullptr || !device) {
            return false;
        }
        DC::Capture word{
            .host = host,
            .device = *device,
            .size = read.size,
            .what = read.what,
        };
        std::memcpy(word.bytes.data(), read.bytes.data(), read.size);
        watch.Hold(word.device, word.size);
        recipe.inputs.words.push_back(word);
        // Memory must still hold what the resolve read.
        return std::memcmp(host, read.bytes.data(), read.size) == 0;
    };
    for (const DC::InputRead& read : capture.reads) {
        if (!add_word(read)) {
            return false;
        }
    }
    const auto add_descriptor = [&](bool sampler, u32 index) {
        DC::Capture word{.size = 32, .what = sampler ? DC::Capture::What::Tsc : DC::Capture::What::Tic};
        GPUVAddr address{};
        if (!texture_cache.CaptureDescriptor(compute, sampler, index, address, word.host,
                                             std::span<u8, 32>{word.bytes})) {
            return false;
        }
        if (word.host == nullptr) {
            return true;
        }
        const std::optional<DAddr> device = memory.GpuToCpuAddress(address);
        if (!device) {
            return false;
        }
        word.device = *device;
        watch.Hold(word.device, word.size);
        recipe.inputs.words.push_back(word);
        return std::memcmp(word.host, word.bytes.data(), word.size) == 0;
    };
    recipe.views = capture.views;
    for (DC::RecipeView& view : recipe.views) {
        view.generation = gens.Get(Versions::Kind::ImageView, view.id);
        const VideoCommon::ImageViewId id{view.id};
        if (id && !add_descriptor(false, view.index)) {
            return false;
        }
        if (const auto range = texture_cache.ViewImageRange(id)) {
            recipe.inputs.content.push_back({range->first, range->second, true});
        }
    }
    for (const u32 index : capture.tsc_indices) {
        if (!add_descriptor(true, index)) {
            return false;
        }
    }
    for (const u32 sampler : capture.samplers) {
        recipe.samplers.push_back({sampler, gens.Get(Versions::Kind::Sampler, sampler)});
    }
    recipe.push = capture.push;
    recipe.push_words = capture.push_words;
    recipe.heap_words = capture.heap_words;
    recipe.inputs.SealPages();
    ok = true;
    return true;
}

void DrawCacheGlue::Drop(const Recipe& recipe) {
    for (const DC::Capture& word : recipe.inputs.words) {
        watch.Drop(word.device, word.size);
    }
}

void DrawCacheGlue::Store(DC::RecipeTable& table, u64 key, Recipe& recipe,
                          const Decision& decision, bool compute) {
    DC::Counters& counters = compute ? ComputeCounters() : GraphicsCounters();
    recipe.key = key;
    if (decision.verify && decision.recipe != nullptr) {
        // The hit's recipe against the full resolve, unless the memo deferred a CPU write the
        // resolve saw (the guest did not order it before this work: not a mismatch).
        DC::CapturedInputs fresh = decision.recipe->inputs;
        if (DC::ValidateInputs(fresh, memo, Versions::Pages(), false) != DC::Miss::Count) {
            ++counters.verify_deferred;
        } else {
            std::string reasons;
            if (!DC::CompareRecipes(*decision.recipe, recipe, reasons)) {
                ++counters.verify_mismatches;
                DC::LogMismatch(compute, key, reasons);
            }
        }
    }
    bool evicted = false;
    Recipe& slot = table.Insert(key, evicted, [this](const Recipe& old) { Drop(old); });
    counters.evictions += evicted ? 1 : 0;
    ++counters.inserts;
    // The built recipe's pages are held already (FillFromCapture); the slot takes them over.
    slot = recipe;
    counters.watched_pages = watch.Pages();
}

void DrawCacheGlue::FinishDrawCapture(GraphicsPipeline& pipeline, const Decision& decision,
                                      bool is_indexed, const Tegra::Engines::Maxwell3D& maxwell3d,
                                      Tegra::MemoryManager& memory, TextureCache& texture_cache,
                                      BufferCache& buffer_cache) {
    if (!pending_draw.capturing) {
        return;
    }
    pending_draw.capturing = false;
    static_cast<void>(is_indexed);
    const Versions::Generations& gens = Versions::Gens();
    Recipe& recipe = built;
    recipe.pipeline = &pipeline;
    recipe.engine = &maxwell3d;
    recipe.memory = &memory;
    recipe.pipeline_epoch = gens.GetEpoch(Versions::Epoch::Pipelines);
    recipe.rescale_epoch = gens.GetEpoch(Versions::Epoch::Rescale);
    recipe.map_serial = memory.MapSerial();
    recipe.host_serial = memory.HostPointerSerial();
    if (!FillFromCapture(recipe, false, memory, texture_cache)) {
        return;
    }
    for (size_t stage = 0; stage < Tegra::Engines::Maxwell3D::Regs::MaxShaderStage; ++stage) {
        if (pipeline.StorageConstantBuffers()[stage] == 0) {
            continue;
        }
        buffer_cache.ForEachGraphicsStorageBinding(
            stage, [&](u32 index, bool is_written, const VideoCommon::Binding& binding) {
                recipe.storage.push_back({
                    .stage = static_cast<u8>(stage),
                    .index = static_cast<u8>(index),
                    .written = is_written,
                    .device_addr = binding.device_addr,
                    .gpu_addr = binding.gpu_addr,
                    .size = binding.size,
                    .segment_first = binding.segment_first,
                    .segment_count = binding.segment_count,
                });
                recipe.inputs.content.push_back({binding.device_addr, binding.size, false});
            });
    }
    const auto& targets = texture_cache.GetRenderTargets();
    for (size_t rt = 0; rt < DC::NumRecipeTargets - 1; ++rt) {
        const u32 id = targets.color_buffer_ids[rt].index;
        recipe.targets[rt] = {id, gens.Get(Versions::Kind::ImageView, id)};
    }
    const u32 depth = targets.depth_buffer_id.index;
    recipe.targets[DC::NumRecipeTargets - 1] = {depth, gens.Get(Versions::Kind::ImageView, depth)};
    if (Framebuffer* const framebuffer = texture_cache.GetFramebuffer()) {
        recipe.render_pass = reinterpret_cast<u64>(framebuffer->RenderPass());
    }
    // The shape learns what the bound shaders use (P3's relevance masking).
    DC::ShapeInfo info{.shape = pending_draw.shape};
    info.relevant_groups = (pipeline.ReadsVertexAttributes() ? GroupBit(RegGroup::VertexStreams)
                                                             : 0) |
                           (pipeline.UsesTexturePools() ? GroupBit(RegGroup::TexPools) : 0);
    for (size_t stage = 0; stage < info.key_cbufs.size(); ++stage) {
        info.key_cbufs[stage] =
            pipeline.HandleConstantBuffers()[stage] | pipeline.StorageConstantBuffers()[stage];
    }
    graphics_shapes.Insert(info);
    Store(graphics_table, GraphicsKey(pending_draw.shape, info, maxwell3d), recipe, decision, false);
    DC::ThreadStats().graphics_occupied = graphics_table.Occupied();
    DC::ThreadStats().graphics_capacity = graphics_table.Capacity();
}

void DrawCacheGlue::NoteReplayed(Recipe& recipe) {
    ++recipe.uses;
    ++GraphicsCounters().hits;
}

void DrawCacheGlue::NoteReplayRebuilt(Recipe& recipe) {
    static_cast<void>(recipe);
    ++GraphicsCounters().replays_failed;
}

void DrawCacheGlue::NoteDrawTime(bool replay, u64 ns) {
    DC::Counters& counters = GraphicsCounters();
    if (replay) {
        counters.replay_ns += ns;
        ++counters.replay_draws;
    } else {
        counters.resolve_ns += ns;
        ++counters.resolve_draws;
    }
}

DrawCacheGlue::Decision DrawCacheGlue::DecideDispatch(const Tegra::Engines::KeplerCompute& kepler,
                                                      const Tegra::MemoryManager& memory,
                                                      bool eligible) {
    DC::Counters& counters = ComputeCounters();
    ++counters.lookups;
    pending_dispatch = {};
    const Decision decision = DC::Decide(
        false, eligible,
        [&](DC::Miss& miss) -> Recipe* {
            const u64 shape = ComputeShape(kepler, memory);
            pending_dispatch.shape = shape;
            pending_dispatch.valid = true;
            const DC::ShapeInfo* const info = compute_shapes.Find(shape);
            if (info == nullptr) {
                miss = DC::Miss::NoShape;
                return nullptr;
            }
            Recipe* const recipe = compute_table.Find(ComputeKey(shape, *info, kepler));
            if (recipe == nullptr) {
                miss = DC::Miss::NoEntry;
            }
            return recipe;
        },
        [&](Recipe& recipe) { return Validate(recipe, &kepler, memory, true); },
        [this] { return V2::Verify(V2::Switch::DrawCache) && compute_sampler.Next(); });
    if (decision.path == DC::Path::Replay) {
        if (decision.verify) {
            ++counters.verify_samples;
        }
    } else {
        ++counters.misses[static_cast<size_t>(decision.miss)];
        counters.invalidated += decision.invalidated ? 1 : 0;
    }
    return decision;
}

DrawCacheGlue::RecipeCapture* DrawCacheGlue::BeginDispatchCapture(const ComputePipeline& pipeline,
                                                                  bool eligible) {
    if (!pending_dispatch.valid || !eligible || !pipeline.TakesRecipes()) {
        return nullptr;
    }
    capture.Clear();
    built.Clear();
    built.inputs.stamp = Versions::Pages().Stamp();
    pending_dispatch.capturing = true;
    return &capture;
}

void DrawCacheGlue::FinishDispatchCapture(ComputePipeline& pipeline, u64 native_key,
                                          const Decision& decision,
                                          const Tegra::Engines::KeplerCompute& kepler,
                                          Tegra::MemoryManager& memory,
                                          TextureCache& texture_cache) {
    if (!pending_dispatch.capturing) {
        return;
    }
    pending_dispatch.capturing = false;
    const Versions::Generations& gens = Versions::Gens();
    Recipe& recipe = built;
    recipe.pipeline = &pipeline;
    recipe.engine = &kepler;
    recipe.memory = &memory;
    recipe.native_key = native_key;
    recipe.pipeline_epoch = gens.GetEpoch(Versions::Epoch::Pipelines);
    recipe.rescale_epoch = gens.GetEpoch(Versions::Epoch::Rescale);
    recipe.map_serial = memory.MapSerial();
    recipe.host_serial = memory.HostPointerSerial();
    if (!FillFromCapture(recipe, true, memory, texture_cache)) {
        return;
    }
    DC::ShapeInfo info{.shape = pending_dispatch.shape};
    info.key_cbufs[0] = pipeline.HandleConstantBuffers();
    compute_shapes.Insert(info);
    Store(compute_table, ComputeKey(pending_dispatch.shape, info, kepler), recipe, decision, true);
    DC::ThreadStats().compute_occupied = compute_table.Occupied();
    DC::ThreadStats().compute_capacity = compute_table.Capacity();
}

void DrawCacheGlue::NoteDispatchReplayed(Recipe& recipe) {
    ++recipe.uses;
    ++ComputeCounters().hits;
}

void DrawCacheGlue::NoteDispatchRebuilt(Recipe& recipe) {
    static_cast<void>(recipe);
    ++ComputeCounters().replays_failed;
}

void DrawCacheGlue::NoteDispatchTime(bool replay, u64 ns) {
    DC::Counters& counters = ComputeCounters();
    if (replay) {
        counters.replay_ns += ns;
        ++counters.replay_draws;
    } else {
        counters.resolve_ns += ns;
        ++counters.resolve_draws;
    }
}

void DrawCacheGlue::Report() {
    DC::Report(VideoCommon::GpuSplit::OnBackThread() ? "GPU back thread" : "GPU thread", &memo);
}

} // namespace Vulkan
