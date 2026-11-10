// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Pipeline v2 P6 Step 0 (EDEN_NX2_V2_DRAWREC, notes/STEP0_SPEC.md): a draw resolved into a
// relocatable record (ResolveDraw, pure), its cache mutations applied (ApplyDraw) and the draw
// encoded from it (EncodeDraw). The resolve section reads only the StateView, guest memory and
// const cache lookups: no engine, no dirty flags, no recording.

#include <array>
#include <bit>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <fmt/ranges.h>

#include "common/logging.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_handle_program.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/textures/texture.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/resolve/dynamic_state.h"
#include "video_core/v2/resolve/state_view.h"
#include "video_core/v2/versions.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace R = VideoCommon::V2::Resolve;
namespace C = VideoCommon::V2::Commit;
using Tegra::Texture::TexturePair;
using VideoCore::Surface::PixelFormat;

// RESOLVE BEGIN

namespace {

/// A view of the record before its lookup: what ConfigureImpl collects as ImageViewInOut.
struct PendingView {
    u32 index;
    u8 stage;
    u8 kind;
    bool blacklist;
};

constexpr u8 KindTextureBuffer = 0;
constexpr u8 KindImageBuffer = 1;
constexpr u8 KindTexture = 2;
constexpr u8 KindImage = 3;

} // Anonymous namespace

void GraphicsPipeline::PushViewRequests(std::span<R::ResolvedView> views, bool blacklist,
                                        C::RequestList& textures) const {
    // PredictImageViews, every ready view as an ImagePrepare request (no prepare in place: as
    // the first requests of the list it has the same effect).
    const u64 table_changes = texture_cache.GraphicsImageTableChanges();
    for (R::ResolvedView& entry : views) {
        const auto lookup = texture_cache.LookupImageView(entry.tic_index, false);
        u32 id = lookup.id.index;
        if (lookup.request) {
            C::Request request{};
            request.kind = lookup.kind;
            request.resource = C::Resource::ImageView;
            request.address = entry.tic_index;
            request.size = table_changes;
            request.arg = 0;
            request.slot = lookup.id.index;
            const u32 proxy = textures.Push(request, true);
            // A prepare keeps the view it found (a deletion while applying rebuilds the list);
            // a table update or a creation is known once applied.
            if (lookup.kind != C::Kind::ImagePrepare) {
                id = proxy;
            }
            if (blacklist && entry.blacklist != 0 &&
                !(lookup.kind == C::Kind::ViewBind &&
                  lookup.id == VideoCommon::NULL_IMAGE_VIEW_ID)) {
                C::Request rescale{};
                rescale.kind = C::Kind::Rescale;
                rescale.resource = C::Resource::ImageView;
                rescale.slot = proxy;
                rescale.arg = C::ArgBlacklist;
                textures.Push(rescale);
            }
        }
        entry.view = {id, C::IsProxy(id)
                              ? 0U
                              : texture_cache.ViewGeneration(VideoCommon::ImageViewId{id})};
    }
}

template <class View>
void GraphicsPipeline::ResolveTextures(const View& view, R::ResolvedDraw& out,
                                       C::RequestList& textures) const {
    const auto& regs = view.Regs();
    const bool via_header_index{regs.sampler_binding ==
                                R::MaxwellRegs::SamplerBinding::ViaHeaderBinding};
    boost::container::small_vector<PendingView, 160> pending;
    boost::container::small_vector<R::ResolvedSampler, 96> samplers;
    boost::container::small_vector<VideoCommon::V2::DrawCache::InputRead, 64> reads;
    bool has_images = false;
    const auto add_read = [&](GPUVAddr address, u32 word) {
        VideoCommon::V2::DrawCache::InputRead read{
            .address = address, .size = 4, .what = VideoCommon::V2::DrawCache::Capture::What::Handle};
        std::memcpy(read.bytes.data(), &word, sizeof(word));
        reads.push_back(read);
    };
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!recipe_stages[stage]) {
            continue;
        }
        const Shader::Info& info{stage_infos[stage]};
        const auto cbuf = [&](u32 index) { return view.Cbuf(stage, index); };
        // The storage buffer descriptors (CaptureStorageDescriptor), before the handles.
        for (const auto& desc : info.storage_buffers_descriptors) {
            const GPUVAddr address = cbuf(desc.cbuf_index).address + desc.cbuf_offset;
            if (desc.indirect_offset != ~0U || !gpu_memory->IsFullyMappedRange(address, 16)) {
                out.h.flags |= R::DrawFlagTablePointers;
                continue;
            }
            VideoCommon::V2::DrawCache::InputRead read{
                .address = address,
                .size = 16,
                .what = VideoCommon::V2::DrawCache::Capture::What::StorageDesc,
            };
            gpu_memory->ReadBlockUnsafe(address, read.bytes.data(), read.bytes.size());
            reads.push_back(read);
        }
        const auto read_handle = [&](const auto& desc, u32 index) {
            if (desc.handle_program != ~0U) {
                // Read by the shader from a storage buffer: computed for this draw.
                out.h.flags |= R::DrawFlagComputedHandles;
                return TexturePair(
                    VideoCommon::EvaluateTextureHandle(
                        info, desc.handle_program, *gpu_memory,
                        [&](u32 index_) -> GPUVAddr {
                            if (index_ >= VideoCommon::V2::Resolve::CbufSlots) {
                                return GPUVAddr{};
                            }
                            const auto binding = cbuf(index_);
                            return binding.enabled != 0 ? binding.address : GPUVAddr{};
                        },
                        device.GetStorageBufferAlignment()),
                    via_header_index);
            }
            const u32 offset{desc.cbuf_offset + (index << desc.size_shift)};
            const GPUVAddr addr{cbuf(desc.cbuf_index).address + offset};
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(desc)>,
                                         Shader::TextureDescriptor> ||
                          std::is_same_v<std::remove_cvref_t<decltype(desc)>,
                                         Shader::TextureBufferDescriptor>) {
                if (desc.has_secondary) {
                    const GPUVAddr separate_addr{cbuf(desc.secondary_cbuf_index).address +
                                                 desc.secondary_cbuf_offset};
                    const u32 lhs_word{gpu_memory->Read<u32>(addr)};
                    const u32 rhs_word{gpu_memory->Read<u32>(separate_addr)};
                    add_read(addr, lhs_word);
                    add_read(separate_addr, rhs_word);
                    const u32 raw{(lhs_word << desc.shift_left) |
                                  (rhs_word << desc.secondary_shift_left)};
                    return TexturePair(raw, via_header_index);
                }
            }
            const u32 word{gpu_memory->Read<u32>(addr)};
            add_read(addr, word);
            return TexturePair(word, via_header_index);
        };
        const auto add_image = [&](const auto& desc, u8 kind, bool blacklist) {
            for (u32 index = 0; index < desc.count; ++index) {
                pending.push_back({read_handle(desc, index).first, static_cast<u8>(stage), kind,
                                   blacklist});
            }
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            add_image(desc, KindTextureBuffer, false);
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            add_image(desc, KindImageBuffer, false);
        }
        for (const auto& desc : info.texture_descriptors) {
            for (u32 index = 0; index < desc.count; ++index) {
                const auto handle{VideoCommon::HandleIsStoragePointer(info, desc, index)
                                      ? VideoCommon::NULL_TEXTURE_HANDLE
                                      : read_handle(desc, index)};
                pending.push_back({handle.first, static_cast<u8>(stage), KindTexture, false});
                // GetSamplerId in a sampler scope: a ready sampler, or a table update or a
                // creation (with a proxy) applied before the views.
                const auto lookup = texture_cache.LookupSampler(handle.second, false);
                u32 id = lookup.id.index;
                if (lookup.request) {
                    C::Request request{};
                    request.kind = lookup.kind;
                    request.resource = C::Resource::Sampler;
                    request.address = handle.second;
                    request.arg = 0;
                    request.slot = lookup.id.index;
                    const bool create = lookup.kind == C::Kind::SamplerCreate;
                    const u32 proxy = textures.Push(request, create);
                    id = create ? proxy : lookup.id.index;
                }
                samplers.push_back(R::ResolvedSampler{
                    .tsc_index = handle.second,
                    .sampler = {id, C::IsProxy(id) ? 0U
                                                   : texture_cache.SamplerGeneration(
                                                         VideoCommon::SamplerId{id})},
                });
            }
        }
        for (const auto& desc : info.image_descriptors) {
            has_images = true;
            add_image(desc, KindImage, desc.is_written);
        }
    }
    boost::container::small_vector<R::ResolvedView, 160> views;
    for (const PendingView& entry : pending) {
        views.push_back(R::ResolvedView{
            .tic_index = entry.index,
            .stage = entry.stage,
            .kind = entry.kind,
            .blacklist = static_cast<u8>(entry.blacklist ? 1 : 0),
        });
    }
    if (has_images) {
        out.h.flags |= R::DrawFlagBlacklistedImages;
    }
    PushViewRequests(std::span<R::ResolvedView>(views.data(), views.size()), has_images, textures);
    out.h.views = out.Push(std::span<const R::ResolvedView>(views.data(), views.size()));
    out.h.samplers = out.Push(std::span<const R::ResolvedSampler>(samplers.data(), samplers.size()));
    out.h.reads = out.Push(
        std::span<const VideoCommon::V2::DrawCache::InputRead>(reads.data(), reads.size()));
}

namespace {

/// The status of a draw the record path does not resolve (counted per reason).
R::ResolveStatus CommitOnly(R::ResolvedDraw& out, R::CommitOnlyReason reason) {
    out.h.status = R::ResolveStatus::CommitOnly;
    out.h.reason = reason;
    return R::ResolveStatus::CommitOnly;
}

} // Anonymous namespace

template <class View>
R::ResolveStatus GraphicsPipeline::ResolveBuffers(const View& view, const R::ResolveContext& ctx,
                                                  R::ResolvedDraw& out,
                                                  C::RequestList& buffers) const {
    const auto& regs = view.Regs();
    if (regs.transform_feedback_enabled != 0) {
        return CommitOnly(out, R::CommitOnlyReason::TransformFeedback);
    }
    if (ctx.is_indirect) {
        return CommitOnly(out, R::CommitOnlyReason::Indirect);
    }
    // A buffer the lookup can't find is created when the list applies (FindBuffer's proxy).
    const auto resolve_buffer = [&](VideoCommon::V2::Versions::WeakRef& ref, DAddr device_addr,
                                    u32 size) {
        if (ref.id != VideoCommon::BufferId{}.index) {
            return;
        }
        ref.id = buffers.Push(buffer_cache.MakeCreateRequest(device_addr, size, false), true);
        ref.generation = 0;
    };
    // Step 17: a sub-record of the previous record (same pipeline, buffers neither remapped nor
    // deleted since, its handles patched) is reused when its register groups hash equal.
    const R::ResolvedDraw* const previous = ctx.previous;
    // F2-01: sub-records compared by value (uniform VAs, texture buffer views) need only the
    // same pipeline (same lists in the same order) and buffers neither remapped nor deleted.
    const bool same_buffers = previous != nullptr && previous->h.pipeline == ctx.pipeline &&
                              previous->h.status == R::ResolveStatus::Ready &&
                              previous->h.map_serial == out.h.map_serial &&
                              previous->h.buffer_deletions == out.h.buffer_deletions;
    const bool reusable = same_buffers && out.h.groups_valid && previous->h.groups_valid;
    const auto same_groups = [&](std::initializer_list<VideoCommon::V2::RegGroup> groups) {
        for (const VideoCommon::V2::RegGroup group : groups) {
            const size_t index = static_cast<size_t>(group);
            if (previous->h.group_hashes[index] != out.h.group_hashes[index]) {
                return false;
            }
        }
        return true;
    };
    // UpdateIndexBuffer.
    if (ctx.is_indexed) {
        if (view.InlineIndexDraw()) {
            return CommitOnly(out, R::CommitOnlyReason::InlineIndex);
        }
        if (reusable && previous->h.indexed && previous->h.index.valid &&
            !C::IsProxy(previous->h.index.buffer.id) &&
            std::memcmp(&previous->h.draw, &out.h.draw, sizeof(out.h.draw)) == 0 &&
            same_groups({VideoCommon::V2::RegGroup::IndexBuffer,
                         VideoCommon::V2::RegGroup::DrawParams})) {
            out.h.index = previous->h.index;
        } else {
            out.h.index = buffer_cache.ResolveIndexBinding(view);
            if (out.h.index.valid) {
                resolve_buffer(out.h.index.buffer, out.h.index.daddr, out.h.index.size);
            }
        }
    } else {
        out.h.index = {}; // a base copied from an indexed draw (ResolveFromBase)
    }
    // UpdateVertexBuffers: every slot (a clean slot holds what its registers give).
    boost::container::small_vector<R::ResolvedVertex, VideoCommon::NUM_VERTEX_BUFFERS> vertex;
    out.h.vertex_mask = 0;
    const auto previous_vertex = reusable && same_groups({VideoCommon::V2::RegGroup::VertexStreams})
                                     ? previous->Span<const R::ResolvedVertex>(previous->h.vertex)
                                     : std::span<const R::ResolvedVertex>{};
    const bool reuse_vertex =
        !previous_vertex.empty() && std::ranges::none_of(previous_vertex, [](const auto& entry) {
            return C::IsProxy(entry.buffer.id);
        });
    if (reuse_vertex) {
        vertex.assign(previous_vertex.begin(), previous_vertex.end());
        out.h.vertex_mask = previous->h.vertex_mask;
    }
    for (u32 slot = 0; !reuse_vertex && slot < VideoCommon::NUM_VERTEX_BUFFERS; ++slot) {
        std::optional<R::ResolvedVertex> binding = buffer_cache.ResolveVertexBinding(view, slot);
        if (!binding) {
            continue;
        }
        resolve_buffer(binding->buffer, binding->daddr, binding->size);
        if (binding->buffer.id != VideoCommon::NULL_BUFFER_ID.index && binding->size != 0) {
            out.h.vertex_mask |= 1U << slot;
        }
        vertex.push_back(*binding);
    }
    // Per stage: UpdateUniformBuffers, UpdateStorageBuffers, UpdateTextureBuffers.
    boost::container::small_vector<R::ResolvedUbo, 32> ubos;
    boost::container::small_vector<R::ResolvedSsbo, 32> ssbos;
    boost::container::small_vector<R::ResolvedTbo, 16> tbos;
    const auto views = out.Span<const R::ResolvedView>(out.h.views);
    size_t view_index = 0;
    // F2-01: the previous record's uniform and texture buffer lists, entry for entry (same
    // pipeline: the same masks and descriptors in the same order).
    const auto previous_ubos = same_buffers
                                   ? previous->Span<const R::ResolvedUbo>(previous->h.ubo)
                                   : std::span<const R::ResolvedUbo>{};
    const auto previous_views = same_buffers
                                    ? previous->Span<const R::ResolvedView>(previous->h.views)
                                    : std::span<const R::ResolvedView>{};
    const auto previous_tbos = same_buffers && previous_views.size() == views.size()
                                   ? previous->Span<const R::ResolvedTbo>(previous->h.tbo)
                                   : std::span<const R::ResolvedTbo>{};
    u32 reused_ubos = 0;
    u32 reused_tbos = 0;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        out.h.ubo_masks[stage] = 0;
        if (!recipe_stages[stage]) {
            continue;
        }
        const Shader::Info& info{stage_infos[stage]};
        const u32 ubo_mask = enabled_uniform_buffer_masks[stage];
        out.h.ubo_masks[stage] = ubo_mask;
        for (u32 mask = ubo_mask; mask != 0; mask &= mask - 1) {
            const u32 slot = static_cast<u32>(std::countr_zero(mask));
            // A constant buffer bound at the same VA and size translates to the same device
            // range in the same buffer while nothing was remapped or deleted.
            if (const size_t at = ubos.size(); at < previous_ubos.size()) {
                const R::ResolvedUbo& kept = previous_ubos[at];
                const R::CbufBinding cbuf = view.Cbuf(stage, slot);
                if (kept.stage == stage && kept.slot == slot && kept.daddr != 0 &&
                    cbuf.enabled != 0 && kept.gpu_addr == cbuf.address &&
                    kept.size == cbuf.size && !C::IsProxy(kept.buffer.id)) {
                    ubos.push_back(kept);
                    ++reused_ubos;
                    continue;
                }
            }
            R::ResolvedUbo ubo = buffer_cache.ResolveUniformBinding(view, stage, slot);
            if (ubo.size != 0 || ubo.daddr != 0) {
                resolve_buffer(ubo.buffer, ubo.daddr, ubo.size);
            }
            ubos.push_back(ubo);
        }
        u32 ssbo_index = 0;
        for (const auto& desc : info.storage_buffers_descriptors) {
            bool multi_range = false;
            R::ResolvedSsbo ssbo = buffer_cache.ResolveStorageBinding(
                view, stage, ssbo_index, desc.cbuf_index, desc.cbuf_offset, desc.is_written,
                desc.indirect_offset, multi_range);
            if (multi_range) {
                return CommitOnly(out, R::CommitOnlyReason::MultiRangeStorage);
            }
            if (ssbo.size != 0 || ssbo.daddr != 0) {
                resolve_buffer(ssbo.buffer, ssbo.daddr, ssbo.size);
            }
            ssbos.push_back(ssbo);
            ++ssbo_index;
        }
        // Texture and image buffers from the stage's ready views (BindGraphicsTextureBuffer).
        u32 tbo_index = 0;
        const auto add_buffer = [&](const auto& desc, bool is_image) {
            for (u32 i = 0; i < desc.count; ++i, ++view_index, ++tbo_index) {
                const R::ResolvedView& entry = views[view_index];
                if (C::IsProxy(entry.view.id)) {
                    return false;
                }
                // The same view (id and generation): the same range and format.
                if (const size_t at = tbos.size(); at < previous_tbos.size() &&
                                                   previous_views[view_index].view == entry.view &&
                                                   previous_tbos[at].stage == stage &&
                                                   previous_tbos[at].index == tbo_index &&
                                                   !C::IsProxy(previous_tbos[at].buffer.id)) {
                    tbos.push_back(previous_tbos[at]);
                    ++reused_tbos;
                    continue;
                }
                const ImageView& image_view{
                    texture_cache.GetImageView(VideoCommon::ImageViewId{entry.view.id})};
                PixelFormat format{image_view.format};
                bool is_written = false;
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(desc)>,
                                             Shader::ImageBufferDescriptor>) {
                    is_written = desc.is_written;
                    if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                        format = *explicit_format;
                    }
                }
                R::ResolvedTbo tbo = buffer_cache.ResolveTextureBufferBinding(
                    stage, tbo_index, image_view.GpuAddr(), image_view.BufferSize(), format,
                    is_written, is_image);
                if (tbo.size != 0) {
                    resolve_buffer(tbo.buffer, tbo.daddr, tbo.size);
                }
                tbos.push_back(tbo);
            }
            return true;
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            if (!add_buffer(desc, false)) {
                return CommitOnly(out, R::CommitOnlyReason::TextureBufferViewCreate);
            }
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            if (!add_buffer(desc, true)) {
                return CommitOnly(out, R::CommitOnlyReason::TextureBufferViewCreate);
            }
        }
        view_index += Shader::NumDescriptors(info.texture_descriptors);
        view_index += Shader::NumDescriptors(info.image_descriptors);
    }
    out.h.vertex = out.Push(std::span<const R::ResolvedVertex>(vertex.data(), vertex.size()));
    out.h.ubo = out.Push(std::span<const R::ResolvedUbo>(ubos.data(), ubos.size()));
    out.h.ssbo = out.Push(std::span<const R::ResolvedSsbo>(ssbos.data(), ssbos.size()));
    out.h.tbo = out.Push(std::span<const R::ResolvedTbo>(tbos.data(), tbos.size()));
    if ((reused_ubos | reused_tbos) != 0) {
        R::RecordStats& stats = R::GetRecordStats();
        R::CountRecordStat(stats.reused_ubos, reused_ubos);
        R::CountRecordStat(stats.reused_tbos, reused_tbos);
    }
    return R::ResolveStatus::Ready;
}

template R::ResolveStatus GraphicsPipeline::ResolveBuffers<R::LiveView>(
    const R::LiveView&, const R::ResolveContext&, R::ResolvedDraw&, C::RequestList&) const;
template R::ResolveStatus GraphicsPipeline::ResolveBuffers<R::SnapshotView>(
    const R::SnapshotView&, const R::ResolveContext&, R::ResolvedDraw&, C::RequestList&) const;

template <class View>
R::ResolveStatus GraphicsPipeline::ResolveDraw(const View& view, const R::ResolveContext& ctx,
                                               R::ResolvedDraw& out, C::RequestList& textures,
                                               C::RequestList& buffers) const {
    namespace Versions = VideoCommon::V2::Versions;
    out.Clear();
    R::DrawHeader& h = out.h;
    // 1. Header and stamps.
    h.ordinal = ctx.ordinal;
    h.fingerprint = view.Groups().Fingerprint(VideoCommon::V2::AllGroups);
    h.group_hashes = view.Groups().hashes;
    h.groups_valid = view.Groups().valid;
    h.draw = view.Draw();
    h.topology = view.Topology();
    h.indexed = ctx.is_indexed;
    h.indirect = ctx.is_indirect;
    h.pipeline = ctx.pipeline;
    h.pipeline_epoch = ctx.pipeline_epoch;
    h.map_serial = gpu_memory->MapSerial();
    h.image_table_changes = texture_cache.GraphicsImageTableChanges();
    h.sampler_table_changes = texture_cache.GraphicsSamplerTableChanges();
    h.texture_bindings_serial = texture_cache.TextureBindingsSerial();
    h.buffer_deletions = buffer_cache.BufferDeletions();
    h.rescale_epoch = Versions::Gens().GetEpoch(Versions::Epoch::Rescale);
    h.residency = texture_cache.CurrentResidency();
    h.stages = recipe_stages;
    // 2. What the record path does not take yet (D3).
    if (const R::CommitOnlyReason reason = CommitOnlyOf(view, ctx);
        reason != R::CommitOnlyReason::None) {
        return CommitOnly(out, reason);
    }
    // 3. The descriptor tables this draw reads (SynchronizeDescriptors' inputs).
    h.tables = R::TablesOf(view);
    if (!texture_cache.DescriptorTablesMatch(h.tables)) {
        h.status = R::ResolveStatus::Stale;
        h.stale = R::StaleReason::DescriptorTables;
        return R::ResolveStatus::Stale;
    }
    // 4-5. Textures, then buffers (texture buffers come from the views).
    ResolveTextures(view, out, textures);
    if (const R::ResolveStatus status = ResolveBuffers(view, ctx, out, buffers);
        status != R::ResolveStatus::Ready) {
        return status;
    }
    // 6-8. Render targets, dynamic state, render area.
    ResolveTargetsAndDynamic(view, ctx, nullptr, out);
    h.texture_requests = {0, static_cast<u32>(textures.Requests().size())};
    h.buffer_requests = {0, static_cast<u32>(buffers.Requests().size())};
    h.status = R::ResolveStatus::Ready;
    return R::ResolveStatus::Ready;
}

template <class View>
R::CommitOnlyReason GraphicsPipeline::CommitOnlyOf(const View& view,
                                                   const R::ResolveContext& ctx) const {
    if (IsMeshPipeline()) {
        return IsEmulatedMesh() ? R::CommitOnlyReason::EmulatedMesh : R::CommitOnlyReason::Mesh;
    }
    if (ctx.frame_dump) {
        return R::CommitOnlyReason::FrameDump;
    }
    if (ctx.sampled_recording) {
        return R::CommitOnlyReason::SampledRecording;
    }
    if (view.Regs().transform_feedback_enabled != 0) {
        return R::CommitOnlyReason::TransformFeedback;
    }
    if (ctx.is_indexed && view.InlineIndexDraw()) {
        return R::CommitOnlyReason::InlineIndex;
    }
    if (ctx.is_indirect) {
        return R::CommitOnlyReason::Indirect;
    }
    return R::CommitOnlyReason::None;
}

template <class View>
void GraphicsPipeline::ResolveTargetsAndDynamic(const View& view, const R::ResolveContext& ctx,
                                                const R::ResolvedDraw* base,
                                                R::ResolvedDraw& out) const {
    R::DrawHeader& h = out.h;
    // 6. Render targets (a base with the same registers keeps its prediction).
    h.rt = R::RtKeyOf(view);
    if (base == nullptr || !(base->h.rt == h.rt)) {
        const auto prediction = texture_cache.PredictRenderTargets(h.rt);
        h.rt_ids = prediction.ids;
        h.rt_known = prediction.known;
        h.rt_inputs_hold = prediction.inputs_hold;
    }
    // 7. Dynamic state (scale-independent): ComputeDynamic reads the registers and the
    // topology; a base whose register groups that hold them hash equal has the same values.
    const R::ResolvedDraw* const dynamic_base = base != nullptr ? base : ctx.previous;
    const auto same_dynamic = [&] {
        using VideoCommon::V2::RegGroup;
        if (dynamic_base == nullptr || dynamic_base->h.status != R::ResolveStatus::Ready ||
            !dynamic_base->h.groups_valid || !h.groups_valid ||
            dynamic_base->h.topology != h.topology) {
            return false;
        }
        for (const RegGroup group : {RegGroup::Pipeline, RegGroup::Viewport,
                                     RegGroup::VertexStreams, RegGroup::IndexBuffer,
                                     RegGroup::RenderTargets}) {
            const size_t index = static_cast<size_t>(group);
            if (dynamic_base->h.group_hashes[index] != h.group_hashes[index]) {
                return false;
            }
        }
        return true;
    };
    if (same_dynamic()) {
        h.dynamic = dynamic_base->h.dynamic;
    } else {
        h.dynamic = R::ComputeDynamic(view, *ctx.features);
    }
    // 8. Render area, as ConfigureImpl pushes it (Switch 2 titles may never program the
    // surface clip; colour target 0 then).
    const auto& regs = view.Regs();
    h.render_area = {};
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (recipe_stages[stage] && stage_infos[stage].uses_render_area) {
            const u32 clip_width = regs.surface_clip.width, clip_height = regs.surface_clip.height;
            const bool clip_set = clip_width != 0 && clip_height != 0;
            h.render_area.uses = true;
            h.render_area.w = static_cast<f32>(clip_set ? clip_width : u32{regs.rt[0].width});
            h.render_area.h = static_cast<f32>(clip_set ? clip_height : u32{regs.rt[0].height});
        }
    }
}

template <class View>
R::ResolveStatus GraphicsPipeline::ResolveFromBase(const View& view, const R::ResolveContext& ctx,
                                                   const R::ResolvedDraw& base,
                                                   R::ResolvedDraw& out,
                                                   C::RequestList& buffers) const {
    namespace Versions = VideoCommon::V2::Versions;
    // The base's views, samplers and reads hold (a recipe validated its inputs; a continued
    // draw changed no texture state); its buffer lists are resolved again and appended.
    out.h = base.h;
    out.arena.assign(base.arena.begin(), base.arena.end());
    R::DrawHeader& h = out.h;
    h.ordinal = ctx.ordinal;
    h.fingerprint = view.Groups().Fingerprint(VideoCommon::V2::AllGroups);
    h.group_hashes = view.Groups().hashes;
    h.groups_valid = view.Groups().valid;
    h.draw = view.Draw();
    h.topology = view.Topology();
    h.indexed = ctx.is_indexed;
    h.indirect = ctx.is_indirect;
    h.pipeline = ctx.pipeline;
    h.pipeline_epoch = ctx.pipeline_epoch;
    h.map_serial = gpu_memory->MapSerial();
    h.image_table_changes = texture_cache.GraphicsImageTableChanges();
    h.sampler_table_changes = texture_cache.GraphicsSamplerTableChanges();
    h.texture_bindings_serial = texture_cache.TextureBindingsSerial();
    h.buffer_deletions = buffer_cache.BufferDeletions();
    h.rescale_epoch = Versions::Gens().GetEpoch(Versions::Epoch::Rescale);
    h.residency = texture_cache.CurrentResidency();
    h.status = R::ResolveStatus::Ready;
    h.reason = R::CommitOnlyReason::None;
    h.stale = R::StaleReason::None;
    if (const R::CommitOnlyReason reason = CommitOnlyOf(view, ctx);
        reason != R::CommitOnlyReason::None) {
        return CommitOnly(out, reason);
    }
    h.tables = R::TablesOf(view);
    if (!texture_cache.DescriptorTablesMatch(h.tables)) {
        h.status = R::ResolveStatus::Stale;
        h.stale = R::StaleReason::DescriptorTables;
        return R::ResolveStatus::Stale;
    }
    if (const R::ResolveStatus status = ResolveBuffers(view, ctx, out, buffers);
        status != R::ResolveStatus::Ready) {
        return status;
    }
    ResolveTargetsAndDynamic(view, ctx, &base, out);
    h.texture_requests = {0, 0};
    h.buffer_requests = {0, static_cast<u32>(buffers.Requests().size())};
    return R::ResolveStatus::Ready;
}

template R::ResolveStatus GraphicsPipeline::ResolveFromBase<R::LiveView>(
    const R::LiveView&, const R::ResolveContext&, const R::ResolvedDraw&, R::ResolvedDraw&,
    C::RequestList&) const;
template R::ResolveStatus GraphicsPipeline::ResolveFromBase<R::SnapshotView>(
    const R::SnapshotView&, const R::ResolveContext&, const R::ResolvedDraw&, R::ResolvedDraw&,
    C::RequestList&) const;

template R::ResolveStatus GraphicsPipeline::ResolveDraw<R::LiveView>(
    const R::LiveView&, const R::ResolveContext&, R::ResolvedDraw&, C::RequestList&,
    C::RequestList&) const;
template R::ResolveStatus GraphicsPipeline::ResolveDraw<R::SnapshotView>(
    const R::SnapshotView&, const R::ResolveContext&, R::ResolvedDraw&, C::RequestList&,
    C::RequestList&) const;

template void GraphicsPipeline::ResolveTextures<R::LiveView>(const R::LiveView&,
                                                             R::ResolvedDraw&,
                                                             C::RequestList&) const;
template void GraphicsPipeline::ResolveTextures<R::SnapshotView>(const R::SnapshotView&,
                                                                 R::ResolvedDraw&,
                                                                 C::RequestList&) const;

// RESOLVE END

void GraphicsPipeline::ObserveDrawRecord(std::span<const VideoCommon::ImageViewInOut> views,
                                         std::span<const VideoCommon::SamplerId> samplers,
                                         const RescalingPushConstant& rescaling,
                                         const RenderAreaPushConstant& render_area) {
    DrawRecObserver& observer = *drawrec_observer;
    if (observer.done || observer.resolved == nullptr) {
        return;
    }
    observer.done = true;
    observer.heap_words = heap_push.words;
    observer.rescaling = rescaling.Data();
    observer.payload.assign(guest_descriptor_queue.UpdateData(),
                            guest_descriptor_queue.UpdateData() +
                                guest_descriptor_queue.UpdateCount());
    observer.uses_render_area = render_area.uses_render_area;
    observer.render_area = render_area.words;
    observer.buffers_match = false;
    observer.derived_ok = false;
    const R::ResolvedDraw& resolved = *observer.resolved;
    if (resolved.h.status != R::ResolveStatus::Ready) {
        return;
    }
    R::ResolvedDraw& observed = observer.observed;
    observed.Clear();
    R::DrawHeader& h = observed.h;
    h.pipeline = this;
    h.indexed = resolved.h.indexed;
    // The views and samplers the configure bound, in ConfigureImpl's order.
    boost::container::small_vector<R::ResolvedView, 160> observed_views;
    size_t view_index = 0;
    const auto add = [&](u32 count, u8 stage, u8 kind) {
        for (u32 i = 0; i < count && view_index < views.size(); ++i, ++view_index) {
            const VideoCommon::ImageViewInOut& view = views[view_index];
            observed_views.push_back(R::ResolvedView{
                .tic_index = view.index,
                .view = {view.id.index, texture_cache.ViewGeneration(view.id)},
                .stage = stage,
                .kind = kind,
                .blacklist = static_cast<u8>(view.blacklist ? 1 : 0),
            });
        }
    };
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!recipe_stages[stage]) {
            continue;
        }
        const Shader::Info& info{stage_infos[stage]};
        const u8 s = static_cast<u8>(stage);
        add(Shader::NumDescriptors(info.texture_buffer_descriptors), s, KindTextureBuffer);
        add(Shader::NumDescriptors(info.image_buffer_descriptors), s, KindImageBuffer);
        add(Shader::NumDescriptors(info.texture_descriptors), s, KindTexture);
        add(Shader::NumDescriptors(info.image_descriptors), s, KindImage);
    }
    h.views = observed.Push(
        std::span<const R::ResolvedView>(observed_views.data(), observed_views.size()));
    // A sampler's TSC index is not kept by the configure: the resolved one, by position.
    const auto resolved_samplers = resolved.Span<const R::ResolvedSampler>(resolved.h.samplers);
    boost::container::small_vector<R::ResolvedSampler, 96> observed_samplers;
    for (size_t i = 0; i < samplers.size(); ++i) {
        observed_samplers.push_back(R::ResolvedSampler{
            .tsc_index = i < resolved_samplers.size() ? resolved_samplers[i].tsc_index : ~0U,
            .sampler = {samplers[i].index, texture_cache.SamplerGeneration(samplers[i])},
        });
    }
    h.samplers = observed.Push(
        std::span<const R::ResolvedSampler>(observed_samplers.data(), observed_samplers.size()));
    // The words read are not kept either (the recipe capture compares them, DRAWCACHE=verify).
    h.reads = observed.Push(resolved.Span<const VideoCommon::V2::DrawCache::InputRead>(
        resolved.h.reads));
    buffer_cache.ObserveGraphicsBindings(observed, resolved.h.indexed, recipe_stages);
    h.index.format = resolved.h.index.format;
    h.index.first = resolved.h.index.first;
    h.index.count = resolved.h.index.count;
    {
        // Texture buffer bindings keep no guest address.
        const auto resolved_tbos = resolved.Span<const R::ResolvedTbo>(resolved.h.tbo);
        auto observed_tbos = observed.Span<R::ResolvedTbo>(h.tbo);
        for (size_t i = 0; i < observed_tbos.size() && i < resolved_tbos.size(); ++i) {
            observed_tbos[i].gpu_addr = resolved_tbos[i].gpu_addr;
        }
    }
    const auto& targets = texture_cache.GetRenderTargets();
    for (size_t rt = 0; rt < R::NumRecordTargets - 1; ++rt) {
        h.rt_ids[rt] = targets.color_buffer_ids[rt].index;
    }
    h.rt_ids[R::NumRecordTargets - 1] = targets.depth_buffer_id.index;
    h.rt_known = true;
    h.dynamic = R::ComputeDynamic(R::LiveView{*maxwell3d}, *observer.features);
    h.render_area.uses = render_area.uses_render_area;
    h.render_area.w = render_area.uses_render_area ? render_area.words[0] : 0.0f;
    h.render_area.h = render_area.uses_render_area ? render_area.words[1] : 0.0f;

    // Creations during the configure: a proxy of that kind is a dependent match.
    u32 created = 0;
    const auto bit = [](R::RecordGroup group) { return 1U << static_cast<u32>(group); };
    if (texture_cache.ImageViewSlots() > observer.views_before) {
        created |= bit(R::RecordGroup::Views) | bit(R::RecordGroup::Rt);
    }
    if (texture_cache.SamplerSlots() > observer.samplers_before) {
        created |= bit(R::RecordGroup::Samplers);
    }
    if (buffer_cache.BufferSlots() > observer.buffers_before) {
        created |= bit(R::RecordGroup::Vertex);
    }
    // A table update to a view the lookup found binds that view: compared as the view.
    R::ResolvedDraw& predicted = observer.predicted;
    predicted.h = resolved.h;
    predicted.arena.assign(resolved.arena.begin(), resolved.arena.end());
    if (observer.textures != nullptr) {
        R::PredictKnownViewProxies(predicted, observer.textures->Requests());
    }
    R::RecordStats& stats = R::GetRecordStats();
    std::string reasons;
    u32 groups = 0;
    u32 dependent = 0;
    const bool equal =
        R::CompareRecords(predicted, observed, created, reasons, &groups, &dependent);
    constexpr u32 encoded_groups = (1U << static_cast<u32>(R::RecordGroup::Views)) |
                                   (1U << static_cast<u32>(R::RecordGroup::Samplers)) |
                                   (1U << static_cast<u32>(R::RecordGroup::Vertex)) |
                                   (1U << static_cast<u32>(R::RecordGroup::Index)) |
                                   (1U << static_cast<u32>(R::RecordGroup::Ubo)) |
                                   (1U << static_cast<u32>(R::RecordGroup::Ssbo)) |
                                   (1U << static_cast<u32>(R::RecordGroup::Tbo));
    observer.buffers_match = (groups & encoded_groups) == 0;
    if (!equal) {
        // A log budget per group: a frequent group must not hide the examples of the others.
        static std::array<std::atomic<u32>, R::NumRecordGroups> logged{};
        for (size_t group = 0; group < R::NumRecordGroups; ++group) {
            if (((groups >> group) & 1) == 0) {
                continue;
            }
            R::CountRecordStat(stats.mismatches[group]);
            if (logged[group].fetch_add(1, std::memory_order_relaxed) < 4) {
                LOG_WARNING(Render_Vulkan,
                            "V2 draw record verify: draw {} pipeline {:016x}: the record differs "
                            "from the old configure in {} ({}): {}",
                            resolved.h.ordinal, state_hash, R::record_group_names[group],
                            reasons,
                            R::DescribeRecordGroup(predicted, observed,
                                                   static_cast<R::RecordGroup>(group)));
            }
        }
    }
    R::CountRecordStat(stats.dependent, dependent);
}

// APPLY: the serial mutator of a record.

namespace {

/// Every buffer reference of a record (index, vertex, uniform, storage, texture buffers) with
/// its guest range.
template <typename Fn>
void ForEachRecordBuffer(R::ResolvedDraw& record, Fn&& fn) {
    R::DrawHeader& h = record.h;
    if (h.indexed && h.index.valid) {
        fn(h.index.buffer, h.index.daddr, h.index.size);
    }
    for (R::ResolvedVertex& entry : record.Span<R::ResolvedVertex>(h.vertex)) {
        fn(entry.buffer, entry.daddr, entry.size);
    }
    for (R::ResolvedUbo& entry : record.Span<R::ResolvedUbo>(h.ubo)) {
        if (entry.daddr != 0 || entry.size != 0) {
            fn(entry.buffer, entry.daddr, entry.size);
        }
    }
    for (R::ResolvedSsbo& entry : record.Span<R::ResolvedSsbo>(h.ssbo)) {
        if (entry.daddr != 0 || entry.size != 0) {
            fn(entry.buffer, entry.daddr, entry.size);
        }
    }
    for (R::ResolvedTbo& entry : record.Span<R::ResolvedTbo>(h.tbo)) {
        if (entry.size != 0) {
            fn(entry.buffer, entry.daddr, entry.size);
        }
    }
}

} // Anonymous namespace

void GraphicsPipeline::PatchRecordViews(R::ResolvedDraw& record,
                                        const C::RequestList& textures) const {
    for (R::ResolvedSampler& entry : record.Span<R::ResolvedSampler>(record.h.samplers)) {
        if (C::IsProxy(entry.sampler.id)) {
            entry.sampler.id = textures.Resolve(entry.sampler.id);
            entry.sampler.generation =
                texture_cache.SamplerGeneration(VideoCommon::SamplerId{entry.sampler.id});
        }
    }
    for (R::ResolvedView& entry : record.Span<R::ResolvedView>(record.h.views)) {
        if (C::IsProxy(entry.view.id)) {
            entry.view.id = textures.Resolve(entry.view.id);
            entry.view.generation =
                texture_cache.ViewGeneration(VideoCommon::ImageViewId{entry.view.id});
        }
    }
}

void GraphicsPipeline::PatchRecordBuffers(R::ResolvedDraw& record,
                                          const C::RequestList& buffers) const {
    ForEachRecordBuffer(record, [&](VideoCommon::V2::Versions::WeakRef& ref, DAddr, u32) {
        if (C::IsProxy(ref.id)) {
            ref.id = buffers.Resolve(ref.id);
            ref.generation = buffer_cache.BufferGeneration(VideoCommon::BufferId{ref.id});
        }
    });
    // A patched creation is enabled as UpdateVertexBufferSlot enables it.
    record.h.vertex_mask = 0;
    for (const R::ResolvedVertex& entry : record.Span<const R::ResolvedVertex>(record.h.vertex)) {
        if (entry.buffer.id != VideoCommon::NULL_BUFFER_ID.index && entry.size != 0) {
            record.h.vertex_mask |= 1U << entry.slot;
        }
    }
}

void GraphicsPipeline::RepeekRecordBuffers(R::ResolvedDraw& record,
                                           C::RequestList& buffers) const {
    // UpdateGraphicsBuffers' loop after a deletion: every binding looked up again.
    ForEachRecordBuffer(record, [&](VideoCommon::V2::Versions::WeakRef& ref, DAddr device_addr,
                                    u32 size) {
        const VideoCommon::BufferId id = buffer_cache.PeekBuffer(device_addr, size, false);
        if (id) {
            ref = {id.index, buffer_cache.BufferGeneration(id)};
        } else {
            ref = {buffers.Push(buffer_cache.MakeCreateRequest(device_addr, size, false), true),
                   0};
        }
    });
}

void GraphicsPipeline::RelookupRecordProxies(R::ResolvedDraw& record) const {
    // DeriveOnly: the old configure created what the proxies stood for; find it again.
    for (R::ResolvedSampler& entry : record.Span<R::ResolvedSampler>(record.h.samplers)) {
        if (C::IsProxy(entry.sampler.id)) {
            const auto lookup = texture_cache.LookupSampler(entry.tsc_index, false);
            entry.sampler = {lookup.id.index, texture_cache.SamplerGeneration(lookup.id)};
        }
    }
    for (R::ResolvedView& entry : record.Span<R::ResolvedView>(record.h.views)) {
        if (C::IsProxy(entry.view.id)) {
            const auto lookup = texture_cache.LookupImageView(entry.tic_index, false);
            entry.view = {lookup.id.index, texture_cache.ViewGeneration(lookup.id)};
        }
    }
    ForEachRecordBuffer(record, [&](VideoCommon::V2::Versions::WeakRef& ref, DAddr device_addr,
                                    u32 size) {
        if (C::IsProxy(ref.id)) {
            const VideoCommon::BufferId id = buffer_cache.PeekBuffer(device_addr, size, false);
            ref = {id.index, buffer_cache.BufferGeneration(id)};
        }
    });
}

bool GraphicsPipeline::RefreshRecordTextureBuffers(R::ResolvedDraw& record) const {
    // After the deleted-images loop: a texture buffer view may be another one now.
    const auto views = record.Span<const R::ResolvedView>(record.h.views);
    auto tbos = record.Span<R::ResolvedTbo>(record.h.tbo);
    size_t view_index = 0;
    size_t tbo_index = 0;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!recipe_stages[stage]) {
            continue;
        }
        const Shader::Info& info{stage_infos[stage]};
        u32 index = 0;
        const auto refresh = [&](const auto& desc, bool is_image) {
            for (u32 i = 0; i < desc.count; ++i, ++view_index, ++index, ++tbo_index) {
                if (tbo_index >= tbos.size() || C::IsProxy(views[view_index].view.id)) {
                    return false;
                }
                const ImageView& image_view{
                    texture_cache.GetImageView(VideoCommon::ImageViewId{views[view_index].view.id})};
                PixelFormat format{image_view.format};
                bool is_written = false;
                if constexpr (std::is_same_v<std::remove_cvref_t<decltype(desc)>,
                                             Shader::ImageBufferDescriptor>) {
                    is_written = desc.is_written;
                    if (const auto explicit_format{PixelFormatFromImageFormat(desc.format)}) {
                        format = *explicit_format;
                    }
                }
                tbos[tbo_index] = buffer_cache.ResolveTextureBufferBinding(
                    stage, index, image_view.GpuAddr(), image_view.BufferSize(), format,
                    is_written, is_image);
            }
            return true;
        };
        for (const auto& desc : info.texture_buffer_descriptors) {
            if (!refresh(desc, false)) {
                return false;
            }
        }
        for (const auto& desc : info.image_buffer_descriptors) {
            if (!refresh(desc, true)) {
                return false;
            }
        }
        view_index += Shader::NumDescriptors(info.texture_descriptors);
        view_index += Shader::NumDescriptors(info.image_descriptors);
    }
    return true;
}

GraphicsPipeline::ApplyResult GraphicsPipeline::ApplyDraw(
    R::ResolvedDraw& record, C::RequestList& textures, C::RequestList& buffers, DrawApplyOut& out,
    ApplyMode mode, C::Stage& stage, VideoCommon::V2::DrawCache::Recipe* recipe) {
    R::RecordStats& stats = R::GetRecordStats();
    R::DrawHeader& h = record.h;
    const bool derive_only = mode == ApplyMode::DeriveOnly;
    const bool from_template = mode == ApplyMode::Template && recipe != nullptr;
    bool views_relooked = false; ///< the deleted-images loop looked the views up again
    out.Clear();
    if (derive_only) {
        RelookupRecordProxies(record);
    } else if (from_template) {
        // 1. A hit's views are the template's: what FillImageViews does to the views it finds
        // (uploads, alias copies, the LRU), once per residency stamp (F2-09).
        for (const R::ResolvedView& entry : record.Span<const R::ResolvedView>(h.views)) {
            out.views.push_back(VideoCommon::ImageViewInOut{
                .index = entry.tic_index,
                .blacklist = entry.blacklist != 0,
                .id = VideoCommon::ImageViewId{entry.view.id},
            });
        }
        if (!texture_cache.PrepareResidentViews(
                std::span<const VideoCommon::ImageViewInOut>(out.views.data(), out.views.size()),
                recipe->residency)) {
            return ApplyResult::Rebuild;
        }
        // 2-3. Buffer creations (relocated bindings outside every buffer).
        static_cast<void>(buffer_cache.TakeDeletedBuffers());
        for (;;) {
            stage.Apply(buffers, [&](const C::Request& request) {
                return buffer_cache.ApplyDrawRequest(request);
            });
            PatchRecordBuffers(record, buffers);
            if (!buffer_cache.TakeDeletedBuffers()) {
                break;
            }
            R::CountRecordStat(stats.deleted_loops);
            buffers.Reset(buffers.Segment());
            RepeekRecordBuffers(record, buffers);
        }
    } else {
        // 1. Samplers, then views (CommitSamplerLookups, FillImageViewsCommit's loop: a deletion,
        // or a blacklisted image scaled down, looks the views up again).
        const bool blacklist = (h.flags & R::DrawFlagBlacklistedImages) != 0;
        static_cast<void>(texture_cache.TakeDeletedImages());
        bool& looped = views_relooked;
        for (;;) {
            bool has_blacklisted = false;
            stage.Apply(textures, [&](const C::Request& request) {
                return texture_cache.ApplyDrawRequest(request, has_blacklisted);
            });
            PatchRecordViews(record, textures);
            if (!texture_cache.TakeDeletedImages() && !(blacklist && has_blacklisted)) {
                break;
            }
            R::CountRecordStat(stats.deleted_loops);
            looped = true;
            textures.Reset(textures.Segment());
            PushViewRequests(record.Span<R::ResolvedView>(h.views), blacklist, textures);
        }
        if (looped && h.tbo.count != 0 && !RefreshRecordTextureBuffers(record)) {
            return ApplyResult::Failed;
        }
        // 2. Buffer creations (UpdateGraphicsBuffers' loop: a deletion looks every binding up
        // again).
        static_cast<void>(buffer_cache.TakeDeletedBuffers());
        for (;;) {
            stage.Apply(buffers, [&](const C::Request& request) {
                return buffer_cache.ApplyDrawRequest(request);
            });
            // 3. The record's proxies patched.
            PatchRecordBuffers(record, buffers);
            if (!buffer_cache.TakeDeletedBuffers()) {
                break;
            }
            R::CountRecordStat(stats.deleted_loops);
            buffers.Reset(buffers.Segment());
            RepeekRecordBuffers(record, buffers);
        }
    }
    // 4. TraceTablePointer stays with the old bind (an env diagnostic of table pointers).
    // 5. The image descriptors (handles, heap slots, modification marks, rescaling bits).
    if (!from_template) {
        out.views.clear();
        for (const R::ResolvedView& entry : record.Span<const R::ResolvedView>(h.views)) {
            out.views.push_back(VideoCommon::ImageViewInOut{
                .index = entry.tic_index,
                .blacklist = entry.blacklist != 0,
                .id = VideoCommon::ImageViewId{entry.view.id},
            });
        }
    }
    for (const R::ResolvedSampler& entry : record.Span<const R::ResolvedSampler>(h.samplers)) {
        out.samplers.push_back(VideoCommon::SamplerId{entry.sampler.id});
    }
    // F2-04 (§7 descriptor row): a hit replays the recipe's heap and rescaling words while no
    // heap slot was recycled or freed since a hit found them equal; the slots are marked used
    // once per submission.
    const u64 heap_releases =
        heap_pipeline && heap_push.heap != nullptr ? heap_push.heap->Releases() : 0;
    namespace Versions = VideoCommon::V2::Versions;
    const u64 rescale_epoch = Versions::Gens().GetEpoch(Versions::Epoch::Rescale);
    // Step 18: a continued draw samples the last draw's views and samplers; its words hold while
    // no heap slot was recycled or freed and the rescale epoch is the same.
    const bool keepable = heap_pipeline && heap_push.heap != nullptr && !heap_push.verify &&
                          HeapImagesOnly();
    if (mode == ApplyMode::Continued && keepable && !views_relooked &&
        kept_heap_releases == heap_releases &&
        kept_rescale_epoch == rescale_epoch &&
        kept_rescaling.size() == out.rescaling.Data().size()) {
        out.heap_words.assign(kept_heap_words.begin(), kept_heap_words.end());
        out.rescaling.Assign(std::span<const u32>(kept_rescaling.data(), kept_rescaling.size()));
        out.heap_replayed = true;
        if (const u64 tick = scheduler.CurrentTick(); kept_heap_tick != tick) {
            kept_heap_tick = tick;
            TouchHeapWords(kept_heap_words);
        }
    } else if (from_template && HeapWordsReplayable(*recipe)) {
        out.heap_words.assign(recipe->heap_words.begin(), recipe->heap_words.end());
        out.rescaling.Assign(std::span<const u32>(recipe->push.data(), out.rescaling.Data().size()));
        out.heap_replayed = true;
        if (const u64 tick = scheduler.CurrentTick(); recipe->heap_tick != tick) {
            recipe->heap_tick = tick;
            TouchHeapWords(recipe->heap_words);
        }
    } else {
        BindlessHeap* const heap = heap_pipeline ? heap_push.heap : nullptr;
        const VideoCommon::SamplerId* samplers_it{out.samplers.data()};
        const VideoCommon::ImageViewInOut* views_it{out.views.data()};
        for (size_t index = 0; index < NUM_STAGES; ++index) {
            if (!recipe_stages[index]) {
                continue;
            }
            const size_t words_before = out.heap_words.size();
            DeriveImageDescriptors(texture_cache, stage_infos[index], samplers_it, views_it, heap,
                                   heap_push.verify, out, derive_only);
            out.heap_stage_words[index] = static_cast<u32>(out.heap_words.size() - words_before);
        }
        if (from_template && HeapWordsLearnable(*recipe) && out.heap_words == recipe->heap_words &&
            std::equal(out.rescaling.Data().begin(), out.rescaling.Data().end(),
                       recipe->push.begin())) {
            // This hit found the recipe's words: later hits replay them while no slot moves.
            recipe->heap_releases = heap_releases;
            recipe->heap_tick = scheduler.CurrentTick();
        }
    }
    if (keepable && !derive_only && !(mode == ApplyMode::Continued && out.heap_replayed)) {
        // The words of the views a continued draw after this one samples (whatever path).
        kept_heap_words.assign(out.heap_words.begin(), out.heap_words.end());
        kept_rescaling.assign(out.rescaling.Data().begin(), out.rescaling.Data().end());
        kept_heap_releases = heap_push.heap->Releases();
        kept_rescale_epoch = rescale_epoch;
        kept_heap_tick = scheduler.CurrentTick();
    }
    if (!derive_only) {
        // 6. The heap slots reserved get their descriptors.
        if (heap_pipeline) {
            heap_push.heap->Commit();
        }
        // 7-8. Render targets (D4: the old update), the feedback-loop check.
        if (from_template) {
            // A hit (ConfigureRecipe's rule): with clean registers and the recipe's targets
            // bound the update resolves nothing; other views mean the cache moved under them.
            constexpr size_t NumColors = VideoCommon::V2::DrawCache::NumRecipeTargets - 1;
            bool targets_unchanged = false;
            if (texture_cache.CommitActive()) [[unlikely]] {
                std::array<u32, NumColors> colors{};
                for (size_t rt = 0; rt < NumColors; ++rt) {
                    colors[rt] = recipe->targets[rt].id;
                }
                targets_unchanged =
                    texture_cache.RenderTargetsUnchanged(colors, recipe->targets[NumColors].id);
                texture_cache.NoteRecipeRenderTargets();
            }
            texture_cache.UpdateRenderTargets(false);
            if (!targets_unchanged) {
                const auto& targets = texture_cache.GetRenderTargets();
                for (size_t rt = 0; rt < NumColors; ++rt) {
                    if (targets.color_buffer_ids[rt].index != recipe->targets[rt].id) {
                        return ApplyResult::Rebuild;
                    }
                }
                if (targets.depth_buffer_id.index != recipe->targets[NumColors].id) {
                    return ApplyResult::Rebuild;
                }
            }
        } else {
            texture_cache.UpdateRenderTargets(false);
        }
        texture_cache.CheckFeedbackLoop(
            std::span<const VideoCommon::ImageViewInOut>(out.views.data(), out.views.size()));
    }
    // 9.
    return IsBuilt() && (!pipeline || build->Failed()) ? ApplyResult::Failed : ApplyResult::Ok;
}

void GraphicsPipeline::VerifyDrawApply(R::ResolvedDraw& record, C::Stage& stage) {
    DrawRecObserver* const observer = drawrec_observer;
    if (observer == nullptr || !observer->done || record.h.status != R::ResolveStatus::Ready) {
        return;
    }
    C::RequestList unused;
    DrawApplyOut& out = observer->derived;
    if (ApplyDraw(record, unused, unused, out, ApplyMode::DeriveOnly, stage) != ApplyResult::Ok) {
        return;
    }
    observer->derived_ok = true;
    R::RecordStats& stats = R::GetRecordStats();
    const bool rescaling_equal = out.rescaling.Data() == observer->rescaling;
    // The configure's heap words interleave each stage's texel buffer words (pushed by its buffer
    // binds) ahead of its image words; compare where no stage has texel words in the heap.
    const bool texels = std::ranges::any_of(heap_texels, [](const HeapTexelMasks& masks) {
        return masks.heap != 0;
    });
    const bool heap_equal = texels || out.heap_words == observer->heap_words;
    if (!rescaling_equal) {
        R::CountRecordStat(stats.mismatches[static_cast<size_t>(R::RecordGroup::Push)]);
    }
    if (!heap_equal) {
        R::CountRecordStat(stats.mismatches[static_cast<size_t>(R::RecordGroup::HeapWords)]);
    }
    if (!rescaling_equal || !heap_equal) {
        static std::atomic<u32> logged{};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
            LOG_WARNING(Render_Vulkan,
                        "V2 draw record verify: draw {} pipeline {:016x}: the derived {} differ "
                        "from the old configure's ({} heap words vs {})",
                        record.h.ordinal, state_hash,
                        !rescaling_equal && !heap_equal ? "rescaling and heap words"
                        : !rescaling_equal              ? "rescaling words"
                                                        : "heap words",
                        out.heap_words.size(), observer->heap_words.size());
        }
    }
}

void GraphicsPipeline::NoteRecordDelta(const DrawApplyOut& out) {
    // ConfigureImpl's CaptureNextDelta block, for a draw the record path resolved.
    if (!std::exchange(capture_delta, false) || delta_blocker != DeltaBlocker::None) {
        return;
    }
    const auto same_view = [](const VideoCommon::ImageViewInOut& lhs,
                              const VideoCommon::ImageViewInOut& rhs) {
        return lhs.index == rhs.index && lhs.id == rhs.id;
    };
    if (std::exchange(verify_delta, false)) {
        delta_mismatch = !delta_views_valid || !std::ranges::equal(out.views, delta_views, same_view) ||
                         !std::ranges::equal(out.samplers, delta_samplers);
    }
    delta_views.assign(out.views.begin(), out.views.end());
    delta_samplers.assign(out.samplers.begin(), out.samplers.end());
    delta_recipe = nullptr;
    configured_stages = recipe_stages;
    delta_views_valid = true;
}

void GraphicsPipeline::FillRecipeCapture(const R::ResolvedDraw& record, const DrawApplyOut& out,
                                         VideoCommon::V2::DrawCache::RecipeCapture& capture) const {
    const R::DrawHeader& h = record.h;
    capture.Clear();
    capture.ineligible =
        (h.flags & (R::DrawFlagComputedHandles | R::DrawFlagTablePointers)) != 0;
    const auto reads = record.Span<const VideoCommon::V2::DrawCache::InputRead>(h.reads);
    capture.reads.assign(reads.begin(), reads.end());
    for (const R::ResolvedView& entry : record.Span<const R::ResolvedView>(h.views)) {
        capture.views.push_back({
            .index = entry.tic_index,
            .id = entry.view.id,
            .generation = 0,
            .blacklist = entry.blacklist != 0,
        });
    }
    for (const R::ResolvedSampler& entry : record.Span<const R::ResolvedSampler>(h.samplers)) {
        capture.samplers.push_back(entry.sampler.id);
        capture.tsc_indices.push_back(entry.tsc_index);
    }
    RenderAreaPushConstant render_area;
    render_area.uses_render_area = h.render_area.uses;
    if (h.render_area.uses) {
        render_area.words = {h.render_area.w, h.render_area.h};
    }
    // The heap words as the encode pushed them (texel words interleaved, as the configure's).
    NoteRecipePush(capture, out.rescaling, render_area, heap_push.words);
}

void GraphicsPipeline::NoteRecipeDelta(const VideoCommon::V2::DrawCache::Recipe& recipe,
                                       const DrawApplyOut& out) {
    // ConfigureRecipe's tail: a delta draw after this hit keeps what it bound (F2-14).
    if (delta_recipe != &recipe || delta_recipe_stamp != recipe.inputs.stamp ||
        !delta_views_valid) {
        delta_views.assign(out.views.begin(), out.views.end());
        delta_samplers.assign(out.samplers.begin(), out.samplers.end());
        delta_recipe = &recipe;
        delta_recipe_stamp = recipe.inputs.stamp;
    }
    configured_stages = recipe_stages;
    delta_views_valid = true;
    // The template is the next continued draw's base (its buffers are resolved again then).
    if (last_record_recipe != &recipe || last_record_stamp != recipe.inputs.stamp) {
        StoreLastRecord(recipe.record.Get());
        last_record_recipe = &recipe;
        last_record_stamp = recipe.inputs.stamp;
    }
}

// ENCODE: the draw recorded from its record.

bool GraphicsPipeline::EncodeDraw(const R::ResolvedDraw& record, const DrawApplyOut& out,
                                  const R::ResolvedDraw* previous, EncodeMode mode,
                                  VideoCommon::V2::DrawCache::Recipe* recipe,
                                  VideoCommon::V2::Versions::EpochMemo* memo,
                                  BufferReplay* replay) {
    static_cast<void>(previous);
    namespace Profile = VideoCommon::Nx2Profile;
    const R::DrawHeader& h = record.h;
    const bool try_replay = mode == EncodeMode::Replay && recipe != nullptr && memo != nullptr;
    // A write after this stamp is newer than the buffers recorded again below.
    const u64 buffer_stamp = try_replay ? VideoCommon::V2::Versions::Pages().Stamp() : 0;
    bool buffers_replayed = false;
    // 1-2. The channel state as the update passes leave it, then the bind scope they open.
    buffer_cache.SetUniformBuffersState(enabled_uniform_buffer_masks, &uniform_buffer_sizes);
    {
        Profile::Scope scope{Profile::Zone::UpdateBuffers};
        buffer_cache.EncodeGraphicsBindings(record);
        // 3. A hit binds from the recipe's buffers where they still hold (the channel state is
        // the record's: the replay only decides which syncs it skips).
        if (try_replay) {
            buffers_replayed = buffer_cache.ReplayGraphicsBuffers(recipe->buffers, h.indexed, *memo);
        }
        if (!buffers_replayed) {
            buffer_cache.BeginBindScope();
            buffer_cache.BindHostGeometryBuffers(h.indexed);
        }
    }
    // 4.
    {
        Profile::Scope scope{Profile::Zone::DescriptorAcquire};
        guest_descriptor_queue.Acquire(scheduler, num_descriptor_entries, uses_descriptor_buffer);
    }
    // 5. Per stage: its buffers (heap texel words first), then its image descriptors.
    heap_push.words.clear();
    ImageDescriptorCursor cursor{};
    size_t heap_cursor = 0;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (!recipe_stages[stage]) {
            continue;
        }
        {
            Profile::Scope scope{Profile::Zone::BindStageBuffers};
            if (buffers_replayed) {
                WithHeapTexels(stage, [&] {
                    buffer_cache.ReplayHostStageBuffers(stage, recipe->buffers, *memo);
                });
            } else {
                WithHeapTexels(stage, [&] { buffer_cache.BindHostStageBuffers(stage); });
            }
        }
        if (!out.heap_replayed) {
            const size_t words = out.heap_stage_words[stage];
            heap_push.words.insert(heap_push.words.end(), out.heap_words.begin() + heap_cursor,
                                   out.heap_words.begin() + heap_cursor + words);
            heap_cursor += words;
        }
        Profile::Scope scope{Profile::Zone::PushDescriptors};
        EmitImageDescriptors(guest_descriptor_queue, stage_infos[stage], heap_pipeline, out,
                             cursor);
    }
    if (out.heap_replayed) {
        heap_push.words.assign(out.heap_words.begin(), out.heap_words.end());
    }
    // 6.
    if (heap_pipeline) {
        AddHeapWords();
    }
    buffer_cache.CommitBinds();
    if (buffer_cache.any_buffer_uploaded) {
        buffer_cache.runtime.PostCopyBarrier();
        buffer_cache.any_buffer_uploaded = false;
    }
    if (try_replay) {
        const auto [replayed, synced] = buffer_cache.TakeReplayCounts();
        if (replay != nullptr) {
            *replay = BufferReplay{
                .replayed = buffers_replayed,
                .binds_replayed = replayed,
                .binds_synced = synced,
            };
        }
        if (!buffers_replayed || synced != 0) {
            // Some binding took the sync path: the recipe holds what was bound now.
            buffer_cache.RecordGraphicsBuffers(recipe->buffers, h.indexed, buffer_stamp);
        }
    }
    if (mode == EncodeMode::PayloadOnly) {
        // Verify: the old configure's draw is already set up (its render pass included).
        return true;
    }
    // 7. The fragment barrier decision, last before the draw (as in the old order).
    const std::span<const VideoCommon::ImageViewInOut> view_span{out.views.data(),
                                                                 out.views.size()};
    if (scheduler.HasFragmentBarrier() &&
        (scheduler.RenderPassHasStores() || texture_cache.SamplesRenderTarget(view_span))) {
        scheduler.RequestOutsideRenderPassOperationContext();
    }
    RenderAreaPushConstant render_area;
    render_area.uses_render_area = h.render_area.uses;
    if (h.render_area.uses) {
        render_area.words = {h.render_area.w, h.render_area.h};
    }
    // 8.
    Profile::Scope scope{Profile::Zone::ConfigureDraw};
    return ConfigureDraw(out.rescaling, render_area);
}

void GraphicsPipeline::VerifyDrawEncode(const R::ResolvedDraw& record) {
    DrawRecObserver* const observer = drawrec_observer;
    if (observer == nullptr || !observer->done || !observer->derived_ok ||
        !observer->buffers_match || record.h.status != R::ResolveStatus::Ready) {
        return;
    }
    // R-8: a second encode of equal bindings: the vertex and index binds repeat (harmless), its
    // CommitBinds finds clean ranges. Its stage buffers may take new stream allocations (a
    // version table miss, the root table bank, no CBUFVER/UBOREUSE): a stream entry matches
    // when buffer and range are equal and both allocations hold the same bytes.
    if (!EncodeDraw(record, observer->derived, nullptr, EncodeMode::PayloadOnly)) {
        return;
    }
    R::RecordStats& stats = R::GetRecordStats();
    const size_t count = guest_descriptor_queue.UpdateCount();
    const auto same_entry = [&](const DescriptorUpdateEntry& lhs,
                                const DescriptorUpdateEntry& rhs) {
        if (std::memcmp(&lhs, &rhs, sizeof(DescriptorUpdateEntry)) == 0) {
            return true;
        }
        if (lhs.buffer.buffer != rhs.buffer.buffer || lhs.buffer.range != rhs.buffer.range) {
            return false;
        }
        const auto& runtime = buffer_cache.runtime;
        const std::span<const u8> a =
            runtime.StreamBytes(lhs.buffer.buffer, lhs.buffer.offset, lhs.buffer.range);
        const std::span<const u8> b =
            runtime.StreamBytes(rhs.buffer.buffer, rhs.buffer.offset, rhs.buffer.range);
        return !a.empty() && a.size() == b.size() &&
               std::memcmp(a.data(), b.data(), a.size()) == 0;
    };
    bool payload_equal = count == observer->payload.size();
    for (size_t i = 0; payload_equal && i < count; ++i) {
        payload_equal = same_entry(guest_descriptor_queue.UpdateData()[i], observer->payload[i]);
    }
    // The heap words as the encode interleaves them (texel words, then image words, per stage).
    const bool push_equal =
        observer->derived.rescaling.Data() == observer->rescaling &&
        observer->uses_render_area == record.h.render_area.uses &&
        (!record.h.render_area.uses ||
         (observer->render_area[0] == record.h.render_area.w &&
          observer->render_area[1] == record.h.render_area.h)) &&
        heap_push.words == observer->heap_words;
    if (!payload_equal) {
        R::CountRecordStat(stats.mismatches[static_cast<size_t>(R::RecordGroup::Payload)]);
    }
    if (!push_equal) {
        R::CountRecordStat(stats.mismatches[static_cast<size_t>(R::RecordGroup::Push)]);
    }
    if (!payload_equal || !push_equal) {
        // The first differing payload entry, as raw words (which descriptor it is follows from
        // the pipeline's layout order: uniform, storage, texel, image buffers, then images).
        std::string detail;
        if (!payload_equal && count == observer->payload.size()) {
            const DescriptorUpdateEntry* const encoded = guest_descriptor_queue.UpdateData();
            for (size_t i = 0; i < count; ++i) {
                if (std::memcmp(&encoded[i], &observer->payload[i], sizeof(DescriptorUpdateEntry)) !=
                    0) {
                    std::array<u64, sizeof(DescriptorUpdateEntry) / sizeof(u64)> lhs{};
                    std::array<u64, sizeof(DescriptorUpdateEntry) / sizeof(u64)> rhs{};
                    std::memcpy(lhs.data(), &encoded[i], sizeof(lhs));
                    std::memcpy(rhs.data(), &observer->payload[i], sizeof(rhs));
                    detail = fmt::format(": entry {} record {:#x} configure {:#x}", i,
                                         fmt::join(lhs, ","), fmt::join(rhs, ","));
                    break;
                }
            }
        }
        static std::array<std::atomic<u32>, 2> logged{};
        if (logged[payload_equal ? 1 : 0].fetch_add(1, std::memory_order_relaxed) < 8) {
            LOG_WARNING(Render_Vulkan,
                        "V2 draw record verify: draw {} pipeline {:016x}: the encode's {} differ "
                        "from the old configure's ({} payload entries vs {}){}",
                        record.h.ordinal, state_hash,
                        !payload_equal ? "descriptor payload" : "push words", count,
                        observer->payload.size(), detail);
        }
    }
}

} // namespace Vulkan
