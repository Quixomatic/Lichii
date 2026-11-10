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
#include <type_traits>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "video_core/renderer_vulkan/pipeline_helper.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/shader_handle_program.h"
#include "video_core/texture_cache/texture_cache.h"
#include "video_core/textures/texture.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/resolve/state_view.h"
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
    // PredictImageViews, every ready view as an ImagePrepare request (no prepare in place: as the
    // first requests of the list it has the same effect).
    const bool blacklist = has_images;
    const u64 table_changes = texture_cache.GraphicsImageTableChanges();
    boost::container::small_vector<R::ResolvedView, 160> views;
    for (const PendingView& entry : pending) {
        const auto lookup = texture_cache.LookupImageView(entry.index, false);
        u32 id = lookup.id.index;
        if (lookup.request) {
            C::Request request{};
            request.kind = lookup.kind;
            request.resource = C::Resource::ImageView;
            request.address = entry.index;
            request.size = table_changes;
            request.arg = 0;
            request.slot = lookup.id.index;
            const u32 proxy = textures.Push(request, true);
            // A prepare keeps the view it found (a deletion while applying rebuilds the list);
            // a table update or a creation is known once applied.
            if (lookup.kind != C::Kind::ImagePrepare) {
                id = proxy;
            }
            if (blacklist && entry.blacklist &&
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
        views.push_back(R::ResolvedView{
            .tic_index = entry.index,
            .view = {id, C::IsProxy(id) ? 0U
                                        : texture_cache.ViewGeneration(VideoCommon::ImageViewId{id})},
            .stage = entry.stage,
            .kind = entry.kind,
            .blacklist = static_cast<u8>(entry.blacklist ? 1 : 0),
        });
    }
    if (blacklist) {
        out.h.flags |= R::DrawFlagBlacklistedImages;
    }
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
    // UpdateIndexBuffer.
    if (ctx.is_indexed) {
        if (view.InlineIndexDraw()) {
            return CommitOnly(out, R::CommitOnlyReason::InlineIndex);
        }
        out.h.index = buffer_cache.ResolveIndexBinding(view);
        if (out.h.index.valid) {
            resolve_buffer(out.h.index.buffer, out.h.index.daddr, out.h.index.size);
        }
    }
    // UpdateVertexBuffers: every slot (a clean slot holds what its registers give).
    boost::container::small_vector<R::ResolvedVertex, VideoCommon::NUM_VERTEX_BUFFERS> vertex;
    out.h.vertex_mask = 0;
    for (u32 slot = 0; slot < VideoCommon::NUM_VERTEX_BUFFERS; ++slot) {
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
    return R::ResolveStatus::Ready;
}

template R::ResolveStatus GraphicsPipeline::ResolveBuffers<R::LiveView>(
    const R::LiveView&, const R::ResolveContext&, R::ResolvedDraw&, C::RequestList&) const;
template R::ResolveStatus GraphicsPipeline::ResolveBuffers<R::SnapshotView>(
    const R::SnapshotView&, const R::ResolveContext&, R::ResolvedDraw&, C::RequestList&) const;

template void GraphicsPipeline::ResolveTextures<R::LiveView>(const R::LiveView&,
                                                             R::ResolvedDraw&,
                                                             C::RequestList&) const;
template void GraphicsPipeline::ResolveTextures<R::SnapshotView>(const R::SnapshotView&,
                                                                 R::ResolvedDraw&,
                                                                 C::RequestList&) const;

// RESOLVE END

} // namespace Vulkan
