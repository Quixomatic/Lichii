// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// Pipeline v2 P6 Step 0 (EDEN_NX2_V2_DRAWREC, notes/STEP0_SPEC.md): a draw resolved into a
// relocatable record (ResolveDraw, pure), its cache mutations applied (ApplyDraw) and the draw
// encoded from it (EncodeDraw). The resolve section reads only the StateView, guest memory and
// const cache lookups: no engine, no dirty flags, no recording.

#include <array>
#include <cstring>
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

template void GraphicsPipeline::ResolveTextures<R::LiveView>(const R::LiveView&,
                                                             R::ResolvedDraw&,
                                                             C::RequestList&) const;
template void GraphicsPipeline::ResolveTextures<R::SnapshotView>(const R::SnapshotView&,
                                                                 R::ResolvedDraw&,
                                                                 C::RequestList&) const;

// RESOLVE END

} // namespace Vulkan
