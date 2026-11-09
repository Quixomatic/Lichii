// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/v2/regsem.h"

// Pipeline v2 P3 (EDEN_NX2_V2_CONTDRAW): Cemu's continued draw (notes/06 §4.1). The register
// groups changed since the last recorded draw decide whether the next draw can keep its
// pipeline, image views, samplers and render pass (the delta draw of nx2_fast_draw.h).
namespace VideoCommon::V2 {

enum class GroupClass : u8 {
    Light,          ///< Re-sync the dirty buffers and emit the draw
    PipelineCheck,  ///< Light if the pipeline found for the new state is the same object
    Heavy,          ///< Resolve in full
};

/// Groups that never need more than the delta draw's buffer re-sync: draw parameters, index
/// buffer, constant buffer binds and data (their texture handle ranges are checked apart), and
/// state no draw reads.
inline constexpr u32 LightGroups = GroupBit(RegGroup::DrawParams) |
                                   GroupBit(RegGroup::IndexBuffer) | GroupBit(RegGroup::CbufBind) |
                                   GroupBit(RegGroup::CbufData) | GroupBit(RegGroup::Misc);
/// Groups that change the pipeline key or the shaders, or only dynamic state: light when the
/// pipeline lookup returns the same pipeline.
inline constexpr u32 PipelineCheckGroups =
    GroupBit(RegGroup::Pipeline) | GroupBit(RegGroup::Shaders) | GroupBit(RegGroup::Viewport) |
    GroupBit(RegGroup::VertexStreams);

/// The class of a set of changed groups for a pipeline that does (or doesn't) read the texture
/// and sampler pools.
[[nodiscard]] constexpr GroupClass ClassifyGroups(u32 groups, bool uses_texture_pools) {
    u32 heavy = GroupBit(RegGroup::RenderTargets);
    if (uses_texture_pools) {
        heavy |= GroupBit(RegGroup::TexPools);
    }
    if ((groups & heavy) != 0) {
        return GroupClass::Heavy;
    }
    if ((groups & PipelineCheckGroups) != 0) {
        return GroupClass::PipelineCheck;
    }
    return GroupClass::Light;
}

} // namespace VideoCommon::V2
