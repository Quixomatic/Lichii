// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>

#include "common/common_types.h"
#include "video_core/surface.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/resolve/state_view.h"

// Pipeline v2 P6 Step 0 (notes/STEP0_SPEC.md §2 item 19): the dynamic state a draw records,
// computed from registers alone. Each helper is the value part of one RasterizerVulkan::Update*:
// the legacy updates call them, and ComputeDynamic(view) fills a record's DynamicState with the
// same values. Host API mapping, state tracking and recording stay in the rasterizer.
namespace VideoCommon::V2::Resolve {

struct DepthBiasValues {
    f32 constant{};
    f32 clamp{};
    f32 slope{};
};

[[nodiscard]] inline bool IsD24DepthFormat(Tegra::DepthFormat format) {
    return format == Tegra::DepthFormat::Z24_UNORM_S8_UINT ||
           format == Tegra::DepthFormat::X8Z24_UNORM ||
           format == Tegra::DepthFormat::S8Z24_UNORM || format == Tegra::DepthFormat::V8Z24_UNORM;
}

/// UpdateDepthBias. `d24_rescale`: the host has no D24 depth and the title is in NEEDS_D24.
[[nodiscard]] inline DepthBiasValues DepthBiasOf(const MaxwellRegs& regs, bool d24_rescale) {
    f32 units = regs.depth_bias / 2.0f;
    if (d24_rescale && IsD24DepthFormat(regs.zeta.format)) {
        // https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-output-merger-stage-depth-bias
        const double rescale_factor =
            static_cast<double>(1ULL << (32 - 24)) / (static_cast<double>(0x1.ep+127));
        units = static_cast<f32>(static_cast<double>(units) * rescale_factor);
    }
    return DepthBiasValues{
        .constant = units,
        .clamp = regs.depth_bias_clamp,
        .slope = regs.slope_scale_depth_bias,
    };
}

[[nodiscard]] inline std::array<f32, 4> BlendConstantsOf(const MaxwellRegs& regs) {
    return {regs.blend_color.r, regs.blend_color.g, regs.blend_color.b, regs.blend_color.a};
}

[[nodiscard]] inline f32 LineWidthOf(const MaxwellRegs& regs) {
    return regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased;
}

/// UpdatePrimitiveRestartEnable. `restart_topologies`: bit per topology the host restarts.
[[nodiscard]] inline bool PrimitiveRestartOf(const MaxwellRegs& regs,
                                             MaxwellRegs::PrimitiveTopology topology,
                                             bool molten_vk, u32 restart_topologies) {
    if (molten_vk) {
        return true;
    }
    if (regs.primitive_restart.enabled == 0) {
        return false;
    }
    const u32 index = static_cast<u32>(topology);
    return index < 32 && ((restart_topologies >> index) & 1) != 0;
}

/// UpdateDepthBiasEnable: the polygon offset enable of the topology's primitive class.
[[nodiscard]] inline bool DepthBiasEnableOf(const MaxwellRegs& regs,
                                            MaxwellRegs::PrimitiveTopology topology) {
    constexpr size_t POINT = 0;
    constexpr size_t LINE = 1;
    constexpr size_t POLYGON = 2;
    static constexpr std::array POLYGON_OFFSET_ENABLE_LUT = {
        POINT,   // Points
        LINE,    // Lines
        LINE,    // LineLoop
        LINE,    // LineStrip
        POLYGON, // Triangles
        POLYGON, // TriangleStrip
        POLYGON, // TriangleFan
        POLYGON, // Quads
        POLYGON, // QuadStrip
        POLYGON, // Polygon
        LINE,    // LinesAdjacency
        LINE,    // LineStripAdjacency
        POLYGON, // TrianglesAdjacency
        POLYGON, // TriangleStripAdjacency
        POLYGON, // Patches
    };
    const std::array enabled_lut{
        regs.polygon_offset_point_enable,
        regs.polygon_offset_line_enable,
        regs.polygon_offset_fill_enable,
    };
    const u32 topology_index = static_cast<u32>(topology);
    return enabled_lut[POLYGON_OFFSET_ENABLE_LUT[topology_index]] != 0;
}

[[nodiscard]] inline bool DepthClampOf(const MaxwellRegs& regs) {
    using GeometryClip = MaxwellRegs::ViewportClipControl::GeometryClip;
    const GeometryClip clip = regs.viewport_clip_control.geometry_clip;
    return !(clip == GeometryClip::Passthrough || clip == GeometryClip::FrustumXYZ ||
             clip == GeometryClip::FrustumZ);
}

/// G3: the logic op enable after the AMD workaround (no logic op with float vertex attributes),
/// without writing the register.
[[nodiscard]] inline bool EffectiveLogicOpEnable(const MaxwellRegs& regs, bool amd) {
    const bool enable = regs.logic_op.enable != 0;
    if (!enable || !amd) {
        return enable;
    }
    const bool has_float = std::any_of(
        regs.vertex_attrib_format.begin(), regs.vertex_attrib_format.end(), [](const auto& attrib) {
            return attrib.type == MaxwellRegs::VertexAttribute::Type::Float;
        });
    return !has_float;
}

/// A render target's guest colour mask as RGBA bits (1, 2, 4, 8: the host component bits).
[[nodiscard]] inline u8 ColorMaskOf(const MaxwellRegs& regs, size_t index) {
    const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
    return static_cast<u8>((mask.R ? 1 : 0) | (mask.G ? 2 : 0) | (mask.B ? 4 : 0) |
                           (mask.A ? 8 : 0));
}

[[nodiscard]] inline bool BlendEnableOf(const MaxwellRegs& regs, size_t index) {
    bool is_integer = false;
    if (regs.rt[index].format != Tegra::RenderTargetFormat::NONE) {
        const auto format = VideoCore::Surface::PixelFormatFromRenderTargetFormat(regs.rt[index].format);
        is_integer = VideoCore::Surface::IsPixelFormatInteger(format);
    }
    return !is_integer && regs.blend.enable[index] != 0;
}

template <class GuestBlend>
[[nodiscard]] inline BlendEquation BlendEquationOf(const GuestBlend& guest) {
    return BlendEquation{
        .color_source = guest.color_source,
        .color_dest = guest.color_dest,
        .color_op = guest.color_op,
        .alpha_source = guest.alpha_source,
        .alpha_dest = guest.alpha_dest,
        .alpha_op = guest.alpha_op,
    };
}

/// UpdateBlending's equations. True when the squashed iterated-blend workaround replaces them
/// (`equations` then stay value-initialized).
[[nodiscard]] inline bool BlendEquationsOf(
    const MaxwellRegs& regs, bool squashed_iterated_blend,
    std::array<BlendEquation, MaxwellRegs::NumRenderTargets>& equations) {
    if (!regs.blend_per_target_enabled) {
        if (regs.iterated_blend.enable && squashed_iterated_blend) {
            equations = {};
            return true;
        }
        equations.fill(BlendEquationOf(regs.blend));
        return false;
    }
    for (size_t index = 0; index < MaxwellRegs::NumRenderTargets; ++index) {
        equations[index] = BlendEquationOf(regs.blend_per_target[index]);
    }
    return false;
}

[[nodiscard]] inline StencilFaceOp StencilFaceOpOf(const MaxwellRegs::StencilOp& op) {
    return StencilFaceOp{.fail = op.fail, .zfail = op.zfail, .zpass = op.zpass, .func = op.func};
}

/// UpdateVertexInput's guest part: the attributes and bindings the device takes.
inline void VertexInputOf(const MaxwellRegs& regs, u32 max_attributes, u32 max_bindings,
                          DynamicValues& values) {
    values.num_attributes = max_attributes;
    values.num_bindings = max_bindings;
    for (u32 index = 0; index < max_attributes; ++index) {
        const MaxwellRegs::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        const u32 binding = attribute.buffer;
        values.attributes[index] = VertexAttributeInput{
            .enabled = static_cast<u8>(!attribute.constant && binding < max_bindings),
            .binding = static_cast<u8>(binding),
            .type = attribute.type,
            .size = attribute.size,
            .offset = attribute.offset,
        };
    }
    for (u32 binding = 0; binding < max_bindings; ++binding) {
        const auto& stream = regs.vertex_streams[binding];
        const bool instanced = regs.vertex_stream_instances.IsInstancingEnabled(binding);
        values.bindings[binding] = VertexBindingInput{
            .stride = stream.stride.Value(),
            .divisor = instanced ? stream.frequency : 1U,
            .instanced = static_cast<u8>(instanced),
        };
    }
}

[[nodiscard]] inline ViewportInputs ViewportInputsOf(const MaxwellRegs& regs) {
    ViewportInputs inputs{};
    inputs.transform = regs.viewport_transform;
    inputs.scissor = regs.scissor_test;
    inputs.surface_clip = regs.surface_clip;
    inputs.window_origin = regs.window_origin;
    inputs.depth_mode = regs.depth_mode;
    inputs.viewport_scale_offset_enabled = regs.viewport_scale_offset_enabled;
    return inputs;
}

/// Every dynamic value a draw records (scale-independent), from the view alone.
template <StateView View>
[[nodiscard]] DynamicState ComputeDynamic(const View& view, const DynamicFeatures& features) {
    const MaxwellRegs& regs = view.Regs();
    const MaxwellRegs::PrimitiveTopology topology = view.Topology();
    DynamicState state{};
    state.viewport = ViewportInputsOf(regs);
    DynamicValues& v = state.values;
    const DepthBiasValues bias = DepthBiasOf(regs, features.d24_bias_rescale);
    v.depth_bias_constant = bias.constant;
    v.depth_bias_clamp = bias.clamp;
    v.depth_bias_slope = bias.slope;
    v.blend_constants = BlendConstantsOf(regs);
    v.depth_bounds_min = regs.depth_bounds[0];
    v.depth_bounds_max = regs.depth_bounds[1];
    v.stencil_two_side = regs.stencil_two_side_enable;
    v.stencil_front_ref = regs.stencil_front_ref;
    v.stencil_back_ref = regs.stencil_back_ref;
    v.stencil_front_write_mask = regs.stencil_front_mask;
    v.stencil_back_write_mask = regs.stencil_back_mask;
    v.stencil_front_compare_mask = regs.stencil_front_func_mask;
    v.stencil_back_compare_mask = regs.stencil_back_func_mask;
    v.line_width = LineWidthOf(regs);
    v.cull_enable = regs.gl_cull_test_enabled;
    v.cull_face = regs.gl_cull_face;
    v.depth_compare = regs.depth_test_func;
    v.front_face = regs.gl_front_face;
    v.front_face_flip = static_cast<u8>(regs.window_origin.flip_y != 0);
    v.stencil_front = StencilFaceOpOf(regs.stencil_front_op);
    v.stencil_back = StencilFaceOpOf(regs.stencil_back_op);
    v.depth_bounds_test = static_cast<u8>(regs.depth_bounds_enable != 0 && features.depth_bounds);
    v.depth_test = static_cast<u8>(regs.depth_test_enable != 0);
    v.depth_write = static_cast<u8>(regs.depth_write_enabled != 0);
    v.stencil_test = static_cast<u8>(regs.stencil_enable != 0);
    v.primitive_restart = static_cast<u8>(
        PrimitiveRestartOf(regs, topology, features.molten_vk, features.restart_topologies));
    v.rasterizer_discard = static_cast<u8>(regs.rasterize_enable == 0);
    v.depth_bias_enable = static_cast<u8>(DepthBiasEnableOf(regs, topology));
    v.depth_clamp = static_cast<u8>(DepthClampOf(regs));
    v.line_stipple = static_cast<u8>(regs.line_stipple_enable != 0);
    v.alpha_to_coverage = static_cast<u8>(regs.anti_alias_alpha_control.alpha_to_coverage != 0);
    v.alpha_to_one = static_cast<u8>(regs.anti_alias_alpha_control.alpha_to_one != 0);
    v.logic_op_enable = static_cast<u8>(EffectiveLogicOpEnable(regs, features.amd_logic_op));
    v.line_smooth = static_cast<u8>(regs.line_anti_alias_enable != 0 && features.smooth_lines);
    v.conservative = static_cast<u8>(regs.conservative_raster_enable != 0);
    v.logic_op = static_cast<u32>(regs.logic_op.op);
    for (size_t index = 0; index < MaxwellRegs::NumRenderTargets; ++index) {
        v.color_masks[index] = ColorMaskOf(regs, index);
        v.blend_enables[index] = static_cast<u8>(BlendEnableOf(regs, index));
        v.color_write_enables[index] = static_cast<u8>(v.color_masks[index] != 0);
    }
    v.iterated_blend_squash = static_cast<u8>(
        BlendEquationsOf(regs, features.squashed_iterated_blend, v.blend_equations));
    if (features.vertex_input_dynamic) {
        VertexInputOf(regs, features.max_vertex_attributes, features.max_vertex_bindings, v);
    }
    return state;
}

} // namespace VideoCommon::V2::Resolve
