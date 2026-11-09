// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/engines/maxwell_3d.h"
#include "video_core/v2/draw_model.h"

namespace VideoCommon::V2 {

namespace {

using Tegra::Engines::Maxwell3D;
using Regs = Maxwell3D::Regs;

enum Mode : u8 {
    General = 0,
    Instance = 1,
    InlineIndex = 2,
};

const Regs& View(const u32* regs) {
    return *reinterpret_cast<const Regs*>(regs);
}

} // Anonymous namespace

bool DrawModel::IsInstanceLoopMethod(u32 method) {
    return Maxwell3D::DrawManager::IsInstanceLoopMethod(method);
}

bool DrawModel::IsDrawMethod(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
        return true;
    default:
        return false;
    }
}

void DrawModel::UpdateTopology(const u32* regs) {
    const Regs& r = View(regs);
    if (r.primitive_topology_control != Regs::PrimitiveTopologyControl::UseSeparateState) {
        return;
    }
    switch (r.topology_override) {
    case Regs::PrimitiveTopologyOverride::None:
        break;
    case Regs::PrimitiveTopologyOverride::Points:
        state.topology = static_cast<u8>(Regs::PrimitiveTopology::Points);
        break;
    case Regs::PrimitiveTopologyOverride::Lines:
        state.topology = static_cast<u8>(Regs::PrimitiveTopology::Lines);
        break;
    case Regs::PrimitiveTopologyOverride::LineStrip:
        state.topology = static_cast<u8>(Regs::PrimitiveTopology::LineStrip);
        break;
    default:
        state.topology = static_cast<u8>(r.topology_override);
        break;
    }
}

void DrawModel::ProcessDraw(bool indexed, u32 instance_count, const u32* regs, DrawOutput& out) {
    UpdateTopology(regs);
    out.Push(Ir::PredictedDraw{
        .kind = indexed ? Ir::DrawKind::Indexed : Ir::DrawKind::Array,
        .topology = state.topology,
        .reserved = 0,
        .first = indexed ? state.index_first : state.vertex_first,
        .count = indexed ? state.index_count : state.vertex_count,
        .instances = instance_count,
        .base_vertex = state.base_index,
        .base_instance = state.base_instance,
    });
}

void DrawModel::DrawEnd(u32 instance_count, bool force_draw, const u32* regs, DrawOutput& out) {
    const Regs& r = View(regs);
    switch (state.mode) {
    case Instance:
        if (!force_draw) {
            break;
        }
        [[fallthrough]];
    case General:
        state.base_instance = r.global_base_instance_index;
        if (state.mode == Instance && state.drawn_instances != 0) {
            state.base_instance += state.drawn_instances + 1;
        }
        state.base_index = r.global_base_vertex_index;
        if (state.draw_indexed) {
            state.index_first = r.index_buffer.first;
            state.index_count = r.index_buffer.count;
            ProcessDraw(true, instance_count, regs, out);
        } else {
            state.vertex_first = r.vertex_buffer.first;
            state.vertex_count = r.vertex_buffer.count;
            ProcessDraw(false, instance_count, regs, out);
        }
        state.draw_indexed = false;
        break;
    case InlineIndex:
        state.base_instance = r.global_base_instance_index;
        state.base_index = r.global_base_vertex_index;
        state.index_first = r.index_buffer.first;
        state.index_count = state.inline_bytes / 4;
        ProcessDraw(true, instance_count, regs, out);
        state.inline_bytes = 0;
        break;
    }
}

void DrawModel::DrawDeferred(bool loop_continues, const u32* regs, DrawOutput& out) {
    if (state.mode != Instance || state.instance_count == 0) {
        if (!loop_continues) {
            state.drawn_instances = 0;
        }
        return;
    }
    const bool continued = state.drawn_instances != 0;
    DrawEnd(state.instance_count + (continued ? 0 : 1), true, regs, out);
    state.drawn_instances = loop_continues ? state.drawn_instances + state.instance_count : 0;
    state.instance_count = 0;
}

void DrawModel::OnStore(u32 method, const u32* regs, DrawOutput& out) {
    if (HasDeferredDraw() && !IsInstanceLoopMethod(method)) {
        DrawDeferred(true, regs, out);
    }
}

void DrawModel::OnMultiStart(u32 method, const u32* regs, DrawOutput& out) {
    OnStore(method, regs, out);
}

void DrawModel::OnExecute(u32 method, u32 argument, const u32* regs, DrawOutput& out) {
    const Regs& r = View(regs);
    const auto index_small = [&](u32 value) {
        const Regs::IndexBufferSmall small{value};
        state.base_instance = r.global_base_instance_index;
        state.base_index = r.global_base_vertex_index;
        state.index_first = small.first;
        state.index_count = small.count;
        state.topology = static_cast<u8>(small.topology.Value());
        ProcessDraw(true, 1, regs, out);
    };
    const auto array_instanced = [&](const Regs::VertexArray& array, bool subsequent) {
        state.topology = static_cast<u8>(array.topology.Value());
        state.vertex_first = array.start;
        state.vertex_count = array.count;
        if (!subsequent) {
            state.instance_count = 1;
            state.drawn_instances = 0;
        }
        state.base_instance = state.instance_count - 1;
        state.mode = Instance;
        ++state.instance_count;
        ProcessDraw(false, 1, regs, out);
    };
    switch (method) {
    case MAXWELL3D_REG_INDEX(clear_surface):
        out.Push(Ir::PredictedDraw{.kind = Ir::DrawKind::Clear, .first = 1});
        return;
    case MAXWELL3D_REG_INDEX(draw.begin):
        if (r.draw.instance_id == Regs::Draw::InstanceId::First) {
            DrawDeferred(false, regs, out);
            state.instance_count = 0;
            state.mode = General;
        } else if (r.draw.instance_id == Regs::Draw::InstanceId::Subsequent) {
            ++state.instance_count;
            state.mode = Instance;
        }
        state.topology = static_cast<u8>(r.draw.topology.Value());
        return;
    case MAXWELL3D_REG_INDEX(draw.end):
        DrawEnd(1, false, regs, out);
        return;
    case MAXWELL3D_REG_INDEX(index_buffer.count):
        state.draw_indexed = true;
        return;
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
        ++state.instance_count;
        index_small(argument);
        return;
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
        index_small(argument);
        return;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
        state.inline_bytes += 4;
        state.mode = InlineIndex;
        return;
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
        state.inline_bytes += 8;
        state.mode = InlineIndex;
        return;
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        state.inline_bytes += 16;
        state.mode = InlineIndex;
        return;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
        array_instanced(r.vertex_array_instance_first, false);
        return;
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
        array_instanced(r.vertex_array_instance_subsequent, true);
        return;
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
        out.Push(Ir::PredictedDraw{.kind = Ir::DrawKind::Texture});
        return;
    default:
        return;
    }
}

void DrawModel::OnInlineIndex(u32 method, u32 amount) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw_inline_index):
        state.inline_bytes += amount * 4;
        break;
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
        state.inline_bytes += amount * 8;
        break;
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        state.inline_bytes += amount * 16;
        break;
    default:
        return;
    }
    state.mode = InlineIndex;
}

void DrawModel::OnMacroEnd(const u32* regs, DrawOutput& out) {
    DrawDeferred(false, regs, out);
}

void DrawModel::OnIndirect(bool indexed, u32 topology, const u32* regs, DrawOutput& out) {
    state.topology = static_cast<u8>(topology);
    if (indexed) {
        state.index_first = 0;
    }
    UpdateTopology(regs);
    out.Push(Ir::PredictedDraw{
        .kind = indexed ? Ir::DrawKind::IndexedIndirect : Ir::DrawKind::Indirect,
        .topology = state.topology,
    });
}

} // namespace VideoCommon::V2
