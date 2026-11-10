// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include "common/assert.h"
#include "common/bit_util.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/query_cache/frame_trace.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/texture.h"
#include "video_core/texture_cache/image_info.h"
#include "video_core/texture_cache/util.h"
#include "video_core/v2/cbuf_ring.h"
#include "video_core/v2/cbuf_watch.h"
#include "video_core/v2/draw_model.h"
#include "video_core/v2/front_writes.h"
#include "video_core/v2/ir_exec.h"
#include "video_core/v2/macro_memo.h"
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/resolve/split_resolve.h"
#include "video_core/v2/s1_deltas.h"
#include "video_core/v2/shader_writes.h"
#include "video_core/v2/word_readback.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif

namespace Tegra::Engines {

/// First register id that is actually a Macro call.
constexpr u32 MacroRegistersStart = 0xE00;

Maxwell3D::Maxwell3D(MemoryManager& memory_manager_)
    : draw_manager()
    , memory_manager{memory_manager_}
#ifdef ARCHITECTURE_x86_64
    , macro_engine(bool(Settings::values.disable_macro_jit))
#else
    , macro_engine(true)
#endif
    , upload_state{memory_manager, regs.upload}
{
    dirty.flags.flip();
    InitializeRegisterDefaults();
    reports_fast_draw_breaks = true;
    is_maxwell3d = true;
    execution_mask.reset();
    for (size_t i = 0; i < execution_mask.size(); i++)
        execution_mask[i] = IsMethodExecutable(u32(i));
    VideoCommon::V2::RegSem::InitTable3D(execution_mask);
    v2_back_mode = VideoCommon::V2::Get(VideoCommon::V2::Switch::RegSem);
    v2_table = &VideoCommon::V2::RegSem::GetTable3D();
    fast_maxwell3d = !record_write_history &&
                     !trace_first_method_writes && !VideoCommon::Nx2Profile::Enabled();
}

Maxwell3D::~Maxwell3D() = default;

void Maxwell3D::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer_);
}

void Maxwell3D::InitializeRegisterDefaults() {
    // Initializes registers to their default values - what games expect them to be at boot. This is
    // for certain registers that may not be explicitly set by games.

    // Reset all registers to zero
    std::memset(&regs, 0, sizeof(regs));

    // Depth range near/far is not always set, but is expected to be the default 0.0f, 1.0f. This is
    // needed for ARMS.
    for (auto& viewport : regs.viewports) {
        viewport.depth_range_near = 0.0f;
        viewport.depth_range_far = 1.0f;
    }
    for (auto& viewport : regs.viewport_transform) {
        viewport.swizzle.x.Assign(Regs::ViewportSwizzle::PositiveX);
        viewport.swizzle.y.Assign(Regs::ViewportSwizzle::PositiveY);
        viewport.swizzle.z.Assign(Regs::ViewportSwizzle::PositiveZ);
        viewport.swizzle.w.Assign(Regs::ViewportSwizzle::PositiveW);
    }

    // Doom and Bomberman seems to use the uninitialized registers and just enable blend
    // so initialize blend registers with sane values
    regs.blend.color_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.color_source = Regs::Blend::Factor::One_D3D;
    regs.blend.color_dest = Regs::Blend::Factor::Zero_D3D;
    regs.blend.alpha_op = Regs::Blend::Equation::Add_D3D;
    regs.blend.alpha_source = Regs::Blend::Factor::One_D3D;
    regs.blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    for (auto& blend : regs.blend_per_target) {
        blend.color_op = Regs::Blend::Equation::Add_D3D;
        blend.color_source = Regs::Blend::Factor::One_D3D;
        blend.color_dest = Regs::Blend::Factor::Zero_D3D;
        blend.alpha_op = Regs::Blend::Equation::Add_D3D;
        blend.alpha_source = Regs::Blend::Factor::One_D3D;
        blend.alpha_dest = Regs::Blend::Factor::Zero_D3D;
    }
    regs.stencil_front_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_front_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_front_func_mask = 0xFFFFFFFF;
    regs.stencil_front_mask = 0xFFFFFFFF;
    regs.stencil_two_side_enable = 1;
    regs.stencil_back_op.fail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zfail = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.zpass = Regs::StencilOp::Op::Keep_D3D;
    regs.stencil_back_op.func = Regs::ComparisonOp::Always_GL;
    regs.stencil_back_func_mask = 0xFFFFFFFF;
    regs.stencil_back_mask = 0xFFFFFFFF;

    regs.depth_test_func = Regs::ComparisonOp::Always_GL;
    regs.gl_front_face = Regs::FrontFace::CounterClockWise;
    regs.gl_cull_face = Regs::CullFace::Back;

    // TODO(Rodrigo): Most games do not set a point size. I think this is a case of a
    // register carrying a default value. Assume it's OpenGL's default (1).
    regs.point_size = 1.0f;

    // TODO(bunnei): Some games do not initialize the color masks (e.g. Sonic Mania). Assuming a
    // default of enabled fixes rendering here.
    for (auto& color_mask : regs.color_mask) {
        color_mask.R.Assign(1);
        color_mask.G.Assign(1);
        color_mask.B.Assign(1);
        color_mask.A.Assign(1);
    }

    for (auto& format : regs.vertex_attrib_format) {
        format.constant.Assign(1);
    }

    // NVN games expect these values to be enabled at boot
    regs.rasterize_enable = 1;
    regs.color_target_mrt_enable = 1;
    regs.framebuffer_srgb = 1;
    regs.line_width_aliased = 1.0f;
    regs.line_width_smooth = 1.0f;
    regs.gl_front_face = Maxwell3D::Regs::FrontFace::ClockWise;
    regs.polygon_mode_back = Maxwell3D::Regs::PolygonMode::Fill;
    regs.polygon_mode_front = Maxwell3D::Regs::PolygonMode::Fill;

    shadow_state = regs;
}

bool Maxwell3D::IsMethodExecutable(u32 method) {
    if (method >= MacroRegistersStart) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(draw.begin):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(index_buffer.count):
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(index_buffer32_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer16_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer8_subsequent):
    case MAXWELL3D_REG_INDEX(index_buffer32_first):
    case MAXWELL3D_REG_INDEX(index_buffer16_first):
    case MAXWELL3D_REG_INDEX(index_buffer8_first):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_first):
    case MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent):
    case MAXWELL3D_REG_INDEX(draw_texture.src_y0):
    case RootTables::SelectMethod:
    case RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(topology_override):
    case MAXWELL3D_REG_INDEX(clear_surface):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMacro(Core::System& system, u32 method, const u32* base_start, u32 amount, bool is_last_call) {
    if (executing_macro == 0) {
        // A macro call must begin by writing the macro method's register, not its argument.
        ASSERT((method % 2) == 0 && "Can't start macro execution by writing to the ARGS register");
        executing_macro = method;
    }

    macro_segments.emplace_back(current_dma_segment, amount);
    current_macro_dirty |= current_dirty;
    current_dirty = false;

    // Nearly every call arrives whole in one method run (title 1 makes ~1,000,000 a second):
    // copy it to the stack instead of growing the parameter vector. Still a copy: the command
    // list is often guest memory read in place, which the guest may reuse while the macro runs
    // (and the macro memo stores the parameters after it ran). Only accurate GPU mode rewrites
    // the parameters in place (RefreshParameters), so it keeps the vector.
    // EDEN_NX2_MACRO_COPY=1: always use the vector (for comparisons).
    static const bool always_copy = std::getenv("EDEN_NX2_MACRO_COPY") != nullptr;
    if (is_last_call && macro_params.empty() && amount <= 64 && !always_copy &&
        !Settings::IsGPULevelHigh()) {
        std::array<u32, 64> parameters;
        std::copy_n(base_start, amount, parameters.begin());
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, std::span<const u32>{parameters.data(), amount});
        macro_segments.clear();
        current_macro_dirty = false;
        return;
    }
    macro_params.insert(macro_params.end(), base_start, base_start + amount);

    // Call the macro when there are no more parameters in the command buffer
    if (is_last_call) {
        ConsumeSink(system);
        CallMacroMethod(system, executing_macro, macro_params);
        macro_params.clear();
        macro_segments.clear();
        current_macro_dirty = false;
    }
}

void Maxwell3D::RefreshParametersImpl() {
    if (!Settings::IsGPULevelHigh()) {
        return;
    }
    size_t current_index = 0;
    for (auto& segment : macro_segments) {
        if (segment.first == 0) {
            current_index += segment.second;
            continue;
        }
        memory_manager.ReadBlock(segment.first, &macro_params[current_index],
                                 sizeof(u32) * segment.second);
        current_index += segment.second;
    }
}

u32 Maxwell3D::GetMaxCurrentVertices() {
    u32 num_vertices = 0;
    for (size_t index = 0; index < Regs::NumVertexArrays; ++index) {
        const auto& array = regs.vertex_streams[index];
        if (array.enable == 0) {
            continue;
        }
        const auto& attribute = regs.vertex_attrib_format[index];
        if (attribute.constant) {
            num_vertices = (std::max)(num_vertices, 1U);
            continue;
        }
        const auto& limit = regs.vertex_stream_limits[index];
        const GPUVAddr gpu_addr_begin = array.Address();
        const GPUVAddr gpu_addr_end = limit.Address() + 1;
        const u64 stream_size = GetNativeMacroEngine().native_turing
                                    ? regs.NativeVertexStreamSize(index)
                                    : gpu_addr_end - gpu_addr_begin;
        const u32 address_size = static_cast<u32>((std::min<u64>)(stream_size, (std::numeric_limits<u32>::max)()));
        num_vertices = (std::max)(
            num_vertices, address_size / (std::max)(attribute.SizeInBytes(), array.stride.Value()));
        break;
    }
    return num_vertices;
}

size_t Maxwell3D::EstimateIndexBufferSize() {
    GPUVAddr start_address = regs.index_buffer.StartAddress();
    GPUVAddr end_address = regs.index_buffer.EndAddress();
    auto const byte_size = regs.index_buffer.FormatSizeInBytes();
    auto const max_size = 1ull << (byte_size * CHAR_BIT);
    auto const upper_cap = GetMaxCurrentVertices() * 4 * byte_size;
    auto const lower_cap = std::min<size_t>(size_t(end_address - start_address), upper_cap);
    return std::min<size_t>(memory_manager.GetMemoryLayoutSize(start_address, byte_size * max_size) / byte_size, lower_cap);
}

u32 Maxwell3D::ProcessShadowRam(u32 method, u32 argument) {
    // Keep track of the register value in shadow_state when requested.
    auto const c = shadow_state.shadow_ram_control;
    if (c == Regs::ShadowRamControl::Track || c == Regs::ShadowRamControl::TrackWithFilter)
        return shadow_state.reg_array[method] = argument;
    else if (c == Regs::ShadowRamControl::Replay)
        return shadow_state.reg_array[method];
    return argument;
}

void Maxwell3D::ConsumeSinkImpl(Core::System& system) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    if (v2_back_mode != VideoCommon::V2::Mode::Off) [[unlikely]] {
        return ConsumeSinkV2();
    }
    const auto control = shadow_state.shadow_ram_control;
    // Sunk methods are state writes only (not in execution_mask): skip unchanged values, as
    // CallMethod does.
    const bool skip_same = SkipSameValueWrites();
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    method_sink.clear();
}

void Maxwell3D::ProcessDirtyRegisters(u32 method, u32 argument) {
    // The instances after the first are drawn together once their loop ends, and they must see
    // the state they were begun with: NVN's draw macros reset the base instance (and its copy in
    // c[0][0x644]) right after the loop, before the macro returns.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    if (VideoCommon::Nx2FastDraw::Enabled() && regs.reg_array[method] != argument &&
        !IsFastDrawParameter(method)) {
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Register, method);
    }
    if (VideoCommon::V2::Stats::Enabled()) [[unlikely]] {
        VideoCommon::V2::Stats::NoteWrite(v2_draw_tracker, method, regs.reg_array[method],
                                          argument);
    }
    if (VideoCommon::V2::TrackGroups()) [[unlikely]] {
        VideoCommon::V2::RegSem::NoteWrite3D(v2_groups, method, regs.reg_array[method], argument);
    }
    regs.reg_array[method] = argument;
    for (auto const& table : dirty.tables)
        dirty.flags[table[method]] = true;
}

bool Maxwell3D::IsFastDrawParameter(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.size):
    case MAXWELL3D_REG_INDEX(const_buffer.address_high):
    case MAXWELL3D_REG_INDEX(const_buffer.address_low):
    case MAXWELL3D_REG_INDEX(const_buffer.offset):
    case MAXWELL3D_REG_INDEX(global_base_vertex_index):
    case MAXWELL3D_REG_INDEX(global_base_instance_index):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.start_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_high):
    case MAXWELL3D_REG_INDEX(index_buffer.limit_addr_low):
    case MAXWELL3D_REG_INDEX(index_buffer.format):
        return true;
    default:
        return DrawManager::IsInstanceLoopMethod(method) || IsFastDrawDeltaMethod(method) ||
               IsFastDrawVertexBuffer(method);
    }
}

bool Maxwell3D::IsFastDrawVertexBuffer(u32 method) {
    constexpr u32 streams = MAXWELL3D_REG_INDEX(vertex_streams);
    constexpr u32 stream_words = sizeof(Regs::VertexStream) / sizeof(u32);
    if (method >= streams && method < streams + Regs::NumVertexArrays * stream_words) {
        const u32 word = (method - streams) % stream_words;
        return word == 1 || word == 2; // address_high, address_low
    }
    constexpr u32 sizes = 0x600 / sizeof(u32); // SET_VERTEX_STREAM_SIZE_A/B (NativeVertexStreamSize)
    constexpr u32 limits = MAXWELL3D_REG_INDEX(vertex_stream_limits);
    constexpr u32 limit_words = sizeof(Regs::VertexStreamLimit) / sizeof(u32);
    return (method >= sizes && method < sizes + Regs::NumVertexArrays * 2) ||
           (method >= limits && method < limits + Regs::NumVertexArrays * limit_words);
}

bool Maxwell3D::IsFastDrawDeltaMethod(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + Regs::NumCBData) {
        return true;
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return true;
    default:
        return false;
    }
}

void Maxwell3D::ProcessMethodCall(u32 method, u32 argument, u32 nonshadow_argument, bool is_last_call) {
    if (!DrawManager::IsInstanceLoopMethod(method) && !IsFastDrawDeltaMethod(method)) {
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Method, method);
    }
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (method) {
    case RootTables::SelectMethod:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case RootTables::LoadMethod:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case MAXWELL3D_REG_INDEX(wait_for_idle):
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    // With macros on the split GPU thread's front (FrontWrite), the front loads and binds them.
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case MAXWELL3D_REG_INDEX(falcon[4]):
        return ProcessFirmwareCall4();
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        return ProcessCBData(argument);
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
        return ProcessQueryGet();
    case MAXWELL3D_REG_INDEX(render_enable.mode):
        return ProcessQueryCondition();
    case MAXWELL3D_REG_INDEX(clear_report_value):
        return ProcessCounterReset();
    case MAXWELL3D_REG_INDEX(sync_info):
        return ProcessSyncPoint();
    case MAXWELL3D_REG_INDEX(launch_dma):
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() == Regs::LaunchDMA::Layout::Pitch);
    case MAXWELL3D_REG_INDEX(inline_data):
        upload_state.ProcessData(argument, is_last_call);
        return;
    case MAXWELL3D_REG_INDEX(fragment_barrier):
        return rasterizer->FragmentBarrier();
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        break;
    }
}

void Maxwell3D::CallMacroMethod(Core::System& system, u32 method, std::span<const u32> parameters) {
    // Reset the current macro.
    executing_macro = 0;

    // Lookup the macro offset
    const u32 entry =
        ((method - MacroRegistersStart) >> 1) % static_cast<u32>(GetNativeMacroPositions().size());

    // Execute the current macro.
    GetNativeMacroEngine().Execute(system, *this, GetNativeMacroPositions()[entry], parameters);
    draw_manager.DrawDeferred(*this);
}

void Maxwell3D::TraceFirstMethodWrite(u32 method, u32 argument) {
    static std::array<bool, Regs::NUM_REGS> seen{};
    if (method < seen.size() && !seen[method]) {
        seen[method] = true;
        LOG_INFO(HW_GPU, "3D method first write: {:#x} (byte {:#x}) = {:#x}", method, method * 4, argument);
    }
}

#if defined(_MSC_VER) && !defined(__clang__)
#define M3D_FORCE_INLINE __forceinline
#else
#define M3D_FORCE_INLINE inline __attribute__((always_inline))
#endif

M3D_FORCE_INLINE void Maxwell3D::TableWrite(u32 method, u32 method_argument, bool is_last_call,
                                            bool fast) {
    // EDEN_NX2_V2_REGSEM=1: the register update through the semantics table the front mirror
    // uses (shadow RAM, compare-on-write, the action per register), then the action's handler.
    namespace RegSem = VideoCommon::V2::RegSem;
    using RegSem::Action3D;
    const RegSem::Table3D& table = *v2_table;
    const u32 argument = RegSem::ApplyShadow3D(shadow_state.shadow_ram_control,
                                               shadow_state.reg_array.data(), method,
                                               method_argument);
    const Action3D action = table.actions[method];
    if (action == Action3D::Store) {
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    switch (action) {
    case Action3D::StoreExec:
        return;
    case Action3D::TopologyOverride:
        if (!fast) {
            // ProcessMethodCall: a method break, nothing for the draw manager to do.
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Method, method);
        }
        return;
    case Action3D::CbData:
        return ProcessCBData(argument);
    case Action3D::CbufBind0:
        return ProcessCBBind(0);
    case Action3D::CbufBind1:
        return ProcessCBBind(1);
    case Action3D::CbufBind2:
        return ProcessCBBind(2);
    case Action3D::CbufBind3:
        return ProcessCBBind(3);
    case Action3D::CbufBind4:
        return ProcessCBBind(4);
    case Action3D::DrawBegin:
        return draw_manager.DrawBegin(*this);
    case Action3D::DrawEnd:
        return draw_manager.DrawEnd(*this);
    default:
        return TableMethodCall(action, method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    if (!fast_maxwell3d || executing_macro != 0 || method >= MacroRegistersStart ||
        method == MAXWELL3D_REG_INDEX(shadow_ram_control)) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    if (perf_verify) {
        return VerifyCallMethod(system, method, method_argument, is_last_call);
    }
    if (v2_back_mode != VideoCommon::V2::Mode::Off) [[unlikely]] {
        if (v2_back_mode == VideoCommon::V2::Mode::On) [[likely]] {
            return TableWrite(method, method_argument, is_last_call, true);
        }
        return CallMethodV2(method, method_argument, is_last_call, true);
    }
    CallMethodFast(method, method_argument, is_last_call);
}

namespace {

// Executable methods the draw manager does nothing for (ProcessDirtyRegisters is all they do).
bool IsStoreOnlyExecutable(u32 method) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return true;
    default:
        return false;
    }
}

} // Anonymous namespace

// CallMethodSlow without the checks that cannot apply (CallMethod): no macro collects parameters,
// no write history, trace or profile, not a macro call or SET_SHADOW_RAM_CONTROL. The hot
// executable methods go straight to their handler.
void Maxwell3D::CallMethodFast(u32 method, u32 method_argument, bool is_last_call) {
    const u32 argument = ProcessShadowRam(method, method_argument);
    if (!execution_mask[method]) {
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method - cb_data < 16) {
        return ProcessCBData(argument);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        return draw_manager.DrawBegin(*this);
    case MAXWELL3D_REG_INDEX(draw.end):
        return draw_manager.DrawEnd(*this);
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
    case MAXWELL3D_REG_INDEX(index_buffer.first):
    case MAXWELL3D_REG_INDEX(topology_override):
        return;
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
        return ProcessCBBind(0);
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
        return ProcessCBBind(1);
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
        return ProcessCBBind(2);
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
        return ProcessCBBind(3);
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
        return ProcessCBBind(4);
    default:
        return ProcessMethodCall(method, argument, method_argument, is_last_call);
    }
}

void Maxwell3D::VerifyCallMethod(Core::System& system, u32 method, u32 method_argument,
                                 bool is_last_call) {
    // Only register writes can run twice: handlers draw, upload and bind, and a deferred draw
    // ProcessDirtyRegisters may issue is a draw too.
    if ((execution_mask[method] && !IsStoreOnlyExecutable(method)) ||
        draw_manager.HasDeferredDraw()) {
        return CallMethodSlow(system, method, method_argument, is_last_call);
    }
    const u32 reg = regs.reg_array[method];
    const u32 shadow = shadow_state.reg_array[method];
    const DirtyState::Flags flags = dirty.flags;
    CallMethodFast(method, method_argument, is_last_call);
    const u32 fast_reg = regs.reg_array[method];
    const u32 fast_shadow = shadow_state.reg_array[method];
    const DirtyState::Flags fast_flags = dirty.flags;
    regs.reg_array[method] = reg;
    shadow_state.reg_array[method] = shadow;
    dirty.flags = flags;
    CallMethodSlow(system, method, method_argument, is_last_call);
    if (fast_reg == regs.reg_array[method] && fast_shadow == shadow_state.reg_array[method] &&
        fast_flags == dirty.flags) {
        return;
    }
    static std::atomic<u32> reports{};
    if (reports.fetch_add(1, std::memory_order_relaxed) >= 64) {
        return;
    }
    std::string flag_differences;
    const DirtyState::Flags differing = fast_flags ^ dirty.flags;
    for (size_t i = 0; i < differing.size(); ++i) {
        if (differing[i]) {
            flag_differences += fmt::format(" {}={}", i, fast_flags[i] ? 1 : 0);
        }
    }
    LOG_ERROR(HW_GPU,
              "Nx2Perf method: {:#x} = {:#x}: register {:#x} (old {:#x}), shadow {:#x} (old {:#x}), "
              "fast dirty flags differ:{}",
              method, method_argument, fast_reg, regs.reg_array[method], fast_shadow,
              shadow_state.reg_array[method], flag_differences);
}

void Maxwell3D::CallConstBufferWords(Core::System& system, const u32* methods,
                                     const u32* arguments, u32 count) {
    const u32 offset = regs.const_buffer.offset;
    const u64 bytes = u64{count} * sizeof(u32);
    const GPUVAddr address = regs.const_buffer.Address();
    // Keep per-word translation across device pages and per-word clipping at buffer limits.
    if (!fast_maxwell3d || executing_macro != 0 || address == 0 ||
        offset > regs.const_buffer.size || bytes > regs.const_buffer.size - offset ||
        bytes > Core::DEVICE_PAGESIZE - ((address + offset) & Core::DEVICE_PAGEMASK)) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        return;
    }
    std::array<u32, 64> values;
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    for (u32 i = 0; i < count; ++i) {
        values[i] = replay ? shadow_state.reg_array[methods[i]] : arguments[i];
    }
    if (perf_verify) {
        for (u32 i = 0; i < count; ++i) {
            CallMethod(system, methods[i], arguments[i], true);
        }
        std::array<u32, 64> written;
        const bool mapped = memory_manager.IsFullyMappedRange(address + offset, bytes);
        if (mapped) {
            memory_manager.ReadBlockUnsafe(address + offset, written.data(), bytes);
        }
        if (regs.const_buffer.offset != offset + bytes ||
            (mapped && std::memcmp(written.data(), values.data(), bytes) != 0)) {
            static std::atomic<u32> reports{};
            if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU,
                          "Nx2Perf method: {} constant buffer words at {:#x}+{:#x}: offset {:#x}, "
                          "expected {:#x}, memory {}",
                          count, address, offset, regs.const_buffer.offset, offset + bytes,
                          mapped && std::memcmp(written.data(), values.data(), bytes) != 0
                              ? "differs"
                              : "matches");
            }
        }
        return;
    }
    // As CallMethodFast for each word: the first may issue a deferred draw, before any upload.
    for (u32 i = 0; i < count; ++i) {
        ProcessDirtyRegisters(methods[i], ProcessShadowRam(methods[i], arguments[i]));
    }
    if (VideoCommon::Nx2FastDraw::Enabled()) {
        NoteFastDrawConstantData(address + offset, address + offset + bytes);
    }
    memory_manager.WriteWordsCached(address + offset, values.data(), count);
    regs.const_buffer.offset += static_cast<u32>(bytes);
    if (v2_back_mode == VideoCommon::V2::Mode::Verify) [[unlikely]] {
        // The words as CbData table writes (the mirror was current before the batch).
        auto& mirror = BackMirror();
        const auto& table = VideoCommon::V2::RegSem::GetTable3D();
        for (u32 i = 0; i < count; ++i) {
            VideoCommon::V2::RegSem::DryWrite3D(mirror, table, methods[i], arguments[i],
                                                SkipSameValueWrites());
        }
        VideoCommon::V2::RegSem::CompareBackWrite3D(mirror, table, methods[count - 1],
                                                    arguments[count - 1], regs.reg_array.data(),
                                                    shadow_state.reg_array.data());
    }
}

void Maxwell3D::CallMethodSlow(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    // Every word of a macro register run is a parameter of the macro it started, whichever macro
    // register it writes (as on the split front, FrontCallMethod). Only a write to an ordinary
    // register before the macro has all its parameters is an error.
    if (executing_macro != 0 && method < MacroRegistersStart) {
        DumpRecentWrites(method);
    }
    if (executing_macro != 0) {
        ASSERT(method >= MacroRegistersStart);
    }
    RecordWrite(method, method_argument, 1, is_last_call ? 1 : 2);

    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, &method_argument, 1, is_last_call);
        return;
    }
    if (trace_first_method_writes) [[unlikely]] {
        TraceFirstMethodWrite(method, method_argument);
    }

    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register, increase the size of the Regs structure");
    if (VideoCommon::Nx2Profile::Enabled()) [[unlikely]] {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::MethodWrites, 1);
        const bool same = regs.reg_array[method] == method_argument;
        if (same) {
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::SameValueWrites, 1);
        }
        VideoCommon::Nx2Profile::CountMethod(method, same);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && front_macros) [[unlikely]] {
        back_control_writes[back_control_count++ % back_control_writes.size()] = {
            method_argument, static_cast<u32>(shadow_state.shadow_ram_control), is_last_call ? 1U : 0U};
    }
    if (v2_back_mode != VideoCommon::V2::Mode::Off) [[unlikely]] {
        return CallMethodV2(method, method_argument, is_last_call, false);
    }
    CallMethodSlowTail(method, method_argument, is_last_call);
}

void Maxwell3D::CallMethodSlowTail(u32 method, u32 method_argument, bool is_last_call) {
    const u32 argument = ProcessShadowRam(method, method_argument);
    // A state register written with the value it holds changes nothing: no dirty flags (they
    // would make the next draw re-check that state), no method processing. Title 1 makes ~6
    // million 3D register writes a second, ~60% of them same-value, most from macro replays.
    // Methods with side effects (execution_mask: draws, clears, constant buffer data and
    // binds, semaphores, ...) always run. EDEN_NX2_SKIP_SAME=0 turns it off.
    if (!execution_mask[method]) {
        // ProcessMethodCall (and the draw manager behind it) acts only on executable methods.
        if (!SkipSameValueWrites() || regs.reg_array[method] != argument) {
            ProcessDirtyRegisters(method, argument);
        }
        return;
    }
    ProcessDirtyRegisters(method, argument);
    ProcessMethodCall(method, argument, method_argument, is_last_call);
}

VideoCommon::V2::RegSem::BackMirror3D& Maxwell3D::BackMirror() {
    if (!v2_back_mirror) [[unlikely]] {
        v2_back_mirror = std::make_unique<VideoCommon::V2::RegSem::BackMirror3D>();
        VideoCommon::V2::RegSem::ResyncBackMirror3D(*v2_back_mirror, regs.reg_array.data(),
                                                    shadow_state.reg_array.data());
    }
    return *v2_back_mirror;
}

void Maxwell3D::CallMethodV2(u32 method, u32 method_argument, bool is_last_call, bool fast) {
    namespace RegSem = VideoCommon::V2::RegSem;
    if (v2_back_mode == VideoCommon::V2::Mode::On) {
        return TableWrite(method, method_argument, is_last_call, fast);
    }
    // Verify: the engine's own path decides; the table-driven update function runs on a copy of
    // the registers and is compared.
    auto& mirror = BackMirror();
    if (fast) {
        CallMethodFast(method, method_argument, is_last_call);
    } else {
        CallMethodSlowTail(method, method_argument, is_last_call);
    }
    const auto& table = RegSem::GetTable3D();
    RegSem::DryWrite3D(mirror, table, method, method_argument, SkipSameValueWrites());
    RegSem::CompareBackWrite3D(mirror, table, method, method_argument, regs.reg_array.data(),
                               shadow_state.reg_array.data());
}

void Maxwell3D::VerifyMultiV2(VideoCommon::V2::RegSem::BackMirror3D& mirror, u32 method,
                              const u32* base_start, u32 amount) {
    namespace RegSem = VideoCommon::V2::RegSem;
    const auto& table = RegSem::GetTable3D();
    const u16 flags = table.flags[method];
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    // Runs of other methods went through CallMethod word by word (verified there).
    if ((flags & (RegSem::CbData | RegSem::InlineData)) == 0 &&
        ((flags & RegSem::InlineIndex) == 0 || replay)) {
        return;
    }
    RegSem::DryWriteMulti3D(mirror, table, method, base_start, amount, SkipSameValueWrites());
    RegSem::CompareBackWrite3D(mirror, table, method, amount != 0 ? base_start[amount - 1] : 0,
                               regs.reg_array.data(), shadow_state.reg_array.data());
}

void Maxwell3D::ConsumeSinkV2() {
    namespace RegSem = VideoCommon::V2::RegSem;
    const auto& table = RegSem::GetTable3D();
    const bool skip_same = SkipSameValueWrites();
    if (v2_back_mode == VideoCommon::V2::Mode::On) {
        // Sunk methods are state writes (Action3D::Store): the table path's store.
        for (const auto [method, value] : method_sink) {
            const u32 argument = RegSem::ApplyShadow3D(shadow_state.shadow_ram_control,
                                                       shadow_state.reg_array.data(), method,
                                                       value);
            if (!skip_same || regs.reg_array[method] != argument) {
                ProcessDirtyRegisters(method, argument);
            }
        }
        method_sink.clear();
        return;
    }
    // Verify: the engine's loop, then the same writes on the mirror.
    auto& mirror = BackMirror();
    const auto control = shadow_state.shadow_ram_control;
    const auto process = [&](u32 method, u32 value) {
        if (!skip_same || regs.reg_array[method] != value) {
            ProcessDirtyRegisters(method, value);
        }
    };
    if (control == Regs::ShadowRamControl::Track || control == Regs::ShadowRamControl::TrackWithFilter) {
        for (auto [method, value] : method_sink) {
            shadow_state.reg_array[method] = value;
            process(method, value);
        }
    } else if (control == Regs::ShadowRamControl::Replay) {
        for (auto [method, value] : method_sink)
            process(method, shadow_state.reg_array[method]);
    } else {
        for (auto [method, value] : method_sink)
            process(method, value);
    }
    if (!method_sink.empty()) {
        for (const auto [method, value] : method_sink) {
            RegSem::DryWrite3D(mirror, table, method, value, skip_same);
        }
        const auto [last_method, last_value] = method_sink.back();
        RegSem::CompareBackWrite3D(mirror, table, last_method, last_value, regs.reg_array.data(),
                                   shadow_state.reg_array.data());
    }
    method_sink.clear();
}

void Maxwell3D::ExecuteS1Run(const u64* it, const u64* const end) {
    // The front ran the register semantics: store, mark the v1 dirty flags, and run only what
    // has side effects. No shadow RAM, no compare, no sink, no draw state machine, no hashing.
    namespace S1 = VideoCommon::V2::S1;
    const auto& tables = dirty.tables;
    const auto store = [&](u32 index, u32 value) {
        regs.reg_array[index] = value;
        dirty.flags[tables[0][index]] = true;
        dirty.flags[tables[1][index]] = true;
    };
    while (it < end) {
        const u64 word = *it;
        switch (S1::TagOf(word)) {
        case S1::Tag::Delta:
            if ((word & S1::DeltaQuiet) == 0) [[likely]] {
                store(S1::IndexOf(word), S1::ValueOf(word));
            } else {
                regs.reg_array[S1::IndexOf(word)] = S1::ValueOf(word);
            }
            ++it;
            break;
        case S1::Tag::Exec: {
            const u32 index = S1::IndexOf(word);
            const u32 value = S1::ValueOf(word);
            store(index, value);
            TableMethodCall(VideoCommon::V2::RegSem::GetTable3D().actions[index], index, value,
                            value, (word & S1::IsLast) != 0);
            ++it;
            break;
        }
        case S1::Tag::Cbuf: {
            const u32 count = S1::CbufCount(word);
            const u32* const words = reinterpret_cast<const u32*>(it + 2);
            if ((word & S1::CbufStore) != 0) {
                constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
                for (u32 k = 0; k < count; ++k) {
                    store(S1::CbufMethod(word, cb_data, k), words[k]);
                }
            }
            S1CbufWrite(it[1], S1::ValueOf(word), words, count);
            it += 2 + (count + 1) / 2;
            break;
        }
        case S1::Tag::Draw:
            draw_manager.DrawPredicted(*this, S1::DrawOf(it), static_cast<u8>(word >> 8));
            it += S1::DrawWords;
            break;
        case S1::Tag::CbufVersion: {
            // The window's decode-time version, by device address for the uniform binds.
            auto& table = VideoCommon::V2::CbufRing::Get().table;
            if (const auto device = memory_manager.GpuToCpuAddress(it[1])) {
                if (it[2] == VideoCommon::V2::CbufRing::NoPosition) {
                    table.Drop(*device);
                } else {
                    table.Set(*device, S1::ValueOf(word), it[2]);
                }
            }
            it += S1::CbufVersionWords;
            break;
        }
        case S1::Tag::Inline: {
            const u32 count = S1::ValueOf(word);
            draw_manager.SetInlineIndexBuffer(*this, S1::IndexOf(word),
                                              reinterpret_cast<const u32*>(it + 1), count);
            it += 1 + (count + 1) / 2;
            break;
        }
        case S1::Tag::Groups: {
            // The front's groups and hashes: what PIPEKEY, CONTDRAW and DRAWCACHE read.
            const u32 groups = S1::GroupsDirty(word);
            const u32 hashed = groups & VideoCommon::V2::HashedGroups;
            ++it;
            for (u32 bits = hashed; bits != 0; bits &= bits - 1) {
                v2_groups.hashes[std::countr_zero(bits)] = *it++;
            }
            v2_groups.dirty |= groups;
            if (hashed == VideoCommon::V2::HashedGroups) {
                v2_groups.valid = true;
            }
            if ((word & S1::GroupsRegisterBreak) != 0) {
                VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Register);
            }
            if ((word & S1::GroupsMethodBreak) != 0) {
                // Clears and texture draws between draws change no state a draw resolves
                // from: with CONTDRAW the groups decide (a register break), not an event.
                VideoCommon::Nx2FastDraw::Break(
                    VideoCommon::V2::On(VideoCommon::V2::Switch::ContDraw)
                        ? VideoCommon::Nx2FastDraw::Reason::Register
                        : VideoCommon::Nx2FastDraw::Reason::Method,
                    S1::ValueOf(word));
            }
            break;
        }
        default:
            // Not a classified entry: the run is corrupt. Skip the word.
            ++it;
            break;
        }
    }
}

void Maxwell3D::S1CbufWrite(GPUVAddr buffer, u32 offset, const u32* words, u32 count) {
    // ProcessCBMultiData with the front's address and offset (the registers hold the same).
    const size_t copy_size = size_t{count} * sizeof(u32);
    const size_t room = offset < regs.const_buffer.size ? regs.const_buffer.size - offset : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address = buffer + offset;
        if (VideoCommon::Nx2FastDraw::Enabled()) {
            NoteFastDrawConstantData(address, address + write_size);
        }
        memory_manager.WriteBlockCached(address, words, write_size);
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      offset, regs.const_buffer.size, copy_size, write_size);
        }
    }
    regs.const_buffer.offset = offset + static_cast<u32>(copy_size);
}

struct Maxwell3D::S1CheckState {
    std::array<u32, Regs::NUM_REGS> regs{};
    bool synced{};
    u64 entries{};
    u64 mismatches{};
};

std::pair<u64, u64> Maxwell3D::S1CheckCounts() const {
    return s1_check ? std::pair<u64, u64>{s1_check->entries, s1_check->mismatches}
                    : std::pair<u64, u64>{0, 0};
}

size_t Maxwell3D::S1CheckEntry(const u64* entry) {
    namespace S1 = VideoCommon::V2::S1;
    if (!s1_check) {
        s1_check = std::make_unique<S1CheckState>();
    }
    S1CheckState& check = *s1_check;
    // The engine's registers with the state writes still in its method sink applied.
    const bool replay = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay;
    const auto expected = [&](u32 index) {
        for (auto it = method_sink.rbegin(); it != method_sink.rend(); ++it) {
            if (it->first == index) {
                return replay ? shadow_state.reg_array[index] : it->second;
            }
        }
        return regs.reg_array[index];
    };
    if (!check.synced) {
        for (u32 index = 0; index < Regs::NUM_REGS; ++index) {
            check.regs[index] = expected(index);
        }
        check.synced = true;
    }
    static std::atomic<u32> reports{};
    const auto report = [&](u32 index) {
        ++check.mismatches;
        check.synced = false;
        if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_WARNING(HW_GPU,
                        "V2 S1 check: entry {:#x} register {:#x}: deltas {:#x}, executed {:#x} "
                        "({} entries, {} mismatches)",
                        entry[0], index, check.regs[index], expected(index), check.entries,
                        check.mismatches);
        }
    };
    ++check.entries;
    std::array<u32, 20> touched;
    u32 num_touched = 0;
    S1::ApplyRegisters(entry, MAXWELL3D_REG_INDEX(const_buffer.buffer),
                       MAXWELL3D_REG_INDEX(const_buffer.offset), [&](u32 index, u32 value) {
                           check.regs[index] = value;
                           if (num_touched < touched.size()) {
                               touched[num_touched++] = index;
                           }
                       });
    for (u32 i = 0; i < num_touched; ++i) {
        if (check.regs[touched[i]] != expected(touched[i])) {
            report(touched[i]);
            return S1::EntryWords(entry);
        }
    }
    if (S1::TagOf(entry[0]) == S1::Tag::Cbuf && entry[1] != regs.const_buffer.Address()) {
        ++check.mismatches;
        if (reports.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_WARNING(HW_GPU, "V2 S1 check: constant buffer upload to {:#x}, executed at {:#x}",
                        entry[1], regs.const_buffer.Address());
        }
    }
    if ((check.entries & 4095) == 0) {
        for (u32 index = 0; index < Regs::NUM_REGS; ++index) {
            if (check.regs[index] != expected(index)) {
                report(index);
                break;
            }
        }
        if ((check.entries & 0xfffff) == 0) {
            LOG_INFO(HW_GPU, "V2 S1 check: {} entries, {} mismatches", check.entries,
                     check.mismatches);
        }
    }
    return S1::EntryWords(entry);
}

void Maxwell3D::TableMethodCall(VideoCommon::V2::RegSem::Action3D action, u32 method,
                                u32 argument, u32 nonshadow_argument, bool is_last_call) {
    // ProcessMethodCall with the action from the table instead of a switch on the method.
    using VideoCommon::V2::RegSem::Action3D;
    const auto& table = VideoCommon::V2::RegSem::GetTable3D();
    if ((table.flags[method] & VideoCommon::V2::RegSem::FastDrawQuiet) == 0) {
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Method, method);
    }
    if (GetNativeMacroEngine().native_turing && method == MAXWELL3D_REG_INDEX(clear_surface)) {
        LOG_DEBUG(HW_GPU, "Native GPU surface clear: value={:#x}", argument);
    }
    switch (action) {
    case Action3D::RootSelect:
        root_tables.selected = argument & 7;
        root_tables.offset = (argument >> 8) & 0xff;
        return;
    case Action3D::RootLoad:
        if (root_tables.offset < 256 && root_tables.offset % 4 == 0) {
            root_tables.tables[root_tables.selected][root_tables.offset / 4] = argument;
            root_tables.loaded = true;
        }
        root_tables.offset += 4;
        return;
    case Action3D::WaitForIdle:
        return rasterizer->WaitForIdle();
    case Action3D::ShadowControl:
        shadow_state.shadow_ram_control = static_cast<Regs::ShadowRamControl>(nonshadow_argument);
        return;
    case Action3D::MacroCodePointer:
        if (front_macros) return;
        return GetNativeMacroEngine().ClearCode(regs.load_mme.instruction_ptr);
    case Action3D::MacroCode:
        if (front_macros) return;
        return GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr, argument);
    case Action3D::MacroBind:
        if (front_macros) {
            ++regs.load_mme.start_address_ptr;
            return;
        }
        return ProcessMacroBind(argument);
    case Action3D::Falcon4:
        return ProcessFirmwareCall4();
    case Action3D::CbData:
        return ProcessCBData(argument);
    case Action3D::CbufBind0:
        return ProcessCBBind(0);
    case Action3D::CbufBind1:
        return ProcessCBBind(1);
    case Action3D::CbufBind2:
        return ProcessCBBind(2);
    case Action3D::CbufBind3:
        return ProcessCBBind(3);
    case Action3D::CbufBind4:
        return ProcessCBBind(4);
    case Action3D::ReportSemaphore:
        return ProcessQueryGet();
    case Action3D::RenderEnable:
        return ProcessQueryCondition();
    case Action3D::ClearReportValue:
        return ProcessCounterReset();
    case Action3D::SyncInfo:
        return ProcessSyncPoint();
    case Action3D::LaunchDma:
        return upload_state.ProcessExec(regs.launch_dma.memory_layout.Value() ==
                                        Regs::LaunchDMA::Layout::Pitch);
    case Action3D::InlineData:
        upload_state.ProcessData(argument, is_last_call);
        return;
    case Action3D::FragmentBarrier:
        return rasterizer->FragmentBarrier();
    case Action3D::InvalidateTextureDataCache:
        rasterizer->InvalidateGPUCache();
        return rasterizer->WaitForIdle();
    case Action3D::TiledCacheBarrier:
        return rasterizer->TiledCacheBarrier();
    default:
        draw_manager.ProcessMethodCall(*this, method, argument);
        return;
    }
}

void Maxwell3D::DumpRecentWrites(u32 method) {
    static u32 dumps{};
    if (dumps++ >= 4) {
        return;
    }
    LOG_ERROR(HW_GPU, "Write to {:#x} while macro {:#x} collects parameters ({} so far); last writes:",
              method, executing_macro, macro_params.size());
    if (!record_write_history) {
        LOG_ERROR(HW_GPU, "Register write history disabled; EDEN_NX2_3D_HISTORY=1 enables it");
        return;
    }
    for (size_t i = 0; i < recent_writes.size(); ++i) {
        const auto& w = recent_writes[(recent_write_index + i) % recent_writes.size()];
        if (w.amount != 0) {
            LOG_ERROR(HW_GPU, "  method {:#x} arg {:#x} amount {} pending {}", w.method, w.argument,
                      w.amount, w.pending);
        }
    }
}

void Maxwell3D::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    RecordWrite(method, amount ? base_start[0] : 0, amount, methods_pending);
    // Methods after 0xE00 are special, they're actually triggers for some microcode that was
    // uploaded to the GPU during initialization.
    if (method >= MacroRegistersStart) {
        ProcessMacro(system, method, base_start, amount, amount == methods_pending);
        return;
    }
    // EDEN_NX2_V2_REGSEM=verify: the runs below that don't go through CallMethod per word.
    VideoCommon::V2::RegSem::BackMirror3D* const v2_mirror =
        v2_back_mode == VideoCommon::V2::Mode::Verify ? &BackMirror() : nullptr;
    SCOPE_EXIT {
        if (v2_mirror != nullptr) [[unlikely]] {
            VerifyMultiV2(*v2_mirror, method, base_start, amount);
        }
    };
    // As in ProcessDirtyRegisters: the paths below write without it.
    if (draw_manager.HasDeferredDraw() && !DrawManager::IsInstanceLoopMethod(method)) [[unlikely]] {
        draw_manager.DrawDeferred(*this, true);
    }
    switch (method) {
    case MAXWELL3D_REG_INDEX(const_buffer.buffer):
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 1:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 2:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 3:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 4:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 5:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 6:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 7:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 8:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 9:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 10:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 11:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 12:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 13:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 14:
    case MAXWELL3D_REG_INDEX(const_buffer.buffer) + 15:
        ProcessCBMultiData(base_start, amount);
        break;
    case MAXWELL3D_REG_INDEX(inline_data): {
        ASSERT(methods_pending == amount);
        VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::InlineData);
        upload_state.ProcessData(base_start, amount);
        return;
    }
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        if (shadow_state.shadow_ram_control != Regs::ShadowRamControl::Replay) {
            ProcessInlineIndexMultiData(method, base_start, amount);
            break;
        }
        [[fallthrough]];
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

void Maxwell3D::ProcessMacroUpload(u32 data) {
    GetNativeMacroEngine().AddCode(regs.load_mme.instruction_ptr++, data);
}

void Maxwell3D::ProcessMacroBind(u32 data) {
    const auto positions = GetNativeMacroPositions();
    const u32 index = regs.load_mme.start_address_ptr++;
    if (index < positions.size()) {
        positions[index] = data;
    }
}

void Maxwell3D::ProcessFirmwareCall4() {
    LOG_DEBUG(HW_GPU, "(STUBBED) called");

    // Firmware call 4 is a blob that changes some registers depending on its parameters.
    // These registers don't affect emulation and so are stubbed by setting 0xd00 to 1.
    regs.shadow_scratch[0] = 1;
}

void Maxwell3D::StampQueryResult(Core::System& system, u64 payload, bool long_query) {
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    if (long_query) {
        memory_manager.Write<u64>(sequence_address + sizeof(u64), system.GPU().GetTicks());
        memory_manager.Write<u64>(sequence_address, payload);
    } else {
        memory_manager.Write<u32>(sequence_address, static_cast<u32>(payload));
    }
    if (const u64 trace = VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::Write)) {
        LOG_INFO(HW_GPU,
                 "DKB_QUERY write n={} path=stamp as={} gpu={:#x} bytes={} value={:#x}",
                 trace, memory_manager.GetID(), sequence_address, long_query ? 8 : 4,
                 long_query ? payload : static_cast<u32>(payload));
    }
}

void Maxwell3D::ProcessQueryGet() {
    VideoCommon::QueryPropertiesFlags flags{};
    if (regs.report_semaphore.query.short_query == 0) {
        flags |= VideoCommon::QueryPropertiesFlags::HasTimeout;
    }
    const GPUVAddr sequence_address{regs.report_semaphore.Address()};
    const VideoCommon::QueryType query_type =
        static_cast<VideoCommon::QueryType>(regs.report_semaphore.query.report.Value());
    const u32 payload = regs.report_semaphore.payload;
    const u32 subreport = regs.report_semaphore.query.sub_report;
    switch (regs.report_semaphore.query.operation) {
    case Regs::ReportSemaphore::Operation::Release:
        if (regs.report_semaphore.query.short_query != 0) {
            flags |= VideoCommon::QueryPropertiesFlags::IsAFence;
        }
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Acquire:
        // TODO(Blinkhawk): Under this operation, the GPU waits for the CPU to write a value that
        // matches the current payload.
        UNIMPLEMENTED_MSG("Unimplemented query operation ACQUIRE");
        break;
    case Regs::ReportSemaphore::Operation::ReportOnly:
        rasterizer->Query(sequence_address, query_type, flags, payload, subreport);
        break;
    case Regs::ReportSemaphore::Operation::Trap:
        UNIMPLEMENTED_MSG("Unimplemented query operation TRAP");
        break;
    default:
        UNIMPLEMENTED_MSG("Unknown query operation");
        break;
    }
}

void Maxwell3D::ProcessQueryCondition() {
    constexpr u32 mode_index = MAXWELL3D_REG_INDEX(render_enable.mode);
    if (regs.reg_array[mode_index] == ~0U) [[unlikely]] {
        // MACROHLE: a Replay of the shadow copy the front doesn't know (FrontHleRenderEnable's
        // ShadowUnknown, sent as S1 deltas are, post-shadow): the mode this back evaluated last
        // under tracking is that copy. The front never waits for it (F3F5b-6).
        regs.reg_array[mode_index] = v2_hle_tracked_mode;
    }
    if (rasterizer->AccelerateConditionalRendering()) {
        execute_on = true;
        return;
    }
    const GPUVAddr condition_address{regs.render_enable.Address()};
    Regs::ReportSemaphore::Compare cmp{};
    bool operands_read = false;
    switch (regs.render_enable_override) {
    case Regs::RenderEnable::Override::AlwaysRender:
        execute_on = true;
        break;
    case Regs::RenderEnable::Override::NeverRender:
        execute_on = false;
        break;
    case Regs::RenderEnable::Override::UseRenderEnable: {
        switch (regs.render_enable.mode) {
        case Regs::RenderEnable::Mode::True: {
            execute_on = true;
            break;
        }
        case Regs::RenderEnable::Mode::False: {
            execute_on = false;
            break;
        }
        case Regs::RenderEnable::Mode::Conditional: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != 0U && cmp.initial_mode != 0U;
            break;
        }
        case Regs::RenderEnable::Mode::IfEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence == cmp.current_sequence &&
                         cmp.initial_mode == cmp.current_mode;
            break;
        }
        case Regs::RenderEnable::Mode::IfNotEqual: {
            memory_manager.ReadBlock(condition_address, &cmp, sizeof(cmp));
            operands_read = true;
            execute_on = cmp.initial_sequence != cmp.current_sequence ||
                         cmp.initial_mode != cmp.current_mode;
            break;
        }
        default: {
            UNIMPLEMENTED_MSG("Uninplemented Condition Mode!");
            execute_on = true;
            break;
        }
        }
        break;
    }
    }
    if (const u64 trace =
            VideoCommon::QueryTrace::Sample(VideoCommon::QueryTrace::Event::CpuCondition)) {
        LOG_INFO(HW_GPU,
                 "DKB_CONDITION cpu n={} as={} gpu={:#x} mode={} override={} execute={} "
                 "operands_read={} initial={:#x}/{:#x} current={:#x}/{:#x}",
                 trace, memory_manager.GetID(), condition_address,
                 static_cast<u32>(regs.render_enable.mode),
                 static_cast<u32>(regs.render_enable_override), execute_on, operands_read,
                 cmp.initial_sequence, cmp.initial_mode, cmp.current_sequence, cmp.current_mode);
    }
}

void Maxwell3D::ProcessCounterReset() {
    const auto query_type = [clear_report = regs.clear_report_value]() {
        switch (clear_report) {
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::ZPassPixelCount:
            return VideoCommon::QueryType::ZPassPixelCount64;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::StreamingPrimitivesSucceeded:
            return VideoCommon::QueryType::StreamingPrimitivesSucceeded;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::PrimitivesGenerated:
            return VideoCommon::QueryType::PrimitivesGenerated;
        case Tegra::Engines::Maxwell3D::Regs::ClearReport::VtgPrimitivesOut:
            return VideoCommon::QueryType::VtgPrimitivesOut;
        default:
            LOG_DEBUG(HW_GPU, "Unimplemented counter reset={}", clear_report);
            return VideoCommon::QueryType::Payload;
        }
    }();
    rasterizer->ResetCounter(query_type);
}

void Maxwell3D::ProcessSyncPoint() {
    if (GetNativeMacroEngine().native_turing) {
        LOG_DEBUG(HW_GPU, "Native 3D syncpoint increment: id={}, raw={:#x}", regs.sync_info.sync_point.Value(), regs.reg_array[MAXWELL3D_REG_INDEX(sync_info)]);
    }
    const u32 sync_point = regs.sync_info.sync_point.Value();
    [[maybe_unused]] const u32 cache_flush = regs.sync_info.clean_l2.Value();
    rasterizer->SignalSyncPoint(sync_point);
}

void Maxwell3D::ProcessCBBind(size_t stage_index) {
    // Bind the buffer currently in CB_ADDRESS to the specified index in the desired shader
    // stage.
    const auto& bind_data = regs.bind_groups[stage_index];
    static std::array<std::array<bool, 32>, 5> seen{};
    if (!seen[stage_index][bind_data.shader_slot]) {
        seen[stage_index][bind_data.shader_slot] = true;
        LOG_INFO(HW_GPU, "Constant buffer bind: stage {} slot {} address {:#x} size {:#x}", stage_index,
                 bind_data.shader_slot.Value(), regs.const_buffer.Address(), regs.const_buffer.size);
    }
    if (bind_data.shader_slot >= Regs::MaxConstBuffers) {
        return; // slot field is 5 bits; the state array holds 18
    }
    auto& buffer = state.shader_stages[stage_index].const_buffers[bind_data.shader_slot];
    if (VideoCommon::Nx2FastDraw::Enabled() &&
        (buffer.enabled != (bind_data.valid.Value() != 0) ||
         buffer.address != regs.const_buffer.Address() || buffer.size != regs.const_buffer.size)) {
        VideoCommon::Nx2FastDraw::NoteBind(stage_index, bind_data.shader_slot);
    }
    buffer.enabled = bind_data.valid.Value() != 0;
    buffer.address = regs.const_buffer.Address();
    buffer.size = regs.const_buffer.size;

    const bool is_enabled = bind_data.valid.Value() != 0;
    if (!is_enabled) {
        rasterizer->DisableGraphicsUniformBuffer(stage_index, bind_data.shader_slot);
        return;
    }
    const GPUVAddr gpu_addr = regs.const_buffer.Address();
    const u32 size = regs.const_buffer.size;
    rasterizer->BindGraphicsUniformBuffer(stage_index, bind_data.shader_slot, gpu_addr, size);
}

void Maxwell3D::NoteFastDrawConstantData(GPUVAddr begin, GPUVAddr end) {
    namespace FastDraw = VideoCommon::Nx2FastDraw;
    // Texture and sampler headers are read when image views are resolved, which delta draws skip.
    constexpr u64 HeaderSize = 0x20;
    const GPUVAddr tic = regs.tex_header.Address();
    const u64 tic_size = (u64{regs.tex_header.limit} + 1) * HeaderSize;
    // As TextureCache::SynchronizeDescriptors sizes the sampler table.
    const GPUVAddr tsc = regs.tex_sampler.Address();
    const u64 tsc_size = (u64{regs.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding
                                  ? regs.tex_header.limit
                                  : regs.tex_sampler.limit} +
                          1) *
                         HeaderSize;
    if ((begin < tic + tic_size && tic < end) || (begin < tsc + tsc_size && tsc < end)) {
        FastDraw::Break(FastDraw::Reason::TextureHeaders);
        return;
    }
    FastDraw::NoteConstantData(begin, end);
    // The same device memory can be mapped at other GPU addresses: compare where it lands.
    fast_draw_header_spans.Update(memory_manager, tic, tic_size);
    fast_draw_sampler_spans.Update(memory_manager, tsc, tsc_size);
    constexpr u64 PageSize = 0x1000; // the smallest GPU page
    for (GPUVAddr page = begin; page < end;) {
        const GPUVAddr page_end = (std::min)(end, (page & ~(PageSize - 1)) + PageSize);
        if (const std::optional<DAddr> device = memory_manager.GpuToCpuAddress(page)) {
            const DAddr device_end = *device + (page_end - page);
            if (fast_draw_header_spans.Intersects(*device, device_end) ||
                fast_draw_sampler_spans.Intersects(*device, device_end)) {
                FastDraw::Break(FastDraw::Reason::TextureHeaders);
                return;
            }
            FastDraw::NoteDeviceData(*device, device_end);
        }
        page = page_end;
    }
}

void Maxwell3D::ProcessCBMultiData(const u32* start_base, u32 amount) {
    if (VideoCommon::V2::Stats::Enabled()) [[unlikely]] {
        VideoCommon::V2::Stats::NoteConstantData(v2_draw_tracker);
    }
    if (VideoCommon::V2::TrackGroups()) [[unlikely]] {
        v2_groups.Touch(VideoCommon::V2::RegGroup::CbufData);
    }
    // Write the input value to the current const buffer at the current position.
    const GPUVAddr buffer_address = regs.const_buffer.Address();
    ASSERT(buffer_address != 0);

    // Don't allow writing past the end of the buffer.
    const size_t copy_size = amount * sizeof(u32);
    const size_t room = regs.const_buffer.offset < regs.const_buffer.size
                            ? regs.const_buffer.size - regs.const_buffer.offset
                            : 0;
    const size_t write_size = std::min(copy_size, room / sizeof(u32) * sizeof(u32));
    if (write_size != 0) {
        const GPUVAddr address{buffer_address + regs.const_buffer.offset};
        if (VideoCommon::Nx2FastDraw::Enabled()) {
            NoteFastDrawConstantData(address, address + write_size);
        }
        memory_manager.WriteBlockCached(address, start_base, write_size);
    }
    if (write_size != copy_size) {
        static std::atomic<bool> warned{};
        if (!warned.exchange(true, std::memory_order_relaxed)) {
            LOG_ERROR(HW_GPU,
                      "Dropping constant buffer words past the end: offset={:#x}, size={:#x}, "
                      "bytes={:#x}, written={:#x}",
                      regs.const_buffer.offset, regs.const_buffer.size, copy_size, write_size);
        }
    }

    // Increment the current buffer position.
    regs.const_buffer.offset += static_cast<u32>(copy_size);
}

void Maxwell3D::ProcessCBData(u32 value) {
    ProcessCBMultiData(&value, 1);
}

void Maxwell3D::ProcessInlineIndexMultiData(u32 method, const u32* start_base, u32 amount) {
    if (amount == 0) {
        return;
    }
    const u32 argument = ProcessShadowRam(method, start_base[amount - 1]);
    ProcessDirtyRegisters(method, argument);
    draw_manager.SetInlineIndexBuffer(*this, method, start_base, amount);
}

Texture::TICEntry Maxwell3D::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tex_header.Address() +
                                   tic_index * sizeof(Texture::TICEntry)};
    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry Maxwell3D::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tex_sampler.Address() +
                                   tsc_index * sizeof(Texture::TSCEntry)};
    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

u32 Maxwell3D::GetRegisterValue(u32 method) const {
    ASSERT(method < Regs::NUM_REGS && "Invalid Maxwell3D register");
    return regs.reg_array[method];
}

void Maxwell3D::SetHLEReplacementAttributeType(u32 bank, u32 offset,
                                               HLEReplacementAttributeType name) {
    const u64 key = (static_cast<u64>(bank) << 32) | offset;
    replace_table.emplace(key, name);
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2). The front keeps
// what macros read: the registers as the back thread will hold them when it reaches the same
// point of the method stream. That is every write the front sends, with the shadow RAM applied,
// plus the few changes the back makes on its own (constant buffer offset, shadow_scratch[0] on
// falcon[4], the macro bind pointer); the front loads and binds the macro code itself.
struct Maxwell3D::FrontState {
    std::array<ControlWrite, 16> control_writes{};
    u32 control_count{};
    std::array<u32, Regs::NUM_REGS> regs{};
    std::array<u32, Regs::NUM_REGS> shadow{};
    Regs::ShadowRamControl control{};
    u32 executing_macro{};
    std::vector<u32> parameters;
    u64 calls{};
    // Pipeline v2 P2 write ranges (FrontNoteWrites): the upload being sent, and the device
    // ranges of the render targets, recomputed when their registers change.
    std::optional<std::pair<GPUVAddr, u64>> v2_upload;
    // EDEN_NX2_V2_MACRO_DUMP / MACROHLE: macro identities by start, for one code generation,
    // and what the macro being dumped does.
    std::unordered_map<u32, VideoCommon::V2::MacroHle::Identity> v2_macros;
    struct MacroSlot {
        u32 start{~0U};
        VideoCommon::V2::MacroHle::Identity identity{};
        const VideoCommon::V2::MacroHle::Entry* entry{};
        bool dump_checked{};
    };
    std::array<MacroSlot, 64> v2_macro_slots{};
    u64 v2_macro_generation{~u64{0}};
    /// What the macro being dumped or checked (MACROHLE=verify) does.
    std::unique_ptr<VideoCommon::V2::MacroHle::Observation> v2_observation;
    /// MACROHLE=1: since a conditional render call the back evaluated, regs[render_enable.mode]
    /// here may be older than the back's (FrontSyncRenderEnable takes the back's before an MME
    /// read of it). Its shadow copy, when the call ran under shadow RAM tracking, holds
    /// ShadowUnknown: a Replay of it sends that (S1) or the method word (the back's own shadow
    /// RAM replays it), and the back resolves it (ProcessQueryCondition); the front never waits.
    /// A replayed ShadowUnknown in regs counts as unknown too.
    bool v2_render_enable_unknown{};
    /// EDEN_NX2_V2_MACROMEMO: replayed macro calls (Fix F5-02).
    std::unique_ptr<VideoCommon::V2::MacroMemo::Memo> v2_memo;
    /// EDEN_NX2_V2_STATS: calls per macro and exact repeats (Fix F5-01).
    std::unique_ptr<VideoCommon::V2::MacroMemo::CallStats> v2_call_stats;
    // FrontNoteDraw's verdict on the bound programs' stores, kept until a Shaders-group register,
    // a register it reads, a summary (ShaderWrites::Generation) or a mapping (MapEpoch) changes.
    bool v2_programs_dirty{true};
    bool v2_programs_unbounded{};
    u32 v2_programs_state{};
    u32 v2_programs_generation{};
    u32 v2_programs_map_epoch{};
    /// A method the draw manager handles was sent since the back last ran DrawDeferred: only
    /// then can it have a deferred instanced draw to issue at the end of a macro.
    bool draw_pending{true};
    /// The back holds state writes in its method sink, which its next executable method consumes
    /// (issuing a deferred draw a changed one ends). Front() consumes it before copying the state.
    bool sink_pending{};
    u32 split_start{}; ///< GpuSplit::Starts() when this copy was made
    bool skip_same{}; ///< SkipSameValueWrites() && FrontSkipSame(), read once
    /// The back's draw manager is in an instance loop (DrawMode::Instance): draw.end does nothing.
    bool instance_loop{};
    /// EDEN_NX2_V2_REGSEM (null when off): groups and fingerprints of these registers (=1), or
    /// the table-driven copy compared with them (verify).
    VideoCommon::V2::GroupTracker v2_groups;
    /// EDEN_NX2_V2_REGSEM=1: the hashes of v2_groups wait here until something reads them
    /// (folded by FrontDrawGroups and FrontParallelDraw; Perf fix D).
    VideoCommon::V2::RegSem::HashLog3D v2_hash_log;
    /// EDEN_NX2_V2_REGSEM=1: the constant buffer bindings these registers made (P6c snapshots),
    /// kept only with EDEN_NX2_V2_PARALLEL (v2_cbufs_wanted).
    VideoCommon::V2::Resolve::CbufTable v2_cbufs;
    bool v2_cbufs_wanted{};
    /// EDEN_NX2_V2_CBUFRING (S1 with the hazard table only): decode-time cbuf versions.
    std::unique_ptr<VideoCommon::V2::CbufRing::Decode> v2_cbuf_ring;
    /// F2-03: the split whose range write notes reach v2_cbuf_ring (cleared on release).
    VideoCommon::GpuSplit* v2_listener_split{};
    /// Register blocks of 64 changed since the last P6 snapshot (all after a copy or resync).
    u64 v2_dirty_blocks{~u64{0}};
    std::unique_ptr<VideoCommon::V2::RegSem::Front3D> v2;
    /// EDEN_NX2_V2_REGSEM=1 (v2 non-null and On): FrontWrite's table-driven fast path.
    bool v2_on{};
    const VideoCommon::V2::RegSem::Table3D* v2_table{};
    /// Registers FrontNoteWrites acts on (GpuSplit::TracksWrites): any other write costs no
    /// call (Perf fix D). Filled when the copy is made.
    std::array<bool, Regs::NUM_REGS> v2_note_writes{};
    /// EDEN_NX2_V2_IR_DRAWS: the back's draw manager as the front predicts it (Draw records).
    VideoCommon::V2::DrawModel v2_draw_model;
    bool back_skip_same{}; ///< SkipSameValueWrites(): the back stores only changed state writes
    /// S1 single owner (Fix F1): GpuSplit::S1Exec (classified runs only) and S1Check (method
    /// words plus the classified entries the back checks), fixed when the copy is made.
    bool s1_exec{};
    bool s1_check{};
    /// The register maps to a v1 dirty flag: an executable write sets it even when unchanged.
    std::array<bool, Regs::NUM_REGS> s1_touch{};
    /// S1Hook* bits per register: what FrontS1Hooks does after a write (none: no call). Only the
    /// registers with write notes or stats have any; the profile counts below need none.
    std::array<u8, Regs::NUM_REGS> s1_hooks{};
    /// S1 with write tracking: the sink notes each constant buffer word's range (Fix F3d).
    bool s1_note_cb{};
    /// Nx2Profile write counters, counted here (no thread-local access per write) and added to
    /// the profile at each draw.
    u64 s1_writes{};
    u64 s1_same_writes{};
    bool s1_profile{};
    /// Fast draw breaks for the next S1 Groups entry (V2::S1::Groups*Break) and the method.
    u64 s1_breaks{};
    u32 s1_break_method{};
};

namespace {

// EDEN_NX2_SPLIT_VERIFY=1: every 4096 front macro calls, wait for the back and compare the
// front's registers with the back's (logs and resynchronizes differences).
// F1-X3: read once, then one relaxed load and a predictable branch per call (a function-local
// static paid its thread-safe initialization check, a TLS load, out of line on every macro call).
std::atomic<s8> g_front_verify{-1};

bool ReadFrontVerify() {
    const bool verify = std::getenv("EDEN_NX2_SPLIT_VERIFY") != nullptr;
    g_front_verify.store(verify ? 1 : 0, std::memory_order_relaxed);
    return verify;
}

inline bool FrontVerifyEnabled() {
    const s8 verify = g_front_verify.load(std::memory_order_relaxed);
    if (verify >= 0) [[likely]] {
        return verify != 0;
    }
    return ReadFrontVerify();
}

// Executable methods ProcessMethodCall handles itself; the rest go to the draw manager.
bool FrontHandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case MAXWELL3D_REG_INDEX(wait_for_idle):
    case MAXWELL3D_REG_INDEX(shadow_ram_control):
    case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
    case MAXWELL3D_REG_INDEX(load_mme.instruction):
    case MAXWELL3D_REG_INDEX(load_mme.start_address):
    case MAXWELL3D_REG_INDEX(falcon[4]):
    case MAXWELL3D_REG_INDEX(bind_groups[0].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[1].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[2].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[3].raw_config):
    case MAXWELL3D_REG_INDEX(bind_groups[4].raw_config):
    case MAXWELL3D_REG_INDEX(report_semaphore.query):
    case MAXWELL3D_REG_INDEX(render_enable.mode):
    case MAXWELL3D_REG_INDEX(clear_report_value):
    case MAXWELL3D_REG_INDEX(sync_info):
    case MAXWELL3D_REG_INDEX(launch_dma):
    case MAXWELL3D_REG_INDEX(inline_data):
    case MAXWELL3D_REG_INDEX(fragment_barrier):
    case MAXWELL3D_REG_INDEX(invalidate_texture_data_cache):
    case MAXWELL3D_REG_INDEX(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

/// MACROHLE: the front's shadow copy of render_enable.mode after a conditional render the back
/// evaluated under shadow RAM tracking (the mode register holds 0..4; no game writes this).
constexpr u32 ShadowUnknown = ~0U;
constexpr u32 RenderEnableMode = MAXWELL3D_REG_INDEX(render_enable.mode);

// EDEN_NX2_SPLIT_SKIP=0: send state writes that keep the register's value too.
bool FrontSkipSame() {
    static const bool skip = [] {
        const char* value = std::getenv("EDEN_NX2_SPLIT_SKIP");
        return value == nullptr || value[0] != '0';
    }();
    return skip;
}

} // Anonymous namespace

Maxwell3D::FrontState& Maxwell3D::Front(Core::System& system, VideoCommon::GpuSplit& split) {
    // Copied during an earlier start: while the split was stopped, state writes went to the
    // method sink without reaching it (they hand the front back only once consumed).
    if (front && front->split_start != split.Starts()) [[unlikely]] {
        split.Drain(VideoCommon::DrainSite::FrontStart); // The engine's state is the back's.
        ReleaseFront();
    }
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle (state writes it
        // still holds in its method sink first).
        split.Drain(VideoCommon::DrainSite::FrontStart);
        ConsumeSink(system);
        front = std::make_unique<FrontState>();
        std::memcpy(front->regs.data(), regs.reg_array.data(), sizeof(front->regs));
        std::memcpy(front->shadow.data(), shadow_state.reg_array.data(), sizeof(front->shadow));
        front->control = shadow_state.shadow_ram_control;
        front->skip_same = SkipSameValueWrites() && FrontSkipSame();
        front->split_start = split.Starts();
        front->executing_macro = executing_macro;
        front->back_skip_same = SkipSameValueWrites();
        {
            // The draw manager's state as the back left it (it is idle).
            auto& model = front->v2_draw_model.state;
            const auto& draw = draw_manager.draw_state;
            model.mode = static_cast<u8>(draw.draw_mode);
            model.topology = static_cast<u8>(draw.topology);
            model.draw_indexed = draw.draw_indexed;
            model.instance_count = draw.instance_count;
            model.drawn_instances = draw.drawn_instances;
            model.base_index = draw.base_index;
            model.base_instance = draw.base_instance;
            model.inline_bytes = static_cast<u32>(draw.inline_index_draw_indexes.size());
            model.vertex_first = draw.vertex_buffer.first;
            model.vertex_count = draw.vertex_buffer.count;
            model.index_first = draw.index_buffer.first;
            model.index_count = draw.index_buffer.count;
        }
        front->parameters = std::move(macro_params);
        macro_params.clear();
        executing_macro = 0;
        front_macros = true;
        {
            // What FrontNoteWrites looks at (keep the two in step).
            namespace V2 = VideoCommon::V2;
            const auto& table = V2::RegSem::GetTable3D();
            front->v2_table = &table;
            for (u32 method = 0; method < Regs::NUM_REGS; ++method) {
                const u32 flags = table.flags[method];
                front->v2_note_writes[method] =
                    table.groups[method] == V2::RegGroup::Shaders ||
                    method == MAXWELL3D_REG_INDEX(clear_surface) ||
                    method == MAXWELL3D_REG_INDEX(launch_dma) ||
                    (flags & (V2::RegSem::DrawEnd | V2::RegSem::InlineData | V2::RegSem::CbData |
                              V2::RegSem::ReportSemaphore)) != 0;
            }
        }
        if (VideoCommon::V2::On(VideoCommon::V2::Switch::RegSem)) [[unlikely]] {
            front->v2 = std::make_unique<VideoCommon::V2::RegSem::Front3D>();
            front->v2->mode = VideoCommon::V2::Get(VideoCommon::V2::Switch::RegSem);
            front->v2_on = front->v2->mode == VideoCommon::V2::Mode::On;
            front->v2_cbufs_wanted = split.Parallel() != nullptr;
            FrontRegSemResync(*front);
            front->s1_exec = split.S1Exec() && front->v2_on;
            front->s1_check = split.S1Check();
            if (front->s1_exec || front->s1_check) {
                const auto& table = VideoCommon::V2::RegSem::GetTable3D();
                front->s1_profile = VideoCommon::Nx2Profile::Enabled();
                const bool stats = VideoCommon::V2::Stats::Enabled();
                for (u32 method = 0; method < Regs::NUM_REGS; ++method) {
                    front->s1_touch[method] =
                        dirty.tables[0][method] != 0 || dirty.tables[1][method] != 0;
                    u8 hooks = 0;
                    // CB data words are noted by the S1 sink's Cbuf (F3d).
                    const bool cb_data = (table.flags[method] & VideoCommon::V2::RegSem::CbData) != 0;
                    if (split.TracksWrites() && front->v2_note_writes[method] &&
                        !(front->s1_exec && cb_data)) {
                        hooks |= S1HookNotes;
                    }
                    if (stats && (table.flags[method] & VideoCommon::V2::RegSem::ReportSemaphore)) {
                        hooks |= S1HookStats;
                    }
                    front->s1_hooks[method] = hooks;
                }
                front->s1_note_cb = front->s1_exec && split.TracksWrites();
            }
            if (front->s1_exec && split.TracksWrites() &&
                VideoCommon::V2::On(VideoCommon::V2::Switch::CbufRing) &&
                VideoCommon::V2::CbufRing::Get().Ready()) [[unlikely]] {
                front->v2_cbuf_ring = std::make_unique<VideoCommon::V2::CbufRing::Decode>(
                    VideoCommon::V2::CbufRing::Get(), VideoCommon::V2::CbufWatch::Enabled());
                front->v2_cbufs_wanted = true; // the binds make versions (FrontCbufBindVersions)
                if (VideoCommon::V2::CbufWatch::Enabled() &&
                    split.RangeWriteContext() == nullptr) {
                    // F2-03: other producers' ranged writes drop the windows they overlap.
                    split.SetRangeWriteListener(
                        [](void* context, u64 device_address, u64 size) {
                            static_cast<VideoCommon::V2::CbufRing::Decode*>(context)->NoteWrite(
                                device_address, size);
                        },
                        front->v2_cbuf_ring.get());
                    front->v2_listener_split = &split;
                }
            }
            if (front->s1_exec) {
                // The front's DrawModel decides every draw from now on (it took the state above);
                // the back's state machine stays idle so that nothing on it issues a draw.
                auto& draw = draw_manager.draw_state;
                draw.draw_mode = DrawManager::DrawMode::General;
                draw.instance_count = 0;
                draw.drawn_instances = 0;
                draw.draw_indexed = false;
            }
        }
    }
    return *front;
}

// With the split on, engine methods run on its back only: one here on another thread means the
// split stopped (accuracy settings changed mid-game). Hand the macros back: the front's copy
// goes stale from now on, and a restart copies the registers again (Front).
void Maxwell3D::ReleaseFront() {
    if (front->s1_exec) {
        // The draw state machine as the front's model left it (the back's stayed idle).
        const auto& model = front->v2_draw_model.state;
        auto& draw = draw_manager.draw_state;
        draw.draw_mode = static_cast<DrawManager::DrawMode>(model.mode);
        draw.topology = static_cast<Regs::PrimitiveTopology>(model.topology);
        draw.draw_indexed = model.draw_indexed;
        draw.instance_count = model.instance_count;
        draw.drawn_instances = model.drawn_instances;
    }
    executing_macro = front->executing_macro;
    macro_params = std::move(front->parameters);
    if (front->shadow[RenderEnableMode] == ShadowUnknown) {
        // MACROHLE: the mode the back evaluated last under shadow RAM tracking.
        front->shadow[RenderEnableMode] = v2_hle_tracked_mode;
    }
    std::memcpy(shadow_state.reg_array.data(), front->shadow.data(), sizeof(front->shadow));
    shadow_state.shadow_ram_control = front->control;
    if (front->v2_listener_split != nullptr &&
        front->v2_listener_split->RangeWriteContext() == front->v2_cbuf_ring.get()) {
        front->v2_listener_split->SetRangeWriteListener(nullptr, nullptr);
    }
    front.reset();
    front_macros = false;
    LOG_INFO(HW_GPU, "GPU thread split stopped: 3D macros back on the engine");
}

// EDEN_NX2_V2_REGSEM=1, every front register write (~6M/s, most from macros): inlined into
// FrontWrite, with the table read through the front's pointer, the hashes deferred to the
// log (HashLog3D), and the draw model and write notes called only for methods they act on.
M3D_FORCE_INLINE void Maxwell3D::FrontWriteRegSemOn(VideoCommon::GpuSplit& split,
                                                    FrontState& fs, u32 method, u32 argument,
                                                    bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    namespace RegSem = VideoCommon::V2::RegSem;
    const RegSem::Table3D& table = *fs.v2_table;
    const u16 flags = table.flags[method];
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    if ((flags & RegSem::ShadowControl) != 0 && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    const bool v2_draws = split.DrawRecords();
    if (v2_draws && fs.v2_draw_model.HasDeferredDraw()) [[unlikely]] {
        FrontDrawPre(split, fs, method,
                     fs.control == Regs::ShadowRamControl::Replay ? fs.shadow[method] : argument);
    }
    const u32 old_value = fs.regs[method];
    const u32 old_offset = fs.regs[table.const_buffer_offset];
    if (!RegSem::Write3D(fs, table, method, argument, from_macro,
                         [this](RegSem::Effect effect, u32 a, u32 b) {
                             return FrontRegSemEffect(effect, a, b);
                         })) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        return;
    }
    // FrontDrawPost acts on executable draw manager methods only (DrawModel::IsDrawMethod).
    if (v2_draws && (flags & RegSem::DrawManager) != 0) [[unlikely]] {
        FrontDrawPost(split, fs, method, fs.regs[method]);
    }
    if (from_macro) {
        split.MacroWrite(this, method, argument);
    } else {
        split.Method(this, method, argument, is_last_call, nullptr);
    }
    if (fs.s1_check) [[unlikely]] {
        FrontCheckS1(split, fs, fs.regs.data(), method, old_value, old_offset, is_last_call);
    }
    if (split.TracksWrites() && fs.v2_note_writes[method]) [[unlikely]] {
        FrontNoteWrites(split, fs, method, 1);
    }
    if ((flags & RegSem::ReportSemaphore) != 0 && VideoCommon::V2::Stats::Enabled())
        [[unlikely]] {
        VideoCommon::V2::Stats::NoteReportSemaphore(memory_manager.GetID(), fs.regs.data());
    }
}

/// The S1 entries for the split (V2::S1::Classify3D's sink).
struct Maxwell3D::S1Sink {
    VideoCommon::GpuSplit& split;
    Maxwell3D* engine;
    FrontState* fs{}; ///< Set where EDEN_NX2_V2_CBUFRING versions the data words

    void Delta(u32 index, u32 value) {
        namespace S1 = VideoCommon::V2::S1;
        split.S1Word(engine, S1::MakeRegister(S1::Tag::Delta, index, value));
    }
    void Quiet(u32 index, u32 value) {
        namespace S1 = VideoCommon::V2::S1;
        split.S1Word(engine, S1::MakeRegister(S1::Tag::Delta, index, value, S1::DeltaQuiet));
    }
    void Exec(u32 index, u32 value, bool is_last) {
        namespace S1 = VideoCommon::V2::S1;
        split.S1Word(engine,
                     S1::MakeRegister(S1::Tag::Exec, index, value, is_last ? S1::IsLast : 0));
    }
    void Cbuf(bool store, u32 method, u64 buffer, u32 offset, u32 word) {
        split.S1Cbuf(engine, store, method - MAXWELL3D_REG_INDEX(const_buffer.buffer), buffer,
                     offset, word);
        if (fs == nullptr) {
            return; // S1Check's copy: notes nothing
        }
        if (fs->s1_note_cb) {
            // Fix F3d (Fix F1's design: CB data ranges per Cbuf entry, not per word through
            // FrontS1Hooks -> FrontNoteWrites): the split coalesces the words of an entry.
            split.NoteCbWrite(engine->memory_manager, buffer + offset, sizeof(u32));
        }
        if (fs->v2_cbuf_ring) [[unlikely]] {
            engine->FrontCbufVersion(split, *fs, buffer, offset, &word, 1);
        }
    }
    void Inline(u32 method, const u32* words, u32 count) {
        split.S1Inline(engine, method, words, count);
    }
};

// S1 single owner (Fix F1, GpuSplit::S1Exec): every front register write. The semantics run
// here once (Write3D); the back gets the post-shadow deltas of the registers that changed and a
// typed entry for the writes with side effects (V2::S1::Classify3D), never the method words.
M3D_FORCE_INLINE void Maxwell3D::FrontWriteS1(VideoCommon::GpuSplit& split, FrontState& fs,
                                              u32 method, u32 argument, bool is_last_call,
                                              bool from_macro) {
    namespace RegSem = VideoCommon::V2::RegSem;
    const RegSem::Table3D& table = *fs.v2_table;
    const u16 flags = table.flags[method];
    // Profile counters, write notes and stats for this method, decided when the copy was made:
    // one table load and one branch per write (F1-09).
    const u8 hooks = fs.s1_hooks[method];
    ++fs.s1_writes;
    if (fs.v2_draw_model.HasDeferredDraw()) [[unlikely]] {
        FrontDrawPre(split, fs, method,
                     fs.control == Regs::ShadowRamControl::Replay ? fs.shadow[method] : argument);
    }
    const u32 old_value = fs.regs[method];
    const u32 old_offset = fs.regs[table.const_buffer_offset];
    if (!RegSem::Write3D(fs, table, method, argument, from_macro,
                         [this](RegSem::Effect effect, u32 a, u32 b) {
                             return FrontRegSemEffect(effect, a, b);
                         })) {
        ++fs.s1_same_writes; // the hooks only follow sent writes
        return;
    }
    S1Sink sink{split, this, &fs};
    VideoCommon::V2::S1::Classify3D(fs.regs.data(), table, fs.s1_touch.data(), fs.back_skip_same,
                                    method, old_value, old_offset, is_last_call, sink);
    if ((flags & RegSem::DrawManager) != 0) [[unlikely]] {
        // What TableMethodCall breaks fast draw reuse for, at the next draw.
        if ((flags & RegSem::FastDrawQuiet) == 0 &&
            table.actions[method] == RegSem::Action3D::DrawManager) {
            fs.s1_breaks |= VideoCommon::V2::S1::GroupsMethodBreak;
            fs.s1_break_method = method;
        }
        FrontDrawPost(split, fs, method, fs.regs[method]);
    }
    if (hooks != 0) [[unlikely]] {
        FrontS1Hooks(split, fs, method, hooks);
    }
}

void Maxwell3D::FrontS1Hooks(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                             u8 hooks) {
    if ((hooks & S1HookNotes) != 0) {
        FrontNoteWrites(split, fs, method, 1);
    }
    if ((hooks & S1HookStats) != 0) {
        VideoCommon::V2::Stats::NoteReportSemaphore(memory_manager.GetID(), fs.regs.data());
    }
}

void Maxwell3D::FrontMultiS1(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                             const u32* base_start, u32 amount, u32 methods_pending) {
    namespace V2 = VideoCommon::V2;
    namespace RegSem = V2::RegSem;
    const RegSem::Table3D& table = *fs.v2_table;
    const u16 flags = table.flags[method];
    V2::DrawModel& model = fs.v2_draw_model;
    // As CallMultiMethod: a run of a method outside instance loops ends a deferred batch first.
    if (!V2::DrawModel::IsInstanceLoopMethod(method)) {
        V2::DrawOutput out;
        model.OnMultiStart(method, fs.regs.data(), out);
        if (split.Parallel() != nullptr) [[unlikely]] {
            FrontParallelDraw(split, fs, out, ~0U, true);
        }
        FrontS1Draws(split, fs, out);
    }
    const auto effects = [this](RegSem::Effect effect, u32 a, u32 b) {
        return FrontRegSemEffect(effect, a, b);
    };
    if ((flags & RegSem::CbData) != 0) {
        const u32 offset = fs.regs[table.const_buffer_offset];
        const u64 buffer = (u64{fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
                           fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)];
        RegSem::WriteMulti3D(fs, table, method, base_start, amount, effects);
        split.S1CbufBlock(this, buffer, offset, base_start, amount);
        if (fs.v2_cbuf_ring) [[unlikely]] {
            FrontCbufVersion(split, fs, buffer, offset, base_start, amount);
        }
    } else if ((flags & RegSem::InlineData) != 0) {
        // Upload data only: the back's CallMultiMethod reads no register for it.
        RegSem::WriteMulti3D(fs, table, method, base_start, amount, effects);
        split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
    } else if ((flags & RegSem::InlineIndex) != 0 &&
               fs.control != Regs::ShadowRamControl::Replay) {
        if (amount == 0) {
            return;
        }
        const u32 old_value = fs.regs[method];
        model.OnInlineIndex(method, amount);
        RegSem::WriteMulti3D(fs, table, method, base_start, amount, effects);
        if (fs.regs[method] != old_value || fs.s1_touch[method]) {
            split.S1Word(this, V2::S1::MakeRegister(V2::S1::Tag::Delta, method, fs.regs[method]));
        }
        split.S1Inline(this, method, base_start, amount);
    } else {
        // Each word as a write of its own (CallMultiMethod calls CallMethod per word).
        for (u32 i = 0; i < amount; ++i) {
            FrontWriteS1(split, fs, method, base_start[i], methods_pending - i <= 1, false);
        }
        return;
    }
    if (split.TracksWrites() && fs.v2_note_writes[method]) [[unlikely]] {
        FrontNoteWrites(split, fs, method, amount);
    }
}

void Maxwell3D::FrontCheckS1(VideoCommon::GpuSplit& split, FrontState& fs, const u32* table_regs,
                             u32 method, u32 old_value, u32 old_offset, bool is_last_call) {
    S1Sink sink{split, this};
    VideoCommon::V2::S1::Classify3D(table_regs, VideoCommon::V2::RegSem::GetTable3D(),
                                    fs.s1_touch.data(), fs.back_skip_same, method, old_value,
                                    old_offset, is_last_call, sink);
}

void Maxwell3D::FrontCheckMultiS1(VideoCommon::GpuSplit& split, const u32* table_regs,
                                  u32 method, const u32* base_start, u32 amount, u32 old_offset) {
    namespace V2 = VideoCommon::V2;
    const auto& table = V2::RegSem::GetTable3D();
    const u16 flags = table.flags[method];
    if ((flags & V2::RegSem::CbData) != 0) {
        const u64 buffer =
            (u64{table_regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            table_regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)];
        // One entry for the whole run: the check compares after each entry, and the back
        // executed the run's MethodData records whole before it.
        split.S1CbufBlock(this, buffer, old_offset, base_start, amount);
        return;
    }
    if ((flags & V2::RegSem::InlineData) != 0) {
        return;
    }
    // The registers the run can leave changed (the back ran it word by word).
    for (const u32 index : {method, table.start_address_ptr, table.shadow_scratch}) {
        split.S1Word(this, V2::S1::MakeRegister(V2::S1::Tag::Delta, index, table_regs[index]));
    }
}

void Maxwell3D::FrontS1Draws(VideoCommon::GpuSplit& split, FrontState& fs,
                             const VideoCommon::V2::DrawOutput& out) {
    namespace S1 = VideoCommon::V2::S1;
    if (out.count == 0) {
        return;
    }
    if (fs.v2_cbuf_ring) [[unlikely]] {
        FrontCbufBindVersions(split, fs);
        if (fs.v2_cbuf_ring->Pending()) {
            FrontPublishCbufVersions(split, fs);
        }
    }
    FrontS1Groups(split, fs);
    for (u32 i = 0; i < out.count; ++i) {
        u64* const entry = split.S1Begin(this, S1::DrawWords);
        entry[0] = static_cast<u64>(S1::Tag::Draw) | (u64{out.variants[i]} << 8);
        std::memcpy(entry + 1, &out.draws[i], sizeof(out.draws[i]));
        split.S1Commit(S1::DrawWords);
    }
    split.S1PaceDraws(out.count);
}

namespace {

/// EDEN_NX2_V2_CBUFRING: what the decode-time versions read (v2/cbuf_ring.h Decode's Env).
struct CbufRingEnv {
    VideoCommon::GpuSplit& split;
    const Tegra::MemoryManager& memory;

    u32 Writers(u64 va, u32 size) {
        namespace V2 = VideoCommon::V2;
        const auto first = memory.GpuToCpuAddress(va);
        const auto last = memory.GpuToCpuAddress(va + size - 1);
        if (!first || !last || *last != *first + size - 1) {
            return V2::CbufRing::Decode::UnknownWriters;
        }
        // (a): in-flight writes of any producer over the window, this front's constant
        // buffer data included, by the hazard table. Unbounded writers (shader stores the
        // front can't place) are left to the back's compare at the bind.
        V2::HazardFilter filter{};
        filter.ranged = V2::AllProducers;
        const V2::HazardResult result = split.FindWrites(*first, size, filter);
        if (result.kind == V2::HazardResult::Kind::Clear) {
            return 0;
        }
        return result.producers != 0 ? result.producers : V2::CbufRing::Decode::UnknownWriters;
    }
    bool Read(u64 va, u32 size, u8* dst) {
        memory.ReadBlockUnsafe(va, dst, size);
        return true;
    }
    bool Same(u64 va, u32 size, const u8* bytes) {
        thread_local std::array<u8, VideoCommon::V2::CbufRing::Decode::MaxWindow> scratch;
        memory.ReadBlockUnsafe(va, scratch.data(), size);
        return std::memcmp(scratch.data(), bytes, size) == 0;
    }
    // F2-03: where the window lives in device memory (the overlap index and the watch).
    u64 Device(u64 va, u32 size) {
        const auto first = memory.GpuToCpuAddress(va);
        const auto last = memory.GpuToCpuAddress(va + size - 1);
        if (!first || !last || *last != *first + size - 1) {
            return VideoCommon::V2::CbufRing::NoPosition;
        }
        return *first;
    }
    u64 Maps() {
        return VideoCommon::GpuSplit::MapsExecuted();
    }
};

} // Anonymous namespace

void Maxwell3D::FrontCbufVersion(VideoCommon::GpuSplit& split, FrontState& fs, u64 buffer,
                                 u32 offset, const u32* words, u32 count) {
    namespace V2 = VideoCommon::V2;
    CbufRingEnv env{split, memory_manager};
    const u32 size = fs.regs[MAXWELL3D_REG_INDEX(const_buffer.size)];
    fs.v2_cbuf_ring->Write(buffer, size, offset, words, count,
                           V2::ProducerBit(V2::Producer::CbData), env);
}

void Maxwell3D::FrontCbufBindVersions(VideoCommon::GpuSplit& split, FrontState& fs) {
    // The windows bound since the last draw: a version for those without one (F2-02 coverage:
    // the guest CPU fills most of them, so constant buffer data never makes theirs).
    namespace V2 = VideoCommon::V2;
    CbufRingEnv env{split, memory_manager};
    auto& cbufs = fs.v2_cbufs;
    for (size_t stage = 0; stage < V2::Resolve::CbufStages; ++stage) {
        for (u32 bits = std::exchange(cbufs.bound[stage], 0U); bits != 0; bits &= bits - 1) {
            const V2::Resolve::CbufBinding& bind = cbufs.stages[stage][std::countr_zero(bits)];
            if (bind.enabled != 0 && bind.address != 0) {
                fs.v2_cbuf_ring->Bind(bind.address, bind.size,
                                      V2::ProducerBit(V2::Producer::CbData), env);
            }
        }
    }
}

void Maxwell3D::FrontPublishCbufVersions(VideoCommon::GpuSplit& split, FrontState& fs) {
    namespace S1 = VideoCommon::V2::S1;
    fs.v2_cbuf_ring->Publish([&](u64 va, u32 size, u64 position) {
        u64* const entry = split.S1Begin(this, S1::CbufVersionWords);
        entry[0] = static_cast<u64>(S1::Tag::CbufVersion) | (u64{size} << 32);
        entry[1] = va;
        entry[2] = position;
        split.S1Commit(S1::CbufVersionWords);
    });
}

void Maxwell3D::FrontS1Groups(VideoCommon::GpuSplit& split, FrontState& fs) {
    namespace V2 = VideoCommon::V2;
    namespace S1 = V2::S1;
    if (fs.s1_profile) [[unlikely]] {
        namespace Profile = VideoCommon::Nx2Profile;
        Profile::Add(Profile::Counter::MethodWrites, std::exchange(fs.s1_writes, u64{0}));
        Profile::Add(Profile::Counter::SameValueWrites, std::exchange(fs.s1_same_writes, u64{0}));
    }
    // The groups the back's draw consumers see, from the front's own fingerprints.
    fs.v2_hash_log.Fold(fs.v2_groups, *fs.v2_table, fs.regs.data());
    V2::GroupTracker& groups = fs.v2_groups;
    const u32 changed = groups.dirty;
    if (changed != 0 || fs.s1_breaks != 0) {
        const u64 breaks =
            fs.s1_breaks | ((changed & ~S1::LightGroups) != 0 ? S1::GroupsRegisterBreak : 0);
        const u32 hashed = changed & V2::HashedGroups;
        const size_t size = 1 + static_cast<size_t>(std::popcount(hashed));
        u64* const entry = split.S1Begin(this, size);
        entry[0] = S1::MakeGroups(changed, breaks, fs.s1_break_method);
        size_t slot = 1;
        for (u32 bits = hashed; bits != 0; bits &= bits - 1) {
            entry[slot++] = groups.hashes[std::countr_zero(bits)];
        }
        split.S1Commit(size);
        fs.s1_breaks = 0;
    }
    groups.BeginDraw();
}

void Maxwell3D::FrontWrite(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                           u32 argument, bool is_last_call, bool from_macro) {
    if (fs.s1_exec) {
        FrontWriteS1(split, fs, method, argument, is_last_call, from_macro);
        return;
    }
    if (fs.v2_on) {
        FrontWriteRegSemOn(split, fs, method, argument, is_last_call, from_macro);
        return;
    }
    if (fs.v2 != nullptr) [[unlikely]] {
        return FrontWriteRegSem(split, fs, method, argument, is_last_call, from_macro);
    }
    FrontWriteV1(split, fs, method, argument, is_last_call, from_macro);
}

void Maxwell3D::FrontWriteRegSem(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                 u32 argument, bool is_last_call, bool from_macro) {
    namespace RegSem = VideoCommon::V2::RegSem;
    if (fs.v2->mode != VideoCommon::V2::Mode::Verify) {
        FrontWriteRegSemOn(split, fs, method, argument, is_last_call, from_macro);
        return;
    }
    // The hand-written mirror decides; the table-driven copy follows and is compared.
    const RegSem::Table3D& table = RegSem::GetTable3D();
    const u64 before = split.SentWords();
    FrontWriteV1(split, fs, method, argument, is_last_call, from_macro);
    const bool sent = split.SentWords() != before;
    RegSem::Mirror3D& mirror = *fs.v2->mirror;
    const u32 old_value = mirror.regs[method];
    const u32 old_offset = mirror.regs[table.const_buffer_offset];
    const bool table_sent =
        RegSem::Write3D(mirror, table, method, argument, from_macro,
                        [](RegSem::Effect, u32, u32) { return true; });
    if (fs.s1_check && table_sent) {
        // What S1 would send for the write, checked by the back against its execution.
        FrontCheckS1(split, fs, mirror.regs.data(), method, old_value, old_offset, is_last_call);
    }
    RegSem::CompareWrite3D(*fs.v2, fs, table, method, argument, sent, table_sent);
}

bool Maxwell3D::FrontRegSemEffect(VideoCommon::V2::RegSem::Effect effect, u32 a, u32 b) {
    switch (effect) {
    case VideoCommon::V2::RegSem::Effect::ClearCode:
        GetNativeMacroEngine().ClearCode(a);
        return true;
    case VideoCommon::V2::RegSem::Effect::AddCode:
        GetNativeMacroEngine().AddCode(a, b);
        return true;
    case VideoCommon::V2::RegSem::Effect::Bind: {
        const auto positions = GetNativeMacroPositions();
        if (a < positions.size()) {
            positions[a] = b;
            return true;
        }
        return false;
    }
    }
    return true;
}

void Maxwell3D::FrontMultiRegSem(FrontState& fs, u32 method, const u32* base_start, u32 amount,
                                 bool apply) {
    namespace RegSem = VideoCommon::V2::RegSem;
    const RegSem::Table3D& table = RegSem::GetTable3D();
    if (apply) {
        RegSem::WriteMulti3D(fs, table, method, base_start, amount,
                             [this](RegSem::Effect effect, u32 a, u32 b) {
                                 return FrontRegSemEffect(effect, a, b);
                             });
        return;
    }
    RegSem::WriteMulti3D(*fs.v2->mirror, table, method, base_start, amount,
                         [](RegSem::Effect, u32, u32) { return true; });
    RegSem::CompareWrite3D(*fs.v2, fs, table, method, amount != 0 ? base_start[amount - 1] : 0,
                           true, true);
}

void Maxwell3D::FrontRegSemResync(FrontState& fs) {
    // The front's registers were set without the write path (copied, or resynchronized with
    // the back by EDEN_NX2_SPLIT_VERIFY).
    fs.v2_dirty_blocks = ~u64{0};
    for (size_t stage = 0; stage < VideoCommon::V2::Resolve::CbufStages; ++stage) {
        for (size_t slot = 0; slot < VideoCommon::V2::Resolve::CbufSlots; ++slot) {
            const auto& cbuf = state.shader_stages[stage].const_buffers[slot];
            fs.v2_cbufs.stages[stage][slot] = {cbuf.address, cbuf.size, cbuf.enabled ? 1U : 0U};
        }
    }
    ++fs.v2_cbufs.changes;
    if (fs.v2->mode == VideoCommon::V2::Mode::Verify) {
        if (!fs.v2->mirror) {
            fs.v2->mirror = std::make_unique<VideoCommon::V2::RegSem::Mirror3D>();
        }
        VideoCommon::V2::RegSem::CopyMirror3D(*fs.v2->mirror, fs);
    } else {
        fs.v2_hash_log.Clear();
        fs.v2_groups.valid = false;
        VideoCommon::V2::RegSem::Rehash3D(fs.v2_groups, fs.regs.data());
        fs.v2_groups.dirty = VideoCommon::V2::AllGroups;
    }
}

void Maxwell3D::FrontWriteV1(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                             u32 argument, bool is_last_call, bool from_macro) {
    namespace Profile = VideoCommon::Nx2Profile;
    // As ProcessShadowRam does on the back.
    u32 value = argument;
    bool shadow_kept = true;
    if (fs.control == Regs::ShadowRamControl::Track ||
        fs.control == Regs::ShadowRamControl::TrackWithFilter) {
        shadow_kept = fs.shadow[method] == argument;
        fs.shadow[method] = argument;
    } else if (fs.control == Regs::ShadowRamControl::Replay) {
        value = fs.shadow[method];
    }
    if (Profile::Enabled()) [[unlikely]] {
        Profile::Add(Profile::Counter::MethodWrites, 1);
    }
    if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
        fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
            argument, static_cast<u32>(fs.control), from_macro ? 2U : 1U};
    }
    const bool v2_draws = split.DrawRecords();
    if (v2_draws) [[unlikely]] {
        FrontDrawPre(split, fs, method, value);
    }
    if (execution_mask[method] && FrontSkipsLoopWrite(fs, method, value, shadow_kept)) {
        if (Profile::Enabled()) [[unlikely]] {
            Profile::Add(Profile::Counter::SameValueWrites, 1);
        }
        return;
    }
    if (!execution_mask[method]) {
        // Nothing would change on the back: CallMethod and ConsumeSinkImpl skip same-value state
        // writes, and the shadow RAM (tracked by default) holds the value already.
        if (fs.regs[method] == value && shadow_kept && fs.skip_same) {
            if (Profile::Enabled()) [[unlikely]] {
                Profile::Add(Profile::Counter::SameValueWrites, 1);
            }
            return;
        }
        fs.regs[method] = value;
    } else {
        fs.regs[method] = value;
        constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
        switch (method) {
        case MAXWELL3D_REG_INDEX(shadow_ram_control):
            // The back's mode is shadow_state.shadow_ram_control: the shadow copy of this very
            // register, which a later Replay write reads back.
            fs.control = static_cast<Regs::ShadowRamControl>(argument);
            fs.shadow[method] = argument;
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
            GetNativeMacroEngine().ClearCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)]);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.instruction):
            GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
            break;
        case MAXWELL3D_REG_INDEX(load_mme.start_address): {
            const auto positions = GetNativeMacroPositions();
            u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
            if (pointer < positions.size()) {
                positions[pointer] = value;
            }
            ++pointer;
            break;
        }
        case MAXWELL3D_REG_INDEX(falcon[4]):
            fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
            break;
        default:
            if (method >= cb_data && method < cb_data + 16) {
                fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += sizeof(u32);
            } else if (!FrontHandledOutsideDrawManager(method)) {
                fs.draw_pending = true;
            }
            break;
        }
    }
    if (v2_draws) [[unlikely]] {
        FrontDrawPost(split, fs, method, value);
    }
    if (from_macro) {
        // Direct writes consume the sink first.
        split.MacroWrite(this, method, argument);
        fs.sink_pending = false;
    } else {
        split.Method(this, method, argument, is_last_call, nullptr);
        fs.sink_pending = !execution_mask[method];
    }
    if (split.TracksWrites() && fs.v2_note_writes[method]) [[unlikely]] {
        FrontNoteWrites(split, fs, method, 1);
    }
    if (method == MAXWELL3D_REG_INDEX(report_semaphore.query) &&
        VideoCommon::V2::Stats::Enabled()) [[unlikely]] {
        VideoCommon::V2::Stats::NoteReportSemaphore(memory_manager.GetID(), fs.regs.data());
    }
}

void Maxwell3D::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split, u32 method,
                                u32 argument, bool is_last_call) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    FrontWrite(split, fs, method, argument, is_last_call, false);
}

void Maxwell3D::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                     u32 method, const u32* base_start, u32 amount,
                                     u32 methods_pending) {
    auto& fs = Front(system, split);
    if (method >= MacroRegistersStart) {
        if (fs.executing_macro == 0) {
            fs.executing_macro = method;
        }
        fs.parameters.insert(fs.parameters.end(), base_start, base_start + amount);
        if (amount == methods_pending) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    if (fs.s1_exec) {
        FrontMultiS1(split, fs, method, base_start, amount, methods_pending);
        return;
    }
    if (split.DrawRecords()) [[unlikely]] {
        FrontDrawMulti(split, fs, method, base_start, amount);
    }
    if (fs.v2 != nullptr && fs.v2->mode == VideoCommon::V2::Mode::On) [[unlikely]] {
        const u32 old_offset = fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
        FrontMultiRegSem(fs, method, base_start, amount, true);
        split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
        if (fs.s1_check) [[unlikely]] {
            FrontCheckMultiS1(split, fs.regs.data(), method, base_start, amount, old_offset);
        }
        if (split.TracksWrites() && fs.v2_note_writes[method]) [[unlikely]] {
            FrontNoteWrites(split, fs, method, amount);
        }
        return;
    }
    // What CallMultiMethod does to the registers on the back, then one entry for all the words.
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const auto tracked = [&fs] {
        return fs.control == Regs::ShadowRamControl::Track ||
               fs.control == Regs::ShadowRamControl::TrackWithFilter;
    };
    if (method >= cb_data && method < cb_data + 16) {
        fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)] += amount * sizeof(u32);
    } else if (method == MAXWELL3D_REG_INDEX(inline_data)) {
        // Upload data only.
    } else if ((method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0)) &&
               fs.control != Regs::ShadowRamControl::Replay) {
        FrontTrackInstanceLoop(fs, method, 0);
        if (amount != 0) {
            const u32 last = base_start[amount - 1];
            if (tracked()) {
                fs.shadow[method] = last;
            }
            fs.regs[method] = last;
        }
        fs.draw_pending = true;
    } else {
        // CallMethod for each word, macro uploads among them (load_mme.instruction). A word may
        // change the shadow RAM mode for the next (a run of SET_SHADOW_RAM_CONTROL writes).
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = base_start[i];
            u32 value = argument;
            if (method == MAXWELL3D_REG_INDEX(shadow_ram_control) && FrontVerifyEnabled()) [[unlikely]] {
                fs.control_writes[fs.control_count++ % fs.control_writes.size()] = {
                    argument, static_cast<u32>(fs.control), 3U};
            }
            if (tracked()) {
                fs.shadow[method] = argument;
            } else if (fs.control == Regs::ShadowRamControl::Replay) {
                value = fs.shadow[method];
            }
            fs.regs[method] = value;
            FrontTrackInstanceLoop(fs, method, value); // DrawBegin runs for every word
            if (!execution_mask[method]) {
                continue;
            }
            switch (method) {
            case MAXWELL3D_REG_INDEX(shadow_ram_control):
                fs.control = static_cast<Regs::ShadowRamControl>(argument);
                fs.shadow[method] = argument; // The same word on the back (see FrontWrite)
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction_ptr):
                GetNativeMacroEngine().ClearCode(value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.instruction):
                GetNativeMacroEngine().AddCode(fs.regs[MAXWELL3D_REG_INDEX(load_mme.instruction_ptr)], value);
                break;
            case MAXWELL3D_REG_INDEX(load_mme.start_address): {
                const auto positions = GetNativeMacroPositions();
                u32& pointer = fs.regs[MAXWELL3D_REG_INDEX(load_mme.start_address_ptr)];
                if (pointer < positions.size()) {
                    positions[pointer] = value;
                }
                ++pointer;
                break;
            }
            case MAXWELL3D_REG_INDEX(falcon[4]):
                fs.regs[MAXWELL3D_REG_INDEX(shadow_scratch)] = 1;
                break;
            default:
                if (!FrontHandledOutsideDrawManager(method)) {
                    fs.draw_pending = true;
                }
                break;
            }
        }
    }
    fs.sink_pending = false; // The back consumes the sink before a run.
    u32 check_offset = 0;
    if (fs.v2 != nullptr) [[unlikely]] {
        check_offset = fs.v2->mirror->regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
        FrontMultiRegSem(fs, method, base_start, amount, false); // Verify
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, nullptr);
    if (fs.s1_check && fs.v2 != nullptr) [[unlikely]] {
        FrontCheckMultiS1(split, fs.v2->mirror->regs.data(), method, base_start, amount,
                          check_offset);
    }
    if (split.TracksWrites() && fs.v2_note_writes[method]) [[unlikely]] {
        FrontNoteWrites(split, fs, method, amount);
    }
}

void Maxwell3D::FrontNoteWrites(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                u32 amount) {
    namespace V2 = VideoCommon::V2;
    const auto& table = V2::RegSem::GetTable3D();
    const u32 flags = table.flags[method];
    if (table.groups[method] == V2::RegGroup::Shaders) {
        fs.v2_programs_dirty = true;
    }
    if ((flags & V2::RegSem::DrawEnd) != 0 || method == MAXWELL3D_REG_INDEX(clear_surface)) {
        FrontNoteDraw(split, fs);
        return;
    }
    if (method == MAXWELL3D_REG_INDEX(launch_dma) ||
        (flags & V2::RegSem::InlineData) != 0) {
        for (u32 i = 0; i < std::max<u32>(amount, 1); ++i) {
            V2::NoteUploadMethod(split, memory_manager, fs.regs.data(), method, fs.regs[method],
                                 fs.v2_upload);
        }
        return;
    }
    if ((flags & V2::RegSem::CbData) != 0) {
        // The words went to the constant buffer at the offset before they advanced it.
        const GPUVAddr buffer =
            (GPUVAddr{fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_high)]} << 32) |
            fs.regs[MAXWELL3D_REG_INDEX(const_buffer.address_low)];
        const u64 bytes = u64{amount} * sizeof(u32);
        const u32 offset = fs.regs[MAXWELL3D_REG_INDEX(const_buffer.offset)];
        // Coalesced in the split while the words stay contiguous on one GPU page (macros write
        // them one by one): one translation and one table entry per run.
        split.NoteCbWrite(memory_manager, buffer + offset - bytes, bytes);
    } else if ((flags & V2::RegSem::ReportSemaphore) != 0 && amount != 0) {
        V2::NoteReport3D(split, memory_manager, fs.regs.data());
    }
}

void Maxwell3D::FrontNoteDraw(VideoCommon::GpuSplit& split, FrontState& fs) {
    namespace V2 = VideoCommon::V2;
    // The front's registers have the engine's layout.
    const auto& r = *reinterpret_cast<const Regs*>(fs.regs.data());
    // Shader stores: every bound program must be known to store nothing. Render and depth
    // targets are not noted (MASTER_PLAN §5.1): their data reaches guest memory only through
    // texture cache writebacks (S3), and reads of GPU-written pages already take the back.
    // The verdict is cached: recomputed only when what it reads may have changed (B4).
    const bool native = GetNativeMacroEngine().native_turing;
    const u32 inputs = (r.transform_feedback_enabled != 0 ? 1U : 0U) |
                       (r.rasterize_enable != 0 ? 2U : 0U) | (r.IsMeshEnabled() ? 4U : 0U) |
                       (native ? 8U : 0U);
    const u32 generation = V2::ShaderWrites::Generation();
    if (fs.v2_programs_dirty || inputs != fs.v2_programs_state ||
        generation != fs.v2_programs_generation || split.MapEpoch() != fs.v2_programs_map_epoch) {
        fs.v2_programs_dirty = false;
        fs.v2_programs_state = inputs;
        fs.v2_programs_generation = generation;
        fs.v2_programs_map_epoch = split.MapEpoch();
        fs.v2_programs_unbounded = FrontProgramsUnbounded(r, native);
    }
    if (fs.v2_programs_unbounded) {
        split.NoteUnboundedWrite(V2::Producer::Draw);
    }
}

bool Maxwell3D::FrontProgramsUnbounded(const Regs& r, bool native) {
    namespace V2 = VideoCommon::V2;
    bool unbounded = r.transform_feedback_enabled != 0;
    for (size_t index = 0; index < Regs::MaxShaderProgram && !unbounded; ++index) {
        if (!r.IsShaderConfigEnabled(index)) {
            continue;
        }
        const auto type = static_cast<Regs::ShaderType>(index);
        if ((type == Regs::ShaderType::Pixel && !r.rasterize_enable) ||
            (native && type == Regs::ShaderType::Geometry && r.IsMeshEnabled())) {
            continue; // As ShaderCache::RefreshStages: not run, or a header without code
        }
        const auto& config = r.pipelines[index];
        const GPUVAddr address =
            native ? (GPUVAddr{config.reserved[0] & 0xff} << 32) | config.reserved[1]
                   : r.program_region.Address() + config.offset;
        const auto device = memory_manager.GpuToCpuAddress(address);
        const auto summary =
            device ? V2::ShaderWrites::Lookup(*device) : std::optional<V2::ShaderWrites::Summary>{};
        unbounded = !summary || !summary->WritesNothing();
    }
    return unbounded;
}

void Maxwell3D::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    const u32 method = fs.executing_macro;
    fs.executing_macro = 0;
    VideoCommon::V2::Resolve::MultiFront::Note(
        VideoCommon::V2::Resolve::MultiFront::EventKind::MacroCall, method);
    auto& engine = GetNativeMacroEngine();
    const auto positions = GetNativeMacroPositions();
    // Fix F3d: both position tables hold 0x80 entries: a mask, not a division per call.
    static_assert(sizeof(macro_positions) / sizeof(u32) == 0x80 &&
                  sizeof(Tegra::NativeMacroContext::positions) / sizeof(u32) == 0x80);
    const u32 start = positions[((method - MacroRegistersStart) >> 1) & 0x7f];
    VideoCommon::Nx2Profile::CountMethod(method, false);
    namespace MacroHle = VideoCommon::V2::MacroHle;
    const bool dump = VideoCommon::V2::On(VideoCommon::V2::Switch::MacroDump);
    const VideoCommon::V2::Mode hle = VideoCommon::V2::Get(VideoCommon::V2::Switch::MacroHle);
    std::vector<u32> dump_parameters;
    bool dumping = false;
    const MacroHle::Entry* check_entry = nullptr; // MACROHLE=verify: compared after the run
    MacroHle::Identity identity{};
    u64 call_key = 0; // MacroMemo::Hash of the call, made once for the stats and the memo
    if (VideoCommon::V2::Stats::Enabled() && engine.native_turing) [[unlikely]] {
        if (!fs.v2_call_stats) {
            fs.v2_call_stats = std::make_unique<VideoCommon::V2::MacroMemo::CallStats>();
        }
        call_key = VideoCommon::V2::MacroMemo::Hash(start, fs.parameters);
        fs.v2_call_stats->Note("3D", start, fs.parameters, engine.native_code,
                               engine.native_code_generation,
                               fs.v2_memo ? &fs.v2_memo->statistics : nullptr, call_key);
    }
    if ((dump || hle != VideoCommon::V2::Mode::Off) && engine.native_turing) [[unlikely]] {
        if (fs.v2_macro_generation != engine.native_code_generation) {
            fs.v2_macro_generation = engine.native_code_generation;
            fs.v2_macros.clear();
            fs.v2_macro_slots.fill({});
        }
        // ~1M macro calls/s: a direct-mapped slot per start holds the identity, its HLE entry
        // (the table is fixed after loading) and whether the dump already saw it; the map
        // behind it identifies each start once per code generation.
        FrontState::MacroSlot& slot = fs.v2_macro_slots[start % fs.v2_macro_slots.size()];
        if (slot.start != start) [[unlikely]] {
            auto [it, inserted] = fs.v2_macros.try_emplace(start);
            if (inserted) {
                it->second = MacroHle::Identify(engine.native_code, start);
            }
            slot = FrontState::MacroSlot{
                .start = start,
                .identity = it->second,
                .entry = MacroHle::Find(it->second.hash),
                .dump_checked = false,
            };
        }
        identity = slot.identity;
        const MacroHle::Entry* const entry =
            hle != VideoCommon::V2::Mode::Off ? slot.entry : nullptr;
        if (entry != nullptr && entry->kind == MacroHle::Kind::RenderEnableCompare) {
            // F3F5b-4: not only under Passthrough (OoT runs with shadow RAM tracking, the
            // default, so no call was ever HLE'd): under Track the shadow copy of the mode is
            // the back's value, resolved on the back when a Replay sends it (F3F5b-6).
            // Replay makes the macro's writes the shadow's values: the macro runs.
            const bool replay = fs.control == Regs::ShadowRamControl::Replay;
            if (!replay && MacroHle::Applies(*entry, fs.parameters)) {
                if (hle == VideoCommon::V2::Mode::On) {
                    FrontHleRenderEnable(system, split, fs, *entry);
                    return;
                }
                check_entry = entry;
            } else {
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_INFO(HW_GPU,
                             "V2 macro HLE: macro {:016x} runs ({}; {} parameters, p0 {:#x})",
                             identity.hash, replay ? "shadow RAM replay" : "parameters",
                             fs.parameters.size(),
                             fs.parameters.empty() ? 0U : fs.parameters[0]);
                }
            }
        } else if (entry != nullptr) {
            const MacroHle::Draw draw = MacroHle::Resolve(*entry, fs.parameters);
            if (hle == VideoCommon::V2::Mode::On) {
                // The back draws from the GPU buffer; the front reads nothing.
                fs.parameters.clear();
                FrontEmitHostIndirect(system, split, fs, draw);
                return;
            }
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1) < 32) {
                LOG_INFO(HW_GPU,
                         "V2 macro HLE verify: macro {:016x} would draw kind {} topology {} from "
                         "{:#x} (count {:#x}, {} draws, stride {})",
                         identity.hash, static_cast<u32>(draw.kind), draw.topology, draw.address,
                         draw.count_address, draw.max_draws, draw.stride);
            }
        }
        // ShouldDump answers true once per macro, so a slot asks it once.
        if (dump && !slot.dump_checked) {
            slot.dump_checked = true;
            if (MacroHle::ShouldDump("3d", identity.hash)) {
                dumping = true;
                dump_parameters = fs.parameters;
            }
        }
        if (dumping || check_entry != nullptr) {
            fs.v2_observation = std::make_unique<MacroHle::Observation>();
            if (!dumping) {
                dump_parameters = fs.parameters;
            }
        }
    }
    {
        VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
        engine.native_interpreter.active_dram = &native_data_ram;
        const auto parameter_count = fs.parameters.size();
        const auto read = [this, &split, &fs](u32 address) {
            if (address == RenderEnableMode &&
                (fs.v2_render_enable_unknown || fs.regs[address] == ShadowUnknown)) [[unlikely]] {
                FrontSyncRenderEnable(split, fs);
            }
            return address < Regs::NUM_REGS ? fs.regs[address] : 0U;
        };
        const auto write = [&](u32 address, u32 value) {
            if (address >= MacroRegistersStart) {
                return false;
            }
            if (fs.v2_observation && fs.v2_observation->writes.size() < 64) [[unlikely]] {
                fs.v2_observation->writes.emplace_back(address, value);
            }
            // Fix F3d (F3c's test was a second compare in front of Write3D's): under S1 the
            // write path is inlined here and Write3D's compare-on-write drops a write that
            // changes nothing, once. Memo hits leave those writes out before this (F3e).
            if (fs.s1_exec) [[likely]] {
                FrontWriteS1(split, fs, address, value, true, true);
                return true;
            }
            FrontWrite(split, fs, address, value, true, true);
            return true;
        };
        // MME_DMA_READ (into data RAM) and MME_DMA_READ_FIFOED (into the load FIFO).
        const auto dma = [&](u32 address, u32 value) {
            const size_t before = fs.parameters.size();
            FrontDmaRead(split, fs, method, address, value);
            if (fs.v2_observation) [[unlikely]] {
                auto& words = fs.v2_observation->read_words;
                words.insert(words.end(), fs.parameters.begin() + before, fs.parameters.end());
            }
        };
        const auto emit = [&](u32 address, u32 value) {
            if (!write(address, value)) {
                return false;
            }
            if (address == 0x157 || address == 0x158) {
                dma(address, value);
            }
            return true;
        };
        // `dram_hooks`: the memo's JIT recordings of code that touches data RAM (F3d).
        const auto live = [&](auto& live_read, auto& live_emit, auto&... dram_hooks) {
#ifdef ARCHITECTURE_x86_64
            if (!engine.native_jit) {
                engine.native_jit = std::make_shared<TuringMacroJit>();
            }
            engine.native_jit->Prepare(engine.native_code, engine.native_code_generation,
                                       sizeof...(dram_hooks) != 0);
            return engine.native_jit->Execute(start, fs.parameters,
                                              std::span<u32>{native_data_ram}, live_read,
                                              live_emit, dram_hooks...);
#else
            ((void)dram_hooks, ...); // jit_data_ram is off: never called with them
            return engine.native_interpreter.Execute(engine.native_code, start, fs.parameters,
                                                     live_read, live_emit);
#endif
        };
        const VideoCommon::V2::Mode memo = VideoCommon::V2::Get(VideoCommon::V2::Switch::MacroMemo);
        Tegra::TuringMacroInterpreter::Result result;
        if (memo != VideoCommon::V2::Mode::Off && engine.native_turing) [[unlikely]] {
            namespace MacroMemo = VideoCommon::V2::MacroMemo;
            if (!fs.v2_memo) {
                fs.v2_memo = std::make_unique<MacroMemo::Memo>();
            }
            fs.v2_memo->Synchronize(engine.native_code_generation);
            // Registers a write changes besides its own (RegSem::Write3D's effects).
            static constexpr std::array<u32, 3> side_effects{
                MAXWELL3D_REG_INDEX(const_buffer.offset),
                MAXWELL3D_REG_INDEX(load_mme.start_address_ptr),
                MAXWELL3D_REG_INDEX(shadow_scratch),
            };
            const MacroMemo::CallInfo call{
                .start = start,
                .fifo = &fs.parameters,
                .dram = native_data_ram,
                .code = engine.native_code,
                .generation = &engine.native_code_generation,
                .regs = fs.regs.data(),
                .num_regs = Regs::NUM_REGS,
                .side_effects = side_effects,
                .dma_fifoed = 0x158,
                .dma_ram = 0x157,
                .verify = memo == VideoCommon::V2::Mode::Verify,
                // A record that read the mode the back evaluates isn't checked against the
                // front's copy until `read` took the back's.
                .stale_register = fs.v2_render_enable_unknown ||
                                          fs.regs[RenderEnableMode] == ShadowUnknown
                                      ? RenderEnableMode
                                      : ~0U,
                .key = call_key,
                // Fix F3e: a replay leaves out the writes Write3D would drop (S1's
                // FrontWriteS1, skip_same, not under Replay; every write seen when the HLE
                // checks or the dump observe them).
                .skip_same = fs.s1_exec && fs.skip_same && !fs.v2_observation &&
                             fs.control != Regs::ShadowRamControl::Replay,
                .flags = fs.v2_table->flags.data(),
                .exec_mask = VideoCommon::V2::RegSem::Exec,
                .barrier_mask = VideoCommon::V2::RegSem::ShadowControl,
                .shadow = fs.control == Regs::ShadowRamControl::Track ||
                                  fs.control == Regs::ShadowRamControl::TrackWithFilter
                              ? fs.shadow.data()
                              : nullptr,
                .writes_skippable =
                    [](const void* model) {
                        return !static_cast<const VideoCommon::V2::DrawModel*>(model)
                                    ->HasDeferredDraw();
                    },
                .skippable_arg = &fs.v2_draw_model,
                .dropped_writes = &fs.s1_same_writes,
#ifdef ARCHITECTURE_x86_64
                .jit_data_ram = true,
#endif
            };
            result = MacroMemo::Call(*fs.v2_memo, engine.native_interpreter, call, live, read,
                                     write, dma);
            // The writes a hit left out, counted as FrontWriteS1 counts dropped ones.
            fs.s1_writes += fs.v2_memo->last_skipped;
            fs.s1_same_writes += fs.v2_memo->last_skipped;
        } else {
            result = live(read, emit);
        }
        if (!result.success) {
            engine.native_execution_failed = true;
            LOG_ERROR(HW_GPU, "Native MME (split front) failed: start={:#x}, consumed={}/{}, error={}",
                      start, result.consumed, parameter_count, result.error);
        }
    }
    if (fs.v2_observation) [[unlikely]] {
        if (check_entry != nullptr) {
            FrontCheckHle(*check_entry, identity.hash, dump_parameters, *fs.v2_observation);
        }
        if (dumping) {
            MacroHle::Dump("3d", method, start, identity, engine.native_code, dump_parameters,
                           *fs.v2_observation);
        }
        fs.v2_observation.reset();
    }
    fs.parameters.clear();
    FrontEndMacro(system, split, fs);
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

bool Maxwell3D::FrontEndMacro(Core::System& system, VideoCommon::GpuSplit& split,
                              FrontState& fs) {
    if (fs.v2 != nullptr && fs.v2->mirror) [[unlikely]] {
        fs.v2->mirror->draw_pending = false;
    }
    if (!fs.draw_pending) {
        return false;
    }
    fs.draw_pending = false;
    FrontEmitMacroEnd(system, split, fs);
    return true;
}

void Maxwell3D::RunMacroHleDraw(Core::System& system, const VideoCommon::V2::MacroHle::Draw& draw) {
    // As Switch 1's HLE_DrawArraysIndirect / HLE_DrawIndexedIndirect /
    // HLE_MultiDrawIndexedIndirectCount: the draw manager reads the arguments from the GPU
    // buffer at `draw.address` (host indirect draw), the count from `draw.count_address`.
    using Kind = VideoCommon::V2::MacroHle::Kind;
    ConsumeSink(system);
    const auto topology = static_cast<Regs::PrimitiveTopology>(draw.topology);
    const bool indexed = draw.kind != Kind::DrawArraysIndirect;
    const u32 words = indexed ? 5 : 4;
    auto& params = draw_manager.indirect_state;
    params.is_byte_count = false;
    params.is_indexed = indexed;
    params.include_count = draw.kind == Kind::MultiDrawIndexedIndirectCount;
    params.count_start_address = draw.count_address;
    params.indirect_start_address = draw.address;
    params.max_draw_counts = draw.max_draws;
    params.stride = draw.stride;
    params.buffer_size = (draw.stride != 0 ? draw.stride : words * sizeof(u32)) * draw.max_draws;
    if (draw.base_vertex_attribute) {
        engine_state = EngineHint::OnHLEMacro;
        SetHLEReplacementAttributeType(0, 0x640, indexed ? HLEReplacementAttributeType::BaseVertex
                                                         : HLEReplacementAttributeType::BaseInstance);
        if (indexed) {
            SetHLEReplacementAttributeType(0, 0x644, HLEReplacementAttributeType::BaseInstance);
        }
    }
    if (indexed) {
        dirty.flags[VideoCommon::Dirty::IndexBuffer] = true;
        draw_manager.DrawIndexedIndirect(*this, topology, 0,
                                         static_cast<u32>(EstimateIndexBufferSize()));
    } else {
        draw_manager.DrawArrayIndirect(*this, topology);
    }
    if (draw.base_vertex_attribute) {
        engine_state = EngineHint::None;
        replace_table.clear();
    }
}

VideoCommon::V2::GroupTracker* Maxwell3D::FrontDrawGroups(FrontState& fs) {
    if (fs.v2 == nullptr) {
        return nullptr;
    }
    if (fs.v2->mode == VideoCommon::V2::Mode::On) {
        // The hashes as of the registers now (deferred per write, HashLog3D).
        fs.v2_hash_log.Fold(fs.v2_groups, *fs.v2_table, fs.regs.data());
        return &fs.v2_groups;
    }
    return fs.v2->mirror ? &fs.v2->mirror->v2_groups : nullptr;
}

void Maxwell3D::FrontFillDraw(FrontState& fs, VideoCommon::V2::Ir::DrawBody& body, u32 method,
                              u32 value, const VideoCommon::V2::DrawOutput& out, u32 first,
                              const VideoCommon::V2::GroupTracker* hashes) {
    namespace V2 = VideoCommon::V2;
    body.method = method;
    body.argument = value;
    body.num_draws =
        static_cast<u8>(std::min<u32>(out.count - first, V2::Ir::MaxPredictedDraws));
    for (u32 i = 0; i < body.num_draws; ++i) {
        body.draws[i] = out.draws[first + i];
    }
    // The register groups changed since the previous Draw record, and their hashes as the back
    // will have them at the draw (`hashes` when the mirror's are not those yet).
    V2::GroupTracker* const groups = FrontDrawGroups(fs);
    if (groups == nullptr || !groups->valid) {
        return;
    }
    const V2::GroupTracker& source = hashes != nullptr ? *hashes : *groups;
    body.flags |= V2::Ir::DrawHasGroups;
    body.dirty_groups = groups->dirty;
    size_t slot = 0;
    for (u32 bits = V2::HashedGroups; bits != 0; bits &= bits - 1) {
        body.group_hashes[slot++] = source.hashes[std::countr_zero(bits)];
    }
    body.fingerprint = source.Fingerprint(V2::HashedGroups);
    groups->BeginDraw();
}

VideoCommon::V2::Ir::CompactBody Maxwell3D::FrontCompactDraw(VideoCommon::GpuSplit& split,
                                                              FrontState& fs,
                                                              VideoCommon::V2::Ir::DrawOp op,
                                                              u32 count) {
    namespace V2 = VideoCommon::V2;
    V2::Ir::CompactBody body{};
    body.full_kind = V2::Ir::Kind::Draw;
    body.op = static_cast<u8>(op);
    body.count = static_cast<u16>(std::min<u32>(count, 0xffff));
    body.engine = split.Ref(this);
    if (V2::GroupTracker* const groups = FrontDrawGroups(fs); groups != nullptr && groups->valid) {
        body.dirty_groups = groups->dirty;
        body.fingerprint = groups->Fingerprint(V2::HashedGroups);
        groups->BeginDraw();
    }
    return body;
}

void Maxwell3D::FrontEmitDraws(VideoCommon::GpuSplit& split, FrontState& fs, u32 method, u32 value,
                               const VideoCommon::V2::DrawOutput& out,
                               const VideoCommon::V2::GroupTracker* hashes) {
    if (out.count == 0) {
        return;
    }
    if (fs.s1_exec) {
        FrontS1Draws(split, fs, out); // Draw entries in the classified run the back executes
        return;
    }
    if (!split.FullDrawRecords()) {
        // Nothing reads the predictions: one compact record per trigger (MASTER_PLAN §5.2).
        split.Note(VideoCommon::V2::Ir::Kind::Compact,
                   FrontCompactDraw(split, fs, VideoCommon::V2::Ir::DrawOp::Predicted, out.count));
        return;
    }
    // Draw records go before the method that makes the back draw (it consumes them in order).
    for (u32 first = 0; first < out.count; first += VideoCommon::V2::Ir::MaxPredictedDraws) {
        VideoCommon::V2::Ir::DrawBody body{};
        body.op = VideoCommon::V2::Ir::DrawOp::Predicted;
        body.engine = split.Ref(this);
        FrontFillDraw(fs, body, method, value, out, first, hashes);
        split.Note(VideoCommon::V2::Ir::Kind::Draw, body);
    }
}

void Maxwell3D::FrontParallelDraw(VideoCommon::GpuSplit& split, FrontState& fs,
                                  const VideoCommon::V2::DrawOutput& out, u32 pending_method,
                                  bool may_cut) {
    namespace Resolve = VideoCommon::V2::Resolve;
    Resolve::SplitResolve* const parallel = split.Parallel();
    if (out.count == 0 || fs.v2 == nullptr || fs.v2->mode != VideoCommon::V2::Mode::On) {
        return;
    }
    fs.v2_hash_log.Fold(fs.v2_groups, *fs.v2_table, fs.regs.data()); // the snapshot's hashes
    const Resolve::SplitResolve::FrontView view{
        .regs = fs.regs.data(),
        .shadow = fs.shadow.data(),
        .groups = &fs.v2_groups,
        .cbufs = &fs.v2_cbufs,
        .control = static_cast<u32>(fs.control),
        .instance_loop = fs.instance_loop,
        .sink_pending = fs.sink_pending,
        .draw_pending = fs.draw_pending,
        .dirty_blocks = &fs.v2_dirty_blocks,
        .deltas = fs.s1_exec,
    };
    const Resolve::SplitResolve::EngineInfo engine{
        .ref = split.Ref(this),
        .engine = reinterpret_cast<uintptr_t>(this),
        .memory = reinterpret_cast<uintptr_t>(&memory_manager),
        .native_turing = GetNativeMacroEngine().native_turing,
    };
    parallel->FrontDraw(view, engine, out, pending_method, may_cut);
}

void Maxwell3D::FrontDrawPre(VideoCommon::GpuSplit& split, FrontState& fs, u32 method, u32 value) {
    // A write the back stores (ProcessDirtyRegisters) ends a deferred instanced batch first.
    if (!fs.v2_draw_model.HasDeferredDraw() ||
        VideoCommon::V2::DrawModel::IsInstanceLoopMethod(method)) {
        return;
    }
    if (!execution_mask[method] && fs.back_skip_same && fs.regs[method] == value) {
        return;
    }
    VideoCommon::V2::DrawOutput out;
    fs.v2_draw_model.OnStore(method, fs.regs.data(), out);
    if (split.Parallel() != nullptr) [[unlikely]] {
        FrontParallelDraw(split, fs, out, ~0U, true);
    }
    FrontEmitDraws(split, fs, method, value, out, nullptr);
}

void Maxwell3D::FrontDrawPost(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                              u32 value) {
    namespace V2 = VideoCommon::V2;
    if (!execution_mask[method] || !V2::DrawModel::IsDrawMethod(method)) {
        return;
    }
    V2::DrawOutput out;
    fs.v2_draw_model.OnExecute(method, value, fs.regs.data(), out);
    if (out.count == 0) {
        return;
    }
    // REGSEM=verify: the table copy takes this write only after the hand-written mirror.
    V2::GroupTracker pending;
    const V2::GroupTracker* hashes = nullptr;
    if (fs.v2 != nullptr && fs.v2->mode == V2::Mode::Verify && fs.v2->mirror) {
        pending = fs.v2->mirror->v2_groups;
        const u32 old_value = fs.v2->mirror->regs[method];
        if (old_value != value) {
            pending.Change(V2::RegSem::GetTable3D().groups[method], method, old_value, value);
        }
        hashes = &pending;
    }
    if (split.Parallel() != nullptr) [[unlikely]] {
        // The mirror holds this write already; the ring gets it after the records.
        FrontParallelDraw(split, fs, out, method, true);
    }
    FrontEmitDraws(split, fs, method, value, out, hashes);
}

void Maxwell3D::FrontDrawMulti(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                               const u32* base_start, u32 amount) {
    // As CallMultiMethod on the back: a run of a method outside instance loops ends a deferred
    // batch first, inline index runs append, other runs are a CallMethod per word.
    namespace V2 = VideoCommon::V2;
    V2::DrawModel& model = fs.v2_draw_model;
    V2::DrawOutput out;
    if (!V2::DrawModel::IsInstanceLoopMethod(method)) {
        model.OnMultiStart(method, fs.regs.data(), out);
        if (split.Parallel() != nullptr) [[unlikely]] {
            FrontParallelDraw(split, fs, out, ~0U, true);
        }
        FrontEmitDraws(split, fs, method, amount != 0 ? base_start[0] : 0, out, nullptr);
    }
    const bool replay = fs.control == Regs::ShadowRamControl::Replay;
    constexpr u32 cb_data = MAXWELL3D_REG_INDEX(const_buffer.buffer);
    const bool inline_index = method == MAXWELL3D_REG_INDEX(draw_inline_index) ||
                              method == MAXWELL3D_REG_INDEX(inline_index_2x16.even) ||
                              method == MAXWELL3D_REG_INDEX(inline_index_4x8.index0);
    if ((method >= cb_data && method < cb_data + 16) ||
        method == MAXWELL3D_REG_INDEX(inline_data)) {
        return;
    }
    if (inline_index && !replay) {
        model.OnInlineIndex(method, amount);
        return;
    }
    if (!V2::DrawModel::IsDrawMethod(method) && !model.HasDeferredDraw()) {
        return;
    }
    // Word by word, with the hashes as the back has them at each draw.
    V2::GroupTracker* const groups = FrontDrawGroups(fs);
    V2::GroupTracker hashes = groups != nullptr ? *groups : V2::GroupTracker{};
    const auto group = V2::RegSem::GetTable3D().groups[method];
    const u32 saved = fs.regs[method];
    for (u32 i = 0; i < amount; ++i) {
        const u32 value = replay ? fs.shadow[method] : base_start[i];
        out.count = 0;
        if (model.HasDeferredDraw() && !V2::DrawModel::IsInstanceLoopMethod(method) &&
            (execution_mask[method] || !fs.back_skip_same || fs.regs[method] != value)) {
            model.OnStore(method, fs.regs.data(), out);
            if (split.Parallel() != nullptr) [[unlikely]] {
                FrontParallelDraw(split, fs, out, ~0U, false);
            }
            FrontEmitDraws(split, fs, method, value, out, &hashes);
            out.count = 0;
        }
        if (fs.regs[method] != value) {
            hashes.Change(group, method, fs.regs[method], value);
        }
        fs.regs[method] = value;
        if (execution_mask[method]) {
            model.OnExecute(method, value, fs.regs.data(), out);
            if (split.Parallel() != nullptr) [[unlikely]] {
                FrontParallelDraw(split, fs, out, ~0U, false);
            }
            FrontEmitDraws(split, fs, method, value, out, &hashes);
        }
    }
    fs.regs[method] = saved; // The mirror applies the run itself after this.
}

void Maxwell3D::FrontEmitMacroEnd(Core::System& system, VideoCommon::GpuSplit& split,
                                  FrontState& fs) {
    // The back consumes its sink and issues the deferred instanced draw (FinishFrontMacro).
    if (fs.s1_exec) {
        // S1: the deferred batch as Draw entries; the back has no sink or state machine.
        VideoCommon::V2::DrawOutput out;
        fs.v2_draw_model.OnMacroEnd(fs.regs.data(), out);
        if (split.Parallel() != nullptr) [[unlikely]] {
            FrontParallelDraw(split, fs, out, ~0U, true);
        }
        FrontS1Draws(split, fs, out);
        return;
    }
    if (!split.FullDrawRecords()) [[likely]] {
        VideoCommon::V2::Ir::CompactBody compact{};
        if (split.DrawRecords()) {
            VideoCommon::V2::DrawOutput out;
            fs.v2_draw_model.OnMacroEnd(fs.regs.data(), out);
            compact = FrontCompactDraw(split, fs, VideoCommon::V2::Ir::DrawOp::MacroEnd, out.count);
        } else {
            compact.full_kind = VideoCommon::V2::Ir::Kind::Draw;
            compact.op = static_cast<u8>(VideoCommon::V2::Ir::DrawOp::MacroEnd);
            compact.engine = split.Ref(this);
        }
        split.Emit(VideoCommon::V2::Ir::Kind::Compact, compact,
                   [this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
                   VideoCommon::GpuSplit::FastDrawNeutral);
        return;
    }
    VideoCommon::V2::Ir::DrawBody body{};
    body.op = VideoCommon::V2::Ir::DrawOp::MacroEnd;
    if (split.DrawRecords()) [[unlikely]] {
        VideoCommon::V2::DrawOutput out;
        fs.v2_draw_model.OnMacroEnd(fs.regs.data(), out);
        FrontFillDraw(fs, body, 0, 0, out, 0, nullptr);
    }
    body.engine = split.Ref(this);
    split.Emit(VideoCommon::V2::Ir::Kind::Draw, body,
               [this, system_ptr = &system] { FinishFrontMacro(*system_ptr); },
               VideoCommon::GpuSplit::FastDrawNeutral);
}

void Maxwell3D::FrontEmitHostIndirect(Core::System& system, VideoCommon::GpuSplit& split,
                                      FrontState& fs, const VideoCommon::V2::MacroHle::Draw& draw) {
    // NVN2 macro HLE (EDEN_NX2_V2_MACROHLE): the back draws from the GPU buffer.
    if (fs.s1_exec) {
        FrontS1Groups(split, fs); // what the draw sees changed, before its record
    }
    VideoCommon::V2::Ir::DrawBody body{};
    body.op = VideoCommon::V2::Ir::DrawOp::HostIndirect;
    body.engine = split.Ref(this);
    body.hle = {
        .kind = static_cast<u32>(draw.kind),
        .topology = draw.topology,
        .address = draw.address,
        .count_address = draw.count_address,
        .max_draws = draw.max_draws,
        .stride = draw.stride,
        .base_vertex_attribute = draw.base_vertex_attribute ? 1U : 0U,
        .reserved = 0,
    };
    if (split.DrawRecords()) [[unlikely]] {
        VideoCommon::V2::DrawOutput out;
        fs.v2_draw_model.OnIndirect(draw.kind != VideoCommon::V2::MacroHle::Kind::DrawArraysIndirect,
                                    draw.topology, fs.regs.data(), out);
        FrontFillDraw(fs, body, 0, 0, out, 0, nullptr);
    }
    body.engine = split.Ref(this);
    split.Emit(VideoCommon::V2::Ir::Kind::Draw, body, [this, system_ptr = &system, draw] {
        RunMacroHleDraw(*system_ptr, draw);
    });
}

void Maxwell3D::FrontHleRenderEnable(Core::System& system, VideoCommon::GpuSplit& split,
                                     FrontState& fs, const VideoCommon::V2::MacroHle::Entry& entry) {
    // Fix F5-03/04: the macro's writes to the MME DMA source registers (and the DMA count),
    // without the read: the back reads the report pair where its writers are done (class C/D
    // read moved off the front) and writes the render enable mode the macro would have.
    for (u32 i = 0; i < entry.num_writes; ++i) {
        FrontWrite(split, fs, entry.writes[i].method, entry.writes[i].value.Get(fs.parameters),
                   true, true);
    }
    const GPUVAddr source = (GPUVAddr{fs.regs[0x154] & 0x1ffffff} << 32) | fs.regs[0x155];
    const bool invert = entry.invert.Get(fs.parameters) != 0;
    const u32 bytes = entry.words * sizeof(u32);
    const bool s1 = fs.s1_exec;
    const bool tracked = fs.control == Regs::ShadowRamControl::Track ||
                         fs.control == Regs::ShadowRamControl::TrackWithFilter;
    fs.parameters.clear();
    // The mode is the back's from now on (F3F5b-4): read back when the front needs it.
    fs.v2_render_enable_unknown = true;
    if (tracked) {
        fs.shadow[RenderEnableMode] = ShadowUnknown;
    }
    const VideoCommon::V2::Ir::ReadbackBody body{
        .op = VideoCommon::V2::Ir::ReadbackOp::MacroRenderEnable,
        .object = split.Ref(this),
        .object2 = VideoCommon::V2::Ir::NoObject,
        .cache = (invert ? 1U : 0U) | (s1 ? 2U : 0U) | (tracked ? 4U : 0U),
        .address = source,
        .bytes = bytes,
    };
    split.Emit(VideoCommon::V2::Ir::Kind::Readback, body,
               [this, system_ptr = &system, source, bytes, invert, s1, tracked] {
                   RunMacroHleRenderEnable(*system_ptr, source, bytes, invert, s1, tracked);
               });
    static u64 calls{}; // Front thread only
    if ((++calls & 0xfffff) == 1) {
        LOG_INFO(HW_GPU, "V2 macro HLE: {} conditional render calls evaluated on the back", calls);
    }
    FrontEndMacro(system, split, fs);
    if (FrontVerifyEnabled() && (++fs.calls & 4095) == 0) [[unlikely]] {
        FrontVerify(split, fs);
    }
}

void Maxwell3D::RunMacroHleRenderEnable(Core::System& system, GPUVAddr source, u32 bytes,
                                        bool invert, bool s1, bool tracked) {
    // The macro's DMA read with the flushes the v1 macro made, here where everything before it
    // in command order ran on the back.
    VideoCommon::V2::IrExec::MacroDmaFlush(memory_manager, *rasterizer, source, bytes);
    std::array<u32, 6> words{};
    memory_manager.ReadBlockUnsafe(source, words.data(),
                                   std::min<size_t>(bytes, sizeof(words)));
    const u32 mode = VideoCommon::V2::MacroHle::RenderEnableValue(invert, words);
    constexpr u32 method = MAXWELL3D_REG_INDEX(render_enable.mode);
    if (tracked) {
        v2_hle_tracked_mode = mode; // The front's shadow copy (ProcessQueryCondition)
    }
    if (s1) {
        // As an S1 Exec entry of the macro's write.
        namespace S1 = VideoCommon::V2::S1;
        const u64 entry = S1::MakeRegister(S1::Tag::Exec, method, mode, S1::IsLast);
        ExecuteS1Run(&entry, &entry + 1);
        return;
    }
    // As the macro's write on the word stream (GpuSplit::ExecuteMethod3D, Direct).
    ConsumeSink(system);
    CallMethod(system, method, mode, true);
}

void Maxwell3D::FrontCheckHle(const VideoCommon::V2::MacroHle::Entry& entry, u64 hash,
                              std::span<const u32> parameters,
                              const VideoCommon::V2::MacroHle::Observation& observation) {
    static std::atomic<u64> checks{};
    static std::atomic<u64> mismatches{};
    const auto mismatch = VideoCommon::V2::MacroHle::CheckRenderEnable(
        entry, parameters, observation.writes, observation.read_words);
    const u64 checked = checks.fetch_add(1) + 1;
    if (mismatch) {
        if (mismatches.fetch_add(1) < 32) {
            LOG_WARNING(HW_GPU, "V2 macro HLE verify: macro {:016x}: {}", hash, *mismatch);
        }
    }
    if ((checked & 0xffff) == 1) {
        LOG_INFO(HW_GPU, "V2 macro HLE verify: {} calls checked, {} mismatches", checked,
                 mismatches.load());
    }
}

void Maxwell3D::FrontSyncRenderEnable(VideoCommon::GpuSplit& split, FrontState& fs) {
    // The back ran every evaluation sent: its register is the macro's value (or a later write's).
    split.Drain(VideoCommon::DrainSite::RenderEnable);
    fs.v2_render_enable_unknown = false;
    const u32 value = regs.reg_array[RenderEnableMode];
    if (fs.v2_on) {
        // The groups hashed the front's old value: change it as a write would.
        VideoCommon::V2::RegSem::Store3D(fs, *fs.v2_table, RenderEnableMode, value);
    } else {
        fs.regs[RenderEnableMode] = value;
        if (fs.v2 != nullptr) {
            FrontRegSemResync(fs); // REGSEM=verify: the mirror and groups follow
        }
    }
    // Each one drains: a macro that reads the mode every call would make this costly.
    static u64 syncs{}; // Front thread only
    if ((++syncs & 1023) == 1) {
        LOG_INFO(HW_GPU, "V2 macro HLE: an MME read of render_enable.mode took the back's ({} so far)",
                 syncs);
    }
}

u32 Maxwell3D::FrontRegisterForTest(u32 method) const {
    return front ? front->regs[method] : 0;
}

u32 Maxwell3D::FrontShadowForTest(u32 method) const {
    return front ? front->shadow[method] : 0;
}

bool Maxwell3D::FrontRenderEnableUnknownForTest() const {
    return front && front->v2_render_enable_unknown;
}

void Maxwell3D::FrontTrackInstanceLoop(FrontState& fs, u32 method, u32 value) {
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.begin):
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        if (((value >> 26) & 3) == 0) {
            fs.instance_loop = false;
        } else if (((value >> 26) & 3) == 1) {
            fs.instance_loop = true;
        }
        break;
    case MAXWELL3D_REG_INDEX(draw_inline_index):
    case MAXWELL3D_REG_INDEX(inline_index_2x16.even):
    case MAXWELL3D_REG_INDEX(inline_index_4x8.index0):
        fs.instance_loop = false; // DrawMode::InlineIndex: draw.end draws
        break;
    default:
        break;
    }
}

bool Maxwell3D::FrontSkipsLoopWrite(FrontState& fs, u32 method, u32 value, bool shadow_kept) {
    FrontTrackInstanceLoop(fs, method, value);
    switch (method) {
    case MAXWELL3D_REG_INDEX(draw.end):
    case MAXWELL3D_REG_INDEX(vertex_buffer.first):
    case MAXWELL3D_REG_INDEX(vertex_buffer.count):
        break;
    default:
        return false;
    }
    // Sent while state writes wait in the back's sink: it consumes them at the next executable
    // method, and a changed one ends a deferred draw that must issue before the next engine's work
    // (a compute dispatch after a dropped vertex_buffer.count would otherwise run first).
    if (!fs.skip_same || fs.sink_pending || !shadow_kept || fs.regs[method] != value) {
        return false;
    }
    // Instance loop methods flush no deferred draw, and the draw manager ignores these: DrawEnd
    // draws nothing in an instance loop, and it only reads the vertex buffer first and count
    // (index_buffer writes also mark the index buffer dirty, so they are always sent).
    return method != MAXWELL3D_REG_INDEX(draw.end) || fs.instance_loop;
}

void Maxwell3D::FrontDmaRead(VideoCommon::GpuSplit& split, FrontState& fs, u32 macro,
                             u32 method, u32 count) {
    // The words may be written by methods still queued for the back (indirect draw arguments
    // from a culling dispatch), and the caches that may hold newer ones belong to it.
    const GPUVAddr source = (GPUVAddr{fs.regs[0x154] & 0x1ffffff} << 32) | fs.regs[0x155];
    const u32 ram = fs.regs[0x156];
    const u32 words = std::min<u32>(count, 0x1000);
    const size_t read_bytes = size_t{words} * sizeof(u32);
    std::vector<u32> diag_before;
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        diag_before.resize(words);
        memory_manager.ReadBlockUnsafe(source, diag_before.data(), read_bytes);
    }
    VideoCommon::Nx2Profile::CountDmaRead(macro);
    if (fs.v2_observation) [[unlikely]] {
        fs.v2_observation->dma_reads.emplace_back(source, words);
        for (u32 i = 0; i < words && fs.v2_observation->dma_words.size() < 64; ++i) {
            fs.v2_observation->dma_words.push_back(
                memory_manager.Read<u32>(source + u64{i} * sizeof(u32)));
        }
    }
    const VideoCommon::V2::Mode hazard_mode = split.HazardMode();
    std::vector<u32> hazard_words;
    bool hazard_read = false;
    if (hazard_mode != VideoCommon::V2::Mode::Off && !Tegra::Nx2CmdPageDiag::g_enabled &&
        !VideoCommon::QueryDma::TraceEnabled()) [[unlikely]] {
        // EDEN_NX2_V2_HAZARD: wait only for the queued writes the words depend on.
        hazard_words.resize(words);
        hazard_read = FrontDmaReadHazard(split, source, read_bytes, hazard_words);
        if (hazard_mode == VideoCommon::V2::Mode::On) {
            if (!hazard_read) {
                // One wait instead of up to three: the flushes run on the back after
                // everything queued, then the front reads.
                const VideoCommon::V2::Ir::ReadbackBody body{
                    .op = VideoCommon::V2::Ir::ReadbackOp::MacroDmaFlush,
                    .object = split.Ref(&memory_manager),
                    .object2 = split.Ref(rasterizer),
                    .cache = 0,
                    .address = source,
                    .bytes = read_bytes,
                };
                split.Emit(VideoCommon::V2::Ir::Kind::Readback, body,
                           [this, memory = &memory_manager, source, read_bytes] {
                    VideoCommon::V2::IrExec::MacroDmaFlush(*memory, *rasterizer, source,
                                                           read_bytes);
                });
                split.Drain(VideoCommon::DrainSite::MacroDmaRead);
                for (u32 i = 0; i < words; ++i) {
                    hazard_words[i] = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
                }
            }
            for (u32 i = 0; i < words; ++i) {
                if (method == 0x158) {
                    fs.parameters.push_back(hazard_words[i]);
                } else if (ram + i < native_data_ram.size()) {
                    native_data_ram[ram + i] = hazard_words[i];
                }
            }
            return;
        }
    }
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    if (VideoCommon::QueryDma::TraceEnabled()) {
        split.Run(
            [this, macro, method, source, read_bytes] {
                if (VideoCommon::QueryDma::HasPendingQueries(memory_manager, *rasterizer, source,
                                                             read_bytes)) {
                    VideoCommon::QueryDma::Trace(memory_manager, *rasterizer, macro, method,
                                                 source, read_bytes);
                }
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    if (VideoCommon::Nx2Profile::Enabled()) {
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::DmaReadBytes, read_bytes);
    }
    if (memory_manager.IsMemoryDirty(source, read_bytes, VideoCommon::CacheType::BufferCache)) {
        const VideoCommon::V2::Ir::ReadbackBody body{
            .op = VideoCommon::V2::Ir::ReadbackOp::MemoryFlush,
            .object = split.Ref(&memory_manager),
            .address = source,
            .bytes = read_bytes,
        };
        split.RunTyped(
            VideoCommon::V2::Ir::Kind::Readback, body,
            [memory = &memory_manager, source, read_bytes] {
                memory->FlushRegion(source, read_bytes, VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::MacroDmaFlush);
    }
    // Write out the reports held in the query cache, or the read returns stale occlusion counts.
    const VideoCommon::V2::Ir::ReadbackBody query_body{
        .op = VideoCommon::V2::Ir::ReadbackOp::QueryFlush,
        .object = split.Ref(&memory_manager),
        .object2 = split.Ref(rasterizer),
        .cache = 0,
        .address = source,
        .bytes = read_bytes,
    };
    split.RunTyped(
        VideoCommon::V2::Ir::Kind::Readback, query_body,
        [this, source, read_bytes] {
            VideoCommon::QueryDma::FlushPending(memory_manager, *rasterizer, source, read_bytes);
        },
        VideoCommon::DrainSite::MacroDmaFlush);
    u32 changed{};
    for (u32 i = 0; i < words; ++i) {
        const u32 word = memory_manager.Read<u32>(source + u64{i} * sizeof(u32));
        if (method == 0x158) {
            fs.parameters.push_back(word);
        } else if (ram + i < native_data_ram.size()) {
            native_data_ram[ram + i] = word;
        }
        if (!diag_before.empty() && diag_before[i] != word) {
            ++changed;
        }
        if (hazard_read && hazard_words[i] != word) [[unlikely]] {
            // EDEN_NX2_V2_HAZARD=verify: the read without the drain disagrees.
            VideoCommon::V2::Stats::NoteMismatch(VideoCommon::DrainSite::MacroDmaRead);
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1) < 32) {
                LOG_WARNING(HW_GPU,
                            "V2 hazard verify: macro {:#x} DMA read {:#x} word {} read {:#x} "
                            "without the drain, {:#x} after it",
                            macro, source, i, hazard_words[i], word);
            }
            hazard_read = false;
        }
    }
    if (Tegra::Nx2CmdPageDiag::g_enabled) {
        bool gpu_written = false;
        for (GPUVAddr page = source & ~GPUVAddr{0xfff}; page < source + read_bytes; page += 0x1000) {
            const auto device = memory_manager.GpuToCpuAddress(page);
            gpu_written |= device && Tegra::Nx2IsGpuWritten(*device);
        }
        Tegra::Nx2CmdPageDiag::RecordDmaRead(source, words, changed, gpu_written);
    }
}

bool Maxwell3D::FrontDmaReadHazard(VideoCommon::GpuSplit& split, GPUVAddr source, size_t bytes,
                                   std::span<u32> out) {
    namespace V2 = VideoCommon::V2;
    using V2::Stats::Decision;
    using V2::Stats::Reason;
    constexpr auto site = VideoCommon::DrainSite::MacroDmaRead;
    const auto fall_back = [](Reason reason) {
        V2::Stats::NoteDecision(site, Decision::Waited, reason);
        return false;
    };
    // The words in device memory, in one piece.
    const auto device = memory_manager.GpuToCpuAddress(source);
    if (!device || bytes == 0) {
        return fall_back(Reason::Unmapped);
    }
    for (GPUVAddr page = (source & ~GPUVAddr{0xfff}) + 0x1000; page < source + bytes;
         page += 0x1000) {
        const auto page_device = memory_manager.GpuToCpuAddress(page);
        if (!page_device || *page_device != *device + (page - source)) {
            return fall_back(Reason::Unmapped);
        }
    }
    // Fix F3-08 (MASTER_PLAN §6): no drain. Counters and timestamps still to land (class D,
    // strict) are waited for by landing ticket; words the buffer cache may hold newer than
    // memory (class C) are read back behind their writers, through the readback path.
    // What can have written the words in command order (7429fba04e): methods that write memory
    // (ranged, or unbounded: dispatches, copies, uploads) and the work before the last
    // host-level wait. Draws after it are not visible to the 3D engine's reads before a wait.
    V2::HazardFilter filter;
    filter.ranged = V2::AllProducers;
    filter.unbounded = V2::ProducerBit(V2::Producer::Dispatch) |
                       V2::ProducerBit(V2::Producer::Copy) |
                       V2::ProducerBit(V2::Producer::Upload);
    // Draws: their shader stores, those sent before the last host-level wait.
    filter.unbounded_synced = V2::ProducerBit(V2::Producer::Draw);
    bool narrowed = false;
    bool read_back = false;
    bool reports_flushed = false;
    for (int attempt = 0; attempt < 4; ++attempt) {
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = memory_manager.Read<u32>(source + i * sizeof(u32));
        }
        // Releases still to land are applied, then the queued writes (GpuSplit::FindWrites).
        // Once the range's reports were written out, memory holds what they leave.
        const V2::HazardResult result =
            reports_flushed ? split.FindQueuedWrites(*device, out.size() * sizeof(u32), filter,
                                                     reinterpret_cast<u8*>(out.data()))
                            : split.FindWrites(*device, out.size() * sizeof(u32), filter,
                                               reinterpret_cast<u8*>(out.data()));
        if (result.kind == V2::HazardResult::Kind::Wait && result.landing) {
            // A counter or timestamp only its operation writes, and that operation waits for a
            // fence an idle back never makes (SESSION_HANDOFF §5g): as the old path's deferred
            // QueryFlush (Perf fix A item 9), the back writes the range's reports out itself,
            // after the entry it runs, submitting and waiting as it needs to.
            if (!FrontFlushReports(split, source, bytes, result.position)) {
                return fall_back(Reason::Unlanded);
            }
            reports_flushed = true;
            narrowed = true;
            continue;
        }
        const u64 wait = result.kind == V2::HazardResult::Kind::Wait ? result.position : 0;
        if (wait == 0) {
            if (!read_back && Nx2MayBeGpuWritten(*device, bytes)) {
                // Every writer the table knows has run: read the words back behind them.
                FrontReadBack(split, *device, bytes);
                read_back = true;
                narrowed = true;
                continue; // Memory holds them now; the queued writes still go on top
            }
            const bool forwarded = result.kind == V2::HazardResult::Kind::Forward;
            V2::Stats::NoteDecision(site,
                                    narrowed    ? Decision::Narrowed
                                    : forwarded ? Decision::Forwarded
                                                : Decision::Avoided,
                                    read_back ? Reason::GpuWritten : Reason::None);
            return true;
        }
        if (attempt == 3) {
            break;
        }
        V2::Stats::NoteNarrowedBy(site, result.producers);
        split.WaitForPosition(wait, site);
        narrowed = true;
    }
    return fall_back(Reason::Overflow);
}

bool Maxwell3D::FrontFlushReports(VideoCommon::GpuSplit& split, GPUVAddr source, size_t bytes,
                                  u64 ticket) {
    struct Job {
        Tegra::MemoryManager* memory{};
        VideoCore::RasterizerInterface* rasterizer{};
        GPUVAddr source{};
        size_t bytes{};
    };
    static Job job; // Front thread only, one at a time (CompleteLanding waits for it)
    job = {&memory_manager, rasterizer, source, bytes};
    return split.CompleteLanding(
        ticket, VideoCommon::DrainSite::MacroDmaRead,
        [](void* argument) -> u64 {
            const auto& self = *static_cast<const Job*>(argument);
            VideoCommon::QueryDma::FlushPending(*self.memory, *self.rasterizer, self.source,
                                                self.bytes);
            // F3F5b-1: the early write of a final value leaves a 16-byte report's timestamp
            // (words 2-3) to its operation; the read must see it as the drained path does.
            return VideoCommon::V2::IrExec::LandCaptured(*self.rasterizer);
        },
        &job);
}

void Maxwell3D::FrontReadBack(VideoCommon::GpuSplit& split, DAddr device, size_t bytes) {
    // The back records the download after the entry it runs (its writers ran: the caller
    // waited for them) and the front waits for that download's tick only.
    struct Job {
        VideoCommon::V2::WordReadback slot;
        VideoCore::RasterizerInterface* rasterizer{};
        DAddr address{};
        u64 size{};
    };
    static Job job; // The front thread's only one in flight
    job.rasterizer = rasterizer;
    job.address = device;
    job.size = bytes;
    split.RequestBackJob(
        [](void* argument) {
            auto& self = *static_cast<Job*>(argument);
            self.slot.Run(*self.rasterizer, self.address, self.size);
        },
        &job);
    VideoCommon::Nx2Profile::Scope scope{VideoCommon::Nx2Profile::Zone::WriterWait};
    while (!job.slot.Ready()) {
        std::this_thread::yield();
    }
    job.slot.Land();
}

void Maxwell3D::FrontVerify(VideoCommon::GpuSplit& split, FrontState& fs) {
    split.Drain(VideoCommon::DrainSite::FrontVerify);
    static u64 checks{};
    static u64 differences{};
    ++checks;
    // State writes the back still holds in its method sink count as done.
    std::array<u32, Regs::NUM_REGS> expected;
    std::memcpy(expected.data(), regs.reg_array.data(), sizeof(expected));
    for (const auto& [method, value] : method_sink) {
        expected[method] = shadow_state.shadow_ram_control == Regs::ShadowRamControl::Replay
                               ? shadow_state.reg_array[method]
                               : value;
    }
    static u64 control_differences{};
    if (!fs.s1_exec &&
        (fs.control != shadow_state.shadow_ram_control ||
         fs.shadow[MAXWELL3D_REG_INDEX(shadow_ram_control)] != shadow_state.reg_array[MAXWELL3D_REG_INDEX(shadow_ram_control)] ||
         fs.regs[MAXWELL3D_REG_INDEX(shadow_ram_control)] != expected[MAXWELL3D_REG_INDEX(shadow_ram_control)]) &&
        control_differences < 4) {
        LOG_WARNING(HW_GPU, "Split shadow RAM control writes, front ({} total) / back ({} total):",
                    fs.control_count, back_control_count);
        for (u32 i = 0; i < 16; ++i) {
            const auto& f = fs.control_writes[(fs.control_count + i) % 16];
            const auto& b = back_control_writes[(back_control_count + i) % 16];
            LOG_WARNING(HW_GPU, "  front {:#x} (mode {}, path {}) | back {:#x} (mode {}, last {})",
                        f.argument, f.before, f.path, b.argument, b.before, b.path);
        }
    }
    // MACROHLE: the back evaluates the render enable mode of conditional render macros.
    const u32 hle_register =
        VideoCommon::V2::Get(VideoCommon::V2::Switch::MacroHle) == VideoCommon::V2::Mode::On
            ? MAXWELL3D_REG_INDEX(render_enable.mode)
            : ~0U;
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.regs[i] != expected[i]) {
            if (i != hle_register && differences++ < 64) {
                LOG_WARNING(HW_GPU, "Split front register {:#x}: front {:#x}, back {:#x}", i,
                            fs.regs[i], expected[i]);
            }
            fs.regs[i] = expected[i];
        }
    }
    fs.v2_render_enable_unknown = false; // The back's value, just taken
    if (fs.s1_exec) {
        // S1: the shadow RAM lives on the front only (the back never sees the method words).
        if (fs.v2 != nullptr) {
            FrontRegSemResync(fs);
        }
        return;
    }
    if (fs.control != shadow_state.shadow_ram_control) {
        if (control_differences++ < 16) {
            LOG_WARNING(HW_GPU, "Split front shadow RAM control {} vs back {}",
                        static_cast<u32>(fs.control),
                        static_cast<u32>(shadow_state.shadow_ram_control));
        }
        fs.control = shadow_state.shadow_ram_control;
    }
    for (u32 i = 0; i < Regs::NUM_REGS; ++i) {
        if (fs.shadow[i] != shadow_state.reg_array[i]) {
            if (i != hle_register && control_differences++ < 16) {
                LOG_WARNING(HW_GPU, "Split front shadow register {:#x}: front {:#x}, back {:#x}", i,
                            fs.shadow[i], shadow_state.reg_array[i]);
            }
            fs.shadow[i] = shadow_state.reg_array[i];
        }
    }
    if ((checks & 63) == 1) {
        LOG_INFO(HW_GPU, "Split front register checks: {} done, {} differences", checks, differences);
    }
    if (fs.v2 != nullptr) [[unlikely]] {
        FrontRegSemResync(fs);
    }
}


void Maxwell3D::FinishFrontMacro(Core::System& system) {
    // What ProcessMacro and CallMacroMethod do around the macro.
    ConsumeSink(system);
    draw_manager.DrawDeferred(*this);
}

} // namespace Tegra::Engines
