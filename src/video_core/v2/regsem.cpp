// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <string>

#include "common/assert.h"
#include "common/logging.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/v2_stats.h"

#define OFF(field_name) MAXWELL3D_REG_INDEX(field_name)
#define NUM(field_name) (sizeof(::Tegra::Engines::Maxwell3D::Regs::field_name) / (sizeof(u32)))

namespace VideoCommon::V2::RegSem {

namespace {

using Tegra::Engines::Maxwell3D;
static_assert(Table3D::NumRegs == Maxwell3D::Regs::NUM_REGS);

// The P0 group table (v2_stats.cpp used to own it).
void FillGroups(std::array<RegGroup, Table3D::NumRegs>& table) {
    using enum RegGroup;
    table.fill(Pipeline);
    const auto fill = [&table](size_t begin, size_t count, RegGroup group) {
        if (begin < table.size()) {
            std::fill_n(table.begin() + begin, std::min(count, table.size() - begin), group);
        }
    };
    // Methods before 0x80: object binds, notifies, WFI, macro upload, shadow RAM control,
    // inline-to-memory uploads.
    fill(0, OFF(run_ds_now), Misc);
    fill(OFF(invalidate_shader_cache), 1, Misc);
    fill(OFF(sync_info), 1, Misc);
    fill(OFF(flush_invalidate_rop_mini_cache), 1, Misc);
    fill(OFF(decompress_surface), 1, Misc);
    fill(OFF(peer_semaphore_release), 1, Misc);
    fill(OFF(statistics_count), 1, Misc);
    fill(OFF(clear_rect), NUM(clear_rect), Misc);
    fill(OFF(clear_color), NUM(clear_color), Misc);
    fill(OFF(clear_depth), 1, Misc);
    fill(OFF(clear_stencil), 1, Misc);
    fill(OFF(clear_control), 1, Misc);
    fill(OFF(clear_surface), 1, Misc);
    fill(OFF(clear_clip_id_surface), 1, Misc);
    fill(OFF(clear_report_value), 1, Misc);
    fill(OFF(invalidate_shader_cache_no_wfi), 1, Misc);
    fill(OFF(fragment_barrier), 1, Misc);
    fill(OFF(invalidate_texture_data_cache), 1, Misc);
    fill(OFF(tiled_cache_barrier), 1, Misc);
    fill(OFF(invalidate_sampler_cache), 1, Misc);
    fill(OFF(invalidate_texture_header_cache), 1, Misc);
    fill(OFF(invalidate_texture_cache_no_wfi), 1, Misc);
    fill(OFF(invalidate_sampler_cache_lines), 1, Misc);
    fill(OFF(invalidate_texture_header_cache_lines), 1, Misc);
    fill(OFF(invalidate_texture_data_cache_lines), 1, Misc);
    fill(OFF(invalidate_sampler_cache_no_wfi), 1, Misc);
    fill(OFF(invalidate_texture_header_cache_no_wfi), 1, Misc);
    fill(OFF(invalidate_zcull), 1, Misc);
    fill(OFF(render_enable), NUM(render_enable), Misc);
    fill(OFF(report_semaphore), NUM(report_semaphore), Misc);
    fill(OFF(falcon), NUM(falcon), Misc);
    fill(OFF(shadow_scratch), NUM(shadow_scratch), Misc);
    // Binds are flagged (CbufBindFlag): the bound address comes from the const_buffer registers.
    fill(OFF(bind_groups), NUM(bind_groups), Misc);
    fill(OFF(const_buffer), NUM(const_buffer), CbufData);

    fill(OFF(viewport_transform), NUM(viewport_transform), Viewport);
    fill(OFF(viewports), NUM(viewports), Viewport);
    fill(OFF(scissor_test), NUM(scissor_test), Viewport);
    fill(OFF(window_offset_x), 1, Viewport);
    fill(OFF(window_offset_y), 1, Viewport);

    fill(OFF(vertex_streams), NUM(vertex_streams), VertexStreams);
    fill(OFF(vertex_stream_limits), NUM(vertex_stream_limits), VertexStreams);
    fill(OFF(vertex_stream_instances), NUM(vertex_stream_instances), VertexStreams);
    // start address, limit, format; first and count are draw parameters.
    fill(OFF(index_buffer), 5, IndexBuffer);

    fill(OFF(tex_header), NUM(tex_header), TexPools);
    fill(OFF(tex_sampler), NUM(tex_sampler), TexPools);
    fill(OFF(sampler_binding), NUM(sampler_binding), TexPools);
    fill(OFF(bindless_texture_const_buffer_slot), 1, TexPools);

    fill(OFF(rt), NUM(rt), RenderTargets);
    fill(OFF(zeta), NUM(zeta), RenderTargets);
    fill(OFF(zeta_size), NUM(zeta_size), RenderTargets);
    fill(OFF(zeta_enable), 1, RenderTargets);
    fill(OFF(rt_control), 1, RenderTargets);
    fill(OFF(surface_clip), NUM(surface_clip), RenderTargets);
    fill(OFF(rt_layer), 1, RenderTargets);
    fill(OFF(zeta_layer_offset), 1, RenderTargets);

    fill(OFF(pipelines), NUM(pipelines), Shaders);
    fill(OFF(program_region), NUM(program_region), Shaders);

    fill(OFF(vertex_buffer), NUM(vertex_buffer), DrawParams);
    fill(OFF(index_buffer) + 5, 2, DrawParams);
    fill(OFF(index_buffer32_first), OFF(index_buffer8_subsequent) - OFF(index_buffer32_first) + 1,
         DrawParams);
    fill(OFF(draw), NUM(draw), DrawParams);
    fill(OFF(draw_inline_index), 1, DrawParams);
    fill(OFF(inline_index_2x16), NUM(inline_index_2x16), DrawParams);
    fill(OFF(inline_index_4x8), NUM(inline_index_4x8), DrawParams);
    fill(OFF(vertex_array_instance_first), 1, DrawParams);
    fill(OFF(vertex_array_instance_subsequent), 1, DrawParams);
    fill(OFF(global_base_vertex_index), 1, DrawParams);
    fill(OFF(global_base_instance_index), 1, DrawParams);
    fill(OFF(vertex_id_base), 1, DrawParams);
    fill(OFF(draw_zero_index), 1, DrawParams);
    fill(OFF(draw_auto_byte_count), 1, DrawParams);
}

// Executable methods Maxwell3D::ProcessMethodCall handles itself (the rest go to the draw
// manager), as FrontHandledOutsideDrawManager in maxwell_3d.cpp.
bool HandledOutsideDrawManager(u32 method) {
    constexpr u32 cb_data = OFF(const_buffer.buffer);
    if (method >= cb_data && method < cb_data + 16) {
        return true;
    }
    switch (method) {
    case Maxwell3D::RootTables::SelectMethod:
    case Maxwell3D::RootTables::LoadMethod:
    case OFF(wait_for_idle):
    case OFF(shadow_ram_control):
    case OFF(load_mme.instruction_ptr):
    case OFF(load_mme.instruction):
    case OFF(load_mme.start_address):
    case OFF(falcon[4]):
    case OFF(bind_groups[0].raw_config):
    case OFF(bind_groups[1].raw_config):
    case OFF(bind_groups[2].raw_config):
    case OFF(bind_groups[3].raw_config):
    case OFF(bind_groups[4].raw_config):
    case OFF(report_semaphore.query):
    case OFF(render_enable.mode):
    case OFF(clear_report_value):
    case OFF(sync_info):
    case OFF(launch_dma):
    case OFF(inline_data):
    case OFF(fragment_barrier):
    case OFF(invalidate_texture_data_cache):
    case OFF(tiled_cache_barrier):
        return true;
    default:
        return false;
    }
}

Table3D BuildTable3D(const std::bitset<(std::numeric_limits<u16>::max)()>& execution_mask) {
    Table3D table;
    FillGroups(table.groups);
    table.const_buffer_offset = OFF(const_buffer.offset);
    table.instruction_ptr = OFF(load_mme.instruction_ptr);
    table.start_address_ptr = OFF(load_mme.start_address_ptr);
    table.shadow_scratch = OFF(shadow_scratch);
    table.shadow_control = OFF(shadow_ram_control);
    constexpr u32 cb_data = OFF(const_buffer.buffer);
    for (u32 method = 0; method < Table3D::NumRegs; ++method) {
        u16 flags = 0;
        if (execution_mask[method]) {
            flags |= Exec;
            if (!HandledOutsideDrawManager(method)) {
                flags |= DrawManager;
            }
        }
        if (method >= cb_data && method < cb_data + 16) {
            flags |= CbData;
        }
        table.flags[method] = flags;
    }
    const auto add = [&table](u32 method, u16 flags) { table.flags[method] |= flags; };
    add(OFF(shadow_ram_control), ShadowControl);
    add(OFF(load_mme.instruction_ptr), MacroCodePointer);
    add(OFF(load_mme.instruction), MacroCode);
    add(OFF(load_mme.start_address), MacroBind);
    add(OFF(falcon[4]), Falcon4);
    add(OFF(draw.begin), DrawBegin);
    add(OFF(draw_inline_index), InlineIndex);
    add(OFF(inline_index_2x16.even), InlineIndex);
    add(OFF(inline_index_4x8.index0), InlineIndex);
    add(OFF(draw.end), LoopSkippable | DrawEnd);
    add(OFF(vertex_buffer.first), LoopSkippable);
    add(OFF(vertex_buffer.count), LoopSkippable);
    add(OFF(inline_data), InlineData);
    add(OFF(report_semaphore.query), ReportSemaphore);
    add(OFF(bind_groups[0].raw_config), CbufBindFlag);
    add(OFF(bind_groups[1].raw_config), CbufBindFlag);
    add(OFF(bind_groups[2].raw_config), CbufBindFlag);
    add(OFF(bind_groups[3].raw_config), CbufBindFlag);
    add(OFF(bind_groups[4].raw_config), CbufBindFlag);
    // Methods ProcessMethodCall doesn't break fast draw reuse for.
    for (u32 method = 0; method < Table3D::NumRegs; ++method) {
        if (Maxwell3D::DrawManager::IsInstanceLoopMethod(method) ||
            Maxwell3D::IsFastDrawDeltaMethod(method)) {
            table.flags[method] |= FastDrawQuiet;
        }
    }
    // The executing engine's action per register (Maxwell3D::TableWrite).
    for (u32 method = 0; method < Table3D::NumRegs; ++method) {
        Action3D action = Action3D::Store;
        if (execution_mask[method]) {
            action = (table.flags[method] & CbData) != 0 ? Action3D::CbData : Action3D::DrawManager;
        }
        table.actions[method] = action;
    }
    const auto set = [&table, &execution_mask](u32 method, Action3D action) {
        if (execution_mask[method]) {
            table.actions[method] = action;
        }
    };
    set(OFF(vertex_buffer.first), Action3D::StoreExec);
    set(OFF(vertex_buffer.count), Action3D::StoreExec);
    set(OFF(index_buffer.first), Action3D::StoreExec);
    set(OFF(topology_override), Action3D::TopologyOverride);
    set(OFF(bind_groups[0].raw_config), Action3D::CbufBind0);
    set(OFF(bind_groups[1].raw_config), Action3D::CbufBind1);
    set(OFF(bind_groups[2].raw_config), Action3D::CbufBind2);
    set(OFF(bind_groups[3].raw_config), Action3D::CbufBind3);
    set(OFF(bind_groups[4].raw_config), Action3D::CbufBind4);
    set(OFF(draw.begin), Action3D::DrawBegin);
    set(OFF(draw.end), Action3D::DrawEnd);
    set(Maxwell3D::RootTables::SelectMethod, Action3D::RootSelect);
    set(Maxwell3D::RootTables::LoadMethod, Action3D::RootLoad);
    set(OFF(wait_for_idle), Action3D::WaitForIdle);
    set(OFF(shadow_ram_control), Action3D::ShadowControl);
    set(OFF(load_mme.instruction_ptr), Action3D::MacroCodePointer);
    set(OFF(load_mme.instruction), Action3D::MacroCode);
    set(OFF(load_mme.start_address), Action3D::MacroBind);
    set(OFF(falcon[4]), Action3D::Falcon4);
    set(OFF(report_semaphore.query), Action3D::ReportSemaphore);
    set(OFF(render_enable.mode), Action3D::RenderEnable);
    set(OFF(clear_report_value), Action3D::ClearReportValue);
    set(OFF(sync_info), Action3D::SyncInfo);
    set(OFF(launch_dma), Action3D::LaunchDma);
    set(OFF(inline_data), Action3D::InlineData);
    set(OFF(fragment_barrier), Action3D::FragmentBarrier);
    set(OFF(invalidate_texture_data_cache), Action3D::InvalidateTextureDataCache);
    set(OFF(tiled_cache_barrier), Action3D::TiledCacheBarrier);
    // The flags the front mirror acts on must be on executable methods (FrontWrite).
    for (const size_t method : {OFF(shadow_ram_control), OFF(load_mme.instruction_ptr),
                             OFF(load_mme.instruction), OFF(load_mme.start_address),
                             OFF(falcon[4]), OFF(draw.begin), OFF(draw.end)}) {
        ASSERT(execution_mask[method]);
    }
    return table;
}

const Table3D* g_table3d{};

struct DriftData {
    u64 checks{};
    u64 drifted{};
};
thread_local DriftData t_drift{};

} // Anonymous namespace

void InitTable3D(const std::bitset<(std::numeric_limits<u16>::max)()>& execution_mask) {
    static const Table3D table = BuildTable3D(execution_mask);
    g_table3d = &table;
}

const Table3D& GetTable3D() {
    return *g_table3d;
}

u32 Rehash3D(GroupTracker& tracker, const u32* regs) {
    const Table3D& table = GetTable3D();
    std::array<u64, NumRegGroups> hashes{};
    for (u32 i = 0; i < Table3D::NumRegs; ++i) {
        hashes[static_cast<size_t>(table.groups[i])] ^= MixRegister(i, regs[i]);
    }
    u32 drifted = 0;
    for (u32 bits = HashedGroups; bits != 0; bits &= bits - 1) {
        const u32 group = static_cast<u32>(std::countr_zero(bits));
        if (tracker.valid && hashes[group] != tracker.hashes[group]) {
            drifted |= 1U << group;
        }
        tracker.hashes[group] = hashes[group];
    }
    tracker.valid = true;
    return drifted;
}

void NoteDraw3D(GroupTracker& tracker, const u32* regs) {
    if (!tracker.valid) [[unlikely]] {
        Rehash3D(tracker, regs);
        tracker.dirty = AllGroups;
    } else if ((Verify(Switch::RegSem) || Stats::Enabled()) && (tracker.draws & 4095) == 4095)
        [[unlikely]] {
        const u32 drifted = Rehash3D(tracker, regs);
        DriftData& data = t_drift;
        ++data.checks;
        if (drifted != 0) {
            Stats::NoteHashDrift();
            if (data.drifted++ < 16) {
                std::string names;
                for (u32 bits = drifted; bits != 0; bits &= bits - 1) {
                    names += ' ';
                    names += reg_group_names[static_cast<size_t>(std::countr_zero(bits))];
                }
                LOG_WARNING(HW_GPU, "V2 regsem: group hashes drifted from the registers:{}",
                            names);
            }
        }
        if ((data.checks & 63) == 1) {
            LOG_INFO(HW_GPU, "V2 regsem: {} group hash checks, {} drifted", data.checks,
                     data.drifted);
        }
    }
    tracker.BeginDraw();
}

const std::array<ComputeGroup, 0x1000>& TableComputeGroups() {
    static const std::array<ComputeGroup, 0x1000> table = [] {
        std::array<ComputeGroup, 0x1000> groups;
        groups.fill(ComputeGroup::Pipeline);
        const auto fill = [&groups](u32 begin, u32 end, ComputeGroup group) {
            std::fill(groups.begin() + begin, groups.begin() + end, group);
        };
        fill(0x60, 0x6e, ComputeGroup::Upload);
        fill(0xa0, 0xc0, ComputeGroup::Launch);
        fill(0xc6, 0x108, ComputeGroup::Qmd);
        fill(0x557, 0x563, ComputeGroup::Pools); // TSC (0x557) and TIC (0x55d) pools
        fill(0x582, 0x584, ComputeGroup::Pools); // code location
        fill(0x982, 0x983, ComputeGroup::Pools); // bindless texture constant buffer slot
        // Triggers and state no dispatch reads.
        fill(0x45, 0x49, ComputeGroup::Misc);    // macro upload and bind
        fill(0xae, 0xb1, ComputeGroup::Misc);    // launch, QMD signals
        fill(0x107, 0x108, ComputeGroup::Misc);  // inline QMD launch
        fill(0x6c0, 0x6c4, ComputeGroup::Misc);  // report semaphore
        fill(0xe00, 0x1000, ComputeGroup::Misc); // macro calls
        return groups;
    }();
    return table;
}

u32 RehashCompute(ComputeGroupTracker& tracker, const u32* registers, u32 count) {
    const auto& table = TableComputeGroups();
    std::array<u64, NumComputeGroups> hashes{};
    for (u32 i = 0; i < count; ++i) {
        hashes[static_cast<size_t>(table[i])] ^= MixRegister(i, registers[i]);
    }
    u32 drifted = 0;
    for (u32 bits = HashedComputeGroups; bits != 0; bits &= bits - 1) {
        const u32 group = static_cast<u32>(std::countr_zero(bits));
        if (tracker.valid && hashes[group] != tracker.hashes[group]) {
            drifted |= 1U << group;
        }
        tracker.hashes[group] = hashes[group];
    }
    tracker.valid = true;
    return drifted;
}

void NoteDispatchCompute(ComputeGroupTracker& tracker, const u32* registers, u32 count) {
    if (!tracker.valid) [[unlikely]] {
        RehashCompute(tracker, registers, count);
        tracker.dirty = AllComputeGroups;
    } else if ((Verify(Switch::RegSem) || Stats::Enabled()) && (tracker.draws & 4095) == 4095)
        [[unlikely]] {
        static std::atomic<u32> logged{};
        if (const u32 drifted = RehashCompute(tracker, registers, count);
            drifted != 0 && logged.fetch_add(1, std::memory_order_relaxed) < 16) {
            Stats::NoteHashDrift();
            LOG_WARNING(HW_GPU, "V2 regsem: compute group hashes drifted from the registers, "
                                "groups {:#x}",
                        drifted);
        }
    }
    tracker.BeginDraw();
}

const std::bitset<0xe00>& TableComputePlain() {
    static const std::bitset<0xe00> plain = [] {
        std::bitset<0xe00> bits;
        const auto& flags = TableCompute();
        for (u32 method = 0; method < 0xe00; ++method) {
            bits[method] = flags[method] == 0 && method != 0x6c && method != 0x6d;
        }
        return bits;
    }();
    return plain;
}

const std::array<u8, 0x1000>& TableCompute() {
    static const std::array<u8, 0x1000> table = [] {
        std::array<u8, 0x1000> flags{};
        flags[0x45] = ComputeClearCode | ComputeUploaded;
        flags[0x46] = ComputeAddCode;
        flags[0x47] = ComputeBindPointer | ComputeUploaded;
        flags[0x48] = ComputeBind;
        flags[0x107] = ComputeInlineQmd;
        flags[0xae] = ComputeLaunch;
        flags[0xaf] = ComputeLaunch;
        flags[0xb0] = ComputeLaunch;
        flags[0x6c3] = ComputeRelease;
        return flags;
    }();
    return table;
}

} // namespace VideoCommon::V2::RegSem
