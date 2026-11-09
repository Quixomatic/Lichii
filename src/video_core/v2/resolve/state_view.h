// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <bit>
#include <concepts>
#include <cstddef>

#include "common/common_types.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/v2/draw_cache.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/resolve/cbuf_table.h"

// Pipeline v2 P6c: what a draw's resolve reads from the 3D engine, behind one interface, so the
// same resolve code runs on the commit thread against the live Maxwell3D (LiveView) and on a P6
// shard against a segment's copy-on-write snapshot plus the deltas it replayed (SnapshotView).
//
// Both views are final classes with inline accessors; resolve functions take them as a template
// parameter (StateView), so the live path compiles to the same direct engine reads as before.
//
// Covered (the engine state resolve reads that is not a cache):
//  - registers: Regs() (a snapshot's register file has the Maxwell3D layout);
//  - the constant buffer bindings (state.shader_stages[].const_buffers): Cbuf();
//  - the draw manager: Topology() (after the topology override), Draw(), InlineIndexDraw();
//  - the engine hint (engine_state) and the native-Turing flag;
//  - the register groups and their fingerprints: Groups();
//  - dirty flags: a snapshot has none. DirtyKnown() is false there and resolve code must take
//    the "everything dirty" (force) branch, which by construction gives the same result as the
//    incremental path (P3 PIPEKEY verify compares exactly that).
//  - graphics program addresses: ProgramAddress() (as ShaderCache::RefreshStages computes them).
namespace VideoCommon::V2::Resolve {

using MaxwellRegs = Tegra::Engines::Maxwell3D::Regs;
using EngineHint = Tegra::Engines::Maxwell3D::EngineHint;

/// A draw's parameters as the draw manager holds them when the rasterizer draws.
struct ViewDraw {
    u32 first = 0; ///< vertex first, or index first
    u32 count = 0; ///< vertex count, or index count
    u32 instances = 0;
    u32 base_vertex = 0; ///< indexed: the base index (vertex offset)
    u32 base_instance = 0;
    bool indexed = false;

    bool operator==(const ViewDraw&) const = default;
};

template <typename V>
concept StateView = requires(const V& v, size_t index) {
    { v.Regs() } -> std::same_as<const MaxwellRegs&>;
    { v.Topology() } -> std::same_as<MaxwellRegs::PrimitiveTopology>;
    { v.EngineState() } -> std::same_as<EngineHint>;
    { v.Cbuf(index, index) } -> std::same_as<CbufBinding>;
    { v.Groups() } -> std::same_as<const GroupTracker&>;
    { v.NativeTuring() } -> std::same_as<bool>;
    { v.InlineIndexDraw() } -> std::same_as<bool>;
    { v.Draw() } -> std::same_as<ViewDraw>;
    { v.DirtyKnown() } -> std::same_as<bool>;
};

/// The live engine (the commit thread, and every path with EDEN_NX2_V2_PARALLEL off).
class LiveView final {
public:
    explicit LiveView(const Tegra::Engines::Maxwell3D& engine_) noexcept : engine{&engine_} {}

    [[nodiscard]] const MaxwellRegs& Regs() const noexcept {
        return engine->regs;
    }
    [[nodiscard]] MaxwellRegs::PrimitiveTopology Topology() const noexcept {
        return engine->draw_manager.draw_state.topology;
    }
    [[nodiscard]] EngineHint EngineState() const noexcept {
        return engine->engine_state;
    }
    [[nodiscard]] CbufBinding Cbuf(size_t stage, size_t slot) const noexcept {
        const auto& cbuf = engine->state.shader_stages[stage].const_buffers[slot];
        return CbufBinding{cbuf.address, cbuf.size, cbuf.enabled ? 1U : 0U};
    }
    [[nodiscard]] const GroupTracker& Groups() const noexcept {
        return engine->v2_groups;
    }
    [[nodiscard]] bool NativeTuring() const noexcept {
        return const_cast<Tegra::Engines::Maxwell3D*>(engine)->GetNativeMacroEngine().native_turing;
    }
    [[nodiscard]] bool InlineIndexDraw() const noexcept {
        return !engine->draw_manager.draw_state.inline_index_draw_indexes.empty();
    }
    [[nodiscard]] ViewDraw Draw() const noexcept {
        const auto& draw = engine->draw_manager.draw_state;
        const bool indexed = draw.draw_indexed;
        return ViewDraw{
            .first = indexed ? draw.index_buffer.first : draw.vertex_buffer.first,
            .count = indexed ? draw.index_buffer.count : draw.vertex_buffer.count,
            .instances = draw.instance_count,
            .base_vertex = indexed ? draw.base_index : 0,
            .base_instance = draw.base_instance,
            .indexed = indexed,
        };
    }
    [[nodiscard]] bool DirtyKnown() const noexcept {
        return true;
    }
    [[nodiscard]] const Tegra::Engines::Maxwell3D& Engine() const noexcept {
        return *engine;
    }

private:
    const Tegra::Engines::Maxwell3D* engine;
};

/// A shard's state at a draw: its mirror (the segment's snapshot plus the replayed deltas) and
/// the draw's Draw record.
class SnapshotView final {
public:
    SnapshotView(const u32* regs_, const CbufTable* cbufs_, const GroupTracker* groups_,
                 MaxwellRegs::PrimitiveTopology topology_, const ViewDraw& draw_,
                 bool native_turing_) noexcept
        : regs{regs_}, cbufs{cbufs_}, groups{groups_}, topology{topology_}, draw{draw_},
          native_turing{native_turing_} {}

    [[nodiscard]] const MaxwellRegs& Regs() const noexcept {
        return *reinterpret_cast<const MaxwellRegs*>(regs);
    }
    [[nodiscard]] MaxwellRegs::PrimitiveTopology Topology() const noexcept {
        return topology;
    }
    /// Shards resolve only draws without an engine hint (DecideDraw rejects the others).
    [[nodiscard]] EngineHint EngineState() const noexcept {
        return EngineHint::None;
    }
    [[nodiscard]] CbufBinding Cbuf(size_t stage, size_t slot) const noexcept {
        return cbufs->stages[stage][slot];
    }
    [[nodiscard]] const GroupTracker& Groups() const noexcept {
        return *groups;
    }
    [[nodiscard]] bool NativeTuring() const noexcept {
        return native_turing;
    }
    /// The front's draw model gives no per-draw inline flag: the commit checks it (DecideDraw).
    [[nodiscard]] bool InlineIndexDraw() const noexcept {
        return false;
    }
    [[nodiscard]] ViewDraw Draw() const noexcept {
        return draw;
    }
    [[nodiscard]] bool DirtyKnown() const noexcept {
        return false;
    }

private:
    const u32* regs;
    const CbufTable* cbufs;
    const GroupTracker* groups;
    MaxwellRegs::PrimitiveTopology topology;
    ViewDraw draw;
    bool native_turing;
};

static_assert(StateView<LiveView>);
static_assert(StateView<SnapshotView>);
static_assert(sizeof(MaxwellRegs) == MaxwellRegs::NUM_REGS * sizeof(u32));

/// The address of graphics program `index` (0 when it does not run), as RefreshStages finds it.
template <StateView View>
[[nodiscard]] GPUVAddr ProgramAddress(const View& view, size_t index) {
    const MaxwellRegs& regs = view.Regs();
    if (!regs.IsShaderConfigEnabled(index)) {
        return 0;
    }
    const auto program = static_cast<MaxwellRegs::ShaderType>(index);
    if (program == MaxwellRegs::ShaderType::Pixel && !regs.rasterize_enable) {
        return 0;
    }
    const bool native = view.NativeTuring();
    if (native && regs.IsMeshEnabled() && program == MaxwellRegs::ShaderType::Geometry) {
        return 0;
    }
    const auto& config = regs.pipelines[index];
    return native ? (static_cast<u64>(config.reserved[0] & 0xff) << 32) | config.reserved[1]
                  : regs.program_region.Address() + config.offset;
}

/// Step (a): the P4 draw-cache key's constant buffer part. The bindings of the slots in
/// `key_cbufs` (per stage: the handle and storage-descriptor buffers) are mixed into `key`.
template <StateView View>
[[nodiscard]] u64 KeyCbufs(u64 key, const std::array<u32, CbufStages>& key_cbufs,
                           const View& view) {
    for (size_t stage = 0; stage < CbufStages; ++stage) {
        for (u32 mask = key_cbufs[stage]; mask != 0; mask &= mask - 1) {
            const CbufBinding cbuf = view.Cbuf(stage, static_cast<size_t>(std::countr_zero(mask)));
            key = DrawCache::Mix(key, cbuf.address);
            key = DrawCache::Mix(key, (u64{cbuf.size} << 1) | (cbuf.enabled != 0 ? 1 : 0));
        }
    }
    return key;
}

/// The whole binding table of a view (verify, tests).
template <StateView View>
[[nodiscard]] CbufTable CbufTableOf(const View& view) {
    CbufTable table;
    for (size_t stage = 0; stage < CbufStages; ++stage) {
        for (size_t slot = 0; slot < CbufSlots; ++slot) {
            table.stages[stage][slot] = view.Cbuf(stage, slot);
        }
    }
    return table;
}

} // namespace VideoCommon::V2::Resolve
