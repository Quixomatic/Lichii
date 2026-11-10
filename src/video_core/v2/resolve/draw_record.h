// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/draw_cache.h"
#include "video_core/v2/residency.h"
#include "video_core/v2/resolve/state_view.h"
#include "video_core/v2/versions.h"

// Pipeline v2 P6 Step 0 (notes/STEP0_SPEC.md §1.1, EDEN_NX2_V2_DRAWREC): a draw's resolved state
// as one relocatable record. ResolveDraw<View> fills it from a StateView alone (pure), ApplyDraw
// runs the cache mutations it requests and EncodeDraw records the draw from it. Every handle is
// a slot id with a generation (or a commit proxy), every address a VA or device address, so a
// record can be copied (a recipe template, a continued draw) and checked by stamps.
namespace VideoCommon::V2::Resolve {

enum class ResolveStatus : u8 { Ready, Stale, CommitOnly };

/// Why a draw runs the old Configure (D3): what Step 0 does not resolve into a record yet.
enum class CommitOnlyReason : u8 {
    None,
    Mesh,
    EmulatedMesh,
    TransformFeedback,
    InlineIndex,
    Indirect,
    MultiRangeStorage,
    TextureBufferViewCreate,
    FrameDump,
    SampledRecording,
    Retry,
    Count,
};
inline constexpr size_t NumCommitOnlyReasons = static_cast<size_t>(CommitOnlyReason::Count);
inline constexpr std::array<const char*, NumCommitOnlyReasons> commit_only_names{
    "none",   "mesh",          "emulated mesh",      "transform feedback",
    "inline index", "indirect", "multi-range storage", "texture buffer view create",
    "frame dump", "sampled recording", "retry",
};

enum class StaleReason : u8 { None, DescriptorTables };

/// What SynchronizeDescriptors(false) reads: the graphics TIC and TSC tables.
struct DescriptorTablesKey {
    GPUVAddr tic_addr{};
    u32 tic_limit{};
    GPUVAddr tsc_addr{};
    u32 tsc_limit{};
    bool via_header{};

    bool operator==(const DescriptorTablesKey&) const = default;
};

/// The register part of the texture cache's RenderTargetInputs.
struct RtKey {
    std::array<MaxwellRegs::RenderTargetConfig, MaxwellRegs::NumRenderTargets> rt{};
    MaxwellRegs::Zeta zeta{};
    MaxwellRegs::SurfaceClip surface_clip{};
    MaxwellRegs::RtControl rt_control{};
    MaxwellRegs::ZetaSize zeta_size{};
    u32 zeta_enable{};
    Tegra::Texture::MsaaMode anti_alias_samples_mode{};

    bool operator==(const RtKey& other) const {
        return std::memcmp(this, &other, sizeof(*this)) == 0;
    }
};

/// The scale-dependent dynamic state, kept as the register slice it is computed from (encode
/// applies the resolution scale).
struct ViewportInputs {
    std::array<MaxwellRegs::ViewportTransform, MaxwellRegs::NumViewports> transform{};
    std::array<MaxwellRegs::ScissorTest, MaxwellRegs::NumViewports> scissor{};
    MaxwellRegs::SurfaceClip surface_clip{};
    MaxwellRegs::WindowOrigin window_origin{};
    MaxwellRegs::DepthMode depth_mode{};
    u32 viewport_scale_offset_enabled{};
};

/// One blend equation in guest terms (encode maps it to the host API).
struct BlendEquation {
    MaxwellRegs::Blend::Factor color_source{};
    MaxwellRegs::Blend::Factor color_dest{};
    MaxwellRegs::Blend::Equation color_op{};
    MaxwellRegs::Blend::Factor alpha_source{};
    MaxwellRegs::Blend::Factor alpha_dest{};
    MaxwellRegs::Blend::Equation alpha_op{};

    bool operator==(const BlendEquation&) const = default;
};

struct StencilFaceOp {
    MaxwellRegs::StencilOp::Op fail{};
    MaxwellRegs::StencilOp::Op zfail{};
    MaxwellRegs::StencilOp::Op zpass{};
    MaxwellRegs::ComparisonOp func{};

    bool operator==(const StencilFaceOp&) const = default;
};

struct VertexAttributeInput {
    u8 enabled{}; ///< not constant and its binding exists
    u8 binding{};
    u16 pad{};
    MaxwellRegs::VertexAttribute::Type type{};
    MaxwellRegs::VertexAttribute::Size size{};
    u32 offset{}; ///< guest offset (encode adds the vertex repack adjustment)

    bool operator==(const VertexAttributeInput&) const = default;
};

struct VertexBindingInput {
    u32 stride{}; ///< guest stride (encode substitutes a repacked binding's)
    u32 divisor{};
    u8 instanced{};
    std::array<u8, 3> pad{};

    bool operator==(const VertexBindingInput&) const = default;
};

/// The scale-independent dynamic state, computed (v2/resolve/dynamic_state.h). Blending holds
/// the guest colour masks; encode applies the pipeline's colour output mask.
struct DynamicValues {
    f32 depth_bias_constant{};
    f32 depth_bias_clamp{};
    f32 depth_bias_slope{};
    std::array<f32, 4> blend_constants{};
    f32 depth_bounds_min{};
    f32 depth_bounds_max{};
    u32 stencil_two_side{};
    u32 stencil_front_ref{};
    u32 stencil_back_ref{};
    u32 stencil_front_write_mask{};
    u32 stencil_back_write_mask{};
    u32 stencil_front_compare_mask{};
    u32 stencil_back_compare_mask{};
    f32 line_width{};
    u32 cull_enable{};
    MaxwellRegs::CullFace cull_face{};
    MaxwellRegs::ComparisonOp depth_compare{};
    MaxwellRegs::FrontFace front_face{}; ///< the register (encode flips it with front_face_flip)
    StencilFaceOp stencil_front{};
    StencilFaceOp stencil_back{}; ///< the register (used while two-sided stencil is on)
    u8 depth_bounds_test{};
    u8 depth_test{};
    u8 depth_write{};
    u8 stencil_test{};
    u8 primitive_restart{};
    u8 rasterizer_discard{};
    u8 depth_bias_enable{};
    u8 depth_clamp{};
    u8 line_stipple{};
    u8 alpha_to_coverage{}; ///< the register bit; encode ands the pipeline's support
    u8 alpha_to_one{};      ///< the register bit; encode ands the pipeline's support
    u8 logic_op_enable{};   ///< the effective enable (G3)
    u8 line_smooth{};       ///< smooth line rasterization (requested and supported)
    u8 conservative{};
    u8 iterated_blend_squash{}; ///< the squashed iterated-blend workaround replaces equations
    u8 front_face_flip{};       ///< window_origin.flip_y
    u32 logic_op{}; ///< the register value (encode maps 0x1500..0x150f, else no-op)
    std::array<u8, MaxwellRegs::NumRenderTargets> color_masks{}; ///< RGBA bits, unmasked
    std::array<u8, MaxwellRegs::NumRenderTargets> blend_enables{};
    std::array<u8, MaxwellRegs::NumRenderTargets> color_write_enables{};
    std::array<BlendEquation, MaxwellRegs::NumRenderTargets> blend_equations{};
    std::array<VertexAttributeInput, MaxwellRegs::NumVertexAttributes> attributes{};
    std::array<VertexBindingInput, MaxwellRegs::NumVertexArrays> bindings{};
    u32 num_attributes{}; ///< attributes the device takes (the rest are not read)
    u32 num_bindings{};
};

struct DynamicState {
    ViewportInputs viewport{};
    DynamicValues values{};
};

/// The device features ComputeDynamic depends on (fixed at startup).
struct DynamicFeatures {
    bool eds1{};
    bool eds2{};
    bool eds2_extras{};
    bool eds3_enables{};
    bool eds3_blending{};
    bool color_write_enable{};
    bool vertex_input_dynamic{};
    bool amd_logic_op{};          ///< the AMD workaround: logic op off with float attributes
    bool nv_viewport_swizzle{};   ///< the host applies the viewport swizzle
    bool depth_range_unrestricted{};
    bool depth_bounds{};
    bool smooth_lines{};
    bool molten_vk{};             ///< primitive restart always on
    bool d24_bias_rescale{};      ///< D24 depth bias rescaled (no D24 on the host, listed title)
    bool squashed_iterated_blend{};
    u32 restart_topologies{};     ///< bit per Maxwell topology: primitive restart supported
    u32 max_viewports{};
    u32 max_vertex_attributes{};
    u32 max_vertex_bindings{};
};

/// A record's resolved texture view. `view.id` may be a commit proxy (Commit::IsProxy).
struct ResolvedView {
    u32 tic_index{};
    Versions::WeakRef view{};
    u8 stage{};
    u8 kind{}; ///< 0 texture buffer, 1 image buffer, 2 texture, 3 image
    u8 blacklist{};
    u8 pad{};
};
struct ResolvedSampler {
    u32 tsc_index{};
    Versions::WeakRef sampler{};
};
struct ResolvedVertex {
    u8 slot{};
    std::array<u8, 3> pad{};
    Versions::WeakRef buffer{};
    DAddr daddr{};
    u32 size{};
    u32 stride{};
};
struct ResolvedIndex {
    Versions::WeakRef buffer{};
    DAddr daddr{};
    u32 size{};
    u32 format{};
    u32 first{};
    u32 count{};
    bool valid{};
};
struct ResolvedUbo {
    u8 stage{};
    u8 slot{};
    DAddr daddr{};
    u32 size{}; ///< the constant buffer's bound size
    Versions::WeakRef buffer{};
    u64 version{}; ///< F2-02: a CbufVersion reference (0: none in Step 0)
    GPUVAddr gpu_addr{}; ///< the bound constant buffer's VA (F2-01: sub-record reuse by value)
};
struct ResolvedSsbo {
    u8 stage{};
    u8 index{};
    u8 written{};
    u8 table_pointer{};
    DAddr daddr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    Versions::WeakRef buffer{};
};
struct ResolvedTbo {
    u8 stage{};
    u8 index{};
    u8 written{};
    u8 image{};
    GPUVAddr gpu_addr{};
    DAddr daddr{};
    u32 size{};
    u32 format{};
    Versions::WeakRef buffer{};
};

/// A span of a record's arena: byte offset and element count.
struct Seg {
    u32 offset{};
    u32 count{};
};

inline constexpr size_t NumRecordTargets = 9;
inline constexpr size_t NumRecordStages = 5;

/// The fixed part of a record (trivially copyable).
struct DrawHeader {
    u64 ordinal{};
    u64 segment{};
    u64 fingerprint{}; ///< view.Groups().Fingerprint(AllGroups)
    /// view.Groups().hashes (step 17: a group equal to the previous record's reuses its
    /// sub-record); `groups_valid`: the tracker had computed them.
    std::array<u64, NumRegGroups> group_hashes{};
    bool groups_valid{};
    ViewDraw draw{};
    MaxwellRegs::PrimitiveTopology topology{};
    bool indexed{};
    bool indirect{};
    ResolveStatus status{};
    CommitOnlyReason reason{};
    StaleReason stale{};
    u8 flags{}; ///< bit 0 computed handles, bit 1 table pointers, bit 2 blacklisted images
    void* pipeline{}; ///< GraphicsPipeline*, valid under pipeline_epoch
    u64 pipeline_epoch{};
    // Stamps.
    u64 map_serial{};
    u64 image_table_changes{};
    u64 sampler_table_changes{};
    u64 texture_bindings_serial{};
    u64 buffer_deletions{};
    u64 rescale_epoch{};
    u64 shape_generation{};
    ResidencyStamp residency{};
    DescriptorTablesKey tables{};
    RtKey rt{};
    std::array<u32, NumRecordTargets> rt_ids{};
    bool rt_inputs_hold{};
    bool rt_known{};
    DynamicState dynamic{};
    ResolvedIndex index{};
    u32 vertex_mask{};
    std::array<u32, NumRecordStages> ubo_masks{};
    std::array<bool, NumRecordStages> stages{};
    struct {
        bool uses{};
        f32 w{};
        f32 h{};
    } render_area;
    Seg views, samplers, reads, vertex, ubo, ssbo, tbo;
    struct {
        u32 begin{};
        u32 count{};
    } texture_requests, buffer_requests;
};

enum DrawFlags : u8 {
    DrawFlagComputedHandles = 1 << 0,
    DrawFlagTablePointers = 1 << 1,
    DrawFlagBlacklistedImages = 1 << 2,
};

/// A draw record: the header plus a per-record arena holding its variable-length lists. A
/// copy is one header copy and one arena assign.
struct ResolvedDraw {
    DrawHeader h{};
    std::vector<std::byte> arena;

    template <class T>
    [[nodiscard]] std::span<T> Span(Seg s) {
        static_assert(std::is_trivially_copyable_v<T>);
        return {reinterpret_cast<T*>(arena.data() + s.offset), s.count};
    }
    template <class T>
    [[nodiscard]] std::span<const T> Span(Seg s) const {
        static_assert(std::is_trivially_copyable_v<T>);
        return {reinterpret_cast<const T*>(arena.data() + s.offset), s.count};
    }
    /// Appends `items` to the arena (aligned to 8 bytes) and returns their segment.
    template <class T>
    Seg Push(std::span<const T> items) {
        static_assert(std::is_trivially_copyable_v<T> && alignof(T) <= 8);
        const size_t offset = (arena.size() + 7) & ~size_t{7};
        const size_t bytes = items.size_bytes();
        arena.resize(offset + bytes);
        if (bytes != 0) {
            std::memcpy(arena.data() + offset, items.data(), bytes);
        }
        return Seg{static_cast<u32>(offset), static_cast<u32>(items.size())};
    }
    void Clear() {
        h = DrawHeader{};
        arena.clear();
    }
};

static_assert(std::is_trivially_copyable_v<DrawHeader>);

/// A record compare's field groups (RecordStats::mismatches).
enum class RecordGroup : u8 {
    Pipeline,
    Views,
    Samplers,
    Reads,
    Vertex,
    Index,
    Ubo,
    Ssbo,
    Tbo,
    Rt,
    Dynamic,
    RenderArea,
    Payload,
    Push,
    HeapWords,
    Requests,
    Count,
};
inline constexpr size_t NumRecordGroups = static_cast<size_t>(RecordGroup::Count);
inline constexpr std::array<const char*, NumRecordGroups> record_group_names{
    "pipeline", "views", "samplers", "reads",  "vertex",  "index",      "ubo",      "ssbo",
    "tbo",      "rt",    "dynamic",  "render area", "payload", "push", "heap words", "requests",
};

/// The resolve context of one draw.
struct ResolveContext {
    void* pipeline{}; ///< from PrepareDraw's ladder (R3 moves the lookup in)
    u64 pipeline_epoch{};
    bool is_indexed{};
    bool is_indirect{};
    bool frame_dump{};         ///< frame_dump.Capturing()
    bool sampled_recording{};  ///< the frame dump records sampled inputs
    u64 ordinal{};
    const ResolvedDraw* previous{}; ///< the pipeline's last record (record diff), may be null
    const DynamicFeatures* features{};
};

/// Compares a resolved record with the one observed from the old Configure. `created_mask`:
/// resource kinds (bit per RecordGroup) whose slot vectors grew during the draw; a proxy in
/// `resolved` matches any id there (dependent, not a mismatch). Appends the differing groups to
/// `reasons`; returns whether they are equal. `dependent` counts proxy matches.
[[nodiscard]] bool CompareRecords(const ResolvedDraw& resolved, const ResolvedDraw& observed,
                                  u32 created_mask, std::string& reasons, u32* mismatched_groups,
                                  u32* dependent);

/// Whether any handle of the record is still a commit proxy (not usable as a template).
[[nodiscard]] bool HasProxies(const ResolvedDraw& record);

struct RecordStats {
    std::atomic<u64> drawrec_draws{};
    std::array<std::atomic<u64>, 3> by_path{}; ///< full, delta, template
    std::array<std::atomic<u64>, NumCommitOnlyReasons> commit_only{};
    std::atomic<u64> stale{};
    std::atomic<u64> verify_draws{};
    std::array<std::atomic<u64>, NumRecordGroups> mismatches{};
    std::atomic<u64> dependent{};
    std::atomic<u64> purity_violations{};
    std::atomic<u64> deleted_loops{}; ///< ApplyDraw's deleted-images/buffers loops (R-7)
    /// S0-04: time stamp counter ticks per path (full, delta, template): the whole draw's
    /// resolve + apply + encode, and its resolve alone.
    std::array<std::atomic<u64>, 3> path_ticks{};
    std::array<std::atomic<u64>, 3> resolve_ticks{};
    /// F2-01: sub-records a resolve took from its base (uniform and texture buffer entries).
    std::atomic<u64> reused_ubos{};
    std::atomic<u64> reused_tbos{};
    std::atomic<u64> nearest_bases{}; ///< full resolves over a shape's template
};
[[nodiscard]] RecordStats& GetRecordStats();

/// Adds the ticks of one record-path draw (S0-04).
inline void CountRecordTime(size_t path, u64 total_ticks, u64 resolve_ticks) {
    RecordStats& stats = GetRecordStats();
    stats.path_ticks[path].fetch_add(total_ticks, std::memory_order_relaxed);
    stats.resolve_ticks[path].fetch_add(resolve_ticks, std::memory_order_relaxed);
}

/// Counts one relaxed increment (stats).
inline void CountRecordStat(std::atomic<u64>& counter, u64 value = 1) {
    counter.fetch_add(value, std::memory_order_relaxed);
}

/// The graphics descriptor tables a draw's view selects (SynchronizeDescriptors(false)'s inputs).
/// Here, not in state_view.h, which this header includes.
template <StateView View>
[[nodiscard]] DescriptorTablesKey TablesOf(const View& view) {
    const MaxwellRegs& regs = view.Regs();
    const bool via_header = regs.sampler_binding == MaxwellRegs::SamplerBinding::ViaHeaderBinding;
    const u32 tic_limit = regs.tex_header.limit;
    return DescriptorTablesKey{
        .tic_addr = regs.tex_header.Address(),
        .tic_limit = tic_limit,
        .tsc_addr = regs.tex_sampler.Address(),
        .tsc_limit = via_header ? tic_limit : regs.tex_sampler.limit,
        .via_header = via_header,
    };
}

/// The render target registers the texture cache's render target update reads.
template <StateView View>
[[nodiscard]] RtKey RtKeyOf(const View& view) {
    const MaxwellRegs& regs = view.Regs();
    RtKey key{};
    key.rt = regs.rt;
    key.zeta = regs.zeta;
    key.surface_clip = regs.surface_clip;
    key.rt_control = regs.rt_control;
    key.zeta_size = regs.zeta_size;
    key.zeta_enable = regs.zeta_enable;
    key.anti_alias_samples_mode = regs.anti_alias_samples_mode;
    return key;
}

/// The P6 stats-line part of the record path ("drawrec ..."), counters taken; empty when
/// nothing was counted.
[[nodiscard]] std::string TakeRecordStatsText(double seconds);

} // namespace VideoCommon::V2::Resolve
