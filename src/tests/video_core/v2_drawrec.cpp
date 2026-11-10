// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// P6 Step 0 (notes/STEP0_SPEC.md): draw records, ResolveDraw / ApplyDraw / EncodeDraw.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/commit.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/resolve/dynamic_state.h"
#include "video_core/v2/v2_switches.h"

namespace V2 = VideoCommon::V2;
namespace R = VideoCommon::V2::Resolve;

namespace {

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // Anonymous namespace

TEST_CASE("V2 DRAWREC: the switch parses 0, 1 and verify", "[video_core][v2_drawrec]") {
    constexpr size_t index = static_cast<size_t>(V2::Switch::DrawRec);
    REQUIRE(std::string_view{V2::switch_names[index]} == "DRAWREC");
    const auto read = [&](const char* value) {
        SetEnv("EDEN_NX2_V2_DRAWREC", value);
        return V2::Detail::ReadAll()[index];
    };
    REQUIRE(read("0") == V2::Mode::Off);
    REQUIRE(read("1") == V2::Mode::On);
    REQUIRE(read("verify") == V2::Mode::Verify);
    REQUIRE(read("bogus") == V2::Mode::Off);
    SetEnv("EDEN_NX2_PIPELINE", "v1");
    REQUIRE(read("1") == V2::Mode::Off);
    SetEnv("EDEN_NX2_PIPELINE", "");
    REQUIRE(read("") == V2::Mode::Off);
}

namespace {

/// A record with two views, a sampler, a vertex binding and an index.
R::ResolvedDraw SampleRecord() {
    R::ResolvedDraw record;
    record.h.ordinal = 7;
    record.h.pipeline = reinterpret_cast<void*>(std::uintptr_t{0x1000});
    record.h.draw = R::ViewDraw{.first = 3, .count = 36, .instances = 1, .indexed = true};
    const std::array views{
        R::ResolvedView{.tic_index = 4, .view = {11, 2}, .stage = 4, .kind = 2},
        R::ResolvedView{.tic_index = 9, .view = {12, 0}, .stage = 4, .kind = 3},
    };
    record.h.views = record.Push(std::span<const R::ResolvedView>(views));
    const std::array samplers{R::ResolvedSampler{.tsc_index = 1, .sampler = {5, 1}}};
    record.h.samplers = record.Push(std::span<const R::ResolvedSampler>(samplers));
    const std::array vertex{
        R::ResolvedVertex{.slot = 0, .buffer = {3, 0}, .daddr = 0x20000, .size = 0x400, .stride = 16}};
    record.h.vertex = record.Push(std::span<const R::ResolvedVertex>(vertex));
    record.h.vertex_mask = 1;
    record.h.index = R::ResolvedIndex{
        .buffer = {4, 0}, .daddr = 0x30000, .size = 72, .format = 1, .first = 3, .count = 36, .valid = true};
    record.h.rt_known = true;
    record.h.rt_ids = {21, 0, 0, 0, 0, 0, 0, 0, 22};
    return record;
}

} // Anonymous namespace

TEST_CASE("V2 DRAWREC: record arena push, span and copy", "[video_core][v2_drawrec]") {
    R::ResolvedDraw record = SampleRecord();
    const auto views = record.Span<const R::ResolvedView>(record.h.views);
    REQUIRE(views.size() == 2);
    REQUIRE(views[0].tic_index == 4);
    REQUIRE(views[1].view.id == 12);
    REQUIRE(record.h.views.offset % 8 == 0);
    REQUIRE(record.h.samplers.offset % 8 == 0);
    REQUIRE(record.Span<const R::ResolvedSampler>(record.h.samplers)[0].sampler.id == 5);
    REQUIRE(record.Span<const R::ResolvedVertex>(record.h.vertex)[0].daddr == 0x20000);
    // An empty list is a valid segment.
    const R::Seg empty = record.Push(std::span<const R::ResolvedTbo>{});
    REQUIRE(empty.count == 0);
    REQUIRE(record.Span<const R::ResolvedTbo>(empty).empty());

    // A copy is the header and the arena: byte-equal.
    const R::ResolvedDraw copy = record;
    REQUIRE(std::memcmp(&copy.h, &record.h, sizeof(R::DrawHeader)) == 0);
    REQUIRE(copy.arena == record.arena);
    std::string reasons;
    REQUIRE(R::CompareRecords(copy, record, 0, reasons, nullptr, nullptr));
    REQUIRE(reasons.empty());

    record.Clear();
    REQUIRE(record.arena.empty());
    REQUIRE(record.h.views.count == 0);
}

TEST_CASE("V2 DRAWREC: a recipe keeps its record template", "[video_core][v2_drawrec]") {
    const R::ResolvedDraw record = SampleRecord();
    VideoCommon::V2::DrawCache::Recipe recipe;
    REQUIRE(!recipe.record.Valid());
    recipe.record.Store(record);
    REQUIRE(recipe.record.Valid());
    REQUIRE(std::memcmp(&recipe.record.Get().h, &record.h, sizeof(R::DrawHeader)) == 0);
    REQUIRE(recipe.record.Get().arena == record.arena);
    // A recipe copy is deep; Clear drops the template.
    const VideoCommon::V2::DrawCache::Recipe copy = recipe;
    recipe.Clear();
    REQUIRE(!recipe.record.Valid());
    REQUIRE(copy.record.Valid());
    REQUIRE(copy.record.Get().arena == record.arena);
    std::string reasons;
    REQUIRE(R::CompareRecords(copy.record.Get(), record, 0, reasons, nullptr, nullptr));
}

TEST_CASE("V2 DRAWREC: record compare names the differing groups", "[video_core][v2_drawrec]") {
    const R::ResolvedDraw resolved = SampleRecord();
    std::string reasons;
    u32 groups = 0;
    u32 dependent = 0;

    R::ResolvedDraw observed = SampleRecord();
    observed.Span<R::ResolvedView>(observed.h.views)[1].view.id = 13;
    REQUIRE_FALSE(R::CompareRecords(resolved, observed, 0, reasons, &groups, &dependent));
    REQUIRE(reasons == "views");
    REQUIRE(groups == (1U << static_cast<u32>(R::RecordGroup::Views)));

    observed = SampleRecord();
    observed.h.index.first = 4;
    observed.h.rt_ids[8] = 23;
    reasons.clear();
    REQUIRE_FALSE(R::CompareRecords(resolved, observed, 0, reasons, nullptr, nullptr));
    REQUIRE(reasons == "index, rt");

    observed = SampleRecord();
    observed.h.dynamic.values.line_width = 2.0f;
    observed.h.pipeline = nullptr;
    reasons.clear();
    REQUIRE_FALSE(R::CompareRecords(resolved, observed, 0, reasons, nullptr, nullptr));
    REQUIRE(reasons == "pipeline, dynamic");
}

TEST_CASE("V2 DRAWREC: a proxy matches a resource created in the draw", "[video_core][v2_drawrec]") {
    // The resolve could not know the id a creation will get: its proxy matches any id created
    // during the draw (dependent), and is a mismatch when nothing of that kind was created.
    R::ResolvedDraw resolved = SampleRecord();
    resolved.Span<R::ResolvedView>(resolved.h.views)[0].view.id = V2::Commit::MakeProxy(0);
    resolved.h.vertex = resolved.Push(std::span<const R::ResolvedVertex>(std::array{
        R::ResolvedVertex{.slot = 0, .buffer = {V2::Commit::MakeProxy(1), 0}, .daddr = 0x20000,
                          .size = 0x400, .stride = 16}}));
    const R::ResolvedDraw observed = SampleRecord();
    std::string reasons;
    u32 dependent = 0;
    constexpr u32 views_created = 1U << static_cast<u32>(R::RecordGroup::Views);
    constexpr u32 buffers_created = 1U << static_cast<u32>(R::RecordGroup::Vertex);
    REQUIRE(R::CompareRecords(resolved, observed, views_created | buffers_created, reasons, nullptr,
                              &dependent));
    REQUIRE(dependent == 2);
    REQUIRE_FALSE(R::CompareRecords(resolved, observed, views_created, reasons, nullptr, nullptr));
    REQUIRE(reasons == "vertex");
}

namespace {

/// A register file of pseudo-random words with the fields the helpers index made valid.
std::vector<u32> RandomRegisters(u32 seed) {
    std::vector<u32> words(R::MaxwellRegs::NUM_REGS);
    u32 state = seed;
    for (u32& word : words) {
        state = state * 1664525U + 1013904223U;
        word = state;
    }
    auto& regs = *reinterpret_cast<R::MaxwellRegs*>(words.data());
    for (auto& rt : regs.rt) {
        rt.format = Tegra::RenderTargetFormat::NONE;
    }
    regs.rt[0].format = Tegra::RenderTargetFormat::A8B8G8R8_UNORM;
    regs.rt[1].format = Tegra::RenderTargetFormat::R32_UINT;
    regs.zeta.format = Tegra::DepthFormat::Z24_UNORM_S8_UINT;
    regs.depth_bias = 3.0f;
    regs.slope_scale_depth_bias = 0.5f;
    regs.depth_bias_clamp = 0.25f;
    regs.line_width_smooth = 1.5f;
    regs.line_width_aliased = 2.0f;
    return words;
}

} // Anonymous namespace

TEST_CASE("V2 DRAWREC: view helpers read what the registers hold", "[video_core][v2_drawrec]") {
    using Regs = R::MaxwellRegs;
    const std::vector<u32> words = RandomRegisters(0x5eed);
    const Regs& regs = *reinterpret_cast<const Regs*>(words.data());
    const R::CbufTable cbufs{};
    const V2::GroupTracker groups{};
    const R::SnapshotView view{words.data(), &cbufs, &groups, Regs::PrimitiveTopology::Triangles,
                               R::ViewDraw{}, false};

    const R::DescriptorTablesKey tables = R::TablesOf(view);
    REQUIRE(tables.tic_addr == regs.tex_header.Address());
    REQUIRE(tables.tic_limit == regs.tex_header.limit);
    REQUIRE(tables.tsc_addr == regs.tex_sampler.Address());
    const bool via_header = regs.sampler_binding == Regs::SamplerBinding::ViaHeaderBinding;
    REQUIRE(tables.tsc_limit == (via_header ? regs.tex_header.limit : regs.tex_sampler.limit));

    const R::RtKey key = R::RtKeyOf(view);
    REQUIRE(std::memcmp(key.rt.data(), regs.rt.data(), sizeof(regs.rt)) == 0);
    REQUIRE(std::memcmp(&key.zeta, &regs.zeta, sizeof(regs.zeta)) == 0);
    REQUIRE(key.zeta_enable == regs.zeta_enable);
    R::RtKey other = key;
    REQUIRE(other == key);
    other.zeta_enable ^= 1;
    REQUIRE_FALSE(other == key);

    R::DynamicFeatures features{};
    features.depth_bounds = true;
    features.smooth_lines = true;
    features.restart_topologies = 1U << static_cast<u32>(Regs::PrimitiveTopology::Triangles);
    features.vertex_input_dynamic = true;
    features.max_vertex_attributes = 32;
    features.max_vertex_bindings = 16;
    const R::DynamicState state = R::ComputeDynamic(view, features);
    const R::DynamicValues& v = state.values;
    REQUIRE(std::memcmp(state.viewport.transform.data(), regs.viewport_transform.data(),
                        sizeof(regs.viewport_transform)) == 0);
    REQUIRE(state.viewport.viewport_scale_offset_enabled == regs.viewport_scale_offset_enabled);
    // The old formulas (vk_rasterizer.cpp Update*), written out.
    REQUIRE(v.depth_bias_constant == regs.depth_bias / 2.0f);
    REQUIRE(v.depth_bias_slope == regs.slope_scale_depth_bias);
    REQUIRE(v.line_width ==
            (regs.line_anti_alias_enable ? regs.line_width_smooth : regs.line_width_aliased));
    REQUIRE(v.depth_bounds_test == (regs.depth_bounds_enable != 0 ? 1 : 0));
    REQUIRE(v.rasterizer_discard == (regs.rasterize_enable == 0 ? 1 : 0));
    REQUIRE(v.primitive_restart == (regs.primitive_restart.enabled != 0 ? 1 : 0));
    REQUIRE(v.depth_bias_enable == (regs.polygon_offset_fill_enable != 0 ? 1 : 0));
    REQUIRE(v.logic_op_enable == (regs.logic_op.enable != 0 ? 1 : 0));
    for (size_t index = 0; index < Regs::NumRenderTargets; ++index) {
        const auto& mask = regs.color_mask[regs.color_mask_common ? 0 : index];
        const u8 bits = static_cast<u8>((mask.R ? 1 : 0) | (mask.G ? 2 : 0) | (mask.B ? 4 : 0) |
                                        (mask.A ? 8 : 0));
        REQUIRE(v.color_masks[index] == bits);
    }
    // An integer target never blends.
    REQUIRE(v.blend_enables[1] == 0);
    REQUIRE(v.blend_enables[0] == (regs.blend.enable[0] != 0 ? 1 : 0));
    REQUIRE(v.num_attributes == 32);
    REQUIRE(v.num_bindings == 16);
    for (u32 index = 0; index < 32; ++index) {
        const Regs::VertexAttribute attribute{regs.vertex_attrib_format[index]};
        REQUIRE(v.attributes[index].enabled ==
                ((!attribute.constant && attribute.buffer < 16) ? 1 : 0));
        REQUIRE(v.attributes[index].offset == attribute.offset);
    }

    // The D24 rescale only applies to D24 formats, and the AMD rule only with float attributes.
    features.d24_bias_rescale = true;
    REQUIRE(R::ComputeDynamic(view, features).values.depth_bias_constant !=
            regs.depth_bias / 2.0f);
    features.amd_logic_op = true;
    std::vector<u32> amd_words = words;
    auto& amd_regs = *reinterpret_cast<Regs*>(amd_words.data());
    amd_regs.logic_op.enable = 1;
    for (auto& attrib : amd_regs.vertex_attrib_format) {
        attrib.type.Assign(Regs::VertexAttribute::Type::UnusedEnumDoNotUseBecauseItWillGoAway);
    }
    REQUIRE(R::EffectiveLogicOpEnable(amd_regs, true));
    amd_regs.vertex_attrib_format[3].type.Assign(Regs::VertexAttribute::Type::Float);
    REQUIRE_FALSE(R::EffectiveLogicOpEnable(amd_regs, true));
    REQUIRE(R::EffectiveLogicOpEnable(amd_regs, false));
    // A snapshot of the same registers resolves the same dynamic state.
    const R::SnapshotView again{words.data(), &cbufs, &groups, Regs::PrimitiveTopology::Triangles,
                                R::ViewDraw{}, true};
    features = {};
    const R::DynamicState lhs = R::ComputeDynamic(view, features);
    const R::DynamicState rhs = R::ComputeDynamic(again, features);
    REQUIRE(std::memcmp(&lhs, &rhs, sizeof(lhs)) == 0);
}
