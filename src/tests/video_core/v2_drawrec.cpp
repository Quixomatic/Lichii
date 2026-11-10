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

#include "video_core/v2/cbuf_ring.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/hazard.h"
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

TEST_CASE("V2 DRAWREC: a miss base is peeked without touching the recipe table",
          "[video_core][v2_drawrec]") {
    // F2-01: MissBase finds the shape's last template through Peek, which must not change the
    // table's replacement order (DRAWREC=0 keeps the same evictions).
    namespace DC = VideoCommon::V2::DrawCache;
    DC::RecipeTable table{1}; // one set of 4 ways
    bool evicted = false;
    for (u64 key = 1; key <= 4; ++key) {
        table.Insert(key, evicted);
    }
    table.Insert(1, evicted).record.Store(SampleRecord()); // 1 is the newest now
    const DC::Recipe* const peeked = table.Peek(2);
    REQUIRE(peeked != nullptr);
    REQUIRE(!peeked->record.Valid());
    REQUIRE(table.Peek(1)->record.Valid());
    REQUIRE(table.Peek(9) == nullptr);
    table.Insert(5, evicted); // 2 stays the oldest: the peek did not touch it
    REQUIRE(evicted);
    REQUIRE(table.Peek(2) == nullptr);
    REQUIRE(table.Peek(3) != nullptr);
}

TEST_CASE("V2 DRAWREC: the stats line reports reuse and time per path",
          "[video_core][v2_drawrec]") {
    R::RecordStats& stats = R::GetRecordStats();
    static_cast<void>(R::TakeRecordStatsText(1.0)); // the counter's origin, old counts taken
    R::CountRecordStat(stats.drawrec_draws, 3);
    R::CountRecordStat(stats.by_path[2], 3);
    R::CountRecordStat(stats.reused_ubos, 7);
    R::CountRecordStat(stats.reused_tbos, 2);
    R::CountRecordTime(2, 3000, 1200);
    const std::string text = R::TakeRecordStatsText(1.0);
    REQUIRE(text.find("template 3") != std::string::npos);
    REQUIRE(text.find("reused ubo 7/s tbo 2/s") != std::string::npos);
    REQUIRE(stats.path_ticks[2].load() == 0);
    REQUIRE(stats.resolve_ticks[2].load() == 0);
    REQUIRE(stats.reused_ubos.load() == 0);
    // A record's uniform keeps the VA it was bound at (reuse compares it with the view's).
    REQUIRE(R::ResolvedUbo{}.gpu_addr == 0);
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

TEST_CASE("V2 DRAWREC: a table update to a known view compares as that view",
          "[video_core][v2_drawrec]") {
    // F2e (c236: views differ on 13% of draws): a ViewBind proxy (the TIC entry changed, its
    // view is cached) creates nothing, so the created rule rejected it. Verify replaces it with
    // the view the lookup found; creations stay proxies.
    namespace C = V2::Commit;
    R::ResolvedDraw resolved = SampleRecord();
    C::RequestList textures;
    textures.Reset(1);
    C::Request bind{};
    bind.kind = C::Kind::ViewBind;
    bind.resource = C::Resource::ImageView;
    bind.slot = 11;
    const u32 bind_proxy = textures.Push(bind, true);
    C::Request create{};
    create.kind = C::Kind::ViewCreate;
    create.resource = C::Resource::ImageView;
    const u32 create_proxy = textures.Push(create, true);
    auto views = resolved.Span<R::ResolvedView>(resolved.h.views);
    views[0].view = {bind_proxy, 0};
    views[1].view = {create_proxy, 0};

    const R::ResolvedDraw observed = SampleRecord();
    std::string reasons;
    REQUIRE_FALSE(R::CompareRecords(resolved, observed, 0, reasons, nullptr, nullptr));
    REQUIRE(R::PredictKnownViewProxies(resolved, textures.Requests()) == 1);
    REQUIRE(views[0].view.id == 11);
    REQUIRE(C::IsProxy(views[1].view.id));
    constexpr u32 views_created = 1U << static_cast<u32>(R::RecordGroup::Views);
    u32 dependent = 0;
    reasons.clear();
    REQUIRE(R::CompareRecords(resolved, observed, views_created, reasons, nullptr, &dependent));
    REQUIRE(dependent == 1);
    // A wrong prediction stays a mismatch.
    views[0].view.id = 10;
    reasons.clear();
    REQUIRE_FALSE(R::CompareRecords(resolved, observed, views_created, reasons, nullptr, nullptr));
    REQUIRE(reasons == "views");
}

TEST_CASE("V2 DRAWREC: a mismatch log names the first differing entry of its group",
          "[video_core][v2_drawrec]") {
    const R::ResolvedDraw resolved = SampleRecord();
    R::ResolvedDraw observed = SampleRecord();
    observed.Span<R::ResolvedView>(observed.h.views)[1].view.id = 13;
    observed.h.index.size = 0x258;
    const std::string views = R::DescribeRecordGroup(resolved, observed, R::RecordGroup::Views);
    REQUIRE(views.find("[1]") != std::string::npos);
    REQUIRE(views.find("view 12") != std::string::npos);
    REQUIRE(views.find("view 13") != std::string::npos);
    const std::string index = R::DescribeRecordGroup(resolved, observed, R::RecordGroup::Index);
    REQUIRE(index.find("size 0x48") != std::string::npos);
    REQUIRE(index.find("size 0x258") != std::string::npos);
}

TEST_CASE("V2 F2-02: the version ring retires by timeline value", "[video_core][v2_drawrec]") {
    V2::CbufRing::Ring ring{0x1000};
    // Front: aligned allocations; a tail too short is skipped, the ring never overruns.
    REQUIRE(ring.Allocate(0x100, 0x100) == 0);
    REQUIRE(ring.Allocate(0x10, 0x100) == 0x100);
    REQUIRE(ring.Allocate(0xd00, 0x100) == 0x200);
    REQUIRE(ring.Head() == 0xf00);
    REQUIRE_FALSE(ring.Allocate(0x200, 0x100).has_value()); // would wrap over unretired bytes
    REQUIRE(ring.Refusals() == 1);
    // Back: the bind floor is head - capacity / 2; nothing below it is bound again, so it is
    // free once no running tick references it.
    u64 completed = 0;
    const auto is_free = [&](u64 tick) { return tick <= completed; };
    ring.Roll(5, is_free);
    REQUIRE(ring.Floor() == 0x700);
    REQUIRE_FALSE(ring.Usable(0x200));
    REQUIRE(ring.Usable(0x700));
    ring.Use(5, 0x700);
    REQUIRE(ring.Retired() == 0x700);
    const auto wrapped = ring.Allocate(0x200, 0x100);
    REQUIRE(wrapped == 0x1000); // the tail is skipped
    REQUIRE(ring.Offset(*wrapped) == 0);
    ring.Roll(6, is_free);
    REQUIRE(ring.Floor() == 0xa00);
    REQUIRE_FALSE(ring.Usable(0x700));
    REQUIRE(ring.Retired() == 0x700); // tick 5 still runs and holds 0x700
    REQUIRE_FALSE(ring.Allocate(0x700, 0x100).has_value());
    REQUIRE(ring.Refusals() == 2);
    completed = 5;
    ring.Roll(7, is_free);
    REQUIRE(ring.Retired() == 0xa00);
    REQUIRE(ring.Allocate(0x700, 0x100) == 0x1200);
}

namespace {

/// A guest memory of 0x10000 bytes at GPU address 0 and a hazard answer for the decode tests.
struct DecodeEnv {
    std::vector<u8> memory = std::vector<u8>(0x10000);
    u32 writers = 0;
    u32 reads = 0;
    u32 Writers(u64, u32) {
        return writers;
    }
    bool Read(u64 va, u32 size, u8* dst) {
        ++reads;
        std::memcpy(dst, memory.data() + va, size);
        return true;
    }
};

struct Published {
    u64 va;
    u32 size;
    u64 position;
};

std::vector<Published> PublishAll(V2::CbufRing::Decode& decode) {
    std::vector<Published> out;
    decode.Publish([&](u64 va, u32 size, u64 position) { out.push_back({va, size, position}); });
    return out;
}

} // Anonymous namespace

TEST_CASE("V2 F2-02: constant buffer versions are made at decode", "[video_core][v2_drawrec]") {
    static V2::CbufRing::Shared shared;
    std::vector<u8> mapped(V2::CbufRing::Shared::Capacity);
    shared.Init(mapped.data(), 256);
    V2::CbufRing::Decode decode{shared};
    DecodeEnv env;
    constexpr u32 cbuf_bit = V2::ProducerBit(V2::Producer::CbData);
    for (u32 i = 0; i < 0x100; ++i) {
        env.memory[0x1000 + i] = static_cast<u8>(i);
    }
    // First write to a window with nothing in flight: guest memory is the base.
    const u32 a = 0xaabbccdd;
    decode.Write(0x1000, 0x100, 0x10, &a, 1, cbuf_bit, env);
    const u32 b = 0x11223344;
    decode.Write(0x1000, 0x100, 0x14, &b, 1, cbuf_bit, env); // patched in place
    REQUIRE(env.reads == 1);
    auto published = PublishAll(decode);
    REQUIRE(published.size() == 1);
    REQUIRE(published[0].va == 0x1000);
    REQUIRE(published[0].size == 0x100);
    const u64 first = published[0].position;
    const u8* const ring = mapped.data() + shared.ring.Offset(first);
    REQUIRE(ring[0x20] == 0x20);
    u32 word = 0;
    std::memcpy(&word, ring + 0x10, 4);
    REQUIRE(word == a);
    std::memcpy(&word, ring + 0x14, 4);
    REQUIRE(word == b);
    REQUIRE(PublishAll(decode).empty()); // nothing changed since the draw
    // A write after the draw: a new version. Only this data in flight: the previous version.
    env.writers = cbuf_bit;
    const u32 c = 0x55667788;
    decode.Write(0x1000, 0x100, 0x10, &c, 1, cbuf_bit, env);
    REQUIRE(env.reads == 1);
    published = PublishAll(decode);
    REQUIRE(published.size() == 1);
    REQUIRE(published[0].position != first);
    const u8* const second = mapped.data() + shared.ring.Offset(published[0].position);
    std::memcpy(&word, second + 0x10, 4);
    REQUIRE(word == c);
    std::memcpy(&word, second + 0x14, 4);
    REQUIRE(word == b);
    std::memcpy(&word, ring + 0x10, 4);
    REQUIRE(word == a); // published versions never change
    // Another producer in flight over the window: no version, the back drops its own.
    env.writers = cbuf_bit | V2::ProducerBit(V2::Producer::Copy);
    decode.Write(0x1000, 0x100, 0x10, &a, 1, cbuf_bit, env);
    published = PublishAll(decode);
    REQUIRE(published.size() == 1);
    REQUIRE(published[0].position == V2::CbufRing::NoPosition);
    // Writes outside the window or windows too large for the stream path make none either.
    env.writers = 0;
    decode.Write(0x2000, 0x10, 0x10, &a, 1, cbuf_bit, env);
    decode.Write(0x3000, 0x10000, 0, &a, 1, cbuf_bit, env);
    published = PublishAll(decode);
    REQUIRE(published.size() == 2);
    REQUIRE(published[0].position == V2::CbufRing::NoPosition);
    REQUIRE(published[1].position == V2::CbufRing::NoPosition);
}

TEST_CASE("V2 F2-02: the back table maps versions by device address", "[video_core][v2_drawrec]") {
    V2::CbufRing::BackTable table;
    table.Set(0x8000, 0x100, 0x400);
    REQUIRE(table.Find(0x8000, 0x80) != nullptr);
    REQUIRE(table.Find(0x8000, 0x80)->position == 0x400);
    REQUIRE(table.Find(0x8000, 0x200) == nullptr); // the bind reads more than the version
    REQUIRE(table.Find(0x8040, 0x40) == nullptr);
    table.Drop(0x8000);
    REQUIRE(table.Find(0x8000, 0x80) == nullptr);
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
