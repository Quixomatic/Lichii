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

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/commit.h"
#include "video_core/v2/resolve/draw_record.h"
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
