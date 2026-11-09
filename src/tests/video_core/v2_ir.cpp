// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/ir.h"
#include "video_core/v2/ir_arena.h"
#include "video_core/v2/regsem_3d.h"

namespace Ir = VideoCommon::V2::Ir;

TEST_CASE("V2 IR records encode and decode", "[video_core][v2_ir]") {
    // A record with a payload reference and a body, as the split's emitters write them.
    std::array<u64, Ir::RecordWordsOf<Ir::BarrierBody>(true)> record{};
    const Ir::BarrierBody body{.op = Ir::BarrierOp::Invalidate, .object = 7, .a = 0x1000, .b = 64};
    record[0] = Ir::MakeHeader(Ir::Kind::Barrier, Ir::BodyWordsOf<Ir::BarrierBody>, record.size(),
                               Ir::HasPayload);
    record[1] = 42;
    record[2] = Ir::PackRef(123, 9);
    std::memcpy(record.data() + 3, &body, sizeof(body));
    const Ir::RecordView view = Ir::Decode(record.data());
    REQUIRE(view.kind == Ir::Kind::Barrier);
    REQUIRE(view.sequence == 42);
    REQUIRE(view.words == record.size());
    REQUIRE(view.body_words == Ir::BodyWordsOf<Ir::BarrierBody>);
    REQUIRE(view.has_payload);
    REQUIRE(view.payload.index == 123);
    REQUIRE(view.payload.words == 9);
    REQUIRE(view.As<Ir::BarrierBody>().op == Ir::BarrierOp::Invalidate);
    REQUIRE(view.As<Ir::BarrierBody>().object == 7);
    REQUIRE(view.As<Ir::BarrierBody>().a == 0x1000);
    REQUIRE(view.As<Ir::BarrierBody>().b == 64);
    // Without a payload the body follows the header.
    std::array<u64, Ir::RecordWordsOf<Ir::EngineBody>(false)> engine{};
    engine[0] = Ir::MakeHeader(Ir::Kind::Engine, 1, engine.size(), 0);
    engine[1] = 43;
    const Ir::EngineBody engine_body{5, 0};
    std::memcpy(engine.data() + 2, &engine_body, sizeof(engine_body));
    const Ir::RecordView engine_view = Ir::Decode(engine.data());
    REQUIRE(!engine_view.has_payload);
    REQUIRE(engine_view.As<Ir::EngineBody>().engine == 5);
    // Every kind has a name; the layouts are whole words.
    REQUIRE(Ir::BodyWordsOf<Ir::DrawBody> * 8 == sizeof(Ir::DrawBody));
    REQUIRE(Ir::BodyWordsOf<Ir::MapBody> * 8 == sizeof(Ir::MapBody));
}

TEST_CASE("V2 IR payload arena retires by executed sequence", "[video_core][v2_ir]") {
    Ir::PayloadArena arena;
    arena.Allocate(64, 8);
    // Spans are contiguous and tagged with their record.
    u64* const a = arena.Begin(10, 1, 100);
    REQUIRE(a != nullptr);
    u64* const b = arena.Begin(20, 2, 200);
    REQUIRE(b == a + 10);
    REQUIRE(arena.Used() == 30);
    // The newest span grows in place.
    u64* const more = arena.Extend(5);
    REQUIRE(more == b + 20);
    REQUIRE(arena.Used() == 35);
    // No room for 30 more: the oldest record must run first.
    REQUIRE(arena.Begin(30, 3, 300) == nullptr);
    REQUIRE(arena.Oldest().sequence == 1);
    REQUIRE(arena.Oldest().position == 100);
    arena.Retire(0);
    REQUIRE(arena.Used() == 35);
    arena.Retire(1);
    REQUIRE(arena.Used() == 25);
    // 29 words left before the end: a 30-word span wraps to the start, taking the rest with it.
    REQUIRE(arena.Begin(30, 3, 300) == nullptr); // 25 + 29 skipped + 30 > 64
    arena.Retire(2);
    REQUIRE(arena.Used() == 0);
    u64* const c = arena.Begin(30, 3, 300);
    REQUIRE(c != nullptr);
    REQUIRE(arena.IndexOf(c) == 0);
    REQUIRE(arena.Used() == 30 + 29);
    // A wrapped span can't be extended past the end, nor a span that ends at it.
    u64* const d = arena.Begin(34, 4, 400);
    REQUIRE(d == nullptr);
    arena.Retire(3);
    REQUIRE(arena.Used() == 0);
    u64* const e = arena.Begin(34, 4, 400);
    REQUIRE(arena.IndexOf(e) == 30);
    REQUIRE(arena.Extend(1) == nullptr); // Ends at the arena's end
    arena.Retire(4);
    // Marks are bounded too.
    for (u64 i = 0; i < 8; ++i) {
        REQUIRE(arena.Begin(1, 10 + i, 1000 + i) != nullptr);
    }
    REQUIRE(arena.Begin(1, 18, 1018) == nullptr);
    arena.Retire(10);
    REQUIRE(arena.Begin(1, 18, 1018) != nullptr);
}

TEST_CASE("V2 IR payload arena keeps live spans intact under random traffic",
          "[video_core][v2_ir]") {
    // A model of the front (allocating, filling) and the back (executing in order, checking).
    Ir::PayloadArena arena;
    arena.Allocate(1024, 64);
    std::mt19937 rng{5};
    struct Live {
        u64 sequence;
        u32 index;
        u32 words;
    };
    std::deque<Live> live;
    u64 sequence = 0;
    u64 executed = 0;
    for (u32 step = 0; step < 200000; ++step) {
        if (rng() % 3 != 0) {
            const u32 words = 1 + static_cast<u32>(rng() % 200);
            u64* span = arena.Begin(words, sequence + 1, 0);
            if (span == nullptr) {
                arena.Retire(executed);
                span = arena.Begin(words, sequence + 1, 0);
            }
            if (span == nullptr) {
                continue; // The back must catch up.
            }
            ++sequence;
            for (u32 i = 0; i < words; ++i) {
                span[i] = (sequence << 16) | i;
            }
            if (rng() % 4 == 0) {
                const u32 extra = static_cast<u32>(rng() % 8);
                if (u64* const more = arena.Extend(extra)) {
                    for (u32 i = 0; i < extra; ++i) {
                        more[i] = (sequence << 16) | (words + i);
                    }
                    live.push_back({sequence, arena.IndexOf(span), words + extra});
                    continue;
                }
            }
            live.push_back({sequence, arena.IndexOf(span), words});
        } else if (!live.empty()) {
            // The back executes the oldest record: its payload is what the front wrote.
            const Live record = live.front();
            live.pop_front();
            const u64* const words = arena.At(record.index);
            for (u32 i = 0; i < record.words; ++i) {
                REQUIRE(words[i] == ((record.sequence << 16) | i));
            }
            executed = record.sequence;
        }
        REQUIRE(arena.Used() <= arena.Capacity());
    }
}

TEST_CASE("V2 IR object table interns pointers once", "[video_core][v2_ir]") {
    auto table = std::make_unique<Ir::ObjectTable>();
    std::vector<int> objects(Ir::ObjectTable::Capacity + 1);
    for (u32 i = 0; i < Ir::ObjectTable::Capacity; ++i) {
        REQUIRE(table->Intern(&objects[i]) == i);
    }
    for (u32 i = 0; i < Ir::ObjectTable::Capacity; i += 97) {
        REQUIRE(table->Intern(&objects[i]) == i);
        REQUIRE(table->As<int>(i) == &objects[i]);
    }
    REQUIRE(table->Intern(&objects.back()) == Ir::ObjectTable::None);
    table->Clear();
    REQUIRE(table->Size() == 0);
    REQUIRE(table->Intern(&objects.back()) == 0);
    REQUIRE(table->Intern(nullptr) == 1);
    REQUIRE(table->Get(1) == nullptr);
}

TEST_CASE("V2 compute register groups: compare-on-write and incremental fingerprints",
          "[video_core][v2_regsem]") {
    namespace RegSem = VideoCommon::V2::RegSem;
    using VideoCommon::V2::ComputeGroup;
    using VideoCommon::V2::ComputeGroupBit;
    const auto& groups = RegSem::TableComputeGroups();
    REQUIRE(groups[0x62] == ComputeGroup::Upload);
    REQUIRE(groups[0xad] == ComputeGroup::Launch);
    REQUIRE(groups[0xc8] == ComputeGroup::Qmd);
    REQUIRE(groups[0x106] == ComputeGroup::Qmd);
    REQUIRE(groups[0x107] == ComputeGroup::Misc); // the inline QMD launch
    REQUIRE(groups[0xb0] == ComputeGroup::Misc);
    REQUIRE(groups[0x557] == ComputeGroup::Pools);
    REQUIRE(groups[0x6c3] == ComputeGroup::Misc);
    REQUIRE(groups[0x200] == ComputeGroup::Pipeline);
    auto mirror = std::make_unique<RegSem::MirrorCompute>();
    RegSem::RehashCompute(mirror->v2_groups, mirror->registers.data());
    mirror->v2_groups.TakeDirty();
    const auto none = [](RegSem::Effect, u32, u32) { return true; };
    // A changed QMD word dirties Qmd and moves its fingerprint; the same value does neither.
    RegSem::ApplyCompute(*mirror, 0xd0, 5, none);
    REQUIRE(mirror->v2_groups.TakeDirty() == ComputeGroupBit(ComputeGroup::Qmd));
    const u64 qmd = mirror->v2_groups.Fingerprint(ComputeGroupBit(ComputeGroup::Qmd));
    RegSem::ApplyCompute(*mirror, 0xd0, 5, none);
    REQUIRE(mirror->v2_groups.TakeDirty() == 0);
    REQUIRE(mirror->v2_groups.Fingerprint(ComputeGroupBit(ComputeGroup::Qmd)) == qmd);
    // Upload registers are tracked but not hashed.
    const u64 all = mirror->v2_groups.Fingerprint(VideoCommon::V2::HashedComputeGroups);
    RegSem::ApplyCompute(*mirror, 0x61, 7, none);
    REQUIRE(mirror->v2_groups.TakeDirty() == ComputeGroupBit(ComputeGroup::Upload));
    REQUIRE(mirror->v2_groups.Fingerprint(VideoCommon::V2::HashedComputeGroups) == all);
    // Random writes: incremental hashes equal a rehash; dispatch takes the dirty bits.
    std::mt19937 rng{11};
    for (u32 i = 0; i < 100000; ++i) {
        u32 method = static_cast<u32>(rng() % 0xe00);
        if (method >= 0x45 && method <= 0x48) {
            method = 0x100; // macro uploads and binds need a macro engine
        }
        RegSem::ApplyCompute(*mirror, method, static_cast<u32>(rng() % 4), none);
    }
    VideoCommon::V2::ComputeGroupTracker check = mirror->v2_groups;
    REQUIRE(RegSem::RehashCompute(check, mirror->registers.data()) == 0);
    RegSem::NoteDispatchCompute(mirror->v2_groups, mirror->registers.data(), 0x1000);
    REQUIRE(mirror->v2_groups.dirty == 0);
    REQUIRE(mirror->v2_groups.draw_dirty != 0);
}
