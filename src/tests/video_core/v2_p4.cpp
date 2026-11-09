// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/draw_cache.h"
#include "video_core/v2/versions.h"

namespace {

namespace DC = VideoCommon::V2::DrawCache;
namespace Versions = VideoCommon::V2::Versions;
using Versions::PageVersions;
using Versions::Source;

constexpr u64 Page = u64{1} << PageVersions::PageBits;

} // Anonymous namespace

TEST_CASE("V2 P4: page versions follow writes after a stamp", "[video_core][v2_p4]") {
    PageVersions versions;
    const u64 v = versions.Stamp();
    // Nothing written: unchanged.
    REQUIRE(versions.Unchanged(0x10000, 64, v));
    // A write after the stamp: changed for the range and only near it.
    versions.Note(0x10010, 4, Source::Inline);
    REQUIRE_FALSE(versions.Unchanged(0x10000, 64, v));
    REQUIRE_FALSE(versions.Unchanged(0x10000 + 0x20, 4, v)); // same page
    REQUIRE(versions.Unchanged(0x40000, 64, v));
    // A new stamp vouches for it again.
    const u64 w = versions.Stamp();
    REQUIRE(versions.Unchanged(0x10000, 64, w));
    // A CPU write moves the page; a repeated one is a load and compare.
    versions.NoteCpuWrite(0x10008, 4);
    REQUIRE_FALSE(versions.Unchanged(0x10000, 64, w));
    const u64 before = versions.Newest(0x10000, 64);
    versions.NoteCpuWrite(0x10008, 4);
    REQUIRE(versions.Newest(0x10000, 64) == before);
    // Writes before a stamp never fail it.
    const u64 x = versions.Stamp();
    REQUIRE(versions.Unchanged(0x10000, Page, x));
}

TEST_CASE("V2 P4: long writes use regions, longer ones the floor", "[video_core][v2_p4]") {
    PageVersions versions;
    const u64 v = versions.Stamp();
    // 2 MiB (> MaxPageWrite pages): region slots.
    versions.Note(0x2000000, 2ULL << 20, Source::Gpu);
    REQUIRE_FALSE(versions.Unchanged(0x2000000 + 0x1000, 4, v));
    REQUIRE_FALSE(versions.Unchanged(0x2000000 + (2ULL << 20) - 4, 4, v));
    REQUIRE(versions.Unchanged(0x8000000, 4, v));
    // A huge write moves the floor: every range changed.
    const u64 w = versions.Stamp();
    versions.Note(0, 1ULL << 40, Source::Host);
    REQUIRE_FALSE(versions.Unchanged(0x8000000, 4, w));
    const u64 x = versions.Stamp();
    REQUIRE(versions.Unchanged(0x8000000, 4, x));
}

TEST_CASE("V2 P4: versions are safe to bump from many threads", "[video_core][v2_p4]") {
    PageVersions versions;
    const u64 v = versions.Stamp();
    std::vector<std::thread> threads;
    for (u32 t = 0; t < 4; ++t) {
        threads.emplace_back([&versions, t] {
            for (u32 i = 0; i < 20000; ++i) {
                versions.NoteCpuWrite(0x100000 + (u64{t} << 20) + (i % 64) * Page, 4);
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    for (u32 t = 0; t < 4; ++t) {
        REQUIRE_FALSE(versions.Unchanged(0x100000 + (u64{t} << 20), 4, v));
    }
    REQUIRE(versions.Unchanged(0x900000, 4, versions.Stamp()));
}

TEST_CASE("V2 P4: the epoch memo answers repeats and sees writes to its range only",
          "[video_core][v2_p4]") {
    PageVersions versions;
    Versions::EpochMemo memo{versions};
    const u64 v = versions.Stamp();
    const DAddr image = 0x400000;
    const u64 size = 64 * Page;
    REQUIRE(memo.Unchanged(image, size, v));
    REQUIRE(memo.Unchanged(image, size, v));
    auto counters = memo.TakeCounters();
    REQUIRE(counters.queries == 2);
    REQUIRE(counters.hits == 1);
    REQUIRE(counters.walks == 1);

    // A guest CPU write to the range is seen at the next query, as a walk sees it.
    versions.NoteCpuWrite(image + 5 * Page, 4);
    REQUIRE_FALSE(memo.Unchanged(image, size, v));
    REQUIRE_FALSE(memo.UnchangedNow(image, size, v));

    // A stream write (inline upload, DMA, GPU) is seen at once.
    const u64 w = versions.Stamp();
    REQUIRE(memo.Unchanged(image, size, w));
    versions.Note(image + 7 * Page, 16, Source::Dma);
    REQUIRE_FALSE(memo.Unchanged(image, size, w));

    // Thousands of writes elsewhere (per-draw constant buffer data) keep it answering without a
    // walk: it looks only at the range's own write buckets.
    const u64 x = versions.Stamp();
    REQUIRE(memo.Unchanged(image, size, x));
    static_cast<void>(memo.TakeCounters());
    for (u32 i = 0; i < 5000; ++i) {
        versions.Note(0x9000000 + (i % 64) * 16, 16, Source::Inline);
        versions.NoteCpuWrite(0x9800000 + (i % 16) * Page, 4);
        REQUIRE(memo.Unchanged(image, size, x));
    }
    counters = memo.TakeCounters();
    REQUIRE(counters.walks == 0);
    REQUIRE(counters.hits == 5000);
    // An epoch change alone keeps it too.
    Versions::Epochs().Advance(Versions::EpochReason::Submit);
    REQUIRE(memo.Unchanged(image, size, x));
    REQUIRE(memo.TakeCounters().walks == 0);

    // A write over the range among them: seen, then answered from the new walk.
    versions.Note(image + 3 * Page, 16, Source::Inline);
    REQUIRE_FALSE(memo.Unchanged(image, size, x));
    const u64 y = versions.Stamp();
    REQUIRE(memo.Unchanged(image, size, y));
    REQUIRE(memo.Unchanged(image, size, y));
    counters = memo.TakeCounters();
    REQUIRE(counters.walks == 1);
    REQUIRE(counters.hits == 2);

    // Long writes (region slots) and floor writes reach every range they cover.
    versions.Note(image - 64 * Page, 256 * Page, Source::Gpu);
    REQUIRE_FALSE(memo.Unchanged(image, size, y));
    const u64 z = versions.Stamp();
    REQUIRE(memo.Unchanged(image, size, z));
    versions.Note(0, u64{1} << 40, Source::Host);
    REQUIRE_FALSE(memo.Unchanged(image, size, z));

    // A long range (over 4 MiB): checked at 4 MiB, the same answers.
    const DAddr big = 0x10000000;
    const u64 big_size = 16ULL << 20;
    const u64 s = versions.Stamp();
    REQUIRE(memo.Unchanged(big, big_size, s));
    versions.Note(0x20000000, 16, Source::Inline);
    REQUIRE(memo.Unchanged(big, big_size, s));
    versions.Note(big + (9ULL << 20), 16, Source::Inline);
    REQUIRE_FALSE(memo.Unchanged(big, big_size, s));
    REQUIRE(memo.Unchanged(big, big_size, versions.Stamp()));
}

TEST_CASE("V2 P4: the epoch memo agrees with a walk on random writes and ranges",
          "[video_core][v2_p4]") {
    PageVersions versions;
    Versions::EpochMemo memo{versions};
    std::mt19937_64 rng{7};
    std::vector<std::pair<DAddr, u64>> ranges;
    for (u32 i = 0; i < 64; ++i) {
        const DAddr address = (rng() % 4096) * Page + (rng() % 2 == 0 ? 0 : 0x123);
        const u64 size = (rng() % 4 == 0) ? (1 + rng() % 2048) * Page : 1 + rng() % (8 * Page);
        ranges.emplace_back(address, size);
    }
    u64 stamp = versions.Stamp();
    for (u32 step = 0; step < 200000; ++step) {
        const u32 kind = static_cast<u32>(rng() % 100);
        if (kind < 30) {
            const DAddr address = (rng() % 4096) * Page + rng() % Page;
            const u64 size = rng() % 8 == 0 ? (1 + rng() % 600) * Page : 1 + rng() % 256;
            versions.Note(address, size, Source::Inline);
        } else if (kind < 40) {
            versions.NoteCpuWrite((rng() % 4096) * Page + rng() % 4000, 4);
        } else if (kind < 42) {
            stamp = versions.Stamp();
        } else {
            const auto& [address, size] = ranges[rng() % ranges.size()];
            REQUIRE(memo.Unchanged(address, size, stamp) ==
                    memo.UnchangedNow(address, size, stamp));
        }
    }
}

TEST_CASE("V2 P4: page watch holds and lets go of tracked pages", "[video_core][v2_p4]") {
    Versions::PageWatch watch;
    std::array<s32, 16> counts{};
    watch.SetTracker([&](DAddr address, u64 size, s32 delta) {
        REQUIRE(size == Page);
        counts[address / Page] += delta;
    });
    watch.Hold(1 * Page + 8, 4);
    watch.Hold(1 * Page + 16, 4);
    watch.Hold(2 * Page - 4, 8); // pages 1 and 2
    REQUIRE(counts[1] == 1);
    REQUIRE(counts[2] == 1);
    REQUIRE(watch.Pages() == 2);
    watch.Drop(1 * Page + 8, 4);
    watch.Drop(1 * Page + 16, 4);
    watch.Drop(2 * Page - 4, 8);
    // Idle pages stay tracked until the idle list overflows (cheap re-holds).
    REQUIRE(counts[1] == 1);
    watch.Clear();
    REQUIRE(counts[1] == 0);
    REQUIRE(counts[2] == 0);
    REQUIRE(watch.Pages() == 0);
}

TEST_CASE("V2 P4: generations invalidate weak handles", "[video_core][v2_p4]") {
    Versions::Generations gens;
    REQUIRE(gens.Get(Versions::Kind::ImageView, 5) == 0);
    gens.Bump(Versions::Kind::ImageView, 5);
    REQUIRE(gens.Get(Versions::Kind::ImageView, 5) == 1);
    REQUIRE(gens.Get(Versions::Kind::Image, 5) == 0);
    REQUIRE(gens.Get(Versions::Kind::ImageView, 70000) == 0);
    gens.Bump(Versions::Kind::ImageView, 70000);
    REQUIRE(gens.Get(Versions::Kind::ImageView, 70000) == 1);
    // The invalid slot id never matches a bumped generation and never moves.
    REQUIRE(gens.Get(Versions::Kind::ImageView, 0xffffffffU) == ~0U);
    gens.Bump(Versions::Kind::ImageView, 0xffffffffU);
    REQUIRE(gens.Get(Versions::Kind::ImageView, 0xffffffffU) == ~0U);

    DC::Recipe recipe;
    recipe.views.push_back({.index = 3, .id = 5, .generation = 1});
    recipe.samplers.push_back({.id = 2, .generation = 0});
    recipe.targets[0] = {7, 0};
    recipe.targets[8] = {0xffffffffU, ~0U};
    REQUIRE(DC::HandlesAlive(recipe, gens));
    gens.Bump(Versions::Kind::ImageView, 7); // a render target view recreated
    REQUIRE_FALSE(DC::HandlesAlive(recipe, gens));
    recipe.targets[0].generation = 1;
    REQUIRE(DC::HandlesAlive(recipe, gens));
    gens.Bump(Versions::Kind::Sampler, 2);
    REQUIRE_FALSE(DC::HandlesAlive(recipe, gens));
    recipe.samplers[0].generation = 1;
    gens.Bump(Versions::Kind::ImageView, 5); // rescale/invalidation freed the view
    REQUIRE_FALSE(DC::HandlesAlive(recipe, gens));

    const u64 epoch = gens.GetEpoch(Versions::Epoch::Pipelines);
    gens.BumpEpoch(Versions::Epoch::Pipelines);
    REQUIRE(gens.GetEpoch(Versions::Epoch::Pipelines) == epoch + 1);
}

TEST_CASE("V2 P4: captured words are compared with memory on every check", "[video_core][v2_p4]") {
    PageVersions versions;
    Versions::EpochMemo memo{versions};
    // Guest memory stand-ins: a constant buffer page with a handle and a TIC entry.
    std::array<u8, 64> cbuf{};
    std::array<u8, 32> tic{};
    const u32 handle = 0x1234;
    std::memcpy(&cbuf[16], &handle, 4);
    tic.fill(0xab);

    DC::CapturedInputs inputs;
    inputs.stamp = versions.Stamp();
    DC::Capture word{.host = &cbuf[16], .device = 0x10010, .size = 4,
                     .what = DC::Capture::What::Handle};
    std::memcpy(word.bytes.data(), &handle, 4);
    inputs.words.push_back(word);
    DC::Capture header{.host = tic.data(), .device = 0x20000, .size = 32,
                       .what = DC::Capture::What::Tic};
    std::memcpy(header.bytes.data(), tic.data(), 32);
    inputs.words.push_back(header);
    inputs.content.push_back({0x400000, 16 * Page, true});
    inputs.content.push_back({0x800000, Page, false});

    // Nothing written: the words compare equal.
    REQUIRE(DC::ValidateInputs(inputs, memo, versions) == DC::Miss::Count);

    // Per-draw uniforms written on the handle page, the handle itself untouched: still equal.
    versions.Note(0x10020, 16, Source::Inline);
    REQUIRE(DC::ValidateInputs(inputs, memo, versions) == DC::Miss::Count);

    // The handle rewritten by a write no page version saw (an untrapped CPU write): a miss.
    const u32 other = 0x5678;
    std::memcpy(&cbuf[16], &other, 4);
    REQUIRE(DC::ValidateInputs(inputs, memo, versions) == DC::Miss::Handle);
    std::memcpy(&cbuf[16], &handle, 4);

    // A texture header changed the same way.
    tic[3] = 0;
    REQUIRE(DC::ValidateInputs(inputs, memo, versions) == DC::Miss::Tic);
    tic[3] = 0xab;
    REQUIRE(DC::ValidateInputs(inputs, memo, versions) == DC::Miss::Count);

    // Buffer pages written: reported, never a miss.
    bool buffers = false;
    versions.Note(0x800010, 4, Source::Dma);
    REQUIRE(DC::ValidateInputs(inputs, memo, versions, true, nullptr, &buffers) ==
            DC::Miss::Count);
    REQUIRE(buffers);

    // A sampled image's pages written: a miss whatever the words say.
    versions.Note(0x400000 + 3 * Page, 64, Source::Dma);
    log.Note(0x400000 + 3 * Page, 0x400000 + 3 * Page + 64);
    REQUIRE(DC::ValidateInputs(inputs, memo, versions) == DC::Miss::ImagePages);
}

TEST_CASE("V2 P4: a key gets a recipe on its second sighting", "[video_core][v2_p4]") {
    DC::SeenFilter seen;
    REQUIRE(!seen.Admit(0x1234));
    REQUIRE(seen.Admit(0x1234));
    REQUIRE(seen.Admit(0x1234));
    // Many unique keys never get in.
    u32 admitted = 0;
    for (u64 key = 1; key <= 100000; ++key) {
        admitted += seen.Admit(key * 0x9e3779b97f4a7c15ULL) ? 1 : 0;
    }
    REQUIRE(admitted == 0);
    seen.Clear();
    REQUIRE(!seen.Admit(0x1234));
    REQUIRE(!seen.Admit(0));
    REQUIRE(seen.Admit(0));
}

TEST_CASE("V2 P4: recipe table keys, LRU eviction and drops", "[video_core][v2_p4]") {
    DC::RecipeTable table{4}; // 4 sets x 4 ways
    REQUIRE(table.Capacity() == 16);
    std::vector<u64> dropped;
    const auto drop = [&](const DC::Recipe& recipe) { dropped.push_back(recipe.key); };
    // Fill: every insert of a new key into a set with a free way does not evict.
    bool evicted = false;
    for (u64 key = 100; key < 116; ++key) {
        DC::Recipe& recipe = table.Insert(key, evicted, drop);
        recipe.uses = key;
    }
    REQUIRE(table.Occupied() <= 16);
    // Re-insert of an existing key drops its old recipe but evicts nothing.
    const size_t before = table.Occupied();
    dropped.clear();
    if (table.Find(100) != nullptr) {
        DC::Recipe& again = table.Insert(100, evicted, drop);
        REQUIRE_FALSE(evicted);
        REQUIRE(dropped == std::vector<u64>{100});
        REQUIRE(again.uses == 0); // cleared
        REQUIRE(table.Occupied() == before);
    }
    // Many more keys: the table stays bounded and evicts the least recently used way.
    dropped.clear();
    u64 evictions = 0;
    for (u64 key = 1000; key < 1200; ++key) {
        table.Insert(key, evicted, drop);
        evictions += evicted ? 1 : 0;
    }
    REQUIRE(table.Occupied() == 16);
    REQUIRE(evictions == dropped.size());
    REQUIRE(evictions >= 200 - 16);
    // The most recent keys of each set survive; a touched key outlives its set's others.
    DC::RecipeTable lru{1}; // one set of 4 ways
    lru.Insert(1, evicted);
    lru.Insert(2, evicted);
    lru.Insert(3, evicted);
    lru.Insert(4, evicted);
    REQUIRE(lru.Find(1) != nullptr); // touch 1: 2 is now the oldest
    lru.Insert(5, evicted);
    REQUIRE(evicted);
    REQUIRE(lru.Find(2) == nullptr);
    REQUIRE(lru.Find(1) != nullptr);
    REQUIRE(lru.Find(5) != nullptr);
    lru.Erase(5);
    REQUIRE(lru.Find(5) == nullptr);
    REQUIRE(lru.Occupied() == 3);
    dropped.clear();
    lru.Clear(drop);
    REQUIRE(dropped.size() == 3);
    REQUIRE(lru.Occupied() == 0);
}

TEST_CASE("V2 P4: shape table and key mixing", "[video_core][v2_p4]") {
    DC::ShapeTable shapes;
    REQUIRE(shapes.Find(42) == nullptr);
    DC::ShapeInfo info{.shape = 42, .relevant_groups = 0x44};
    info.key_cbufs[0] = 0x3;
    shapes.Insert(info);
    const DC::ShapeInfo* const found = shapes.Find(42);
    REQUIRE(found != nullptr);
    REQUIRE(found->relevant_groups == 0x44);
    REQUIRE(found->key_cbufs[0] == 0x3);
    REQUIRE(shapes.Find(43) == nullptr);
    // Mixing is order dependent and spreads single-bit changes.
    REQUIRE(DC::Mix(DC::Mix(1, 2), 3) != DC::Mix(DC::Mix(1, 3), 2));
    REQUIRE(DC::Mix(1, 0x1000) != DC::Mix(1, 0x1001));
    const std::array<u32, 3> a{1, 2, 3};
    const std::array<u32, 3> b{1, 2, 4};
    REQUIRE(DC::HashWords(a, 7) != DC::HashWords(b, 7));
    REQUIRE(DC::HashWords(a, 7) == DC::HashWords(a, 7));
}

TEST_CASE("V2 P4: the fast-path ladder decides in order", "[video_core][v2_p4]") {
    DC::Recipe recipe;
    int lookups = 0;
    int validations = 0;
    const auto found = [&](DC::Miss&) -> DC::Recipe* {
        ++lookups;
        return &recipe;
    };
    const auto absent = [&](DC::Miss& miss) -> DC::Recipe* {
        ++lookups;
        miss = DC::Miss::NoShape;
        return nullptr;
    };
    const auto holds = [&](DC::Recipe&) {
        ++validations;
        return DC::Miss::Count;
    };
    const auto handle_changed = [&](DC::Recipe&) {
        ++validations;
        return DC::Miss::Handle;
    };
    int samples = 0;
    const auto no = [&] {
        ++samples;
        return false;
    };
    const auto yes = [&] {
        ++samples;
        return true;
    };

    // 1. A continued draw wins and looks nothing up.
    DC::Decision decision = DC::Decide(true, true, found, holds, no);
    REQUIRE(decision.path == DC::Path::Continued);
    REQUIRE(lookups == 0);
    REQUIRE(validations == 0);

    // The cache off or the draw ineligible: resolve.
    decision = DC::Decide(false, false, found, holds, no);
    REQUIRE(decision.path == DC::Path::Resolve);
    REQUIRE(decision.miss == DC::Miss::Ineligible);
    REQUIRE(lookups == 0);

    // 2. A validated recipe replays.
    decision = DC::Decide(false, true, found, holds, no);
    REQUIRE(decision.path == DC::Path::Replay);
    REQUIRE(decision.recipe == &recipe);
    REQUIRE_FALSE(decision.verify);
    REQUIRE(validations == 1);

    // A verify sample still decides Replay, flagged (the caller resolves and compares).
    decision = DC::Decide(false, true, found, holds, yes);
    REQUIRE(decision.path == DC::Path::Replay);
    REQUIRE(decision.verify);

    // The sampler is asked on hits only (1-in-N of hits, not of draws).
    REQUIRE(samples == 2);

    // 3. A changed input or no entry: resolve, with the reason.
    decision = DC::Decide(false, true, found, handle_changed, no);
    REQUIRE(decision.path == DC::Path::Resolve);
    REQUIRE(decision.miss == DC::Miss::Handle);
    REQUIRE(decision.invalidated);
    decision = DC::Decide(false, true, absent, holds, no);
    REQUIRE(decision.path == DC::Path::Resolve);
    REQUIRE(decision.miss == DC::Miss::NoShape);
    REQUIRE_FALSE(decision.invalidated);
    REQUIRE(samples == 2);

    // 1-in-N sampling.
    DC::VerifySampler every{1};
    REQUIRE(every.Next());
    REQUIRE(every.Next());
    DC::VerifySampler sampler{4};
    int sampled = 0;
    for (int i = 0; i < 16; ++i) {
        sampled += sampler.Next() ? 1 : 0;
    }
    REQUIRE(sampled == 4);
}

TEST_CASE("V2 P4: recipes compare field by field", "[video_core][v2_p4]") {
    DC::Recipe a;
    a.pipeline = reinterpret_cast<void*>(0x1000);
    a.views.push_back({.index = 1, .id = 2, .generation = 0});
    a.samplers.push_back({.id = 3, .generation = 0});
    a.storage.push_back({.stage = 4, .index = 0, .device_addr = 0x100, .size = 64});
    a.targets[0] = {9, 0};
    a.push_words = 2;
    a.push[0] = 1;
    DC::Recipe b = a;
    std::string reasons;
    REQUIRE(DC::CompareRecipes(a, b, reasons));
    REQUIRE(reasons.empty());
    b.views[0].id = 5;
    b.storage[0].size = 128;
    b.push[1] = 7;
    REQUIRE_FALSE(DC::CompareRecipes(a, b, reasons));
    REQUIRE(reasons.find("image views") != std::string::npos);
    REQUIRE(reasons.find("storage buffers") != std::string::npos);
    REQUIRE(reasons.find("push data") != std::string::npos);
    REQUIRE(reasons.find("pipeline") == std::string::npos);
    // Generations are not compared (a recipe rebuilt after a bump holds the same objects).
    DC::Recipe c = a;
    c.views[0].generation = 4;
    reasons.clear();
    REQUIRE(DC::CompareRecipes(a, c, reasons));
}

TEST_CASE("V2 P4: buffer recipes compare what a replay would bind", "[video_core][v2_p4]") {
    using Kind = DC::BufferBind::Kind;
    DC::BufferBind vb{.stage = 0, .index = 1, .kind = Kind::Cached, .buffer_id = 7,
                      .generation = 0, .device_addr = 0x1000, .size = 0x400};
    REQUIRE(vb.Contains(0x1000, 0x400));
    REQUIRE(vb.Contains(0x1100, 0x100));
    REQUIRE_FALSE(vb.Contains(0x0ff0, 0x20));
    REQUIRE_FALSE(vb.Contains(0x1300, 0x200));

    DC::BufferRecipe hit;
    hit.valid = true;
    hit.has_index = true;
    hit.index = {.buffer_id = 3, .device_addr = 0x8000, .size = 0x600};
    hit.vertex.push_back(vb);
    hit.uniform.push_back({.stage = 4, .index = 2, .kind = Kind::Version, .buffer_id = 9,
                           .device_addr = 0x20000, .size = 0x100});
    REQUIRE(DC::BufferRecipe::Find(hit.uniform, 4, 2) != nullptr);
    REQUIRE(DC::BufferRecipe::Find(hit.uniform, 4, 3) == nullptr);

    DC::BufferRecipe fresh = hit;
    std::string reasons;
    REQUIRE(DC::CompareBufferRecipes(hit, fresh, reasons));
    REQUIRE(reasons.empty());

    // Fewer indices this draw (a sub-range of the recorded one): same buffer, fine.
    fresh.index.size = 0x300;
    REQUIRE(DC::CompareBufferRecipes(hit, fresh, reasons));
    // A relocated uniform buffer (outside the recorded range): the replay syncs it, not compared.
    fresh.uniform[0].device_addr = 0x40000;
    fresh.uniform[0].buffer_id = 11;
    REQUIRE(DC::CompareBufferRecipes(hit, fresh, reasons));
    // The full resolve found another buffer for a range the replay would take: a mismatch.
    fresh.vertex[0].buffer_id = 8;
    REQUIRE_FALSE(DC::CompareBufferRecipes(hit, fresh, reasons));
    REQUIRE(reasons.find("buffers (vertex 0:1)") != std::string::npos);
    // A cached binding the full resolve now streams: a mismatch in path.
    fresh = hit;
    fresh.uniform[0].kind = Kind::Cached;
    reasons.clear();
    REQUIRE_FALSE(DC::CompareBufferRecipes(hit, fresh, reasons));

    // Clearing a recipe clears its buffers.
    DC::Recipe recipe;
    recipe.buffers = hit;
    recipe.Clear();
    REQUIRE_FALSE(recipe.buffers.valid);
    REQUIRE(recipe.buffers.vertex.empty());
}
