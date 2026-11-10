// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// F2-03 (notes/F2-03_DESIGN.md): the one-shot constant buffer write watch.

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/cbuf_ring.h"
#include "video_core/v2/cbuf_watch.h"
#include "video_core/v2/hazard.h"
#include "video_core/v2/v2_switches.h"
#include "video_core/v2/versions.h"

namespace V2 = VideoCommon::V2;
namespace W = VideoCommon::V2::CbufWatch;

namespace {

/// The device memory's cached counts, per page.
struct FakeCounts {
    std::map<u64, int> counts;
    u32 holds = 0; ///< Hold calls (batched protection changes)

    void Hold(std::span<const std::pair<u64, u64>> ranges, s32 delta) {
        ++holds;
        for (const auto& [addr, size] : ranges) {
            for (u64 page = addr; page < addr + size; page += W::PageSize) {
                counts[page] += delta;
            }
        }
    }
    bool TryHold(u64 page) {
        if (counts[page] < 1) {
            return false;
        }
        ++counts[page];
        return true;
    }
    int Count(u64 page) {
        return counts[page];
    }
};

constexpr u64 PageA = 0x10000;
constexpr u64 PageB = 0x11000;

} // Anonymous namespace

TEST_CASE("V2 F2-03: an armed page traps once and gives its count back", "[video_core][v2_cbufwatch]") {
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    REQUIRE_FALSE(watch->Take(PageA + 0x100, 0x100).armed);
    watch->RequestArm(PageA + 0x100, 0x100);
    watch->RequestArm(PageA + 0x200, 0x100); // one request per page
    REQUIRE(watch->stats.requests.load() == 1);
    arming.Tick(1, false, counts);
    REQUIRE(counts.Count(PageA) == 1);
    REQUIRE(arming.Armed() == 1);
    const W::Snap snap = watch->Take(PageA + 0x100, 0x100);
    REQUIRE(snap.armed);
    // Foreign writes elsewhere and re-requests change nothing.
    watch->NoteForeign(PageB, 0x10);
    watch->RequestArm(PageA, 0x10);
    REQUIRE(watch->Take(PageA + 0x100, 0x100).words == snap.words);
    // The first trapped store disarms the page and gives the count back on the trapping thread.
    u32 released = 0;
    const auto release = [&](u64 page) {
        ++released;
        counts.Hold(std::span<const std::pair<u64, u64>>{std::array{std::pair{page, W::PageSize}}},
                    -1);
    };
    watch->OnCpuWrite(PageA + 0x180, 4, release);
    watch->OnCpuWrite(PageA + 0x184, 4, release); // not armed any more: no trap
    REQUIRE(released == 1);
    REQUIRE(counts.Count(PageA) == 0);
    REQUIRE(watch->stats.traps.load() == 1);
    const W::Snap after = watch->Take(PageA + 0x100, 0x100);
    REQUIRE_FALSE(after.armed);
    REQUIRE(after.words != snap.words);
    // The back sees it disarmed: off the list, no second -1.
    arming.Tick(2, false, counts);
    REQUIRE(arming.Armed() == 0);
    REQUIRE(counts.Count(PageA) == 0);
    // A two-page window is armed only when both pages are.
    watch->RequestArm(PageA + 0xf00, 0x200);
    arming.Tick(40, false, counts);
    REQUIRE(watch->Take(PageA + 0xf00, 0x200).armed);
    REQUIRE(counts.Count(PageA) == 1);
    REQUIRE(counts.Count(PageB) == 1);
    // Released at shutdown: every count back.
    arming.ReleaseAll(counts);
    REQUIRE(counts.Count(PageA) == 0);
    REQUIRE(counts.Count(PageB) == 0);
    REQUIRE_FALSE(watch->Take(PageA + 0xf00, 0x200).armed);
}

TEST_CASE("V2 F2-03: a foreign write disarms an armed page with its count owed",
          "[video_core][v2_cbufwatch]") {
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    watch->RequestArm(PageA, 0x100);
    arming.Tick(1, false, counts);
    const W::Snap snap = watch->Take(PageA, 0x100);
    REQUIRE(snap.armed);
    // A snapshot after the note, before the bytes land, is unarmed: never trusted.
    watch->NoteForeign(PageA + 0x20, 0x10);
    REQUIRE(watch->stats.foreign_disarms.load() == 1);
    const W::Snap after = watch->Take(PageA, 0x100);
    REQUIRE_FALSE(after.armed);
    REQUIRE(after.words != snap.words);
    REQUIRE(counts.Count(PageA) == 1); // still protected until the back gives it back
    // A trap now finds nothing armed (no second -1).
    u32 released = 0;
    watch->OnCpuWrite(PageA, 4, [&](u64) { ++released; });
    REQUIRE(released == 0);
    arming.Tick(2, false, counts);
    REQUIRE(counts.Count(PageA) == 0);
    REQUIRE(arming.Armed() == 0);
    // An unarmed page the back used moves its sequence on a foreign write (the memo's key).
    watch->MarkUsed(PageB, 0x100);
    const W::Snap unarmed = watch->Take(PageB, 0x100);
    watch->NoteForeign(PageB + 0x80, 4);
    REQUIRE(watch->Take(PageB, 0x100).words != unarmed.words);
    REQUIRE(watch->stats.foreign_bumps.load() == 1);
}

TEST_CASE("V2 F2-03: a write while arming leaves the page unarmed with its count balanced",
          "[video_core][v2_cbufwatch]") {
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    // The foreign write lands between the +1 and the arm-complete CAS.
    struct Racing : FakeCounts {
        W::Watch* watch{};
        void Hold(std::span<const std::pair<u64, u64>> ranges, s32 delta) {
            FakeCounts::Hold(ranges, delta);
            if (delta > 0) {
                watch->NoteForeign(PageA, 4);
            }
        }
    } counts;
    counts.watch = watch.get();
    watch->RequestArm(PageA, 0x100);
    arming.Tick(1, false, counts);
    REQUIRE(arming.Armed() == 0);
    REQUIRE(counts.Count(PageA) == 0);
    REQUIRE_FALSE(watch->Take(PageA, 0x100).armed);
    // The request is cleared: the front asks again, the next tick arms it.
    counts.watch = nullptr;
    struct Plain : FakeCounts {} plain;
    watch->RequestArm(PageA, 0x100);
    arming.Tick(2, false, plain);
    REQUIRE(watch->Take(PageA, 0x100).armed);
    REQUIRE(plain.Count(PageA) == 1);
    // An unmap gives an armed page's count back at once; the back does not give it twice.
    u32 released = 0;
    watch->ReleaseRange(PageA, W::PageSize, [&](u64 page) {
        ++released;
        plain.Hold(std::span<const std::pair<u64, u64>>{std::array{std::pair{page, W::PageSize}}},
                   -1);
    });
    REQUIRE(released == 1);
    arming.Tick(3, false, plain);
    REQUIRE(plain.Count(PageA) == 0);
    REQUIRE(arming.Armed() == 0);
}

TEST_CASE("V2 F2-03: pages trapped soon after their arm back off longer", "[video_core][v2_cbufwatch]") {
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    const auto trap = [&] {
        watch->OnCpuWrite(PageA, 4, [&](u64 page) {
            counts.Hold(
                std::span<const std::pair<u64, u64>>{std::array{std::pair{page, W::PageSize}}}, -1);
        });
    };
    u64 frame = 1;
    // A hot page: trapped the frame after each arm; the backoff doubles to the cap.
    u32 expected = 1;
    for (int i = 0; i < 8; ++i) {
        watch->RequestArm(PageA, 0x100);
        // Not armed before its backoff ends; the request waits (one request outstanding).
        while (!watch->Take(PageA, 0x100).armed) {
            arming.Tick(frame++, false, counts);
        }
        trap();
        arming.Tick(frame++, false, counts);
        expected = (std::min)(expected * 2, W::Arming::MaxBackoff);
        REQUIRE(arming.Backoff(PageA) == expected);
    }
    REQUIRE(expected == W::Arming::MaxBackoff);
    REQUIRE(counts.Count(PageA) == 0);
    // A request inside the backoff stays waiting.
    watch->RequestArm(PageA, 0x100);
    arming.Tick(frame, false, counts);
    REQUIRE_FALSE(watch->Take(PageA, 0x100).armed);
    REQUIRE(arming.Waiting() == 1);
    // Armed again after it; trapped long after the arm: the backoff halves.
    frame += W::Arming::MaxBackoff;
    arming.Tick(frame, false, counts);
    REQUIRE(watch->Take(PageA, 0x100).armed);
    frame += 200;
    trap();
    arming.Tick(frame, false, counts);
    REQUIRE(arming.Backoff(PageA) == W::Arming::MaxBackoff / 2);
    REQUIRE(counts.Count(PageA) == 0);
}

TEST_CASE("V2 F2-03: reactive flushing arms only pages others already hold",
          "[video_core][v2_cbufwatch]") {
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    counts.counts[PageB] = 1; // the buffer cache tracks it
    watch->RequestArm(PageA, 0x100);
    watch->RequestArm(PageB, 0x100);
    arming.Tick(1, true, counts);
    REQUIRE_FALSE(watch->Take(PageA, 0x100).armed); // would change its protection
    REQUIRE(watch->Take(PageB, 0x100).armed);
    REQUIRE(counts.Count(PageA) == 0);
    REQUIRE(counts.Count(PageB) == 2);
    REQUIRE(counts.holds == 0); // no batched protection change
    // A page that could not be armed is retried only after a while.
    watch->RequestArm(PageA, 0x100);
    counts.counts[PageA] = 1;
    arming.Tick(2, true, counts);
    REQUIRE_FALSE(watch->Take(PageA, 0x100).armed);
    arming.Tick(2 + W::Arming::UnarmableRetry, true, counts);
    REQUIRE(watch->Take(PageA, 0x100).armed);
    arming.ReleaseAll(counts);
    REQUIRE(counts.Count(PageA) == 1);
    REQUIRE(counts.Count(PageB) == 1);
}

TEST_CASE("V2 F2-03: the armed list releases its oldest pages over the cap",
          "[video_core][v2_cbufwatch]") {
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    u64 frame = 1;
    const u64 total = W::Arming::MaxArmed + 300;
    for (u64 i = 0; i < total; ++i) {
        watch->RequestArm(0x100000 + i * W::PageSize, 4);
    }
    while (arming.Waiting() != 0 || watch->stats.arms.load() < total) {
        arming.Tick(frame++, false, counts);
        REQUIRE(frame < 100);
    }
    arming.Tick(frame++, false, counts);
    REQUIRE(arming.Armed() == W::Arming::MaxArmed);
    REQUIRE_FALSE(watch->Take(0x100000, 4).armed); // the oldest went
    REQUIRE(watch->Take(0x100000 + (total - 1) * W::PageSize, 4).armed);
    int held = 0;
    for (const auto& [page, count] : counts.counts) {
        REQUIRE((count == 0 || count == 1));
        held += count;
    }
    REQUIRE(held == static_cast<int>(W::Arming::MaxArmed));
    arming.ReleaseAll(counts);
    for (const auto& [page, count] : counts.counts) {
        REQUIRE(count == 0);
    }
}

TEST_CASE("V2 F2-03: over the cap the armed list keeps the pages binds use (CLOCK)",
          "[video_core][v2_cbufwatch]") {
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    u64 frame = 1;
    const auto page_at = [](u64 i) { return 0x100000 + i * W::PageSize; };
    for (u64 i = 0; i < W::Arming::MaxArmed; ++i) {
        watch->RequestArm(page_at(i), 4);
    }
    while (arming.Waiting() != 0 || watch->stats.arms.load() < W::Arming::MaxArmed) {
        arming.Tick(frame++, false, counts);
        REQUIRE(frame < 100);
    }
    // The oldest 16 pages are bound; the referenced bit is bookkeeping, not part of a snapshot.
    const W::Snap before = watch->Take(page_at(0), 4);
    for (u64 i = 0; i < 16; ++i) {
        watch->Touch(page_at(i), 4);
    }
    REQUIRE(watch->Take(page_at(0), 4).words == before.words);
    for (u64 i = 0; i < 32; ++i) {
        watch->RequestArm(page_at(W::Arming::MaxArmed + i), 4);
    }
    arming.Tick(frame++, false, counts);
    REQUIRE(arming.Armed() == W::Arming::MaxArmed);
    for (u64 i = 0; i < 16; ++i) {
        REQUIRE(watch->Take(page_at(i), 4).armed); // used: second chance
    }
    for (u64 i = 16; i < 48; ++i) {
        REQUIRE_FALSE(watch->Take(page_at(i), 4).armed); // the oldest unused went
    }
    REQUIRE(watch->Take(page_at(W::Arming::MaxArmed + 31), 4).armed);
    // Without new uses their second chance is spent: they go first next time.
    for (u64 i = 0; i < 16; ++i) {
        watch->RequestArm(page_at(2 * W::Arming::MaxArmed + i), 4);
    }
    arming.Tick(frame++, false, counts);
    for (u64 i = 0; i < 16; ++i) {
        REQUIRE_FALSE(watch->Take(page_at(i), 4).armed);
    }
    arming.ReleaseAll(counts);
    for (const auto& [page, count] : counts.counts) {
        REQUIRE(count == 0);
    }
}

namespace {

constexpr u64 DeviceBase = 0x100000;

/// Guest memory of 0x10000 bytes; GPU address va maps to device DeviceBase + (va & 0xffff), so
/// 0x11000 aliases 0x1000.
struct WatchEnv {
    std::vector<u8> memory = std::vector<u8>(0x10000);
    u32 writers = 0;
    u64 maps = 0;
    u32 same_calls = 0;
    u32 Writers(u64, u32) {
        return writers;
    }
    bool Read(u64 va, u32 size, u8* dst) {
        std::memcpy(dst, memory.data() + (va & 0xffff), size);
        return true;
    }
    bool Same(u64 va, u32 size, const u8* bytes) {
        ++same_calls;
        return std::memcmp(memory.data() + (va & 0xffff), bytes, size) == 0;
    }
    u64 shift = 0; ///< a remap moves every window
    u64 Device(u64 va, u32) {
        return DeviceBase + shift + (va & 0xffff);
    }
    u64 Maps() {
        return maps;
    }
    W::Watch* watch = nullptr; ///< the write watch (none: never armed)
    W::Snap Words(u64 device, u32 size) {
        return watch != nullptr ? watch->Take(device, size) : W::Snap{};
    }
    void Arm(u64 device, u32 size) {
        if (watch != nullptr) {
            watch->RequestArm(device, size);
        }
    }
};

struct Version {
    u64 va;
    u32 size;
    u64 position;
};

std::vector<Version> PublishAll(V2::CbufRing::Decode& decode) {
    std::vector<Version> out;
    decode.Publish([&](u64 va, u32 size, u64 position) { out.push_back({va, size, position}); });
    return out;
}

const Version* Find(const std::vector<Version>& versions, u64 va) {
    for (const Version& version : versions) {
        if (version.va == va) {
            return &version;
        }
    }
    return nullptr;
}

u32 WordAt(const std::vector<u8>& mapped, const V2::CbufRing::Shared& shared, u64 position,
           u32 offset) {
    u32 word = 0;
    std::memcpy(&word, mapped.data() + shared.ring.Offset(position) + offset, 4);
    return word;
}

} // Anonymous namespace

TEST_CASE("V2 F2-03: constant buffer data through one window reaches every window over it",
          "[video_core][v2_cbufwatch]") {
    static V2::CbufRing::Shared shared;
    std::vector<u8> mapped(V2::CbufRing::Shared::Capacity);
    shared.Init(mapped.data(), 256);
    auto decode = std::make_unique<V2::CbufRing::Decode>(shared, true);
    WatchEnv env;
    constexpr u32 cbuf_bit = V2::ProducerBit(V2::Producer::CbData);
    // B = [0x1080, 0x1180) overlaps A = [0x1000, 0x1100); C aliases A's device bytes.
    decode->Bind(0x1080, 0x100, cbuf_bit, env);
    decode->Bind(0x11000, 0x100, cbuf_bit, env);
    auto versions = PublishAll(*decode);
    REQUIRE(versions.size() == 2);
    const u64 first_b = Find(versions, 0x1080)->position;
    const u64 cows = shared.cow_versions.load();
    // Data through A over B's first bytes: B was published, so it is copied, then patched.
    const u32 x = 0x11111111;
    decode->Write(0x1000, 0x100, 0x80, &x, 1, cbuf_bit, env);
    REQUIRE(shared.cow_versions.load() == cows + 2); // B and C
    env.writers = cbuf_bit; // the data is in flight from now on: kept bases
    const u32 y = 0x22222222;
    decode->Write(0x1000, 0x100, 0x84, &y, 1, cbuf_bit, env);
    REQUIRE(shared.cow_versions.load() == cows + 2); // unpublished now: patched in place
    // The alias C gets A's bytes too (its first word lies outside B).
    const u32 z = 0x33333333;
    decode->Write(0x1000, 0x100, 0x00, &z, 1, cbuf_bit, env);
    versions = PublishAll(*decode);
    REQUIRE(versions.size() == 3);
    const Version* const b = Find(versions, 0x1080);
    const Version* const c = Find(versions, 0x11000);
    REQUIRE(b != nullptr);
    REQUIRE(c != nullptr);
    REQUIRE(b->position != first_b);
    REQUIRE(WordAt(mapped, shared, b->position, 0) == x);
    REQUIRE(WordAt(mapped, shared, b->position, 4) == y);
    REQUIRE(WordAt(mapped, shared, first_b, 0) == 0); // published versions never change
    REQUIRE(WordAt(mapped, shared, c->position, 0) == z);
    REQUIRE(WordAt(mapped, shared, c->position, 0x80) == x);
    // Another producer's ranged write over B's tail (not A's bytes) drops B only.
    decode->NoteWrite(DeviceBase + 0x1150, 4);
    versions = PublishAll(*decode);
    REQUIRE(versions.size() == 1);
    REQUIRE(versions[0].va == 0x1080);
    REQUIRE(versions[0].position == V2::CbufRing::NoPosition);
    // A window whose mapping moved loses its version (the back drops the old device's entry).
    env.maps = 1;
    env.shift = 0x10000;
    env.writers = 0;
    decode->Bind(0x1000, 0x100, cbuf_bit, env);
    versions = PublishAll(*decode);
    REQUIRE(Find(versions, 0x1000) != nullptr);
    for (const Version& version : versions) {
        REQUIRE(version.position == V2::CbufRing::NoPosition);
    }
}

TEST_CASE("V2 F2-03 (R2-F2-03): windows of one address and three sizes see each other's data, "
          "across unrelated maps",
          "[video_core][v2_cbufwatch]") {
    static V2::CbufRing::Shared shared;
    std::vector<u8> mapped(V2::CbufRing::Shared::Capacity);
    shared.Init(mapped.data(), 256);
    auto decode = std::make_unique<V2::CbufRing::Decode>(shared, true);
    WatchEnv env;
    constexpr u32 cbuf_bit = V2::ProducerBit(V2::Producer::CbData);
    constexpr u64 va = 0x4000;
    // The 0x40f4000 family of the 0fe4 verify log: one address bound with 0x210, 0x550, 0x790.
    decode->Bind(va, 0x210, cbuf_bit, env);
    decode->Bind(va, 0x550, cbuf_bit, env);
    decode->Bind(va, 0x790, cbuf_bit, env);
    REQUIRE(PublishAll(*decode).size() == 3);
    const u64 patches = shared.overlap_patches.load();
    env.maps = 1; // a map elsewhere: every window still at the same device address
    const u32 x = 0xaaaa0001;
    decode->Write(va, 0x210, 0x10, &x, 1, cbuf_bit, env);
    REQUIRE(shared.overlap_patches.load() == patches + 2);
    env.writers = cbuf_bit;
    const u32 y = 0xaaaa0002;
    decode->Write(va, 0x550, 0x500, &y, 1, cbuf_bit, env); // past 0x210: the largest only
    REQUIRE(shared.overlap_patches.load() == patches + 3);
    const u32 z = 0xaaaa0003;
    decode->Write(va, 0x790, 0x700, &z, 1, cbuf_bit, env); // past 0x550: nobody else
    REQUIRE(shared.overlap_patches.load() == patches + 3);
    const u32 w = 0xaaaa0004;
    decode->Write(va, 0x790, 0x20, &w, 1, cbuf_bit, env); // inside all three
    REQUIRE(shared.overlap_patches.load() == patches + 5);
    std::vector<V2::CbufRing::Published> versions;
    decode->Publish([&](const V2::CbufRing::Published& version) { versions.push_back(version); });
    REQUIRE(versions.size() == 3);
    for (const auto& version : versions) {
        REQUIRE(version.position != V2::CbufRing::NoPosition);
        REQUIRE(WordAt(mapped, shared, version.position, 0x10) == x);
        REQUIRE(WordAt(mapped, shared, version.position, 0x20) == w);
        if (version.size >= 0x550) {
            REQUIRE(WordAt(mapped, shared, version.position, 0x500) == y);
        }
        if (version.size == 0x790) {
            REQUIRE(WordAt(mapped, shared, version.position, 0x700) == z);
        }
    }
}

TEST_CASE("V2 F2-03: a version on armed pages is trusted until its pages see a write",
          "[video_core][v2_cbufwatch]") {
    static V2::CbufRing::Shared shared;
    std::vector<u8> mapped(V2::CbufRing::Shared::Capacity);
    shared.Init(mapped.data(), 256);
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    auto decode = std::make_unique<V2::CbufRing::Decode>(shared, true);
    WatchEnv env;
    env.watch = watch.get();
    constexpr u32 cbuf_bit = V2::ProducerBit(V2::Producer::CbData);
    constexpr u64 page = DeviceBase + 0x2000;
    env.memory[0x2000] = 0x42;
    // A version taken on an unarmed page asks for the arm and is not trusted.
    decode->Bind(0x2000, 0x100, cbuf_bit, env);
    auto versions = PublishAll(*decode);
    REQUIRE(versions.size() == 1);
    const u64 first = versions[0].position;
    REQUIRE(watch->stats.requests.load() == 1);
    arming.Tick(1, false, counts);
    REQUIRE(watch->Take(page, 0x100).armed);
    // Armed now: one compare at the next bind trusts it, and the back hears of it (same
    // position, no new version).
    const u32 upgrades = static_cast<u32>(shared.trust_upgrades.load());
    decode->Bind(0x2000, 0x100, cbuf_bit, env);
    REQUIRE(shared.trust_upgrades.load() == upgrades + 1);
    versions = PublishAll(*decode);
    REQUIRE(versions.size() == 1);
    REQUIRE(versions[0].position == first);
    // Trusted with its words unchanged: no compare at all.
    const u32 compares = env.same_calls;
    for (int i = 0; i < 4; ++i) {
        decode->Bind(0x2000, 0x100, cbuf_bit, env);
    }
    REQUIRE(env.same_calls == compares);
    REQUIRE(PublishAll(*decode).empty());
    // A trapped CPU store: the words move, the next bind compares and takes the new bytes.
    watch->OnCpuWrite(page + 0x10, 4, [&](u64 p) {
        counts.Hold(std::span<const std::pair<u64, u64>>{std::array{std::pair{p, W::PageSize}}},
                    -1);
    });
    env.memory[0x2010] = 0x99;
    decode->Bind(0x2000, 0x100, cbuf_bit, env);
    REQUIRE(env.same_calls == compares + 1);
    versions = PublishAll(*decode);
    REQUIRE(versions.size() == 1);
    REQUIRE(versions[0].position != first);
    REQUIRE(mapped[shared.ring.Offset(versions[0].position) + 0x10] == 0x99);
    REQUIRE(counts.Count(page) == 0);
    // A foreign write the front did not decode also ends the trust (the page is re-armed after
    // its backoff and trusted again after one compare).
    arming.Tick(2, false, counts);
    arming.Tick(10, false, counts);
    REQUIRE(watch->Take(page, 0x100).armed);
    decode->Bind(0x2000, 0x100, cbuf_bit, env); // trusted again (one compare)
    // The upgrade is published with the next draw (until then the window is dirty and binds
    // return before any check).
    REQUIRE(PublishAll(*decode).size() == 1);
    const u32 before_foreign = env.same_calls;
    decode->Bind(0x2000, 0x100, cbuf_bit, env);
    REQUIRE(env.same_calls == before_foreign);
    watch->NoteForeign(page + 0x20, 4);
    decode->Bind(0x2000, 0x100, cbuf_bit, env);
    REQUIRE(env.same_calls == before_foreign + 1);
    PublishAll(*decode);
    arming.ReleaseAll(counts);
    REQUIRE(counts.Count(page) == 0);
}

TEST_CASE("V2 F2-03: published versions carry their trust; bases read in flight never have it",
          "[video_core][v2_cbufwatch]") {
    static V2::CbufRing::Shared shared;
    std::vector<u8> mapped(V2::CbufRing::Shared::Capacity);
    shared.Init(mapped.data(), 256);
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    auto decode = std::make_unique<V2::CbufRing::Decode>(shared, true);
    WatchEnv env;
    env.watch = watch.get();
    constexpr u32 cbuf_bit = V2::ProducerBit(V2::Producer::CbData);
    const auto publish = [&] {
        std::vector<V2::CbufRing::Published> out;
        decode->Publish([&](const V2::CbufRing::Published& version) { out.push_back(version); });
        return out;
    };
    watch->RequestArm(DeviceBase + 0x3000, 0x2000);
    arming.Tick(1, false, counts);
    // Data on an armed window with nothing in flight: a trusted guest base, then patched.
    const u32 a = 0xa;
    decode->Write(0x3000, 0x100, 0, &a, 1, cbuf_bit, env);
    auto versions = publish();
    REQUIRE(versions.size() == 1);
    REQUIRE(versions[0].trusted);
    REQUIRE(versions[0].device == DeviceBase + 0x3000);
    REQUIRE(versions[0].words == watch->Take(DeviceBase + 0x3000, 0x100).words);
    // Its next version keeps the trusted base while only its data is in flight.
    env.writers = cbuf_bit;
    decode->Write(0x3000, 0x100, 4, &a, 1, cbuf_bit, env);
    versions = publish();
    REQUIRE(versions[0].trusted);
    // G2: a window without a version whose own data is in flight takes guest bytes that may
    // lack it: never trusted.
    decode->Write(0x4000, 0x100, 0, &a, 1, cbuf_bit, env);
    versions = publish();
    REQUIRE(versions.size() == 1);
    REQUIRE(versions[0].position != V2::CbufRing::NoPosition);
    REQUIRE_FALSE(versions[0].trusted);
    // A trap on the first window's page: its kept chain loses the trust.
    watch->OnCpuWrite(DeviceBase + 0x3000, 4, [&](u64 p) {
        counts.Hold(std::span<const std::pair<u64, u64>>{std::array{std::pair{p, W::PageSize}}},
                    -1);
    });
    decode->Write(0x3000, 0x100, 8, &a, 1, cbuf_bit, env);
    versions = publish();
    REQUIRE_FALSE(versions[0].trusted);
    // Data whose device address cannot be told untrusts every version (the epoch moves).
    const u32 epoch = shared.decode_epoch.load();
    struct Untranslatable : WatchEnv {
        u64 Device(u64, u32) {
            return V2::CbufRing::NoPosition;
        }
    } lost;
    lost.watch = watch.get();
    decode->Write(0x9000, 0x100, 0x200, &a, 1, cbuf_bit, lost); // outside its window
    REQUIRE(shared.decode_epoch.load() == epoch + 1);
    publish();
    // The back table keeps what the entries carried.
    V2::CbufRing::BackTable table;
    table.Set(0x8000, 0x100, 0x400, 0x1234, 7, true);
    const auto* const entry = table.Find(0x8000, 0x100);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->trusted);
    REQUIRE(entry->words == 0x1234);
    REQUIRE(entry->epoch == 7);
    REQUIRE(entry->checked_epoch == 0);
    arming.ReleaseAll(counts);
}

TEST_CASE("V2 F2-03: the back binds trusted versions without a compare, others once per epoch",
          "[video_core][v2_cbufwatch]") {
    using Path = V2::CbufRing::BindPath;
    auto watch = std::make_unique<W::Watch>();
    W::Arming arming{*watch};
    FakeCounts counts;
    constexpr u64 page = 0x50000;
    watch->RequestArm(page, 0x100);
    arming.Tick(1, false, counts);
    const W::Snap snap = watch->Take(page, 0x100);
    V2::CbufRing::BackTable table;
    table.Set(page, 0x100, 0x400, snap.words, 3, true);
    const auto* const entry = table.Find(page, 0x100);
    const auto choose = [&](u64 collect_epoch, u32 decode_epoch = 3, u32 decoders = 1,
                            u32 size = 0x100) {
        return V2::CbufRing::ChooseBindPath(*entry, size, watch->Take(page, 0x100).words,
                                            collect_epoch, decode_epoch, decoders);
    };
    REQUIRE(choose(10) == Path::Trusted);
    REQUIRE(choose(11) == Path::Trusted); // across epochs too
    REQUIRE(choose(10, 4) == Path::Compared);    // the decoder untrusted everything since
    REQUIRE(choose(10, 3, 2) == Path::Compared); // two decoders: neither knows the other's windows
    // A trapped CPU store moves the words: compared, then memoised for the rest of the epoch.
    watch->OnCpuWrite(page + 8, 4, [&](u64 p) {
        counts.Hold(std::span<const std::pair<u64, u64>>{std::array{std::pair{p, W::PageSize}}},
                    -1);
    });
    REQUIRE(choose(10) == Path::Compared);
    watch->MarkUsed(page, 0x100);
    entry->checked_epoch = 10;
    entry->checked_words = watch->Take(page, 0x100).words;
    entry->checked_size = 0x40; // a shorter bind of the entry compared
    REQUIRE(choose(10, 3, 1, 0x40) == Path::Memo);
    REQUIRE(choose(10, 3, 1, 0x20) == Path::Memo);
    REQUIRE(choose(10) == Path::Compared); // its tail was never compared (R2-F2-02)
    entry->checked_size = 0x100;
    REQUIRE(choose(10) == Path::Memo);
    REQUIRE(choose(11) == Path::Compared); // a new collect epoch: CPU writes ordered since
    // A foreign write within the epoch ends the memo too (the page is unarmed: its words move).
    watch->NoteForeign(page + 0x40, 4);
    REQUIRE(choose(10) == Path::Compared);
    arming.Tick(2, false, counts);
    REQUIRE(counts.Count(page) == 0);
}

TEST_CASE("V2 F2-03: the collect epoch moves with the ring on and the draw cache off",
          "[video_core][v2_cbufwatch]") {
    namespace Versions = V2::Versions;
    auto& modes = V2::g_modes;
    const auto saved = modes;
    modes[static_cast<size_t>(V2::Switch::DrawCache)] = V2::Mode::Off;
    modes[static_cast<size_t>(V2::Switch::CbufRing)] = V2::Mode::Off;
    const u64 before = Versions::Epochs().Current();
    Versions::HookEpoch(Versions::EpochReason::Entry);
    REQUIRE(Versions::Epochs().Current() == before); // v1: no epochs
    modes[static_cast<size_t>(V2::Switch::CbufRing)] = V2::Mode::On;
    Versions::HookEpoch(Versions::EpochReason::Acquire);
    Versions::HookEpoch(Versions::EpochReason::Submit);
    REQUIRE(Versions::Epochs().Current() == before + 2);
    modes = saved;
}

TEST_CASE("V2 F2-13: a reader's stamp is kept while no write was logged since",
          "[video_core][v2_cbufwatch]") {
    namespace Versions = V2::Versions;
    auto versions = std::make_unique<Versions::PageVersions>();
    const u64 first = versions->ReaderStamp();
    REQUIRE(versions->ReaderStamp() == first); // no write since: no clock move
    versions->Note(0x1000, 4, Versions::Source::Host);
    REQUIRE_FALSE(versions->Unchanged(0x1000, 4, first)); // a write after a kept stamp is newer
    const u64 second = versions->ReaderStamp();
    REQUIRE(second > first);
    REQUIRE(versions->Unchanged(0x1000, 4, second));
    versions->Note(0x1000, 4, Versions::Source::Host);
    REQUIRE_FALSE(versions->Unchanged(0x1000, 4, second));
    versions->Reset();
    const u64 third = versions->ReaderStamp();
    versions->Note(0x2000, 4, Versions::Source::Host);
    REQUIRE_FALSE(versions->Unchanged(0x2000, 4, third)); // not a stamp from before the reset
}

TEST_CASE("V2 S0-G19: the back's collect epoch moves where its stream reaches the front's "
          "ordering points",
          "[video_core][v2_cbufwatch]") {
    namespace Versions = V2::Versions;
    auto epochs = std::make_unique<Versions::CollectEpoch>();
    const u64 start = epochs->Current();
    // The front noted points before ring positions 100 and 300 (two at 300: one move).
    epochs->AdvanceAt(Versions::EpochReason::Entry, 100);
    epochs->AdvanceAt(Versions::EpochReason::Acquire, 300);
    epochs->AdvanceAt(Versions::EpochReason::Wait, 300);
    REQUIRE(epochs->Current() == start); // the back has not reached them
    epochs->Reach(0);
    epochs->Reach(40);
    REQUIRE(epochs->Current() == start); // every encode of these entries: one epoch
    epochs->Reach(100);
    REQUIRE(epochs->Current() == start + 1);
    epochs->Reach(180);
    epochs->Reach(299);
    REQUIRE(epochs->Current() == start + 1);
    epochs->Reach(320); // past 300 without an entry at it
    REQUIRE(epochs->Current() == start + 2);
    epochs->Advance(Versions::EpochReason::Submit); // the back's own point
    REQUIRE(epochs->Current() == start + 3);
    // A full queue: dropped notes move the epoch at every entry up to the newest of them.
    for (u64 i = 0; i < Versions::CollectEpoch::NumNotes + 8; ++i) {
        epochs->AdvanceAt(Versions::EpochReason::Entry, 1000 + i * 10);
    }
    const u64 newest = 1000 + (Versions::CollectEpoch::NumNotes + 7) * 10;
    u64 now = epochs->Current();
    epochs->Reach(1000);
    REQUIRE(epochs->Current() == now + 1);
    for (u64 position = 1001; position < newest; position += 7) {
        now = epochs->Current();
        epochs->Reach(position);
        REQUIRE(epochs->Current() == now + 1); // never fewer moves than the notes asked
    }
    epochs->Reach(newest);
    now = epochs->Current();
    epochs->Reach(newest + 5);
    REQUIRE(epochs->Current() == now); // drained
    // The split stops: positions start over.
    epochs->AdvanceAt(Versions::EpochReason::Entry, newest + 100);
    epochs->ResetStream();
    now = epochs->Current();
    epochs->Reach(newest + 200);
    REQUIRE(epochs->Current() == now);
    epochs->AdvanceAt(Versions::EpochReason::Entry, 4);
    epochs->Reach(4);
    REQUIRE(epochs->Current() == now + 1);
}

TEST_CASE("V2 F2-03: a page the watch alone holds traps only writes", "[video_core][v2_cbufwatch]") {
    using W::Protection;
    using W::ProtectionOf;
    // The watch arms an uncached page: write-protected only (reads stay free with reactive
    // flushing on).
    REQUIRE(ProtectionOf(0, false) == Protection::None);
    REQUIRE(ProtectionOf(1, true) == Protection::Watched);
    // The buffer cache tracks it too: reactive flushing's protection; and back when it lets go.
    REQUIRE(ProtectionOf(2, true) == Protection::Cached);
    REQUIRE(ProtectionOf(1, true) == Protection::Watched);
    // A trap gives the watch's count back: uncached.
    REQUIRE(ProtectionOf(0, false) == Protection::None);
    // Another holder alone, with or without the watch's +1 on top: cached.
    REQUIRE(ProtectionOf(1, false) == Protection::Cached);
    REQUIRE(ProtectionOf(2, true) == Protection::Cached);
}
