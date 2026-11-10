// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// F2-03 (notes/F2-03_DESIGN.md): the one-shot constant buffer write watch.

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/cbuf_watch.h"

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
