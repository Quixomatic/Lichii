// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/native_qmd.h"
#include "video_core/v2/sem_view.h"

// SYNC_SUBMIT_SPEC §2.1: S1's semaphore view.
namespace {

using namespace VideoCommon::V2;
using Retire = SemView::Retire;
using Kind = SemView::Kind;

const auto Reduce = [](u32 op, u32 base, u32 operand) {
    return Tegra::ReduceSemaphore(op, false, base, operand);
};

SemProgress At(u64 consumed, u64 landed = 0, u64 settled = 0) {
    return {.consumed = consumed, .landed = landed, .untracked_settled = settled};
}

} // Anonymous namespace

TEST_CASE("SYNC SemView: literal releases, newest writer, retirement", "[video_core][v2_sem_view]") {
    SemView view;
    constexpr u64 word = 0x1000;
    view.Note(word, 1, true, 10, Retire::Consumed, 0, Kind::Release, At(0));
    view.Note(word, 2, true, 20, Retire::Consumed, 0, Kind::Release, At(0));
    auto answer = view.Find(word, At(5));
    REQUIRE(answer.pending);
    REQUIRE(answer.writer.known);
    REQUIRE(answer.writer.value == 2);
    REQUIRE(answer.writer.position == 20);
    REQUIRE(answer.writer.has_prior);
    REQUIRE(answer.writer.prior == 1);
    // Writes to one word land in order: the newest retiring retires all older ones.
    REQUIRE(view.Find(word, At(19)).pending);
    REQUIRE_FALSE(view.Find(word, At(20)).pending);
    REQUIRE_FALSE(view.Find(word + 4, At(0)).pending);
}

TEST_CASE("SYNC SemView: retire by consumed, landing ticket and untracked settlement",
          "[video_core][v2_sem_view]") {
    SemView view;
    view.Note(0x100, 7, true, 10, Retire::Landing, 0x1234, Kind::Release, At(0));
    view.Note(0x200, 7, true, 10, Retire::Untracked, 10, Kind::QmdPayload, At(0));
    // A landing release stays pending after the back ran it, until its ticket landed.
    REQUIRE(view.Find(0x100, At(100, 0x1233)).pending);
    REQUIRE_FALSE(view.Find(0x100, At(100, 0x1234)).pending);
    // A QMD write: until the entry's untracked operations ran.
    REQUIRE(view.Find(0x200, At(100, 0, 9)).pending);
    REQUIRE(view.Find(0x200, At(9, 0, 100)).pending);
    REQUIRE_FALSE(view.Find(0x200, At(10, 0, 10)).pending);
}

TEST_CASE("SYNC SemView: reductions over a pending payload and over memory",
          "[video_core][v2_sem_view]") {
    SemView view;
    constexpr u64 word = 0x2000;
    u32 memory = 100;
    const auto read = [&memory] { return memory; };
    // A chain over a pending payload release.
    view.Note(word, 5, true, 10, Retire::Untracked, 10, Kind::QmdPayload, At(0));
    view.NoteReduction(word, false, 0, 3, 11, Retire::Untracked, 11, Kind::QmdReduction, At(0),
                       true, Reduce, read);
    view.NoteReduction(word, false, 0, 2, 12, Retire::Untracked, 12, Kind::QmdReduction, At(0),
                       true, Reduce, read);
    auto answer = view.Find(word, At(0));
    REQUIRE(answer.writer.known);
    REQUIRE_FALSE(answer.writer.bound);
    REQUIRE(answer.writer.value == 10);
    // Nothing pending: memory holds every earlier GPU write and is the base.
    view.NoteReduction(word + 4, false, 0, 0x13, 20, Retire::Consumed, 0, Kind::QmdReduction,
                       At(0), true, Reduce, read);
    REQUIRE(view.Find(word + 4, At(0)).writer.value == 0x113);
    // An incomplete base (an unknown writer may still change memory): a lower bound for ADD,
    // unknown for an op that may lower the word.
    view.NoteReduction(word + 8, false, 0, 1, 20, Retire::Consumed, 0, Kind::QmdReduction, At(0),
                       false, Reduce, read);
    answer = view.Find(word + 8, At(0));
    REQUIRE(answer.writer.known);
    REQUIRE(answer.writer.bound);
    view.NoteReduction(word + 12, false, 5, 1, 20, Retire::Consumed, 0, Kind::QmdReduction,
                       At(0), false, Reduce, read);
    REQUIRE_FALSE(view.Find(word + 12, At(0)).writer.known);
}

TEST_CASE("SYNC SemView: a max over an unknown base is unknown", "[video_core][v2_sem_view]") {
    SemView view;
    constexpr u64 word = 0x3000;
    view.Note(word, 0, false, 10, Retire::Landing, 5, Kind::Counter, At(0));
    view.NoteReduction(word, true, 0, 9, 11, Retire::Untracked, 11, Kind::QmdMax, At(0), true,
                       Reduce, [] { return 0U; });
    const auto answer = view.Find(word, At(0));
    REQUIRE(answer.pending);
    REQUIRE_FALSE(answer.writer.known);
    REQUIRE(answer.writer.position == 11);
}

TEST_CASE("SYNC SemView: a CPU write with no writer pending decides by memory",
          "[video_core][v2_sem_view]") {
    SemView view;
    constexpr u64 word = 0x4000;
    view.Note(word, 1, true, 10, Retire::Consumed, 0, Kind::Release, At(0));
    const auto answer = view.Find(word, At(10));
    REQUIRE_FALSE(answer.pending);
    SemInputs in;
    in.memory_word = 1;
    auto decision = DecideFromView(SemaphoreOp::Equal, 2, answer, in);
    REQUIRE(decision.verdict == AcquireVerdict::NotMet);
    REQUIRE((decision.wake & SemWake::Cpu) != 0);
    in.memory_word = 2; // The guest CPU stored the value
    decision = DecideFromView(SemaphoreOp::Equal, 2, answer, in);
    REQUIRE(decision.verdict == AcquireVerdict::Met);
}

TEST_CASE("SYNC SemView: overflow answers unknown at the newest evicted position",
          "[video_core][v2_sem_view]") {
    SemView view;
    // Twice the slots, all pending: the windows fill and the oldest writers are evicted.
    constexpr u64 count = SemView::Slots * 2;
    for (u64 i = 0; i < count; ++i) {
        view.Note(0x10000 + i * 4, static_cast<u32>(i), true, i + 1, Retire::Consumed, 0,
                  Kind::Release, At(0));
    }
    REQUIRE(view.TakeEvictions() > 0);
    u64 unknown = 0;
    u64 newest_evicted = 0;
    for (u64 i = 0; i < count; ++i) {
        const auto answer = view.Find(0x10000 + i * 4, At(0));
        REQUIRE(answer.pending);
        if (answer.overflow) {
            ++unknown;
            REQUIRE_FALSE(answer.writer.known);
            newest_evicted = answer.writer.position;
        } else {
            REQUIRE(answer.writer.value == static_cast<u32>(i));
        }
    }
    REQUIRE(unknown > 0);
    REQUIRE(newest_evicted <= count);
    // Conservative until the back passed the newest evicted writer, then memory decides.
    REQUIRE_FALSE(view.Find(0x10000, At(count, 0, count)).pending);
    view.Clear();
    REQUIRE_FALSE(view.Find(0x10000, At(0)).pending);
}
