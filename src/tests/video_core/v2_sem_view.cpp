// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/native_qmd.h"
#include "video_core/v2/hazard.h"
#include "video_core/v2/sem_view.h"
#include "video_core/v2/submit_requests.h"

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
    REQUIRE(view.Find(word + 4, At(0)).writer.value == 100 + 0x13);
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

TEST_CASE("SYNC step 2: DecideFromView, every row of the table", "[video_core][v2_sem_view]") {
    SemView view;
    constexpr u64 word = 0x5000;
    SemInputs in;
    SECTION("unmapped, a map queued: suspend until it ran") {
        in.mapped = false;
        in.map_queued = true;
        const auto d = DecideFromView(SemaphoreOp::Gequal, 1, {}, in);
        REQUIRE(d.row == SemRow::MapQueued);
        REQUIRE(d.verdict == AcquireVerdict::NotMet);
        REQUIRE(d.wake == SemWake::Map);
    }
    SECTION("unmapped, no map queued: memory as the guest reads it (84de: gequal 0 passes)") {
        in.mapped = false;
        in.memory_word = 0;
        REQUIRE(DecideFromView(SemaphoreOp::Gequal, 0, {}, in).verdict == AcquireVerdict::Met);
        REQUIRE(DecideFromView(SemaphoreOp::Equal, 1, {}, in).verdict == AcquireVerdict::NotMet);
    }
    SECTION("GPU-written word: class C readback") {
        in.gpu_written = true;
        const auto d = DecideFromView(SemaphoreOp::Equal, 1, {}, in);
        REQUIRE(d.row == SemRow::GpuWritten);
        REQUIRE(d.verdict == AcquireVerdict::Fallback);
        REQUIRE(d.wake == SemWake::Readback);
    }
    SECTION("no pending writer: memory decides; Equal waits for a queued unknown QMD") {
        in.memory_word = 3;
        REQUIRE(DecideFromView(SemaphoreOp::Equal, 3, {}, in).verdict == AcquireVerdict::Met);
        const auto unmet = DecideFromView(SemaphoreOp::Gequal, 4, {}, in);
        REQUIRE(unmet.verdict == AcquireVerdict::NotMet);
        REQUIRE((unmet.wake & SemWake::Cpu) != 0);
        REQUIRE((unmet.wake & SemWake::ReleaseParsed) != 0);
        in.unbounded_unknown = 40;
        const auto equal = DecideFromView(SemaphoreOp::Equal, 3, {}, in);
        REQUIRE(equal.verdict == AcquireVerdict::NotMet);
        REQUIRE(equal.wake == SemWake::Consumed);
        REQUIRE(equal.position == 40);
        // A word that only advances: the unknown QMD can't move it back (§10).
        REQUIRE(DecideFromView(SemaphoreOp::Gequal, 3, {}, in).verdict == AcquireVerdict::Met);
    }
    SECTION("pending known writer: class B, or wait for a later release") {
        view.Note(word, 9, true, 50, Retire::Landing, 0x3200, Kind::Release, At(0));
        const auto answer = view.Find(word, At(60, 0x31ff));
        in.memory_word = 8;
        const auto met = DecideFromView(SemaphoreOp::Equal, 9, answer, in);
        REQUIRE(met.row == SemRow::PendingKnown);
        REQUIRE(met.verdict == AcquireVerdict::MetByQueued);
        REQUIRE(met.position == 50);
        const auto unmet = DecideFromView(SemaphoreOp::Gequal, 10, answer, in);
        REQUIRE(unmet.verdict == AcquireVerdict::NotMet);
        REQUIRE((unmet.wake & SemWake::ReleaseParsed) != 0);
        REQUIRE_FALSE(unmet.request);
    }
    SECTION("pending unknown writer at execute") {
        view.Note(word, 0, false, 70, Retire::Consumed, 0, Kind::Release, At(0));
        const auto answer = view.Find(word, At(60));
        in.memory_word = 5;
        // Gequal passes on memory with an unknown writer pending (§10 monotonic).
        REQUIRE(DecideFromView(SemaphoreOp::Gequal, 5, answer, in).verdict ==
                AcquireVerdict::Met);
        // Equal waits for it, then resumes once the back consumed it.
        const auto equal = DecideFromView(SemaphoreOp::Equal, 5, answer, in);
        REQUIRE(equal.row == SemRow::UnknownAtExecute);
        REQUIRE(equal.verdict == AcquireVerdict::NotMet);
        REQUIRE((equal.wake & SemWake::Consumed) != 0);
        REQUIRE(equal.position == 70);
        REQUIRE_FALSE(equal.request);
        const auto after = DecideFromView(SemaphoreOp::Equal, 5, view.Find(word, At(70)), in);
        REQUIRE(after.verdict == AcquireVerdict::Met);
    }
    SECTION("pending unknown writer at a fence (class D): one request for its ticket") {
        view.Note(word, 0, false, 80, Retire::Landing, 0x5000, Kind::Counter, At(0));
        in.memory_word = 1;
        const auto d = DecideFromView(SemaphoreOp::Equal, 2, view.Find(word, At(90)), in);
        REQUIRE(d.row == SemRow::UnknownAtFence);
        REQUIRE(d.verdict == AcquireVerdict::NotMet);
        REQUIRE(d.request);
        REQUIRE(d.ticket == 0x5000);
        REQUIRE((d.wake & SemWake::Landing) != 0);
    }
}

TEST_CASE("SYNC step 2 idle back: an Equal acquire on a word an unknown QMD writes resumes "
          "once the back ran it, without a submit",
          "[video_core][v2_sem_view]") {
    SemView view;
    view.NoteUnboundedUnknown(100);
    SemInputs in;
    in.memory_word = 7;
    in.unbounded_unknown = view.UnboundedUnknown(90);
    auto d = DecideFromView(SemaphoreOp::Equal, 7, view.Find(0x6000, At(90)), in);
    REQUIRE(d.verdict == AcquireVerdict::NotMet);
    REQUIRE(d.wake == SemWake::Consumed);
    REQUIRE_FALSE(d.request);
    // The back ran it and went idle: nothing to submit, memory decides.
    in.unbounded_unknown = view.UnboundedUnknown(100);
    d = DecideFromView(SemaphoreOp::Equal, 7, view.Find(0x6000, At(100)), in);
    REQUIRE(d.verdict == AcquireVerdict::Met);
    REQUIRE_FALSE(d.request);
}

TEST_CASE("SYNC step 3 idle back: a QMD reduction deferred behind a landing still answers a "
          "Gequal acquire by the semaphore view",
          "[video_core][v2_sem_view]") {
    // 8d31/019d: acquires 0x13 short of 19 ADD reductions. The front notes each reduction (QMD
    // writes retire by untracked settlement); one the landing rule defers stays pending at its
    // fence with the back idle, and the view already holds the value it leaves.
    SemView view;
    constexpr u64 word = 0x8000;
    const u32 memory = 0x100;
    for (u64 i = 0; i < 0x13; ++i) {
        view.NoteReduction(word, false, 0, 1, 10 + i, Retire::Untracked, 10 + i,
                           Kind::QmdReduction, At(0), true, Reduce, [memory] { return memory; });
    }
    // The back ran every entry; the last reductions' untracked operations haven't landed.
    const auto answer = view.Find(word, At(100, 0, 20));
    REQUIRE(answer.pending);
    REQUIRE(answer.writer.value == 0x113);
    SemInputs in;
    in.memory_word = 0x10a;
    const auto d = DecideFromView(SemaphoreOp::Gequal, 0x113, answer, in);
    REQUIRE(d.verdict == AcquireVerdict::MetByQueued);
    REQUIRE_FALSE(d.request);
    // Once they landed, memory holds the value and nothing is pending.
    REQUIRE_FALSE(view.Find(word, At(100, 0, 100)).pending);
}

TEST_CASE("SYNC step 4 idle back: every wake row of §2.2 resumes once its event happened",
          "[video_core][v2_sem_view]") {
    // With the back idle and nothing queued, each suspended state names the wake class whose
    // event resolves it; after that event (and nothing else) the same decision passes.
    SemView view;
    SemInputs in;
    SECTION("map: a queued map ran and mapped the word") {
        in.mapped = false;
        in.map_queued = true;
        auto d = DecideFromView(SemaphoreOp::Gequal, 1, {}, in);
        REQUIRE((d.wake & SemWake::Map) != 0);
        in.mapped = true;
        in.map_queued = false;
        in.memory_word = 1;
        d = DecideFromView(SemaphoreOp::Gequal, 1, view.Find(0x9000, At(10)), in);
        REQUIRE(d.verdict == AcquireVerdict::Met);
    }
    SECTION("consumed: the back ran an unknown writer") {
        view.Note(0x9000, 0, false, 20, Retire::Consumed, 0, Kind::Release, At(0));
        in.memory_word = 4;
        auto d = DecideFromView(SemaphoreOp::Equal, 4, view.Find(0x9000, At(10)), in);
        REQUIRE((d.wake & SemWake::Consumed) != 0);
        d = DecideFromView(SemaphoreOp::Equal, 4, view.Find(0x9000, At(20)), in);
        REQUIRE(d.verdict == AcquireVerdict::Met);
    }
    SECTION("landing: a counter report's operation ran") {
        view.Note(0x9000, 0, false, 20, Retire::Landing, 0x1400, Kind::Counter, At(0));
        in.memory_word = 4;
        auto d = DecideFromView(SemaphoreOp::Equal, 4, view.Find(0x9000, At(30, 0x13ff)), in);
        REQUIRE((d.wake & SemWake::Landing) != 0);
        REQUIRE(d.request);
        d = DecideFromView(SemaphoreOp::Equal, 4, view.Find(0x9000, At(30, 0x1400)), in);
        REQUIRE(d.verdict == AcquireVerdict::Met);
    }
    SECTION("readback: the GPU-written word was read back") {
        in.gpu_written = true;
        auto d = DecideFromView(SemaphoreOp::Equal, 4, {}, in);
        REQUIRE((d.wake & SemWake::Readback) != 0);
        in.gpu_written = false; // word_landed: memory holds it
        in.memory_word = 4;
        d = DecideFromView(SemaphoreOp::Equal, 4, {}, in);
        REQUIRE(d.verdict == AcquireVerdict::Met);
    }
    SECTION("CPU write: no writer, the guest stored the value") {
        in.memory_word = 3;
        auto d = DecideFromView(SemaphoreOp::Gequal, 4, view.Find(0x9000, At(10)), in);
        REQUIRE((d.wake & SemWake::Cpu) != 0);
        in.memory_word = 4;
        d = DecideFromView(SemaphoreOp::Gequal, 4, view.Find(0x9000, At(10)), in);
        REQUIRE(d.verdict == AcquireVerdict::Met);
    }
    SECTION("release parsed: another channel's release to the word was parsed") {
        in.memory_word = 3;
        auto d = DecideFromView(SemaphoreOp::Gequal, 4, view.Find(0x9000, At(10)), in);
        REQUIRE((d.wake & SemWake::ReleaseParsed) != 0);
        const u64 notes = view.Notes();
        view.Note(0x9000, 4, true, 30, Retire::Landing, 0x1e00, Kind::Release, At(10));
        REQUIRE(view.Notes() != notes); // the epoch the waiter watches moved
        d = DecideFromView(SemaphoreOp::Gequal, 4, view.Find(0x9000, At(10)), in);
        REQUIRE(d.verdict == AcquireVerdict::MetByQueued);
    }
}

TEST_CASE("SYNC step 4: no submit when nobody waits and the GPU is busy",
          "[video_core][v2_sem_view]") {
    // A suspended acquire with no producer at a fence asks for nothing.
    REQUIRE(PlanSuspendedPoll(false, false, 9, 0, true, true, false) == SuspendedPoll::Wait);
    // An idle hook with only a lazy fence's open tick, no request and DXVK's rule not due:
    // the work is left for the next boundary (no urgent-at-queue request exists any more).
    SubmitRequests requests;
    REQUIRE(requests.AtIdleHook(false, true, false, 9, 9, false) ==
            SubmitRequests::IdleAction::Leave);
}

TEST_CASE("SYNC step 6: macro DMA reads of semaphore words forward known values, wait only for "
          "unknown writers",
          "[video_core][v2_sem_view]") {
    // Maxwell3D's FrontDmaReadSem takes, per word: no pending writer → memory; a pending known
    // writer (a QMD reduction over a pending payload) → its value, no wait; an unknown writer
    // made at execute → its position; an unknown writer at a fence → the targeted landing (one
    // request for its ticket's tick, CompleteLanding; idle-back test: "V2 landing: the back lands
    // its captured operations for a waiting front").
    SemView view;
    constexpr u64 word = 0xa000;
    view.Note(word, 0x40, true, 10, Retire::Untracked, 10, Kind::QmdPayload, At(0));
    view.NoteReduction(word, false, 0, 2, 11, Retire::Untracked, 11, Kind::QmdReduction, At(0),
                       true, Reduce, [] { return 0U; });
    const auto known = view.Find(word, At(5));
    REQUIRE(known.pending);
    REQUIRE(known.writer.known);
    REQUIRE(known.writer.value == 0x42); // forwarded: the read needs no wait
    view.Note(word + 4, 0, false, 20, Retire::Consumed, 0, Kind::Release, At(0));
    const auto at_execute = view.Find(word + 4, At(5));
    REQUIRE(at_execute.pending);
    REQUIRE_FALSE(at_execute.writer.known);
    REQUIRE(at_execute.writer.retire == Retire::Consumed); // wait for position 20 only
    REQUIRE(at_execute.writer.position == 20);
    view.Note(word + 8, 0, false, 30, Retire::Landing, 0x1e01, Kind::Counter, At(0));
    const auto at_fence = view.Find(word + 8, At(40));
    REQUIRE(at_fence.pending);
    REQUIRE(at_fence.writer.retire == Retire::Landing); // targeted landing by its ticket
    REQUIRE(at_fence.writer.ticket == 0x1e01);
    // Once landed nothing is pending and memory holds it.
    REQUIRE_FALSE(view.Find(word + 8, At(40, 0x1e01)).pending);
}

TEST_CASE("SYNC step 2 idle back: an acquire met by a fence-flagged release whose fence is not "
          "submitted passes at once (class B)",
          "[video_core][v2_sem_view]") {
    SemView view;
    // A 4-byte release lands at its fence (#337); the back ran it, its fence is unsubmitted.
    view.Note(0x7000, 3, true, 10, Retire::Landing, (u64{10} << 8), Kind::Release, At(0));
    SemInputs in;
    in.memory_word = 2;
    const auto d =
        DecideFromView(SemaphoreOp::Gequal, 3, view.Find(0x7000, At(10, 0)), in);
    REQUIRE(d.verdict == AcquireVerdict::MetByQueued);
    REQUIRE_FALSE(d.request);
    REQUIRE(d.position == 10);
}
