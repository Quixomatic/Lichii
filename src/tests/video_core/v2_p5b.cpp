// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <bit>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/native_qmd.h"
#include "video_core/v2/barrier_tracker.h"
#include "video_core/v2/flush_heuristic.h"
#include "video_core/v2/frame_pacer.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/gpu_unswizzle.h"
#include "video_core/textures/decoders.h"
#include "video_core/v2/qmd_front.h"
#include "video_core/v2/tick_ring.h"
#include "video_core/v2/timeline_join.h"

namespace {

using VideoCommon::V2::FlushAction;
using VideoCommon::V2::FlushHeuristic;
using VideoCommon::V2::FlushHint;
using VideoCommon::V2::FlushParams;
using VideoCommon::V2::FlushState;
using VideoCommon::V2::RangeBarrierTracker;
using VideoCommon::V2::TimelineJoin;

} // Anonymous namespace

TEST_CASE("V2 P5b barrier tracker range math", "[video_core][v2_p5b]") {
    SECTION("half-open overlap") {
        REQUIRE(RangeBarrierTracker::Overlaps(0, 10, 9, 20));
        REQUIRE_FALSE(RangeBarrierTracker::Overlaps(0, 10, 10, 20));
        REQUIRE_FALSE(RangeBarrierTracker::Overlaps(10, 20, 0, 10));
        REQUIRE(RangeBarrierTracker::Overlaps(5, 6, 0, 100));
        REQUIRE_FALSE(RangeBarrierTracker::Overlaps(5, 5, 0, 100)); // empty
        REQUIRE_FALSE(RangeBarrierTracker::Overlaps(0, 100, 50, 50));
    }
    SECTION("resource or guest range") {
        RangeBarrierTracker tracker;
        tracker.NoteWrite(/*resource*/ 1, 0x1000, 0x2000, 0x10, 0x100);
        tracker.NoteWrite(2, 0x8000, 0x9000, 0x20, 0x200);
        // Same resource, any range (e.g. a rescaled or format-aliased view): hazard.
        REQUIRE(tracker.Check(1, 0, 0).Any());
        // Another resource aliasing the guest range: hazard.
        const auto alias = tracker.Check(3, 0x1800, 0x1801);
        REQUIRE(alias.count == 1);
        REQUIRE(alias.stages == 0x10);
        REQUIRE(alias.access == 0x100);
        // Disjoint: none.
        REQUIRE_FALSE(tracker.Check(3, 0x2000, 0x8000).Any());
        // Both.
        const auto both = tracker.Check(2, 0x1000, 0x1001);
        REQUIRE(both.count == 2);
        REQUIRE(both.stages == 0x30);
        REQUIRE(tracker.Size() == 2);
        // Take removes only what overlaps.
        const auto taken = tracker.Take(3, 0x1fff, 0x2000);
        REQUIRE(taken.count == 1);
        REQUIRE(tracker.Size() == 1);
        REQUIRE_FALSE(tracker.Check(1, 0x1000, 0x2000).Any());
        REQUIRE(tracker.Check(0, 0x8fff, 0x9000).Any());
    }
    SECTION("merge and fold") {
        RangeBarrierTracker tracker;
        tracker.NoteWrite(1, 0, 16, 1, 2);
        tracker.NoteWrite(1, 0, 16, 4, 8);
        REQUIRE(tracker.Size() == 1);
        REQUIRE(tracker.Check(1, 0, 16).stages == 5);
        for (u64 i = 0; i < RangeBarrierTracker::Capacity + 4; ++i) {
            tracker.NoteWrite(100 + i, i * 0x10000, i * 0x10000 + 0x100, 0x40, 0x80);
        }
        REQUIRE(tracker.Folded());
        // Everything hazards when folded, with the union of every mask.
        const auto any = tracker.Check(999, 0xffff0000, 0xffff0001);
        REQUIRE(any.Any());
        REQUIRE(any.stages == (5 | 0x40));
        const auto all = tracker.TakeAll();
        REQUIRE(all.Any());
        REQUIRE(tracker.Empty());
    }
    SECTION("random model") {
        std::mt19937 rng{1234};
        RangeBarrierTracker tracker;
        struct Ref {
            u64 resource, begin, end;
            u32 stages;
        };
        std::vector<Ref> model;
        for (int step = 0; step < 20000; ++step) {
            const u64 resource = rng() % 8 + 1;
            const u64 begin = (rng() % 64) * 0x100;
            const u64 end = begin + (rng() % 4 + 1) * 0x100;
            const u32 stages = 1U << (rng() % 8);
            switch (rng() % 3) {
            case 0:
                if (tracker.Size() < RangeBarrierTracker::Capacity) {
                    tracker.NoteWrite(resource, begin, end, stages, 0);
                    bool merged = false;
                    for (auto& ref : model) {
                        if (ref.resource == resource && ref.begin == begin && ref.end == end) {
                            ref.stages |= stages;
                            merged = true;
                        }
                    }
                    if (!merged) {
                        model.push_back({resource, begin, end, stages});
                    }
                }
                break;
            case 1: {
                u32 expected = 0;
                for (const auto& ref : model) {
                    if (ref.resource == resource || (ref.begin < end && begin < ref.end)) {
                        expected |= ref.stages;
                    }
                }
                REQUIRE(tracker.Check(resource, begin, end).stages == expected);
                break;
            }
            case 2: {
                u32 expected = 0;
                std::erase_if(model, [&](const Ref& ref) {
                    const bool hit = ref.resource == resource || (ref.begin < end && begin < ref.end);
                    if (hit) {
                        expected |= ref.stages;
                    }
                    return hit;
                });
                REQUIRE(tracker.Take(resource, begin, end).stages == expected);
                break;
            }
            }
            REQUIRE(tracker.Size() == model.size());
        }
    }
}

TEST_CASE("V2 P5b flush heuristic decisions", "[video_core][v2_p5b]") {
    FlushParams params;
    const FlushState idle{.pending = 0, .back_behind = false};
    const FlushState idle_behind{.pending = 0, .back_behind = true};
    SECTION("idle GPU is fed once enough draws are recorded") {
        FlushHeuristic heuristic{params};
        u64 now = 1000;
        FlushHeuristic::Decision decision{};
        u32 at = 0;
        for (u32 i = 1; i <= 1000 && at == 0; ++i) {
            now += 20;
            decision = heuristic.Advance(1, idle, now);
            if (decision.action == FlushAction::Flush) {
                at = i;
            }
        }
        REQUIRE(decision.reason == FlushHeuristic::ReasonIdle);
        // Never before min_draws (Perf fix C: 16-draw submissions), at a chunk boundary.
        REQUIRE(at == params.min_draws);
        REQUIRE(heuristic.Draws() == 0);
    }
    // Perf fix C: a CPU-bound recorder always leaves the GPU idle; the idle rule must not turn
    // that into "submit every millisecond".
    SECTION("idle GPU with the recorder behind: no idle flush, only the interval") {
        FlushHeuristic heuristic{params};
        u32 flushes = 0;
        // 10 draws per 100 us for 20 ms, all with the GPU idle and more work queued.
        for (u32 i = 0; i < 2000; ++i) {
            const u64 now = 1000 + u64{i} * 10;
            const auto decision = heuristic.Advance(1, idle_behind, now);
            if (decision.action == FlushAction::Flush) {
                REQUIRE(decision.reason == FlushHeuristic::ReasonInterval);
                ++flushes;
            }
        }
        // 20 ms: one interval flush per max_interval_us (5), not one per ms or per chunk.
        REQUIRE(flushes <= 20000 / params.max_interval_us + 1);
        REQUIRE(flushes >= 1);
    }
    SECTION("idle GPU: no idle flush sooner than the minimum interval") {
        FlushHeuristic heuristic{params};
        u32 flushes = 0;
        // 10 draws per 100 us: 128 draws every 1.28 ms.
        for (u32 i = 0; i < 1280; ++i) {
            const u64 now = 1000 + u64{i} * 10;
            if (heuristic.Advance(1, idle, now).action == FlushAction::Flush) {
                ++flushes;
            }
        }
        REQUIRE(flushes <= 1280 / params.min_draws);
        REQUIRE(flushes >= 1);
    }
    SECTION("busy GPU: threshold grows with pending work, chunks dispatched meanwhile") {
        FlushHeuristic heuristic{params};
        const u64 now = 1000;
        u32 flushed_at = 0;
        u32 dispatches = 0;
        for (u32 i = 1; i <= 2000 && flushed_at == 0; ++i) {
            const auto decision = heuristic.Advance(1, /*pending*/ 4, now);
            if (decision.action == FlushAction::Dispatch) {
                ++dispatches;
            }
            if (decision.action == FlushAction::Flush) {
                REQUIRE(decision.reason == FlushHeuristic::ReasonPending);
                flushed_at = i;
            }
        }
        // min(20, 4 * 3) = 12 chunks of 32 draws.
        REQUIRE(flushed_at == 12 * params.chunk_draws);
        REQUIRE(dispatches == 11);
    }
    SECTION("one submission in flight: no pending flush, no idle flush") {
        FlushHeuristic heuristic{params};
        u32 flushed_at = 0;
        for (u32 i = 1; i <= 2000 && flushed_at == 0; ++i) {
            if (heuristic.Advance(1, /*pending*/ 1, 1000).action == FlushAction::Flush) {
                flushed_at = i;
            }
        }
        REQUIRE(flushed_at == params.max_draws);
    }
    SECTION("deep queue caps at max_draws") {
        FlushHeuristic heuristic{params};
        u32 flushed_at = 0;
        for (u32 i = 1; i <= 2000 && flushed_at == 0; ++i) {
            if (heuristic.Advance(1, 100, 1000).action == FlushAction::Flush) {
                flushed_at = i;
            }
        }
        // min(20, 300) = 20 chunks = 640 draws, below max_draws.
        REQUIRE(flushed_at == 20 * params.chunk_draws);
    }
    SECTION("strong hints flush after 3 chunks, weak after 6") {
        FlushHeuristic strong{params};
        u32 at = 0;
        for (u32 i = 1; i <= 1000 && at == 0; ++i) {
            if (i == 5) {
                strong.Hint(FlushHint::Strong);
            }
            const auto decision = strong.Advance(1, 10, 1000);
            if (decision.action == FlushAction::Flush) {
                REQUIRE(decision.reason == FlushHeuristic::ReasonStrong);
                at = i;
            }
        }
        REQUIRE(at == 3 * params.chunk_draws);
        FlushHeuristic weak{params};
        at = 0;
        for (u32 i = 1; i <= 1000 && at == 0; ++i) {
            weak.Hint(FlushHint::Weak);
            const auto decision = weak.Advance(1, 10, 1000);
            if (decision.action == FlushAction::Flush) {
                REQUIRE(decision.reason == FlushHeuristic::ReasonWeak);
                at = i;
            }
        }
        REQUIRE(at == 6 * params.chunk_draws);
    }
    SECTION("cost cap and time interval") {
        FlushHeuristic heuristic{params};
        FlushHeuristic::Decision decision{};
        u32 i = 0;
        while (decision.action != FlushAction::Flush) {
            decision = heuristic.Advance(/*dispatch cost*/ 400, 10, 1000);
            ++i;
        }
        REQUIRE(decision.reason == FlushHeuristic::ReasonCost);
        REQUIRE(i == 4);
        FlushHeuristic timed{params};
        u64 now = 1000;
        decision = {};
        u32 at = 0;
        for (u32 d = 1; d <= 1000 && decision.action != FlushAction::Flush; ++d) {
            now += 50; // past max_interval_us at draw 80, but below min_draws
            decision = timed.Advance(1, 1, now);
            at = d;
        }
        REQUIRE(decision.reason == FlushHeuristic::ReasonInterval);
        REQUIRE(at == params.min_draws);
    }
    SECTION("fast-flush mode enters and leaves on the guest's unsubmitted waits") {
        FlushHeuristic heuristic{params};
        for (int frame = 0; frame < 40 && !heuristic.FastMode(); ++frame) {
            heuristic.NoteGuestSyncWait(5000);
            heuristic.NoteFrame();
        }
        REQUIRE(heuristic.FastMode());
        u64 now = 1000;
        FlushHeuristic::Decision decision{};
        heuristic.NoteSubmit(now);
        for (u32 d = 1; d <= params.chunk_draws; ++d) {
            now += 200;
            decision = heuristic.Advance(1, FlushState{.pending = 10, .back_behind = true}, now);
        }
        // A recorder that is behind makes the guest wait by itself: no fast flush then.
        REQUIRE(decision.action == FlushAction::Dispatch);
        for (u32 d = 1; d <= params.chunk_draws; ++d) {
            now += 200;
            decision = heuristic.Advance(1, FlushState{.pending = 10, .back_behind = false}, now);
        }
        REQUIRE(decision.action == FlushAction::Flush);
        REQUIRE(decision.reason == FlushHeuristic::ReasonFast);
        for (int frame = 0; frame < 200 && heuristic.FastMode(); ++frame) {
            heuristic.NoteFrame();
        }
        REQUIRE_FALSE(heuristic.FastMode());
    }
    SECTION("immediate strong hint needs 3 chunks, idle GPU or not") {
        FlushHeuristic heuristic{params};
        for (u32 i = 0; i < 16; ++i) {
            (void)heuristic.Advance(1, idle, 1000);
        }
        heuristic.Hint(FlushHint::Strong);
        REQUIRE_FALSE(heuristic.WantsFlushNow()); // Perf fix C: no idle shortcut
        FlushHeuristic later{params};
        for (u32 i = 0; i < params.strong_chunks * params.chunk_draws; ++i) {
            (void)later.Advance(1, 10, 1000);
        }
        later.Hint(FlushHint::Strong);
        REQUIRE(later.WantsFlushNow());
    }
}

TEST_CASE("V2 P5b timeline join bookkeeping", "[video_core][v2_p5b]") {
    TimelineJoin join;
    // Open tick 10. A resource last used at 7 joins the transfer batch and waits for 7.
    REQUIRE(join.CanHoist(7, 10));
    join.Hoist(7, 10);
    join.Hoist(3, 10);
    // Used by the open submission: stays in order.
    REQUIRE_FALSE(join.CanHoist(10, 10));
    auto batch = join.Close(10);
    REQUIRE(batch.transfer);
    REQUIRE(batch.transfer_wait == 7);
    REQUIRE(batch.transfer_signal == 10);
    REQUIRE(batch.graphics_waits_transfer);
    // Nothing hoisted: no batch.
    batch = join.Close(11);
    REQUIRE_FALSE(batch.transfer);
    REQUIRE_FALSE(batch.graphics_waits_transfer);

    // Class B join: an acquire executed during open tick 12 stops hoisting in 12; the batch of
    // 13 waits for 12 even for a resource last used long ago.
    join.NoteAcquire(12);
    REQUIRE_FALSE(join.CanHoist(1, 12));
    batch = join.Close(12);
    REQUIRE(join.CanHoist(1, 13));
    join.Hoist(1, 13);
    batch = join.Close(13);
    REQUIRE(batch.transfer_wait == 12);

    // A queue-family release blocks hoisting until the next submission only.
    join.BlockHoisting();
    REQUIRE_FALSE(join.CanHoist(1, 14));
    (void)join.Close(14);
    REQUIRE(join.CanHoist(1, 15));

    // Readback waits: newest per semaphore, bounded slots.
    REQUIRE(join.AddGraphicsWait(0xA, 5));
    REQUIRE(join.AddGraphicsWait(0xA, 3));
    REQUIRE(join.AddGraphicsWait(0xB, 9));
    batch = join.Close(15);
    REQUIRE(batch.num_extra_waits == 2);
    REQUIRE(batch.extra_waits[0].semaphore == 0xA);
    REQUIRE(batch.extra_waits[0].value == 5);
    REQUIRE(batch.extra_waits[1].value == 9);
    for (u64 i = 0; i < TimelineJoin::MaxExtraWaits; ++i) {
        REQUIRE(join.AddGraphicsWait(0x100 + i, 1));
    }
    REQUIRE_FALSE(join.AddGraphicsWait(0x999, 1));
    REQUIRE(join.AddGraphicsWait(0x100, 2)); // merges
    batch = join.Close(16);
    REQUIRE(batch.num_extra_waits == TimelineJoin::MaxExtraWaits);

    VideoCommon::V2::JobTimeline timeline;
    const u64 first = timeline.Next();
    const u64 second = timeline.Next();
    REQUIRE(second > first);
    REQUIRE_FALSE(timeline.IsComplete(first));
    timeline.NoteCompleted(second);
    REQUIRE(timeline.IsComplete(first));
}

TEST_CASE("V2 P5b fence-tracked ring never waits", "[video_core][v2_p5b]") {
    VideoCommon::V2::TickRing ring{1024, 64};
    // Tick 1 and 2 allocations; nothing completed.
    REQUIRE(ring.Allocate(256, 1, 0) == 0);
    REQUIRE(ring.Allocate(256, 1, 0) == 256);
    REQUIRE(ring.Allocate(256, 2, 0) == 512);
    REQUIRE(ring.Allocate(256, 2, 0) == 768);
    // Full: fails instead of waiting.
    REQUIRE_FALSE(ring.Allocate(64, 3, 0).has_value());
    // Tick 1 done: its 512 bytes come back, allocation wraps to the start.
    REQUIRE(ring.Allocate(300, 3, 1) == 0);
    REQUIRE_FALSE(ring.Allocate(300, 3, 1).has_value()); // 512-320 = 192 free before tick 2's
    REQUIRE(ring.Allocate(128, 3, 1) == 320);
    // Everything done: empty again.
    REQUIRE(ring.Allocate(1024, 4, 3) == 0);
    REQUIRE_FALSE(ring.Allocate(2048, 5, 4).has_value()); // larger than the ring

    // Random model: live allocations never overlap.
    std::mt19937 rng{77};
    VideoCommon::V2::TickRing model{1 << 16, 256};
    struct Live {
        u64 offset, size, tick;
    };
    std::vector<Live> live;
    u64 tick = 1;
    u64 completed = 0;
    for (int step = 0; step < 50000; ++step) {
        if (rng() % 8 == 0) {
            ++tick;
        }
        if (rng() % 6 == 0 && completed + 1 < tick) {
            completed += 1 + rng() % (tick - completed - 1);
        }
        std::erase_if(live, [&](const Live& entry) { return entry.tick <= completed; });
        const u64 size = 1 + rng() % 9000;
        const auto offset = model.Allocate(size, tick, completed);
        if (!offset) {
            continue;
        }
        const u64 aligned = (size + 255) / 256 * 256;
        REQUIRE(*offset + aligned <= model.Size());
        for (const Live& other : live) {
            REQUIRE((*offset + aligned <= other.offset || other.offset + other.size <= *offset));
        }
        live.push_back({*offset, aligned, tick});
    }
}

TEST_CASE("V2 P5b frame pacer limits", "[video_core][v2_p5b]") {
    using VideoCommon::V2::FramesInFlightLimit;
    REQUIRE(FramesInFlightLimit(0) == 1);
    REQUIRE(FramesInFlightLimit(1) == 2);
    REQUIRE(FramesInFlightLimit(-1) == 2);
    REQUIRE(FramesInFlightLimit(7) == 3);
    VideoCommon::V2::FramePacer pacer;
    pacer.SetLimit(2);
    REQUIRE_FALSE(pacer.MustWait());
    REQUIRE(pacer.NoteComposite() == 1);
    REQUIRE_FALSE(pacer.MustWait());
    pacer.NoteComposite();
    REQUIRE(pacer.MustWait());
    REQUIRE(pacer.WaitTarget() == 1);
    pacer.NoteRetired(1);
    REQUIRE_FALSE(pacer.MustWait());
    pacer.NoteRetired(99); // clamped to what was composited
    REQUIRE(pacer.InFlight() == 0);
}

namespace {

using Descriptor = Tegra::NativeQmdScheduler::Descriptor;

Descriptor MakeQmd(u32 dependencies, bool grid, u64 chain_address = 0, u32 chain_action = 0) {
    Descriptor qmd{};
    qmd[18] = 0x30; // QMD version 3.0
    qmd[4] = dependencies << 14;
    if (grid) {
        qmd[12] = 4;
        qmd[13] = 1;
        qmd[14] = 1;
    }
    if (chain_address != 0) {
        qmd[15] = static_cast<u32>(chain_address >> 8);
        qmd[16] = 1 | (chain_action << 1);
    }
    return qmd;
}

u32 Action(u32 action, u32 count = 1) {
    return action | ((count - 1) << 14);
}

} // Anonymous namespace

TEST_CASE("V2 P5b front QMD scheduler counts", "[video_core][v2_p5b]") {
    namespace QmdFront = VideoCommon::V2::QmdFront;
    std::map<u64, Descriptor> memory;
    constexpr u64 A = 0x10000;
    constexpr u64 B = 0x20000;
    constexpr u64 C = 0x30000;
    memory[A] = MakeQmd(2, true, B, /*decrement*/ 4);
    memory[B] = MakeQmd(1, true);
    memory[C] = MakeQmd(0, false); // empty grid: completes without a dispatch

    QmdFront::Model model;
    std::vector<u64> dispatched;
    std::vector<u64> released;
    const auto read = [&](u64 address, Descriptor& descriptor) {
        const auto it = memory.find(address);
        if (it == memory.end()) {
            return QmdFront::ReadResult::Unmapped;
        }
        descriptor = it->second;
        return QmdFront::ReadResult::Ok;
    };
    const auto mapped = [](u64, u64) { return true; };
    const auto on_dispatch = [&](u64 address, const Descriptor&) { dispatched.push_back(address); };
    const auto on_release = [&](u64 address, u32, u32) { released.push_back(address); };
    const auto signal = [&](u64 address, u32 argument) {
        return model.Signal(address, argument, read, mapped, on_dispatch, on_release);
    };

    // B scheduled with one dependency: waits.
    auto result = signal(B, Action(3));
    REQUIRE(result.valid);
    REQUIRE(result.dispatches == 0);
    // A scheduled with two dependencies.
    REQUIRE(signal(A, Action(3)).dispatches == 0);
    // One decrement of A: still waiting.
    REQUIRE(signal(A, Action(7)).dispatches == 0);
    // The second launches A, whose dependent action decrements B, which launches too.
    result = signal(A, Action(7));
    REQUIRE(result.valid);
    REQUIRE(result.dispatches == 2);
    REQUIRE(dispatched == std::vector<u64>{A, B});
    // An empty-grid QMD completes without a dispatch.
    REQUIRE(signal(C, Action(3)).dispatches == 0);
    // A decrement that arrives before the QMD is loaded is deferred, as on the back.
    constexpr u64 D = 0x40000;
    memory[D] = MakeQmd(1, true);
    REQUIRE(signal(D, Action(7)).dispatches == 0);
    REQUIRE(signal(D, Action(3)).dispatches == 1);

    // The front model and the back scheduler give the same counts on a random stream.
    Tegra::NativeQmdScheduler back;
    QmdFront::Model front;
    std::mt19937 rng{4242};
    std::map<u64, Descriptor> random_memory;
    for (u64 i = 0; i < 16; ++i) {
        const u64 address = 0x100000 + i * 0x100;
        const u64 chain = rng() % 3 == 0 ? 0x100000 + (rng() % 16) * 0x100 : 0;
        random_memory[address] = MakeQmd(rng() % 3, rng() % 4 != 0, chain, chain ? 4 : 0);
    }
    const auto random_read = [&](u64 address, Descriptor& descriptor) {
        descriptor = random_memory.at(address);
        return true;
    };
    for (int step = 0; step < 5000; ++step) {
        const u64 address = 0x100000 + (rng() % 16) * 0x100;
        static constexpr std::array<u32, 4> actions{2, 3, 7, 12};
        const u32 argument = Action(actions[rng() % 4], 1 + rng() % 2);
        u32 back_dispatches = 0;
        (void)back.Signal(
            address, argument, random_read, [](u64, u32, u32) { return true; },
            [&](const Descriptor&) {
                ++back_dispatches;
                return true;
            });
        const auto front_result = front.Signal(
            address, argument,
            [&](u64 a, Descriptor& d) {
                d = random_memory.at(a);
                return QmdFront::ReadResult::Ok;
            },
            mapped, [](u64, const Descriptor&) {}, [](u64, u32, u32) {});
        REQUIRE(front_result.valid);
        REQUIRE(front_result.dispatches == back_dispatches);
    }

    // A read the front can't trust makes the model unknown until a resync from the back.
    const auto unknown = model.Signal(
        A, Action(3), [](u64, Descriptor&) { return QmdFront::ReadResult::Unknown; }, mapped,
        on_dispatch, on_release);
    REQUIRE_FALSE(unknown.valid);
    REQUIRE_FALSE(model.Valid());
    REQUIRE_FALSE(signal(B, Action(3)).valid);
    model.Resync(Tegra::NativeQmdScheduler::BackTable());
    REQUIRE(model.Valid());
    REQUIRE(model.Table().size() == Tegra::NativeQmdScheduler::BackTable().size());

    QmdFront::InlineQmds inline_qmds;
    inline_qmds.Note(A, memory[A]);
    REQUIRE(inline_qmds.Find(A) != nullptr);
    REQUIRE(inline_qmds.Find(B) == nullptr);
}

TEST_CASE("V2 P5b GPU untile math matches UnswizzleTexture", "[video_core][v2_p5b]") {
    namespace G = VideoCommon::V2::GpuUnswizzle;
    std::mt19937 rng{99};
    for (int trial = 0; trial < 300; ++trial) {
        const u32 bpp = 1U << (rng() % 5);
        const u32 width = 1 + rng() % 80;
        const u32 height = 1 + rng() % 70;
        const u32 depth = rng() % 4 == 0 ? 1 + rng() % 5 : 1;
        const u32 block_height = rng() % 5;
        const u32 block_depth = depth > 1 ? rng() % 3 : 0;
        const u32 stride_alignment = rng() % 2 == 0 ? 0 : 6 - std::countr_zero(bpp) % 7;
        const bool native = rng() % 2 == 0;
        const auto segment = G::MakeSegment(0, 0, bpp, width, height, depth, block_height,
                                            block_depth, stride_alignment % 8, native);
        // Input big enough for any swizzled offset of the image.
        const u32 input_size = (segment.slice_size + segment.block_size + 4096) *
                               ((depth >> block_depth) + 2);
        std::vector<u8> input(input_size);
        for (auto& value : input) {
            value = static_cast<u8>(rng());
        }
        std::vector<u8> expected(segment.num_bytes);
        Tegra::Texture::UnswizzleTexture(expected, input, bpp, width, height, depth, block_height,
                                         block_depth, stride_alignment % 8, native);
        for (u32 b = 0; b < segment.num_bytes; ++b) {
            const u32 source = G::SourceByte(segment, b);
            REQUIRE(source < input.size());
            REQUIRE(input[source] == expected[b]);
        }
    }
}

TEST_CASE("V2 P5b unused-region uploads join the transfer batch without a graphics wait",
          "[video_core][v2_p5b]") {
    TimelineJoin join;
    join.NoteAcquire(10); // class-B floor at the open tick
    REQUIRE_FALSE(join.CanHoist(3, 10));
    REQUIRE(join.CanHoistUnordered());
    join.HoistUnordered();
    auto batch = join.Close(10);
    REQUIRE(batch.transfer);
    REQUIRE(batch.transfer_wait == 0);
    REQUIRE(batch.graphics_waits_transfer);
    // A track-ordered op in the same batch still sets the wait.
    join.HoistUnordered();
    join.Hoist(11, 12);
    batch = join.Close(12);
    REQUIRE(batch.transfer_wait == 11);
    // A blocked batch takes nothing.
    join.BlockHoisting();
    REQUIRE_FALSE(join.CanHoistUnordered());
}
TEST_CASE("V2 P5b submits are counted by call site", "[video_core][v2_p5b]") {
    using VideoCommon::V2::P5bStats::SubmitSites;
    SubmitSites sites;
    static const char a[] = "src/video_core/renderer_vulkan/vk_fence_manager.cpp";
    static const char b[] = "C:\lichii\src\video_core\renderer_vulkan\vk_rasterizer.cpp";
    // The same file name from another literal (an inline function in two units) is one site.
    std::string a_copy{a};
    for (int i = 0; i < 5; ++i) {
        sites.Note(a, 35);
    }
    sites.Note(a_copy.c_str(), 35);
    sites.Note(a, 36);
    sites.Note(b, 4800);
    sites.Note(b, 4800);
    const auto taken = sites.Take();
    REQUIRE(taken.size() == 3);
    REQUIRE(taken[0].count == 6);
    REQUIRE(taken[0].line == 35);
    REQUIRE(taken[1].count == 2);
    REQUIRE(taken[2].count == 1);
    REQUIRE(SubmitSites::Name(taken[0].file, taken[0].line) == "vk_fence_manager.cpp:35");
    REQUIRE(SubmitSites::Name(b, 4800) == "vk_rasterizer.cpp:4800");
    REQUIRE(sites.Take().empty());
}

TEST_CASE("V2 P5b flush decisions at pass boundaries", "[video_core][v2_p5b]") {
    FlushParams params;
    params.pass_boundaries = true;
    const FlushState busy{.pending = 1, .back_behind = true};
    const FlushState requested{.pending = 1, .back_behind = true, .requested = true};
    SECTION("mid-pass: only the safety nets, no interval or idle submits") {
        FlushHeuristic heuristic{params};
        const FlushState idle{.pending = 0, .back_behind = false};
        u32 flushes = 0;
        // 1000 draws over 7 ms, GPU idle: the old rules would have flushed (idle, interval).
        for (u32 i = 1; i <= 1000; ++i) {
            if (heuristic.Advance(1, idle, 1000 + u64{i} * 7).action == FlushAction::Flush) {
                ++flushes;
            }
        }
        REQUIRE(flushes == 0);
        REQUIRE(heuristic.Advance(1, idle, 9000).action == FlushAction::None);
        // max draws still applies.
        for (u32 i = heuristic.Draws(); i < params.max_draws - 1; ++i) {
            REQUIRE(heuristic.Advance(1, idle, 9000).action != FlushAction::Flush);
        }
        const auto last = heuristic.Advance(1, idle, 9000);
        REQUIRE(last.action == FlushAction::Flush);
        REQUIRE(last.reason == FlushHeuristic::ReasonMaxDraws);
    }
    SECTION("mid-pass safety interval") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        FlushHeuristic::Decision decision{};
        for (u32 i = 1; i <= params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, busy, 1000 + params.mid_pass_interval_us - 1);
        }
        REQUIRE(decision.action == FlushAction::Dispatch);
        for (u32 i = 1; i <= params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, busy, 1000 + params.mid_pass_interval_us);
        }
        REQUIRE(decision.action == FlushAction::Flush);
        REQUIRE(decision.reason == FlushHeuristic::ReasonInterval);
    }
    SECTION("Ryujinx: a boundary after enough draws and time") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        for (u32 i = 0; i < params.pass_min_draws - 1; ++i) {
            heuristic.Advance(1, busy, 1000);
        }
        // Too few draws, however long ago.
        REQUIRE(heuristic.AtPassBoundary(FlushHint::Weak, busy, 1000 + 10 * params.pass_interval_us)
                    .action == FlushAction::None);
        heuristic.Advance(1, busy, 1000);
        // Enough draws, too soon.
        REQUIRE(heuristic.AtPassBoundary(FlushHint::Weak, busy, 1000 + params.pass_interval_us / 2)
                    .action == FlushAction::None);
        const auto decision =
            heuristic.AtPassBoundary(FlushHint::Weak, busy, 1000 + params.pass_interval_us);
        REQUIRE(decision.action == FlushAction::Flush);
        REQUIRE(decision.reason == FlushHeuristic::ReasonPass);
        REQUIRE(heuristic.Draws() == 0);
    }
    SECTION("nothing recorded: no boundary submit") {
        FlushHeuristic heuristic{params};
        REQUIRE(heuristic.AtPassBoundary(FlushHint::Strong, requested, 1'000'000).action ==
                FlushAction::None);
    }
    SECTION("a waiter's request rides the next boundary, mid-pass only after the gap") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        heuristic.Advance(1, busy, 1000);
        auto boundary = heuristic.AtPassBoundary(FlushHint::Weak, requested, 1001);
        REQUIRE(boundary.action == FlushAction::Flush);
        REQUIRE(boundary.reason == FlushHeuristic::ReasonRequest);
        // Mid-pass: the chunk boundary right after the submit waits for the gap.
        FlushHeuristic::Decision decision{};
        for (u32 i = 1; i <= params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, requested, 1001 + params.request_gap_us / 2);
        }
        REQUIRE(decision.action == FlushAction::Dispatch);
        for (u32 i = 1; i <= params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, requested, 1001 + params.request_gap_us);
        }
        REQUIRE(decision.action == FlushAction::Flush);
        REQUIRE(decision.reason == FlushHeuristic::ReasonRequest);
    }
    SECTION("RT0 change is a strong hint after 3 chunks; weak needs min draws and 6 chunks") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        for (u32 i = 0; i < params.strong_chunks * params.chunk_draws; ++i) {
            heuristic.Advance(1, busy, 1000);
        }
        REQUIRE(heuristic.AtPassBoundary(FlushHint::Weak, busy, 1001).action ==
                FlushAction::None);
        const auto strong = heuristic.AtPassBoundary(FlushHint::Strong, busy, 1002);
        REQUIRE(strong.action == FlushAction::Flush);
        REQUIRE(strong.reason == FlushHeuristic::ReasonStrong);
    }
    SECTION("busy GPU: DXVK's pending rule at boundaries") {
        FlushHeuristic heuristic{params};
        const FlushState two{.pending = 2, .back_behind = true};
        heuristic.NoteSubmit(1000);
        for (u32 i = 0; i < 6 * params.chunk_draws - 1; ++i) {
            heuristic.Advance(1, two, 1000);
        }
        REQUIRE(heuristic.AtPassBoundary(FlushHint::None, two, 1001).action == FlushAction::None);
        heuristic.Advance(1, two, 1000);
        const auto decision = heuristic.AtPassBoundary(FlushHint::None, two, 1001);
        REQUIRE(decision.action == FlushAction::Flush);
        REQUIRE(decision.reason == FlushHeuristic::ReasonPending);
    }
}
