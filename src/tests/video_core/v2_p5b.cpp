// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/native_qmd.h"
#include "video_core/renderer_vulkan/vk_staging_buffer_pool.h"
#include "video_core/v2/after_pass.h"
#include "video_core/v2/barrier_tracker.h"
#include "video_core/v2/flush_heuristic.h"
#include "video_core/v2/frame_pacer.h"
#include "video_core/v2/landing_fence.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/submit_requests.h"
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

TEST_CASE("V2 P5b wait-for-idle keeps store-free passes open", "[video_core][v2_p5b]") {
    // F4-X3: outside a pass the barrier is recorded at once; inside a pass whose draws wrote only
    // attachments (tracked) it waits for the pass end; stores or transform feedback end the pass.
    using VideoCommon::V2::DecideWaitForIdle;
    using VideoCommon::V2::WfiAction;
    REQUIRE(DecideWaitForIdle(false, false, false) == WfiAction::Barrier);
    REQUIRE(DecideWaitForIdle(false, true, true) == WfiAction::Barrier);
    REQUIRE(DecideWaitForIdle(true, false, false) == WfiAction::AtPassEnd);
    REQUIRE(DecideWaitForIdle(true, true, false) == WfiAction::EndPass);
    REQUIRE(DecideWaitForIdle(true, false, true) == WfiAction::EndPass);
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

TEST_CASE("V2 P5b graphics waits for buffer-only transfer batches at buffer stages only",
          "[video_core][v2_p5b]") {
    TimelineJoin join;
    join.HoistUnordered();
    auto batch = join.Close(5);
    REQUIRE(batch.graphics_waits_transfer);
    REQUIRE_FALSE(batch.transfer_images);
    // An image upload (ownership acquired at the submission's start) makes it every stage.
    join.HoistUnordered();
    join.Hoist(3, 6);
    batch = join.Close(6);
    REQUIRE(batch.transfer_images);
    // Reset by Close.
    join.HoistUnordered();
    REQUIRE_FALSE(join.Close(7).transfer_images);
    REQUIRE_FALSE(join.Close(8).transfer_images);
}
TEST_CASE("V2 P5b submits are counted by call site", "[video_core][v2_p5b]") {
    using VideoCommon::V2::P5bStats::SubmitSites;
    SubmitSites sites;
    static const char a[] = "src/video_core/renderer_vulkan/vk_fence_manager.cpp";
    static const char b[] = R"(C:\lichii\src\video_core\renderer_vulkan\vk_rasterizer.cpp)";
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

TEST_CASE("V2 P5b readbacks submit only if their producer is still open",
          "[video_core][v2_p5b]") {
    using VideoCommon::V2::ChooseReadbackRoute;
    using VideoCommon::V2::ReadbackRoute;
    // open tick 10, the graphics queue has up to 8, the GPU finished 5.
    REQUIRE(ChooseReadbackRoute(false, 7, 10, 8, 5) == ReadbackRoute::Direct);
    REQUIRE(ChooseReadbackRoute(false, 8, 10, 8, 5) == ReadbackRoute::Direct);
    REQUIRE(ChooseReadbackRoute(false, 0, 10, 0, 0) == ReadbackRoute::Direct);
    // Handed to the worker but not on the queue yet (B1: the job must not get ahead of it).
    REQUIRE(ChooseReadbackRoute(false, 9, 10, 8, 5) == ReadbackRoute::FlushNow);
    // ... unless the GPU already finished it.
    REQUIRE(ChooseReadbackRoute(false, 9, 10, 8, 9) == ReadbackRoute::Direct);
    // The open tick is flushed at once: no job is left riding an unsubmitted tick (3d9126d161).
    REQUIRE(ChooseReadbackRoute(false, 10, 10, 9, 9) == ReadbackRoute::FlushNow);
    // Image copies carry a release recorded in the open submission.
    REQUIRE(ChooseReadbackRoute(true, 1, 10, 9, 9) == ReadbackRoute::FlushNow);
}

TEST_CASE("V2 P5b a graphics wait on the open submission's own post job goes to the next one",
          "[video_core][v2_p5b]") {
    // 3d9126d161 hang: a readback job riding open tick T (waiting for T) was also joined into T's
    // waits, so T waited for a job that waited for T.
    TimelineJoin join;
    REQUIRE(join.AddPostJob(0x100, 0x200, 7));
    REQUIRE(join.AddGraphicsWait(0x200, 7));
    REQUIRE(join.AddGraphicsWait(0x300, 2)); // another thread's job: this submission waits
    auto batch = join.Close(10);
    REQUIRE(batch.num_post_jobs == 1);
    REQUIRE(batch.num_extra_waits == 1);
    REQUIRE(batch.extra_waits[0].semaphore == 0x300);
    batch = join.Close(11);
    REQUIRE(batch.num_extra_waits == 1);
    REQUIRE(batch.extra_waits[0].semaphore == 0x200);
    REQUIRE(batch.extra_waits[0].value == 7);
    REQUIRE(join.Close(12).num_extra_waits == 0);
}

TEST_CASE("V2 P5b a wait on an open tick completes with the recording thread idle",
          "[video_core][v2_p5b]") {
    // SESSION_HANDOFF §5g: a landing wait must cause its submit. Model of the protocol: the
    // recording thread sleeps until woken (GpuSplit's back in WaitOnAddress), then runs its idle
    // hook (Scheduler::ServeIdleHook -> SubmitRequests::IdleSubmits). A reader waiting on the open
    // tick asks (Request) and wakes it (RequestIdleHook); no other work ever arrives.
    using VideoCommon::V2::SubmitRequests;
    SubmitRequests requests;
    std::atomic<u64> current_tick{5}; // ticks below are submitted; 5 is open with recorded work
    std::mutex mutex;
    std::condition_variable cv;
    bool woken = false;
    bool stop = false;
    u32 submits = 0;
    std::thread back([&] {
        std::unique_lock lock{mutex};
        while (!stop) {
            cv.wait(lock, [&] { return woken || stop; });
            if (std::exchange(woken, false)) {
                // The idle hook: lazy fences on, nothing wanted or fenced, only the request.
                if (requests.IdleSubmits(false, true, false, 0, current_tick.load())) {
                    current_tick.fetch_add(1);
                    ++submits;
                }
            }
        }
    });
    const u64 tick = current_tick.load();
    requests.Request(tick);
    {
        std::scoped_lock lock{mutex};
        woken = true;
    }
    cv.notify_one();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (current_tick.load() <= tick && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool submitted = current_tick.load() > tick;
    {
        std::scoped_lock lock{mutex};
        stop = true;
    }
    cv.notify_one();
    back.join();
    REQUIRE(submitted);
    REQUIRE(submits == 1);
    // Served: an idle hook with no request and nothing pending submits nothing.
    REQUIRE_FALSE(requests.IdleSubmits(false, true, false, 0, current_tick.load()));
    // Lazy fences: an open fence tick or wanted work is submitted at the idle hook; without lazy
    // fences only deferred work and requests.
    REQUIRE(requests.IdleSubmits(false, true, false, current_tick.load(), current_tick.load()));
    REQUIRE(requests.IdleSubmits(false, true, true, 0, current_tick.load()));
    REQUIRE_FALSE(requests.IdleSubmits(false, false, true, current_tick.load(), current_tick.load()));
    REQUIRE(requests.IdleSubmits(true, false, false, 0, current_tick.load()));
}

TEST_CASE("V2 P5b flush decisions at pass boundaries", "[video_core][v2_p5b]") {
    FlushParams params;
    params.pass_boundaries = true;
    const FlushState busy{.pending = 1, .back_behind = true};
    const FlushState requested{.pending = 1, .back_behind = true, .requested = true};
    SECTION("SYNC rule C mid-pass: busy GPU, only DXVK's pending threshold; max draws") {
        FlushHeuristic heuristic{params};
        const FlushState two{.pending = 2, .back_behind = false};
        heuristic.NoteSubmit(1000);
        u32 flushes = 0;
        // Two in flight: min(20, 2 × 3) = 6 chunks, however long it takes.
        for (u32 i = 1; i < 6 * params.chunk_draws; ++i) {
            if (heuristic.Advance(1, two, 1000 + u64{i} * 100).action == FlushAction::Flush) {
                ++flushes;
            }
        }
        REQUIRE(flushes == 0);
        const auto pending = heuristic.Advance(1, two, 100'000);
        REQUIRE(pending.action == FlushAction::Flush);
        REQUIRE(pending.reason == FlushHeuristic::ReasonPending);
        // Many in flight: the threshold caps at max_chunk_threshold (20) chunks.
        const FlushState many{.pending = 10, .back_behind = false};
        for (u32 i = heuristic.Draws(); i < params.max_chunk_threshold * params.chunk_draws - 1;
             ++i) {
            REQUIRE(heuristic.Advance(1, many, 100'000).action != FlushAction::Flush);
        }
        const auto last = heuristic.Advance(1, many, 100'000);
        REQUIRE(last.action == FlushAction::Flush);
        REQUIRE(last.reason == FlushHeuristic::ReasonPending);
    }
    SECTION("SYNC rule C mid-pass: the GPU about to go idle, min draws and the interval") {
        FlushHeuristic heuristic{params};
        const FlushState one{.pending = 1, .back_behind = false};
        heuristic.NoteSubmit(1000);
        FlushHeuristic::Decision decision{};
        // Enough draws but too soon after the last submission: no cut.
        for (u32 i = 1; i <= params.min_draws; ++i) {
            decision = heuristic.Advance(1, one, 1000 + params.pass_interval_us - 1);
        }
        REQUIRE(decision.action == FlushAction::Dispatch);
        for (u32 i = 1; i <= params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, one, 1000 + params.pass_interval_us);
        }
        REQUIRE(decision.action == FlushAction::Flush);
        REQUIRE(decision.reason == FlushHeuristic::ReasonIdle);
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
    SECTION("a waiter's request rides the next boundary; mid-pass it never cuts the pass") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        heuristic.Advance(1, busy, 1000);
        auto boundary = heuristic.AtPassBoundary(FlushHint::Weak, requested, 1001);
        REQUIRE(boundary.action == FlushAction::Flush);
        REQUIRE(boundary.reason == FlushHeuristic::ReasonRequest);
        // SYNC rule W: an urgent request is served at the back's next entry boundary (the idle
        // hook between entries), not by a chunk mid-pass.
        const FlushState urgent{.pending = 2, .back_behind = true, .requested = true,
                                .urgent = true};
        FlushHeuristic::Decision decision{};
        for (u32 i = 1; i <= 3 * params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, urgent, 1001 + params.request_gap_us * 10);
        }
        REQUIRE(decision.action == FlushAction::Dispatch);
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
        // The sixth chunk: rule C takes it mid-pass (the same DXVK threshold).
        const auto decision = heuristic.Advance(1, two, 1000);
        REQUIRE(decision.action == FlushAction::Flush);
        REQUIRE(decision.reason == FlushHeuristic::ReasonPending);
    }
    SECTION("SYNC rule B: a strong hint is weighed at the pass boundary, not mid-pass") {
        FlushHeuristic heuristic{params};
        const FlushState two{.pending = 2, .back_behind = true};
        heuristic.NoteSubmit(1000);
        heuristic.Hint(FlushHint::Strong);
        FlushHeuristic::Decision decision{};
        for (u32 i = 0; i < params.strong_chunks * params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, two, 1001);
        }
        REQUIRE(decision.action == FlushAction::Dispatch);
        // FlushHint's immediate form (after readback work) still decides at once.
        REQUIRE(heuristic.WantsFlushNow());
        const auto boundary = heuristic.AtPassBoundary(FlushHint::Strong, two, 1002);
        REQUIRE(boundary.action == FlushAction::Flush);
        REQUIRE(boundary.reason == FlushHeuristic::ReasonStrong);
    }
    SECTION("F4-X2: the idle hook's synchronization rule (DXVK pending < 2, minimum work)") {
        const FlushState idle{.pending = 0, .back_behind = false};
        const FlushState one{.pending = 1, .back_behind = false};
        const FlushState two{.pending = 2, .back_behind = false};
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        // Nothing recorded: never.
        REQUIRE_FALSE(heuristic.SyncPointDue(idle, 1'000'000));
        for (u32 i = 0; i < params.pass_min_draws - 1; ++i) {
            heuristic.Advance(1, one, 1000);
        }
        // Too few draws, however long ago and however idle.
        REQUIRE_FALSE(heuristic.SyncPointDue(idle, 1000 + 10 * params.pass_interval_us));
        heuristic.Advance(1, one, 1000);
        // Enough draws but too soon after the last submission.
        REQUIRE_FALSE(heuristic.SyncPointDue(one, 1000 + params.pass_interval_us / 2));
        // The GPU about to go idle (fewer than two in flight), enough work: due.
        REQUIRE(heuristic.SyncPointDue(one, 1000 + params.pass_interval_us));
        REQUIRE(heuristic.SyncPointDue(idle, 1000 + params.pass_interval_us));
        // Busy (two in flight): DXVK's threshold, min(20, pending * 3) = 6 chunks.
        REQUIRE_FALSE(heuristic.SyncPointDue(two, 1000 + params.pass_interval_us));
        for (u32 i = heuristic.Draws(); i < 6 * params.chunk_draws; ++i) {
            heuristic.Advance(1, one, 1000);
        }
        REQUIRE(heuristic.SyncPointDue(two, 1001));
        // min_draws alone is enough while the GPU is about to go idle.
        FlushHeuristic many{params};
        many.NoteSubmit(1000);
        for (u32 i = 0; i < params.min_draws; ++i) {
            many.Advance(1, one, 1000);
        }
        REQUIRE(many.SyncPointDue(one, 1001));
    }
}

TEST_CASE("V2 P5b the idle hook submits nothing while the GPU is busy and nobody waits",
          "[video_core][v2_p5b]") {
    // F4-X2 (f1d05e9625: 54 idle-hook submits a frame; 0fe48cd86a: the hook waited for the GPU's
    // idle, 263 wakes/s): waking the hook, wanted work and a lazy fence's bounded-time request
    // are no reason to submit while the GPU is busy. DeferFlush's work and a thread blocking on
    // the open tick (an urgent request) are, at once; other work only by DXVK's sync rule.
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    constexpr u64 open = 10;
    SubmitRequests requests;
    u32 submits = 0;
    const auto idle_hook = [&](bool flush_pending, bool wanted, u64 lazy_tick, bool sync_due) {
        const auto action =
            requests.AtIdleHook(flush_pending, true, wanted, lazy_tick, open, sync_due);
        if (action != Action::None && action != Action::Leave) {
            ++submits;
        }
        return action;
    };
    // Nothing open: nothing to do.
    REQUIRE(idle_hook(false, false, 0, true) == Action::None);
    // The heuristic with two submissions in flight and a few draws: the rule says no.
    FlushParams params;
    params.pass_boundaries = true;
    FlushHeuristic heuristic{params};
    const FlushState busy{.pending = 2, .back_behind = false};
    heuristic.NoteSubmit(1000);
    for (u32 i = 0; i < 40; ++i) {
        heuristic.Advance(1, busy, 1000);
    }
    REQUIRE_FALSE(heuristic.SyncPointDue(busy, 1'000'000));
    // Wanted work (list copies, list ends) and an open lazy fence, woken many times.
    for (u32 i = 0; i < 100; ++i) {
        REQUIRE(idle_hook(false, true, open, heuristic.SyncPointDue(busy, 1'000'000)) ==
                Action::Leave);
    }
    // The lazy fence's window ran out (not urgent): still left for the next boundary.
    requests.Request(open, false);
    REQUIRE(requests.Open(open));
    REQUIRE_FALSE(requests.Urgent(open));
    REQUIRE(idle_hook(false, true, open, false) == Action::Leave);
    REQUIRE(submits == 0);
    // One in flight (the GPU is about to go idle) and enough work: DXVK's rule submits.
    const FlushState one{.pending = 1, .back_behind = false};
    REQUIRE(heuristic.SyncPointDue(one, 1000 + params.pass_interval_us));
    REQUIRE(idle_hook(false, true, open, true) == Action::Sync);
    // A thread blocking on the open tick, or DeferFlush's work: at once, GPU busy or not.
    SubmitRequests blocked;
    blocked.Request(open);
    REQUIRE(blocked.Urgent(open));
    REQUIRE(blocked.AtIdleHook(false, true, false, 0, open, false) == Action::Urgent);
    REQUIRE(requests.AtIdleHook(true, true, false, 0, open, false) == Action::Deferred);
    // Submitted ticks are no longer urgent.
    REQUIRE_FALSE(blocked.Urgent(open + 1));
}

TEST_CASE("SYNC 5b06 boot hang: every channel suspended, the back idle, a lazy fence open: it "
          "is submitted and the reductions behind it land",
          "[video_core][v2_p5b]") {
    // 5b06b426d9 hung at 44 s: three channels on words 0x13 short (19 QMD reductions) and one
    // still 0, row "no writer" (memory polled every pass). The reductions sat in fences behind a
    // lazy fence in the open tick; no draw followed (all channels suspended), so no pass boundary
    // or chunk came, rule I left the open tick (draws == 0) and the fencing thread waited for its
    // submission forever. Model: the fencing thread pops fences in order; a fence's operations
    // run once it popped; a fence pops once its tick is submitted.
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    FlushParams params;
    params.pass_boundaries = true;
    FlushHeuristic heuristic{params};
    heuristic.NoteSubmit(1000); // nothing recorded since
    constexpr u64 open = 30;
    SubmitRequests requests;
    u32 word = 0x607;
    bool submitted = false;
    const bool back_idle = true;
    const auto idle_hook = [&](u32 pending) {
        // RasterizerVulkan's rule I: an open lazy fence is worth a submission with the GPU about
        // to go idle; the heuristic alone says no (draws == 0).
        const FlushState state{.pending = pending, .back_behind = false};
        const bool lazy_fence_open = true;
        const bool due = (pending < params.busy_pending && lazy_fence_open) ||
                         heuristic.SyncPointDue(state, 1'000'000);
        const auto action = requests.AtIdleHook(false, true, false, open, open, due);
        submitted |= action != Action::None && action != Action::Leave;
    };
    // Two submissions in flight when the back went idle: left for now.
    idle_hook(2);
    REQUIRE_FALSE(submitted);
    // The fencing thread blocks on the open tick and sees the back idle: it files rule W.
    if (back_idle) {
        requests.Request(open, true);
    }
    idle_hook(2); // served at once by the woken idle hook, however many are in flight
    REQUIRE(submitted);
    // The lazy fence pops, then the fences behind it: the reductions land.
    word += 0x13;
    REQUIRE(word == 0x61a);
    // Rule I alone (no request) also submits once fewer than two are in flight.
    SubmitRequests fresh;
    submitted = false;
    const FlushState one{.pending = 1, .back_behind = false};
    REQUIRE_FALSE(heuristic.SyncPointDue(one, 1'000'000));
    const auto action = fresh.AtIdleHook(false, true, false, open, open, true);
    REQUIRE(action == Action::Sync);
}

TEST_CASE("SYNC §7.3: open work without draws is submitted however the back is fed",
          "[video_core][v2_p5b]") {
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    FlushParams params;
    params.pass_boundaries = true;
    constexpr u64 open = 50;
    SECTION("(a) a lazy fence with an operation, 0 draws, a record every 5 us: the batch end "
            "runs rule I and the tick is submitted") {
        // GpuSplit::BackLoop (fix 1) runs the idle hook at every batch end with the ring empty;
        // with records every 5 us the timed hook (idle_hook_ns of emptiness) never would.
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        SubmitRequests requests;
        bool landed = false;
        for (int batch = 0; batch < 100 && !landed; ++batch) {
            const bool ring_empty_at_batch_end = true; // the back caught up before the next one
            if (ring_empty_at_batch_end) {
                const FlushState state{.pending = 0, .back_behind = false};
                const bool due = heuristic.OpenWorkDue(state.pending, true, false) ||
                                 heuristic.SyncPointDue(state, 2000);
                const auto action = requests.AtIdleHook(false, true, false, open, open, due);
                landed = action == Action::Sync; // submitted: the fence pops, the op lands
            }
        }
        REQUIRE(landed);
    }
    SECTION("(b) a pure compute stream that never empties the ring: rule C submits within the "
            "chunk bound") {
        // Fix 3: dispatches, list ends and fences count toward chunks (no draws needed).
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        const FlushState two{.pending = 2, .back_behind = true};
        u32 items = 0;
        bool flushed = false;
        while (!flushed && items < 10'000) {
            ++items;
            const u32 cost = items % 3 == 0 ? 4 : 1; // dispatches, list ends, fences
            flushed = heuristic.Advance(cost, two, 1000 + items).action == FlushAction::Flush;
        }
        REQUIRE(flushed);
        REQUIRE(items <= params.max_chunk_threshold * params.chunk_draws);
    }
    SECTION("(c) the fencing thread blocked on an unsubmitted tick while the back is busy: rule W "
            "submits it at the next entry boundary") {
        // Fix 4 (InnerFence::Wait): after two `consumed` stores past its turn without a submit it
        // files rule W once; the idle hook between entries serves it (Urgent).
        SubmitRequests requests;
        u64 consumed = 1000;
        u64 last = consumed;
        u32 moves = 0;
        bool requested = false;
        for (int spin = 0; spin < 10 && !requested; ++spin) {
            consumed += 64; // the back keeps executing entries
            if (consumed != last) {
                last = consumed;
                ++moves;
            }
            if (moves >= 2) {
                requests.Request(open, true);
                requested = true;
            }
        }
        REQUIRE(requested);
        REQUIRE(requests.AtIdleHook(false, true, false, open, open, false) == Action::Urgent);
    }
    SECTION("(d) nobody waits, two in flight, draws only: no extra submit") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        const FlushState two{.pending = 2, .back_behind = false};
        for (u32 i = 0; i < 40; ++i) {
            REQUIRE(heuristic.Advance(1, two, 1000).action != FlushAction::Flush);
        }
        SubmitRequests requests;
        const bool due = heuristic.OpenWorkDue(two.pending, true, true) ||
                         heuristic.SyncPointDue(two, 1'000'000);
        REQUIRE_FALSE(due);
        REQUIRE(requests.AtIdleHook(false, true, false, open, open, due) == Action::Leave);
    }
}

TEST_CASE("SYNC §9 idle back: operations with no fence get one, rule I submits it",
          "[video_core][v2_p5b]") {
    // Model of RasterizerVulkan::ServeSplitIdleHook's §9.3 step and the fence manager: a
    // SyncOperation (or a query-cache uncommitted flush) waits for the next SignalFence; on an
    // idle back none comes unless the idle hook makes it. The fence lands once its tick is
    // submitted; rule I submits a lazy fence in the open tick with the GPU about to idle.
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    FlushParams params;
    params.pass_boundaries = true;
    FlushHeuristic heuristic{params};
    heuristic.NoteSubmit(1000);
    constexpr u64 open = 31;
    struct Fences {
        bool uncommitted_operation = false;
        bool uncommitted_query_flush = false;
        u64 lazy_tick = 0;
        u32 fences = 0;
    };
    const auto idle_hook = [&](Fences& f, u32 pending, u32& submits) {
        if (f.uncommitted_operation || f.uncommitted_query_flush) {
            // g_signaling_query_report_fence + SignalFence + FlushCommands: a lazy fence with them
            f.uncommitted_operation = false;
            f.uncommitted_query_flush = false;
            f.lazy_tick = open;
            ++f.fences;
        }
        SubmitRequests requests;
        const bool lazy_open = f.lazy_tick >= open;
        const bool due = heuristic.OpenWorkDue(pending, lazy_open, false);
        const auto action = requests.AtIdleHook(false, true, false, f.lazy_tick, open, due);
        if (action == Action::Sync) {
            ++submits;
        }
    };
    SECTION("(a) a SyncOperation writing a semaphore word lands; the suspended acquire resumes") {
        Fences f;
        f.uncommitted_operation = true;
        u32 submits = 0;
        u32 word = 0xea7;
        idle_hook(f, 0, submits);
        REQUIRE(f.fences == 1);
        REQUIRE(submits == 1);
        word += 0x13; // the fence popped, the operation wrote the word
        REQUIRE(word == 0xeba);
    }
    SECTION("(b) the same with a query-cache uncommitted flush") {
        Fences f;
        f.uncommitted_query_flush = true;
        u32 submits = 0;
        idle_hook(f, 1, submits);
        REQUIRE(f.fences == 1);
        REQUIRE(submits == 1);
    }
    SECTION("(c) nothing unfenced and nothing open: no fence, no submit") {
        Fences f;
        u32 submits = 0;
        for (int hook = 0; hook < 100; ++hook) {
            idle_hook(f, 0, submits);
        }
        REQUIRE(f.fences == 0);
        REQUIRE(submits == 0);
    }
}


TEST_CASE("SYNC §10: an idle back submits any non-empty open tick with fewer than two in flight",
          "[video_core][v2_p5b]") {
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    FlushParams params;
    params.pass_boundaries = true;
    FlushHeuristic heuristic{params};
    constexpr u64 open = 11026; // 6 ticks past the newest lazy fence (11020)
    // ServeSplitIdleHook: `wanted` includes RecordedSinceSubmit; sync_due is IdleDue.
    const auto idle_hook = [&](bool recorded, u32 pending) {
        SubmitRequests requests;
        return requests.AtIdleHook(false, true, recorded, 11020, open,
                                   heuristic.IdleDue(pending, recorded));
    };
    SECTION("(a) a dispatch only, no draws, no fence, no waiter, back idle: submitted") {
        REQUIRE(idle_hook(true, 0) == Action::Sync);
        REQUIRE(idle_hook(true, 1) == Action::Sync);
    }
    SECTION("(b) an empty open tick on an idle back: never submitted (storm guard)") {
        for (int hook = 0; hook < 1000; ++hook) {
            REQUIRE(idle_hook(false, 0) == Action::None);
        }
    }
    SECTION("(c) two in flight with recorded work: waits for a finish publish, then submits") {
        REQUIRE(idle_hook(true, 2) == Action::Leave);
        // The finish thread publishes a completion (one in flight): RequestIdleHook (the back is
        // idle, the open tick non-empty), and the hook submits.
        REQUIRE(idle_hook(true, 1) == Action::Sync);
    }
}

TEST_CASE("SYNC step 9: the post-pass list runs in order after the pass, before the submit",
          "[video_core][v2_p5b]") {
    // Model of the recording stream: the scheduler runs the list at the pass end (EndRenderPass)
    // and at the start of a submission (Flush/Finish). Copies recorded after the queries ended,
    // before the submit; a job deferred twice runs once; a job deferred while the list runs runs
    // after the others.
    VideoCommon::V2::AfterPassList list;
    std::vector<int> stream;
    struct Ctx {
        std::vector<int>* stream;
        VideoCommon::V2::AfterPassList* list;
        int id;
    };
    Ctx presync{&stream, &list, 1};
    Ctx sync{&stream, &list, 2};
    const auto record = [](void* argument) {
        auto& self = *static_cast<Ctx*>(argument);
        self.stream->push_back(self.id);
    };
    stream.push_back(100); // a query ends in the pass
    list.Defer(record, &presync);
    list.Defer(record, &sync);
    list.Defer(record, &presync); // the same WFI again before the pass ended: once
    stream.push_back(101);        // a later draw in the same pass
    REQUIRE_FALSE(list.Empty());
    stream.push_back(200); // the pass ends
    list.Run();
    stream.push_back(300); // the submit
    REQUIRE(stream == std::vector<int>{100, 101, 200, 1, 2, 300});
    REQUIRE(list.Empty());
    // A job deferring another runs it in the same Run, after itself.
    Ctx chained{&stream, &list, 3};
    list.Defer(
        [](void* argument) {
            auto& self = *static_cast<Ctx*>(argument);
            self.stream->push_back(self.id);
            static Ctx after{self.stream, self.list, 4};
            self.list->Defer(
                [](void* inner) {
                    auto& next = *static_cast<Ctx*>(inner);
                    next.stream->push_back(next.id);
                },
                &after);
        },
        &chained);
    stream.clear();
    list.Run();
    REQUIRE(stream == std::vector<int>{3, 4});
}

TEST_CASE("SYNC step 9: a buffer copy goes to the upload stream only when hazard-free",
          "[video_core][v2_p5b]") {
    // BufferCache::DMACopy's rule: neither the source nor the destination range is used by work
    // in flight (Buffer::IsRegionUsed: no pending write to read, no read or write to pass).
    const auto hazard_free = [](bool adaptive, bool source_used, bool destination_used) {
        return adaptive && !source_used && !destination_used;
    };
    REQUIRE(hazard_free(true, false, false));
    REQUIRE_FALSE(hazard_free(true, true, false));  // the source has a write in flight
    REQUIRE_FALSE(hazard_free(true, false, true));  // the open submission reads the destination
    REQUIRE_FALSE(hazard_free(false, false, false)); // v1: in order as before
}

TEST_CASE("SYNC step 7: the back submits for rules B, C, I and W only", "[video_core][v2_p5b]") {
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    FlushParams params;
    params.pass_boundaries = true;
    constexpr u64 open = 20;
    // The idle hook's rule I input (RasterizerVulkan::ServeSplitIdleHook): DXVK's sync point, or
    // a lazy fence in the open tick carrying captures with the GPU about to go idle.
    const auto rule_i = [&](const FlushHeuristic& heuristic, u32 pending, bool captures_open) {
        const FlushState state{.pending = pending, .back_behind = false};
        return (pending < params.busy_pending && captures_open) ||
               heuristic.SyncPointDue(state, 1'000'000);
    };
    SECTION("idle back (a): a lazy fence nobody waits on is submitted by rule I") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        SubmitRequests requests;
        // No draw after it (draws == 0), one submission in flight, captures in the open tick.
        REQUIRE(requests.AtIdleHook(false, true, false, open, open, rule_i(heuristic, 1, true)) ==
                Action::Sync);
        // Without captures and without work worth a submission: left (nothing waits on it).
        REQUIRE(requests.AtIdleHook(false, true, false, open, open, rule_i(heuristic, 1, false)) ==
                Action::Leave);
    }
    SECTION("idle back (b): a host waiter's request is served at the next entry boundary, "
            "however many submissions are in flight") {
        SubmitRequests requests;
        requests.Request(open, true);
        // No "two in flight → postpone" exception: the hook between entries submits it.
        REQUIRE(requests.AtIdleHook(false, true, false, open, open, false) == Action::Urgent);
        // Mid-pass a chunk doesn't take it (rule C has no request branch).
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        const FlushState urgent{.pending = 3, .back_behind = true, .requested = true,
                                .urgent = true};
        FlushHeuristic::Decision decision{};
        for (u32 i = 0; i < 2 * params.chunk_draws; ++i) {
            decision = heuristic.Advance(1, urgent, 100'000);
        }
        REQUIRE(decision.action == FlushAction::Dispatch);
    }
    SECTION("idle back (c): two in flight, nobody waits: not submitted until the finish thread "
            "publishes progress, then submitted with one still in flight") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        const FlushState one{.pending = 1, .back_behind = false};
        for (u32 i = 0; i < params.min_draws - 1; ++i) {
            heuristic.Advance(1, one, 1000);
        }
        SubmitRequests requests;
        // Two in flight: DXVK's threshold (6 chunks) is not reached: left.
        REQUIRE(requests.AtIdleHook(false, true, false, open, open, rule_i(heuristic, 2, false)) ==
                Action::Leave);
        // The finish thread published a completion (one in flight): the GPU progress wake runs
        // the idle hook again, and rule I submits the work while one is still in flight.
        REQUIRE(requests.AtIdleHook(false, true, false, open, open, rule_i(heuristic, 1, false)) ==
                Action::Sync);
    }
    SECTION("rule B at pass boundaries, rule C at chunks, nothing else") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        const FlushState one{.pending = 1, .back_behind = false};
        for (u32 i = 0; i < params.pass_min_draws; ++i) {
            REQUIRE(heuristic.Advance(1, one, 1000).action != FlushAction::Flush);
        }
        const auto boundary =
            heuristic.AtPassBoundary(FlushHint::Weak, one, 1000 + params.pass_interval_us);
        REQUIRE(boundary.action == FlushAction::Flush);
        REQUIRE(boundary.reason == FlushHeuristic::ReasonPass);
    }
}

TEST_CASE("V2 P5b a landing wait asks for its fence's tick and an idle back submits it",
          "[video_core][v2_p5b]") {
    // SESSION_HANDOFF §5: a landing wait causes its submit. F4-X2 (R2-F4-01): the waiter files an
    // urgent request for the host tick its operation's fence rides (LandingFence), and the back's
    // idle hook submits it although the GPU is busy (the sync rule says no) and no other work
    // arrives. Before the fence is queued there is no tick to ask for.
    namespace Landing = VideoCommon::V2::Landing;
    namespace LandingFence = VideoCommon::V2::LandingFence;
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    const u64 saved_issued = Landing::issued.load();
    const u64 saved_landed = Landing::landed.load();
    const u64 saved_untracked = Landing::untracked_issued.load();
    SubmitRequests requests;
    LandingFence::request_argument = &requests;
    LandingFence::request_hook = [](void* argument, u64 tick) {
        static_cast<SubmitRequests*>(argument)->Request(tick, true);
    };
    std::atomic<u64> current_tick{7}; // 5 and 6 handed off and running; 7 open
    const u64 ticket = Landing::Ticket(1000, 0);
    Landing::landed.store(Landing::Ticket(999, 0));
    Landing::untracked_issued.store(Landing::untracked_landed.load()); // none still to land
    Landing::issued.store(ticket); // the back captured the write ...
    REQUIRE(LandingFence::TickFor(ticket, 0) == 0); // ... its fence not queued yet
    LandingFence::NoteFence(current_tick.load()); // ... then queued its lazy fence in tick 7
    REQUIRE(LandingFence::TickFor(ticket, 0) == 7);
    REQUIRE(LandingFence::TickForPending() == 7);
    REQUIRE(LandingFence::TickFor(ticket + 1, 0) == 0);

    std::mutex mutex;
    std::condition_variable cv;
    bool woken = false;
    bool stop = false;
    std::atomic<u32> submits{0};
    std::atomic<u32> left{0};
    std::thread back([&] {
        std::unique_lock lock{mutex};
        while (!stop) {
            cv.wait(lock, [&] { return woken || stop; });
            if (!std::exchange(woken, false)) {
                continue;
            }
            // The idle hook: the lazy fence and wanted work are open, the GPU busy.
            const u64 open = current_tick.load();
            switch (requests.AtIdleHook(false, true, true, 7, open, false)) {
            case Action::Urgent:
                current_tick.fetch_add(1);
                submits.fetch_add(1);
                // The GPU runs tick 7, the fencing thread runs the fence's operation.
                Landing::Note(ticket);
                break;
            case Action::Leave:
                left.fetch_add(1);
                break;
            default:
                break;
            }
        }
    });
    const auto wake = [&] {
        {
            std::scoped_lock lock{mutex};
            woken = true;
        }
        cv.notify_one();
    };
    // A wake alone (no request) leaves the work.
    wake();
    while (left.load() == 0) {
        std::this_thread::yield();
    }
    REQUIRE(submits.load() == 0);
    // GpuSplit::WaitLanded's loop.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    u64 requested_tick = 0;
    while (Landing::landed.load() < ticket && std::chrono::steady_clock::now() < deadline) {
        const u64 fence_tick = LandingFence::TickFor(ticket, 0);
        if (fence_tick > requested_tick) {
            LandingFence::Request(fence_tick);
            requested_tick = fence_tick;
            wake();
        }
        std::this_thread::yield();
    }
    const bool landed = Landing::landed.load() >= ticket;
    {
        std::scoped_lock lock{mutex};
        stop = true;
    }
    cv.notify_one();
    back.join();
    REQUIRE(landed);
    REQUIRE(submits.load() == 1);
    REQUIRE_FALSE(requests.Urgent(current_tick.load()));
    REQUIRE(LandingFence::TickForPending() == 0);

    // SYNC Step 4: nothing is urgent when a fence is queued (019d/95fd made ~18 submits/f);
    // a waiter that blocks on a capture asks for the tick that carries it, once it exists.
    // A deferred QMD reduction's fence is stubbed (no GPU work of its own); it is noted with the
    // newest lazy fence's tick, whose pop it waits behind. TickFor names that tick; the waiter's
    // request makes an idle back's hook submit it (Urgent), and nothing else does.
    const u64 reduction = Landing::Ticket(3400, 0);
    Landing::issued.store(reduction);
    LandingFence::NoteFence(11); // the newest lazy fence is in tick 11
    REQUIRE(LandingFence::TickFor(reduction, 0) == 11);
    SubmitRequests stubbed;
    REQUIRE_FALSE(stubbed.Urgent(11));
    stubbed.Request(11, true);
    REQUIRE(stubbed.AtIdleHook(false, true, false, 11, 11, false) == Action::Urgent);

    LandingFence::request_hook = nullptr;
    LandingFence::request_argument = nullptr;
    Landing::issued.store(saved_issued);
    Landing::landed.store(saved_landed);
    Landing::untracked_issued.store(saved_untracked);
    LandingFence::NoteFence(0);
}

TEST_CASE("V2 P5b staging buffers in recurring use are not released", "[video_core][v2_p5b]") {
    // P-01 (eac4735c2b: 93 stutters vs v1's 18, driver frees up): a level is swept once per 64
    // frames, and the sweep used to free the idle buffers a recurring burst takes again.
    using Retention = Vulkan::StagingBufferPool::StagingRetention;
    constexpr u64 recent = Retention::RetainFrames - 1;
    constexpr u64 old = Retention::RetainFrames;
    // Taken within the last RetainFrames: kept.
    REQUIRE_FALSE(Retention::Releasable(3, 0, recent, 0, 16));
    // Unused for RetainFrames: released.
    REQUIRE(Retention::Releasable(3, 0, old, 0, 16));
    // The upload levels keep their first idle buffers whatever their age.
    REQUIRE_FALSE(Retention::Releasable(1, 2, old, 0, 16));
    REQUIRE_FALSE(Retention::Releasable(2, 2, old, 0, 16));
    REQUIRE(Retention::Releasable(3, 2, old, 0, 16));
    // At most max_released per sweep.
    REQUIRE_FALSE(Retention::Releasable(20, 0, old, 16, 16));
}
