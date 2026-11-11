// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <functional>
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
#include "video_core/v2/cpu_readback.h"
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
    REQUIRE(ring.Allocate(256, 1, 0) == u64{0});
    REQUIRE(ring.Allocate(256, 1, 0) == u64{256});
    REQUIRE(ring.Allocate(256, 2, 0) == u64{512});
    REQUIRE(ring.Allocate(256, 2, 0) == u64{768});
    // Full: fails instead of waiting.
    REQUIRE_FALSE(ring.Allocate(64, 3, 0).has_value());
    // Tick 1 done: its 512 bytes come back, allocation wraps to the start.
    REQUIRE(ring.Allocate(300, 3, 1) == u64{0});
    REQUIRE_FALSE(ring.Allocate(300, 3, 1).has_value()); // 512-320 = 192 free before tick 2's
    REQUIRE(ring.Allocate(128, 3, 1) == u64{320});
    // Everything done: empty again.
    REQUIRE(ring.Allocate(1024, 4, 3) == u64{0});
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
    SECTION("SYNC §11.2: the sleep point's rule S (any open work; two in flight with a wake)") {
        FlushHeuristic heuristic{params};
        REQUIRE_FALSE(heuristic.SleepSubmitDue(0, false, true)); // an empty tick never
        REQUIRE(heuristic.SleepSubmitDue(0, true, true));
        REQUIRE(heuristic.SleepSubmitDue(1, true, true));
        REQUIRE_FALSE(heuristic.SleepSubmitDue(2, true, true)); // the finish thread wakes it
        REQUIRE(heuristic.SleepSubmitDue(2, true, false));      // no wake would come
    }
}

// SYNC_SUBMIT_SPEC §12.4: 9acdba08ad made every fence with an operation a one-chunk strong hint
// (6829: 2-3k such fences/s, 23-25 of ~50 submits/f). Only committed downloads hint (strong,
// DXVK's strong_chunks, at pass boundaries), and a host-waited syncpoint files rule W; deferred
// reductions, releases, reports and SyncOperations ride their tick.
TEST_CASE("SYNC §12.4 R3: fences with operations ride their tick; downloads hint at boundaries",
          "[video_core][v2_p5b]") {
    FlushParams params;
    params.pass_boundaries = true;
    const FlushState one{.pending = 1, .back_behind = false};
    SECTION("10 chunks, each ending in a deferred reduction's fence, no waiter: no mid-pass submit") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        for (u32 chunk = 0; chunk < 10; ++chunk) {
            for (u32 i = 1; i < params.chunk_draws; ++i) {
                REQUIRE(heuristic.Advance(1, one, 1000).action != FlushAction::Flush);
            }
            // The fence (RasterizerVulkan::SignalFence → NoteNonDrawWork): work, no hint.
            REQUIRE(heuristic.Advance(1, one, 1000).action != FlushAction::Flush);
        }
        // Too soon for the pass boundary's rules: still open.
        REQUIRE(heuristic.AtPassBoundary(FlushHint::None, one, 1001).action ==
                FlushAction::None);
        // The boundary's own rule (Ryujinx's interval, DXVK's thresholds) decides.
        const auto boundary =
            heuristic.AtPassBoundary(FlushHint::None, one, 1000 + params.pass_interval_us);
        REQUIRE(boundary.action == FlushAction::Flush);
        REQUIRE(boundary.reason == FlushHeuristic::ReasonPass);
    }
    SECTION("a committed download: strong, at a boundary after strong_chunks, never mid-pass") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        heuristic.Hint(FlushHint::Strong);
        for (u32 i = 0; i < params.chunk_draws; ++i) {
            REQUIRE(heuristic.Advance(1, one, 1000).action != FlushAction::Flush);
        }
        REQUIRE(heuristic.AtPassBoundary(FlushHint::Weak, one, 1001).action ==
                FlushAction::None);
        for (u32 i = params.chunk_draws; i < params.strong_chunks * params.chunk_draws; ++i) {
            REQUIRE(heuristic.Advance(1, one, 1001).action != FlushAction::Flush);
        }
        const auto boundary = heuristic.AtPassBoundary(FlushHint::Weak, one, 1002);
        REQUIRE(boundary.action == FlushAction::Flush);
        REQUIRE(boundary.reason == FlushHeuristic::ReasonStrong);
    }
    SECTION("a host-waited syncpoint fence: rule W for its tick at the next entry boundary") {
        using VideoCommon::V2::SubmitRequests;
        SubmitRequests requests;
        // Nobody waits: the idle hook leaves the open tick (lazy fence) for B/C/S.
        REQUIRE(requests.AtIdleHook(false, true, false, 9, 9) == SubmitRequests::IdleAction::Leave);
        requests.Request(9, true); // RasterizerVulkan::RequestHostWaitedFence
        REQUIRE(requests.AtIdleHook(false, true, false, 9, 9) ==
                SubmitRequests::IdleAction::Urgent);
        REQUIRE_FALSE(requests.Urgent(10)); // submitted
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
    const auto idle_hook = [&](bool flush_pending, bool wanted, u64 lazy_tick) {
        const auto action = requests.AtIdleHook(flush_pending, true, wanted, lazy_tick, open);
        if (action != Action::None && action != Action::Leave) {
            ++submits;
        }
        return action;
    };
    // Nothing open: nothing to do.
    REQUIRE(idle_hook(false, false, 0) == Action::None);
    FlushParams params;
    params.pass_boundaries = true;
    FlushHeuristic heuristic{params};
    // Wanted work (list copies, list ends) and an open lazy fence, woken many times.
    for (u32 i = 0; i < 100; ++i) {
        REQUIRE(idle_hook(false, true, open) == Action::Leave);
    }
    // The lazy fence's window ran out (not urgent): still left for the next boundary.
    requests.Request(open, false);
    REQUIRE(requests.Open(open));
    REQUIRE_FALSE(requests.Urgent(open));
    REQUIRE(idle_hook(false, true, open) == Action::Leave);
    REQUIRE(submits == 0);
    // SYNC §11: the open work goes out at the back's sleep point once fewer than two are in
    // flight (rule S), never from a wake of the hook.
    REQUIRE_FALSE(heuristic.SleepSubmitDue(2, true, true));
    REQUIRE(heuristic.SleepSubmitDue(1, true, true));
    // A thread blocking on the open tick, or DeferFlush's work: at once, GPU busy or not.
    SubmitRequests blocked;
    blocked.Request(open);
    REQUIRE(blocked.Urgent(open));
    REQUIRE(blocked.AtIdleHook(false, true, false, 0, open) == Action::Urgent);
    REQUIRE(requests.AtIdleHook(true, true, false, 0, open) == Action::Deferred);
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
        // The idle hook serves rule W; then the back's sleep point (rule S): an open lazy fence
        // makes the open tick non-empty, submitted with fewer than two in flight.
        const auto action = requests.AtIdleHook(false, true, false, open, open);
        submitted |= action != Action::None && action != Action::Leave;
        const bool lazy_fence_open = true;
        submitted |= heuristic.SleepSubmitDue(pending, lazy_fence_open, true);
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
    // Rule S alone (no request) also submits once fewer than two are in flight.
    SubmitRequests fresh;
    submitted = false;
    REQUIRE(fresh.AtIdleHook(false, true, false, open, open) == Action::Leave);
    REQUIRE(heuristic.SleepSubmitDue(1, true, true));
}

TEST_CASE("SYNC §7.3: open work without draws is submitted however the back is fed",
          "[video_core][v2_p5b]") {
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    FlushParams params;
    params.pass_boundaries = true;
    constexpr u64 open = 50;
    SECTION("(a) a lazy fence with an operation, 0 draws, a record every 5 us: each record counts "
            "toward rule C, and a front stall reaches the sleep point (S)") {
        // §11.6 item 3: no idle hook at batch ends any more. While the front keeps publishing,
        // the fences and list ends it carries advance the chunk count; once it stops, the back
        // sleeps only after submitting the non-empty open tick.
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        SubmitRequests requests;
        const FlushState state{.pending = 0, .back_behind = false};
        bool landed = false;
        u32 records = 0;
        for (; records < 10'000 && !landed; ++records) {
            REQUIRE(requests.AtIdleHook(false, true, false, open, open) == Action::Leave);
            landed = heuristic.Advance(1, state, 1000 + 5 * records).action == FlushAction::Flush;
        }
        REQUIRE(landed);
        REQUIRE(records <= params.max_draws);
        // The front stalls after a single fence: the sleep point submits it.
        REQUIRE(heuristic.SleepSubmitDue(state.pending, true, true));
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
        REQUIRE(requests.AtIdleHook(false, true, false, open, open) == Action::Urgent);
    }
    SECTION("(d) nobody waits, two in flight, draws only: no extra submit") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        const FlushState two{.pending = 2, .back_behind = false};
        for (u32 i = 0; i < 40; ++i) {
            REQUIRE(heuristic.Advance(1, two, 1000).action != FlushAction::Flush);
        }
        SubmitRequests requests;
        REQUIRE_FALSE(heuristic.SleepSubmitDue(two.pending, true, true));
        REQUIRE(requests.AtIdleHook(false, true, false, open, open) == Action::Leave);
    }
}

TEST_CASE("SYNC §9 idle back: operations with no fence get one, rule S submits it",
          "[video_core][v2_p5b]") {
    // Model of RasterizerVulkan::ServeIdleSleep's step 1 (FenceUnfenced) and the fence manager: a
    // SyncOperation (or a query-cache uncommitted flush) waits for the next SignalFence; on an
    // idle back none comes unless the sleep point makes it. The fence lands once its tick is
    // submitted; rule S submits the lazy fence's open tick with the GPU about to idle.
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
        const bool lazy_open = f.lazy_tick >= open;
        if (heuristic.SleepSubmitDue(pending, lazy_open, true)) {
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


TEST_CASE("SYNC §10/§11: a sleeping back submits any non-empty open tick with fewer than two in "
          "flight",
          "[video_core][v2_p5b]") {
    FlushParams params;
    params.pass_boundaries = true;
    FlushHeuristic heuristic{params};
    // The back's sleep point (ServeIdleSleep): RecordedSinceSubmit is the open tick's work.
    const auto sleep_point = [&](bool recorded, u32 pending) {
        return heuristic.SleepSubmitDue(pending, recorded, true);
    };
    SECTION("(a) a dispatch only, no draws, no fence, no waiter, back idle: submitted") {
        REQUIRE(sleep_point(true, 0));
        REQUIRE(sleep_point(true, 1));
    }
    SECTION("(b) an empty open tick on an idle back: never submitted (storm guard)") {
        for (int sleep = 0; sleep < 1000; ++sleep) {
            REQUIRE_FALSE(sleep_point(false, 0));
        }
    }
    SECTION("(c) two in flight with recorded work: waits for a finish publish, then submits") {
        REQUIRE_FALSE(sleep_point(true, 2));
        // The finish thread publishes a completion (one in flight) and wakes the sleeping back
        // (open tick non-empty); its next sleep point submits.
        REQUIRE(sleep_point(true, 1));
    }
}

namespace {
// SYNC_SUBMIT_SPEC §11.2 invariant S. Model of GpuSplit::BackLoop's sleep block with
// RasterizerVulkan::ServeIdleSleep (the back) and Scheduler::OnGpuProgress (the finish thread),
// with the same stores, fences and loads; ticks as MasterSemaphore counts them.
struct SleepModel {
    FlushHeuristic heuristic{};
    std::atomic<u64> current{4}; // the open tick: 1-3 submitted
    std::atomic<u64> known{1};   // two in flight
    std::atomic<bool> nonempty{true};
    std::atomic<bool> back_waiting{false};
    std::atomic<u32> wake{0};
    std::atomic<bool> hook_requested{false};
    std::atomic<u32> submits{0};
    bool wake_guaranteed = true; // a finish thread publishes the tick

    u32 Pending() const {
        const u64 cur = current.load(std::memory_order_acquire);
        const u64 done = known.load(std::memory_order_acquire);
        return static_cast<u32>(cur > done + 1 ? cur - done - 1 : 0);
    }
    bool Hook(bool prepare) {
        const bool due = heuristic.SleepSubmitDue(
            Pending(), nonempty.load(std::memory_order_relaxed), wake_guaranteed);
        if (prepare && due) {
            current.fetch_add(1);
            nonempty.store(false, std::memory_order_seq_cst);
            submits.fetch_add(1);
            return false;
        }
        return !prepare && due;
    }
    u32 Announce() {
        back_waiting.store(true, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        return wake.load(std::memory_order_seq_cst);
    }
    void Publish() {
        known.fetch_add(1, std::memory_order_release); // MasterSemaphore::Refresh's CAS
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (back_waiting.load(std::memory_order_seq_cst) &&
            nonempty.load(std::memory_order_seq_cst)) {
            hook_requested.store(true, std::memory_order_seq_cst); // RequestIdleHook
            wake.fetch_add(1, std::memory_order_seq_cst);
        }
    }
};
} // namespace

TEST_CASE("SYNC §11.2 invariant S: no completion wake is lost and no open tick sleeps unsubmitted",
          "[video_core][v2_p5b]") {
    SECTION("3fd8: a completion between the back's check and its sleep: the back submits") {
        SleepModel m;
        m.Hook(true); // two in flight: no submit
        REQUIRE(m.submits.load() == 0);
        m.Publish(); // the back is not waiting yet: no wake (the 3fd8 race)
        REQUIRE_FALSE(m.hook_requested.load());
        const u32 seen = m.Announce();
        // The old protocol slept here on `seen` and nothing changed `wake`: the freeze.
        REQUIRE(m.wake.load() == seen);
        const bool must_submit = m.Hook(false); // the Dekker re-check sees one in flight
        REQUIRE(must_submit);
        m.back_waiting.store(false);
        m.Hook(true);
        REQUIRE(m.submits.load() == 1);
        REQUIRE(m.Pending() == 2); // the open tick went out with one still in flight
    }
    SECTION("the completion after the back's sleep announcement wakes it, then it submits") {
        SleepModel m;
        m.Hook(true);
        const u32 seen = m.Announce();
        REQUIRE_FALSE(m.Hook(false)); // still two in flight: sleep
        m.Publish();
        REQUIRE(m.wake.load() != seen); // wake.wait(seen) returns
        REQUIRE(m.hook_requested.load());
        m.back_waiting.store(false);
        m.Hook(true);
        REQUIRE(m.submits.load() == 1);
    }
    SECTION("threads: the finish thread publishes at any moment of the back's sleep block") {
        u32 lost = 0;
        for (u32 iteration = 0; iteration < 1000 && lost == 0; ++iteration) {
            SleepModel m;
            std::jthread finish([&] {
                for (u32 spin = 0; spin < (iteration * 7) % 97; ++spin) {
                    std::atomic_signal_fence(std::memory_order_seq_cst);
                }
                m.Publish();
            });
            while (m.submits.load() == 0) {
                m.Hook(true);
                if (m.submits.load() != 0) {
                    break;
                }
                const u32 seen = m.Announce();
                const bool must_submit = m.Hook(false);
                if (!must_submit && !m.hook_requested.exchange(false)) {
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{1};
                    while (m.wake.load() == seen && std::chrono::steady_clock::now() < deadline) {
                        std::this_thread::yield();
                    }
                    if (m.wake.load() == seen) {
                        ++lost; // asleep on a non-empty open tick with one in flight
                        break;
                    }
                }
                m.back_waiting.store(false);
                m.hook_requested.store(false);
            }
            finish.join();
        }
        REQUIRE(lost == 0);
    }
    SECTION("9c4a b2 / d5ed / 49e1: idle back, open tick non-empty, nothing in flight: submitted "
            "before the sleep") {
        SleepModel m;
        m.known.store(3); // nothing in flight
        m.Hook(true);
        REQUIRE(m.submits.load() == 1);
        REQUIRE_FALSE(m.nonempty.load());
        // The new open tick is empty: the back sleeps, and no number of sleeps submits it.
        for (int sleep = 0; sleep < 1000; ++sleep) {
            m.Hook(true);
            REQUIRE_FALSE(m.Hook(false));
        }
        REQUIRE(m.submits.load() == 1);
    }
    SECTION("§11.6 item 3, steady state: the front publishes every 5 us with draws: no submit "
            "but B/C (3fd8's 55/f guard)") {
        // Model of BackLoop: a batch end runs no hook; the sleep point is reached only when the
        // ring stays empty past the spin window. One 16.6 ms frame, a draw per publish, a pass
        // boundary every 150 draws, one submission in flight.
        FlushParams params;
        params.pass_boundaries = true;
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1);
        constexpr u64 spin_window_us = 20;
        constexpr u64 gap_us = 5;
        const FlushState one{.pending = 1, .back_behind = false};
        u32 rule_bc = 0;
        u32 rule_s = 0;
        u32 draws = 0;
        for (u64 now = 1; now < 16'600; now += gap_us) {
            ++draws;
            if (draws % 150 == 0 && heuristic.AtPassBoundary(FlushHint::Weak, one, now).action ==
                                         FlushAction::Flush) {
                ++rule_bc;
            }
            if (heuristic.Advance(1, one, now).action == FlushAction::Flush) {
                ++rule_bc;
            }
            if (gap_us > spin_window_us && heuristic.SleepSubmitDue(one.pending, true, true)) {
                ++rule_s; // never: the next publish comes inside the spin window
            }
        }
        REQUIRE(rule_s == 0);
        REQUIRE(rule_bc >= 1);
        REQUIRE(rule_bc <= 40);
    }
    SECTION("§11.6 item 4, 0c86 / e412: a zero-draw stream (copies, clears, fences) that never "
            "empties the ring is submitted within the chunk bound") {
        // NoteNonDrawWork(1) after each DMA copy/clear, surface copy, fence and list end: the back
        // never reaches its sleep point, rule C alone bounds the open tick.
        FlushParams params;
        params.pass_boundaries = true;
        for (const u32 pending : {0u, 1u, 2u, 5u}) {
            FlushHeuristic heuristic{params};
            heuristic.NoteSubmit(1);
            const FlushState state{.pending = pending, .back_behind = true};
            u32 items = 0;
            bool submitted = false;
            while (!submitted && items < 10'000) {
                ++items;
                submitted = heuristic.Advance(1, state, 1 + items).action == FlushAction::Flush;
            }
            REQUIRE(submitted);
            REQUIRE(items <= params.max_draws);
        }
    }
    SECTION("no finish thread (no completion wake): a non-empty open tick never sleeps") {
        SleepModel m;
        m.wake_guaranteed = false;
        m.known.store(0); // three in flight
        m.Hook(true);
        REQUIRE(m.submits.load() == 1);
    }
}

TEST_CASE("SYNC §11.6 item 7: submits by rule are only B/C/W/S/present/readback (+ non-split)",
          "[video_core][v2_p5b]") {
    namespace P5b = VideoCommon::V2::P5bStats;
    using P5b::Counter;
    const auto has = [](Counter counter) {
        return std::find(P5b::SubmitRuleCounters.begin(), P5b::SubmitRuleCounters.end(),
                         counter) != P5b::SubmitRuleCounters.end();
    };
    for (const Counter rule : {Counter::SubmitsPassBoundary, Counter::SubmitsStrong,
                               Counter::SubmitsChunk, Counter::SubmitsWaiter, Counter::SubmitsSleep,
                               Counter::SubmitsPresent, Counter::SubmitsReadback}) {
        REQUIRE(has(rule));
    }
    // DeferFlush's and Scheduler::ServeIdleHook's own submits exist only without lazy fences or
    // the split; nothing else may be a submit rule.
    REQUIRE(P5b::SubmitRuleCounters.size() == 9);
    REQUIRE(has(Counter::SubmitsDeferred));
    REQUIRE(has(Counter::SubmitsIdle));
    // No counter is listed twice (the "other" figure would go negative).
    for (size_t i = 0; i < P5b::SubmitRuleCounters.size(); ++i) {
        for (size_t j = i + 1; j < P5b::SubmitRuleCounters.size(); ++j) {
            REQUIRE(P5b::SubmitRuleCounters[i] != P5b::SubmitRuleCounters[j]);
        }
    }
}

TEST_CASE("SYNC §11.2 step 1: the unfenced and owed writebacks are fenced before the back sleeps",
          "[video_core][v2_p5b]") {
    // Model of RasterizerVulkan::FenceUnfenced + the fence manager's SignalFence with
    // EDEN_NX2_V2_WRITEBACK (defer_writebacks): a non-observable fence leaves the caches'
    // uncommitted downloads owed; an observable one commits them, and they land at its pop.
    struct Fences {
        bool uncommitted_operation = false; // SyncOperation / a deferred reduction's op
        bool owed_download = false;         // a cache download nobody committed
        bool defer_writebacks = true;
        bool fence_queued = false;
        bool fence_carries_download = false;
        bool open_work = false;
        u32 fences = 0;
        void SignalFence(bool observable) {
            fence_carries_download = owed_download && (observable || !defer_writebacks);
            owed_download = owed_download && !fence_carries_download;
            uncommitted_operation = false; // moved into the fence's pending operations
            fence_queued = true;
            open_work = true; // a lazy fence in the open tick (NoteLazyFence)
            ++fences;
        }
        bool FenceUnfenced(bool owed_writebacks) {
            if (!uncommitted_operation && !(owed_writebacks && owed_download)) {
                return false;
            }
            SignalFence(true);
            return true;
        }
    };
    FlushHeuristic heuristic{};
    SECTION("8d31 / 019d / 5b06: a reduction deferred behind a landing plus an unfenced op, back "
            "idle: fenced, submitted, landed, the acquire resumes") {
        Fences f;
        f.uncommitted_operation = true;
        u32 word = 0x1000;
        const u32 target = 0x1013; // the acquire waits for the 0x13 reduction
        REQUIRE(f.FenceUnfenced(true));
        REQUIRE(heuristic.SleepSubmitDue(0, f.open_work, true)); // step 2 submits it
        // The finish thread completes the tick, the fencing thread pops the fence: the operation
        // writes the word and Landing::Note rings the front.
        word += 0x13;
        REQUIRE(word == target);
        REQUIRE_FALSE(f.FenceUnfenced(true)); // nothing left: no second fence
        REQUIRE(f.fences == 1);
    }
    SECTION("WRITEBACK: an owed download, the observable fence at the sleep puts it in memory") {
        Fences f;
        f.owed_download = true;
        // §9.3's old check (operations and query flushes only) saw nothing to fence: W7.
        REQUIRE_FALSE(f.FenceUnfenced(false));
        REQUIRE(f.owed_download);
        REQUIRE(f.FenceUnfenced(true));
        REQUIRE(f.fence_carries_download);
        REQUIRE_FALSE(f.owed_download);
        // A non-observable fence would have left it owed.
        Fences g;
        g.owed_download = true;
        g.SignalFence(false);
        REQUIRE(g.owed_download);
    }
    SECTION("nothing unfenced, nothing owed: no fence") {
        Fences f;
        for (int sleep = 0; sleep < 100; ++sleep) {
            REQUIRE_FALSE(f.FenceUnfenced(true));
        }
        REQUIRE(f.fences == 0);
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

TEST_CASE("SYNC step 7 / §11: the back submits for rules B, C, W and S only",
          "[video_core][v2_p5b]") {
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    FlushParams params;
    params.pass_boundaries = true;
    constexpr u64 open = 20;
    SECTION("idle back (a): a lazy fence nobody waits on is submitted at the sleep point (S)") {
        FlushHeuristic heuristic{params};
        heuristic.NoteSubmit(1000);
        SubmitRequests requests;
        // No draw after it (draws == 0), one submission in flight: the idle hook leaves it, the
        // sleep point submits it (the lazy fence makes the open tick non-empty).
        REQUIRE(requests.AtIdleHook(false, true, false, open, open) == Action::Leave);
        REQUIRE(heuristic.SleepSubmitDue(1, true, true));
        // Nothing in the open tick: the back sleeps without a submit.
        REQUIRE_FALSE(heuristic.SleepSubmitDue(1, false, true));
    }
    SECTION("idle back (b): a host waiter's request is served at the next entry boundary, "
            "however many submissions are in flight") {
        SubmitRequests requests;
        requests.Request(open, true);
        // No "two in flight → postpone" exception: the hook between entries submits it.
        REQUIRE(requests.AtIdleHook(false, true, false, open, open) == Action::Urgent);
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
        // Two in flight: the idle hook leaves it and the back may sleep (a wake will come).
        REQUIRE(requests.AtIdleHook(false, true, false, open, open) == Action::Leave);
        REQUIRE_FALSE(heuristic.SleepSubmitDue(2, true, true));
        // The finish thread published a completion (one in flight) and woke the back: its sleep
        // point submits the work while one is still in flight.
        REQUIRE(heuristic.SleepSubmitDue(1, true, true));
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
            switch (requests.AtIdleHook(false, true, true, 7, open)) {
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
    REQUIRE(stubbed.AtIdleHook(false, true, false, 11, 11) == Action::Urgent);

    LandingFence::request_hook = nullptr;
    LandingFence::request_argument = nullptr;
    Landing::issued.store(saved_issued);
    Landing::landed.store(saved_landed);
    Landing::untracked_issued.store(saved_untracked);
    LandingFence::NoteFence(0);
}

// SYNC_SUBMIT_SPEC §12 R4: df94's back blocked in Finish for guest CPU reads (buffer_cache.h
// DownloadMemory 15-18 ms/s, vk_texture_cache.cpp:1507 25.7 ms/s). The reader now lands them.
TEST_CASE("SYNC §12 R4: a guest CPU read lands on the reader; an idle back submits, never finishes",
          "[video_core][v2_p5b]") {
    namespace CpuReadback = VideoCommon::V2::CpuReadback;
    using VideoCommon::V2::SubmitRequests;
    using Action = SubmitRequests::IdleAction;
    SECTION("guest memory keeps FlushRegion's order: images, then buffers") {
        REQUIRE(CpuReadback::ImagesOnReader(false, false, false)); // no buffer data owed
        REQUIRE(CpuReadback::ImagesOnReader(false, true, true));   // buffers land on it too
        REQUIRE_FALSE(CpuReadback::ImagesOnReader(false, true, false)); // buffers written now
        REQUIRE_FALSE(CpuReadback::ImagesOnReader(true, false, false)); // verify
        std::string order;
        CpuReadback::Chain([&] { order += 'i'; }, [&] { order += 'b'; })();
        REQUIRE(order == "ib");
        order.clear();
        CpuReadback::Chain({}, [&] { order += 'b'; })();
        REQUIRE(order == "b");
        order.clear();
        CpuReadback::Chain([&] { order += 'i'; }, {})();
        REQUIRE(order == "i");
    }
    SECTION("idle back, the writer's tick unsubmitted: the reader's request submits it") {
        SubmitRequests requests;
        std::atomic<u64> open_tick{7}; // the writer and its download copy are in tick 7
        std::atomic<u64> gpu_done{6};
        std::atomic<u32> staging{0};
        std::atomic<u32> submits{0};
        std::atomic<u32> finishes{0};
        std::mutex mutex;
        std::condition_variable cv;
        bool woken = false;
        bool stop = false;
        std::thread back([&] {
            std::unique_lock lock{mutex};
            while (!stop) {
                cv.wait(lock, [&] { return woken || stop; });
                if (!std::exchange(woken, false)) {
                    continue;
                }
                // The back's idle hook (lazy fences, nothing wanted): only a request submits.
                if (requests.AtIdleHook(false, true, false, 0, open_tick.load()) ==
                    Action::Urgent) {
                    open_tick.fetch_add(1);
                    submits.fetch_add(1);
                    staging.store(0x1234); // the GPU runs the writer and the copy
                    gpu_done.store(7);
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
        // A wake without a request leaves the tick open.
        wake();
        u32 guest = 0;
        // The reader's land (WaitForHostRead: EnsureSubmitted, then the timeline wait).
        const std::function<void()> land = [&] {
            requests.Request(7, true);
            wake();
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
            while (gpu_done.load() < 7 && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            guest = staging.load();
        };
        CpuReadback::Chain({}, land)();
        {
            std::scoped_lock lock{mutex};
            stop = true;
        }
        cv.notify_one();
        back.join();
        REQUIRE(guest == 0x1234);
        REQUIRE(submits.load() == 1);
        REQUIRE(finishes.load() == 0);
        REQUIRE_FALSE(requests.Urgent(open_tick.load()));
    }
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
