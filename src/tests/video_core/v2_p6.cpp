// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <deque>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"

#include "video_core/v2/commit.h"
#include "video_core/v2/epoch_publish.h"
#include "video_core/v2/pipeline_lookup.h"
#include "video_core/v2/resolve/cow_regs.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/resolve/dynamic_state.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/v2/resolve/parallel_resolve.h"
#include "video_core/v2/resolve/segment.h"
#include "video_core/v2/resolve/split_resolve.h"
#include "video_core/v2/resolve/state_view.h"
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/ir_arena.h"

namespace {

namespace V2 = VideoCommon::V2;
namespace Resolve = VideoCommon::V2::Resolve;
namespace Commit = VideoCommon::V2::Commit;
namespace MF = VideoCommon::V2::Resolve::MultiFront;
using Resolve::CutReason;
using Resolve::RecordClass;
using Resolve::SerialReason;

Resolve::DrawEvent Draw(u32 dirty = 0, u64 engine = 1) {
    return Resolve::DrawEvent{.dirty_groups = dirty, .engine = engine};
}

} // Anonymous namespace

TEST_CASE("V2 P6: segmenter cuts at render passes, sizes and engine switches", "[video_core][v2_p6]") {
    SECTION("policy strings") {
        CHECK(Resolve::SegmentPolicy::Parse(nullptr).kind == Resolve::SegmentPolicy::Kind::Adaptive);
        CHECK(Resolve::SegmentPolicy::Parse("adaptive").k == Resolve::SegmentPolicy::DefaultK);
        const auto pass = Resolve::SegmentPolicy::Parse("pass");
        CHECK(pass.kind == Resolve::SegmentPolicy::Kind::Pass);
        CHECK(pass.k == Resolve::SegmentPolicy::MaxK);
        const auto fixed = Resolve::SegmentPolicy::Parse("12");
        CHECK(fixed.kind == Resolve::SegmentPolicy::Kind::Fixed);
        CHECK(fixed.k == 12);
        CHECK(Resolve::SegmentPolicy::Parse("bogus").kind ==
              Resolve::SegmentPolicy::Kind::Adaptive);
    }
    SECTION("boundaries") {
        Resolve::Segmenter segmenter{Resolve::SegmentPolicy::Parse("4")};
        CHECK(segmenter.OnDraw(Draw()) == CutReason::First);
        CHECK(segmenter.OpenId() == 0);
        for (int i = 0; i < 3; ++i) {
            CHECK(segmenter.OnDraw(Draw()) == CutReason::None);
        }
        CHECK(segmenter.OnDraw(Draw()) == CutReason::Size);
        CHECK(segmenter.OpenId() == 1);
        // A render target change in a segment of fewer than min(16, K) draws joins it ...
        CHECK(segmenter.OnDraw(Draw(V2::GroupBit(V2::RegGroup::RenderTargets))) ==
              CutReason::None);
        CHECK(segmenter.OnDraw(Draw()) == CutReason::None);
        CHECK(segmenter.OnDraw(Draw()) == CutReason::None);
        // ... and cuts a longer one (a render pass change) ...
        CHECK(segmenter.OnDraw(Draw(V2::GroupBit(V2::RegGroup::RenderTargets))) ==
              CutReason::RenderPass);
        CHECK(segmenter.OpenId() == 2);
        // ... other groups don't.
        CHECK(segmenter.OnDraw(Draw(V2::GroupBit(V2::RegGroup::Pipeline))) == CutReason::None);
        // Another channel's 3D engine.
        CHECK(segmenter.OnDraw(Draw(0, 2)) == CutReason::EngineSwitch);
        CHECK(segmenter.OpenId() == 3);
        CHECK(segmenter.OpenDraws() == 1);
    }
    SECTION("render passes only") {
        Resolve::Segmenter segmenter{Resolve::SegmentPolicy::Parse("pass")};
        segmenter.OnDraw(Draw());
        for (u32 i = 1; i < Resolve::SegmentPolicy::MaxK; ++i) {
            REQUIRE(segmenter.OnDraw(Draw()) == CutReason::None);
        }
        CHECK(segmenter.OnDraw(Draw()) == CutReason::Size); // the cap
    }
    SECTION("adaptive size") {
        Resolve::Segmenter segmenter{};
        segmenter.SetSize(8);
        CHECK(segmenter.Size() == Resolve::SegmentPolicy::MinK);
        segmenter.SetSize(100);
        CHECK(segmenter.Size() == 100);
        Resolve::AdaptiveSize adaptive;
        u32 size = 0;
        for (u32 i = 0; i < Resolve::AdaptiveSize::Window; ++i) {
            size = adaptive.Note(i % 4 == 0, false, 100); // 25% waits
        }
        CHECK(size == 75);
        for (u32 i = 0; i < Resolve::AdaptiveSize::Window; ++i) {
            size = adaptive.Note(false, false, 100); // shards ahead, never waited
        }
        CHECK(size == 126);
        Resolve::Segmenter fixed{Resolve::SegmentPolicy::Parse("32")};
        fixed.SetSize(100);
        CHECK(fixed.Size() == 32); // only the adaptive policy moves
    }
}

TEST_CASE("V2 P6: segmenter serial classification keeps the first reason and isolates",
          "[video_core][v2_p6]") {
    Resolve::Segmenter segmenter{Resolve::SegmentPolicy::Parse("64")};
    segmenter.OnRecord(RecordClass::Transfer); // before the first draw: nothing open
    CHECK(segmenter.OnDraw(Draw()) == CutReason::First);
    CHECK(segmenter.OpenReason() == SerialReason::None);
    // Register-only resolvers read no record's effects: no cut, no serial segment.
    segmenter.OnRecord(RecordClass::Transfer);
    segmenter.OnRecord(RecordClass::Readback);
    segmenter.OnRecord(RecordClass::Map);
    segmenter.OnRecord(RecordClass::CacheInvalidate);
    CHECK(segmenter.OpenReason() == SerialReason::None);
    CHECK(segmenter.OnDraw(Draw()) == CutReason::None);
    // Resolvers that read memory: the draws after a transfer start a segment, the open one
    // stays parallel (no isolate cascade).
    segmenter.SetRecordCapabilities(1U << static_cast<u32>(RecordClass::Transfer));
    segmenter.OnRecord(RecordClass::Neutral);
    segmenter.OnRecord(RecordClass::Map);
    CHECK(segmenter.OnDraw(Draw()) == CutReason::None);
    segmenter.OnRecord(RecordClass::Transfer);
    CHECK(segmenter.OpenReason() == SerialReason::None);
    CHECK(segmenter.OnDraw(Draw()) == CutReason::Isolate);
    CHECK(segmenter.OpenReason() == SerialReason::None);
    CHECK(segmenter.OnDraw(Draw()) == CutReason::None);

    Resolve::DrawEvent indirect = Draw();
    indirect.indirect = true;
    CHECK(segmenter.OnDraw(indirect) == CutReason::None);
    CHECK(segmenter.OpenReason() == SerialReason::IndirectCount);
    Resolve::DrawEvent inline_index = Draw();
    inline_index.inline_index = true;
    CHECK(segmenter.OnDraw(inline_index) == CutReason::Isolate);
    CHECK(segmenter.OpenReason() == SerialReason::InlineIndex);
    segmenter.OnDraw(Draw());
    segmenter.MarkSerial(SerialReason::GpuWrittenHandles);
    CHECK(segmenter.OpenReason() == SerialReason::GpuWrittenHandles);
    for (size_t i = 0; i < Resolve::NumSerialReasons; ++i) {
        CHECK(Resolve::serial_reason_names[i] != nullptr);
    }
}

TEST_CASE("V2 P6: copy-on-write snapshots equal the registers at their time",
          "[video_core][v2_p6]") {
    constexpr size_t Words = 0xe00;
    using Snapshot = Resolve::RegisterSnapshot<Words>;
    std::mt19937 rng{1234};
    std::vector<u32> plain(Words, 0);
    Resolve::CowRegisterFile<Words> cow;
    Resolve::SnapshotBuilder<Words> builder;
    struct Taken {
        std::vector<u32> copy;
        Snapshot cow;
        Snapshot built;
    };
    std::vector<Taken> taken;
    for (u32 step = 0; step < 40000; ++step) {
        // Writes cluster in a few blocks, as register traffic does.
        const u32 index = rng() % 256 == 0 ? rng() % Words : (rng() % 4) * 64 + rng() % 64;
        const u32 value = rng() % 16;
        plain[index] = value;
        cow.Write(index, value);
        if (step % 997 == 0) {
            taken.push_back(Taken{plain, cow.TakeSnapshot(), builder.Take(plain.data())});
        }
    }
    REQUIRE(taken.size() > 30);
    std::vector<u32> out(Words);
    for (const Taken& t : taken) {
        t.cow.CopyTo(out.data());
        REQUIRE(out == t.copy);
        t.built.CopyTo(out.data());
        REQUIRE(out == t.copy);
        CHECK(t.cow.Fingerprint() == Snapshot::FingerprintOf(t.copy.data()));
        CHECK(t.built.Fingerprint() == Snapshot::FingerprintOf(t.copy.data()));
        for (u32 i = 0; i < Words; i += 101) {
            REQUIRE(t.built.Read(i) == t.copy[i]);
        }
    }
    // Consecutive snapshots share the blocks nobody wrote.
    CHECK(taken[5].built.SharedWith(taken[4].built) > Snapshot::NumBlocks / 2);
    CHECK(builder.shared > builder.copied);
    CHECK(cow.copies > 0);

    // A file loaded from a snapshot writes copies, never the snapshot's blocks.
    Resolve::CowRegisterFile<Words> loaded;
    loaded.Load(taken[3].cow);
    loaded.Write(7, 0xdead);
    taken[3].cow.CopyTo(out.data());
    CHECK(out == taken[3].copy);
    CHECK(loaded.Read(7) == 0xdead);
}

TEST_CASE("V2 P6: epoch-reclaimed snapshot blocks stay intact while their segment is unreleased",
          "[video_core][v2_p6]") {
    constexpr size_t Words = 0xe00;
    using Snapshot = Resolve::RegisterSnapshot<Words>;
    std::mt19937 rng{77};
    std::vector<u32> plain(Words, 0);
    Resolve::SnapshotBuilder<Words> builder;
    struct Taken {
        std::vector<u32> copy;
        Snapshot snapshot;
    };
    std::deque<Taken> live; // the segments the commit has not released
    u64 released = 0;
    for (u64 epoch = 1; epoch <= 4000; ++epoch) {
        for (u32 w = 0; w < 40; ++w) {
            plain[(rng() % 8) * 64 + rng() % 64] = rng() % 16;
        }
        live.push_back(Taken{plain, builder.Take(plain.data(), ~u64{0}, epoch)});
        // The commit lags by a few segments, then releases some.
        while (live.size() > 1 + rng() % 6) {
            live.pop_front();
            ++released;
        }
        builder.Reclaim(released);
        std::vector<u32> out(Words);
        for (const Taken& t : live) {
            t.snapshot.CopyTo(out.data());
            REQUIRE(out == t.copy);
        }
    }
    CHECK(builder.copied > 4000);
}

TEST_CASE("V2 P6: a segment's snapshot plus its deltas gives the serial state at every draw",
          "[video_core][v2_p6]") {
    // A synthetic IR stream: register writes and draws. Each segment starts from the snapshot taken
    // at its cut and replays only its own writes; the state at each of its draws must equal the
    // serial state there.
    constexpr size_t Words = 512;
    struct Op {
        bool draw;
        u32 index;
        u32 value;
        u32 dirty;
    };
    std::mt19937 rng{99};
    std::vector<Op> stream;
    for (u32 i = 0; i < 20000; ++i) {
        if (rng() % 6 == 0) {
            const u32 dirty = rng() % 40 == 0 ? V2::GroupBit(V2::RegGroup::RenderTargets) : 0;
            stream.push_back(Op{true, 0, 0, dirty});
        } else {
            stream.push_back(Op{false, static_cast<u32>(rng() % Words), rng() % 8, 0});
        }
    }
    // Serial pass: the state at every draw.
    std::vector<std::vector<u32>> serial_states;
    std::vector<u32> regs(Words, 0);
    for (const Op& op : stream) {
        if (op.draw) {
            serial_states.push_back(regs);
        } else {
            regs[op.index] = op.value;
        }
    }
    // Segmented pass.
    Resolve::Segmenter segmenter{Resolve::SegmentPolicy::Parse("16")};
    Resolve::SnapshotBuilder<Words> builder;
    struct Segment {
        Resolve::RegisterSnapshot<Words> snapshot;
        std::vector<Op> ops;
    };
    std::vector<Segment> segments;
    std::fill(regs.begin(), regs.end(), 0);
    for (const Op& op : stream) {
        if (op.draw && segmenter.OnDraw(Draw(op.dirty)) != CutReason::None) {
            segments.push_back(Segment{builder.Take(regs.data()), {}});
        }
        if (!segments.empty()) {
            segments.back().ops.push_back(op);
        }
        if (!op.draw) {
            regs[op.index] = op.value;
        }
    }
    size_t draw = 0;
    for (const Segment& segment : segments) {
        std::vector<u32> state(Words);
        segment.snapshot.CopyTo(state.data());
        for (const Op& op : segment.ops) {
            if (op.draw) {
                REQUIRE(state == serial_states[draw]);
                ++draw;
            } else {
                state[op.index] = op.value;
            }
        }
    }
    CHECK(draw == serial_states.size());
    CHECK(segments.size() > serial_states.size() / 16);
}

namespace {

// ---- A mock world for the merge and stress tests -------------------------------------------------

constexpr u32 NumHandles = 512;

/// The mock texture cache: handle -> view slot, published with release stores (lookups are pure
/// acquire loads, as P5a's published maps); only the commit applier creates views.
struct MockCache {
    std::array<std::atomic<u32>, NumHandles> view_of{};
    u32 next_slot = 0;
    u64 prepares = 0;

    MockCache() {
        for (auto& view : view_of) {
            view.store(Commit::InvalidSlot, std::memory_order_relaxed);
        }
    }
    u32 Apply(const Commit::Request& request) {
        if (request.kind == Commit::Kind::ViewCreate) {
            const u32 handle = static_cast<u32>(request.address);
            u32 slot = view_of[handle].load(std::memory_order_relaxed);
            if (slot == Commit::InvalidSlot) { // ensure semantics
                slot = next_slot++;
                view_of[handle].store(slot, std::memory_order_release);
            }
            return slot;
        }
        ++prepares;
        return request.slot;
    }
};

struct SynthDraw {
    std::array<u32, 3> handles;
    u32 pipeline_state;
    u32 delay_us; ///< how long a shard takes (out-of-order completion)
};

struct MockInput {
    std::vector<SynthDraw> draws;
    u64 first_draw = 0;
};

struct MockRecord {
    u64 draw = 0;
    std::array<u32, 3> handles{};
    std::array<u32, 3> views{};
    u64 pipeline = 0;

    bool operator==(const MockRecord&) const = default;
};

MockCache* g_cache = nullptr;

struct MockTraits {
    using Input = MockInput;
    using Record = MockRecord;
    struct Shard {
        u32 index = 0;
    };

    static SerialReason Resolve(Shard&, const Input& input, std::vector<Record>& records,
                                Commit::RequestList& requests) {
        for (size_t i = 0; i < input.draws.size(); ++i) {
            const SynthDraw& draw = input.draws[i];
            if (draw.delay_us != 0) {
                // Busy wait: sleeps are timer-quantised on some hosts.
                const auto until = std::chrono::steady_clock::now() +
                                   std::chrono::microseconds(draw.delay_us);
                while (std::chrono::steady_clock::now() < until) {
                }
            }
            MockRecord record;
            record.draw = input.first_draw + i;
            record.pipeline = V2::Mix64(draw.pipeline_state); // a pure pipeline lookup
            for (size_t t = 0; t < 3; ++t) {
                const u32 handle = draw.handles[t];
                record.handles[t] = handle;
                const u32 view = g_cache->view_of[handle].load(std::memory_order_acquire);
                if (view != Commit::InvalidSlot) {
                    record.views[t] = view;
                } else {
                    Commit::Request request{};
                    request.kind = Commit::Kind::ViewCreate;
                    request.resource = Commit::Resource::ImageView;
                    request.address = handle;
                    record.views[t] = requests.Push(request, true);
                }
                // Every bound view is prepared at commit (an upload or the LRU).
                Commit::Request prepare{};
                prepare.kind = Commit::Kind::ImagePrepare;
                prepare.resource = Commit::Resource::ImageView;
                prepare.slot = record.views[t];
                requests.Push(prepare);
            }
            records.push_back(record);
        }
        return SerialReason::None;
    }
    static void Patch(Record& record, const Commit::RequestList& requests) {
        for (u32& view : record.views) {
            view = requests.Resolve(view);
        }
    }
    static const char* Compare(const Record& a, const Record& b) {
        if (a.draw != b.draw) {
            return "draw";
        }
        if (a.handles != b.handles) {
            return "handles";
        }
        for (size_t t = 0; t < 3; ++t) {
            if (!Commit::IsProxy(a.views[t]) && a.views[t] != b.views[t]) {
                return "views";
            }
        }
        return a.pipeline != b.pipeline ? "pipeline" : nullptr;
    }
};

std::vector<SynthDraw> MakeDraws(u32 count, u32 seed, u32 max_delay_us) {
    std::mt19937 rng{seed};
    std::vector<SynthDraw> draws(count);
    for (SynthDraw& draw : draws) {
        for (u32& handle : draw.handles) {
            // Mostly a hot set, sometimes a new handle (a miss: a create request with a proxy).
            handle = rng() % 4 == 0 ? rng() % NumHandles : rng() % 32;
        }
        draw.pipeline_state = rng() % 16;
        draw.delay_us = max_delay_us != 0 ? rng() % max_delay_us : 0;
    }
    return draws;
}

/// The serial reference: one thread, resolve and apply each draw at once.
std::vector<MockRecord> SerialReference(const std::vector<SynthDraw>& draws, u64* prepares) {
    MockCache cache;
    MockCache* const saved = g_cache;
    g_cache = &cache;
    std::vector<MockRecord> output;
    MockTraits::Shard shard;
    for (size_t i = 0; i < draws.size(); ++i) {
        MockInput input{{draws[i]}, i};
        input.draws[0].delay_us = 0;
        std::vector<MockRecord> records;
        Commit::RequestList list;
        list.Reset(i);
        MockTraits::Resolve(shard, input, records, list);
        Commit::Stage stage;
        stage.Apply(list, [&](const Commit::Request& request) { return cache.Apply(request); });
        MockTraits::Patch(records[0], list);
        output.push_back(records[0]);
    }
    *prepares = cache.prepares;
    g_cache = saved;
    return output;
}

/// Runs the parallel resolve: a producer thread segments and submits, the calling thread commits.
std::vector<MockRecord> ParallelRun(const std::vector<SynthDraw>& draws, u32 shards, u32 depth,
                                    u32 segment, bool verify, u64* prepares,
                                    std::vector<u64>* serial_segments = nullptr) {
    MockCache cache;
    g_cache = &cache;
    std::vector<MockRecord> output;
    {
        using Resolver = Resolve::ParallelResolver<MockTraits>;
        Resolver::Config config;
        config.shards = shards;
        config.depth = depth;
        config.verify = verify;
        config.stats = true;
        Resolver resolver{config, [](u32 index) { return MockTraits::Shard{index}; }};
        const u64 segments = (draws.size() + segment - 1) / segment;
        const auto input_of = [&](u64 id) {
            MockInput input;
            input.first_draw = id * segment;
            const size_t end = (std::min)(draws.size(), static_cast<size_t>((id + 1) * segment));
            input.draws.assign(draws.begin() + static_cast<std::ptrdiff_t>(id * segment),
                               draws.begin() + static_cast<std::ptrdiff_t>(end));
            return input;
        };
        std::jthread producer{[&] {
            for (u64 id = 0; id < segments; ++id) {
                // Every 13th segment is serial up front (an indirect draw, say).
                const SerialReason reason =
                    id % 13 == 5 ? SerialReason::IndirectCount : SerialReason::None;
                resolver.Submit(id, input_of(id), reason);
            }
        }};
        for (u64 id = 0; id < segments; ++id) {
            const SerialReason reason = resolver.Commit(
                id,
                [&](Resolver::Result& result) {
                    // Refused (back pressure) or not submitted yet: resolve from our own copy.
                    MockTraits::Shard shard;
                    MockTraits::Resolve(shard, input_of(id), result.records, result.requests);
                },
                [&](const Commit::Request& request) { return cache.Apply(request); },
                [&](std::span<const MockRecord> records) {
                    output.insert(output.end(), records.begin(), records.end());
                });
            if (serial_segments != nullptr && reason != SerialReason::None) {
                serial_segments->push_back(id);
            }
        }
    }
    *prepares = cache.prepares;
    g_cache = nullptr;
    return output;
}

} // Anonymous namespace

TEST_CASE("V2 P6: the ordered merge commits in segment order whatever order shards finish in",
          "[video_core][v2_p6]") {
    const std::vector<SynthDraw> draws = MakeDraws(1200, 7, 120);
    u64 serial_prepares = 0;
    const std::vector<MockRecord> reference = SerialReference(draws, &serial_prepares);
    const u64 failures_before =
        Resolve::GetStats().determinism_failures.load(std::memory_order_relaxed);
    u64 prepares = 0;
    std::vector<u64> serial_segments;
    const std::vector<MockRecord> output =
        ParallelRun(draws, 4, 8, 12, false, &prepares, &serial_segments);
    REQUIRE(output.size() == reference.size());
    for (size_t i = 0; i < output.size(); ++i) {
        REQUIRE(output[i].draw == i);
        REQUIRE(output[i] == reference[i]);
    }
    CHECK(prepares == serial_prepares);
    CHECK(Resolve::GetStats().determinism_failures.load(std::memory_order_relaxed) ==
          failures_before);
    // Serial-up-front segments were resolved on the commit thread.
    CHECK(std::ranges::find(serial_segments, 5) != serial_segments.end());
    CHECK(std::ranges::find(serial_segments, 18) != serial_segments.end());
}

TEST_CASE("V2 P6: stress: parallel resolve equals serial resolve on synthetic streams",
          "[video_core][v2_p6]") {
    for (u32 round = 0; round < 6; ++round) {
        const std::vector<SynthDraw> draws = MakeDraws(4000, 100 + round, round % 2 ? 8 : 0);
        u64 serial_prepares = 0;
        const std::vector<MockRecord> reference = SerialReference(draws, &serial_prepares);
        const u32 shards = 1 + round % 4;
        const u32 depth = round % 3 == 0 ? 2 : 16; // depth 2 forces back pressure
        const u32 segment = 8 + round * 5;
        const bool verify = round % 2 == 0;
        const u64 mismatches_before =
            Resolve::GetStats().verify_mismatches.load(std::memory_order_relaxed);
        u64 prepares = 0;
        const std::vector<MockRecord> output =
            ParallelRun(draws, shards, depth, segment, verify, &prepares);
        REQUIRE(output == reference);
        CHECK(prepares == serial_prepares);
        if (verify) {
            // Shards that saw a view missing which an earlier segment's commit created count as
            // dependent, never as mismatches.
            CHECK(Resolve::GetStats().verify_mismatches.load(std::memory_order_relaxed) ==
                  mismatches_before);
        }
    }
    CHECK(Resolve::GetStats().determinism_failures.load(std::memory_order_relaxed) == 0);
}

TEST_CASE("V2 P6: verify request comparison separates dependent differences",
          "[video_core][v2_p6]") {
    Commit::RequestList shard;
    Commit::RequestList serial;
    shard.Reset(1);
    serial.Reset(1);
    Commit::Request create{};
    create.kind = Commit::Kind::ViewCreate;
    create.address = 5;
    Commit::Request prepare{};
    prepare.kind = Commit::Kind::ImagePrepare;
    // The shard creates (proxy) and prepares the proxy; serially the view existed (slot 9).
    prepare.slot = shard.Push(create, true);
    shard.Push(prepare);
    prepare.slot = 9;
    serial.Push(prepare);
    Resolve::RequestCompare result =
        Resolve::CompareRequests(shard.Requests(), serial.Requests());
    CHECK(result.field == nullptr);
    CHECK(result.dependent == 2);
    // A real difference.
    Commit::Request upload{};
    upload.kind = Commit::Kind::BufferUpload;
    upload.address = 0x1000;
    serial.Push(upload);
    result = Resolve::CompareRequests(shard.Requests(), serial.Requests());
    CHECK(result.field != nullptr);
}

namespace {

MF::Event Ev(u32 channel, MF::EventKind kind, u64 address = 0, u64 value = 0,
             MF::AcquireOp op = MF::AcquireOp::GreaterEqual) {
    return MF::Event{address, value, channel, kind, op};
}

/// Producer/consumer channels with semaphores, an independent channel with a macro upload, and two
/// channels sharing the QMD table.
std::vector<MF::Event> CoupledStream(u32 seed) {
    std::mt19937 rng{seed};
    std::vector<MF::Event> events;
    u64 released = 0;
    u64 acquired = 0;
    for (u32 i = 0; i < 3000; ++i) {
        switch (rng() % 9) {
        case 0:
            events.push_back(Ev(1, MF::EventKind::Release, 0x100, ++released));
            break;
        case 1:
            // The consumer waits for the producer's next value (often not released yet).
            if (i % 6 == 1 && acquired < released + 2) {
                events.push_back(Ev(2, MF::EventKind::Acquire, 0x100, ++acquired));
            }
            break;
        case 2:
            events.push_back(Ev(4, MF::EventKind::QmdUse, rng() % 8));
            break;
        case 3:
            events.push_back(Ev(5, MF::EventKind::QmdUse, rng() % 8));
            break;
        case 4:
            if (i % 500 == 4) {
                events.push_back(Ev(3, MF::EventKind::MacroUpload, i));
            } else {
                events.push_back(Ev(2, MF::EventKind::MacroCall));
            }
            break;
        default:
            events.push_back(Ev(1 + rng() % 5, MF::EventKind::Work));
            break;
        }
    }
    // Whatever the consumer still waits for gets released at the end.
    while (released < acquired) {
        events.push_back(Ev(1, MF::EventKind::Release, 0x100, ++released));
    }
    return events;
}

} // Anonymous namespace

TEST_CASE("V2 P6b: versioned stores answer as of a ticket", "[video_core][v2_p6]") {
    MF::VersionedStore<u64> store;
    store.Publish(0x10, 5, 1);
    store.Publish(0x10, 9, 2);
    store.Publish(0x10, 7, 3); // out of order publication (another engine)
    CHECK(!store.Read(0x10, 5).has_value());
    CHECK(store.Read(0x10, 6)->value == 1);
    CHECK(store.Read(0x10, 8)->value == 3);
    CHECK(store.Read(0x10, 100)->value == 2);
    CHECK(store.FirstWhere(0x10, [](u64 v) { return v >= 2; })->ticket == 7);
    CHECK(!store.Read(0x20, 100).has_value());
    store.Retire(8);
    CHECK(store.Read(0x10, 8)->value == 3);
    CHECK(store.Versions() == 3);
}

TEST_CASE("V2 P6b: channel grouping and single-front fallbacks", "[video_core][v2_p6]") {
    SECTION("QMD sharing joins channels, semaphores stay edges") {
        MF::Grouper grouper;
        for (const MF::Event& event : CoupledStream(3)) {
            grouper.Note(event);
        }
        const MF::Plan plan = grouper.Decide(4);
        CHECK(plan.fallback == MF::Fallback::None);
        CHECK(plan.GroupOf(4) == plan.GroupOf(5));
        CHECK(plan.GroupOf(1) != plan.GroupOf(2));
        CHECK(plan.groups >= 2);
        CHECK(plan.cross_edges > 0);
        CHECK(plan.joins_qmd > 0);
    }
    SECTION("one channel") {
        MF::Grouper grouper;
        for (u32 i = 0; i < 100; ++i) {
            grouper.Note(Ev(7, MF::EventKind::Work));
        }
        CHECK(grouper.Decide(4).fallback == MF::Fallback::SingleChannel);
    }
    SECTION("everything joined") {
        MF::Grouper grouper;
        for (u32 i = 0; i < 100; ++i) {
            grouper.Note(Ev(1 + i % 3, MF::EventKind::QmdUse, 0));
        }
        CHECK(grouper.Decide(4).fallback == MF::Fallback::OneGroup);
    }
    SECTION("too coupled by semaphores") {
        MF::Grouper grouper;
        for (u32 i = 0; i < 400; ++i) {
            grouper.Note(Ev(1, MF::EventKind::Release, 0x40, i));
            grouper.Note(Ev(2, MF::EventKind::Acquire, 0x40, i));
        }
        const MF::Plan plan = grouper.Decide(4);
        CHECK(plan.fallback == MF::Fallback::SemaphoreRate);
        CHECK(plan.groups == 1);
    }
}

TEST_CASE("V2 P6b: multi-front merge keeps the cross-channel order semaphores define",
          "[video_core][v2_p6]") {
    for (u32 stream_seed = 0; stream_seed < 8; ++stream_seed) {
        const std::vector<MF::Event> events = CoupledStream(40 + stream_seed);
        const std::vector<u64> reference = MF::SingleFrontOrder(events);
        REQUIRE(reference.size() == events.size());
        REQUIRE(MF::MergeInvariantHolds(events, reference));
        MF::Grouper grouper;
        for (const MF::Event& event : events) {
            grouper.Note(event);
        }
        MF::Plan plan = grouper.Decide(4);
        plan.fallback = MF::Fallback::None; // exercise the merge even at a high edge rate
        if (plan.groups < 2) {
            continue;
        }
        for (u64 seed = 0; seed < 24; ++seed) {
            std::vector<MF::Processed> log;
            const std::vector<u64> merged = MF::MultiFrontOrder(events, plan, seed, &log);
            REQUIRE(merged.size() == events.size());
            REQUIRE(MF::MergeInvariantHolds(events, merged));
            const MF::CompareResult result = MF::Compare(events, reference, merged);
            INFO("stream " << stream_seed << " seed " << seed << " first "
                           << (result.first_field ? result.first_field : "-") << " at "
                           << result.first_index);
            REQUIRE(result.mismatches == 0);
            CHECK(result.compared > 0);
        }
    }
}

TEST_CASE("V2 P6b: the merge reports a dependency cycle instead of hanging",
          "[video_core][v2_p6]") {
    // Channel 1 waits for channel 2's release, which comes after channel 2 waits for channel 1's.
    const std::vector<MF::Event> events{
        Ev(1, MF::EventKind::Acquire, 0xa, 1),
        Ev(2, MF::EventKind::Acquire, 0xb, 1),
        Ev(1, MF::EventKind::Release, 0xb, 1),
        Ev(2, MF::EventKind::Release, 0xa, 1),
    };
    MF::Plan plan;
    plan.groups = 2;
    plan.group_of = {{1, 0}, {2, 1}};
    CHECK(MF::MultiFrontOrder(events, plan, 0).empty());
}

namespace {

/// A hand-built typed IR ring for the split's shard resolver.
struct RingBuilder {
    static constexpr size_t RingWords = size_t{1} << 20;
    std::vector<u64> ring = std::vector<u64>(RingWords, 0);
    V2::Ir::PayloadArena arena;
    u64 position = 0;
    u64 sequence = 0;

    RingBuilder() {
        arena.Allocate(1 << 16, 1 << 10);
    }
    void Wrap() {
        ring[position & (RingWords - 1)] = 0; // Kind::Wrap
        position += RingWords - (position & (RingWords - 1));
    }
    u64* Reserve(size_t words) {
        if ((position & (RingWords - 1)) + words > RingWords) {
            Wrap();
        }
        u64* const record = &ring[position & (RingWords - 1)];
        position += words;
        return record;
    }
    void StateRun(const std::vector<std::pair<u32, u32>>& writes) {
        const size_t words = V2::Ir::HeaderWords + 1;
        u64* const record = Reserve(words);
        u64* const span = arena.Begin(writes.size(), ++sequence, position);
        for (size_t i = 0; i < writes.size(); ++i) {
            span[i] = 2 | (u64{writes[i].first & 0xffff} << 8) | (u64{writes[i].second} << 32);
        }
        record[0] = V2::Ir::MakeHeader(V2::Ir::Kind::StateRun, 0, words, V2::Ir::HasPayload);
        record[1] = sequence;
        record[2] = V2::Ir::PackRef(arena.IndexOf(span), static_cast<u32>(writes.size()));
    }
    void Draw(const V2::Ir::DrawBody& body) {
        const size_t body_words = V2::Ir::BodyWordsOf<V2::Ir::DrawBody>;
        const size_t words = V2::Ir::HeaderWords + body_words;
        u64* const record = Reserve(words);
        record[0] = V2::Ir::MakeHeader(V2::Ir::Kind::Draw, body_words, words, 0);
        record[1] = ++sequence;
        std::memcpy(record + V2::Ir::HeaderWords, &body, sizeof(body));
    }
};

Resolve::ShardKey FakeShardKey(const Resolve::ShardKeyInputs& in) {
    Resolve::ShardKey key;
    key.shape = V2::Mix64(in.groups->Fingerprint(V2::AllGroups) ^ in.regs[0x200] ^ in.topology);
    key.partial = key.shape + 1;
    key.has_shape = key.has_key = true;
    return key;
}

} // Anonymous namespace

TEST_CASE("V2 P6: a shard replays its segment from the IR ring onto the snapshot",
          "[video_core][v2_p6]") {
    namespace RegSem = V2::RegSem;
    // A Maxwell3D builds the register semantics table from its execution mask.
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Engines::Maxwell3D engine{memory};
    const RegSem::Table3D& table = RegSem::GetTable3D();
    const auto no_effects = [](RegSem::Effect, u32, u32) {};
    // Plain state registers of the pipeline/viewport groups (stores with no side effect).
    std::vector<u32> plain;
    for (u32 index = 0x200; index < 0xd00 && plain.size() < 64; ++index) {
        if ((table.flags[index] & RegSem::Exec) == 0 &&
            table.groups[index] != V2::RegGroup::Misc &&
            table.groups[index] != V2::RegGroup::CbufData) {
            plain.push_back(index);
        }
    }
    REQUIRE(plain.size() == 64);
    auto serial = std::make_unique<RegSem::Mirror3D>();
    std::mt19937 rng{5};
    for (u32 i = 0; i < 2000; ++i) {
        serial->regs[plain[rng() % plain.size()]] = rng() % 4;
    }
    RegSem::Rehash3D(serial->v2_groups, serial->regs.data());
    // The snapshot at the cut.
    auto snapshot = std::make_shared<Resolve::FrontSnapshot>();
    Resolve::SnapshotBuilder<0xe00> builder;
    snapshot->regs = builder.Take(serial->regs.data());
    Resolve::SnapshotBuilder<0xe00> shadow_builder;
    snapshot->shadow = shadow_builder.Take(serial->shadow.data());
    snapshot->groups = serial->v2_groups;
    snapshot->engine_ref = 3;
    snapshot->engine = 0x1000;
    snapshot->memory = 0x2000;

    RingBuilder ring;
    ring.position = RingBuilder::RingWords - 40; // the segment crosses the ring's end
    const u64 begin = ring.position;
    std::vector<u64> expected_shapes;
    for (u32 draw = 0; draw < 12; ++draw) {
        std::vector<std::pair<u32, u32>> writes;
        for (u32 w = 0; w < 5; ++w) {
            writes.emplace_back(plain[rng() % plain.size()], rng() % 4);
        }
        ring.StateRun(writes);
        for (const auto& [method, value] : writes) {
            RegSem::Write3D(*serial, table, method, value, false, no_effects);
        }
        V2::Ir::DrawBody body{};
        body.op = V2::Ir::DrawOp::Predicted;
        body.engine = 3;
        body.flags = V2::Ir::DrawHasGroups;
        body.num_draws = 1;
        body.draws[0].kind = V2::Ir::DrawKind::Indexed;
        body.draws[0].topology = 4;
        body.fingerprint = serial->v2_groups.Fingerprint(V2::HashedGroups);
        if (draw == 7) {
            body.fingerprint ^= 1; // a record the replay disagrees with: drift
        }
        ring.Draw(body);
        expected_shapes.push_back(V2::Mix64(serial->v2_groups.Fingerprint(V2::AllGroups) ^
                                            serial->regs[0x200] ^ 4));
        // Another engine's records in between are not this segment's.
        V2::Ir::DrawBody other = body;
        other.engine = 9;
        ring.Draw(other);
    }
    Resolve::SegmentInput input;
    input.id = 1;
    input.begin = begin;
    input.end = ring.position;
    input.snapshot = snapshot.get();
    input.ring = ring.ring.data();
    input.arena = &ring.arena;

    const Resolve::ShardKeyFn saved = Resolve::g_shard_key;
    Resolve::g_shard_key = &FakeShardKey;
    Resolve::SplitTraits::Shard shard;
    std::vector<Resolve::PreDraw> records;
    Commit::RequestList requests;
    requests.Reset(1);
    const SerialReason reason = Resolve::SplitTraits::Resolve(shard, input, records, requests);
    Resolve::g_shard_key = saved;
    REQUIRE(reason == SerialReason::None);
    REQUIRE(records.size() == 12);
    for (u32 draw = 0; draw < 12; ++draw) {
        INFO("draw " << draw);
        // Ordinals count the other engine's draw after each one too (F1-35).
        CHECK(records[draw].ordinal == draw * 2);
        CHECK(records[draw].indexed);
        CHECK(records[draw].engine == 0x1000);
        if (draw == 7) {
            CHECK(records[draw].drift);
            CHECK(!records[draw].has_shape);
        } else {
            CHECK(!records[draw].drift);
            CHECK(records[draw].shape == expected_shapes[draw]);
        }
    }
    CHECK(requests.Empty());
}

TEST_CASE("V2 P6: an early-submitted segment's shard stops at the end the front sets (F1-X2)",
          "[video_core][v2_p6]") {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Engines::Maxwell3D engine{memory};
    auto snapshot = std::make_unique<Resolve::FrontSnapshot>();
    Resolve::SnapshotBuilder<0xe00> builder;
    const std::vector<u32> zero(0xe00, 0);
    snapshot->regs = builder.Take(zero.data());
    snapshot->engine_ref = 3;
    const Resolve::ShardKeyFn saved = Resolve::g_shard_key;
    Resolve::g_shard_key = &FakeShardKey;
    V2::Ir::DrawBody body{};
    body.op = V2::Ir::DrawOp::Predicted;
    body.engine = 3;
    body.num_draws = 1;
    body.draws[0].kind = V2::Ir::DrawKind::Array;
    V2::Ir::DrawBody cut = body;
    cut.engine = 9; // stands for the next segment's Segment record (not this engine's draw)
    constexpr u32 Draws = 16;
    u32 wrong = 0;
    for (u32 round = 0; round < 200; ++round) {
        RingBuilder ring;
        std::atomic<u64> published{0};
        Resolve::SegmentProgress progress;
        progress.Reset();
        Resolve::SegmentInput input;
        input.id = 1;
        input.begin = 0;
        input.snapshot = snapshot.get();
        input.ring = ring.ring.data();
        input.arena = &ring.arena;
        input.progress = &progress;
        input.published = &published;
        std::vector<Resolve::PreDraw> records;
        SerialReason reason = SerialReason::NoSnapshot;
        std::jthread shard{[&] {
            Resolve::SplitTraits::Shard state;
            Commit::RequestList requests;
            requests.Reset(1);
            reason = Resolve::SplitTraits::Resolve(state, input, records, requests);
        }};
        // The front: its draws, published one by one, the cut (end stored), then the next
        // segment's draws published as fast as the shard reads.
        for (u32 draw = 0; draw < Draws; ++draw) {
            ring.Draw(body);
            published.store(ring.position, std::memory_order_seq_cst);
        }
        ring.Draw(cut);
        progress.end.store(ring.position, std::memory_order_release);
        for (u32 draw = 0; draw < Draws; ++draw) {
            ring.Draw(body);
            published.store(ring.position, std::memory_order_seq_cst);
        }
        shard.join();
        if (reason != SerialReason::None || records.size() != Draws) {
            ++wrong;
        }
    }
    Resolve::g_shard_key = saved;
    CHECK(wrong == 0);
}

TEST_CASE("V2 P6: an abandoned early segment keeps the draws its shard resolved (F1-36)",
          "[video_core][v2_p6]") {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Engines::Maxwell3D engine{memory};
    auto snapshot = std::make_unique<Resolve::FrontSnapshot>();
    Resolve::SnapshotBuilder<0xe00> builder;
    const std::vector<u32> zero(0xe00, 0);
    snapshot->regs = builder.Take(zero.data());
    snapshot->engine_ref = 3;
    const Resolve::ShardKeyFn saved = Resolve::g_shard_key;
    Resolve::g_shard_key = &FakeShardKey;
    V2::Ir::DrawBody body{};
    body.op = V2::Ir::DrawOp::Predicted;
    body.engine = 3;
    body.num_draws = 1;
    body.draws[0].kind = V2::Ir::DrawKind::Array;
    for (const u32 published_draws : {0U, 5U}) {
        RingBuilder ring;
        std::atomic<u64> published{0};
        Resolve::SegmentProgress progress;
        progress.Reset();
        for (u32 draw = 0; draw < published_draws; ++draw) {
            ring.Draw(body);
        }
        published.store(ring.position);
        Resolve::SegmentInput input;
        input.id = 1;
        input.snapshot = snapshot.get();
        input.ring = ring.ring.data();
        input.arena = &ring.arena;
        input.progress = &progress;
        input.published = &published;
        std::vector<Resolve::PreDraw> records;
        SerialReason reason = SerialReason::None;
        std::jthread shard{[&] {
            Resolve::SplitTraits::Shard state;
            Commit::RequestList requests;
            requests.Reset(1);
            reason = Resolve::SplitTraits::Resolve(state, input, records, requests);
        }};
        // The commit reaches the segment while the shard waits for more of it.
        while (!progress.waiting.load()) {
            std::this_thread::yield();
        }
        progress.abandon.store(true);
        shard.join();
        INFO("published draws " << published_draws);
        CHECK(records.size() == published_draws);
        CHECK(reason == (published_draws == 0 ? SerialReason::Stolen : SerialReason::None));
    }
    Resolve::g_shard_key = saved;
}

// ---- R3: the pipeline lookup on the shards -------------------------------------------------------

TEST_CASE("V2 P6 R3: thread epoch sections register once and nest", "[video_core][v2_p6]") {
    // Sections run on threads that end before the domain does (a thread keeps its slot).
    V2::EpochDomain domain;
    bool freed = false;
    bool valid = false;
    bool held = false;
    std::thread{[&] {
        {
            const V2::ThreadEpochGuard outer{domain};
            valid = outer.Valid();
            {
                const V2::ThreadEpochGuard inner{domain};
                valid = valid && inner.Valid();
            }
            // The inner section's end does not end the outer one: what it may hold stays.
            domain.Retire(&freed, [](void* pointer) { *static_cast<bool*>(pointer) = true; });
            held = !freed;
        }
        domain.Reclaim();
    }}.join();
    CHECK(valid);
    CHECK(held);
    CHECK(freed);
    for (u32 i = 0; i < V2::EpochDomain::MaxReaders * 2; ++i) {
        std::thread{[&] { const V2::ThreadEpochGuard guard{domain}; }}.join();
    }
    std::thread{[&] {
        const V2::ThreadEpochGuard guard{domain};
        CHECK(guard.Valid()); // slots came back
    }}.join();
}

TEST_CASE("V2 P6 R3: shared lookups find inserted pipelines and refuse cleared ones",
          "[video_core][v2_p6]") {
    using Lookup = V2::PipelineLookup<int>;
    V2::EpochDomain domain;
    Lookup lookup;
    lookup.SetDomain(&domain);
    const Lookup::Hashes hashes{1, 2, 3, 0, 0, 5};
    std::array<u8, 24> key{};
    key[3] = 7;
    int value = 42;
    const auto shared = [&](const Lookup::Hashes& set, const std::array<u8, 24>& bytes) {
        int* found = nullptr;
        std::thread{[&] {
            const V2::ThreadEpochGuard guard{domain};
            found = lookup.FindShared(set, 9, bytes.data(), bytes.size());
        }}.join();
        return found;
    };
    CHECK(shared(hashes, key) == nullptr);
    const u64 serial = lookup.Serial();
    lookup.Insert(hashes, 9, key.data(), key.size(), &value);
    CHECK(lookup.Serial() != serial);
    // The table keeps its own copy of the key: the caller's bytes may go away.
    std::array<u8, 24> copy = key;
    key.fill(0xff);
    const u64 generation = lookup.Generation();
    CHECK(shared(hashes, copy) == &value);
    CHECK(shared(hashes, key) == nullptr);
    CHECK(shared(Lookup::Hashes{1, 2, 3, 0, 0, 6}, copy) == nullptr);
    // A clear while a reader is inside: the generation moves (the commit refuses what was found
    // before), the sets stay readable until the reader leaves.
    std::atomic<int> phase{0};
    int* held = nullptr;
    std::thread reader{[&] {
        const V2::ThreadEpochGuard guard{domain};
        held = lookup.FindShared(hashes, 9, copy.data(), copy.size());
        phase.store(1);
        while (phase.load() != 2) {
            std::this_thread::yield();
        }
    }};
    while (phase.load() != 1) {
        std::this_thread::yield();
    }
    lookup.Clear();
    CHECK(lookup.Generation() != generation);
    CHECK(domain.PendingRetired() == 1); // the reader is inside: not freed
    phase.store(2);
    reader.join();
    CHECK(held == &value);
    CHECK(shared(hashes, copy) == nullptr);
    domain.Reclaim();
    CHECK(domain.PendingRetired() == 0);
}

TEST_CASE("V2 P6 R3: stress: shard lookups during inserts and clears", "[video_core][v2_p6]") {
    using Lookup = V2::PipelineLookup<u64>;
    V2::EpochDomain domain;
    Lookup lookup;
    lookup.SetDomain(&domain);
    constexpr u32 Sets = 16;
    constexpr u32 Variants = 24;
    // Values encode their (set, variant): a reader checks what it found is for what it asked.
    std::vector<u64> values(Sets * Variants);
    for (u32 i = 0; i < values.size(); ++i) {
        values[i] = i;
    }
    const auto hashes_of = [](u32 set) { return Lookup::Hashes{set + 1, 0, 0, 0, 0, 7}; };
    const auto key_of = [](u32 variant) {
        std::array<u8, 32> key{};
        key[0] = static_cast<u8>(variant);
        key[31] = static_cast<u8>(variant * 3);
        return key;
    };
    std::atomic<bool> stop{false};
    std::atomic<u64> wrong{0};
    std::atomic<u64> hits{0};
    std::vector<std::jthread> readers;
    for (u32 thread = 0; thread < 3; ++thread) {
        readers.emplace_back([&, thread] {
            std::mt19937 rng{thread + 11};
            while (!stop.load(std::memory_order_relaxed)) {
                const u32 set = rng() % Sets;
                const u32 variant = rng() % Variants;
                const auto key = key_of(variant);
                const V2::ThreadEpochGuard guard{domain};
                const u64* const found =
                    lookup.FindShared(hashes_of(set), variant * 0x9e37ULL, key.data(), key.size());
                if (found != nullptr) {
                    hits.fetch_add(1, std::memory_order_relaxed);
                    if (*found != set * Variants + variant) {
                        wrong.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }
    std::mt19937 rng{3};
    for (u32 round = 0; round < 200; ++round) {
        for (u32 i = 0; i < 64; ++i) {
            const u32 set = rng() % Sets;
            const u32 variant = rng() % Variants;
            const auto key = key_of(variant);
            if (lookup.Find(hashes_of(set), variant * 0x9e37ULL, key.data(), key.size()) ==
                nullptr) {
                lookup.Insert(hashes_of(set), variant * 0x9e37ULL, key.data(), key.size(),
                              &values[set * Variants + variant]);
            }
        }
        if (round % 25 == 24) {
            lookup.Clear();
        }
    }
    stop.store(true);
    readers.clear();
    CHECK(wrong.load() == 0);
    CHECK(hits.load() != 0);
}

TEST_CASE("V2 P6 R3: published programs answer by memory and address", "[video_core][v2_p6]") {
    Resolve::ClearPrograms();
    u64 hash = 0;
    CHECK(!Resolve::FindProgram(0x1000, 0x8000, &hash));
    const u64 generation = Resolve::ProgramGeneration();
    Resolve::PublishProgram(0x1000, 0x8000, 0xabc);
    CHECK(Resolve::ProgramGeneration() != generation);
    REQUIRE(Resolve::FindProgram(0x1000, 0x8000, &hash));
    CHECK(hash == 0xabc);
    CHECK(!Resolve::FindProgram(0x2000, 0x8000, &hash)); // another channel's memory
    CHECK(!Resolve::FindProgram(0x1000, 0x8100, &hash));
    // Publishing what is there already does not move the generation.
    const u64 same = Resolve::ProgramGeneration();
    Resolve::PublishProgram(0x1000, 0x8000, 0xabc);
    CHECK(Resolve::ProgramGeneration() == same);
    Resolve::PublishProgram(0x1000, 0x8000, 0xdef);
    REQUIRE(Resolve::FindProgram(0x1000, 0x8000, &hash));
    CHECK(hash == 0xdef);
    Resolve::ClearPrograms();
    CHECK(!Resolve::FindProgram(0x1000, 0x8000, &hash));
}

namespace {

std::shared_ptr<const Resolve::PipelineKey> FakePipelineKey(const Resolve::SnapshotView& view) {
    auto key = std::make_shared<Resolve::PipelineKey>();
    key->hash = view.Regs().reg_array[0x200] + 1;
    key->bytes.assign(8, static_cast<u8>(key->hash));
    return key;
}

/// A shard lookup that names a pipeline by its key (as the real lookup does, keyed by state).
void FakeShardPipeline(const Resolve::SnapshotView&, uintptr_t memory, Resolve::PreDraw& pre) {
    pre.hashes = {memory, 0, 0, 0, 0, 1};
    pre.pipeline = reinterpret_cast<const void*>(static_cast<uintptr_t>(pre.pipeline_key->hash << 4));
    pre.has_pipeline = true;
    pre.lookup_generation = 1;
    pre.lookup_serial = 5;
}

} // Anonymous namespace

TEST_CASE("V2 P6 R3: shards look up each draw's pipeline; verify compares the objects",
          "[video_core][v2_p6]") {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Engines::Maxwell3D engine{memory};
    auto snapshot = std::make_unique<Resolve::FrontSnapshot>();
    Resolve::SnapshotBuilder<0xe00> builder;
    std::vector<u32> regs(0xe00, 0);
    regs[0x200] = 0x30;
    snapshot->regs = builder.Take(regs.data());
    snapshot->engine_ref = 3;
    snapshot->memory = 0x7000;
    RingBuilder ring;
    V2::Ir::DrawBody body{};
    body.op = V2::Ir::DrawOp::Predicted;
    body.engine = 3;
    body.flags = V2::Ir::DrawHasGroups;
    body.num_draws = 1;
    body.draws[0].kind = V2::Ir::DrawKind::Indexed;
    body.fingerprint = V2::GroupTracker{}.Fingerprint(V2::HashedGroups);
    ring.Draw(body);
    body.fingerprint ^= 1; // drift: no key, no lookup
    ring.Draw(body);
    Resolve::SegmentInput input;
    input.id = 1;
    input.begin = 0;
    input.end = ring.position;
    input.snapshot = snapshot.get();
    input.ring = ring.ring.data();
    input.arena = &ring.arena;
    const auto saved_key = Resolve::g_shard_key;
    const auto saved_pipeline_key = Resolve::g_pipeline_key;
    const auto saved_pipeline = Resolve::g_shard_pipeline;
    Resolve::g_shard_key = &FakeShardKey;
    Resolve::g_pipeline_key = &FakePipelineKey;
    Resolve::g_shard_pipeline = &FakeShardPipeline;
    Resolve::SplitTraits::Shard shard;
    std::vector<Resolve::PreDraw> records;
    Commit::RequestList requests;
    requests.Reset(1);
    const SerialReason reason = Resolve::SplitTraits::Resolve(shard, input, records, requests);
    Resolve::g_shard_key = saved_key;
    Resolve::g_pipeline_key = saved_pipeline_key;
    Resolve::g_shard_pipeline = saved_pipeline;
    REQUIRE(reason == SerialReason::None);
    REQUIRE(records.size() == 2);
    CHECK(records[0].has_pipeline);
    CHECK(records[0].pipeline == reinterpret_cast<const void*>(uintptr_t{0x31 << 4}));
    CHECK(records[0].hashes[0] == 0x7000);
    CHECK(!records[1].has_pipeline);
    // Verify: another object for the same tables is a mismatch; after an insert or a program
    // publish in between it is dependent.
    Resolve::PreDraw serial = records[0];
    CHECK(Resolve::SplitTraits::Compare(records[0], serial) == nullptr);
    serial.pipeline = nullptr;
    serial.has_pipeline = false;
    CHECK(Resolve::SplitTraits::Compare(records[0], serial) != nullptr);
    serial.lookup_serial = 6;
    CHECK(Resolve::SplitTraits::Compare(records[0], serial) == nullptr);
    serial = records[0];
    serial.hashes[5] = 2;
    CHECK(Resolve::SplitTraits::Compare(records[0], serial) != nullptr);
    serial.program_generation += 1;
    CHECK(Resolve::SplitTraits::Compare(records[0], serial) == nullptr);
}

TEST_CASE("V2 P6 R5: a shard's dynamic state equals ComputeDynamic on the same view",
          "[video_core][v2_p6]") {
    using Tegra::Engines::Maxwell3D;
    // A Maxwell3D builds the register semantics table.
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Maxwell3D engine{memory};
    auto& mode = V2::g_modes[static_cast<size_t>(V2::Switch::DrawRec)];
    const V2::Mode saved = mode;
    mode = V2::Mode::Off;
    std::vector<u32> regs(Maxwell3D::Regs::NUM_REGS, 0);
    std::mt19937 rng{17};
    const V2::RegSem::Table3D& table = V2::RegSem::GetTable3D();
    for (u32 i = 0; i < 4000; ++i) {
        const u32 index = 0x200 + rng() % 0xb00;
        if (table.groups[index] != V2::RegGroup::Misc) {
            regs[index] = rng() % 4; // small: register fields stay valid enum and index values
        }
    }
    const Resolve::CbufTable cbufs{};
    V2::GroupTracker groups;
    V2::RegSem::Rehash3D(groups, regs.data());
    const Resolve::SnapshotView view{regs.data(),
                                     &cbufs,
                                     &groups,
                                     Maxwell3D::Regs::PrimitiveTopology::Triangles,
                                     Resolve::ViewDraw{.count = 3},
                                     false};
    u64 generation = 0;
    Resolve::DynamicFeatures features{};
    features.eds1 = features.eds2 = features.eds3_enables = true;
    features.depth_bounds = true;
    // No features published yet, or DRAWREC off: nothing (the commit computes).
    CHECK(Resolve::ShardDynamic(view, generation) == nullptr);
    const u64 first = Resolve::PublishDynamicFeatures(features);
    CHECK(Resolve::PublishDynamicFeatures(features) == first); // unchanged: same generation
    CHECK(Resolve::ShardDynamic(view, generation) == nullptr);
    mode = V2::Mode::On;
    const auto dynamic = Resolve::ShardDynamic(view, generation);
    REQUIRE(dynamic != nullptr);
    CHECK(generation == first);
    const Resolve::DynamicState serial = Resolve::ComputeDynamic(view, features);
    CHECK(std::memcmp(dynamic.get(), &serial, sizeof(serial)) == 0);
    // Unchanged groups, topology and features: the same result, not computed again.
    CHECK(Resolve::ShardDynamic(view, generation) == dynamic);
    // New features: a new generation and a new result.
    features.eds3_blending = true;
    const u64 second = Resolve::PublishDynamicFeatures(features);
    CHECK(second != first);
    const auto again = Resolve::ShardDynamic(view, generation);
    REQUIRE(again != nullptr);
    CHECK(again != dynamic);
    CHECK(generation == second);
    const Resolve::DynamicState serial2 = Resolve::ComputeDynamic(view, features);
    CHECK(std::memcmp(again.get(), &serial2, sizeof(serial2)) == 0);
    // Verify: another dynamic state at the same generation is a mismatch, else dependent.
    Resolve::PreDraw a;
    a.dynamic = dynamic;
    a.dynamic_generation = first;
    Resolve::PreDraw b = a;
    CHECK(Resolve::SplitTraits::Compare(a, b) == nullptr);
    auto changed = std::make_shared<Resolve::DynamicState>(*dynamic);
    changed->values.line_width += 1.0f;
    b.dynamic = changed;
    CHECK(Resolve::SplitTraits::Compare(a, b) != nullptr);
    b.dynamic_generation = second;
    CHECK(Resolve::SplitTraits::Compare(a, b) == nullptr);
    mode = saved;
}

TEST_CASE("V2 P6: published shapes are found by the shards", "[video_core][v2_p6]") {
    Resolve::ClearShapes();
    u32 relevant = 0;
    CHECK(!Resolve::FindShape(0x1234, &relevant));
    Resolve::PublishShape(0x1234, 0x44);
    REQUIRE(Resolve::FindShape(0x1234, &relevant));
    CHECK(relevant == 0x44);
    CHECK(!Resolve::FindShape(0x9999, &relevant));
    Resolve::ClearShapes();
    CHECK(!Resolve::FindShape(0x1234, &relevant));
}

// ---- P6c: the state view -------------------------------------------------------------------------

using Tegra::Engines::Maxwell3D;

namespace {

/// The writes a resolve-relevant register stream is made of: program offsets and enables, the
/// program region, constant buffer address/size, constant buffer binds and plain pipeline state.
struct ViewStream {
    std::vector<u32> plain;
    std::vector<u32> programs;
    std::array<u32, 5> binds{};
    u32 cbuf = 0;

    explicit ViewStream(const V2::RegSem::Table3D& table) {
        using Regs = Tegra::Engines::Maxwell3D::Regs;
        for (u32 index = 0x200; index < 0xd00 && plain.size() < 48; ++index) {
            if ((table.flags[index] & V2::RegSem::Exec) == 0 &&
                table.groups[index] == V2::RegGroup::Pipeline) {
                plain.push_back(index);
            }
        }
        for (u32 program = 0; program < Regs::MaxShaderProgram; ++program) {
            programs.push_back(static_cast<u32>(MAXWELL3D_REG_INDEX(pipelines)) +
                               program * static_cast<u32>(sizeof(Regs::Pipeline) / 4));
        }
        for (u32 stage = 0; stage < 5; ++stage) {
            binds[stage] = static_cast<u32>(MAXWELL3D_REG_INDEX(bind_groups)) +
                           stage * static_cast<u32>(sizeof(Regs::BindGroup) / 4) + 4;
        }
        cbuf = static_cast<u32>(MAXWELL3D_REG_INDEX(const_buffer));
    }

    std::pair<u32, u32> Next(std::mt19937& rng) const {
        switch (rng() % 6) {
        case 0:
            return {plain[rng() % plain.size()], rng() % 4};
        case 1: // a program's enable word (bit 0) or its offset
            return {programs[rng() % programs.size()] + (rng() % 2), rng() % 2 == 0 ? 1U : rng()};
        case 2:
            return {static_cast<u32>(MAXWELL3D_REG_INDEX(program_region)) + (rng() % 2),
                    rng() % 16};
        case 3:
            return {cbuf + rng() % 3, rng() % 0x10000};
        default: // a bind: slot 0-23 (18 and above bind nothing), valid or not
            return {binds[rng() % binds.size()], ((rng() % 24) << 4) | (rng() % 4 != 0 ? 1U : 0U)};
        }
    }
};

/// The live engine as the back's methods leave it: the registers, and ProcessCBBind's rule.
void EngineWrite(Tegra::Engines::Maxwell3D& engine, const V2::RegSem::Table3D& table, u32 method,
                 u32 value) {
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    engine.regs.reg_array[method] = value;
    if ((table.flags[method] & V2::RegSem::CbufBindFlag) == 0) {
        return;
    }
    const u32 stage = (method - static_cast<u32>(MAXWELL3D_REG_INDEX(bind_groups))) /
                      static_cast<u32>(sizeof(Regs::BindGroup) / 4);
    const auto& bind = engine.regs.bind_groups[stage];
    if (bind.shader_slot >= Regs::MaxConstBuffers) {
        return;
    }
    auto& buffer = engine.state.shader_stages[stage].const_buffers[bind.shader_slot];
    buffer.enabled = bind.valid.Value() != 0;
    buffer.address = engine.regs.const_buffer.Address();
    buffer.size = engine.regs.const_buffer.size;
}

} // Anonymous namespace

TEST_CASE("V2 P6c: a snapshot view after replay equals the live engine view",
          "[video_core][v2_p6c]") {
    namespace RegSem = V2::RegSem;
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::Engines::Maxwell3D engine{memory};
    const RegSem::Table3D& table = RegSem::GetTable3D();
    const auto no_effects = [](RegSem::Effect, u32, u32) {};
    const ViewStream stream{table};

    // The front mirror starts from the engine (FrontRegSemResync).
    auto front = std::make_unique<RegSem::Mirror3D>();
    std::memcpy(front->regs.data(), engine.regs.reg_array.data(), sizeof(front->regs));
    front->v2_cbufs = Resolve::CbufTableOf(Resolve::LiveView{engine});
    RegSem::Rehash3D(front->v2_groups, front->regs.data());

    struct Expected {
        std::vector<u32> regs;
        Resolve::CbufTable cbufs;
        std::array<GPUVAddr, 6> programs{};
        u64 fingerprint = 0;
        u32 topology = 0;
    };
    struct Segment {
        Resolve::RegisterSnapshot<0xe00> regs;
        std::shared_ptr<const Resolve::CbufTable> cbufs;
        V2::GroupTracker groups;
        std::vector<std::vector<std::pair<u32, u32>>> draws; ///< deltas before each draw
        std::vector<Expected> expected;
    };
    std::vector<Segment> segments;
    Resolve::SnapshotBuilder<0xe00> builder;
    std::shared_ptr<const Resolve::CbufTable> last_cbufs;
    std::mt19937 rng{61};
    u64 binds_changed = 0;
    for (u32 cut = 0; cut < 24; ++cut) {
        Segment& segment = segments.emplace_back();
        segment.regs = builder.Take(front->regs.data());
        if (last_cbufs == nullptr || !(*last_cbufs == front->v2_cbufs)) {
            last_cbufs = std::make_shared<const Resolve::CbufTable>(front->v2_cbufs);
        }
        segment.cbufs = last_cbufs;
        segment.groups = front->v2_groups;
        const u32 draws = 1 + rng() % 24;
        for (u32 draw = 0; draw < draws; ++draw) {
            auto& writes = segment.draws.emplace_back();
            for (u32 w = rng() % 9; w > 0; --w) {
                writes.push_back(stream.Next(rng));
            }
            const Resolve::CbufTable before = front->v2_cbufs;
            for (const auto& [method, value] : writes) {
                RegSem::Write3D(*front, table, method, value, false, no_effects);
                EngineWrite(engine, table, method, value);
            }
            binds_changed += before == front->v2_cbufs ? 0 : 1;
            Expected& expected = segment.expected.emplace_back();
            expected.topology = rng() % 10;
            engine.draw_manager.draw_state.topology =
                static_cast<Resolve::MaxwellRegs::PrimitiveTopology>(expected.topology);
            const Resolve::LiveView live{engine};
            expected.regs.assign(engine.regs.reg_array.begin(), engine.regs.reg_array.end());
            expected.cbufs = Resolve::CbufTableOf(live);
            for (size_t program = 0; program < 6; ++program) {
                expected.programs[program] = Resolve::ProgramAddress(live, program);
            }
            V2::GroupTracker groups;
            RegSem::Rehash3D(groups, engine.regs.reg_array.data());
            expected.fingerprint = groups.Fingerprint(V2::HashedGroups);
        }
    }
    REQUIRE(binds_changed > 20);

    // Shards replay every segment afterwards: the snapshots must still hold their time's state.
    u64 programs_seen = 0;
    for (size_t index = 0; index < segments.size(); ++index) {
        const Segment& segment = segments[index];
        auto shard = std::make_unique<RegSem::Mirror3D>();
        segment.regs.CopyTo(shard->regs.data());
        shard->v2_cbufs = *segment.cbufs;
        shard->v2_groups = segment.groups;
        for (size_t draw = 0; draw < segment.draws.size(); ++draw) {
            INFO("segment " << index << " draw " << draw);
            for (const auto& [method, value] : segment.draws[draw]) {
                RegSem::Write3D(*shard, table, method, value, false, no_effects);
            }
            const Expected& expected = segment.expected[draw];
            const Resolve::SnapshotView view{
                shard->regs.data(),
                &shard->v2_cbufs,
                &shard->v2_groups,
                static_cast<Resolve::MaxwellRegs::PrimitiveTopology>(expected.topology),
                Resolve::ViewDraw{},
                false};
            REQUIRE(std::memcmp(&view.Regs(), expected.regs.data(),
                                sizeof(Resolve::MaxwellRegs)) == 0);
            REQUIRE(Resolve::CbufTableOf(view) == expected.cbufs);
            CHECK(static_cast<u32>(view.Topology()) == expected.topology);
            CHECK(view.Groups().Fingerprint(V2::HashedGroups) == expected.fingerprint);
            for (size_t program = 0; program < 6; ++program) {
                CHECK(Resolve::ProgramAddress(view, program) == expected.programs[program]);
                programs_seen += expected.programs[program] != 0 ? 1 : 0;
            }
        }
    }
    CHECK(programs_seen > 0);
}

namespace {

Vulkan::DynamicFeatures g_test_features{};
constexpr std::array<u32, 5> TestKeyCbufs{0x3, 0x0, 0x10, 0x0, 0x6};
constexpr u64 TestShape = 0x5eed5eed;

/// The shard side of step (a) as the Vulkan glue does it, from the shard's state.
Resolve::ShardKey TestShardKey(const Resolve::ShardKeyInputs& in) {
    Resolve::ShardKey key;
    key.shape = TestShape;
    key.has_shape = true;
    u32 relevant = 0;
    std::array<u32, 5> masks{};
    if (Resolve::FindShape(key.shape, &relevant, &masks)) {
        key.partial = in.groups->Fingerprint(V2::HashedGroups);
        key.relevant = relevant;
        key.has_key = true;
        const Resolve::SnapshotView view{in.regs,
                                         in.cbufs,
                                         in.groups,
                                         static_cast<Resolve::MaxwellRegs::PrimitiveTopology>(
                                             in.topology),
                                         Resolve::ViewDraw{},
                                         in.native_turing};
        key.full = Resolve::KeyCbufs(key.partial, masks, view);
        key.cbuf_masks = Resolve::CbufMasksHash(masks);
        key.has_full = true;
    }
    return key;
}

std::shared_ptr<const Resolve::PipelineKey> TestPipelineKey(const Resolve::SnapshotView& view) {
    Vulkan::FixedPipelineState state;
    std::memset(&state, 0, sizeof(state));
    state.RefreshFull(view.Regs(),
                      Vulkan::KeyInputs{.topology = view.Topology(),
                                        .engine_state = view.EngineState(),
                                        .squashed_iterated_blend = false},
                      g_test_features);
    auto key = std::make_shared<Resolve::PipelineKey>();
    key->hash = state.Hash();
    key->bytes.resize(sizeof(state));
    std::memcpy(key->bytes.data(), &state, sizeof(state));
    return key;
}

} // Anonymous namespace

TEST_CASE("V2 P6c: shards resolve the draw key and the pipeline key as the serial path does",
          "[video_core][v2_p6c]") {
    namespace RegSem = V2::RegSem;
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    const RegSem::Table3D& table = [&]() -> const RegSem::Table3D& {
        Tegra::Engines::Maxwell3D first{memory}; // builds the register semantics table
        return RegSem::GetTable3D();
    }();
    const auto no_effects = [](RegSem::Effect, u32, u32) {};
    const ViewStream stream{table};
    // Every register a pipeline key block reads (non-executable ones: plain stores).
    std::vector<u32> key_registers;
    for (size_t block = 0; block < Vulkan::NumKeyBlocks; ++block) {
        for (const auto& [first, count] :
             Vulkan::KeyBlockRegisters(static_cast<Vulkan::KeyBlock>(block))) {
            for (u32 index = first; index < first + count && index < 0xe00; ++index) {
                if ((table.flags[index] & RegSem::Exec) == 0) {
                    key_registers.push_back(index);
                }
            }
        }
    }
    REQUIRE(key_registers.size() > 100);

    Vulkan::DynamicFeatures none{};
    Vulkan::DynamicFeatures eds = none;
    eds.has_extended_dynamic_state = true;
    eds.has_extended_dynamic_state_2 = true;
    eds.has_extended_dynamic_state_2_logic_op = true;
    eds.has_extended_dynamic_state_3_blend = true;
    eds.has_extended_dynamic_state_3_enables = true;
    eds.has_dynamic_state3_alpha_to_coverage = true;
    eds.has_provoking_vertex = true;
    eds.has_provoking_vertex_last_mode = true;
    Vulkan::DynamicFeatures vertex = none;
    vertex.has_dynamic_vertex_input = true;
    vertex.has_extended_dynamic_state = true;
    Vulkan::DynamicFeatures repack = none;
    repack.has_static_vertex_repack = true;
    const std::array feature_sets{none, eds, vertex, repack};

    const Resolve::ShardKeyFn saved_key = Resolve::g_shard_key;
    const Resolve::PipelineKeyFn saved_pipeline = Resolve::g_pipeline_key;
    Resolve::g_shard_key = &TestShardKey;
    Resolve::g_pipeline_key = &TestPipelineKey;
    Resolve::ClearShapes();
    Resolve::PublishShape(TestShape, V2::AllGroups, TestKeyCbufs);

    u64 draws_checked = 0;
    u64 keys_changed = 0;
    for (size_t set = 0; set < feature_sets.size(); ++set) {
        INFO("feature set " << set);
        g_test_features = feature_sets[set];
        Tegra::Engines::Maxwell3D engine{memory};
        auto front = std::make_unique<RegSem::Mirror3D>();
        std::memcpy(front->regs.data(), engine.regs.reg_array.data(), sizeof(front->regs));
        front->v2_cbufs = Resolve::CbufTableOf(Resolve::LiveView{engine});
        RegSem::Rehash3D(front->v2_groups, front->regs.data());
        RegSem::Rehash3D(engine.v2_groups, engine.regs.reg_array.data());

        // The serial path: the incremental key on the live engine, driven by its group sink.
        Vulkan::FixedPipelineState serial_key;
        std::memset(&serial_key, 0, sizeof(serial_key));
        Vulkan::KeyInputs previous{};
        bool first_key = true;
        std::mt19937 rng{static_cast<u32>(97 + set)};
        u64 last_hash = 0;

        for (u32 segment_index = 0; segment_index < 4; ++segment_index) {
            auto snapshot = std::make_shared<Resolve::FrontSnapshot>();
            Resolve::SnapshotBuilder<0xe00> builder;
            Resolve::SnapshotBuilder<0xe00> shadow_builder;
            snapshot->regs = builder.Take(front->regs.data());
            snapshot->shadow = shadow_builder.Take(front->shadow.data());
            snapshot->groups = front->v2_groups;
            const Resolve::CbufTable snapshot_cbufs = front->v2_cbufs;
            snapshot->cbufs = &snapshot_cbufs;
            snapshot->engine_ref = 3;
            snapshot->engine = 0x1000;
            snapshot->memory = 0x2000;

            RingBuilder ring;
            ring.position = RingBuilder::RingWords - 64 * (segment_index + 1);
            const u64 begin = ring.position;
            struct Serial {
                u64 full;
                std::vector<u8> pipeline;
                u64 hash;
            };
            std::vector<Serial> expected;
            const u32 draws = 8 + rng() % 24;
            for (u32 draw = 0; draw < draws; ++draw) {
                std::vector<std::pair<u32, u32>> writes;
                for (u32 w = rng() % 8; w > 0; --w) {
                    if (rng() % 3 == 0) {
                        writes.push_back(stream.Next(rng));
                    } else {
                        const u32 index = key_registers[rng() % key_registers.size()];
                        writes.emplace_back(index, rng() % 2 == 0 ? rng() % 4 : rng());
                    }
                }
                if (!writes.empty()) {
                    ring.StateRun(writes);
                }
                for (const auto& [method, value] : writes) {
                    RegSem::Write3D(*front, table, method, value, false, no_effects);
                    RegSem::NoteWrite3D(engine.v2_groups, method, engine.regs.reg_array[method],
                                        value);
                    EngineWrite(engine, table, method, value);
                }
                const u8 topology = static_cast<u8>(rng() % 10);
                V2::Ir::DrawBody body{};
                body.op = V2::Ir::DrawOp::Predicted;
                body.engine = 3;
                body.flags = V2::Ir::DrawHasGroups;
                body.num_draws = 1;
                body.draws[0].kind = V2::Ir::DrawKind::Indexed;
                body.draws[0].topology = topology;
                body.fingerprint = front->v2_groups.Fingerprint(V2::HashedGroups);
                ring.Draw(body);

                // Serial resolve of the same draw on the live engine.
                engine.draw_manager.draw_state.topology =
                    static_cast<Resolve::MaxwellRegs::PrimitiveTopology>(topology);
                const Resolve::LiveView live{engine};
                Serial& serial = expected.emplace_back();
                serial.full = Resolve::KeyCbufs(engine.v2_groups.Fingerprint(V2::HashedGroups),
                                                TestKeyCbufs, live);
                engine.dirty.flags[Vulkan::Dirty::VertexInput] = true;
                engine.dirty.flags[Vulkan::Dirty::ViewportSwizzles] = true;
                const u32 groups = engine.v2_groups.TakeSink(V2::GroupSink::PipelineKey);
                const u32 repacked = serial_key.RefreshIncremental(
                    engine, g_test_features, groups, previous, first_key, true);
                if (repacked != 0) {
                    serial_key.CanonicalizeDynamic(g_test_features);
                }
                previous = Vulkan::FixedPipelineState::CurrentInputs(engine);
                first_key = false;
                serial.hash = serial_key.Hash();
                serial.pipeline.assign(reinterpret_cast<const u8*>(&serial_key),
                                       reinterpret_cast<const u8*>(&serial_key) +
                                           serial_key.Size());
                keys_changed += serial.hash != last_hash ? 1 : 0;
                last_hash = serial.hash;
            }

            Resolve::SegmentInput input;
            input.id = segment_index + 1;
            input.begin = begin;
            input.end = ring.position;
            input.snapshot = snapshot.get();
            input.ring = ring.ring.data();
            input.arena = &ring.arena;
            Resolve::SplitTraits::Shard shard;
            std::vector<Resolve::PreDraw> records;
            Commit::RequestList requests;
            requests.Reset(input.id);
            REQUIRE(Resolve::SplitTraits::Resolve(shard, input, records, requests) ==
                    SerialReason::None);
            REQUIRE(records.size() == expected.size());
            for (size_t draw = 0; draw < records.size(); ++draw) {
                INFO("segment " << segment_index << " draw " << draw);
                const Resolve::PreDraw& pre = records[draw];
                REQUIRE(!pre.drift);
                REQUIRE(pre.has_full);
                CHECK(pre.full == expected[draw].full);
                CHECK(pre.cbuf_masks == Resolve::CbufMasksHash(TestKeyCbufs));
                REQUIRE(pre.pipeline_key != nullptr);
                CHECK(pre.pipeline_key->hash == expected[draw].hash);
                const std::vector<u8>& serial = expected[draw].pipeline;
                REQUIRE(pre.pipeline_key->bytes.size() >= serial.size());
                size_t offset = 0;
                while (offset < serial.size() && serial[offset] == pre.pipeline_key->bytes[offset]) {
                    ++offset;
                }
                CHECK(offset == serial.size());
                ++draws_checked;
            }
        }
    }
    Resolve::g_shard_key = saved_key;
    Resolve::g_pipeline_key = saved_pipeline;
    Resolve::ClearShapes();
    CHECK(draws_checked > 100);
    CHECK(keys_changed > 40);
}
