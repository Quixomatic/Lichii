// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
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

#include "video_core/v2/commit.h"
#include "video_core/v2/resolve/cow_regs.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/v2/resolve/parallel_resolve.h"
#include "video_core/v2/resolve/segment.h"
#include "video_core/v2/resolve/split_resolve.h"
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
        // A render target change cuts (a render pass change) ...
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
    segmenter.OnRecord(RecordClass::Neutral);
    CHECK(segmenter.OpenReason() == SerialReason::None);
    segmenter.OnRecord(RecordClass::Transfer);
    segmenter.OnRecord(RecordClass::Map);
    CHECK(segmenter.OpenReason() == SerialReason::Transfer); // the first one
    // The next draw starts a clean segment.
    CHECK(segmenter.OnDraw(Draw()) == CutReason::Isolate);
    CHECK(segmenter.OpenReason() == SerialReason::None);

    Resolve::DrawEvent indirect = Draw();
    indirect.indirect = true;
    CHECK(segmenter.OnDraw(indirect) == CutReason::None);
    CHECK(segmenter.OpenReason() == SerialReason::IndirectCount);
    Resolve::DrawEvent inline_index = Draw();
    inline_index.inline_index = true;
    CHECK(segmenter.OnDraw(inline_index) == CutReason::Isolate);
    CHECK(segmenter.OpenReason() == SerialReason::InlineIndex);
    segmenter.OnDraw(Draw());
    segmenter.OnRecord(RecordClass::Readback);
    CHECK(segmenter.OpenReason() == SerialReason::Readback);
    segmenter.OnDraw(Draw());
    segmenter.OnRecord(RecordClass::CacheInvalidate);
    CHECK(segmenter.OpenReason() == SerialReason::CacheInvalidate);
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
    input.snapshot = snapshot;
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
        CHECK(records[draw].ordinal == draw);
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
