// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <exception>

#include "shader_recompiler/shader_info.h"
#include "video_core/gpu_split.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/v2/front_wake.h"
#include "video_core/v2/front_writes.h"
#include "video_core/v2/hazard.h"
#include "video_core/v2/ir_exec.h"
#include "video_core/v2/landing.h"
#include "video_core/v2/macro_hle.h"
#include "video_core/v2/qmd_front.h"
#include "video_core/v2/shader_writes.h"

namespace {

using namespace VideoCommon::V2;
using Kind = HazardResult::Kind;

/// Counts the fences and flushes LandCaptured makes.
class FenceRasterizer final : public VideoCore::RasterizerInterface {
public:
    int delayed_fences = 0;
    int other_fences = 0;
    int flushes = 0;

    void Draw(bool, u32) override {}
    void DrawTexture() override {}
    void Clear(u32) override {}
    void DispatchCompute() override {}
    void ResetCounter(VideoCommon::QueryType) override {}
    void Query(GPUVAddr, VideoCommon::QueryType, VideoCommon::QueryPropertiesFlags, u32,
               u32) override {}
    void BindGraphicsUniformBuffer(size_t, u32, GPUVAddr, u32) override {}
    void DisableGraphicsUniformBuffer(size_t, u32) override {}
    void SignalFence(std::function<void()>&&) override {
        // Delayed (never skipped as empty, so its operations wait for the GPU) only when the
        // report flag is set, as FenceManager::SignalFence decides.
        (VideoCore::g_signaling_query_report_fence ? delayed_fences : other_fences)++;
    }
    void SyncOperation(std::function<void()>&&) override {}
    void SignalSyncPoint(u32) override {}
    void SignalReference() override {}
    void ReleaseFences(bool) override {}
    void FlushAll() override {}
    void FlushRegion(DAddr, u64, VideoCommon::CacheType) override {}
    bool MustFlushRegion(DAddr, u64, VideoCommon::CacheType) override {
        return false;
    }
    VideoCore::RasterizerDownloadArea GetFlushArea(DAddr addr, u64 size) override {
        return {addr, addr + size, false};
    }
    void InvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void OnCacheInvalidation(DAddr, u64) override {}
    bool OnCPUWrite(DAddr, u64) override {
        return false;
    }
    void InvalidateGPUCache() override {}
    void UnmapMemory(DAddr, u64) override {}
    void ModifyGPUMemory(size_t, GPUVAddr, u64) override {}
    void FlushAndInvalidateRegion(DAddr, u64, VideoCommon::CacheType) override {}
    void WaitForIdle() override {}
    void FragmentBarrier() override {}
    void TiledCacheBarrier() override {}
    void FlushCommands() override {
        ++flushes;
    }
    void TickFrame() override {}
    Tegra::Engines::AccelerateDMAInterface& AccessAccelerateDMA() override {
        std::terminate();
    }
    void AccelerateInlineToMemory(GPUVAddr, size_t, std::span<const u8>) override {}
};

u32 Word(const u8* bytes) {
    u32 value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

} // Anonymous namespace

TEST_CASE("V2 hazard table: overlap, forwarding and retirement", "[video_core][v2_hazard]") {
    HazardTable table;
    const HazardFilter all{};

    SECTION("nothing queued is clear") {
        CHECK(table.Find(0x1000, 4, 0, all).kind == Kind::Clear);
    }

    SECTION("a known store forwards its bytes until the back ran it") {
        const u32 payload = 0x1234;
        table.NoteKnown(0x2000, &payload, 4, 10, Producer::Release);
        std::array<u8, 4> out{};
        const HazardResult queued = table.Find(0x2000, 4, 5, all, out.data());
        REQUIRE(queued.kind == Kind::Forward);
        CHECK(queued.covered);
        CHECK(queued.position == 10);
        CHECK(Word(out.data()) == payload);
        // Neighbours don't overlap.
        CHECK(table.Find(0x2004, 4, 5, all).kind == Kind::Clear);
        CHECK(table.Find(0x1ffc, 4, 5, all).kind == Kind::Clear);
        // Consumed: memory holds it.
        CHECK(table.Find(0x2000, 4, 10, all).kind == Kind::Clear);
        table.Retire(10);
        CHECK(table.Size() == 0);
    }

    SECTION("newer stores win, partial coverage keeps memory's bytes") {
        const std::array<u32, 4> wide{1, 0, 0xaaaa, 0xbbbb};
        table.NoteKnown(0x3000, wide.data(), 8, 4, Producer::Report);
        const u32 newer = 7;
        table.NoteKnown(0x3000, &newer, 4, 6, Producer::Release);
        std::array<u8, 12> out;
        out.fill(0xee);
        const HazardResult result = table.Find(0x3000, 12, 0, all, out.data());
        REQUIRE(result.kind == Kind::Forward);
        CHECK(result.position == 6);
        CHECK(!result.covered); // bytes 8-11 come from memory
        CHECK(Word(out.data()) == 7);
        CHECK(Word(out.data() + 4) == 0);
        CHECK(Word(out.data() + 8) == 0xeeeeeeee);
    }

    SECTION("an unknown write makes a lookup wait for it, unless newer stores cover it") {
        table.NoteRange(0x4000, 16, 8, Producer::Report);
        const u32 newer = 3;
        table.NoteKnown(0x4000, &newer, 4, 9, Producer::Release);
        // The report's other bytes are unknown.
        const HazardResult result = table.Find(0x4000, 8, 0, all);
        CHECK(result.kind == Kind::Wait);
        CHECK(result.position == 8);
        // The release covers its word: the report before it doesn't matter.
        std::array<u8, 4> out{};
        const HazardResult word = table.Find(0x4000, 4, 0, all, out.data());
        CHECK(word.kind == Kind::Forward);
        CHECK(word.covered);
        CHECK(Word(out.data()) == 3);
        // Once the back ran the report, only the known store is queued.
        const HazardResult after = table.Find(0x4000, 8, 8, all);
        CHECK(after.kind == Kind::Forward);
        CHECK(!after.covered);
    }

    SECTION("filters choose producers") {
        table.NoteRange(0x5000, 0x100, 12, Producer::CbData);
        HazardFilter semaphores;
        semaphores.ranged = ProducerBit(Producer::Release) | ProducerBit(Producer::Report);
        CHECK(table.Find(0x5010, 4, 0, semaphores).kind == Kind::Clear);
        CHECK(table.Find(0x5010, 4, 0, all).kind == Kind::Wait);
    }

    SECTION("unbounded writes, before and after the last sync") {
        table.NoteUnbounded(Producer::Dispatch, 20);
        table.NoteSync(21);
        table.NoteUnbounded(Producer::Dispatch, 30);
        HazardFilter any;
        any.ranged = 0;
        any.unbounded = ProducerBit(Producer::Dispatch);
        HazardFilter synced;
        synced.ranged = 0;
        synced.unbounded_synced = ProducerBit(Producer::Dispatch);
        CHECK(table.Find(0x9000, 4, 0, any).position == 30);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 20);
        CHECK(table.Find(0x9000, 4, 20, synced).kind == Kind::Clear);
        CHECK(table.Find(0x9000, 4, 20, any).kind == Kind::Wait);
        CHECK(table.PendingUnbounded(Producer::Dispatch, 25) == 30);
        CHECK(table.PendingUnbounded(Producer::Dispatch, 30) == 0);
    }

    SECTION("a sync at an older position orders the unbounded writes before it only") {
        HazardFilter synced;
        synced.ranged = 0;
        synced.unbounded_synced = ProducerBit(Producer::Copy);
        table.NoteUnbounded(Producer::Copy, 20);
        table.NoteUnbounded(Producer::Copy, 30);
        table.NoteUnbounded(Producer::Copy, 40);
        // An acquire met by a release at 25, decided after the copy at 40 was sent.
        table.NoteSync(25);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 20);
        CHECK(table.Find(0x9000, 4, 20, synced).kind == Kind::Clear);
        // Nothing before an older sync: still the newer sync's, never less.
        table.NoteSync(10);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 20);
        table.NoteSync(40);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 40);
        // Past the history: the oldest kept one, later than the true one (waits longer).
        for (u64 position = 100; position < 100 + 10 * HazardTable::UnboundedHistory;
             position += 10) {
            table.NoteUnbounded(Producer::Copy, position);
        }
        table.NoteSync(95);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 100);
        table.NoteUnbounded(Producer::Copy, 200);
        table.NoteSync(105);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 110);
    }

    SECTION("W-01a: a channel's own sync orders its unbounded writes, not other channels'") {
        HazardFilter synced;
        synced.ranged = 0;
        synced.unbounded_synced = ProducerBit(Producer::Dispatch);
        table.SetChannel(2);
        table.NoteUnbounded(Producer::Dispatch, 100); // channel 2's dispatch
        table.SetChannel(1);
        table.NoteUnbounded(Producer::Dispatch, 120); // channel 1's
        table.SetChannel(3);
        table.NoteUnbounded(Producer::Dispatch, 125); // channel 3's, after channel 1's
        table.SetChannel(1);
        table.NoteChannelSync(130); // channel 1's GP entry sync bit (or its WFI)
        const HazardResult own = table.Find(0x9000, 4, 0, synced);
        CHECK(own.kind == Kind::Wait);
        CHECK(own.position == 120); // its own dispatch, not channel 3's at 125
        CHECK(table.Find(0x9000, 4, 120, synced).kind == Kind::Clear);
        // Channel 2 sent no sync: its fetches are not ordered after anything.
        table.SetChannel(2);
        CHECK(table.Find(0x9000, 4, 0, synced).kind == Kind::Clear);
        // An acquire met by a release at 110 orders every channel's work before it.
        table.NoteSync(110);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 100);
        table.SetChannel(1);
        CHECK(table.Find(0x9000, 4, 0, synced).position == 120); // the later of both
        CHECK(table.Find(0x9000, 4, 110, synced).position == 120);
        // A channel with no unbounded write of its own: its sync waits for nothing new.
        table.SetChannel(4);
        table.NoteChannelSync(140);
        CHECK(table.Find(0x9000, 4, 100, synced).kind == Kind::Clear);
    }

    SECTION("constant buffer words grow one write within a page") {
        for (u32 i = 0; i < 64; ++i) {
            table.NoteRange(0x8000 + i * 4, 4, 100 + i, Producer::CbData);
        }
        CHECK(table.Size() == 1);
        const HazardResult result = table.Find(0x8040, 4, 0, all);
        CHECK(result.kind == Kind::Wait);
        CHECK(result.position == 163);
        // Past the page: a new write.
        table.NoteRange(0x8ffc, 4, 200, Producer::CbData);
        table.NoteRange(0x9000, 4, 201, Producer::CbData);
        CHECK(table.Size() == 3);
        CHECK(table.Find(0x9000, 4, 200, all).position == 201);
    }

    SECTION("long writes and long reads") {
        table.NoteRange(0x100000, 0x100000, 40, Producer::Copy); // 256 pages: wide
        CHECK(table.Find(0x180000, 4, 0, all).kind == Kind::Wait);
        CHECK(table.Find(0x300000, 4, 0, all).kind == Kind::Clear);
        CHECK(table.Find(0x0, 0x400000, 0, all).kind == Kind::Wait);
        CHECK(table.Find(0x0, 0x400000, 40, all).kind == Kind::Clear);
    }

    SECTION("many pending writes elsewhere don't slow a lookup down") {
        for (u64 i = 0; i < 3000; ++i) {
            table.NoteRange(0x200000 + i * 0x1000, 0x100, i + 1, Producer::CbData);
        }
        const u32 payload = 5;
        table.NoteKnown(0x10000, &payload, 4, 3001, Producer::Release);
        const HazardResult result = table.Find(0x10000, 4, 0, all);
        CHECK(result.kind == Kind::Forward);
        CHECK(!result.overflow);
    }

    SECTION("overflow folds into one marker") {
        for (u64 i = 0; i < HazardTable::Capacity + 10; ++i) {
            const u32 value = static_cast<u32>(i);
            table.NoteKnown(0x10000 + i * 0x1000, &value, 4, i + 1, Producer::Release);
        }
        CHECK(table.Size() == HazardTable::Capacity);
        CHECK(table.Overflows() == 10);
        // An address no entry covers still waits for the folded ones.
        const HazardResult result = table.Find(0x7000'0000, 4, 0, all);
        CHECK(result.kind == Kind::Wait);
        CHECK(result.overflow);
        CHECK(result.position == 10);
        CHECK(table.Find(0x7000'0000, 4, 10, all).kind == Kind::Clear);
        // An entry still held forwards once the folded ones ran.
        const HazardResult held = table.Find(0x10000 + 4000 * 0x1000, 4, 10, all);
        CHECK(held.kind == Kind::Forward);
        table.Retire(HazardTable::Capacity + 10);
        CHECK(table.Size() == 0);
    }

    SECTION("clear forgets everything") {
        const u32 value = 1;
        table.NoteKnown(0x2000, &value, 4, 3, Producer::Release);
        table.NoteUnbounded(Producer::Copy, 4);
        table.Clear();
        HazardFilter any;
        any.unbounded = AllProducers;
        CHECK(table.Find(0x2000, 4, 0, any).kind == Kind::Clear);
    }
}

// SYNC_SUBMIT_SPEC §13.5 #1: a Wait names its cause, so list-fetch waits can be told apart
// (overlapping producer, unbounded write, scan cap, found cap, overflow marker).
TEST_CASE("SYNC §13 hazard waits name their cause", "[video_core][v2_hazard]") {
    HazardTable table;
    HazardFilter all{};
    SECTION("an overlapping unknown write") {
        table.NoteRange(0x4000, 16, 8, Producer::Dispatch);
        const HazardResult result = table.Find(0x4000, 8, 0, all);
        REQUIRE(result.kind == Kind::Wait);
        CHECK(result.cause == CauseRanged);
        CHECK(result.causes == CauseRanged);
    }
    SECTION("an unbounded write, newer than an overlapping one") {
        table.NoteRange(0x4000, 16, 8, Producer::Dispatch);
        table.NoteUnbounded(Producer::Copy, 9);
        all.unbounded = ProducerBit(Producer::Copy);
        const HazardResult result = table.Find(0x4000, 8, 0, all);
        REQUIRE(result.kind == Kind::Wait);
        CHECK(result.position == 9);
        CHECK(result.cause == CauseUnbounded);
        CHECK(result.causes == (CauseUnbounded | CauseRanged));
    }
    SECTION("the scan cap: more known stores on the page than a lookup visits") {
        for (u64 i = 0; i <= HazardTable::MaxScan; ++i) {
            const u32 value = static_cast<u32>(i);
            table.NoteKnown(0x10004 + i * 4, &value, 4, 10 + i, Producer::Copy);
        }
        const HazardResult result = table.Find(0x10000, 4, 0, all);
        REQUIRE(result.kind == Kind::Wait);
        CHECK(result.cause == CauseScanCap);
        CHECK(result.overflow);
    }
    SECTION("the found cap: more known stores overlap than a lookup forwards") {
        for (u64 i = 0; i < 65; ++i) {
            const u32 value = static_cast<u32>(i);
            table.NoteKnown(0x20000 + i * 4, &value, 4, 10 + i, Producer::Copy);
        }
        const HazardResult result = table.Find(0x20000, 0x200, 0, all);
        REQUIRE(result.kind == Kind::Wait);
        CHECK(result.cause == CauseFoundCap);
    }
    SECTION("the overflow marker: more pending entries than the table holds") {
        for (u64 i = 0; i <= HazardTable::Capacity; ++i) {
            table.NoteRange(0x100000 + i * 0x1000, 16, 10 + i, Producer::CbData);
        }
        CHECK(table.Overflows() >= 1);
        HazardFilter dispatches;
        dispatches.ranged = ProducerBit(Producer::Dispatch);
        const HazardResult result = table.Find(0x9000'0000, 4, 0, dispatches);
        REQUIRE(result.kind == Kind::Wait);
        CHECK(result.cause == CauseOverflow);
        CHECK(result.producers == 0); // no writer overlaps the read
    }
}

TEST_CASE("V2 hazard table: lookups match a model on random writes", "[video_core][v2_hazard]") {
    struct Write {
        u64 address;
        u64 size;
        u64 position;
        bool known;
        std::array<u8, 16> bytes;
    };
    std::mt19937_64 rng{1234};
    HazardTable table;
    std::vector<Write> writes;
    u64 position = 0;
    u64 consumed = 0;
    const HazardFilter all{};
    int overflowed = 0;
    int forwards = 0;
    int waits = 0;
    for (int step = 0; step < 20000; ++step) {
        // Writes within one page: writes across pages are wide and never forwarded (the
        // dedicated section covers them).
        const u64 address = 0x10f00 + (rng() % 32) * 4;
        if (rng() % 3 != 0) {
            Write write{address, u64{4} << (rng() % 3), ++position, rng() % 4 != 0, {}};
            for (u8& byte : write.bytes) {
                byte = static_cast<u8>(rng());
            }
            if (write.known) {
                table.NoteKnown(write.address, write.bytes.data(), static_cast<u32>(write.size),
                                write.position, Producer::Release);
            } else {
                table.NoteRange(write.address, write.size, write.position, Producer::Report);
            }
            writes.push_back(write);
        }
        if (rng() % 5 == 0) {
            consumed = std::min(position, consumed + rng() % 10);
            if (rng() % 2 == 0) {
                table.Retire(consumed);
            }
        }
        const u64 size = u64{4} << (rng() % 2);
        const std::array<u8, 8> memory{};
        std::array<u8, 8> out{};
        const HazardResult result = table.Find(address, size, consumed, all, out.data());
        if (result.overflow) {
            ++overflowed; // A conservative wait
            REQUIRE(result.kind == Kind::Wait);
            continue;
        }
        // Expected: newest first, the pending writes that overlap until known stores cover every
        // byte; an unknown one met before that makes the lookup wait for it.
        u64 wait_for = 0;
        std::array<bool, 8> covered{};
        bool any = false;
        for (auto it = writes.rbegin(); it != writes.rend(); ++it) {
            const Write& write = *it;
            if (write.position <= consumed || write.address + write.size <= address ||
                write.address >= address + size) {
                continue;
            }
            any = true;
            if (!write.known) {
                wait_for = write.position;
                break;
            }
            bool full = true;
            for (u64 byte = 0; byte < size; ++byte) {
                const u64 at = address + byte;
                covered[byte] =
                    covered[byte] || (at >= write.address && at < write.address + write.size);
                full = full && covered[byte];
            }
            if (full) {
                break;
            }
        }
        if (wait_for != 0) {
            ++waits;
            REQUIRE(result.kind == Kind::Wait);
            REQUIRE(result.position == wait_for);
            continue;
        }
        if (!any) {
            REQUIRE(result.kind == Kind::Clear);
            continue;
        }
        // The bytes: every pending known store replayed in order over memory.
        std::array<u8, 8> expected = memory;
        for (const Write& write : writes) {
            if (write.position <= consumed || !write.known) {
                continue;
            }
            for (u64 byte = 0; byte < size; ++byte) {
                const u64 at = address + byte;
                if (at >= write.address && at < write.address + write.size) {
                    expected[byte] = write.bytes[at - write.address];
                }
            }
        }
        ++forwards;
        REQUIRE(result.kind == Kind::Forward);
        bool all_covered = true;
        for (u64 byte = 0; byte < size; ++byte) {
            all_covered = all_covered && covered[byte];
            if (covered[byte]) {
                REQUIRE(out[byte] == expected[byte]);
            }
        }
        REQUIRE(result.covered == all_covered);
        if (writes.size() > 2000) {
            std::erase_if(writes, [consumed](const Write& w) { return w.position <= consumed; });
        }
    }
    CHECK(table.Overflows() == 0);
    CHECK(overflowed < 1000);
    CHECK(forwards > 1000);
    CHECK(waits > 1000);
}

TEST_CASE("V2 acquire decisions", "[video_core][v2_hazard]") {
    HazardResult clear;
    HazardResult forward;
    forward.kind = Kind::Forward;
    forward.covered = true;
    forward.position = 50;
    HazardResult partial = forward;
    partial.covered = false;
    HazardResult unknown;
    unknown.kind = Kind::Wait;
    unknown.position = 40;

    SECTION("compares") {
        CHECK(SemaphoreMet(SemaphoreOp::Gequal, 5, 5));
        CHECK(!SemaphoreMet(SemaphoreOp::Gequal, 4, 5));
        CHECK(SemaphoreMet(SemaphoreOp::Equal, 5, 5));
        CHECK(!SemaphoreMet(SemaphoreOp::Equal, 6, 5));
        CHECK(SemaphoreMet(SemaphoreOp::Mask, 0x6, 0x2));
        CHECK(!SemaphoreMet(SemaphoreOp::Mask, 0x4, 0x2));
        CHECK(SemaphoreMet(SemaphoreOp::MaskZero, 0x4, 0x2));
        CHECK(SemaphoreMet(SemaphoreOp::CircularGequal, 2, 0xfffffff0));
        CHECK(!SemaphoreMet(SemaphoreOp::CircularGequal, 0xfffffff0, 2));
    }

    SECTION("nothing queued: memory decides") {
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 5, clear, 0, 7, 0).verdict ==
              AcquireVerdict::Met);
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 5, clear, 0, 4, 0).verdict ==
              AcquireVerdict::NotMet);
        CHECK(DecideAcquire(SemaphoreOp::Equal, 5, clear, 0, 5, 0).verdict ==
              AcquireVerdict::Met);
        CHECK(DecideAcquire(SemaphoreOp::Mask, 1, clear, 0, 0, 0).verdict ==
              AcquireVerdict::NotMet);
        // A word that only advances ignores queued dispatches; equal waits for them first.
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 5, clear, 0, 7, 99).verdict ==
              AcquireVerdict::Met);
        const AcquireDecision equal = DecideAcquire(SemaphoreOp::Equal, 5, clear, 0, 5, 99);
        CHECK(equal.verdict == AcquireVerdict::WaitFor);
        CHECK(equal.position == 99);
    }

    SECTION("a queued release decides a greater-or-equal acquire") {
        const AcquireDecision met = DecideAcquire(SemaphoreOp::Gequal, 5, forward, 6, 0, 0);
        CHECK(met.verdict == AcquireVerdict::MetByQueued);
        CHECK(met.position == 50);
        // Memory meets it but the queued release lowers the word: the release decides (only a
        // later release can meet it, so the channel suspends).
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 5, forward, 4, 9, 0).verdict ==
              AcquireVerdict::NotMet);
    }

    SECTION("equal and mask acquires: the queued value, unless a dispatch follows it") {
        CHECK(DecideAcquire(SemaphoreOp::Equal, 5, forward, 5, 0, 0).verdict ==
              AcquireVerdict::MetByQueued);
        CHECK(DecideAcquire(SemaphoreOp::Equal, 5, forward, 6, 5, 0).verdict ==
              AcquireVerdict::NotMet);
        CHECK(DecideAcquire(SemaphoreOp::Mask, 1, forward, 1, 0, 49).verdict ==
              AcquireVerdict::MetByQueued); // the dispatch ran before the release
        const AcquireDecision after = DecideAcquire(SemaphoreOp::Mask, 1, forward, 1, 0, 51);
        CHECK(after.verdict == AcquireVerdict::WaitFor);
        CHECK(after.position == 51);
        // A word that only advances ignores it.
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 1, forward, 1, 0, 51).verdict ==
              AcquireVerdict::MetByQueued);
    }

    SECTION("unknown or partial values fall back") {
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 5, unknown, 0, 9, 0).verdict ==
              AcquireVerdict::Fallback);
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 5, partial, 9, 9, 0).verdict ==
              AcquireVerdict::Fallback);
        CHECK(DecideAcquire(SemaphoreOp::Equal, 5, unknown, 0, 5, 0).verdict ==
              AcquireVerdict::Fallback);
    }

    // Fix F3-01 (MASTER_PLAN §6 class B, undoing Fix pass 1's removal): a fence-flagged release
    // queued, or run but not landed, meets the acquire logically; the landing rule
    // (v2/landing.h) keeps later CPU-visible writes behind it.
    SECTION("unlanded fence-flagged releases meet acquires (class B)") {
        HazardTable unlanded;
        const HazardFilter filter{};
        constexpr u64 word_address = 0x2'1070'2a00;
        const u32 payload = 0xf;
        const u64 ticket = (u64{60} << LandingSubBits) | 0;
        unlanded.NoteKnown(word_address, &payload, sizeof(payload), ticket, Producer::Release);
        for (const SemaphoreOp op :
             {SemaphoreOp::Gequal, SemaphoreOp::Equal, SemaphoreOp::CircularGequal}) {
            u32 forwarded = 0xe;
            HazardResult pending = unlanded.Find(word_address, 4, 0, filter,
                                                 reinterpret_cast<u8*>(&forwarded));
            pending.landing = true;
            REQUIRE(pending.kind == Kind::Forward);
            REQUIRE(forwarded == 0xf);
            const AcquireDecision decision = DecideAcquire(op, 0xf, pending, forwarded, 0xe, 0);
            CHECK(decision.verdict == AcquireVerdict::MetByQueued);
            CHECK(decision.position == ticket);
            // The dispatch rule compares ring positions, not tickets.
            if (!SemaphoreWaitsToReach(op)) {
                CHECK(DecideAcquire(op, 0xf, pending, forwarded, 0xe, 59).verdict ==
                      AcquireVerdict::MetByQueued);
                CHECK(DecideAcquire(op, 0xf, pending, forwarded, 0xe, 61).verdict ==
                      AcquireVerdict::WaitFor);
            }
            // Landed: memory holds it.
            const HazardResult landed = unlanded.Find(word_address, 4, ticket, filter, nullptr);
            REQUIRE(landed.kind == Kind::Clear);
            CHECK(DecideAcquire(op, 0xf, landed, 0, 0xf, 0).verdict == AcquireVerdict::Met);
        }
        // A newer release that doesn't meet it: the channel suspends.
        const u32 lower = 0xd;
        unlanded.NoteKnown(word_address, &lower, sizeof(lower), ticket + 1, Producer::Release);
        u32 forwarded = 0;
        HazardResult newest =
            unlanded.Find(word_address, 4, 0, filter, reinterpret_cast<u8*>(&forwarded));
        newest.landing = true;
        CHECK(forwarded == 0xd);
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 0xf, newest, forwarded, 0xe, 0).verdict ==
              AcquireVerdict::NotMet);
    }
}

// 84de5dbac4 regression (Ocarina of Time stuck before its first present): channel 1 suspended on
// an unmapped word (gequal 0, met by any read) the view can't read, and channels 2/3/5 on a word
// whose release the view never recorded; nothing resumed them once the valve was gone.
TEST_CASE("V2 acquire: a suspended acquire whose release the view never records progresses",
          "[video_core][v2_hazard]") {
    SECTION("what the view can't decide (settings it doesn't model) takes the old path at once") {
        // Fix F3h (R2-F3-03): an unmapped word is decided on the guest's read (gequal 0 passes
        // at once) and suspends when not met, waiting only for a queued map.
        CHECK(!ViewSuspends(AcquireVerdict::Fallback, true));
        CHECK(ViewSuspends(AcquireVerdict::Fallback, false));
        CHECK(ViewSuspends(AcquireVerdict::NotMet, false));
        CHECK(!ViewSuspends(AcquireVerdict::Met, false));
    }
    SECTION("a release written where the view doesn't look is found once the back went idle") {
        // Fix F3h (F3-X1): no timed net. The model: memory the view reads holds 0; the release
        // (1) is in work the back hasn't run yet (another channel's queued work). The view keeps
        // deciding NotMet; the poll waits for the back (its `consumed` store rings the front),
        // and once nothing is queued and no producer is known, the old read finds it.
        HazardResult clear;
        u32 memory_word = 0;
        u64 queued = 3; // entries the back still has to run; it runs one per poll
        bool resumed = false;
        for (int poll = 0; poll < 10 && !resumed; ++poll) {
            const AcquireDecision view =
                DecideAcquire(SemaphoreOp::Equal, 1, clear, memory_word, memory_word, 0);
            if (view.verdict == AcquireVerdict::Met) {
                resumed = true; // the poll's own memory read passes it
                break;
            }
            REQUIRE(ViewSuspends(view.verdict, false));
            // SYNC Step 4: no producer at a fence: the poll waits for its wake classes (the
            // back's `consumed`, a guest CPU write read every pass); nothing is asked for.
            REQUIRE(PlanSuspendedPoll(false, false, 0, 0) == SuspendedPoll::Wait);
            if (--queued == 0) {
                memory_word = 1; // the last queued entry wrote it
            }
        }
        CHECK(resumed);
        CHECK(PlanSuspendedPoll(true, false, 0, 0) == SuspendedPoll::ReadBack);
        CHECK(PlanSuspendedPoll(true, true, 5, 0) == SuspendedPoll::ReadBack);
    }
    SECTION("8d31: an at-fence producer whose fence is queued: one request per tick, idle back") {
        // SYNC Step 4: the 8d31 reductions are class B now (the semaphore view knows them). An
        // unknown producer at a fence (a counter report) still waits for its landing: the poll
        // files one request for its fence's tick, served even with the back idle (rule W), and
        // asks again only when the tick changes. The model: the fence lands once its tick is
        // submitted.
        u64 submitted = 0;
        u64 requested = 0;
        u32 requests = 0;
        const u64 fence_tick = 42;
        bool landed = false;
        for (int poll = 0; poll < 50 && !landed; ++poll) {
            switch (PlanSuspendedPoll(false, true, fence_tick, requested)) {
            case SuspendedPoll::Request:
                requested = fence_tick;
                ++requests;
                submitted = fence_tick; // the idle back serves it at once
                break;
            case SuspendedPoll::Wait:
                break;
            default:
                FAIL("only a request or a wait");
            }
            landed = submitted >= fence_tick;
        }
        CHECK(landed);
        CHECK(requests == 1);
        // A newer tick (a later fence carries it) asks once more; the same tick never again.
        CHECK(PlanSuspendedPoll(false, true, 43, 42) == SuspendedPoll::Request);
        CHECK(PlanSuspendedPoll(false, true, 42, 42) == SuspendedPoll::Wait);
        // No producer at a fence: nothing is asked for (a guest CPU release, queued work).
        CHECK(PlanSuspendedPoll(false, false, 42, 0) == SuspendedPoll::Wait);
    }
    SECTION("019d: a producer with no fence yet: the poll waits, nothing is forced") {
        // SYNC §7.3: the open tick holding it is submitted by rule I at the back's batch end or
        // by the fencing thread's rule W ([v2_p5b] "SYNC §7.3" tests); the poll asks only once
        // a fence tick names it.
        CHECK(PlanSuspendedPoll(false, true, 0, 0) == SuspendedPoll::Wait);
        CHECK(PlanSuspendedPoll(false, true, 9, 0) == SuspendedPoll::Request);
    }
    SECTION("an Equal acquire waits for the QMD producer it was given, not a newer one") {
        // 0fe4: the poll passed the newest unknown QMD release noted (moving every signal), so
        // with memory already meeting it the acquire never passed by the view.
        HazardResult clear;
        // The producer (ring position 200) settled: memory decides.
        CHECK(DecideAcquire(SemaphoreOp::Equal, 5, clear, 5, 5, 0).verdict == AcquireVerdict::Met);
        // Not settled: wait for that position.
        const AcquireDecision waiting = DecideAcquire(SemaphoreOp::Equal, 5, clear, 5, 5, 200);
        CHECK(waiting.verdict == AcquireVerdict::WaitFor);
        CHECK(waiting.position == 200);
        // A Gequal acquire doesn't wait for it (a word that only advances).
        CHECK(DecideAcquire(SemaphoreOp::Gequal, 5, clear, 5, 5, 200).verdict ==
              AcquireVerdict::Met);
    }
    SECTION("an untracked release reports its entry when it is queued and when it lands") {
        const u64 saved_tag = Landing::tag;
        const u64 issued = Landing::untracked_issued.load();
        const u64 landed = Landing::untracked_landed.load();
        Landing::untracked_issued.store(0);
        Landing::untracked_landed.store(0);
        Landing::untracked_captured_tag = 0;
        CHECK(Landing::CaptureUntracked(0x1000, 4) == 0); // outside an Untracked scope
        Landing::tag = 700;
        u64 position = 0;
        {
            const Landing::Untracked untracked;
            CHECK(Landing::Capture() == 0); // no ticket: the front never numbered it
            position = Landing::CaptureUntracked(0x1000, 4);
        }
        CHECK(Landing::tag == 700);
        CHECK(Landing::untracked_tag == 0);
        CHECK(position == Landing::UntrackedTicket(700, 0));
        CHECK(Landing::untracked_issued.load() > Landing::untracked_landed.load()); // pending
        Landing::NoteUntracked(position);
        CHECK(Landing::untracked_issued.load() == Landing::untracked_landed.load());
        Landing::tag = saved_tag;
        Landing::untracked_issued.store(issued);
        Landing::untracked_landed.store(landed);
    }
    SECTION("F3-X6: untracked tickets tell one entry's operations apart and settle in order") {
        const u64 saved_tag = Landing::tag;
        Landing::ResetUntracked();
        Landing::untracked_captured_tag = 0;
        u64 a = 0;
        u64 b = 0;
        u64 c = 0;
        {
            Landing::tag = 300;
            const Landing::Untracked untracked;
            a = Landing::CaptureUntracked(0x1000, 16);
            b = Landing::CaptureUntracked(0x2000, 4);
        }
        {
            Landing::tag = 310;
            const Landing::Untracked untracked;
            c = Landing::CaptureUntracked(0x3000, 4);
        }
        CHECK(a == Landing::UntrackedTicket(300, 0));
        CHECK(b == Landing::UntrackedTicket(300, 1));
        CHECK(c == Landing::UntrackedTicket(310, 0));
        // Nothing landed: no entry is settled, whatever the back consumed.
        CHECK(Landing::UntrackedSettled(320) == 0);
        // The ranges: a read of 0x2000 waits for b only, one of 0x4000 for nothing.
        CHECK(Landing::UntrackedTargetFor(0x2000, 4) == b);
        CHECK(Landing::UntrackedTargetFor(0x1008, 4) == a);
        CHECK(Landing::UntrackedTargetFor(0x4000, 4) == 0);
        // The first operation of entry 300 ran: entry 300 is not settled yet (b may not have).
        Landing::NoteUntracked(a);
        CHECK(Landing::UntrackedSettled(320) == 299);
        CHECK(Landing::UntrackedTargetFor(0x1000, 16) == 0);
        // One of a later entry ran: every earlier one did (operations run in capture order).
        Landing::NoteUntracked(b);
        Landing::NoteUntracked(c);
        CHECK(Landing::UntrackedSettled(320) == 320); // all landed: up to what was consumed
        // Ranges that fell out of the slots are waited for conservatively.
        Landing::tag = 400;
        {
            const Landing::Untracked untracked;
            for (size_t i = 0; i <= Landing::UntrackedWriteSlots; ++i) {
                (void)Landing::CaptureUntracked(0x10000 + i * 16, 4);
            }
        }
        CHECK(Landing::UntrackedTargetFor(0x4000, 4) == Landing::untracked_issued.load());
        Landing::tag = saved_tag;
        Landing::ResetUntracked();
    }
}

// Fix F3h (F3-X6b, R2-F3-04): 0fe4's HAZARD=verify line. The front noted a release as made at
// once; the back deferred it (its own ticket for a write the front never numbered was pending),
// the front's note retired at consumption and the narrowed read took memory's older word. The
// back now obeys the front's verdict for every write it could make at once.
TEST_CASE("V2 landing: the back obeys the front's landing verdict", "[video_core][v2_hazard]") {
    const u64 saved_tag = Landing::tag;
    const u64 saved_pending = Landing::pending;
    const u64 saved_landed = Landing::landed.load();
    Landing::ResetVerdicts();
    Landing::verdicts_missing.store(0);
    // Outside the split (v1): nothing is deferred, no verdict is read.
    Landing::tag = 0;
    CHECK(!Landing::DeferAtOnceWrite());
    // A ticket the back captured for a write the front never numbered is still to land.
    Landing::landed.store(Landing::Ticket(50, 0));
    Landing::pending = Landing::Ticket(90, 0);
    Landing::PushVerdict(100, true);  // the front: at once (its own tickets had landed)
    Landing::PushVerdict(110, false); // the front: at an operation (a ticket it numbered)
    Landing::PushVerdict(120, true);  // a write the back never makes
    Landing::tag = 100;
    CHECK(Landing::MustDefer()); // what the back used to decide on its own
    CHECK(!Landing::DeferAtOnceWrite());
    // Landed meanwhile: the front's ticket verdict still takes a ticket (the numbering matches).
    Landing::landed.store(Landing::pending);
    Landing::tag = 110;
    CHECK(!Landing::MustDefer());
    CHECK(Landing::DeferAtOnceWrite());
    // No verdict for this entry (120's was for a write the back didn't make): the back's own
    // classification, counted when it defers.
    Landing::tag = 130;
    CHECK(!Landing::DeferAtOnceWrite());
    Landing::pending = Landing::Ticket(125, 0);
    CHECK(Landing::DeferAtOnceWrite());
    CHECK(Landing::verdicts_missing.load() == 1);
    CHECK(Landing::verdict_tail.load() == Landing::verdict_head.load());
    // Two writes of one entry take their verdicts in order.
    Landing::landed.store(Landing::pending);
    Landing::PushVerdict(140, false);
    Landing::PushVerdict(140, true);
    Landing::tag = 140;
    CHECK(Landing::DeferAtOnceWrite());
    CHECK(!Landing::DeferAtOnceWrite());
    Landing::tag = saved_tag;
    Landing::pending = saved_pending;
    Landing::landed.store(saved_landed);
    Landing::ResetVerdicts();
}

// SYNC_SUBMIT_SPEC §12 R1: the 0x6c3/QMD reduction path classified with MustDeferUntracked on
// the back; the front's model now decides, and SemView keeps the reduction pending until it
// lands at its fence.
TEST_CASE("SYNC §12 R1: a QMD reduction behind a pending ticket lands by the front's verdict",
          "[video_core][v2_hazard]") {
    const u64 saved_tag = Landing::tag;
    const u64 saved_pending = Landing::pending;
    const u64 saved_landed = Landing::landed.load();
    Landing::ResetUntracked();
    Landing::qmd_verdicts_missing.store(0);
    // The front: a ticket it gave out (Ticket(190, 0)) has not landed when its model launches
    // the reduction at 200: at its fence.
    Landing::landed.store(Landing::Ticket(150, 0));
    Landing::pending = Landing::Ticket(190, 0);
    Landing::PushQmdVerdict(200, false);
    Landing::PushQmdVerdict(210, true);
    // The ticket landed before the back ran 200: its own rule would make it at once.
    Landing::landed.store(Landing::pending);
    Landing::tag = 200;
    u64 ticket = 0;
    {
        const Landing::Untracked untracked;
        CHECK(!Landing::MustDeferUntracked());
        CHECK(Landing::DeferAtOnceUntracked());
        ticket = Landing::CaptureUntracked(0x8000, 4);
    }
    CHECK(ticket != 0);
    // SemView retires the reduction by its untracked operation, not by `consumed`.
    SemView view;
    const SemProgress before{.consumed = 200,
                             .landed = Landing::landed.load(),
                             .untracked_settled = Landing::UntrackedSettled(200)};
    view.Note(0x8000, 0x13, true, 200, SemView::Retire::Untracked, 200, SemView::Kind::QmdReduction,
              before);
    CHECK(view.Find(0x8000, before).pending);
    Landing::NoteUntracked(ticket);
    const SemProgress after{.consumed = 200,
                            .landed = Landing::landed.load(),
                            .untracked_settled = Landing::UntrackedSettled(200)};
    CHECK(!view.Find(0x8000, after).pending);
    // The next one: at once by the front's verdict; one without a verdict falls back.
    Landing::tag = 210;
    {
        const Landing::Untracked untracked;
        CHECK(!Landing::DeferAtOnceUntracked());
        Landing::pending = Landing::Ticket(215, 0);
        CHECK(Landing::DeferAtOnceUntracked()); // no verdict left: MustDeferUntracked
    }
    CHECK(Landing::qmd_verdicts_missing.load() == 1);
    // Outside an Untracked scope nothing reads the QMD queue.
    CHECK(!Landing::DeferAtOnceUntracked());
    Landing::tag = saved_tag;
    Landing::pending = saved_pending;
    Landing::landed.store(saved_landed);
    Landing::ResetUntracked();
}

// Fix F3h (W-01c, R2-F3-02): one release patched into a scheduled QMD (NVN2), still to land,
// made the front's QMD model unknown until the next drain (~23 unknown signals each); and the
// model's resync rode the unmapped-acquire drain, which is gone.
TEST_CASE("V2 QMD model: patched release words keep it exact; it resyncs without a drain",
          "[video_core][v2_hazard]") {
    namespace QmdFront = VideoCommon::V2::QmdFront;
    using Descriptor = Tegra::NativeQmdScheduler::Descriptor;
    SECTION("only release words differ: the scheduling is the same for either") {
        Descriptor memory{};
        memory[4] = 1U << 14; // one dependency
        memory[24] = 0x1000;
        memory[25] = (1U << 23) | (1U << 30); // one-word release, enabled
        memory[26] = 5;
        Descriptor patched = memory;
        patched[26] = 6; // the payload
        patched[24] = 0x2000; // and the address
        CHECK(QmdFront::OnlyReleaseWordsDiffer(memory, patched));
        std::vector<std::pair<u64, u64>> releases;
        QmdFront::NoteBothReleases(patched, releases);
        QmdFront::NoteBothReleases(memory, releases);
        CHECK(releases == std::vector<std::pair<u64, u64>>{{0x2000, 4}, {0x1000, 4}});
        patched[4] = 2U << 14; // a dependency count: the model can't know which the back sees
        CHECK(!QmdFront::OnlyReleaseWordsDiffer(memory, patched));
    }
    SECTION("an unknown model resyncs by a back job once the back ran its signals (idle back)") {
        struct FakeSplit {
            std::atomic<u64> consumed{0};
            std::atomic<bool> running{true};
            void (*job)(void*) = nullptr;
            void* argument = nullptr;
            std::atomic<bool> requested{false};
            u64 ConsumedNow() const {
                return consumed.load();
            }
            bool BackRunning() const {
                return running.load();
            }
            void RequestBackJob(void (*new_job)(void*), void* new_argument) {
                job = new_job;
                argument = new_argument;
                requested.store(true);
            }
        } split;
        auto& model = QmdFront::FrontModel();
        model.Invalidate();
        model.NoteSignalSent(500);
        // The back hasn't run the newest signal the model saw: no copy (it would lack it).
        split.consumed.store(400);
        CHECK(!QmdFront::ResyncByJob(split));
        CHECK(!split.requested.load());
        CHECK(!model.Valid());
        // It has: an idle back serves the job after its current entry, the front takes the copy.
        split.consumed.store(500);
        std::jthread back([&] {
            while (!split.requested.load()) {
                std::this_thread::yield();
            }
            split.job(split.argument);
        });
        CHECK(QmdFront::ResyncByJob(split));
        CHECK(model.Valid());
        back.join();
        // A back that stops before running it: no copy, no hang.
        model.Invalidate();
        split.requested.store(false);
        split.running.store(false);
        CHECK(!QmdFront::ResyncByJob(split));
        CHECK(!model.Valid());
        model.Resync(Tegra::NativeQmdScheduler::BackTable());
        model.NoteSignalSent(0);
    }
}

// F3F5b-1/-2: a wait that must see what a report's operation writes (its timestamp, or a release
// the back deferred behind an earlier one) makes the fence behind the back's captured operations,
// and waits for the back's own newest ticket, not the front's numbering.
TEST_CASE("V2 landing: the back lands its captured operations for a waiting front",
          "[video_core][v2_hazard]") {
    FenceRasterizer rasterizer;
    const u64 saved_pending = Landing::pending;
    const u64 saved_landed = Landing::landed.load();
    SECTION("nothing captured above landed: no fence, nothing to wait for") {
        Landing::landed.store(Landing::Ticket(50, 0));
        Landing::pending = Landing::Ticket(50, 0);
        CHECK(IrExec::LandCaptured(rasterizer) == 0);
        CHECK(rasterizer.delayed_fences + rasterizer.other_fences == 0);
        CHECK(rasterizer.flushes == 0);
    }
    SECTION("captured operations still to land: one delayed fence, submitted, their ticket") {
        Landing::landed.store(Landing::Ticket(50, 0));
        // The front's ticket for the report may be (60, 1): the back numbered the write it made
        // at once in that entry not at all, so its own ticket is (60, 0).
        Landing::pending = Landing::Ticket(60, 0);
        CHECK(IrExec::LandCaptured(rasterizer) == Landing::Ticket(60, 0));
        CHECK(rasterizer.delayed_fences == 1);
        CHECK(rasterizer.other_fences == 0);
        CHECK(rasterizer.flushes == 1);
        CHECK(!VideoCore::g_signaling_query_report_fence); // restored
    }
    Landing::pending = saved_pending;
    Landing::landed.store(saved_landed);
}

TEST_CASE("V2 hazard table: wide writes and draw targets", "[video_core][v2_hazard]") {
    HazardTable table;
    const HazardFilter all{};

    SECTION("a wide write is found without scanning unrelated entries") {
        table.NoteRange(0x1000000, 0x200000, 1, Producer::Dispatch); // a 2 MiB storage buffer
        for (u64 i = 0; i < 3000; ++i) {
            table.NoteRange(0x400000 + i * 0x1000, 0x40, i + 2, Producer::CbData);
        }
        const HazardResult hit = table.Find(0x1100000, 4, 0, all);
        CHECK(hit.kind == Kind::Wait);
        CHECK(hit.position == 1);
        CHECK(!hit.overflow);
        CHECK(table.Find(0x3000000, 4, 0, all).kind == Kind::Clear);
        CHECK(table.Find(0x1100000, 4, 1, all).kind == Kind::Clear);
    }

    SECTION("more wide writes than the list holds fall back to scanning") {
        for (u64 i = 0; i < 70; ++i) {
            table.NoteRange(0x10000000 + i * 0x100000, 0x20000, i + 1, Producer::Copy);
        }
        // The oldest ones left the list but are still found.
        CHECK(table.Find(0x10000000 + 0x10, 4, 0, all).position == 1);
        CHECK(table.Find(0x10000000 + 69 * 0x100000, 4, 0, all).position == 70);
    }
}

TEST_CASE("V2 write ranges: uploads and shader write summaries", "[video_core][v2_hazard]") {
    SECTION("pitch-linear upload") {
        // line_length_in, line_count, address high/low, pitch, block, width, height, depth...
        const std::array<u32, 12> upload{0x40, 3, 0x1, 0x2000, 0x100, 0, 0, 0, 1, 0, 0, 0};
        const auto range = UploadRange(upload.data(), 1);
        REQUIRE(range);
        CHECK(range->first == 0x1'0000'2000ULL);
        CHECK(range->second == 2 * 0x100 + 0x40);
    }
    SECTION("block-linear upload covers its whole surface") {
        const std::array<u32, 12> upload{0x100, 4, 0, 0x10000, 0, 0, 0x100, 16, 1, 0, 0, 0};
        const auto range = UploadRange(upload.data(), 0);
        REQUIRE(range);
        CHECK(range->second >= 0x100 * 4);
    }
    SECTION("summaries") {
        constexpr u64 address = 0x7777'0000;
        CHECK(!ShaderWrites::Lookup(address));
        Shader::Info info{};
        ShaderWrites::Record(address, info);
        auto summary = ShaderWrites::Lookup(address);
        REQUIRE(summary);
        CHECK(summary->WritesNothing());
        info.storage_buffers_descriptors.push_back(
            Shader::StorageBufferDescriptor{.cbuf_index = 1, .cbuf_offset = 0x20, .count = 2,
                                            .is_written = true});
        ShaderWrites::Record(address, info);
        summary = ShaderWrites::Lookup(address);
        REQUIRE(summary);
        CHECK(!summary->unbounded);
        REQUIRE(summary->num_storage == 1);
        CHECK(summary->storage[0].cbuf_index == 1);
        CHECK(summary->storage[0].cbuf_offset == 0x20);
        CHECK(summary->storage[0].count == 2);
        info.stores_global_memory = true;
        ShaderWrites::Record(address, info);
        CHECK(ShaderWrites::Lookup(address)->unbounded);
        // Ranges that miss the key keep it: other pages, the same page around it, empty ranges.
        ShaderWrites::Invalidate(address + 0x1000, address + 0x3000);
        ShaderWrites::Invalidate(address - 0x2000, address);
        ShaderWrites::Invalidate(address + 0x10, address + 0x800);
        ShaderWrites::Invalidate(address, address);
        CHECK(ShaderWrites::Lookup(address));
        // A second program on a far page: an invalidation between the two keeps both.
        constexpr u64 far = 0x9'7777'0000;
        ShaderWrites::Record(far, info);
        ShaderWrites::Invalidate(address + 0x1000, far);
        CHECK(ShaderWrites::Lookup(address));
        CHECK(ShaderWrites::Lookup(far));
        ShaderWrites::Invalidate(far - 0x10, far + 1);
        CHECK(!ShaderWrites::Lookup(far));
        ShaderWrites::Invalidate(address, address + 0x100);
        CHECK(!ShaderWrites::Lookup(address));
        // Recorded again after an invalidation: found and invalidated again; each moves the
        // generation the front caches its draw verdicts by, a miss doesn't.
        const u32 generation = ShaderWrites::Generation();
        ShaderWrites::Record(address, info);
        CHECK(ShaderWrites::Generation() != generation);
        const u32 recorded = ShaderWrites::Generation();
        ShaderWrites::Record(address, info);
        ShaderWrites::Invalidate(address + 0x1000, address + 0x2000);
        CHECK(ShaderWrites::Generation() == recorded);
        CHECK(ShaderWrites::Lookup(address));
        ShaderWrites::Invalidate(address & ~u64{0xfff}, (address & ~u64{0xfff}) + 0x1000);
        CHECK(!ShaderWrites::Lookup(address));
        CHECK(ShaderWrites::Generation() != recorded);
    }
}

TEST_CASE("V2 NVN2 macro identification and HLE entries", "[video_core][v2_hazard]") {
    using namespace VideoCommon::V2::MacroHle;
    // Three instructions; the second has the end bit (bit 0 of its third word): the macro is it
    // and the one after (its delay slot).
    std::vector<u32> code(3 * 8, 0x22222222);
    code[3 * 3 + 2] |= 1; // instruction 3 ends after instruction 4
    const Identity identity = Identify(code, 2);
    CHECK(identity.instructions == 3);
    // The same code elsewhere has the same hash; other code doesn't.
    std::vector<u32> moved(3 * 8, 0);
    std::copy(code.begin() + 6, code.begin() + 15, moved.begin() + 12);
    CHECK(Identify(moved, 4).hash == identity.hash);
    code[3 * 2] ^= 1;
    CHECK(Identify(code, 2).hash != identity.hash);

    const auto entry = ParseLine("0123456789abcdef draw_indexed_indirect topology=p0 "
                                 "address=p2,p3 count=p1 stride=20 base_vertex_attribute");
    REQUIRE(entry);
    CHECK(entry->hash == 0x0123456789abcdefULL);
    CHECK(entry->kind == VideoCommon::V2::MacroHle::Kind::DrawIndexedIndirect);
    const std::array<u32, 4> parameters{4, 3, 0x12, 0x3400};
    const Draw draw = VideoCommon::V2::MacroHle::Resolve(*entry, parameters);
    CHECK(draw.topology == 4);
    CHECK(draw.address == 0x12'0000'3400ULL);
    CHECK(draw.max_draws == 3);
    CHECK(draw.stride == 20);
    CHECK(draw.base_vertex_attribute);
    CHECK(!ParseLine("# a comment"));
    CHECK(!ParseLine("1234 draw_arrays_indirect address=p1,p2")); // no topology
    CHECK(!ParseLine("1234 multi_draw_indexed_indirect_count topology=1 address=p1,p2"));
    CHECK(ParseLine("1234 draw_arrays_indirect topology=5 address=p1,p2"));
}

TEST_CASE("V2 front writes: block linear copies and 2D blits are bounded by their surfaces",
          "[video_core][v2_hazard]") {
    SECTION("copy engine block linear destination") {
        DmaWriteTracker tracker;
        // 256 x 64 bytes, block height 2^4 GOBs: 256 bytes x 128 rows (aligned) x 1.
        tracker.dst_params = {4U << 4, 256, 64, 1, 0, 0};
        const auto bytes = tracker.BlockLinearBytes(false);
        REQUIRE(bytes);
        CHECK(*bytes == 256U * 128U);
        // With remapping, 4 components of 4 bytes per pixel.
        tracker.remap_components = (3U << 16) | (3U << 24);
        CHECK(tracker.BlockLinearBytes(true) == 256U * 16U * 128U);
        // Layers and block widths the copies don't handle: not bounded.
        tracker.dst_params[4] = 1;
        CHECK(!tracker.BlockLinearBytes(false));
        tracker.dst_params[4] = 0;
        tracker.dst_params[0] |= 1;
        CHECK(!tracker.BlockLinearBytes(false));
    }
    SECTION("2D destination surface") {
        BlitWriteTracker tracker;
        // A8B8G8R8_UNORM (0xd5), pitch linear, pitch 1024, 100 rows.
        tracker.dst = {0xd5, 1, 0, 1, 0, 1024, 256, 100, 0x1, 0x2000};
        auto range = tracker.DestinationRange();
        REQUIRE(range);
        CHECK(range->first == 0x1'0000'2000);
        CHECK(range->second == 1024U * 100U);
        // Block linear: 256 pixels x 4 bytes, 100 rows aligned to 8 << 2 = 32.
        tracker.dst[1] = 0;
        tracker.dst[2] = 2U << 4;
        range = tracker.DestinationRange();
        REQUIRE(range);
        CHECK(range->second == 1024U * 128U);
        tracker.macros = true; // registers a 2D macro set are not tracked
        CHECK(!tracker.DestinationRange());
    }
}

// Fix F3f (W-01b): the front's sleep while every channel is suspended ends at a ring (a release
// latching a waiter, the back's progress, a readback, a command), and without one after a tick.
TEST_CASE("V2 front wake: a ring ends the suspended front's sleep", "[video_core][v2_hazard]") {
    namespace FrontWake = VideoCommon::V2::FrontWake;
    FrontWake::RegisterFrontThread();
    // No ring: the sleep still ends (the guest CPU's releases are polled).
    FrontWake::Sleep();
    // A ring while nothing sleeps does nothing (and leaves no wake for later).
    FrontWake::Ring();
    // An idle back ringing: every sleep ends, the front makes progress at the ring's pace.
    std::atomic<bool> stop{false};
    std::jthread ringer([&] {
        while (!stop.load()) {
            FrontWake::Ring();
            std::this_thread::yield();
        }
    });
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; ++i) {
        FrontWake::Sleep();
    }
    CHECK(std::chrono::steady_clock::now() - begin < std::chrono::seconds{2});
    stop.store(true);
}

TEST_CASE("SYNC step 5: each wake source rings the front's sleep until a ring",
          "[video_core][v2_hazard]") {
    namespace FrontWake = VideoCommon::V2::FrontWake;
    FrontWake::RegisterFrontThread();
    // A ring between the epoch and the sleep is never lost: the sleep doesn't start.
    u64 seen = FrontWake::Epoch();
    FrontWake::Ring();
    CHECK(FrontWake::SleepUntilRing(seen) == FrontWake::SleepEnd::Ring);
    // Each source the spec's wake table names rings from another thread and ends the sleep.
    const auto ring_from = [&](auto&& source) {
        seen = FrontWake::Epoch();
        std::atomic<bool> go{false};
        std::jthread other([&] {
            while (!go.load()) {
                std::this_thread::yield();
            }
            source();
        });
        go.store(true);
        // Either the ring arrives first (no sleep) or it ends the sleep; never the bound alone,
        // unless the scheduler delayed the other thread past it (then it rings at once after).
        const auto end = FrontWake::SleepUntilRing(seen);
        other.join();
        return end == FrontWake::SleepEnd::Ring || FrontWake::Epoch() != seen;
    };
    // Landing (a ticketed write's operation ran).
    CHECK(ring_from([] { Landing::Note(Landing::landed.load()); }));
    // Untracked landing (a QMD write's operation ran).
    CHECK(ring_from([] { Landing::NoteUntracked(Landing::untracked_landed.load()); }));
    // Map executed.
    CHECK(ring_from([] { VideoCommon::GpuSplit::NoteMapExecuted(); }));
    // Consumed / readback / new work / GPU progress all call Ring directly.
    CHECK(ring_from([] { FrontWake::Ring(); }));
    // A guest CPU waiter keeps the tick sleep (nothing rings for a CPU store).
    FrontWake::NoteTickWaiter();
    CHECK(FrontWake::TakeTickWaiter());
    CHECK_FALSE(FrontWake::TakeTickWaiter());
}
