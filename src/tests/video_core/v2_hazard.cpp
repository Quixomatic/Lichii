// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/shader_info.h"
#include "video_core/v2/front_writes.h"
#include "video_core/v2/hazard.h"
#include "video_core/v2/macro_hle.h"
#include "video_core/v2/shader_writes.h"

namespace {

using namespace VideoCommon::V2;
using Kind = HazardResult::Kind;

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
    const Draw draw = Resolve(*entry, parameters);
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
