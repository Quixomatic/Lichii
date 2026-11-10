// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/macro_turing.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif
#include "video_core/v2/macro_hle.h"
#include "video_core/v2/macro_memo.h"

namespace {

using namespace VideoCommon::V2;
using Pairs = std::vector<std::pair<u32, u32>>;

// Macros from Ocarina of Time's Castle Town dump (tests/runs/...-macrodump-stats), as dumped:
// three words per instruction, the macro starting at instruction 0.
std::vector<u32> Code(const std::string& text) {
    std::vector<u32> code;
    std::istringstream stream{text};
    std::string instruction;
    while (stream >> instruction) {
        for (size_t i = 0; i < 3; ++i) {
            code.push_back(static_cast<u32>(std::stoul(instruction.substr(i * 9, 8), nullptr, 16)));
        }
    }
    code.resize(code.size() + 3 * 8); // room past the end, as in the code RAM
    return code;
}

// 3D 966aeb0e43e7a13c: a conditional render from a report pair read by MME_DMA_READ_FIFOED.
const std::string render_enable_code =
    "00940003:18c00455:31c10300 8a600003:18c01856:318c0300 00000007:18c00400:359c1f00 "
    "00000003:1d000000:31c08300 00000003:18c00000:3bcc0300 00000003:a003c000:3810bf00 "
    "00000003:18c00000:0010bb00 0018155b:18c00000:72213c29 00200003:18c00000:302c0300";
// 3D 7bcd37c4e888f3ca: reads register 0x48c (MME STATE), writes 0x674 that many times.
const std::string state_read_code =
    "00000003:1a058123:31c08300 00000003:18c00000:b00c5f00 00341003:2108019d:301c0300 "
    "00000003:18c00000:318c0301 00000003:18c00000:318c0300";
// 3D 0e913127a3eb5239: writes 0x376 = p0 when data RAM word 0x53b differs, and stores it there.
const std::string data_ram_code =
    "00000003:1c08014e:f1907b00 00000003:18c00000:00105701 003414ec:39c7c0dd:b01c0002";
// 3D 4b30d9cf2c59a44d: writes 0xd00 and then polls it (a read after its own write).
const std::string poll_code =
    "0d147403:18c00011:31c00300 30a00003:18c00000:318c0300 1a600e53:19078231:300c0300 "
    "00000003:18c00040:01804300 0000000b:18c02001:b40c7700 00000003:18c00340:31905b00 "
    "00000007:18c02400:340c7700 01143403:18c00010:31a05b00 00000007:18c017ff:f40c7701 "
    "00000003:18c02000:f18c6300";

/// A front stand-in: registers, data RAM, guest memory for MME_DMA_READ_FIFOED.
struct Front {
    std::vector<u32> code;
    std::array<u32, 0x1000> regs{};
    Tegra::TuringMacroInterpreter::DataRam dram{};
    std::array<u32, 6> memory{};
    Pairs writes;
    u32 dma_reads = 0;
    u64 generation = 1;
    bool status_d00 = false; ///< Register 0xd00 reads back 1 (a status the engine sets)
    u32 stale = ~0U;         ///< CallInfo::stale_register
    u32 live_runs = 0;       ///< Calls of the JIT stand-in (bypassed, verify, JIT recordings)
    u32 hooked_runs = 0;     ///< JIT runs with data RAM hooks (F3d recordings)
    // Fix F3e: Write3D's compare-on-write (skip_same), the table's executable bit (1) and
    // shadow RAM mode barrier (4), shadow RAM tracking, a deferred draw, the count of dropped
    // writes (`drop_counted` false: a write path that doesn't drop what it should). F3d: the
    // JIT's data RAM recordings.
    bool skip_same = false;
    std::array<u16, 0x1000> flags{};
    bool track = false;
    std::array<u32, 0x1000> shadow{};
    u8 deferred_mode = 0;    ///< DrawModelState::mode (1: instance) ...
    u32 deferred_count = 0;  ///< ... with instances: a deferred draw
    /// Fix F3h (R2-F5-01): methods a hit applies through Store, and what it applied.
    std::array<u8, 0x1000> store_class{};
    bool classified = false;
    Pairs applied;
    u64 dropped = 0;
    bool drop_counted = true;
    bool jit_data_ram = false;
    Tegra::TuringMacroInterpreter interpreter;
    MacroMemo::Memo memo;

    explicit Front(const std::string& text) : code{Code(text)} {
        interpreter.active_dram = &dram;
    }

    bool Write(u32 method, u32 value) {
        if (method >= 0xe00) {
            return false;
        }
        const u32 stored = status_d00 && method == 0xd00 ? 1 : value;
        writes.emplace_back(method, value);
        const bool kept = !track || shadow[method] == value;
        if (track) {
            shadow[method] = value;
        }
        if (skip_same && (flags[method] & 1) == 0 && regs[method] == stored && kept) {
            dropped += drop_counted ? 1 : 0; // as Write3D drops it
            return true;
        }
        regs[method] = stored;
        return true;
    }
    void Dma(std::vector<u32>& fifo, u32 method, u32 value) {
        ++dma_reads;
        if (method == 0x158) {
            fifo.insert(fifo.end(), memory.begin(), memory.begin() + std::min<u32>(value, 6));
        }
    }

    /// The call as the front runs it without the memo (the interpreter stands in for the JIT).
    Pairs Plain(std::vector<u32> fifo) {
        writes.clear();
        Tegra::TuringMacroInterpreter plain;
        plain.active_dram = interpreter.active_dram;
        const auto result = plain.Execute(
            code, 0, fifo, [&](u32 address) { return address < regs.size() ? regs[address] : 0U; },
            [&](u32 method, u32 value) {
                if (!Write(method, value)) {
                    return false;
                }
                if (method == 0x158 || method == 0x157) {
                    Dma(fifo, method, value);
                }
                return true;
            });
        REQUIRE(result.success);
        return writes;
    }

    Pairs Memoised(std::vector<u32> fifo, bool verify = false) {
        writes.clear();
        memo.Synchronize(generation);
        const std::array<u32, 1> side_effects{0x8e3}; // a register writes may change
        const MacroMemo::CallInfo call{
            .start = 0,
            .fifo = &fifo,
            .dram = dram,
            .code = code,
            .generation = &generation,
            .regs = regs.data(),
            .num_regs = regs.size(),
            .side_effects = side_effects,
            .dma_fifoed = 0x158,
            .dma_ram = 0x157,
            .verify = verify,
            .stale_register = stale,
            .skip_same = skip_same,
            .flags = flags.data(),
            .exec_mask = 1,
            .barrier_mask = 4,
            .shadow = track ? shadow.data() : nullptr,
            .deferred_mode = &deferred_mode,
            .deferred_count = &deferred_count,
            .dropped_writes = &dropped,
            .store_class = classified ? store_class.data() : nullptr,
            .jit_data_ram = jit_data_ram,
        };
        // The classified apply: the same store as Write, and seen apart.
        const auto store = [&](u32 method, u32 value) {
            applied.emplace_back(method, value);
            Write(method, value);
        };
        const auto read = [&](u32 address) { return address < regs.size() ? regs[address] : 0U; };
        const auto write = [&](u32 method, u32 value) { return Write(method, value); };
        const auto dma = [&](u32 method, u32 value) { Dma(fifo, method, value); };
        const auto live = [&](auto& live_read, auto& emit, auto&... dram_hooks) {
            ++live_runs;
#ifdef ARCHITECTURE_x86_64
            if constexpr (sizeof...(dram_hooks) != 0) {
                // The real JIT with data RAM hooks (F3d).
                ++hooked_runs;
                Tegra::TuringMacroJit jit;
                jit.Prepare(code, generation, true);
                return jit.Execute(0, fifo, std::span<u32>{dram}, live_read, emit,
                                   dram_hooks...);
            } else {
#endif
                ((void)dram_hooks, ...);
                Tegra::TuringMacroInterpreter jit;
                jit.active_dram = interpreter.active_dram;
                jit.allow_uninitialized = true; // as the JIT
                return jit.Execute(code, 0, fifo, live_read, emit);
#ifdef ARCHITECTURE_x86_64
            }
#endif
        };
        applied.clear();
        const auto result =
            MacroMemo::Call(memo, interpreter, call, live, read, write, dma, store);
        REQUIRE(result.success);
        return writes;
    }
};

} // Anonymous namespace

TEST_CASE("V2 macro memo: key", "[video_core][v2_macro]") {
    const std::array<u32, 3> a{0, 0x21, 0x13760dc0};
    const std::array<u32, 3> b{1, 0x21, 0x13760dc0};
    CHECK(MacroMemo::Hash(0x142, a) == MacroMemo::Hash(0x142, a));
    CHECK(MacroMemo::Hash(0x142, a) != MacroMemo::Hash(0x142, b));
    CHECK(MacroMemo::Hash(0x142, a) != MacroMemo::Hash(0x143, a));
    CHECK(MacroMemo::Hash(0x142, std::span{a}.first(2)) != MacroMemo::Hash(0x142, a));
}

TEST_CASE("V2 macro memo: register reads guard the replay", "[video_core][v2_macro]") {
    Front front{state_read_code};
    front.regs[0x48c] = 3;
    const Pairs expected = front.Plain({5});
    REQUIRE(expected.size() == 3);
    CHECK(front.Memoised({5}) == expected); // recorded
    CHECK(front.memo.statistics.stores == 1);
    CHECK(front.Memoised({5}) == expected); // replayed
    CHECK(front.memo.statistics.hits == 1);
    front.regs[0x48c] = 2;
    const Pairs fewer = front.Plain({5});
    CHECK(front.Memoised({5}) == fewer); // the read no longer holds
    CHECK(front.memo.statistics.guard_failures == 1);
    CHECK(front.Memoised({6}) == front.Plain({6})); // another key
    CHECK(front.memo.statistics.misses == 2);
    front.generation = 2; // new code: everything recorded is forgotten
    CHECK(front.Memoised({5}) == fewer);
    CHECK(front.memo.statistics.misses == 3);
}

TEST_CASE("V2 macro memo: a stale register is not checked against its copy",
          "[video_core][v2_macro]") {
    // F3F5b-4: after a conditional render HLE call the front's render_enable.mode is the
    // back's; a record that read it misses, and the recording's read takes the current value.
    Front front{state_read_code};
    front.regs[0x48c] = 3;
    const Pairs expected = front.Plain({5});
    CHECK(front.Memoised({5}) == expected);
    front.stale = 0x48c;
    CHECK(front.Memoised({5}) == expected);
    CHECK(front.memo.statistics.hits == 0);
    CHECK(front.memo.statistics.guard_failures == 1);
    front.stale = 0x123; // another register: the record holds
    CHECK(front.Memoised({5}) == expected);
    CHECK(front.memo.statistics.hits == 1);
}

TEST_CASE("V2 macro memo: direct-mapped table, JIT recordings, adaptive bypass",
          "[video_core][v2_macro]") {
    // F3F5b-3 (report 7833a32: Find 5.5% + Match 1.5% of the front, 41k guard failures/s).
    SECTION("only code that can reach DREAD/DWRITE is recorded by the interpreter") {
        CHECK(MacroMemo::Memo::MayUseDataRam(Code(data_ram_code), 0));
        CHECK(!MacroMemo::Memo::MayUseDataRam(Code(state_read_code), 0));
        CHECK(!MacroMemo::Memo::MayUseDataRam(Code(render_enable_code), 0));
        // Code past the end of the uploaded RAM is unknown.
        CHECK(MacroMemo::Memo::MayUseDataRam(std::vector<u32>(6, 0), 0));
    }
    SECTION("a miss of such code is recorded by the JIT; the replay doesn't run it") {
        Front front{state_read_code};
        front.regs[0x48c] = 3;
        const Pairs expected = front.Plain({5});
        CHECK(front.Memoised({5}) == expected);
        CHECK(front.memo.statistics.jit_records == 1);
        CHECK(front.memo.statistics.stores == 1);
        CHECK(front.live_runs == 1);
        CHECK(front.Memoised({5}) == expected);
        CHECK(front.memo.statistics.hits == 1);
        CHECK(front.live_runs == 1);
    }
    SECTION("two register contexts of one key both hit") {
        Front front{state_read_code};
        front.regs[0x48c] = 3;
        const Pairs three = front.Plain({5});
        front.regs[0x48c] = 2;
        const Pairs two = front.Plain({5});
        for (int round = 0; round < 4; ++round) {
            front.regs[0x48c] = 3;
            CHECK(front.Memoised({5}) == three);
            front.regs[0x48c] = 2;
            CHECK(front.Memoised({5}) == two);
        }
        CHECK(front.memo.statistics.stores == 2); // each context recorded once
        CHECK(front.memo.statistics.guard_failures == 1);
        CHECK(front.memo.statistics.hits == 6);
        CHECK(front.memo.Size() == 2);
    }
    SECTION("a macro whose calls rarely repeat is left to the JIT for a while") {
        Front front{state_read_code};
        front.regs[0x48c] = 1;
        for (u32 call = 0; call < 2048; ++call) {
            front.Memoised({call}); // every key new: no hit
        }
        CHECK(front.memo.statistics.bypassed == 0);
        const u64 stores = front.memo.statistics.stores;
        CHECK(front.Memoised({7}) == front.Plain({7}));
        CHECK(front.memo.statistics.bypassed == 1);
        CHECK(front.memo.statistics.stores == stores); // no recording while bypassed
    }
}

TEST_CASE("V2 macro memo: data RAM reads and writes", "[video_core][v2_macro]") {
    Front front{data_ram_code};
    front.dram[0x53b] = 7;
    CHECK(front.Memoised({9}) == Pairs{{0x376, 9}});
    CHECK(front.dram[0x53b] == 9);
    // Word 0x53b now holds 9: the record's read fails, the new call writes nothing.
    CHECK(front.Memoised({9}).empty());
    CHECK(front.memo.statistics.guard_failures == 1);
    CHECK(front.Memoised({9}).empty());
    CHECK(front.memo.statistics.hits == 1);
    front.dram[0x53b] = 7;
    // The key's other way still holds the first record (F3F5b-3: two contexts per key).
    CHECK(front.Memoised({9}) == Pairs{{0x376, 9}});
    CHECK(front.dram[0x53b] == 9); // the replay's data RAM write
    CHECK(front.memo.statistics.hits == 2);
    CHECK(front.memo.statistics.stores == 2);
    front.dram[0x53b] = 7;
    CHECK(front.Memoised({9}) == Pairs{{0x376, 9}});
    CHECK(front.dram[0x53b] == 9);
    CHECK(front.memo.statistics.hits == 3);
    // Code that touches data RAM is recorded by the interpreter, never the JIT.
    CHECK(front.memo.statistics.jit_records == 0);
}

TEST_CASE("V2 macro memo: DMA words checked at their place in the replay",
          "[video_core][v2_macro]") {
    Front front{render_enable_code};
    front.memory = {1, 0, 0, 0, 0, 0};
    const Pairs taken{{0x154, 0x21}, {0x155, 0x13760dc0}, {0x158, 6}, {0x556, 1}};
    CHECK(front.Memoised({0, 0x21, 0x13760dc0}) == taken);
    CHECK(front.memo.statistics.stores == 1);
    // T-01: the DMA read grows the FIFO past its capacity; the record keeps the parameters as
    // they were, so the repeat hits (a span taken before the read stored freed memory).
    CHECK(front.Memoised({0, 0x21, 0x13760dc0}) == taken);
    CHECK(front.memo.statistics.guard_failures == 0);
    CHECK(front.memo.statistics.hits == 1);
    CHECK(front.dma_reads == 2);
    // Other words: the prefix is not written twice and the interpreter finishes the call.
    front.memory = {1, 0, 0, 0, 1, 0};
    const Pairs equal{{0x154, 0x21}, {0x155, 0x13760dc0}, {0x158, 6}, {0x556, 0}};
    CHECK(front.Memoised({0, 0x21, 0x13760dc0}) == equal);
    CHECK(front.memo.statistics.refetches == 1);
    CHECK(front.dma_reads == 3);
    CHECK(front.Memoised({0, 0x21, 0x13760dc0}) == equal); // the call stored again
    CHECK(front.memo.statistics.hits == 2);
}

TEST_CASE("V2 macro memo: reads after the call's own writes are not stored",
          "[video_core][v2_macro]") {
    Front front{poll_code};
    front.status_d00 = true; // the poll of 0xd00 ends at once
    const Pairs expected = front.Plain({0x418800, 1, 1});
    CHECK(front.Memoised({0x418800, 1, 1}) == expected);
    CHECK(front.Memoised({0x418800, 1, 1}) == expected);
    CHECK(front.memo.statistics.stores == 0);
    CHECK(front.memo.statistics.hits == 0);
    CHECK(front.memo.statistics.unmemoisable == 2);
}

TEST_CASE("V2 macro memo: verify compares the JIT with the record", "[video_core][v2_macro]") {
    Front front{state_read_code};
    front.regs[0x48c] = 2;
    const Pairs expected = front.Plain({1});
    CHECK(front.Memoised({1}, true) == expected);
    CHECK(front.Memoised({1}, true) == expected);
    CHECK(front.memo.statistics.verified == 1);
    CHECK(front.memo.statistics.verify_mismatches == 0);
}

TEST_CASE("V2 macro memo: the slot record is tried only for its key", "[video_core][v2_macro]") {
    // Fix F3e: a call with another key goes to the table without reading the last record.
    Front front{state_read_code};
    front.regs[0x48c] = 2;
    const Pairs five = front.Plain({5});
    const Pairs six = front.Plain({6});
    CHECK(front.Memoised({5}) == five);
    CHECK(front.Memoised({6}) == six);
    CHECK(front.Memoised({5}) == five); // the table's record, not the slot's
    CHECK(front.memo.statistics.slot_hits == 0);
    CHECK(front.Memoised({5}) == five); // now the slot's
    CHECK(front.memo.statistics.slot_hits == 1);
    CHECK(front.memo.statistics.hits == 2);
    CHECK(front.memo.statistics.stores == 2);
}

// 3D f64c931408c5b052 (Castle Town's constant buffer select): writes 0x8e0..0x8e3.
const std::string cb_select_code =
    "00186383:18c0014d:f1907b00 20a00460:18c40240:318c0301 20202f03:98c408c0:018c4300";

TEST_CASE("V2 macro memo: hits leave out the writes Write3D would drop",
          "[video_core][v2_macro]") {
    // Fix F3e (replaces F3d's change clock, which never engaged in game: 64-register blocks
    // were stamped by the writes between calls): decided per write, from the registers.
    Front front{cb_select_code};
    front.skip_same = true;
    const Pairs expected = front.Plain({0x194});
    REQUIRE(expected.size() >= 2);
    // The writes Write3D would send from registers `regs`, in order.
    const auto sent = [&](std::array<u32, 0x1000> regs) {
        Pairs out;
        for (const auto& [method, value] : expected) {
            if (regs[method] != value) {
                regs[method] = value;
                out.emplace_back(method, value);
            }
        }
        return out;
    };
    const auto [method, value] = expected[0];
    front.regs[method] = ~value;
    const std::vector<u32> call{0x194};
    CHECK(front.Memoised(call) == expected); // recorded: every write made
    const Pairs held = sent(front.regs);
    REQUIRE(held.size() < expected.size()); // the registers hold what it wrote
    const u64 dropped = front.dropped;
    CHECK(front.Memoised(call) == held); // hit
    CHECK(front.memo.statistics.hits == 1);
    CHECK(front.memo.statistics.noop_hits == 1);
    CHECK(front.memo.last_skipped == expected.size() - held.size());
    CHECK(front.memo.statistics.skipped_writes == front.memo.last_skipped);
    CHECK(front.dropped == dropped); // left out before the write path
    front.regs[method] = ~value; // changed by another write
    const Pairs changed = sent(front.regs);
    CHECK(!changed.empty());
    CHECK(front.Memoised(call) == changed);
    CHECK(front.regs == [&] {
        std::array<u32, 0x1000> after = front.regs;
        for (const auto& [m, v] : expected) {
            after[m] = v;
        }
        return after;
    }());
    SECTION("not without compare-on-write") {
        front.skip_same = false;
        CHECK(front.Memoised(call) == expected);
        CHECK(front.memo.last_skipped == 0);
    }
    SECTION("executable writes are always made") {
        front.flags[method] = 1;
        CHECK(front.Memoised(call) == Pairs{expected[0]}); // the others still left out
    }
    SECTION("a shadow copy that differs makes the write") {
        front.track = true;
        for (const auto& [m, v] : expected) {
            front.shadow[m] = v; // as the writes left them
        }
        CHECK(front.Memoised(call) == held);
        front.shadow[method] = ~value;
        CHECK(front.Memoised(call) == Pairs{expected[0]});
        CHECK(front.shadow[method] == value);
    }
    SECTION("a deferred draw makes every write") {
        front.deferred_mode = 1;
        front.deferred_count = 2;
        CHECK(front.Memoised(call) == expected);
        CHECK(front.memo.last_skipped == 0);
        front.deferred_count = 0; // instance mode without instances: no deferred draw
        CHECK(front.Memoised(call) == held);
    }
    SECTION("R2-F5-01: a hit applies plain state writes classified, the rest through write") {
        front.classified = true;
        front.store_class[method] = 1;
        front.regs[method] = ~value; // made again
        const Pairs again = sent(front.regs);
        REQUIRE(!again.empty());
        CHECK(front.Memoised(call) == again);
        CHECK(front.applied == Pairs{expected[0]}); // applied, not through the write path
        // Without compare-on-write every write is made: store-class ones still classified.
        front.skip_same = false;
        CHECK(front.Memoised(call) == expected);
        CHECK(front.applied == Pairs{expected[0]});
        // Verify runs the JIT: everything through write.
        CHECK(front.Memoised(call, true) == expected);
        CHECK(front.applied.empty());
    }
    SECTION("verify: every write the replay would leave out must be dropped") {
        const u64 skipped = front.memo.statistics.skipped_writes;
        const u64 dropped_before = front.dropped;
        CHECK(front.Memoised(call, true) == expected); // the JIT makes every write
        CHECK(front.memo.statistics.verify_mismatches == 0);
        const u64 predicted = front.memo.statistics.skipped_writes - skipped;
        CHECK(predicted == expected.size() - held.size());
        CHECK(front.dropped - dropped_before == predicted);
        CHECK(front.memo.last_skipped == 0); // made, so FrontRunMacro counts nothing
        front.drop_counted = false;          // a write path that keeps them
        front.Memoised(call, true);
        CHECK(front.memo.statistics.verify_mismatches == predicted);
    }
}

TEST_CASE("V2 macro memo: a shadow RAM mode write stops the left-out writes",
          "[video_core][v2_macro]") {
    Front front{cb_select_code};
    front.skip_same = true;
    const Pairs expected = front.Plain({0x194});
    REQUIRE(expected.size() >= 2);
    CHECK(front.Memoised({0x194}) == expected);
    CHECK(front.Memoised({0x194}).empty());
    front.flags[expected[0].first] = 1; // executable: made, the rest still left out
    CHECK(front.Memoised({0x194}) == Pairs{expected[0]});
    front.flags[expected[0].first] = 1 | 4; // and a barrier: every later write made
    CHECK(front.Memoised({0x194}) == expected);
    CHECK(front.memo.statistics.hits == 3);
}

#ifdef ARCHITECTURE_x86_64
TEST_CASE("V2 macro memo: the JIT records code that touches data RAM", "[video_core][v2_macro]") {
    // Fix F3d: DREAD/DWRITE through the JIT's data RAM hooks, not the interpreter.
    Front front{data_ram_code};
    front.jit_data_ram = true;
    front.dram[0x53b] = 7;
    CHECK(front.Memoised({9}) == Pairs{{0x376, 9}});
    CHECK(front.dram[0x53b] == 9);
    CHECK(front.hooked_runs == 1);
    CHECK(front.memo.statistics.jit_records == 1);
    CHECK(front.memo.statistics.stores == 1);
    CHECK(front.Memoised({9}).empty()); // the data RAM guard fails; recorded again
    CHECK(front.hooked_runs == 2);
    front.dram[0x53b] = 7;
    CHECK(front.Memoised({9}) == Pairs{{0x376, 9}}); // replayed, no run
    CHECK(front.dram[0x53b] == 9);
    CHECK(front.hooked_runs == 2);
    CHECK(front.memo.statistics.hits == 1);
}
#endif

TEST_CASE("V2 macro HLE: the conditional render macro", "[video_core][v2_macro]") {
    const std::vector<u32> code = Code(render_enable_code);
    const MacroHle::Identity identity = MacroHle::Identify(code, 0);
    CHECK(identity.hash == 0x966aeb0e43e7a13cULL);
    CHECK(identity.instructions == 9);
    const MacroHle::Entry* const entry = MacroHle::Find(0x966aeb0e43e7a13cULL);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->kind == MacroHle::Kind::RenderEnableCompare);

    const std::array<std::array<u32, 6>, 5> reports{{
        {1, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, 1, 0},
        {5, 2, 9, 9, 5, 3},
        {0, 0, 0, 0, 0, 0},
        {7, 7, 1, 2, 7, 7},
    }};
    for (const u32 invert : {0U, 1U}) {
        for (const auto& report : reports) {
            Front front{render_enable_code};
            front.memory = report;
            const std::vector<u32> parameters{invert, 0x21, 0x13760dc0};
            REQUIRE(MacroHle::Applies(*entry, parameters));
            const Pairs writes = front.Plain(parameters);
            // What the macro did is what the entry makes.
            CHECK(!MacroHle::CheckRenderEnable(*entry, parameters, writes, report).has_value());
            const bool differ = report[0] != report[4] || report[1] != report[5];
            CHECK(writes.back() == std::pair<u32, u32>{0x556, (differ ? 1U : 0U) ^ invert});
            CHECK(MacroHle::RenderEnableValue(invert != 0, report) == writes.back().second);
        }
    }
    // A wrong prediction is reported.
    const std::vector<u32> parameters{0, 0x21, 0x13760dc0};
    const std::array<u32, 6> report{1, 0, 0, 0, 0, 0};
    const Pairs wrong{{0x154, 0x21}, {0x155, 0x13760dc0}, {0x158, 6}, {0x556, 0}};
    CHECK(MacroHle::CheckRenderEnable(*entry, parameters, wrong, report).has_value());
    // Only parameter values the macro turns into a render enable mode.
    CHECK(!MacroHle::Applies(*entry, std::vector<u32>{2, 0x21, 0x13760dc0}));
    CHECK(!MacroHle::Applies(*entry, std::vector<u32>{0, 0x21}));
}

// Fix F3h (R2-F5-03): the two hot 3D macros the memo can't help (4488 MACRO_DUMP), built in as
// pure functions of their parameters and data RAM. Evaluate must match the interpreter exactly:
// writes, data RAM, and which calls it leaves to the macro (those that would fail).
const std::string cbuf_bind_code = // 96d36d962a7ff692
    "00000003:1d200000:31c28300 00000003:1d100000:31c18300 00000023:18c02001:f45c6700 "
    "00000003:18c00148:0b900300 00001203:18c00510:0ba0c300 00000003:18c00120:0810c300 "
    "00000003:002f8000:72100300 00000003:18c00000:061c7f00 00000003:18c00000:040c7f00 "
    "0000003f:18c00400:345c6700 00386383:18c000a3:058c4300 20200003:03c00040:058c4300 "
    "00000003:18c00000:f2400300 00042413:18c00318:0bac4301 00200007:18c004e0:01ac4300";
const std::string surface_bind_code = // 5b6e8dc0f6ab7d8e
    "00000003:5c110001:31c00300 00000003:1c400140:32228300 00000000:1a0a4000:70507b00 "
    "00000003:18c02400:008c7700 00001407:2220014b:31907b00 0000140f:22100140:b2218300 "
    "00000003:1d300000:31c38300 00000001:05c7c000:31c28300 00000003:18c00000:0e4c7f00 "
    "00000003:18c00000:0c3c7f00 00000000:20030000:0a2c7f00 00000003:18c02400:300c7700 "
    "00046017:18c0041e:0bac4300 20208c00:f8c40046:0f8c4300 0c206383:18c0003a:4b8c4300 "
    "01400003:18c00000:718c0300 20208c00:d8c40046:0d8c4300 00042413:18c0031f:0bac4300 "
    "00a00003:18c00004:718c0300 0004200f:18c0041e:0bac4300 00200003:18c00048:0b8c4300 "
    "00042003:18c0041e:0bac4301 00200007:18c0041e:0bac4300";

TEST_CASE("V2 macro HLE: the constant buffer and surface binds match the macros",
          "[video_core][v2_macro]") {
    struct Case {
        const std::string* code;
        u64 hash;
        MacroHle::Kind kind;
        u32 parameters;
    };
    const std::array<Case, 2> cases{{
        {&cbuf_bind_code, 0x96d36d962a7ff692ULL, MacroHle::Kind::ConstantBufferBind, 4},
        {&surface_bind_code, 0x5b6e8dc0f6ab7d8eULL, MacroHle::Kind::SurfaceBind, 5},
    }};
    std::mt19937 rng{0xf3a};
    for (const Case& test : cases) {
        const std::vector<u32> code = Code(*test.code);
        CHECK(MacroHle::Identify(code, 0).hash == test.hash);
        const MacroHle::Entry* const entry = MacroHle::Find(test.hash);
        REQUIRE(entry != nullptr);
        REQUIRE(entry->kind == test.kind);
        u32 evaluated = 0;
        u32 with_writes = 0;
        for (int round = 0; round < 4000; ++round) {
            // Values that reach every branch: small indices, equal words, mask bits, the
            // dumped calls' shapes, and out-of-range indices (the macro fails: not modelled).
            const auto pick = [&](u32 index) -> u32 {
                switch (rng() % 6) {
                case 0:
                    return rng() % 16;
                case 1:
                    return index == 0 ? 0x6e0 + rng() % 0x400 : rng();
                case 2:
                    return 0x100000 | (rng() % 0x100);
                case 3:
                    return (rng() % 8) << 24 | (rng() % 8) << 28 | (rng() & 0xfffff);
                default:
                    return rng();
                }
            };
            std::vector<u32> parameters(test.parameters);
            for (u32 i = 0; i < test.parameters; ++i) {
                parameters[i] = pick(i);
            }
            Tegra::TuringMacroInterpreter::DataRam dram{};
            for (u32& word : dram) {
                word = rng() % 4 == 0 ? rng() : rng() % 16;
            }
            if (rng() % 3 == 0 && parameters[0] < 0x100) {
                dram[(0x500 + parameters[0] * 4) % dram.size()] = parameters[1]; // unchanged
            }
            dram[0x52c] = rng() % 2 == 0 ? ~0U : rng();
            const auto before = dram;
            MacroHle::Effects predicted;
            const bool modelled = MacroHle::Evaluate(*entry, parameters, before, predicted);
            Tegra::TuringMacroInterpreter interpreter;
            interpreter.active_dram = &dram;
            Pairs writes;
            std::vector<u32> fifo = parameters;
            const auto result = interpreter.Execute(
                code, 0, fifo, [](u32) { return 0U; },
                [&](u32 method, u32 value) {
                    writes.emplace_back(method, value);
                    return true;
                });
            if (!result.success) {
                CHECK(!modelled); // the macro fails: it runs, the HLE doesn't stand for it
                continue;
            }
            REQUIRE(modelled);
            ++evaluated;
            with_writes += predicted.num_writes != 0 ? 1 : 0;
            const auto mismatch = MacroHle::CheckEffects(predicted, writes, dram);
            CHECK(!mismatch.has_value());
            if (mismatch) {
                FAIL(*mismatch);
            }
            // Data RAM: exactly the predicted words changed.
            auto expected = before;
            for (u32 i = 0; i < predicted.num_dram_writes; ++i) {
                expected[predicted.dram_writes[i].first] = predicted.dram_writes[i].second;
            }
            CHECK(dram == expected);
        }
        CHECK(evaluated > 1000);
        CHECK(with_writes > 200);
        // Another parameter count: the macro runs.
        MacroHle::Effects effects;
        Tegra::TuringMacroInterpreter::DataRam dram{};
        CHECK(!MacroHle::Evaluate(*entry, std::vector<u32>(test.parameters + 1), dram, effects));
    }
    // A wrong prediction is reported.
    MacroHle::Effects predicted;
    predicted.writes[0] = {0x8e0, 1};
    predicted.num_writes = 1;
    const Pairs made{{0x8e0, 2}};
    CHECK(MacroHle::CheckEffects(predicted, made, {}).has_value());
}

TEST_CASE("V2 macro HLE: nx2_macro_hle.txt lines", "[video_core][v2_macro]") {
    const auto entry = MacroHle::ParseLine(
        "966aeb0e43e7a13c render_enable_compare invert=p0 writes=0x154:p1,0x155:p2,0x158:6 "
        "words=6");
    REQUIRE(entry.has_value());
    CHECK(entry->kind == MacroHle::Kind::RenderEnableCompare);
    CHECK(entry->num_writes == 3);
    CHECK(entry->writes[2].method == 0x158);
    CHECK(entry->writes[2].value.Get(std::vector<u32>{}) == 6);
    CHECK(entry->invert.parameter);
    CHECK(entry->words == 6);
    CHECK(!MacroHle::ParseLine("966aeb0e43e7a13c render_enable_compare invert=p0").has_value());
    const auto draw = MacroHle::ParseLine("1234 draw_arrays_indirect topology=p0 address=p1,p2");
    REQUIRE(draw.has_value());
    CHECK(draw->kind == MacroHle::Kind::DrawArraysIndirect);
}
