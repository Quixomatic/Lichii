// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
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
#include "video_core/v2/change_clock.h"
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
    // Fix F3d: the change clock the front keeps, the table's executable bit (1) and shadow
    // RAM mode barrier (4), the context and the JIT's data RAM recordings.
    VideoCommon::V2::ChangeClock clock;
    bool with_clock = false;
    std::array<u16, 0x1000> flags{};
    u32 context = MacroMemo::NoContext;
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
        if (regs[method] != stored) {
            regs[method] = stored;
            clock.Change(method);
        }
        writes.emplace_back(method, value);
        return true;
    }
    /// A register set outside the write path, as the front's resync does.
    void Set(u32 method, u32 value) {
        regs[method] = value;
        clock.Change(method);
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
            .clock = with_clock ? &clock : nullptr,
            .flags = flags.data(),
            .exec_mask = 1,
            .barrier_mask = 4,
            .context = context,
            .jit_data_ram = jit_data_ram,
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
        const auto result = MacroMemo::Call(memo, interpreter, call, live, read, write, dma);
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
    CHECK(front.Memoised({0, 0x21, 0x13760dc0}) == taken);
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

TEST_CASE("V2 change clock: blocks of 64 registers", "[video_core][v2_macro]") {
    ChangeClock clock;
    const u64 before = clock.Now();
    clock.Change(0x48c);
    const u64 after = clock.Now();
    CHECK(!clock.Unchanged(ChangeClock::Block(0x48c), before));
    CHECK(clock.Unchanged(ChangeClock::Block(0x48c), after));
    CHECK(clock.Unchanged(ChangeClock::Block(0x674), before)); // another block
    CHECK(ChangeClock::Block(0x480) == ChangeClock::Block(0x4bf));
    CHECK(!clock.Unchanged(0, 0)); // never known to hold
    clock.Invalidate();
    CHECK(!clock.Unchanged(ChangeClock::Block(0x674), after));
    CHECK(clock.Unchanged(ChangeClock::Block(0x674), clock.Now()));
}

TEST_CASE("V2 macro memo: guards checked by the change clock", "[video_core][v2_macro]") {
    // Fix F3d: the guard walk is replaced by the blocks' change stamps.
    Front front{state_read_code};
    front.with_clock = true;
    front.Set(0x48c, 3);
    const Pairs expected = front.Plain({5});
    CHECK(front.Memoised({5}) == expected); // recorded: the guard holds from now
    CHECK(front.Memoised({5}) == expected);
    CHECK(front.memo.statistics.hits == 1);
    CHECK(front.memo.statistics.fingerprint_hits == 1);
    front.Set(0x48d, 7); // the guard's block, another register: compared, still holds
    CHECK(front.Memoised({5}) == expected);
    CHECK(front.memo.statistics.hits == 2);
    CHECK(front.memo.statistics.fingerprint_hits == 1);
    CHECK(front.Memoised({5}) == expected); // compared at the last call
    CHECK(front.memo.statistics.fingerprint_hits == 2);
    front.stale = 0x48c; // a stale register is always compared (and fails)
    CHECK(front.Memoised({5}) == expected);
    CHECK(front.memo.statistics.guard_failures == 1);
    front.stale = ~0U;
    front.Set(0x48c, 2);
    const Pairs fewer = front.Plain({5});
    CHECK(front.Memoised({5}) == fewer);
    CHECK(front.memo.statistics.guard_failures == 2);
    SECTION("verify compares the guards the clock passed") {
        CHECK(front.Memoised({5}) == fewer); // recorded under 2, holds
        front.regs[0x48c] = 9;               // set without a stamp: a missed change
        front.Memoised({5}, true);
        CHECK(front.memo.statistics.verify_mismatches == 1);
    }
}

TEST_CASE("V2 macro memo: no-op hits make only the executable writes",
          "[video_core][v2_macro]") {
    // Fix F3d: a hit whose state writes are still what their registers hold leaves them out.
    Front front{data_ram_code};
    front.with_clock = true;
    front.context = 0; // Passthrough
    front.dram[0x53b] = 7;
    CHECK(front.Memoised({9}) == Pairs{{0x376, 9}}); // record A
    CHECK(front.Memoised({9}).empty());              // record B (word 0x53b holds 9)
    front.dram[0x53b] = 7;
    CHECK(front.Memoised({9}).empty()); // A: 0x376 holds 9 since A ran
    CHECK(front.memo.statistics.noop_hits == 1);
    CHECK(front.memo.statistics.skipped_writes == 1);
    CHECK(front.memo.last_skipped == 1);
    CHECK(front.dram[0x53b] == 9); // its data RAM write still made
    CHECK(front.regs[0x376] == 9);
    front.dram[0x53b] = 7;
    front.Set(0x376, 1); // another write changed the register
    CHECK(front.Memoised({9}) == Pairs{{0x376, 9}});
    CHECK(front.memo.statistics.noop_hits == 1);
    CHECK(front.memo.statistics.hits == 2);
    SECTION("only under the context it was applied in") {
        front.dram[0x53b] = 7;
        front.context = 1;
        CHECK(front.Memoised({9}) == Pairs{{0x376, 9}});
        CHECK(front.memo.statistics.noop_hits == 1);
    }
    SECTION("verify checks the writes it would leave out") {
        front.dram[0x53b] = 7;
        front.regs[0x376] = 5; // set without a stamp
        CHECK(front.Memoised({9}, true) == Pairs{{0x376, 9}});
        CHECK(front.memo.statistics.verify_mismatches == 1);
    }
    SECTION("executable writes are always made; a barrier write disables it") {
        Front exec{data_ram_code};
        exec.with_clock = true;
        exec.context = 0;
        exec.flags[0x376] = 1;
        exec.dram[0x53b] = 7;
        CHECK(exec.Memoised({9}) == Pairs{{0x376, 9}});
        exec.dram[0x53b] = 7;
        CHECK(exec.Memoised({9}) == Pairs{{0x376, 9}});
        exec.dram[0x53b] = 7;
        CHECK(exec.Memoised({9}) == Pairs{{0x376, 9}});
        CHECK(exec.memo.statistics.noop_hits == 2); // nothing left out
        CHECK(exec.memo.statistics.skipped_writes == 0);
        Front barrier{data_ram_code};
        barrier.with_clock = true;
        barrier.context = 0;
        barrier.flags[0x376] = 4;
        barrier.dram[0x53b] = 7;
        barrier.Memoised({9});
        barrier.Memoised({9});
        barrier.dram[0x53b] = 7;
        CHECK(barrier.Memoised({9}) == Pairs{{0x376, 9}});
        CHECK(barrier.memo.statistics.hits == 1);
        CHECK(barrier.memo.statistics.noop_hits == 0);
    }
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
