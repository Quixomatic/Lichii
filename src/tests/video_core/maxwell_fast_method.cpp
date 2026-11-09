// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/gpu_thread.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/v2/draw_model.h"
#include "video_core/v2/ir_check.h"
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/v2_switches.h"

namespace VideoCommon {

struct GpuSplitTestAccess {
    static void Initialize(GpuSplit& split, Core::System& system, u64 position = 0) {
        split.system = &system;
        split.ring = std::make_unique<u64[]>(GpuSplit::RingWords);
        split.write_position = position;
        split.consumed_cache = position;
        split.published_local = position;
        split.publish_every = GpuSplit::RingWords;
    }

    static u64 End(const GpuSplit& split) {
        return split.write_position;
    }

    static void Start(GpuSplit& split, Core::System& system) {
        Initialize(split, system);
        split.active = true;
        split.back_thread = std::jthread([&split](std::stop_token stop) { split.BackLoop(stop); });
    }

    static u64 NearWrap() {
        return GpuSplit::RingWords - 1;
    }

    static void SetIr(GpuSplit& split, bool ir) {
        split.ir = ir;
        split.ir_verify = ir;
        if (ir) {
            split.AllocateIr();
        }
    }

    static const u64* Ring(const GpuSplit& split) {
        return split.ring.get();
    }

    static void SetDrawRecords(GpuSplit& split, bool verify) {
        split.front_macros = true;
        split.draw_records = true;
        split.draw_verify = verify;
        if (verify) {
            V2::IrCheck::Start();
        }
    }

    static void SetBackThread(bool back) {
        GpuSplit::on_back_thread = back;
    }

    static u64 IrBadRecords(const GpuSplit& split) {
        return split.ir_bad_records;
    }

    static u64 Execute(GpuSplit& split, u64 position, u64 available) {

        const size_t index = position & GpuSplit::RingMask;
        const size_t size = split.Execute(
            &split.ring[index], std::min<u64>(available - position, GpuSplit::RingWords - index));
        return position + (size != 0 ? size : GpuSplit::RingWords - index);
    }
};

} // namespace VideoCommon

namespace {

using Tegra::Engines::Maxwell3D;

constexpr u32 CbSize = MAXWELL3D_REG_INDEX(const_buffer.size);
constexpr u32 CbAddressHigh = MAXWELL3D_REG_INDEX(const_buffer.address_high);
constexpr u32 CbAddressLow = MAXWELL3D_REG_INDEX(const_buffer.address_low);
constexpr u32 CbOffset = MAXWELL3D_REG_INDEX(const_buffer.offset);
constexpr u32 CbData = MAXWELL3D_REG_INDEX(const_buffer.buffer);
constexpr u32 DrawBegin = MAXWELL3D_REG_INDEX(draw.begin);
constexpr u32 DrawEnd = MAXWELL3D_REG_INDEX(draw.end);
constexpr u32 ShadowControl = MAXWELL3D_REG_INDEX(shadow_ram_control);

// Executable methods without memory or macro effects a null rasterizer cannot take.
constexpr std::array<u32, 16> executable_methods{
    DrawBegin,
    DrawEnd,
    MAXWELL3D_REG_INDEX(vertex_buffer.first),
    MAXWELL3D_REG_INDEX(vertex_buffer.count),
    MAXWELL3D_REG_INDEX(index_buffer.first),
    MAXWELL3D_REG_INDEX(index_buffer.count),
    MAXWELL3D_REG_INDEX(topology_override),
    MAXWELL3D_REG_INDEX(bind_groups[0].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[1].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[2].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[3].raw_config),
    MAXWELL3D_REG_INDEX(bind_groups[4].raw_config),
    MAXWELL3D_REG_INDEX(vertex_array_instance_first),
    MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent),
    Maxwell3D::RootTables::SelectMethod,
    Maxwell3D::RootTables::LoadMethod,
};

struct Fixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Maxwell3D slow{memory};
    Maxwell3D fast{memory};
    std::vector<u32> state_methods;

    Fixture() {
        slow.fast_maxwell3d = false;
        fast.fast_maxwell3d = true;
        for (auto* engine : {&slow, &fast}) {
            engine->BindRasterizer(&rasterizer);
            VideoCommon::Dirty::SetupDirtyFlags(engine->dirty.tables);
            // Give every register a flag so that a missed or extra write shows.
            for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
                if (engine->dirty.tables[1][method] == 0) {
                    engine->dirty.tables[1][method] =
                        static_cast<u8>(VideoCommon::Dirty::LastCommonEntry + method % 61);
                }
            }
            engine->dirty.flags.reset();
            // An unmapped buffer: the upload writes nothing, the offset still advances.
            engine->CallMethod(system, CbAddressHigh, 0x1, true);
            engine->CallMethod(system, CbAddressLow, 0x2000'0000, true);
            engine->CallMethod(system, CbSize, 0x10000, true);
        }
        for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
            if (!slow.execution_mask[method] && (method < CbSize || method > CbAddressLow)) {
                state_methods.push_back(method);
            }
        }
    }

    void Call(u32 method, u32 argument, bool is_last) {
        slow.CallMethod(system, method, argument, is_last);
        fast.CallMethod(system, method, argument, is_last);
    }

    void CallMulti(u32 method, const std::vector<u32>& words) {
        const u32 amount = static_cast<u32>(words.size());
        slow.CallMultiMethod(system, method, words.data(), amount, amount);
        fast.CallMultiMethod(system, method, words.data(), amount, amount);
    }

    void Compare() const {
        REQUIRE(fast.regs.reg_array == slow.regs.reg_array);
        REQUIRE(fast.shadow_state.reg_array == slow.shadow_state.reg_array);
        REQUIRE(fast.shadow_state.shadow_ram_control == slow.shadow_state.shadow_ram_control);
        REQUIRE(fast.dirty.flags == slow.dirty.flags);
        const auto& a = fast.draw_manager.draw_state;
        const auto& b = slow.draw_manager.draw_state;
        REQUIRE(a.topology == b.topology);
        REQUIRE(a.draw_mode == b.draw_mode);
        REQUIRE(a.draw_indexed == b.draw_indexed);
        REQUIRE(a.draw_element_count == b.draw_element_count);
        REQUIRE(a.base_index == b.base_index);
        REQUIRE(a.base_instance == b.base_instance);
        REQUIRE(a.instance_count == b.instance_count);
        REQUIRE(a.drawn_instances == b.drawn_instances);
        REQUIRE(std::memcmp(&a.vertex_buffer, &b.vertex_buffer, sizeof(a.vertex_buffer)) == 0);
        REQUIRE(std::memcmp(&a.index_buffer, &b.index_buffer, sizeof(a.index_buffer)) == 0);
        for (size_t stage = 0; stage < fast.state.shader_stages.size(); ++stage) {
            for (size_t slot = 0; slot < Maxwell3D::Regs::MaxConstBuffers; ++slot) {
                const auto& x = fast.state.shader_stages[stage].const_buffers[slot];
                const auto& y = slow.state.shader_stages[stage].const_buffers[slot];
                REQUIRE(x.address == y.address);
                REQUIRE(x.size == y.size);
                REQUIRE(x.enabled == y.enabled);
            }
        }
        REQUIRE(fast.root_tables.tables == slow.root_tables.tables);
        REQUIRE(fast.root_tables.selected == slow.root_tables.selected);
        REQUIRE(fast.root_tables.offset == slow.root_tables.offset);
    }
};

u32 RandomArgument(std::mt19937& rng, u32 method, const Maxwell3D& engine) {
    // Mostly values the register already holds or small ones, as the macros write them.
    switch (rng() % 4) {
    case 0:
        return engine.regs.reg_array[method];
    case 1:
        return rng() % 4;
    default:
        break;
    }
    if (method == DrawBegin) {
        // Instance id in bits 26-27: first, subsequent (an instanced loop) or unchanged.
        return (static_cast<u32>(rng() % 3) << 26) | (rng() % 8);
    }
    if (method == Maxwell3D::RootTables::SelectMethod) {
        return (rng() % 8) | ((rng() % 64 * 4) << 8);
    }
    return rng() % 64;
}

} // Anonymous namespace

TEST_CASE("Maxwell3D fast method dispatch matches CallMethod on random method streams",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    for (const u32 seed : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        CAPTURE(seed);
        std::mt19937 rng{seed};
        for (u32 step = 0; step < 20000; ++step) {
            CAPTURE(step);
            const u32 kind = rng() % 100;
            if (kind < 45) {
                const u32 method = f.state_methods[rng() % f.state_methods.size()];
                f.Call(method, RandomArgument(rng, method, f.slow), rng() % 2 == 0);
            } else if (kind < 75) {
                const u32 method = executable_methods[rng() % executable_methods.size()];
                f.Call(method, RandomArgument(rng, method, f.slow), rng() % 2 == 0);
            } else if (kind < 85) {
                f.Call(CbData + rng() % 16, rng(), true);
            } else if (kind < 88) {
                f.Call(CbOffset, (rng() % 0x4000) * 4, true);
            } else if (kind < 92) {
                std::vector<u32> words(1 + rng() % 8);
                for (u32& word : words) {
                    word = rng() % 3 == 0 ? rng() : rng() % 4;
                }
                const u32 method = rng() % 2 == 0
                                       ? CbData + rng() % 16
                                       : f.state_methods[rng() % f.state_methods.size()];
                f.CallMulti(method, words);
            } else if (kind < 94) {
                // Track, track with filter, passthrough or replay.
                f.Call(ShadowControl, rng() % 4, true);
            } else if (kind < 97) {
                for (auto* engine : {&f.slow, &f.fast}) {
                    engine->dirty.flags.reset();
                }
            } else {
                for (auto* engine : {&f.slow, &f.fast}) {
                    engine->draw_manager.DrawDeferred(*engine);
                }
            }
            f.Compare();
        }
    }
}

TEST_CASE("Maxwell3D fast method dispatch issues deferred instanced draws as CallMethod does",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    constexpr u32 subsequent = 1u << 26;
    const auto draw = [&f](u32 begin) {
        f.Call(DrawBegin, begin, true);
        f.Call(MAXWELL3D_REG_INDEX(vertex_buffer.first), 0, true);
        f.Call(MAXWELL3D_REG_INDEX(vertex_buffer.count), 3, true);
        f.Call(DrawEnd, 0, true);
        f.Compare();
    };
    draw(0);
    draw(subsequent);
    draw(subsequent);
    REQUIRE(f.fast.draw_manager.HasDeferredDraw());
    // A constant buffer upload, a bind and state changes each end the loop.
    f.Call(CbData, 7, true);
    f.Compare();
    REQUIRE_FALSE(f.fast.draw_manager.HasDeferredDraw());
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(bind_groups[4].raw_config), 1 | (3 << 4), true);
    f.Compare();
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(topology_override), 5, true);
    f.Compare();
    draw(subsequent);
    f.Call(MAXWELL3D_REG_INDEX(global_base_instance_index), 9, true);
    f.Compare();
    REQUIRE_FALSE(f.fast.draw_manager.HasDeferredDraw());
}

TEST_CASE("Maxwell3D constant buffer word runs match single CallMethod writes",
          "[video_core][maxwell_fast_method]") {
    Fixture f;
    std::mt19937 rng{11};
    constexpr u32 subsequent = 1u << 26;
    for (u32 step = 0; step < 20000; ++step) {
        CAPTURE(step);
        switch (rng() % 8) {
        case 0:
            f.Call(ShadowControl, rng() % 4, true);
            break;
        case 1:
            // Near the end too, where words are dropped.
            f.Call(CbSize, rng() % 2 == 0 ? 0x10000 : 0x100 + rng() % 0x40, true);
            break;
        case 2:
            f.Call(CbOffset, rng() % 3 == 0 ? 0xF0 + (rng() % 8) * 4 : (rng() % 0x400) * 4, true);
            break;
        case 3:
            // An instanced loop the first word ends.
            f.Call(DrawBegin, rng() % 2 == 0 ? 0 : subsequent, true);
            f.Call(DrawEnd, 0, true);
            break;
        default: {
            const u32 count = 1 + rng() % 64;
            std::vector<u32> methods(count);
            std::vector<u32> arguments(count);
            const u32 first = CbData + rng() % 16;
            for (u32 i = 0; i < count; ++i) {
                methods[i] = rng() % 2 == 0 ? first : CbData + (first - CbData + i) % 16;
                arguments[i] = rng() % 2 == 0 ? rng() : rng() % 4;
            }
            for (u32 i = 0; i < count; ++i) {
                f.slow.CallMethod(f.system, methods[i], arguments[i], true);
            }
            f.fast.CallConstBufferWords(f.system, methods.data(), arguments.data(), count);
            break;
        }
        }
        f.Compare();
    }
}

TEST_CASE("GpuSplit CB batching respects device remaps within GPU big pages",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    auto* replacement = f.physical.GetPointerFromRaw<u8>(0x30000);
    for (u64 page = 0; page < 16; ++page) {
        table.entries.GetAndFault(4 + page)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing) - 0x4000);
    }
    table.entries.GetAndFault(0x20).Store(false, Common::PageType::Memory, 1,
                                        reinterpret_cast<uintptr_t>(replacement) - 0x20000);
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(0x10000, 0x10000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(0x10000, 0x4000, 0x10000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.device_memory.Unmap(0x11000, 0x1000);
    f.device_memory.Map(0x11000, 0x20000, 0x1000, asid);
    REQUIRE(f.memory.GetPointer(0x100ffc) == backing + 0xffc);
    REQUIRE(f.memory.GetPointer(0x101000) == replacement);

    for (auto* engine : {&f.slow, &f.fast}) {
        std::memset(backing, 0, 0x10000);
        std::memset(replacement, 0, 0x1000);
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        split.MacroWrite(engine, CbAddressHigh, 0);
        split.MacroWrite(engine, CbAddressLow, 0x100ffc);
        split.MacroWrite(engine, CbSize, 8);
        split.MacroWrite(engine, CbOffset, 0);
        split.MacroWrite(engine, CbData, 0x12345678);
        split.MacroWrite(engine, CbData + 1, 0x87654321);
        u64 position{};
        while (position < Access::End(split)) {
            position = Access::Execute(split, position, Access::End(split));
        }
        u32 first{}, second{}, stale{};
        std::memcpy(&first, backing + 0xffc, sizeof(first));
        std::memcpy(&second, replacement, sizeof(second));
        std::memcpy(&stale, backing + 0x1000, sizeof(stale));
        CAPTURE(engine == &f.fast);
        CAPTURE(stale);
        REQUIRE(first == 0x12345678);
        REQUIRE(second == 0x87654321);
        REQUIRE(stale == 0);
        REQUIRE(engine->regs.const_buffer.offset == 8);
    }
    f.Compare();
}

TEST_CASE("GpuSplit dispatch preserves mapped uploads and macro boundaries",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    constexpr GPUVAddr gpu_address = 0x100000;
    constexpr DAddr device_address = 0x10000;
    for (u64 page = 0; page < 3; ++page) {
        const VAddr virtual_address = 0x4000 + page * 0x1000;
        auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000 + page * 0x8000);
        table.entries.GetAndFault(virtual_address >> 12)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing) - virtual_address);
    }
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(device_address, 0x3000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(device_address, 0x4000, 0x3000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(gpu_address, device_address, 0x3000, Tegra::PTEKind::PITCH, false);

    struct Snapshot {
        std::array<u8, 0x3000> bytes{};
        std::array<u32, Maxwell3D::Regs::NUM_REGS> regs{};
        std::array<u32, Maxwell3D::Regs::NUM_REGS> shadow{};
        Maxwell3D::DirtyState::Flags dirty{};
        GPUVAddr segment{};
    };
    const auto snapshot = [&](const Maxwell3D& engine) {
        Snapshot result;
        f.memory.ReadBlockUnsafe(gpu_address, result.bytes.data(), result.bytes.size());
        result.regs = engine.regs.reg_array;
        result.shadow = engine.shadow_state.reg_array;
        result.dirty = engine.dirty.flags;
        result.segment = engine.current_dma_segment;
        return result;
    };

    for (const bool wrap : {false, true}) {
        CAPTURE(wrap);
        std::vector<Snapshot> expected;
        for (auto* engine : {&f.slow, &f.fast}) {
            VideoCommon::GpuSplit split;
            struct ProbeEngine final : Tegra::Engines::EngineInterface {
                u32 writes{};
                void CallMethod(Core::System&, u32, u32, bool) override {
                    ++writes;
                }
                void CallMultiMethod(Core::System&, u32, const u32*, u32 amount, u32) override {
                    writes += amount;
                }
            } probe;
            const u64 start = wrap ? Access::NearWrap() : 0;
            Access::Initialize(split, f.system, start);
            std::vector<u64> boundaries;
            std::array<u8, 0x3000> cleared{};
            f.memory.WriteBlockUnsafe(gpu_address, cleared.data(), cleared.size());
            engine->current_dma_segment = 0;
            engine->dirty.flags.reset();
            const auto state = [&](u32 method, u32 value) {
                split.Method(engine, method, value, true, nullptr);
                boundaries.push_back(Access::End(split));
            };
            const auto direct = [&](u32 method, u32 value) {
                split.MacroWrite(engine, method, value);
                boundaries.push_back(Access::End(split));
            };
            size_t observed{};
            const auto inspect = [&] {
                split.Defer([&, engine] {
                    auto actual = snapshot(*engine);
                    if (engine == &f.slow) {
                        expected.push_back(std::move(actual));
                    } else {
                        REQUIRE(observed < expected.size());
                        const auto& reference = expected[observed];
                        REQUIRE(actual.bytes == reference.bytes);
                        REQUIRE(actual.regs == reference.regs);
                        REQUIRE(actual.shadow == reference.shadow);
                        REQUIRE(actual.dirty == reference.dirty);
                        REQUIRE(actual.segment == reference.segment);
                    }
                    ++observed;
                });
                boundaries.push_back(Access::End(split));
            };
            state(ShadowControl, 0);
            state(CbAddressHigh, 0);
            state(CbAddressLow, static_cast<u32>(gpu_address));
            state(CbSize, 0x3000);
            state(CbOffset, 0xffb);
            direct(MAXWELL3D_REG_INDEX(global_base_instance_index), 17);
            inspect();
            for (u32 i = 0; i < 129; ++i) {
                if (i % 2 == 0) {
                    direct(CbData + i % 16, 0x12340000 + i);
                } else {
                    state(CbData + i % 16, 0x56780000 + i);
                }
            }
            inspect();
            state(ShadowControl, 3);
            state(CbOffset, 0x1ffd);
            for (u32 i = 0; i < 33; ++i) {
                direct(CbData + i % 16, 0xdeadbeef);
            }
            inspect();
            state(ShadowControl, 2);
            state(CbOffset, 0x1ffd);
            direct(DrawBegin, 0);
            direct(DrawEnd, 0);
            direct(DrawBegin, 1u << 26);
            direct(DrawEnd, 0);
            direct(CbData, 0x11223344);
            direct(CbData + 1, 0xaabbccdd);
            const GPUVAddr segment = 0x12345678;
            split.Method(engine, CbData + 2, 0x87654321, false, &segment);
            boundaries.push_back(Access::End(split));
            direct(CbData + 3, 0x76543210);
            split.MacroWrite(&probe, 0x10, 42);
            boundaries.push_back(Access::End(split));
            direct(CbData + 4, 0x65432100);
            const std::array<u32, 3> multi{19, 20, 21};
            split.MultiMethod(engine, CbData, multi.data(), 3, 3, &segment);
            boundaries.push_back(Access::End(split));
            inspect();
            state(CbSize, 7);
            state(CbOffset, 0);
            direct(CbData, 0x99aabbcc);
            direct(CbData + 1, 0xddeeff00);
            inspect();

            u64 position = start;
            if (wrap) {
                for (const u64 available : boundaries) {
                    while (position < available) {
                        position = Access::Execute(split, position, available);
                        REQUIRE(position <= available);
                    }
                }
            } else {
                while (position < Access::End(split)) {
                    position = Access::Execute(split, position, Access::End(split));
                    REQUIRE(position <= Access::End(split));
                }
            }
            REQUIRE(observed == expected.size());
            REQUIRE(probe.writes == 1);
        }
        f.Compare();
    }
}

TEST_CASE("GpuSplit CB batching preserves words at the GPU address space end",
          "[video_core][maxwell_fast_method][gpu_split_dispatch]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    table.entries.GetAndFault(4).Store(false, Common::PageType::Memory, 1,
                                     reinterpret_cast<uintptr_t>(backing) - 0x4000);
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    SCOPE_EXIT {
        f.device_memory.Unmap(0x10000, 0x1000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.device_memory.Map(0x10000, 0x4000, 0x1000, asid);
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(0xfffff000, 0x10000, 0x1000, Tegra::PTEKind::PITCH, false);
    for (auto* engine : {&f.slow, &f.fast}) {
        std::memset(backing, 0, 0x1000);
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        split.MacroWrite(engine, CbAddressHigh, 0);
        split.MacroWrite(engine, CbAddressLow, 0xfffffffcu);
        split.MacroWrite(engine, CbSize, 8);
        split.MacroWrite(engine, CbOffset, 0);
        split.MacroWrite(engine, CbData, 0x12345678);
        split.MacroWrite(engine, CbData + 1, 0x87654321);
        u64 position{};
        while (position < Access::End(split)) {
            position = Access::Execute(split, position, Access::End(split));
        }
        u32 word{};
        std::memcpy(&word, backing + 0xffc, sizeof(word));
        CAPTURE(engine == &f.fast);
        REQUIRE(word == 0x12345678);
        REQUIRE(engine->regs.const_buffer.offset == 8);
    }
    f.Compare();
}

TEST_CASE("Maxwell3D fast method dispatch cost", "[.][maxwell_fast_method_bench]") {
    Fixture f;
    // Per draw, as the back thread sees Ocarina's: state writes the front did not drop, a
    // constant buffer update, three binds and a draw.
    std::vector<std::pair<u32, u32>> stream;
    std::mt19937 rng{1};
    for (u32 draw = 0; draw < 4096; ++draw) {
        for (u32 i = 0; i < 12; ++i) {
            stream.emplace_back(f.state_methods[rng() % 512], rng() % 8);
        }
        stream.emplace_back(CbOffset, 0x640);
        stream.emplace_back(CbData, draw);
        stream.emplace_back(CbData + 1, 0);
        stream.emplace_back(CbData + 2, draw * 3);
        for (const u32 stage : {0u, 2u, 4u}) {
            stream.emplace_back(MAXWELL3D_REG_INDEX(bind_groups[0].raw_config) + stage * 8,
                                1 | (rng() % 8) << 4);
        }
        stream.emplace_back(DrawBegin, 0);
        stream.emplace_back(MAXWELL3D_REG_INDEX(vertex_buffer.first), rng() % 4);
        stream.emplace_back(MAXWELL3D_REG_INDEX(vertex_buffer.count), 3 + rng() % 4);
        stream.emplace_back(DrawEnd, 0);
    }
    const auto run = [&](Maxwell3D& engine) {
        const auto start = std::chrono::steady_clock::now();
        for (u32 round = 0; round < 50; ++round) {
            for (const auto& [method, argument] : stream) {
                engine.CallMethod(f.system, method, argument, true);
            }
        }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               (50.0 * static_cast<double>(stream.size()));
    };
    run(f.slow);
    run(f.fast);
    const double slow_ns = run(f.slow);
    const double fast_ns = run(f.fast);
    WARN("CallMethod: " << slow_ns << " ns per method, fast path " << fast_ns << " ns");

    // Three constant buffer words (a draw macro's c[0][0x640..0x648]) one by one or as a run.
    constexpr std::array<u32, 3> methods{CbData, CbData + 1, CbData + 2};
    constexpr std::array<u32, 3> arguments{1, 2, 3};
    const auto words = [&](bool as_run) {
        const auto start = std::chrono::steady_clock::now();
        for (u32 i = 0; i < 1'000'000; ++i) {
            f.fast.CallMethod(f.system, CbOffset, 0x640, true);
            if (as_run) {
                f.fast.CallConstBufferWords(f.system, methods.data(), arguments.data(), 3);
            } else {
                for (u32 w = 0; w < 3; ++w) {
                    f.fast.CallMethod(f.system, methods[w], arguments[w], true);
                }
            }
        }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               1'000'000.0;
    };
    words(false);
    words(true);
    const double single_ns = words(false);
    const double run_ns = words(true);
    WARN("Three constant buffer words: " << single_ns << " ns one by one, " << run_ns
                                         << " ns as a run (unmapped buffer)");
}

TEST_CASE("Full split drains preserve aliased uploads, partial coverage and queued remaps",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    Kernel::KProcess process{f.system.Kernel()};
    SCOPE_EXIT { f.system.Kernel().UnregisterKernelObject(&process); };
    auto& table = process.GetPageTable().GetImpl();
    table.Resize(20, 12);
    constexpr VAddr cpu_address = 0x4000;
    constexpr DAddr device_address = 0x10000;
    constexpr GPUVAddr gpu_address = 0x100000;
    constexpr GPUVAddr alias = 0x200000;
    auto* backing = f.physical.GetPointerFromRaw<u8>(0x10000);
    for (u64 page = 0; page < 2; ++page) {
        const VAddr address = cpu_address + page * 0x1000;
        table.entries.GetAndFault(address >> 12)
            .Store(false, Common::PageType::Memory, 1,
                   reinterpret_cast<uintptr_t>(backing + page * 0x1000) - address);
    }
    process.GetMemory().SetCurrentPageTable(process);
    const auto asid = f.device_memory.RegisterProcess(&process.GetMemory());
    f.device_memory.BindInterface(&f.rasterizer);
    f.device_memory.Map(device_address, cpu_address, 0x2000, asid);
    SCOPE_EXIT {
        f.device_memory.Unmap(device_address, 0x2000);
        f.device_memory.UnregisterProcess(asid);
    };
    f.memory.BindRasterizer(&f.rasterizer);
    f.memory.Map(gpu_address, device_address, 0x2000, Tegra::PTEKind::PITCH, false);
    f.memory.Map(alias, device_address, 0x1000, Tegra::PTEKind::PITCH, false);
    std::memset(backing, 0, 0x2000);

    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    SCOPE_EXIT { split.Stop(); };
    const auto state = [&](u32 method, u32 value) {
        split.Method(&f.fast, method, value, true, nullptr);
    };
    const auto upload = [&](GPUVAddr address, u32 size, std::vector<u32> words) {
        state(ShadowControl, 0);
        state(CbAddressHigh, static_cast<u32>(address >> 32));
        state(CbAddressLow, static_cast<u32>(address));
        state(CbSize, size);
        state(CbOffset, 0);
        split.MultiMethod(&f.fast, CbData, words.data(), static_cast<u32>(words.size()),
                          static_cast<u32>(words.size()), nullptr);
        std::ranges::fill(words, 0xdeadbeef);
    };

    upload(gpu_address, 16, {1, 2, 3, 4});
    upload(alias + 4, 4, {20, 30});
    upload(gpu_address + 0x1000, 8, {50, 60});
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    std::array<u32, 4> read{};
    f.memory.ReadBlockUnsafe(gpu_address, read.data(), sizeof(read));
    REQUIRE(read == std::array<u32, 4>{1, 20, 3, 4});
    REQUIRE(f.memory.Read<u32>(alias + 4) == 20);
    REQUIRE(f.memory.Read<u32>(gpu_address + 0x1000) == 50);

    split.Defer([&] {
        f.memory.Map(alias, device_address + 0x1000, 0x1000, Tegra::PTEKind::PITCH, false);
    });
    upload(alias, 4, {70});
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    REQUIRE(f.memory.Read<u32>(gpu_address) == 1);
    REQUIRE(f.memory.Read<u32>(gpu_address + 0x1000) == 70);
    REQUIRE(f.memory.Read<u32>(alias + 4) == 60);
}

TEST_CASE("Split consumption does not complete deferred downstream callbacks",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    Access::Start(split, f.system);
    SCOPE_EXIT { split.Stop(); };
    std::vector<std::function<void()>> deferred;
    std::array<u32, 3> output{};
    for (u32 i = 0; i < output.size(); ++i) {
        split.Defer([&, i] { deferred.emplace_back([&, i] { output[i] = i + 1; }); });
    }
    split.Drain(VideoCommon::DrainSite::MacroDmaRead);
    REQUIRE(deferred.size() == 3);
    REQUIRE(output == std::array<u32, 3>{});
    split.Run([&] {
        for (auto& operation : deferred) {
            operation();
        }
        deferred.clear();
    });
    REQUIRE(output == std::array<u32, 3>{1, 2, 3});
}

TEST_CASE("Split restart resets writer positions without replaying old releases",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    u32 released{};
    u32 release_count{};
    for (u32 session = 1; session <= 3; ++session) {
        Access::Start(split, f.system);
        SCOPE_EXIT { split.Stop(); };
        REQUIRE(Access::End(split) == 0);
        REQUIRE_FALSE(split.WaitForWriters());
        REQUIRE_FALSE(split.WaitForSync());
        split.Defer([&, session] {
            released = session;
            ++release_count;
        });
        split.NoteWriter();
        split.NoteSync();
        split.Drain(VideoCommon::DrainSite::Acquire);
        REQUIRE(released == session);
        REQUIRE(release_count == session);
        split.Stop();
        REQUIRE_FALSE(split.Active());
        REQUIRE_FALSE(split.BackRunning());
        REQUIRE(Access::End(split) == 0);
    }
}

TEST_CASE("Maxwell3D takes its macros back from the split's front once the split stops",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    constexpr u32 StartAddressPointer = MAXWELL3D_REG_INDEX(load_mme.start_address_ptr);
    constexpr u32 StartAddress = MAXWELL3D_REG_INDEX(load_mme.start_address);
    const u32 state = f.state_methods.front();
    // The front takes the macros over and tracks a state write in its shadow RAM copy.
    engine.FrontCallMethod(f.system, split, state, 0x1234, true);
    REQUIRE(engine.shadow_state.reg_array[state] != 0x1234);
    // The split stopped: the next engine method runs on this thread, not the split's back.
    engine.CallMethod(f.system, StartAddressPointer, 0, true);
    engine.CallMethod(f.system, StartAddress, 0x40, true);
    CHECK(engine.shadow_state.reg_array[state] == 0x1234);
    CHECK(engine.GetNativeMacroPositions()[0] == 0x40);
}

TEST_CASE("Maxwell3D keeps state written after the split stops over the front's shadow RAM",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    const u32 state = f.state_methods.front();
    engine.FrontCallMethod(f.system, split, state, 0x1234, true);
    // After the stop, DmaPusher sinks a state write and consumes the sink before the next
    // executable method: the handback must come first so that write wins.
    engine.method_sink.emplace_back(state, 0x5678);
    engine.ConsumeSink(f.system);
    engine.CallMethod(f.system, MAXWELL3D_REG_INDEX(load_mme.start_address_ptr), 0, true);
    CHECK(engine.shadow_state.reg_array[state] == 0x5678);
    CHECK(engine.regs.reg_array[state] == 0x5678);
}

TEST_CASE("Maxwell3D's split front uploads macro code sent as one multi-method run",
          "[video_core][gpu_split_dependencies]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    auto& engine = f.fast;
    engine.SetNativeTuringMacroMode(true);
    // GCC 13's loop unswitching dropped these words from the front's per-word loop (#500).
    constexpr u32 pointer = 0x262;
    constexpr std::array<u32, 6> code{0x740003, 0x18c00000, 0x318c0001,
                                      0x740003, 0x18c00000, 0x318c0000};
    engine.FrontCallMethod(f.system, split, MAXWELL3D_REG_INDEX(load_mme.instruction_ptr), pointer,
                           true);
    engine.FrontCallMultiMethod(f.system, split, MAXWELL3D_REG_INDEX(load_mme.instruction),
                                code.data(), code.size(), code.size());
    const auto& macros = engine.GetNativeMacroEngine();
    REQUIRE(macros.native_write_offset == pointer * 3 + code.size());
    CHECK(std::equal(code.begin(), code.end(), macros.native_code.begin() + pointer * 3));
}

namespace {

/// Sets a v2 switch for the scope.
struct V2SwitchScope {
    VideoCommon::V2::Switch name;
    VideoCommon::V2::Mode previous;

    V2SwitchScope(VideoCommon::V2::Switch name_, VideoCommon::V2::Mode mode)
        : name{name_}, previous{VideoCommon::V2::Get(name_)} {
        VideoCommon::V2::g_modes[static_cast<size_t>(name)] = mode;
    }
    ~V2SwitchScope() {
        VideoCommon::V2::g_modes[static_cast<size_t>(name)] = previous;
    }
};

} // Anonymous namespace

TEST_CASE("V2 typed IR runs the same engine calls as the word stream",
          "[video_core][gpu_split_dispatch][v2_ir]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit words;
    VideoCommon::GpuSplit typed;
    Access::Initialize(words, f.system);
    Access::Initialize(typed, f.system);
    Access::SetIr(typed, true);
    std::vector<u32> words_calls;
    std::vector<u32> typed_calls;
    u64 words_read = 0;
    u64 typed_read = 0;
    std::mt19937 rng{1234};
    // The word stream runs on f.slow, the IR on f.fast: Compare checks they end up equal.
    for (u32 round = 0; round < 20; ++round) {
        for (u32 i = 0; i < 300; ++i) {
            const u32 choice = static_cast<u32>(rng() % 16);
            if (choice < 7) {
                const u32 method = f.state_methods[rng() % f.state_methods.size()];
                const u32 argument = RandomArgument(rng, method, f.slow);
                const bool is_last = rng() % 4 != 0;
                words.Method(&f.slow, method, argument, is_last, nullptr);
                typed.Method(&f.fast, method, argument, is_last, nullptr);
            } else if (choice < 10) {
                const u32 method = executable_methods[rng() % executable_methods.size()];
                const u32 argument = RandomArgument(rng, method, f.slow);
                if (choice == 9) {
                    words.MacroWrite(&f.slow, method, argument);
                    typed.MacroWrite(&f.fast, method, argument);
                } else {
                    words.Method(&f.slow, method, argument, true, nullptr);
                    typed.Method(&f.fast, method, argument, true, nullptr);
                }
            } else if (choice < 13) {
                // Constant buffer words one by one (the back batches them) or as one run.
                const u32 count = 1 + static_cast<u32>(rng() % 20);
                std::vector<u32> data(count);
                for (auto& word : data) {
                    word = static_cast<u32>(rng());
                }
                if (choice == 12) {
                    words.MultiMethod(&f.slow, CbData, data.data(), count, count, nullptr);
                    typed.MultiMethod(&f.fast, CbData, data.data(), count, count, nullptr);
                } else {
                    for (const u32 word : data) {
                        const u32 method = CbData + static_cast<u32>(rng() % 16);
                        words.MacroWrite(&f.slow, method, word);
                        typed.MacroWrite(&f.fast, method, word);
                    }
                }
            } else if (choice < 15) {
                const u32 value = static_cast<u32>(rng());
                words.Defer([&words_calls, value] { words_calls.push_back(value); });
                typed.Defer([&typed_calls, value] { typed_calls.push_back(value); });
            } else {
                // A typed record the back only reads, between the others.
                const u32 value = static_cast<u32>(rng());
                const VideoCommon::V2::Ir::CopyBody body{.source = value, .destination = value};
                typed.Note(VideoCommon::V2::Ir::Kind::Copy, body);
                words.Defer([&words_calls, value] { words_calls.push_back(~value); },
                            VideoCommon::GpuSplit::FastDrawNeutral);
                typed.Defer([&typed_calls, value] { typed_calls.push_back(~value); },
                            VideoCommon::GpuSplit::FastDrawNeutral);
            }
        }
        // Execute what both sent, as the back would (a publish closes the open StateRun).
        words.Publish();
        typed.Publish();
        while (words_read != Access::End(words)) {
            words_read = Access::Execute(words, words_read, Access::End(words));
        }
        while (typed_read != Access::End(typed)) {
            typed_read = Access::Execute(typed, typed_read, Access::End(typed));
        }
        f.slow.ConsumeSink(f.system);
        f.fast.ConsumeSink(f.system);
        f.Compare();
        REQUIRE(words_calls == typed_calls);
        REQUIRE(Access::IrBadRecords(typed) == 0);
        REQUIRE(typed.Mark().sequence != 0);
        REQUIRE(words.Mark().sequence == 0);
    }
}

TEST_CASE("V2 table-driven 3D front mirror sends what the hand-written one sends",
          "[video_core][gpu_split_dependencies][v2_regsem]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    using VideoCommon::V2::Mode;
    using VideoCommon::V2::Switch;
    Fixture f;
    Maxwell3D table_engine{f.memory};
    Maxwell3D verify_engine{f.memory};
    VideoCommon::GpuSplit hand;
    VideoCommon::GpuSplit table;
    VideoCommon::GpuSplit verify;
    Access::Initialize(hand, f.system);
    Access::Initialize(table, f.system);
    Access::Initialize(verify, f.system);
    std::mt19937 rng{99};
    const auto call = [&](Maxwell3D& engine, VideoCommon::GpuSplit& split, Mode mode, u32 method,
                          u32 argument, bool is_last) {
        const V2SwitchScope scope{Switch::RegSem, mode};
        engine.FrontCallMethod(f.system, split, method, argument, is_last);
    };
    const auto multi = [&](Maxwell3D& engine, VideoCommon::GpuSplit& split, Mode mode, u32 method,
                           const std::vector<u32>& data) {
        const V2SwitchScope scope{Switch::RegSem, mode};
        const u32 amount = static_cast<u32>(data.size());
        engine.FrontCallMultiMethod(f.system, split, method, data.data(), amount, amount);
    };
    constexpr u32 StartAddressPointer = MAXWELL3D_REG_INDEX(load_mme.start_address_ptr);
    constexpr u32 StartAddress = MAXWELL3D_REG_INDEX(load_mme.start_address);
    constexpr u32 Falcon4 = MAXWELL3D_REG_INDEX(falcon[4]);
    for (u32 i = 0; i < 20000; ++i) {
        const u32 choice = static_cast<u32>(rng() % 20);
        u32 method;
        u32 argument;
        if (choice < 10) {
            method = f.state_methods[rng() % f.state_methods.size()];
            argument = rng() % 3 == 0 ? static_cast<u32>(rng()) : static_cast<u32>(rng() % 3);
        } else if (choice < 14) {
            method = executable_methods[rng() % executable_methods.size()];
            argument = RandomArgument(rng, method, f.fast);
        } else if (choice == 14) {
            method = ShadowControl;
            argument = static_cast<u32>(rng() % 4);
        } else if (choice == 15) {
            method = rng() % 2 ? StartAddressPointer : StartAddress;
            argument = static_cast<u32>(rng() % 8);
        } else if (choice == 16) {
            method = rng() % 2 ? Falcon4 : CbData + static_cast<u32>(rng() % 16);
            argument = static_cast<u32>(rng());
        } else {
            method = rng() % 2 ? CbData : f.state_methods[rng() % f.state_methods.size()];
            std::vector<u32> data(1 + rng() % 6);
            for (auto& word : data) {
                word = static_cast<u32>(rng() % 3);
            }
            multi(f.fast, hand, Mode::Off, method, data);
            multi(table_engine, table, Mode::On, method, data);
            multi(verify_engine, verify, Mode::Verify, method, data);
            continue;
        }
        const bool is_last = rng() % 4 != 0;
        call(f.fast, hand, Mode::Off, method, argument, is_last);
        call(table_engine, table, Mode::On, method, argument, is_last);
        call(verify_engine, verify, Mode::Verify, method, argument, is_last);
    }
    // The same ring entries: the same writes dropped and sent, with the same values.
    REQUIRE(Access::End(hand) == Access::End(table));
    REQUIRE(Access::End(hand) == Access::End(verify));
    const auto words = static_cast<size_t>(Access::End(hand));
    for (size_t i = 0; i < words; ++i) {
        const u64 a = Access::Ring(hand)[i];
        if ((a & 0xf) == 1) { // Engine entries hold each ring's own engine pointer
            ++i;
            continue;
        }
        REQUIRE(a == Access::Ring(table)[i]);
        REQUIRE(a == Access::Ring(verify)[i]);
    }
    CHECK(std::equal(f.fast.GetNativeMacroPositions().begin(),
                     f.fast.GetNativeMacroPositions().end(),
                     table_engine.GetNativeMacroPositions().begin()));
}

TEST_CASE("V2 register groups: compare-on-write and incremental fingerprints",
          "[video_core][v2_regsem]") {
    namespace RegSem = VideoCommon::V2::RegSem;
    using VideoCommon::V2::GroupBit;
    using VideoCommon::V2::RegGroup;
    Fixture f; // Constructs a Maxwell3D, which builds the table.
    const RegSem::Table3D& table = RegSem::GetTable3D();
    auto mirror = std::make_unique<RegSem::Mirror3D>();
    mirror->skip_same = true;
    RegSem::Rehash3D(mirror->v2_groups, mirror->regs.data());
    mirror->v2_groups.TakeDirty();
    const auto none = [](RegSem::Effect, u32, u32) { return true; };
    const u32 state = MAXWELL3D_REG_INDEX(viewport_transform);
    const u32 viewport = GroupBit(RegGroup::Viewport);
    REQUIRE(table.groups[state] == RegGroup::Viewport);
    REQUIRE(table.const_buffer_offset == CbOffset);
    REQUIRE(table.instruction_ptr == MAXWELL3D_REG_INDEX(load_mme.instruction_ptr));
    REQUIRE((table.flags[DrawEnd] & (RegSem::LoopSkippable | RegSem::DrawEnd)) ==
            (RegSem::LoopSkippable | RegSem::DrawEnd));
    REQUIRE((table.flags[MAXWELL3D_REG_INDEX(bind_groups[2].raw_config)] &
             RegSem::CbufBindFlag) != 0);
    // A changed write dirties its group and is sent; the same value again does neither.
    REQUIRE(RegSem::Write3D(*mirror, table, state, 5, false, none));
    REQUIRE(mirror->v2_groups.TakeDirty() == viewport);
    const u64 all = mirror->v2_groups.Fingerprint(VideoCommon::V2::HashedGroups);
    REQUIRE_FALSE(RegSem::Write3D(*mirror, table, state, 5, false, none));
    REQUIRE(mirror->v2_groups.TakeDirty() == 0);
    REQUIRE(mirror->v2_groups.Fingerprint(VideoCommon::V2::HashedGroups) == all);
    // Writing the old value back restores the old fingerprint.
    const u64 before = mirror->v2_groups.Fingerprint(viewport);
    REQUIRE(RegSem::Write3D(*mirror, table, state, 6, false, none));
    REQUIRE(mirror->v2_groups.Fingerprint(viewport) != before);
    REQUIRE(RegSem::Write3D(*mirror, table, state, 5, false, none));
    REQUIRE(mirror->v2_groups.Fingerprint(viewport) == before);
    mirror->v2_groups.TakeDirty();
    // Constant buffer data dirties CbufData whatever the words; it is always sent.
    REQUIRE(RegSem::Write3D(*mirror, table, CbData, 0, false, none));
    REQUIRE(mirror->v2_groups.TakeDirty() == GroupBit(RegGroup::CbufData));
    // Random writes: the incremental hashes equal the ones computed from scratch.
    std::mt19937 rng{7};
    for (u32 i = 0; i < 100000; ++i) {
        const u32 method = f.state_methods[rng() % f.state_methods.size()];
        RegSem::Write3D(*mirror, table, method, static_cast<u32>(rng() % 4), false, none);
    }
    VideoCommon::V2::GroupTracker check = mirror->v2_groups;
    REQUIRE(RegSem::Rehash3D(check, mirror->regs.data()) == 0);
}

TEST_CASE("V2 typed records run on the back without thunks, as their deferred calls do",
          "[video_core][gpu_split_dispatch][v2_ir]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    namespace Ir = VideoCommon::V2::Ir;
    Fixture f;
    f.system.Initialize();
    f.memory.BindRasterizer(&f.rasterizer);
    for (const bool ir : {false, true}) {
        VideoCommon::GpuSplit split;
        Access::SetIr(split, ir);
        Access::Start(split, f.system);
        SCOPE_EXIT { split.Stop(); };
        // Barriers on the rasterizer (WFI, invalidations, fences, reference).
        for (const auto op : {Ir::BarrierOp::WaitForIdle, Ir::BarrierOp::InvalidateCaches,
                              Ir::BarrierOp::ReleaseFences, Ir::BarrierOp::SignalReference}) {
            const Ir::BarrierBody body{.op = op, .object = split.Ref(&f.rasterizer), .a = 0, .b = 0};
            split.Emit(Ir::Kind::Barrier, body, [] {});
        }
        // A readback that sets an engine's dirty flag from the caches (none dirty here).
        f.fast.current_dirty = true;
        const Ir::ReadbackBody dirty{.op = Ir::ReadbackOp::SegmentDirty,
                                     .object = split.Ref(&f.fast),
                                     .object2 = split.Ref(&f.memory),
                                     .cache = 0,
                                     .address = 0x2000'0000,
                                     .bytes = 64};
        split.Emit(Ir::Kind::Readback, dirty, [engine = &f.fast, memory = &f.memory] {
            engine->current_dirty = memory->IsMemoryDirty(0x2000'0000, 64);
        });
        // Command list pages in the payload arena.
        const std::array<u64, 3> pages{0x1000, 0x2000, 0x3000};
        const Ir::ReadbackBody list{.op = Ir::ReadbackOp::ListPages,
                                    .object = split.Ref(&f.rasterizer)};
        split.EmitPayload(Ir::Kind::Readback, list, pages.data(), pages.size(), [] {});
        // The CPU queue's command fence.
        auto state_storage = std::make_unique<VideoCommon::GPUThread::SynchState>();
        auto& state = *state_storage;
        const Ir::SyncPointBody signal{.id = 0,
                                       .value = 0,
                                       .op = Ir::SyncPointOp::CommandSignal,
                                       .object = split.Ref(&state),
                                       .a = 77,
                                       .b = 0};
        split.Emit(Ir::Kind::SyncPoint, signal, [&state] {
            VideoCommon::GPUThread::SignalCommandFence(state, 77, false);
        });
        // A record the back only reads.
        split.Note(Ir::Kind::Acquire, Ir::AcquireBody{.address = 0x40, .value = 1});
        split.Drain(VideoCommon::DrainSite::Other);
        CHECK(!f.fast.current_dirty);
        CHECK(state.signaled_fence.load() == 77);
        CHECK(Access::IrBadRecords(split) == 0);
    }
    // Map records: what the GPU runs for a map and an unmap.
    const Ir::MapBody map{.op = Ir::MapOp::GmmuMap,
                          .flags = 0,
                          .gpu_address = 0x4000'0000,
                          .device_address = 0x30'0000,
                          .size = 0x10000,
                          .object = reinterpret_cast<u64>(&f.memory)};
    f.gpu.ExecuteMap(map);
    REQUIRE(f.memory.GpuToCpuAddress(0x4000'0000 + 0x1234) == DAddr{0x30'1234});
    Ir::MapBody unmap = map;
    unmap.op = Ir::MapOp::GmmuUnmap;
    f.gpu.ExecuteMap(unmap);
    REQUIRE(!f.memory.GpuToCpuAddress(0x4000'0000 + 0x1234));
}

TEST_CASE("V2 front Draw records predict every draw the back performs",
          "[video_core][gpu_split_dispatch][v2_ir]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    using VideoCommon::V2::Mode;
    using VideoCommon::V2::Switch;
    namespace IrCheck = VideoCommon::V2::IrCheck;
    constexpr std::array<u32, 20> draw_methods{
        DrawBegin,
        DrawEnd,
        MAXWELL3D_REG_INDEX(vertex_buffer.first),
        MAXWELL3D_REG_INDEX(vertex_buffer.count),
        MAXWELL3D_REG_INDEX(index_buffer.first),
        MAXWELL3D_REG_INDEX(index_buffer.count),
        MAXWELL3D_REG_INDEX(index_buffer32_first),
        MAXWELL3D_REG_INDEX(index_buffer16_subsequent),
        MAXWELL3D_REG_INDEX(draw_inline_index),
        MAXWELL3D_REG_INDEX(inline_index_2x16.even),
        MAXWELL3D_REG_INDEX(vertex_array_instance_first),
        MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent),
        MAXWELL3D_REG_INDEX(global_base_vertex_index),
        MAXWELL3D_REG_INDEX(global_base_instance_index),
        MAXWELL3D_REG_INDEX(clear_surface),
        MAXWELL3D_REG_INDEX(draw_texture.src_y0),
        MAXWELL3D_REG_INDEX(topology_override),
        MAXWELL3D_REG_INDEX(primitive_topology_control),
        MAXWELL3D_REG_INDEX(bind_groups[0].raw_config),
        ShadowControl,
    };
    for (const Mode regsem : {Mode::Off, Mode::On, Mode::Verify}) {
        const V2SwitchScope scope{Switch::RegSem, regsem};
        Fixture f;
        f.system.Initialize();
        VideoCommon::GpuSplit split;
        Access::Initialize(split, f.system);
        Access::SetIr(split, true);
        Access::SetDrawRecords(split, true);
        auto& engine = f.fast;
        engine.SetNativeTuringMacroMode(true);
        std::mt19937 rng{static_cast<u32>(regsem) + 21};
        u64 read = 0;
        for (u32 round = 0; round < 40; ++round) {
            for (u32 i = 0; i < 400; ++i) {
                const u32 choice = static_cast<u32>(rng() % 10);
                if (choice < 3) {
                    const u32 method = f.state_methods[rng() % f.state_methods.size()];
                    engine.FrontCallMethod(f.system, split, method, rng() % 3 == 0 ? rng() : rng() % 4,
                                           true);
                    continue;
                }
                const u32 method = draw_methods[rng() % draw_methods.size()];
                u32 argument = static_cast<u32>(rng() % 8);
                if (method == DrawBegin) {
                    argument = (static_cast<u32>(rng() % 3) << 26) | (rng() % 8);
                } else if (method == ShadowControl) {
                    argument = rng() % 8 == 0 ? static_cast<u32>(rng() % 4) : 0;
                } else if (method == MAXWELL3D_REG_INDEX(index_buffer32_first) ||
                           method == MAXWELL3D_REG_INDEX(index_buffer16_subsequent) ||
                           method == MAXWELL3D_REG_INDEX(vertex_array_instance_first) ||
                           method == MAXWELL3D_REG_INDEX(vertex_array_instance_subsequent)) {
                    argument = static_cast<u32>(rng());
                } else if (method == MAXWELL3D_REG_INDEX(primitive_topology_control)) {
                    argument = rng() % 2;
                }
                if (choice == 9) {
                    std::vector<u32> words(1 + rng() % 4);
                    for (auto& word : words) {
                        word = method == DrawBegin ? argument : static_cast<u32>(rng() % 8);
                    }
                    const u32 amount = static_cast<u32>(words.size());
                    engine.FrontCallMultiMethod(f.system, split, method, words.data(), amount,
                                                amount);
                } else {
                    engine.FrontCallMethod(f.system, split, method, argument, rng() % 4 != 0);
                }
            }
            // The back runs what the front sent.
            split.Publish();
            Access::SetBackThread(true);
            while (read != Access::End(split)) {
                read = Access::Execute(split, read, Access::End(split));
            }
            Access::SetBackThread(false);
        }
        Access::SetBackThread(true);
        engine.ConsumeSink(f.system);
        Access::SetBackThread(false);
        const IrCheck::Counters counters = IrCheck::GetCounters();
        INFO("regsem mode " << static_cast<u32>(regsem));
        CHECK(counters.draws_matched > 1000);
        CHECK(counters.draws_mismatched == 0);
        CHECK(counters.draws_unexpected == 0);
        CHECK(counters.draws_missing == 0);
        CHECK(IrCheck::PendingDraws() == 0);
        if (regsem != Mode::Off) {
            CHECK(counters.fingerprints_compared > 1000);
            CHECK(counters.fingerprints_mismatched == 0);
        }
        CHECK(Access::IrBadRecords(split) == 0);
        IrCheck::Stop();
    }
}

// Fix pass 1 (Ocarina of Time, "dispatches N in 0 windows, N unexpected"): the back's executor
// must open the dispatch window for a Dispatch record it runs (only the never-called
// IrExec::IsInformational did).
TEST_CASE("V2 IR dispatch records open the check's window on the back",
          "[video_core][v2_ir][maxwell_fast_method]") {
    namespace Ir = VideoCommon::V2::Ir;
    namespace IrCheck = VideoCommon::V2::IrCheck;
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    Access::SetIr(split, true);
    Access::SetDrawRecords(split, true);
    Ir::DispatchBody body{};
    body.method = 0x107;
    body.min_dispatches = 1;
    body.max_dispatches = ~0U;
    split.Note(Ir::Kind::Dispatch, body);
    split.Publish();
    Access::SetBackThread(true);
    u64 read = 0;
    while (read != Access::End(split)) {
        read = Access::Execute(split, read, Access::End(split));
    }
    IrCheck::NoteDispatch(0x1000); // the launch the record predicted
    Access::SetBackThread(false);
    const IrCheck::Counters counters = IrCheck::GetCounters();
    CHECK(counters.dispatch_windows == 1);
    CHECK(counters.dispatches == 1);
    CHECK(counters.dispatches_unexpected == 0);
    CHECK(Access::IrBadRecords(split) == 0);
    IrCheck::Stop();
}

TEST_CASE("V2 draw trigger table and the back's draw/dispatch checks",
          "[video_core][v2_ir]") {
    namespace Ir = VideoCommon::V2::Ir;
    namespace IrCheck = VideoCommon::V2::IrCheck;
    using VideoCommon::V2::DrawModel;
    Fixture f; // Builds the register table.
    const auto& table = VideoCommon::V2::RegSem::GetTable3D();
    // Every method the model acts on is executable and handled by the draw manager, and every
    // draw manager method that draws is in the model.
    u32 draw_methods = 0;
    for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
        if (DrawModel::IsDrawMethod(method)) {
            ++draw_methods;
            REQUIRE((table.flags[method] & VideoCommon::V2::RegSem::Exec) != 0);
            REQUIRE((table.flags[method] & VideoCommon::V2::RegSem::DrawManager) != 0);
        }
    }
    REQUIRE(draw_methods == 16);
    REQUIRE(DrawModel::IsDrawMethod(DrawEnd));
    REQUIRE(DrawModel::IsDrawMethod(MAXWELL3D_REG_INDEX(clear_surface)));
    REQUIRE(DrawModel::IsInstanceLoopMethod(DrawBegin));
    REQUIRE(!DrawModel::IsInstanceLoopMethod(MAXWELL3D_REG_INDEX(clear_surface)));
    // Draw checks: in order, parameters compared, unexpected and missing draws counted.
    IrCheck::Start();
    Ir::DrawBody body{};
    body.num_draws = 2;
    body.draws[0] = {.kind = Ir::DrawKind::Array, .first = 1, .count = 3, .instances = 1};
    body.draws[1] = {.kind = Ir::DrawKind::Clear, .first = 1};
    IrCheck::OnRecord(Ir::Kind::Draw);
    IrCheck::NoteDrawRecord(body);
    IrCheck::NoteDraw(body.draws[0], nullptr);
    IrCheck::NoteDraw({.kind = Ir::DrawKind::Clear, .first = 2}, nullptr); // a different clear
    IrCheck::NoteDraw(body.draws[0], nullptr);                             // nothing predicted
    IrCheck::NoteDrawRecord(body);
    for (u32 i = 0; i < 300; ++i) {
        IrCheck::OnRecord(Ir::Kind::StateRun);
    }
    auto counters = IrCheck::GetCounters();
    CHECK(counters.draws_matched == 1);
    CHECK(counters.draws_mismatched == 1);
    CHECK(counters.draws_unexpected == 1);
    CHECK(counters.draws_missing == 2);
    // Dispatch windows: a record covers the method after it; engine records don't count.
    Ir::DispatchBody dispatch{};
    dispatch.min_dispatches = 1;
    dispatch.max_dispatches = ~0U;
    IrCheck::OnRecord(Ir::Kind::Dispatch);
    IrCheck::NoteDispatchRecord(dispatch);
    IrCheck::OnRecord(Ir::Kind::Engine);
    IrCheck::OnRecord(Ir::Kind::StateRun);
    IrCheck::NoteDispatch(0x1000);
    IrCheck::NoteDispatch(0x2000);
    IrCheck::OnRecord(Ir::Kind::StateRun); // closes the window: 2 dispatches, fine
    IrCheck::NoteDispatch(0x3000);         // outside any window
    IrCheck::OnRecord(Ir::Kind::Dispatch);
    IrCheck::NoteDispatchRecord(dispatch);
    IrCheck::OnRecord(Ir::Kind::StateRun);
    IrCheck::OnRecord(Ir::Kind::StateRun); // closes it with none: below the minimum
    counters = IrCheck::GetCounters();
    CHECK(counters.dispatches == 3);
    CHECK(counters.dispatch_windows == 2);
    CHECK(counters.dispatches_unexpected == 1);
    CHECK(counters.dispatch_count_mismatches == 1);
    IrCheck::Stop();
}

namespace {

void SetupEngine(Fixture& f, Maxwell3D& engine, bool fast) {
    engine.fast_maxwell3d = fast;
    engine.BindRasterizer(&f.rasterizer);
    VideoCommon::Dirty::SetupDirtyFlags(engine.dirty.tables);
    for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
        if (engine.dirty.tables[1][method] == 0) {
            engine.dirty.tables[1][method] =
                static_cast<u8>(VideoCommon::Dirty::LastCommonEntry + method % 61);
        }
    }
    engine.dirty.flags.reset();
    engine.CallMethod(f.system, CbAddressHigh, 0x1, true);
    engine.CallMethod(f.system, CbAddressLow, 0x2000'0000, true);
    engine.CallMethod(f.system, CbSize, 0x10000, true);
}

void CompareEngines(const Maxwell3D& a, const Maxwell3D& b) {
    REQUIRE(a.regs.reg_array == b.regs.reg_array);
    REQUIRE(a.shadow_state.reg_array == b.shadow_state.reg_array);
    REQUIRE(a.dirty.flags == b.dirty.flags);
    const auto& x = a.draw_manager.draw_state;
    const auto& y = b.draw_manager.draw_state;
    REQUIRE(x.topology == y.topology);
    REQUIRE(x.draw_mode == y.draw_mode);
    REQUIRE(x.draw_indexed == y.draw_indexed);
    REQUIRE(x.instance_count == y.instance_count);
    REQUIRE(x.drawn_instances == y.drawn_instances);
    REQUIRE(x.base_index == y.base_index);
    REQUIRE(x.base_instance == y.base_instance);
    for (size_t stage = 0; stage < a.state.shader_stages.size(); ++stage) {
        for (size_t slot = 0; slot < Maxwell3D::Regs::MaxConstBuffers; ++slot) {
            REQUIRE(a.state.shader_stages[stage].const_buffers[slot].address ==
                    b.state.shader_stages[stage].const_buffers[slot].address);
        }
    }
    REQUIRE(a.root_tables.tables == b.root_tables.tables);
    REQUIRE(a.root_tables.offset == b.root_tables.offset);
}

/// A recorded random stream of what the back's engines receive: method writes (fast and macro
/// paths), multi-method runs and sink consumption.
struct Step {
    u32 kind; ///< 0 call, 1 multi, 2 sink write, 3 consume sink
    u32 method;
    std::vector<u32> words;
    bool is_last;
};

std::vector<Step> RecordStream(Fixture& f, u32 seed, u32 count) {
    std::mt19937 rng{seed};
    std::vector<Step> steps;
    const std::array<u32, 6> specials{ShadowControl, MAXWELL3D_REG_INDEX(falcon[4]),
                                      MAXWELL3D_REG_INDEX(load_mme.start_address),
                                      MAXWELL3D_REG_INDEX(load_mme.start_address_ptr),
                                      CbData, CbData + 3};
    for (u32 i = 0; i < count; ++i) {
        const u32 choice = static_cast<u32>(rng() % 20);
        Step step{};
        step.is_last = rng() % 4 != 0;
        if (choice < 8) {
            step.kind = 0;
            step.method = f.state_methods[rng() % f.state_methods.size()];
            step.words = {rng() % 3 == 0 ? static_cast<u32>(rng()) : static_cast<u32>(rng() % 3)};
        } else if (choice < 13) {
            step.kind = 0;
            step.method = executable_methods[rng() % executable_methods.size()];
            step.words = {RandomArgument(rng, step.method, f.slow)};
        } else if (choice < 15) {
            step.kind = 0;
            step.method = specials[rng() % specials.size()];
            step.words = {step.method == ShadowControl ? static_cast<u32>(rng() % 4)
                                                       : static_cast<u32>(rng() % 8)};
        } else if (choice < 17) {
            step.kind = 1;
            step.method = rng() % 2 ? CbData : f.state_methods[rng() % f.state_methods.size()];
            step.words.resize(1 + rng() % 6);
            for (auto& word : step.words) {
                word = static_cast<u32>(rng() % 3);
            }
        } else if (choice < 19) {
            step.kind = 2;
            step.method = f.state_methods[rng() % f.state_methods.size()];
            step.words = {static_cast<u32>(rng() % 3)};
        } else {
            step.kind = 3;
        }
        steps.push_back(std::move(step));
    }
    return steps;
}

void Play(Fixture& f, Maxwell3D& engine, const Step& step) {
    switch (step.kind) {
    case 0:
        engine.CallMethod(f.system, step.method, step.words[0], step.is_last);
        break;
    case 1:
        engine.CallMultiMethod(f.system, step.method, step.words.data(),
                               static_cast<u32>(step.words.size()),
                               static_cast<u32>(step.words.size()));
        break;
    case 2:
        engine.method_sink.emplace_back(step.method, step.words[0]);
        break;
    default:
        engine.ConsumeSink(f.system);
        break;
    }
}

} // Anonymous namespace

TEST_CASE("V2 back table path matches the engine's register path on recorded streams",
          "[video_core][v2_regsem]") {
    using VideoCommon::V2::Mode;
    using VideoCommon::V2::Switch;
    Fixture f;
    for (const bool fast : {true, false}) {
        const auto steps = RecordStream(f, fast ? 3 : 4, 30000);
        auto legacy = std::make_unique<Maxwell3D>(f.memory);
        std::unique_ptr<Maxwell3D> table;
        std::unique_ptr<Maxwell3D> verify;
        {
            const V2SwitchScope scope{Switch::RegSem, Mode::On};
            table = std::make_unique<Maxwell3D>(f.memory);
        }
        {
            const V2SwitchScope scope{Switch::RegSem, Mode::Verify};
            verify = std::make_unique<Maxwell3D>(f.memory);
        }
        for (auto* engine : {legacy.get(), table.get(), verify.get()}) {
            SetupEngine(f, *engine, fast);
        }
        for (size_t i = 0; i < steps.size(); ++i) {
            Play(f, *legacy, steps[i]);
            {
                const V2SwitchScope scope{Switch::RegSem, Mode::On};
                Play(f, *table, steps[i]);
            }
            {
                const V2SwitchScope scope{Switch::RegSem, Mode::Verify};
                Play(f, *verify, steps[i]);
            }
            if ((i & 1023) == 0) {
                CompareEngines(*legacy, *table);
                CompareEngines(*legacy, *verify);
            }
        }
        CompareEngines(*legacy, *table);
        CompareEngines(*legacy, *verify);
        CHECK(verify->V2BackMismatches() == 0);
    }
    // The compute table gives the engine's plain stores.
    Tegra::Engines::KeplerCompute compute{f.memory};
    CHECK(compute.NativePlainMethods() == VideoCommon::V2::RegSem::TableComputePlain());
}

TEST_CASE("V2 back table path cost per write", "[.][v2_regsem_bench]") {
    using VideoCommon::V2::Mode;
    using VideoCommon::V2::Switch;
    Fixture f;
    const auto steps = RecordStream(f, 9, 200000);
    std::vector<Step> calls;
    for (const auto& step : steps) {
        if (step.kind == 0) {
            calls.push_back(step);
        }
    }
    // The engine's path against the table path, without the group trackers (the global switch
    // is off while the writes run; the engine took its back mode when constructed).
    const auto run = [&](Mode mode) {
        std::unique_ptr<Maxwell3D> engine;
        {
            const V2SwitchScope scope{Switch::RegSem, mode};
            engine = std::make_unique<Maxwell3D>(f.memory);
        }
        SetupEngine(f, *engine, true);
        const auto start = std::chrono::steady_clock::now();
        for (u32 pass = 0; pass < 20; ++pass) {
            for (const auto& step : calls) {
                engine->CallMethod(f.system, step.method, step.words[0], step.is_last);
            }
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        return std::chrono::duration<double, std::nano>(elapsed).count() /
               (20.0 * static_cast<double>(calls.size()));
    };
    run(Mode::Off); // warm up
    const double legacy = run(Mode::Off);
    const double table = run(Mode::On);
    const double legacy2 = run(Mode::Off);
    const double table2 = run(Mode::On);
    WARN("V2 back register path: engine " << legacy << " / " << legacy2 << " ns per write, table "
                                          << table << " / " << table2 << " ns per write");
}

TEST_CASE("V2 IR: a write noted for the hazard table ends its StateRun at the noted position",
          "[video_core][gpu_split_dispatch][v2_ir]") {
    using Access = VideoCommon::GpuSplitTestAccess;
    Fixture f;
    f.system.Initialize();
    VideoCommon::GpuSplit split;
    Access::Initialize(split, f.system);
    Access::SetIr(split, true);
    constexpr u32 first = MAXWELL3D_REG_INDEX(vertex_buffer.first);
    constexpr u32 count = MAXWELL3D_REG_INDEX(vertex_buffer.count);
    split.Method(&f.fast, first, 11, true, nullptr);
    split.Method(&f.fast, count, 12, true, nullptr);
    // P2 notes a write at the position after the method that made it.
    const u32 payload = 7;
    split.NoteKnownWrite(0x1000, &payload, sizeof(payload), VideoCommon::V2::Producer::Release);
    const u64 noted = split.WritePosition();
    split.Method(&f.fast, first, 21, true, nullptr);
    split.Method(&f.fast, count, 22, true, nullptr);
    split.Publish();
    REQUIRE(Access::End(split) > noted);
    // The back runs up to the noted position: the methods before it, none after.
    u64 read = 0;
    while (read < noted) {
        read = Access::Execute(split, read, Access::End(split));
    }
    REQUIRE(read == noted);
    CHECK(f.fast.regs.vertex_buffer.first == 11);
    CHECK(f.fast.regs.vertex_buffer.count == 12);
    while (read != Access::End(split)) {
        read = Access::Execute(split, read, Access::End(split));
    }
    CHECK(f.fast.regs.vertex_buffer.first == 21);
    CHECK(f.fast.regs.vertex_buffer.count == 22);
    CHECK(Access::IrBadRecords(split) == 0);
}
