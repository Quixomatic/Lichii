// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "core/device_memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/memory_manager.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/renderer_vulkan/fixed_pipeline_state.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/v2/cbuf_version.h"
#include "video_core/v2/continued_draw.h"
#include "video_core/v2/desc_cache.h"
#include "video_core/v2/pipeline_lookup.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/v2_switches.h"

namespace {

using Tegra::Engines::Maxwell3D;

/// Turns the group tracker on for the test and restores the switch afterwards.
struct TrackGroupsScope {
    bool previous = VideoCommon::V2::g_track_groups;
    TrackGroupsScope() {
        VideoCommon::V2::g_track_groups = true;
    }
    ~TrackGroupsScope() {
        VideoCommon::V2::g_track_groups = previous;
    }
};

Vulkan::DynamicFeatures RandomFeatures(std::mt19937& rng) {
    const auto coin = [&] { return (rng() & 1) != 0; };
    Vulkan::DynamicFeatures features{};
    features.driver_id = 4;
    features.driver_version = 0x1234;
    features.has_extended_dynamic_state = coin();
    features.has_extended_dynamic_state_2 = features.has_extended_dynamic_state && coin();
    features.has_extended_dynamic_state_2_logic_op = features.has_extended_dynamic_state_2 && coin();
    features.has_extended_dynamic_state_3_blend = features.has_extended_dynamic_state_2 && coin();
    features.has_extended_dynamic_state_3_enables = features.has_extended_dynamic_state_2 && coin();
    features.has_dynamic_state3_depth_clamp_enable =
        features.has_extended_dynamic_state_3_enables && coin();
    features.has_dynamic_state3_logic_op_enable =
        features.has_extended_dynamic_state_3_enables && coin();
    features.has_dynamic_state3_line_stipple_enable =
        features.has_extended_dynamic_state_3_enables && coin();
    features.has_dynamic_state3_conservative_raster =
        features.has_extended_dynamic_state_3_enables && coin();
    features.has_dynamic_state3_line_rasterization_mode =
        features.has_extended_dynamic_state_3_enables && coin();
    features.has_dynamic_state3_alpha_to_coverage =
        features.has_extended_dynamic_state_3_enables && coin();
    features.has_dynamic_state3_alpha_to_one =
        features.has_extended_dynamic_state_3_enables && coin();
    features.has_dynamic_vertex_input = coin();
    features.has_static_vertex_repack = !features.has_dynamic_vertex_input && coin();
    features.has_color_write_enable = coin();
    features.has_provoking_vertex = coin();
    features.has_provoking_vertex_first_mode = coin();
    features.has_provoking_vertex_last_mode = coin();
    features.has_provoking_vertex_tf_preserve = coin();
    return features;
}

struct KeyFixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device_memory{physical};
    Tegra::MemoryManager memory{system, device_memory, 32, 1ULL << 31};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    Maxwell3D engine{memory};
    std::vector<u32> state_methods;

    KeyFixture() {
        engine.BindRasterizer(&rasterizer);
        VideoCommon::Dirty::SetupDirtyFlags(engine.dirty.tables);
        for (u32 method = 0; method < Maxwell3D::Regs::NUM_REGS; ++method) {
            if (!engine.execution_mask[method]) {
                state_methods.push_back(method);
            }
        }
    }
};

} // Anonymous namespace

TEST_CASE("V2 P3 incremental pipeline key equals a full refresh", "[video_core][v2_p3]") {
    TrackGroupsScope track;
    KeyFixture f;
    std::mt19937 rng{0x5eed'0003};
    // Registers the key reads, written far more often than the rest so every block re-packs.
    std::vector<u32> key_registers;
    for (size_t block = 0; block < Vulkan::NumKeyBlocks; ++block) {
        for (const auto& [first, count] :
             Vulkan::KeyBlockRegisters(static_cast<Vulkan::KeyBlock>(block))) {
            for (u32 reg = first; reg < first + count; ++reg) {
                key_registers.push_back(reg);
            }
        }
    }
    for (int round = 0; round < 24; ++round) {
        Vulkan::DynamicFeatures features = RandomFeatures(rng);
        Vulkan::FixedPipelineState full{};
        Vulkan::FixedPipelineState incremental{};
        std::memset(&full, 0, sizeof(full));
        std::memset(&incremental, 0, sizeof(incremental));
        Vulkan::KeyInputs inputs{};
        bool first = true;
        // A new key starts from flagged state, as after StateTracker::InvalidateState.
        f.engine.dirty.flags[Vulkan::Dirty::VertexInput] = true;
        f.engine.dirty.flags[Vulkan::Dirty::ViewportSwizzles] = true;
        for (int step = 0; step < 1500; ++step) {
            const int writes = static_cast<int>(rng() % 6);
            for (int w = 0; w < writes; ++w) {
                const bool key_register = (rng() % 4) != 0;
                const u32 method = key_register ? key_registers[rng() % key_registers.size()]
                                                : f.state_methods[rng() % f.state_methods.size()];
                // Small values hit the enums and booleans the key packs, large ones the rest.
                const u32 value = (rng() & 1) != 0 ? rng() % 4 : static_cast<u32>(rng());
                f.engine.CallMethod(f.system, method, value, true);
            }
            if (rng() % 8 == 0) {
                f.engine.draw_manager.draw_state.topology =
                    static_cast<Maxwell3D::Regs::PrimitiveTopology>(rng() % 15);
            }
            if (rng() % 16 == 0) {
                f.engine.engine_state =
                    static_cast<Maxwell3D::EngineHint>(rng() % 3);
            }
            if (rng() % 4 == 0) {
                f.engine.dirty.flags[Vulkan::Dirty::VertexInput] = true;
            }
            if (rng() % 4 == 0) {
                f.engine.dirty.flags[Vulkan::Dirty::ViewportSwizzles] = true;
            }
            const u32 groups =
                f.engine.v2_groups.TakeSink(VideoCommon::V2::GroupSink::PipelineKey);
            incremental.RefreshIncremental(f.engine, features, groups, inputs, first, false);
            incremental.CanonicalizeDynamic(features);
            inputs = Vulkan::FixedPipelineState::CurrentInputs(f.engine);
            first = false;
            full.Refresh(f.engine, features);
            full.CanonicalizeDynamic(features);
            REQUIRE(full.Size() == incremental.Size());
            size_t offset = 0;
            const auto* const a = reinterpret_cast<const u8*>(&full);
            const auto* const b = reinterpret_cast<const u8*>(&incremental);
            while (offset < full.Size() && a[offset] == b[offset]) {
                ++offset;
            }
            CAPTURE(round, step, offset, groups, full.Size());
            REQUIRE(offset == full.Size());
        }
    }
}

TEST_CASE("V2 P3 key blocks read registers of the groups their masks hold",
          "[video_core][v2_p3]") {
    KeyFixture f; // builds the register semantics table
    const auto& masks = Vulkan::KeyBlockGroups();
    const auto& table = VideoCommon::V2::RegSem::GetTable3D();
    for (size_t block = 0; block < Vulkan::NumKeyBlocks; ++block) {
        for (const auto& [first, count] :
             Vulkan::KeyBlockRegisters(static_cast<Vulkan::KeyBlock>(block))) {
            REQUIRE(count != 0);
            for (u32 reg = first; reg < first + count; ++reg) {
                REQUIRE((masks[block] & VideoCommon::V2::GroupBit(table.groups[reg])) != 0);
            }
        }
    }
    // The core block reads pipeline state and the render target formats.
    REQUIRE((masks[0] & VideoCommon::V2::GroupBit(VideoCommon::V2::RegGroup::Pipeline)) != 0);
}

TEST_CASE("V2 P3 two-level pipeline lookup", "[video_core][v2_p3]") {
    using Lookup = VideoCommon::V2::PipelineLookup<int>;
    Lookup lookup;
    std::mt19937 rng{7};
    struct Entry {
        Lookup::Hashes hashes;
        std::array<u8, 40> key;
        u64 hash;
        int value;
    };
    std::vector<std::unique_ptr<Entry>> entries;
    for (int set = 0; set < 50; ++set) {
        Lookup::Hashes hashes{};
        for (auto& h : hashes) {
            h = (static_cast<u64>(rng()) << 32) | rng();
        }
        // Up to 40 variants per set: past the linear part into the hashed table. Every third
        // shares its hash with the previous one, with other bytes.
        const int variants = 1 + static_cast<int>(rng() % 40);
        for (int v = 0; v < variants; ++v) {
            auto entry = std::make_unique<Entry>();
            entry->hashes = hashes;
            for (auto& byte : entry->key) {
                byte = static_cast<u8>(rng());
            }
            entry->hash = (v % 3 == 2 && !entries.empty()) ? entries.back()->hash
                                                           : (static_cast<u64>(rng()) << 32) | rng();
            entry->value = static_cast<int>(entries.size());
            REQUIRE(lookup.Find(hashes, entry->hash, entry->key.data(), entry->key.size()) ==
                    nullptr);
            lookup.Insert(hashes, entry->hash, entry->key.data(), entry->key.size(),
                          &entry->value);
            entries.push_back(std::move(entry));
        }
    }
    for (const auto& entry : entries) {
        int* const found =
            lookup.Find(entry->hashes, entry->hash, entry->key.data(), entry->key.size());
        REQUIRE(found == &entry->value);
        // One byte off, or a different length: not found.
        auto other = entry->key;
        other[5] ^= 1;
        REQUIRE(lookup.Find(entry->hashes, entry->hash, other.data(), other.size()) == nullptr);
        REQUIRE(lookup.Find(entry->hashes, entry->hash, entry->key.data(), 39) == nullptr);
    }
    REQUIRE(lookup.NumSets() == 50);
    lookup.Clear();
    REQUIRE(lookup.NumSets() == 0);
    REQUIRE(lookup.Find(entries[0]->hashes, entries[0]->hash, entries[0]->key.data(),
                        entries[0]->key.size()) == nullptr);
}

TEST_CASE("V2 P3 group sinks accumulate until taken", "[video_core][v2_p3]") {
    VideoCommon::V2::GroupTracker tracker;
    using VideoCommon::V2::GroupBit;
    using VideoCommon::V2::GroupSink;
    using VideoCommon::V2::RegGroup;
    REQUIRE(tracker.TakeSink(GroupSink::PipelineKey) == VideoCommon::V2::AllGroups);
    tracker.dirty = 0;
    tracker.sinks.fill(0);
    tracker.Change(RegGroup::Viewport, 1, 0, 1);
    tracker.BeginDraw();
    tracker.Change(RegGroup::Pipeline, 2, 0, 1);
    tracker.BeginDraw();
    tracker.Touch(RegGroup::CbufData);
    // A sink taken between draws sees the changes no draw took yet too.
    REQUIRE(tracker.PeekSink(GroupSink::ContinuedDraw) ==
            (GroupBit(RegGroup::Viewport) | GroupBit(RegGroup::Pipeline) |
             GroupBit(RegGroup::CbufData)));
    REQUIRE(tracker.TakeSink(GroupSink::PipelineKey) ==
            (GroupBit(RegGroup::Viewport) | GroupBit(RegGroup::Pipeline) |
             GroupBit(RegGroup::CbufData)));
    tracker.BeginDraw();
    REQUIRE(tracker.TakeSink(GroupSink::PipelineKey) == GroupBit(RegGroup::CbufData));
    REQUIRE(tracker.TakeSink(GroupSink::PipelineKey) == 0);
    REQUIRE(tracker.TakeSink(GroupSink::ContinuedDraw) ==
            (GroupBit(RegGroup::Viewport) | GroupBit(RegGroup::Pipeline) |
             GroupBit(RegGroup::CbufData)));
}

TEST_CASE("V2 P3 continued draws classify changed groups as light or heavy",
          "[video_core][v2_p3]") {
    using VideoCommon::V2::ClassifyGroups;
    using VideoCommon::V2::GroupBit;
    using VideoCommon::V2::GroupClass;
    using VideoCommon::V2::RegGroup;
    REQUIRE(ClassifyGroups(0, true) == GroupClass::Light);
    REQUIRE(ClassifyGroups(GroupBit(RegGroup::DrawParams) | GroupBit(RegGroup::CbufData) |
                               GroupBit(RegGroup::CbufBind) | GroupBit(RegGroup::IndexBuffer) |
                               GroupBit(RegGroup::Misc),
                           true) == GroupClass::Light);
    for (const RegGroup group : {RegGroup::Pipeline, RegGroup::Shaders, RegGroup::Viewport,
                                 RegGroup::VertexStreams}) {
        REQUIRE(ClassifyGroups(GroupBit(group) | GroupBit(RegGroup::CbufData), true) ==
                GroupClass::PipelineCheck);
    }
    REQUIRE(ClassifyGroups(GroupBit(RegGroup::RenderTargets), false) == GroupClass::Heavy);
    REQUIRE(ClassifyGroups(GroupBit(RegGroup::RenderTargets) | GroupBit(RegGroup::Pipeline),
                           false) == GroupClass::Heavy);
    // The pools only matter to pipelines that read them (relevance by shader use).
    REQUIRE(ClassifyGroups(GroupBit(RegGroup::TexPools), true) == GroupClass::Heavy);
    REQUIRE(ClassifyGroups(GroupBit(RegGroup::TexPools), false) == GroupClass::Light);
    // Every group has a class: light, checked or heavy.
    for (u32 group = 0; group < VideoCommon::V2::NumRegGroups; ++group) {
        const u32 bit = 1U << group;
        const bool known = (bit & VideoCommon::V2::LightGroups) != 0 ||
                           (bit & VideoCommon::V2::PipelineCheckGroups) != 0 ||
                           bit == GroupBit(RegGroup::RenderTargets) ||
                           bit == GroupBit(RegGroup::TexPools);
        REQUIRE(known);
    }
}

TEST_CASE("V2 P3 descriptor content keys", "[video_core][v2_p3]") {
    using Cache = VideoCommon::V2::DescriptorContentCache<int, 4>;
    using VideoCommon::V2::HashDescriptorBytes;
    Cache cache;
    std::vector<std::vector<u8>> payloads;
    std::mt19937 rng{11};
    for (int i = 0; i < 6; ++i) {
        std::vector<u8> bytes(24 * (1 + i % 3));
        for (auto& byte : bytes) {
            byte = static_cast<u8>(rng());
        }
        payloads.push_back(std::move(bytes));
    }
    const auto hash = [&](int i) { return HashDescriptorBytes(payloads[i]); };
    REQUIRE(hash(0) != hash(1));
    REQUIRE(HashDescriptorBytes(payloads[0]) == hash(0));
    for (int i = 0; i < 4; ++i) {
        REQUIRE(cache.Find(payloads[i], hash(i), 1) == nullptr);
        cache.Insert(payloads[i], hash(i), 1) = i;
    }
    for (int i = 0; i < 4; ++i) {
        const int* const value = cache.Find(payloads[i], hash(i), 1);
        REQUIRE(value != nullptr);
        REQUIRE(*value == i);
        // Another epoch: not valid.
        REQUIRE(cache.Find(payloads[i], hash(i), 2) == nullptr);
    }
    // Touch 1..3 so 0 is the least recent, then a fifth payload replaces it.
    for (int i = 1; i < 4; ++i) {
        REQUIRE(cache.Find(payloads[i], hash(i), 1) != nullptr);
    }
    cache.Insert(payloads[4], hash(4), 1) = 4;
    REQUIRE(cache.Find(payloads[0], hash(0), 1) == nullptr);
    REQUIRE(*cache.Find(payloads[4], hash(4), 1) == 4);
    // A new epoch takes the entries of old epochs first.
    cache.Insert(payloads[5], hash(5), 2) = 5;
    REQUIRE(*cache.Find(payloads[5], hash(5), 2) == 5);
    // Equal hash, other bytes: no hit unless the hash is trusted; Collides tells.
    auto forged = payloads[1];
    forged[3] ^= 0x40;
    REQUIRE(cache.Find(forged, hash(1), 1) == nullptr);
    REQUIRE(cache.Find(forged, hash(1), 1, true) != nullptr);
    REQUIRE(cache.Collides(forged, hash(1), 1));
    REQUIRE_FALSE(cache.Collides(payloads[1], hash(1), 1));
    cache.Clear();
    REQUIRE(cache.Find(payloads[1], hash(1), 1) == nullptr);
}

TEST_CASE("V2 P3 constant buffer versions", "[video_core][v2_p3]") {
    using namespace VideoCommon::V2::CbufVer;
    auto log = std::make_unique<WriteLog>();
    auto table = std::make_unique<VersionTable<int>>(*log);
    constexpr DAddr A = 0x10'0000;
    constexpr DAddr B = 0x10'0400;
    constexpr DAddr Far = 0x80'0000;
    table->Insert(A, 0x100, 7, 1);
    table->Insert(B, 0x200, 7, 2);
    REQUIRE(*table->Find(A, 0x100, 7) == 1);
    REQUIRE(*table->Find(B, 0x200, 7) == 2);
    // Another tick (the stream allocation may be gone) or size: no version.
    REQUIRE(table->Find(A, 0x100, 8) == nullptr);
    REQUIRE(table->Find(A, 0x80, 7) == nullptr);
    // An inline write next to A but inside B ends B's version only.
    log->NoteInline(A + 0x100, A + 0x500);
    REQUIRE(*table->Find(A, 0x100, 7) == 1);
    REQUIRE(table->Find(B, 0x200, 7) == nullptr);
    // Writes elsewhere keep A, even many of them, as long as the log still holds them.
    for (int i = 0; i < 100; ++i) {
        log->NoteInline(Far + i * 16, Far + i * 16 + 16);
        REQUIRE(table->Find(A, 0x100, 7) != nullptr);
    }
    // Past the log's size without a lookup: unknown, so the version ends.
    table->Insert(B, 0x200, 7, 3);
    for (size_t i = 0; i < WriteLog::LogSize + 1; ++i) {
        log->NoteInline(Far, Far + 4);
    }
    REQUIRE(table->Find(B, 0x200, 7) == nullptr);
    // A host write to A's page ends it; one to another page doesn't.
    table->Insert(A, 0x100, 7, 4);
    log->NoteHost(Far, 4);
    REQUIRE(*table->Find(A, 0x100, 7) == 4);
    log->NoteHost(A + 0x800, 4);
    REQUIRE(table->Find(A, 0x100, 7) == nullptr);
    // A command list boundary ends every version; a long host write too.
    table->Insert(A, 0x100, 7, 5);
    REQUIRE(table->Find(A, 0x100, 7) != nullptr);
    log->NoteBoundary();
    REQUIRE(table->Find(A, 0x100, 7) == nullptr);
    table->Insert(A, 0x100, 7, 6);
    log->NoteHost(Far, (WriteLog::MaxHostPages + 1) << WriteLog::PageBits);
    REQUIRE(table->Find(A, 0x100, 7) == nullptr);
    // Verify keeps the bytes of a version.
    const std::array<u8, 4> bytes{1, 2, 3, 4};
    table->Insert(A, 4, 9, 7, bytes.data());
    REQUIRE(table->Bytes(A, 4) == std::vector<u8>{1, 2, 3, 4});
    table->Clear();
    REQUIRE(table->Find(A, 4, 9) == nullptr);
}
