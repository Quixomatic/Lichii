// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/core.h"
#include "shader_recompiler/backend/spirv/bindless_heap.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
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

/// Turns the group tracker on (through EDEN_NX2_V2_PIPEKEY) for the test and restores the switch.
struct TrackGroupsScope {
    static constexpr size_t index = static_cast<size_t>(VideoCommon::V2::Switch::PipeKey);
    VideoCommon::V2::Mode previous = VideoCommon::V2::g_modes[index];
    TrackGroupsScope() {
        VideoCommon::V2::g_modes[index] = VideoCommon::V2::Mode::On;
    }
    ~TrackGroupsScope() {
        VideoCommon::V2::g_modes[index] = previous;
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
    // Guest memory stand-ins for A and B.
    std::vector<u8> a(0x100, 0x11);
    std::vector<u8> b(0x200, 0x22);
    table->Insert(A, 0x100, 7, 1, table->Begin(A, 0x100), a.data());
    table->Insert(B, 0x200, 7, 2, table->Begin(B, 0x200), b.data());
    REQUIRE(*table->Find(A, 0x100, 7, a.data()) == 1);
    REQUIRE(*table->Find(B, 0x200, 7, b.data()) == 2);
    // Another tick (the stream allocation may be gone) or size: no version.
    REQUIRE(table->Find(A, 0x100, 8, a.data()) == nullptr);
    REQUIRE(table->Find(A, 0x80, 7, a.data()) == nullptr);
    // An inline write next to A but inside B ends B's version only.
    log->NoteInline(A + 0x100, A + 0x500);
    REQUIRE(*table->Find(A, 0x100, 7, a.data()) == 1);
    REQUIRE(table->Find(B, 0x200, 7, b.data()) == nullptr);
    // Writes elsewhere keep A, even many of them, as long as the log still holds them.
    for (int i = 0; i < 100; ++i) {
        log->NoteInline(Far + i * 16, Far + i * 16 + 16);
        REQUIRE(table->Find(A, 0x100, 7, a.data()) != nullptr);
    }
    // Past the log's size without a lookup: unknown, so the version ends.
    table->Insert(B, 0x200, 7, 3, table->Begin(B, 0x200), b.data());
    for (size_t i = 0; i < WriteLog::LogSize + 1; ++i) {
        log->NoteInline(Far, Far + 4);
    }
    REQUIRE(table->Find(B, 0x200, 7, b.data()) == nullptr);
    // A host write to A's page ends it; one to another page doesn't.
    table->Insert(A, 0x100, 7, 4, table->Begin(A, 0x100), a.data());
    log->NoteHost(Far, 4);
    REQUIRE(*table->Find(A, 0x100, 7, a.data()) == 4);
    log->NoteHost(A + 0x800, 4);
    REQUIRE(table->Find(A, 0x100, 7, a.data()) == nullptr);
    // A boundary ends every version; a long host write too.
    table->Insert(A, 0x100, 7, 5, table->Begin(A, 0x100), a.data());
    REQUIRE(table->Find(A, 0x100, 7, a.data()) != nullptr);
    log->NoteBoundary();
    REQUIRE(table->Find(A, 0x100, 7, a.data()) == nullptr);
    table->Insert(A, 0x100, 7, 6, table->Begin(A, 0x100), a.data());
    log->NoteHost(Far, (WriteLog::MaxHostPages + 1) << WriteLog::PageBits);
    REQUIRE(table->Find(A, 0x100, 7, a.data()) == nullptr);
    // A version keeps its bytes; without the current bytes nothing is found.
    const std::array<u8, 4> bytes{1, 2, 3, 4};
    table->Insert(A, 4, 9, 7, table->Begin(A, 4), bytes.data());
    REQUIRE(table->Bytes(A, 4) == std::vector<u8>{1, 2, 3, 4});
    REQUIRE(table->Find(A, 4, 9, nullptr) == nullptr);
    table->Clear();
    REQUIRE(table->Find(A, 4, 9, bytes.data()) == nullptr);
}

TEST_CASE("V2 P3 constant buffer versions see guest CPU writes through their bytes",
          "[video_core][v2_p3]") {
    using namespace VideoCommon::V2::CbufVer;
    auto log = std::make_unique<WriteLog>();
    auto table = std::make_unique<VersionTable<int>>(*log);
    constexpr DAddr A = 0x10'0f80; // straddles two pages
    std::vector<u8> guest(0x100);
    for (size_t i = 0; i < guest.size(); ++i) {
        guest[i] = static_cast<u8>(i);
    }
    table->Insert(A, 0x100, 1, 1, table->Begin(A, 0x100), guest.data());
    REQUIRE(table->Find(A, 0x100, 1, guest.data()) != nullptr);
    // A CPU write nobody was told about (no trap, no log entry) that changes a byte ends the
    // version at the next lookup; one that writes the same value does not.
    guest[0x90] = 0xff;
    REQUIRE(table->Find(A, 0x100, 1, guest.data()) == nullptr);
    guest[0x90] = 0x90;
    table->Insert(A, 0x100, 1, 2, table->Begin(A, 0x100), guest.data());
    guest[0x10] = 0x10;
    REQUIRE(*table->Find(A, 0x100, 1, guest.data()) == 2);
    // The last byte counts too.
    guest[0xff] ^= 1;
    REQUIRE(table->Find(A, 0x100, 1, guest.data()) == nullptr);
    // Reset forgets everything (a new buffer cache).
    guest[0xff] ^= 1;
    table->Insert(A, 0x100, 1, 3, table->Begin(A, 0x100), guest.data());
    table->Reset();
    REQUIRE(table->Find(A, 0x100, 1, guest.data()) == nullptr);
}

TEST_CASE("V2 P3 constant buffer versions read guest memory only when they may hold",
          "[video_core][v2_p3]") {
    using namespace VideoCommon::V2::CbufVer;
    auto log = std::make_unique<WriteLog>();
    auto table = std::make_unique<VersionTable<int>>(*log);
    constexpr DAddr A = 0x20'0000;
    std::vector<u8> guest(0x40, 0x5a);
    // No version yet: FindWith never reads.
    u32 reads = 0;
    const auto read = [&] {
        ++reads;
        return static_cast<const u8*>(guest.data());
    };
    REQUIRE(table->FindWith(A, 0x40, 3, read) == nullptr);
    REQUIRE(reads == 0);
    // Prepare + fill + Commit is a version, found by its bytes.
    const u64 page_sum = table->Begin(A, 0x40);
    u8* const bytes = table->Prepare(A, 0x40, 3, page_sum);
    std::memcpy(bytes, guest.data(), guest.size());
    REQUIRE(table->FindWith(A, 0x40, 3, read) == nullptr); // not committed yet
    table->Commit(A, 0x40, 9);
    REQUIRE(table->MayHold(A, 0x40, 3, table->Begin(A, 0x40)));
    REQUIRE(*table->FindWith(A, 0x40, 3, read) == 9);
    REQUIRE(reads == 1);
    // A guest CPU write still ends it at once (the bytes are compared on every lookup).
    guest[0x3f] = 0;
    REQUIRE(table->FindWith(A, 0x40, 3, read) == nullptr);
    REQUIRE(reads == 2);
    guest[0x3f] = 0x5a;
    // A host write ends it without reading guest memory.
    log->NoteHost(A, 4);
    REQUIRE(table->FindWith(A, 0x40, 3, read) == nullptr);
    REQUIRE(reads == 2);
    // So does another tick.
    table->Insert(A, 0x40, 3, 10, table->Begin(A, 0x40), guest.data());
    REQUIRE(table->FindWith(A, 0x40, 4, read) == nullptr);
    REQUIRE(reads == 2);
    REQUIRE(*table->Find(A, 0x40, 3, guest.data()) == 10);
}

namespace {
struct HeapModule {
    u32 set1_decorations{};
    u32 set0_decorations{};
    bool words_binding{};
    bool sampled_image{};
    bool non_uniform{};
    u32 push_variables{}; ///< OpVariable ... PushConstant
};

/// A fragment program sampling texture 0 (with `texture_count` elements, read with a dynamic
/// element index when above 1) and, with `image`, reading storage image 0.
HeapModule EmitHeapProgram(u64 texture_mask, u64 image_mask, u32 texture_count, bool image,
                           const char* name, bool pushed = false) {
    using namespace Shader;
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    program.stage = Stage::Fragment;
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    IR::IREmitter ir{block};
    IR::TextureInstInfo flags{};
    flags.type.Assign(TextureType::ColorArray2D);
    flags.descriptor_index.Assign(0);
    const auto coords = ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f));
    // A dynamic element index: a constant buffer word.
    const IR::Value element{texture_count > 1 ? IR::Value{ir.GetCbuf(ir.Imm32(0), ir.Imm32(0))}
                                              : IR::Value{ir.Imm32(0)}};
    const IR::Value sample{ir.ImageSampleExplicitLod(element, coords, ir.Imm32(0.0f), {}, flags)};
    // What the texture pass makes of the bound sample once the descriptor is known.
    sample.InstRecursive()->ReplaceOpcode(IR::Opcode::ImageSampleExplicitLod);
    program.info.texture_descriptors.push_back(TextureDescriptor{
        .type = TextureType::ColorArray2D,
        .is_depth = false,
        .is_multisample = false,
        .is_integer = false,
        .has_secondary = false,
        .cbuf_index = 2,
        .cbuf_offset = 0x20,
        .shift_left = 0,
        .secondary_cbuf_index = 0,
        .secondary_cbuf_offset = 0,
        .secondary_shift_left = 0,
        .count = texture_count,
        .size_shift = 2,
    });
    if (texture_count > 1) {
        program.info.constant_buffer_mask = 1;
        program.info.constant_buffer_descriptors.push_back({.index = 0, .count = 1});
        program.info.constant_buffer_used_sizes[0] = 16;
        program.info.used_constant_buffer_types = IR::Type::U32;
    }
    if (image) {
        IR::TextureInstInfo image_flags{};
        image_flags.type.Assign(TextureType::Color2D);
        image_flags.descriptor_index.Assign(0);
        const IR::Value read{
            ir.ImageRead(ir.Imm32(0), ir.CompositeConstruct(ir.Imm32(0), ir.Imm32(0)),
                         image_flags)};
        read.InstRecursive()->ReplaceOpcode(IR::Opcode::ImageRead);
        program.info.image_descriptors.push_back(ImageDescriptor{
            .type = TextureType::Color2D,
            .format = ImageFormat::R32_UINT,
            .is_written = false,
            .is_read = true,
            .is_integer = true,
            .cbuf_index = 2,
            .cbuf_offset = 0x40,
            .count = 1,
            .size_shift = 0,
        });
    }
    program.info.heap_texture_mask = texture_mask;
    program.info.heap_image_mask = image_mask;
    program.info.heap_word_base = 3;
    program.info.heap_words_pushed = pushed;
    Profile profile;
    profile.supported_spirv = 0x00010500;
    profile.unified_descriptor_binding = true;
    profile.support_sampled_image_array_nonuniform_indexing = true;
    const std::vector<u32> code = Backend::SPIRV::EmitSPIRV(profile, program);
    // EDEN_V2_HEAP_TEST_OUTPUT=<dir>: write the modules for spirv-val.
    if (const char* directory = std::getenv("EDEN_V2_HEAP_TEST_OUTPUT")) {
        std::filesystem::create_directories(directory);
        std::ofstream file{std::filesystem::path{directory} / name, std::ios::binary};
        file.write(reinterpret_cast<const char*>(code.data()),
                   static_cast<std::streamsize>(code.size() * sizeof(u32)));
    }
    HeapModule module;
    for (size_t i = 5; i < code.size();) {
        const u32 opcode = code[i] & 0xffff;
        const u32 words = code[i] >> 16;
        REQUIRE(words != 0);
        if (opcode == 71 && words >= 4 && code[i + 2] == 34) { // OpDecorate DescriptorSet
            ++(code[i + 3] == 1 ? module.set1_decorations : module.set0_decorations);
        }
        if (opcode == 71 && words >= 4 && code[i + 2] == 33 && code[i + 3] == 1023) {
            module.words_binding = true; // OpDecorate Binding 1023
        }
        if (opcode == 71 && words >= 3 && code[i + 2] == 5300) { // NonUniform
            module.non_uniform = true;
        }
        if (opcode == 86) { // OpSampledImage
            module.sampled_image = true;
        }
        if (opcode == 59 && words >= 4 && code[i + 3] == 9) { // OpVariable PushConstant
            ++module.push_variables;
        }
        i += words;
    }
    return module;
}
} // Anonymous namespace

TEST_CASE("V2 P3 heap descriptors index the bindless heap through the word table",
          "[video_core][v2_p3]") {
    namespace SPIRV = Shader::Backend::SPIRV;
    // Bindings: one per image class and component type after the sampler array, then the
    // storage image arrays, all distinct.
    std::vector<u32> bindings;
    for (const auto type : {Shader::TextureType::Color1D, Shader::TextureType::ColorArray1D,
                            Shader::TextureType::Color2D, Shader::TextureType::ColorArray2D,
                            Shader::TextureType::Color3D, Shader::TextureType::ColorCube,
                            Shader::TextureType::ColorArrayCube}) {
        for (const bool integer : {false, true}) {
            bindings.push_back(SPIRV::HeapImageBinding(type, false, integer));
            if (SPIRV::HeapStorageType(type)) {
                bindings.push_back(SPIRV::HeapStorageBinding(type, integer));
            }
        }
    }
    for (const bool integer : {false, true}) {
        bindings.push_back(SPIRV::HeapImageBinding(Shader::TextureType::Color2D, true, integer));
        bindings.push_back(
            SPIRV::HeapImageBinding(Shader::TextureType::ColorArray2D, true, integer));
        // P5a: storage texel buffers.
        bindings.push_back(SPIRV::HeapStorageTexelBinding(integer));
    }
    bindings.push_back(SPIRV::HEAP_UNIFORM_TEXEL_BINDING);
    std::ranges::sort(bindings);
    REQUIRE(std::ranges::adjacent_find(bindings) == bindings.end());
    REQUIRE(bindings.front() == SPIRV::HEAP_SAMPLER_BINDING + 1);
    REQUIRE(bindings.back() == SPIRV::NUM_HEAP_BINDINGS - 1);
    REQUIRE(bindings.size() == SPIRV::NUM_HEAP_BINDINGS - 1);
    REQUIRE(SPIRV::HeapImageBinding(Shader::TextureType::Color2DRect, false, false) ==
            SPIRV::HeapImageBinding(Shader::TextureType::Color2D, false, false));
    REQUIRE_FALSE(SPIRV::HeapStorageType(Shader::TextureType::ColorCube));
    // Words: the stage base, heap textures' elements in order, then heap images' elements.
    Shader::Info info;
    for (const u32 count : {1U, 4U, 2U}) {
        Shader::TextureDescriptor texture{};
        texture.type = Shader::TextureType::Color2D;
        texture.count = count;
        info.texture_descriptors.push_back(texture);
    }
    for (const u32 count : {1U, 3U}) {
        Shader::ImageDescriptor image{};
        image.type = Shader::TextureType::Color2D;
        image.count = count;
        info.image_descriptors.push_back(image);
    }
    info.heap_texture_mask = 0b110; // textures 1 and 2
    info.heap_image_mask = 0b11;
    info.heap_word_base = 5;
    REQUIRE_FALSE(SPIRV::UsesHeap(info, 0));
    REQUIRE(SPIRV::HeapTextureWord(info, 1) == 5);
    REQUIRE(SPIRV::HeapTextureWord(info, 2) == 9);
    REQUIRE(SPIRV::HeapImageWord(info, 0) == 11);
    REQUIRE(SPIRV::HeapImageWord(info, 1) == 12);
    REQUIRE(SPIRV::HeapWordCount(info) == 10);
    REQUIRE_FALSE(SPIRV::UsesHeap(info, 64));
    // Emission: without the masks the descriptors are set 0 bindings as before; with them, the
    // heap arrays of set 1 and the word table at set 0, binding 1023.
    const HeapModule plain = EmitHeapProgram(0, 0, 1, true, "plain.spv");
    REQUIRE(plain.set1_decorations == 0);
    REQUIRE(plain.set0_decorations == 2);
    REQUIRE_FALSE(plain.words_binding);
    REQUIRE_FALSE(plain.sampled_image);
    const HeapModule heap = EmitHeapProgram(1, 1, 1, true, "heap.spv");
    REQUIRE(heap.set1_decorations == 3); // sampled images, samplers, storage images
    REQUIRE(heap.set0_decorations == 1); // the word table
    REQUIRE(heap.words_binding);
    REQUIRE(heap.sampled_image);
    REQUIRE_FALSE(heap.non_uniform);
    REQUIRE(heap.push_variables == 1); // the table base
    // An array read with a dynamic element index: its heap accesses are NonUniform.
    const HeapModule array = EmitHeapProgram(1, 0, 4, false, "heap_array.spv");
    REQUIRE(array.words_binding);
    REQUIRE(array.non_uniform);
    REQUIRE(array.set1_decorations == 2);
    // Words that fit the push constants: no word table, one push constant block (perf fix B).
    const HeapModule pushed = EmitHeapProgram(1, 1, 1, true, "heap_pushed.spv", true);
    REQUIRE_FALSE(pushed.words_binding);
    REQUIRE(pushed.set0_decorations == 0);
    REQUIRE(pushed.set1_decorations == 3);
    REQUIRE(pushed.push_variables == 1);
    REQUIRE(pushed.sampled_image);
    const HeapModule pushed_array =
        EmitHeapProgram(1, 0, 4, false, "heap_pushed_array.spv", true);
    REQUIRE_FALSE(pushed_array.words_binding);
    REQUIRE(pushed_array.non_uniform);
    REQUIRE(pushed_array.push_variables == 1);
    REQUIRE(plain.push_variables == 0);
}

namespace {
struct TexelModule {
    u32 set1_decorations{};
    u32 set0_decorations{};
    bool words_binding{};
    bool uniform_dynamic{};
    bool storage_dynamic{};
};

/// A fragment program fetching texture buffer 0 and reading integer image buffer 0, with
/// `heap` texel buffers in the heap (EDEN_NX2_V2_HEAP_TEXEL).
TexelModule EmitTexelProgram(bool heap, const char* name) {
    using namespace Shader;
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    program.stage = Stage::Fragment;
    program.blocks = {&block};
    program.post_order_blocks = {&block};
    program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                           {{}, IR::AbstractSyntaxNode::Type::Return}};
    IR::IREmitter ir{block};
    IR::TextureInstInfo fetch_flags{};
    fetch_flags.type.Assign(TextureType::Buffer);
    fetch_flags.descriptor_index.Assign(0);
    const IR::Value fetch{
        ir.ImageFetch(ir.Imm32(0), ir.Imm32(5), {}, ir.Imm32(0), {}, fetch_flags)};
    fetch.InstRecursive()->ReplaceOpcode(IR::Opcode::ImageFetch);
    IR::TextureInstInfo read_flags{};
    read_flags.type.Assign(TextureType::Buffer);
    read_flags.descriptor_index.Assign(0);
    const IR::Value read{ir.ImageRead(ir.Imm32(0), ir.Imm32(7), read_flags)};
    read.InstRecursive()->ReplaceOpcode(IR::Opcode::ImageRead);
    program.info.texture_buffer_descriptors.push_back(TextureBufferDescriptor{
        .has_secondary = false,
        .cbuf_index = 2,
        .cbuf_offset = 0x10,
        .shift_left = 0,
        .secondary_cbuf_index = 0,
        .secondary_cbuf_offset = 0,
        .secondary_shift_left = 0,
        .count = 1,
        .size_shift = 0,
    });
    program.info.image_buffer_descriptors.push_back(ImageBufferDescriptor{
        .format = ImageFormat::R32_UINT,
        .is_written = false,
        .is_read = true,
        .is_integer = true,
        .cbuf_index = 2,
        .cbuf_offset = 0x20,
        .count = 1,
        .size_shift = 0,
    });
    program.info.uses_image_buffers = true;
    program.info.heap_texel_mask = heap ? 1 : 0;
    program.info.heap_image_buffer_mask = heap ? 1 : 0;
    program.info.heap_word_base = 2;
    Profile profile;
    profile.supported_spirv = 0x00010500;
    profile.unified_descriptor_binding = true;
    const std::vector<u32> code = Backend::SPIRV::EmitSPIRV(profile, program);
    if (const char* directory = std::getenv("EDEN_V2_HEAP_TEST_OUTPUT")) {
        std::filesystem::create_directories(directory);
        std::ofstream file{std::filesystem::path{directory} / name, std::ios::binary};
        file.write(reinterpret_cast<const char*>(code.data()),
                   static_cast<std::streamsize>(code.size() * sizeof(u32)));
    }
    TexelModule module;
    for (size_t i = 5; i < code.size();) {
        const u32 opcode = code[i] & 0xffff;
        const u32 words = code[i] >> 16;
        REQUIRE(words != 0);
        if (opcode == 71 && words >= 4 && code[i + 2] == 34) { // OpDecorate DescriptorSet
            ++(code[i + 3] == 1 ? module.set1_decorations : module.set0_decorations);
        }
        if (opcode == 71 && words >= 4 && code[i + 2] == 33 && code[i + 3] == 1023) {
            module.words_binding = true;
        }
        if (opcode == 17 && words >= 2) { // OpCapability
            module.uniform_dynamic |= code[i + 1] == 5304;
            module.storage_dynamic |= code[i + 1] == 5305;
        }
        i += words;
    }
    return module;
}
} // Anonymous namespace

TEST_CASE("V2 P5 texel buffers index the bindless heap through the word table",
          "[video_core][v2_p5]") {
    namespace SPIRV = Shader::Backend::SPIRV;
    // Words: per stage the heap texel buffers come first (texture buffers, then image buffers),
    // then the heap textures and storage images.
    Shader::Info info;
    for (int i = 0; i < 2; ++i) {
        Shader::TextureBufferDescriptor texel{};
        texel.count = 1;
        info.texture_buffer_descriptors.push_back(texel);
    }
    Shader::ImageBufferDescriptor image_buffer{};
    image_buffer.count = 3;
    info.image_buffer_descriptors.push_back(image_buffer);
    Shader::TextureDescriptor texture{};
    texture.type = Shader::TextureType::Color2D;
    texture.count = 1;
    info.texture_descriptors.push_back(texture);
    Shader::ImageDescriptor image{};
    image.type = Shader::TextureType::Color2D;
    image.count = 2;
    info.image_descriptors.push_back(image);
    info.heap_texel_mask = 0b10; // texture buffer 1 only
    info.heap_image_buffer_mask = 0b1;
    info.heap_texture_mask = 0b1;
    info.heap_image_mask = 0b1;
    info.heap_word_base = 7;
    REQUIRE_FALSE(SPIRV::UsesHeapTexel(info, 0));
    REQUIRE(SPIRV::UsesHeapTexel(info, 1));
    REQUIRE(SPIRV::HeapTexelWord(info, 1) == 7);
    REQUIRE(SPIRV::HeapImageBufferWord(info, 0) == 8);
    REQUIRE(SPIRV::HeapTexelWordCount(info) == 4);
    REQUIRE(SPIRV::HeapTextureWord(info, 0) == 11);
    REQUIRE(SPIRV::HeapImageWord(info, 0) == 12);
    REQUIRE(SPIRV::HeapWordCount(info) == 7);
    // Emission: without the masks two set 0 texel bindings; with them the heap's uniform and
    // storage texel arrays at set 1 and the word table, with the dynamic indexing capabilities.
    const TexelModule plain = EmitTexelProgram(false, "texel_plain.spv");
    REQUIRE(plain.set0_decorations == 2);
    REQUIRE(plain.set1_decorations == 0);
    REQUIRE_FALSE(plain.words_binding);
    REQUIRE_FALSE(plain.uniform_dynamic);
    const TexelModule heap = EmitTexelProgram(true, "texel_heap.spv");
    REQUIRE(heap.set0_decorations == 1);
    REQUIRE(heap.set1_decorations == 2);
    REQUIRE(heap.words_binding);
    REQUIRE(heap.uniform_dynamic);
    REQUIRE(heap.storage_dynamic);
}
