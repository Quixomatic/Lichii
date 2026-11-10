// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/cbuf_bda.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "video_core/v2/cbuf_address_table.h"

namespace {
namespace SPIRV = Shader::Backend::SPIRV;

struct BdaModule {
    std::vector<u32> code;
    u32 uniform_variables{};  ///< OpVariable ... Uniform
    u32 push_variables{};     ///< OpVariable ... PushConstant
    u32 convert_to_pointer{}; ///< OpConvertUToPtr
    u32 addressing{};         ///< OpMemoryModel's addressing model
    bool psb_capability{};
    bool int64_capability{};
    std::vector<u32> bindings; ///< OpDecorate Binding values
};

struct BdaSetup {
    u32 mask{};
    u32 base{};
    u32 offset{SPIRV::HEAP_PUSH_OFFSET};
    bool pushed{};
    bool keep_bindings{};
    bool rescaling{};    ///< the stage has the rescaling push block (the address member joins it)
    bool heap{};         ///< texture 0 reads the heap (pushed words)
};

/// A fragment program reading cbuf 1 at 0x10 (u32), cbuf 3 at 0x24 (f32) and cbuf 5 at 0x40 (u32),
/// all at immediate offsets.
BdaModule EmitBdaProgram(const BdaSetup& setup, const char* name) {
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
    static_cast<void>(ir.GetCbuf(ir.Imm32(1), ir.Imm32(0x10)));
    static_cast<void>(ir.GetFloatCbuf(ir.Imm32(3), ir.Imm32(0x24)));
    static_cast<void>(ir.GetCbuf(ir.Imm32(5), ir.Imm32(0x40)));
    for (const u32 index : {1U, 3U, 5U}) {
        program.info.constant_buffer_mask |= 1U << index;
        program.info.constant_buffer_descriptors.push_back({.index = index, .count = 1});
    }
    program.info.constant_buffer_used_sizes[1] = 0x20;
    program.info.constant_buffer_used_sizes[3] = 0x30;
    program.info.constant_buffer_used_sizes[5] = 0x50;
    program.info.used_constant_buffer_types = IR::Type::U32 | IR::Type::F32;
    program.info.uses_rescaling_uniform = setup.rescaling;
    if (setup.heap) {
        IR::TextureInstInfo flags{};
        flags.type.Assign(TextureType::Color2D);
        flags.descriptor_index.Assign(0);
        const auto coords = ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f));
        const IR::Value sample{
            ir.ImageSampleExplicitLod(ir.Imm32(0), coords, ir.Imm32(0.0f), {}, flags)};
        sample.InstRecursive()->ReplaceOpcode(IR::Opcode::ImageSampleExplicitLod);
        program.info.texture_descriptors.push_back(TextureDescriptor{
            .type = TextureType::Color2D,
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
            .count = 1,
            .size_shift = 2,
        });
        program.info.heap_texture_mask = 1;
        program.info.heap_words_pushed = true;
    }
    program.info.cbuf_bda_mask = setup.mask;
    program.info.cbuf_bda_base = setup.base;
    program.info.cbuf_bda_offset = setup.offset;
    program.info.cbuf_bda_pushed = setup.pushed;
    program.info.cbuf_bda_keep_bindings = setup.keep_bindings;
    Profile profile;
    profile.supported_spirv = 0x00010500;
    profile.unified_descriptor_binding = true;
    profile.support_descriptor_aliasing = true;
    profile.support_int64 = true;
    BdaModule module;
    module.code = Backend::SPIRV::EmitSPIRV(profile, program);
    // EDEN_V2_HEAP_TEST_OUTPUT=<dir>: write the modules for spirv-val.
    if (const char* directory = std::getenv("EDEN_V2_HEAP_TEST_OUTPUT")) {
        std::filesystem::create_directories(directory);
        std::ofstream file{std::filesystem::path{directory} / name, std::ios::binary};
        file.write(reinterpret_cast<const char*>(module.code.data()),
                   static_cast<std::streamsize>(module.code.size() * sizeof(u32)));
    }
    const std::vector<u32>& code = module.code;
    for (size_t i = 5; i < code.size();) {
        const u32 opcode = code[i] & 0xffff;
        const u32 words = code[i] >> 16;
        REQUIRE(words != 0);
        if (opcode == 17 && code[i + 1] == 5347) { // OpCapability PhysicalStorageBufferAddresses
            module.psb_capability = true;
        }
        if (opcode == 17 && code[i + 1] == 11) { // OpCapability Int64
            module.int64_capability = true;
        }
        if (opcode == 14) { // OpMemoryModel
            module.addressing = code[i + 1];
        }
        if (opcode == 59 && words >= 4 && code[i + 3] == 2) { // OpVariable Uniform
            ++module.uniform_variables;
        }
        if (opcode == 59 && words >= 4 && code[i + 3] == 9) { // OpVariable PushConstant
            ++module.push_variables;
        }
        if (opcode == 120) { // OpConvertUToPtr
            ++module.convert_to_pointer;
        }
        if (opcode == 71 && words >= 4 && code[i + 2] == 33) { // OpDecorate Binding
            module.bindings.push_back(code[i + 3]);
        }
        i += words;
    }
    std::ranges::sort(module.bindings);
    return module;
}
} // Anonymous namespace

TEST_CASE("V2 F2-02 constant buffer address numbering", "[video_core][v2_cbuf_bda]") {
    Shader::Info info;
    for (const u32 index : {0U, 2U, 3U, 7U}) {
        info.constant_buffer_mask |= 1U << index;
        info.constant_buffer_descriptors.push_back({.index = index, .count = 1});
        info.constant_buffer_used_sizes[index] = 0x100;
    }
    info.constant_buffer_used_sizes[3] = 0x10000; // read at a dynamic offset
    REQUIRE(SPIRV::CbufBdaEligible(info, 0));
    REQUIRE(SPIRV::CbufBdaEligible(info, 2));
    REQUIRE_FALSE(SPIRV::CbufBdaEligible(info, 3));
    REQUIRE_FALSE(SPIRV::CbufBdaEligible(info, 5)); // not read
    info.cbuf_bda_mask = (1U << 0) | (1U << 2) | (1U << 7);
    info.cbuf_bda_base = 4;
    REQUIRE(SPIRV::CbufBdaCount(info) == 3);
    REQUIRE(SPIRV::CbufBdaIndex(info, 0) == 4);
    REQUIRE(SPIRV::CbufBdaIndex(info, 2) == 5);
    REQUIRE(SPIRV::CbufBdaIndex(info, 7) == 6);
    REQUIRE(SPIRV::CbufBoundCount(info) == 1);
    REQUIRE(SPIRV::CbufUnboundCount(info) == 3);
    info.cbuf_bda_keep_bindings = true;
    REQUIRE(SPIRV::CbufBoundCount(info) == 4);
    REQUIRE(SPIRV::CbufUnboundCount(info) == 0);
    // The push layout: after the rescaling/render area layouts, after pushed heap words, after the
    // heap word table's base; all within 128 bytes.
    REQUIRE(SPIRV::CbufBdaPushOffset(false, false) == 48);
    REQUIRE(SPIRV::CbufBdaPushOffset(true, true) == 112);
    REQUIRE(SPIRV::CbufBdaPushOffset(true, false) == 56);
    REQUIRE(SPIRV::CbufBdaPushCapacity(48) == 10);
    REQUIRE(SPIRV::CbufBdaPushCapacity(112) == 2);
    REQUIRE(SPIRV::CbufBdaPushCapacity(56) == 9);
    REQUIRE(SPIRV::CbufBdaPushCapacity(128) == 0);
}

TEST_CASE("V2 F2-02 a pipeline's constant buffer addresses and their push layout",
          "[video_core][v2_cbuf_bda]") {
    const auto stage = [](std::initializer_list<std::pair<u32, u32>> cbufs) {
        Shader::Info info;
        for (const auto& [index, used] : cbufs) {
            info.constant_buffer_mask |= 1U << index;
            info.constant_buffer_descriptors.push_back({.index = index, .count = 1});
            info.constant_buffer_used_sizes[index] = used;
        }
        return info;
    };
    // Vertex: cbufs 0 and 2 (2 dynamically indexed); fragment: 1, 3 and 4. Without the heap the
    // four addresses are pushed at 48.
    Shader::Info vertex = stage({{0, 0x40}, {2, 0x10000}});
    Shader::Info fragment = stage({{1, 0x10}, {3, 0x100}, {4, 0x20}});
    std::array<Shader::Info*, 2> infos{&vertex, &fragment};
    REQUIRE(SPIRV::AssignCbufBda(infos, false) == 4);
    REQUIRE(vertex.cbuf_bda_mask == 1U);
    REQUIRE(vertex.cbuf_bda_base == 0);
    REQUIRE(fragment.cbuf_bda_mask == ((1U << 1) | (1U << 3) | (1U << 4)));
    REQUIRE(fragment.cbuf_bda_base == 1);
    REQUIRE(SPIRV::CbufBdaIndex(fragment, 4) == 3);
    REQUIRE(vertex.cbuf_bda_offset == 48);
    REQUIRE(fragment.cbuf_bda_offset == 48);
    REQUIRE(vertex.cbuf_bda_pushed);
    REQUIRE(fragment.cbuf_bda_pushed);
    REQUIRE(SPIRV::CbufBoundCount(vertex) == 1); // cbuf 2 keeps its uniform buffer
    // Heap words pushed: two addresses fit after them, four don't: the table's pointer at 112.
    fragment.heap_texture_mask = 1;
    fragment.heap_words_pushed = true;
    vertex.heap_words_pushed = true;
    REQUIRE(SPIRV::AssignCbufBda(infos, true) == 4);
    REQUIRE(fragment.cbuf_bda_offset == 112);
    REQUIRE_FALSE(fragment.cbuf_bda_pushed);
    REQUIRE_FALSE(vertex.cbuf_bda_pushed);
    REQUIRE(fragment.cbuf_bda_keep_bindings);
    // The heap word table: its base at 48, the addresses from 56.
    fragment.heap_words_pushed = false;
    vertex.heap_words_pushed = false;
    REQUIRE(SPIRV::AssignCbufBda(infos, false) == 4);
    REQUIRE(fragment.cbuf_bda_offset == 56);
    REQUIRE(fragment.cbuf_bda_pushed);
}

TEST_CASE("V2 F2-02 constant buffer address tables", "[video_core][v2_cbuf_bda]") {
    struct Allocation {
        u64 device_address;
        u64 offset;
        std::span<u8> mapped_span;
    };
    std::vector<std::vector<u8>> chunks;
    u32 allocations = 0;
    const auto allocate = [&](size_t bytes) {
        ++allocations;
        chunks.emplace_back(bytes);
        return Allocation{0x1000'0000ULL * allocations, 0x100, chunks.back()};
    };
    VideoCommon::V2::CbufAddressTable table;
    const std::array<u64, 3> first{0x10, 0x20, 0x30};
    bool reused = true;
    const u64 a = table.Append(first, 1, reused, allocate);
    REQUIRE_FALSE(reused);
    REQUIRE(a == 0x1000'0100ULL);
    u64 written[3]{};
    std::memcpy(written, chunks[0].data(), sizeof(written));
    REQUIRE(written[2] == 0x30);
    // The same table in the same tick: its address again, nothing written.
    REQUIRE(table.Append(first, 1, reused, allocate) == a);
    REQUIRE(reused);
    // Another one follows in the chunk.
    const std::array<u64, 3> second{0x10, 0x20, 0x40};
    REQUIRE(table.Append(second, 1, reused, allocate) == a + sizeof(first));
    REQUIRE_FALSE(reused);
    REQUIRE(allocations == 1);
    // A new tick takes a new chunk (the old one may be recycled after its tick).
    REQUIRE(table.Append(second, 2, reused, allocate) == 0x2000'0100ULL);
    REQUIRE_FALSE(reused);
    REQUIRE(allocations == 2);
}

TEST_CASE("V2 F2-02 constant buffers read through buffer device addresses",
          "[video_core][v2_cbuf_bda]") {
    // Without the mask: the old module, whatever the other fields say (CBUFBDA=0 is
    // byte-identical).
    const BdaModule plain = EmitBdaProgram({}, "cbuf_plain.spv");
    BdaSetup unmarked{};
    unmarked.offset = 112;
    unmarked.pushed = true;
    unmarked.keep_bindings = true;
    REQUIRE(EmitBdaProgram(unmarked, "cbuf_unmarked.spv").code == plain.code);
    REQUIRE(plain.uniform_variables == 3);
    REQUIRE(plain.push_variables == 0);
    REQUIRE(plain.convert_to_pointer == 0);
    REQUIRE(plain.addressing == 0); // Logical
    REQUIRE_FALSE(plain.psb_capability);
    REQUIRE(plain.bindings == std::vector<u32>{0, 1, 2});

    // Every constant buffer through pushed addresses: no uniform buffer, one push block.
    const u32 all = (1U << 1) | (1U << 3) | (1U << 5);
    const BdaModule pushed = EmitBdaProgram({.mask = all, .pushed = true}, "cbuf_pushed.spv");
    REQUIRE(pushed.uniform_variables == 0);
    REQUIRE(pushed.push_variables == 1);
    REQUIRE(pushed.convert_to_pointer == 3);
    REQUIRE(pushed.addressing == 5348); // PhysicalStorageBuffer64
    REQUIRE(pushed.psb_capability);
    REQUIRE(pushed.int64_capability);
    REQUIRE(pushed.bindings.empty());

    // The table: the pushed pointer, then the address, per read.
    const BdaModule table = EmitBdaProgram({.mask = all, .pushed = false}, "cbuf_table.spv");
    REQUIRE(table.uniform_variables == 0);
    REQUIRE(table.push_variables == 1);
    REQUIRE(table.convert_to_pointer == 6);

    // Per slot: cbuf 3 through an address, 1 and 5 keep uniform buffers (bindings 0 and 1), or
    // with verify's kept bindings 0 and 2 (cbuf 3's binding stays reserved).
    const BdaModule mixed =
        EmitBdaProgram({.mask = 1U << 3, .base = 2, .pushed = true}, "cbuf_mixed.spv");
    REQUIRE(mixed.uniform_variables == 2);
    REQUIRE(mixed.convert_to_pointer == 1);
    REQUIRE(mixed.bindings == std::vector<u32>{0, 1});
    const BdaModule kept = EmitBdaProgram(
        {.mask = 1U << 3, .pushed = true, .keep_bindings = true}, "cbuf_mixed_kept.spv");
    REQUIRE(kept.uniform_variables == 2);
    REQUIRE(kept.bindings == std::vector<u32>{0, 2});

    // The address member shares the stage's push block: the rescaling block, or the heap's.
    const BdaModule rescaled = EmitBdaProgram(
        {.mask = all, .pushed = true, .rescaling = true}, "cbuf_rescaling.spv");
    REQUIRE(rescaled.push_variables == 1);
    REQUIRE(rescaled.convert_to_pointer == 3);
    const BdaModule heap = EmitBdaProgram(
        {.mask = all, .offset = SPIRV::CbufBdaPushOffset(true, true), .pushed = false,
         .heap = true},
        "cbuf_heap.spv");
    REQUIRE(heap.push_variables == 1);
    REQUIRE(heap.convert_to_pointer == 6);
    const BdaModule heap_rescaled = EmitBdaProgram(
        {.mask = all, .offset = SPIRV::CbufBdaPushOffset(true, true), .pushed = false,
         .rescaling = true, .heap = true},
        "cbuf_heap_rescaling.spv");
    REQUIRE(heap_rescaled.push_variables == 1);
}
