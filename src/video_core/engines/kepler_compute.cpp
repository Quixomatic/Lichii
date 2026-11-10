// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <deque>
#include <algorithm>
#include <atomic>
#include <bit>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "common/assert.h"
#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "core/core.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/gpu_split.h"
#include "video_core/fire_emblem_upscale.h"
#include "video_core/nx2_profile.h"
#include "video_core/nx2.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/native_upscale.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/game_dlss_status.h"
#include "video_core/memory_manager.h"
#include "video_core/native_query_trace.h"
#include "video_core/oot_network.h"
#ifdef ARCHITECTURE_x86_64
#include "video_core/macro_turing_jit.h"
#endif
#include "video_core/rasterizer_interface.h"
#include "video_core/textures/decoders.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/qmd_front.h"
#include "video_core/v2/front_writes.h"
#include "common/scope_exit.h"
#include "video_core/v2/shader_writes.h"
#include "video_core/v2/ir_check.h"
#include "video_core/v2/macro_memo.h"
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/v2_stats.h"
#include "video_core/v2/s1_deltas.h"

namespace Tegra::Engines {

KeplerCompute::KeplerCompute(MemoryManager& memory_manager_)
    : memory_manager{memory_manager_}
    , upload_state{memory_manager, regs.upload}
{
    // EDEN_NX2_QMD_TRACE: log the QMD scheduler's loads, dependency counts and launches.
    if (std::getenv("EDEN_NX2_QMD_TRACE")) {
        native_qmd.trace = [this](std::string_view event, u64 address, s32 value) {
            LOG_INFO(HW_GPU, "QMD trace {} {:#x} {} (engine {} as {})", event, address, value,
                     fmt::ptr(this), memory_manager.GetID());
        };
    }
    native_qmd.skipped_release = [](u64 address, std::size_t slot, std::span<const u32, 4> words) {
        static std::unordered_set<u32> logged;
        if (!logged.insert(words[1]).second) return;
        LOG_WARNING(HW_GPU, "Native QMD {:#x} skipped release slot {} with unsupported format, words={:08x}",
                    address, slot, fmt::join(words, ","));
    };
    execution_mask.reset();
    execution_mask[KEPLER_COMPUTE_REG_INDEX(exec_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(data_upload)] = true;
    execution_mask[KEPLER_COMPUTE_REG_INDEX(launch)] = true;
    // Everything ProcessNativeMethod acts on besides storing the value; the QMD and 0x5a
    // trace range too while native draw diagnostics trace it.
    native_plain_methods.set();
    for (const u32 method : {0x45U, 0x46U, 0x47U, 0x48U, 0xaeU, 0xafU, 0xb0U, 0x107U, 0x6c3U,
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(exec_upload)),
                             static_cast<u32>(KEPLER_COMPUTE_REG_INDEX(data_upload))}) {
        native_plain_methods[method] = false;
    }
    if (std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr) {
        native_plain_methods[0x5a] = false;
        for (u32 method = 0xc6; method <= 0x108; ++method) {
            native_plain_methods[method] = false;
        }
    } else if (VideoCommon::V2::On(VideoCommon::V2::Switch::RegSem)) [[unlikely]] {
        // EDEN_NX2_V2_REGSEM: the plain stores come from the semantics table (=1); verify
        // checks the table gives the same set.
        const auto& table = VideoCommon::V2::RegSem::TableComputePlain();
        if (VideoCommon::V2::Verify(VideoCommon::V2::Switch::RegSem)) {
            for (u32 method = 0; method < 0xe00; ++method) {
                if (table[method] != native_plain_methods[method]) {
                    LOG_WARNING(HW_GPU, "V2 regsem compute: method {:#x} plain table {} engine {}",
                                method, bool{table[method]}, bool{native_plain_methods[method]});
                }
            }
        } else {
            native_plain_methods = table;
            v2_compute_table = true;
        }
    }
    v2_compute_verify = VideoCommon::V2::Verify(VideoCommon::V2::Switch::RegSem);
}

KeplerCompute::~KeplerCompute() = default;

void KeplerCompute::BindRasterizer(VideoCore::RasterizerInterface* rasterizer_) {
    rasterizer = rasterizer_;
    upload_state.BindRasterizer(rasterizer);
}

void KeplerCompute::ConsumeSinkImpl(Core::System& system) {
    for (auto [method, value] : method_sink) {
        if (method < Regs::NUM_REGS) {
            regs.reg_array[method] = value;
        }
    }
    method_sink.clear();
}

// SET_REPORT_SEMAPHORE_A-D (0x6c0-0x6c3): address high/low, payload, then operation in bits 1:0
// (release / acquire / report only / trap), reduction enable bit 3 with the op in bits 11:9, and
// a one-word (no timestamp) structure in bit 28. NVN2 releases queue syncs from compute this way.
void KeplerCompute::ProcessNativeReportSemaphore(u32 argument) {
    const GPUVAddr address = (GPUVAddr{native_registers[0x6c0] & 0x1ffffff} << 32) |
                             native_registers[0x6c1];
    const u32 payload = native_registers[0x6c2];
    const u32 operation = argument & 3;
    const bool one_word = (argument >> 28) & 1;
    static u32 logged{};
    if (logged++ < 32) {
        LOG_INFO(HW_GPU, "Native compute report semaphore {:#x} = {:#x}, control {:#x}", address,
                 payload, argument);
    }
    if (operation != 0 && operation != 2) {
        LOG_WARNING(HW_GPU, "Native compute report semaphore operation {} not supported", operation);
        return;
    }
    if ((argument >> 3) & 1) {
        const u32 reduction = (argument >> 9) & 7;
        // Written at its fence's operation. Without delayed fences that runs at once, unless an
        // earlier write still has to land (v2/landing.h): then it lands behind it, in order.
        namespace Landing = VideoCommon::V2::Landing;
        const bool deferred = Landing::MustDefer();
        const u64 landing =
            deferred || VideoCommon::V2::FencesDelayed() ? Landing::Capture() : 0;
        VideoCore::g_signaling_query_report_fence = deferred;
        rasterizer->SignalFence([this, address, payload, reduction, landing] {
            memory_manager.Write<u32>(address, ReduceSemaphore(reduction, false,
                                                               memory_manager.Read<u32>(address), payload));
            if (landing != 0) {
                Landing::Note(landing);
            }
        });
        VideoCore::g_signaling_query_report_fence = false;
        return;
    }
    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                      one_word ? VideoCommon::QueryPropertiesFlags::IsAFence
                               : VideoCommon::QueryPropertiesFlags::HasTimeout,
                      payload, 0);
}

bool KeplerCompute::ProcessNativeMethod(Core::System& system, u32 method, u32 argument, bool is_last_call) {
    if (native_context && !front_macros) {
        if (method == 0x45 || method == 0x47) native_compute_uploaded = true;
        native_macro = native_compute_uploaded ? &native_context->compute_engine : &native_context->engine;
        native_macro_positions = native_compute_uploaded ? std::span<u32>{native_context->compute_positions} : std::span<u32>{native_context->positions};
    }
    if (method >= native_registers.size()) {
        LOG_ERROR(HW_GPU, "Native compute method out of range: {:#x}", method);
        return true;
    }
    if (method >= 0xe00) {
        if (!native_executing_macro) {
            if (method & 1) { LOG_ERROR(HW_GPU, "Native compute macro starts on argument method"); return true; }
            native_executing_macro = method;
        }
        native_parameters.push_back(argument);
        if (!is_last_call) return true;
        const u32 entry = ((*native_executing_macro - 0xe00) / 2) % native_macro_positions.size();
        native_executing_macro_logged = native_executing_macro;
        native_executing_macro.reset();
        static const u32 watch = [] {
            const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
            return value ? static_cast<u32>(std::strtoull(value, nullptr, 16)) : 0U;
        }();
        const bool watched = watch != 0 && std::ranges::any_of(native_parameters,
            [](u32 parameter) { return (parameter & ~0xfU) == watch; });
        if (watched) {
            LOG_INFO(HW_GPU, "Semaphore watch: compute macro start={:#x} parameters={:08x}",
                     native_macro_positions[entry], fmt::join(native_parameters, ","));
        }
        // EDEN_NX2_MME_DUMP: write the compute macro code once per engine, and log each macro
        // call with its entry and parameters (diagnostics).
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr; mme_dump) {
            if (!native_mme_dumped && native_macro_positions[entry] == 0xda) {
                native_mme_dumped = true;
                std::ofstream dump(fmt::format("native-compute-mme-{}.bin", fmt::ptr(this)),
                                   std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()),
                           sizeof(native_macro->native_code));
            }
            LOG_INFO(HW_GPU, "MME call engine {} method {:#x} start={:#x} parameters={:08x}",
                     fmt::ptr(this), *native_executing_macro_logged, native_macro_positions[entry],
                     fmt::join(native_parameters, ","));
        }
        if (native_data_ram) native_macro->native_interpreter.active_dram = native_data_ram;
        const u32 dram0_before = (*native_macro->native_interpreter.active_dram)[0x27];
        const u32 start = native_macro_positions[entry];
        const auto emit = [&](u32 address, u32 value) {
            static u32 diagnostic_writes{};
            static const bool log_emits = std::getenv("EDEN_NX2_SEM_WATCH_EMITS") != nullptr;
            if (watched && log_emits) LOG_INFO(HW_GPU, "Semaphore watch: compute macro emit {:#x} = {:#x}", address, value);
            if (diagnostic_writes++ < 4096) LOG_DEBUG(HW_GPU, "Native compute MME emit method={:#x}, value={:#x}", address, value);
            if (address >= 0xe00) return false;
            VideoCommon::Nx2Profile::Scope method_scope{VideoCommon::Nx2Profile::Zone::Method};
            CallMethod(system, address, value, true);
            return true;
        };
#ifdef ARCHITECTURE_x86_64
        // Compiled macro code, as for 3D macros (macro_turing_jit.h); EDEN_NX2_MME_JIT=0 off.
        static const bool use_jit = [] {
            const char* value = std::getenv("EDEN_NX2_MME_JIT");
            return value == nullptr || value[0] != '0';
        }();
        if (use_jit && !watched) {
            if (!native_macro->native_jit) {
                native_macro->native_jit = std::make_shared<TuringMacroJit>();
            }
            native_macro->native_jit->Prepare(native_macro->native_code,
                                              native_macro->native_code_generation);
            const auto result = native_macro->native_jit->Execute(
                start, native_parameters,
                std::span<u32>{*native_macro->native_interpreter.active_dram},
                [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
                emit);
            if (!result.success) {
                native_macro->native_execution_failed = true;
                LOG_ERROR(HW_GPU, "Native MME (JIT) failed: compute start={:#x}, consumed={}/{}, error={}",
                          start, result.consumed, native_parameters.size(), result.error);
            }
            native_parameters.clear();
            return true;
        }
#endif
        // Repeated calls replay their writes, as for 3D macros (macro_memo.h).
        static const bool use_memo = [] {
            const char* value = std::getenv("EDEN_NX2_MME_MEMO");
            return value == nullptr || value[0] != '0';
        }();
        auto& memo = native_macro->native_memo;
        if (use_memo && !watched) {
            memo.Synchronize(native_macro->native_code_generation);
            if (std::remove_reference_t<decltype(memo)>::Effects effects;
                memo.Find(start, native_parameters, *native_macro->native_interpreter.active_dram,
                          effects)) {
                auto& data_ram = *native_macro->native_interpreter.active_dram;
                for (size_t i = 0; i < effects.data_ram_writes.size(); i += 2) {
                    data_ram[effects.data_ram_writes[i]] = effects.data_ram_writes[i + 1];
                }
                for (size_t i = 0; i < effects.writes.size(); i += 2) {
                    emit(effects.writes[i], effects.writes[i + 1]);
                }
                native_parameters.clear();
                return true;
            }
            native_macro->native_writes.clear();
        }
        // The interpreter consumes native_parameters as its FIFO: keep the call's own parameters.
        native_memo_parameters.assign(native_parameters.begin(), native_parameters.end());
        const u64 code_generation = native_macro->native_code_generation;
        const auto result = native_macro->native_interpreter.Execute(native_macro->native_code,
            start, native_parameters,
            [&](u32 address) { return address < native_registers.size() ? native_registers[address] : 0U; },
            [&](u32 address, u32 value) {
                if (use_memo) {
                    native_macro->native_writes.emplace_back(address, value);
                }
                return emit(address, value);
            });
        const auto& interpreter = native_macro->native_interpreter;
        if (use_memo && !watched && result.success && !interpreter.read_registers &&
            !interpreter.reread_data_ram && code_generation == native_macro->native_code_generation) {
            memo.Store(start, native_memo_parameters, interpreter.data_ram_reads,
                       interpreter.data_ram_writes, native_macro->native_writes);
        }
        if (static const bool mme_dump = std::getenv("EDEN_NX2_MME_DUMP") != nullptr;
            mme_dump && (*native_macro->native_interpreter.active_dram)[0x27] != dram0_before) {
            LOG_INFO(HW_GPU, "MME data RAM[0x27] {:#x} -> {:#x} by engine {} macro start={:#x}",
                     dram0_before, (*native_macro->native_interpreter.active_dram)[0x27], fmt::ptr(this),
                     native_macro_positions[entry]);
        }
        if (!result.success) {
            native_macro->native_execution_failed = true;
            if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
                diagnostics) {
                std::ofstream dump("native-compute-mme.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            }
            LOG_ERROR(HW_GPU, "Native MME failed: compute start={:#x}, pc={:#x}, consumed={}/{}, error={}, parameters={:08x}",
                native_macro_positions[entry], result.pc, result.consumed, native_parameters.size(), result.error,
                fmt::join(native_parameters, ","));
        }
        native_parameters.clear();
        return true;
    }
    // S1 (F1-05): the front sent this engine's groups; the back hashes nothing.
    if (VideoCommon::V2::On(VideoCommon::V2::Switch::RegSem) && !s1_compute) [[unlikely]] {
        VideoCommon::V2::RegSem::NoteWriteCompute(v2_groups, method, native_registers[method],
                                                  argument);
    }
    native_registers[method] = argument;
    if (method == 0x6c3) { // SET_REPORT_SEMAPHORE_D: the compute class's semaphore release
        ProcessNativeReportSemaphore(argument);
        return true;
    }
    if (method == 0x5a || (method >= 0xc6 && method <= 0x108)) {
        VideoCommon::TraceNativeQuery(12, memory_manager.GetID(), method, 0, 0, argument, 0);
    }
    if (method == 0x107) {
        const GPUVAddr address = ((GPUVAddr{native_registers[0xc6]} << 32) |
            native_registers[0xc7]) << 8;
        const u32 control = native_registers[0xa7];
        if (!NativeInlineQmdUpload()) {
            return true;
        }
        if ((control & (1U << 24)) || !(native_registers[0xcc] & (1U << 12))) {
            const u32 previous = native_registers[0xad];
            const u32 inline_qmd = static_cast<u32>(address >> 8);
            // EDEN_NX2_V2_REGSEM: the launch group's hash follows the temporary QMD address (the
            // dispatch's fingerprint and hash check see it), and its restore (Fix pass 1: the
            // unnoted swap left group 0x4 drifted after every inline launch's hash check).
            const bool regsem = VideoCommon::V2::On(VideoCommon::V2::Switch::RegSem);
            if (regsem) [[unlikely]] {
                VideoCommon::V2::RegSem::NoteWriteCompute(v2_groups, 0xad, previous, inline_qmd);
            }
            native_registers[0xad] = inline_qmd;
            ProcessNativeMethod(system, 0xb0, 3, true);
            if (regsem) [[unlikely]] {
                VideoCommon::V2::RegSem::NoteWriteCompute(v2_groups, 0xad, inline_qmd, previous);
            }
            native_registers[0xad] = previous;
        }
        return true;
    }
    if (front_macros && method >= 0x45 && method <= 0x48) {
        return true; // Loaded and bound by the split GPU thread's front (FrontApply).
    }
    switch (method) {
    case 0x45: native_macro->ClearCode(argument); return true;
    case 0x46: native_macro->AddCode(0, argument); return true;
    case 0x47: native_bind_pointer = argument; return true;
    case 0x48:
        if (native_bind_pointer < native_macro_positions.size()) native_macro_positions[native_bind_pointer++] = argument;
        else LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        return true;
    case 0xae: case 0xaf: case 0xb0: {
        const GPUVAddr qmd_address = GPUVAddr{native_registers[0xad]} << 8;
        VideoCommon::TraceNativeQuery(9, memory_manager.GetID(), qmd_address,
            memory_manager.GpuToCpuAddress(qmd_address).value_or(0), method, argument, 0);
        if (method == 0xae || (method == 0xaf && !(argument & 2)) ||
            (method == 0xb0 && ((argument & 15) == 0 || (argument & 15) == 1))) return true;
        static const bool qmd_diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
        if ((argument & 15) == 12 && qmd_diagnostics) {
            static u32 captured{};
            if (captured++ == 0 && memory_manager.IsFullyMappedRange(qmd_address, 256)) {
                std::array<u32, 64> descriptor{};
                memory_manager.ReadBlockUnsafe(qmd_address, descriptor.data(), sizeof(descriptor));
                std::ofstream dump("native-increment-qmd.bin", std::ios::binary);
                dump.write(reinterpret_cast<const char*>(descriptor.data()), sizeof(descriptor));
                LOG_WARNING(HW_GPU, "Native increment dependency descriptor: address={:#x}, words={:08x}",
                    qmd_address, fmt::join(descriptor, ","));
            }
        }
        // The front can't see QMD releases (Producer::QmdRelease): they take no landing ticket.
        const VideoCommon::V2::Landing::Untracked untracked;
        const auto error = native_qmd.Signal(qmd_address, argument,
            [&](u64 address, NativeQmdScheduler::Descriptor& descriptor) {
                if (!memory_manager.IsFullyMappedRange(address, sizeof(descriptor))) return false;
                memory_manager.ReadBlockUnsafe(address, descriptor.data(), sizeof(descriptor));
                if (static u32 logged{}; (descriptor[53] & (1U << 23)) && logged < 8) {
                    ++logged;
                    LOG_INFO(HW_GPU, "Native QMD {:#x} releases: {:08x} | {:08x} | {:08x}", address,
                             fmt::join(std::span{descriptor}.subspan(24, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(28, 4), ","),
                             fmt::join(std::span{descriptor}.subspan(52, 4), ","));
                }
                for (u32 slot = 0; slot < 2; ++slot) {
                    const u32 flags = descriptor[25 + slot * 4];
                    if (!(flags & (1U << 23))) continue;
                    const GPUVAddr destination = (GPUVAddr{flags & 0xff} << 32) | descriptor[24 + slot * 4];
                    VideoCommon::TraceNativeQuery(7 + slot, memory_manager.GetID(), destination,
                        memory_manager.GpuToCpuAddress(destination).value_or(0),
                        descriptor[4], descriptor[26 + slot * 4], flags);
                }
                return true;
            },
            [&](u64 address, u32 payload, u32 flags) {
                const bool four_words = (flags >> 30) == 0;
                if (!memory_manager.IsFullyMappedRange(address, four_words ? 16 : sizeof(payload))) return false;
                if (flags & (1U << 26)) {
                    rasterizer->SignalFence([this, address, payload, flags] {
                        const u32 value = ReduceSemaphore((flags >> 20) & 7, (flags >> 24) & 1,
                                                          memory_manager.Read<u32>(address), payload);
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(*device, value);
                        }
                        static const u64 watch = [] {
                            const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                            return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                        }();
                        if (watch != 0 && (address & ~u64{0xf}) == watch) {
                            LOG_INFO(HW_GPU, "Semaphore watch: QMD release reduction {:#x} -> {:#x}", address, value);
                        }
                        VideoCommon::TraceNativeQuery(1, memory_manager.GetID(), address,
                            memory_manager.GpuToCpuAddress(address).value_or(0), 0, value, 1U << 26);
                    });
                } else if (four_words) {
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence | VideoCommon::QueryPropertiesFlags::HasTimeout,
                        payload, 0);
                } else if (flags & (1U << 19)) {
                    // NVN2 tracks QMD completion with a max release (bit 19) into a word other
                    // releases also advance: a plain write would move it backwards and leave
                    // GEQ acquires waiting forever.
                    rasterizer->SignalFence([this, address, payload] {
                        const u32 value = (std::max)(memory_manager.Read<u32>(address), payload);
                        memory_manager.Write<u32>(address, value);
                        if (const auto device = memory_manager.GpuToCpuAddress(address)) {
                            Tegra::Nx2SemaphoreWaiters::Get().OnWrite(*device, value);
                        }
                    });
                } else {
                    static const u64 watch = [] {
                        const char* text = std::getenv("EDEN_NX2_SEM_WATCH");
                        return text ? u64{std::strtoull(text, nullptr, 16)} : u64{};
                    }();
                    if (watch != 0 && (address & ~u64{0xf}) == (watch & ~u64{0xf})) {
                        LOG_INFO(HW_GPU, "Semaphore watch: QMD release {:#x} = {:#x} flags {:#x}", address, payload, flags);
                    }
                    rasterizer->Query(address, VideoCommon::QueryType::Payload,
                        VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
                }
                return true;
            }, [&](const auto& descriptor) { return DispatchNativeCompute(descriptor); });
        if (error.empty()) {
            LOG_DEBUG(HW_GPU, "Native QMD control action={:#x}, QMD={:#x}", argument, qmd_address);
            return true;
        }
        native_macro->native_execution_failed = true;
        std::array<u32, 128> qmd{};
        if (memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            memory_manager.ReadBlockUnsafe(qmd_address, qmd.data(), sizeof(qmd));
        }
        // Log a failing program's first few dispatches only (QMD words 48-49 hold its code
        // address); it fails the same way every frame.
        static std::unordered_map<u64, u32> failures;
        const u64 program = (u64{qmd[49]} << 32) | qmd[48];
        if (failures[program ^ std::hash<std::string>{}(std::string{error})]++ >= 2) {
            return true;
        }
        LOG_ERROR(HW_GPU, "Native QMD failed: {}", error);
        if (qmd_diagnostics && memory_manager.IsFullyMappedRange(qmd_address, sizeof(qmd))) {
            std::ofstream dump("native-qmd.bin", std::ios::binary);
            dump.write(reinterpret_cast<const char*>(qmd.data()), sizeof(qmd));
            std::ofstream macros("native-dispatch-mme.bin", std::ios::binary);
            macros.write(reinterpret_cast<const char*>(native_macro->native_code.data()), sizeof(native_macro->native_code));
            std::ofstream registers("native-dispatch-registers.bin", std::ios::binary);
            registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
        }
        LOG_ERROR(HW_GPU, "Native compute dispatch unsupported: method={:#x}, value={:#x}, QMD={:#x}, words={:08x}", method, argument,
                  qmd_address, fmt::join(qmd, ","));
        return true;
    }
    default:
        // The inline upload register block is shared with the older class.
        if (method < Regs::NUM_REGS) regs.reg_array[method] = argument;
        return method != KEPLER_COMPUTE_REG_INDEX(exec_upload) && method != KEPLER_COMPUTE_REG_INDEX(data_upload);
    }
}

bool KeplerCompute::NativeInlineQmdUpload() {
    const GPUVAddr address = ((GPUVAddr{native_registers[0xc6]} << 32) |
        native_registers[0xc7]) << 8;
    const GPUVAddr program = (GPUVAddr{native_registers[0xc8 + 49] & 0x1ffff} << 32) |
        native_registers[0xc8 + 48];
    const u32 control = native_registers[0xa7];
    LOG_DEBUG(HW_GPU, "Native inline QMD: address={:#x}, program={:#x}, control={:#x}, grid={}/{}/{}",
        address, program, control, native_registers[0xc8 + 12],
        native_registers[0xc8 + 13] & 0xffff, native_registers[0xc8 + 14] & 0xffff);
    if (static const bool diagnostics = std::getenv("EDEN_NATIVE_DRAW_DIAGNOSTICS") != nullptr;
        diagnostics) {
        std::ofstream dump("native-inline-qmd.bin", std::ios::binary);
        dump.write(reinterpret_cast<const char*>(native_registers.data() + 0xc8), 256);
        std::ofstream registers("native-inline-compute-registers.bin", std::ios::binary);
        registers.write(reinterpret_cast<const char*>(native_registers.data()), sizeof(native_registers));
        if (memory_manager.IsFullyMappedRange(program, 0x10000)) {
            std::vector<u8> code(0x10000);
            memory_manager.ReadBlockUnsafe(program, code.data(), code.size());
            std::ofstream shader("native-inline-compute-shader.bin", std::ios::binary);
            shader.write(reinterpret_cast<const char*>(code.data()), code.size());
        }
    }
    if (!memory_manager.IsFullyMappedRange(address, 256)) {
        native_macro->native_execution_failed = true;
        LOG_ERROR(HW_GPU, "Unmapped native inline QMD destination");
        return false;
    }
    rasterizer->AccelerateInlineToMemory(address, 256,
        std::span<const u8>{reinterpret_cast<const u8*>(native_registers.data() + 0xc8), 256});
    return true;
}

void KeplerCompute::ExecuteS1Run(Core::System& system, const u64* it, const u64* const end) {
    // The front ran the register semantics and decided the inline QMD launches: store the
    // deltas, take the groups, run only what has side effects (no hashing, no register write
    // of the back's own).
    namespace S1 = VideoCommon::V2::S1;
    while (it < end) {
        const u64 word = *it;
        const u32 index = S1::IndexOf(word);
        const u32 value = S1::ValueOf(word);
        switch (S1::TagOf(word)) {
        case S1::Tag::Delta:
            native_registers[index] = value;
            if (index < Regs::NUM_REGS) {
                regs.reg_array[index] = value;
            }
            ++it;
            break;
        case S1::Tag::Exec:
            ++it;
            if ((word & S1::ComputeNoLaunch) != 0) {
                native_registers[index] = value;
                s1_inline_failed = !NativeInlineQmdUpload();
                break;
            }
            if ((word & S1::ComputeInlineLaunch) != 0 && s1_inline_failed) {
                break; // as ProcessNativeMethod: no launch from an unmapped upload
            }
            CallMethod(system, index, value, (word & S1::IsLast) != 0);
            break;
        case S1::Tag::ComputeGroups: {
            const u32 groups = S1::GroupsDirty(word);
            ++it;
            for (u32 bits = groups & VideoCommon::V2::HashedComputeGroups; bits != 0;
                 bits &= bits - 1) {
                v2_groups.hashes[std::countr_zero(bits)] = *it++;
            }
            v2_groups.dirty |= groups;
            v2_groups.valid = true;
            break;
        }
        default:
            it += S1::EntryWords(it);
            break;
        }
    }
}

void KeplerCompute::CallMethod(Core::System& system, u32 method, u32 method_argument, bool is_last_call) {
    if (front_macros && !VideoCommon::GpuSplit::OnBackThread()) {
        ReleaseFront();
    }
    if (native_mode) {
        // Most of the compute class's register writes (QMD fields replayed by its macros for
        // each of thousands of dispatches a second) only store the value: ~4% of title 1's GPU
        // thread went through ProcessNativeMethod's checks for them.
        if (method < native_plain_methods.size() && native_plain_methods[method]) {
            if (VideoCommon::V2::On(VideoCommon::V2::Switch::RegSem)) [[unlikely]] {
                VideoCommon::V2::RegSem::NoteWriteCompute(v2_groups, method,
                                                          native_registers[method],
                                                          method_argument);
                if (v2_compute_verify) {
                    return VerifyComputeWrite(system, method, method_argument, is_last_call);
                }
            }
            native_registers[method] = method_argument;
            if (method < Regs::NUM_REGS) {
                regs.reg_array[method] = method_argument;
            }
            return;
        }
        if (v2_compute_table && method < 0xe00 &&
            (VideoCommon::V2::RegSem::TableCompute()[method] &
             (VideoCommon::V2::RegSem::ComputeClearCode | VideoCommon::V2::RegSem::ComputeAddCode |
              VideoCommon::V2::RegSem::ComputeBindPointer | VideoCommon::V2::RegSem::ComputeBind)) !=
                0) [[unlikely]] {
            return TableComputeMacroWrite(method, method_argument);
        }
        if (v2_compute_verify && method < 0xe00) [[unlikely]] {
            return VerifyComputeWrite(system, method, method_argument, is_last_call);
        }
        if (ProcessNativeMethod(system, method, method_argument, is_last_call)) return;
    }
    CallMethodKepler(system, method, method_argument, is_last_call);
}

void KeplerCompute::CallMethodKepler(Core::System& system, u32 method, u32 method_argument,
                                     bool is_last_call) {
    if (method >= Regs::NUM_REGS) {
        LOG_ERROR(HW_GPU, "Compute register out of range: {:#x}", method);
        return;
    }

    if (!native_mode && VideoCommon::V2::On(VideoCommon::V2::Switch::RegSem)) [[unlikely]] {
        VideoCommon::V2::RegSem::NoteWriteCompute(v2_groups, method, regs.reg_array[method],
                                                  method_argument);
    }
    regs.reg_array[method] = method_argument;

    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(exec_upload): {
        UploadInfo info{.upload_address = upload_address,
                        .exec_address = upload_state.ExecTargetAddress(),
                        .copy_size = upload_state.GetUploadSize(),
                        .was_dirty = upload_dirty};
        uploads.push_back(info);
        upload_state.ProcessExec(regs.exec_upload.linear != 0);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(data_upload): {
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(method_argument, is_last_call);
        break;
    }
    case KEPLER_COMPUTE_REG_INDEX(launch): {
        const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();

        for (auto& data : uploads) {
            const GPUVAddr offset = data.exec_address - launch_desc_loc;
            if (offset / sizeof(u32) == LAUNCH_REG_INDEX(grid_dim_x)) {
                const bool source_dirty = memory_manager.IsMemoryDirty(data.upload_address, data.copy_size);
                if (data.was_dirty || source_dirty) {
                    indirect_compute = {data.upload_address};
                }
            }
        }
        uploads.clear();
        ProcessLaunch();
        indirect_compute = std::nullopt;
        break;
    }
    default:
        break;
    }
}

void KeplerCompute::CallMultiMethod(Core::System& system, u32 method, const u32* base_start, u32 amount, u32 methods_pending) {
    if (native_mode) {
        for (u32 i = 0; i < amount; ++i) CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        return;
    }
    switch (method) {
    case KEPLER_COMPUTE_REG_INDEX(data_upload):
        upload_address = current_dma_segment;
        upload_dirty = current_dirty;
        current_dirty = false;
        upload_state.ProcessData(base_start, amount);
        return;
    default:
        for (u32 i = 0; i < amount; i++) {
            CallMethod(system, method, base_start[i], methods_pending - i <= 1);
        }
        break;
    }
}

bool KeplerCompute::DispatchNativeCompute(const NativeQmdScheduler::Descriptor& qmd) {
    if (VideoCommon::V2::IrCheck::g_active) [[unlikely]] {
        // EDEN_NX2_V2_IR_DRAWS verify: the front predicts QMD launches. Count the launch here,
        // before the host decides to skip or replace its work (the native network skips and
        // replacements below; inline QMDs read "dispatched 0 times" for those).
        VideoCommon::V2::IrCheck::NoteDispatch(GPUVAddr{native_registers[0xad]} << 8);
    }
    launch_description = {};
    launch_description.grid_dim_x.Assign(qmd[12]);
    launch_description.grid_dim_y.Assign(qmd[13] & 0xffff);
    launch_description.grid_dim_z.Assign(qmd[14] & 0xffff);
    launch_description.shared_alloc.Assign(qmd[17] & 0x3ffff);
    launch_description.block_dim_x.Assign(qmd[18] >> 16);
    launch_description.block_dim_y.Assign(qmd[19] & 0xffff);
    launch_description.block_dim_z.Assign(qmd[19] >> 16);
    launch_description.const_buffer_enable_mask.Assign(qmd[20] & 0xff);
    launch_description.local_pos_alloc.Assign(qmd[23] & 0xffffff);
    launch_description.local_crs_alloc.Assign(qmd[50] & 0xffffff);
    regs.code_loc.address_low = qmd[48];
    regs.code_loc.address_high = qmd[49] & 0x1ffff;
    for (u32 slot = 0; slot < NumConstBuffers; ++slot) {
        const u32 high = qmd[33 + slot * 2];
        if (high & 0x1ff00) {
            LOG_ERROR(HW_GPU, "Native compute constant address exceeds supported GPU width");
            return false;
        }
        auto& buffer = launch_description.const_buffer_config[slot];
        buffer.address_low = qmd[32 + slot * 2];
        buffer.address_high.Assign(high & 0xff);
        buffer.size.Assign(((high >> 19) & 0x1fff) << 4);
    }
    if (!launch_description.block_dim_x || !launch_description.block_dim_y || !launch_description.block_dim_z)
        return false;
    dlss_chain.output_pass = false;
    // Fortune's Weave's evaluation is 13 dispatches. Bound incomplete evaluations, including
    // skipped dispatches, so a missing output pass cannot keep the chain active indefinitely.
    constexpr u32 max_dispatches = 64;
    if (dlss_chain.active && ++dlss_chain.dispatches > max_dispatches) {
        dlss_chain.active = false;
        dlss_chain.pending.clear();
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
    }
    // Ocarina of Time's movie network (port of Eglantine 92f119195f): a pass that reads one
    // colour image and stores none retains its colour and opens a skip window; the pass that
    // stores an image upscaling it is replaced by a plain copy and closes it. The gameplay
    // network is left alone: its pair (+0xc0 -> +0xd0) carries the tonemap.
    if (!native_upscale_disabled && !dlss_chain.active &&
        Settings::GetCurrentProgramID() == VideoCommon::OotNetwork::ProgramId &&
        (launch_description.const_buffer_enable_mask & 1)) {
        const auto& cbuf = launch_description.const_buffer_config[0];
        const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
        const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
            if (offset + sizeof(u32) > cbuf.size.Value()) return std::nullopt;
            const u32 handle = memory_manager.Read<u32>(cbuf_address + offset);
            if (handle == 0 || (handle & 0xfffff) > regs.tic.limit) return std::nullopt;
            return GetTICEntry(handle & 0xfffff);
        };
        const auto upscale_pair = [](const std::optional<Texture::TICEntry>& from,
                                     const std::optional<Texture::TICEntry>& to,
                                     bool same_format) {
            if (!from || !to || (same_format && from->format != to->format) ||
                from->Address() == 0 || to->Address() == 0 ||
                from->Address() == to->Address() || !from->IsBlockLinear() ||
                !to->IsBlockLinear())
                return false;
            const u64 from_width = from->Width(), from_height = from->Height();
            const u64 to_width = to->Width(), to_height = to->Height();
            const u64 cross_a = to_width * from_height, cross_b = to_height * from_width;
            return from_width >= 64 && to_width > from_width && to_width <= from_width * 4 &&
                   (cross_a > cross_b ? cross_a - cross_b : cross_b - cross_a) * 32 <= cross_a;
        };
        const auto [reads, stores] = NativeImageHandles(regs.code_loc.Address());
        std::optional<Texture::TICEntry> replacement_input;
        std::optional<Texture::TICEntry> replacement_output;
        if (native_upscale_input && network_skip_remaining > 0 && !stores.empty()) {
            if (const auto store = FindNativeUpscaleOutput(native_upscale_input, stores, image,
                                                           upscale_pair)) {
                replacement_input = native_upscale_input;
                replacement_output = image(*store);
            }
        }
        bool skipping{};
        if (reads.size() == 1 && stores.empty() && reads.front() == 0x58) {
            // Ocarina's movie network reads its colour at +0x58; anchoring the window there
            // keeps gameplay compute untouched.
            RememberNativeUpscaleInput();
            if (native_upscale_input) {
                network_skip_remaining = 32;
                skipping = true;
            }
        } else if (!replacement_input && network_skip_remaining > 0) {
            if (--network_skip_remaining == 0) {
                native_upscale_input.reset();
            }
            skipping = true;
        }
        if (replacement_input && replacement_output) {
            const auto surface = [](const Texture::TICEntry& tic)
                -> std::optional<Fermi2D::Surface> {
                using RTFormat = Tegra::RenderTargetFormat;
                Fermi2D::Surface result{};
                switch (tic.format) {
                case Texture::TextureFormat::B10G11R11:
                    result.format = RTFormat::B10G11R11_FLOAT;
                    break;
                case Texture::TextureFormat::R16G16B16A16:
                    result.format = RTFormat::R16G16B16A16_FLOAT;
                    break;
                case Texture::TextureFormat::A8B8G8R8:
                    result.format = RTFormat::A8B8G8R8_UNORM;
                    break;
                case Texture::TextureFormat::A2B10G10R10:
                    result.format = RTFormat::A2B10G10R10_UNORM;
                    break;
                default:
                    return std::nullopt;
                }
                if (tic.Address() == 0 || !tic.IsBlockLinear()) {
                    return std::nullopt;
                }
                result.linear = Fermi2D::MemoryLayout::BlockLinear;
                result.block_width.Assign(tic.block_width);
                result.block_height.Assign(tic.block_height);
                result.block_depth.Assign(tic.block_depth);
                result.depth = 1;
                result.width = tic.Width();
                result.height = tic.Height();
                result.addr_upper = static_cast<u32>(tic.Address() >> 32);
                result.addr_lower = static_cast<u32>(tic.Address());
                return result;
            };
            const auto source = surface(*replacement_input);
            const auto dest = surface(*replacement_output);
            if (source && dest) {
                // Does the program decode sRGB itself? (Ocarina's movie network does.)
                const bool decodes_srgb = [&] {
                    const GPUVAddr program = regs.code_loc.Address();
                    std::array<u32, 0x400> block{};
                    bool threshold = false, exponent = false;
                    for (u64 offset = 0; offset < 0x20000 && !(threshold && exponent);
                         offset += sizeof(block)) {
                        if (!memory_manager.IsFullyMappedRange(program + offset, sizeof(block)))
                            break;
                        memory_manager.ReadBlock(program + offset, block.data(), sizeof(block));
                        for (size_t word = 0; word < block.size(); word += 4) {
                            const u32 opcode = block[word] & 0xfff;
                            threshold |= opcode == 0x434 && block[word + 1] == 0x292d292d;
                            exponent |= opcode == 0x832 && block[word + 1] == 0x40cc40cc;
                        }
                    }
                    return threshold && exponent;
                }();
                const Fermi2D::Config config{
                    .operation = decodes_srgb ? Fermi2D::Operation::SrcCopySrgbToLinear
                                              : Fermi2D::Operation::SrcCopy,
                    .filter = Fermi2D::Filter::Bilinear,
                    .must_accelerate = true,
                    .dst_x0 = 0,
                    .dst_y0 = 0,
                    .dst_x1 = static_cast<s32>(replacement_output->Width()),
                    .dst_y1 = static_cast<s32>(replacement_output->Height()),
                    .src_x0 = 0,
                    .src_y0 = 0,
                    .src_x1 = static_cast<s32>(replacement_input->Width()),
                    .src_y1 = static_cast<s32>(replacement_input->Height()),
                };
                const bool done = rasterizer->AccelerateSurfaceCopy(*source, *dest, config);
                LOG_DEBUG(HW_GPU,
                          "Native network pass {:#x} replaced by input copy {}->{} ({}): {}",
                          regs.code_loc.Address(), replacement_input->Address(),
                          replacement_output->Address(), decodes_srgb ? "sRGB" : "plain", done);
                native_upscale_input.reset();
                network_skip_remaining = 0;
                if (done) {
                    return true;
                }
                // Run the real pass, and the whole network on later frames.
                native_upscale_disabled = true;
                skipping = false;
            }
        }
        if (skipping) {
            return true;
        }
    }
    bool dispatched = true;
    NoteV2Dispatch(true);
    try {
        rasterizer->DispatchCompute();
    } catch (const VideoCommon::FireEmblemUpscale::DispatchReplaced&) {
        return true;
    } catch (const Shader::Exception& error) {
        if (Common::DkbFrameTrace::Enabled()) {
            static thread_local Common::DkbFrameTrace::RateLimit limit;
            static thread_local u64 rejected{};
            ++rejected;
            if (limit.Ready(std::chrono::steady_clock::now())) {
                LOG_INFO(HW_GPU,
                         "DKB shader rejected stage=compute address={:#x} count={} reason={}",
                         regs.code_loc.Address() + launch_description.program_start, rejected,
                         error.what());
                rejected = 0;
            }
        }
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        static std::unordered_set<std::string> logged;
        if (logged.insert(error.what()).second) {
            LOG_ERROR(HW_GPU, "Native compute shader rejected: {}", error.what());
        }
        dispatched = false;
        if (dlss_chain.active) {
            ++dlss_chain.rejected;
        } else if (std::string_view{error.what()}.find("matrix multiply") !=
                   std::string_view::npos) {
            DescribeNativeResources(qmd);
            LOG_ERROR(HW_GPU, "Native tensor dispatch is unsupported; no validated replacement was available");
        }
    }
    if (dlss_chain.output_pass) {
        return FinishDlssEvaluation() || dispatched;
    }
    return dispatched;
}

namespace {

/// EDEN_NX2_V2_MULTIFRONT: methods that read or change the process-wide QMD dependency table
/// (the signals 0xae-0xb0 and the inline QMD launch 0x107).
void NoteQmdUse(u32 method) {
    if (method == 0xae || method == 0xaf || method == 0xb0 || method == 0x107) {
        VideoCommon::V2::Resolve::MultiFront::Note(
            VideoCommon::V2::Resolve::MultiFront::EventKind::QmdUse, method);
    }
}

} // Anonymous namespace

namespace {
constexpr u32 DLSS_NETWORK_CLEAN_EVALUATIONS = 8;
/// Health checks in a row the network must pass before it is shown (about a second: a network that
/// goes bad does so within its first few evaluations), and fail to be dropped.
constexpr u32 DLSS_NETWORK_GOOD_CHECKS = 30;
constexpr u32 DLSS_NETWORK_BAD_CHECKS = 10;
/// A check fails when more than this percentage of its blocks is mismatched (working networks:
/// at most 0.2% on Fortune's Weave's title and Kunlun Base; frozen ones: 6-10%), or when any output
/// sample is NaN or infinite.
constexpr u32 DLSS_CHECK_MISMATCH_PERCENT = 3;
} // Anonymous namespace

bool KeplerCompute::ProcessGameDlss(u64 program_id, u64 native_key) {
    using namespace VideoCommon::FireEmblemUpscale;
    const auto passes = CapturedPasses(program_id);
    if (!passes) {
        return false;
    }
    if (passes->spatial_only && (native_key == passes->prepass || native_key == passes->output)) {
        TraceDlssContract(native_key, false);
    }
    if (native_key == passes->prepass) {
        dlss_chain.spatial_only = passes->spatial_only;
        CaptureFireEmblemJitter();
        dlss_chain.active = true;
        dlss_chain.rejected = 0;
        dlss_chain.dispatches = 0;
        dlss_chain.pending.clear();
    }
    if (!dlss_chain.active) {
        return false;
    }
    dlss_chain.pending.insert(native_key);
    dlss_chain.output_pass = native_key == passes->output;
    if (dlss_chain.spatial_only) {
        return false;
    }
    const bool replacement = dlss_chain.failure != VideoCore::GameDlssReason::None ||
                             GameDlssBlocker() != VideoCore::GameDlssReason::None;
    if (dlss_chain.output_pass) {
        return replacement && FinishDlssEvaluation();
    }
    return dlss_chain.failure != VideoCore::GameDlssReason::None &&
           native_key != passes->prepass &&
           dlss_chain.programs.contains(native_key);
}

// The game's own DLSS network (its translated programs) is shown once every pass of an
// evaluation has run several times in a row and its output has passed several health checks in
// a row. Until then, when any pass can't run, and for the rest of the session once the network
// fails its checks, the replacement upscale draws over the output pass's result. The setting
// "Use the game's own DLSS" defaults off; an explicit native override enables it, while fsr/dlss
// overrides keep the replacement. The network's image sizes are fixed by the game, so it is used
// only without resolution scaling; its matrix multiplies need shaders 32 threads wide.
VideoCore::GameDlssReason KeplerCompute::GameDlssBlocker() const {
    using VideoCore::GameDlssReason;
    static const char* const option = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
    const bool forced_replacement = option && VideoCommon::FireEmblemUpscale::UseReplacement(
        VideoCommon::FireEmblemUpscale::ProgramId,
        VideoCommon::FireEmblemUpscale::OutputProgramKey, option);
    const bool forced_native = option && !forced_replacement;
    if (forced_replacement ||
        (!forced_native && !Settings::values.game_dlss_network.GetValue())) {
        return GameDlssReason::TurnedOff;
    }
    // NVIDIA DLSS, when turned on and running, comes first.
    if (rasterizer->UsesNvidiaDlss()) {
        return GameDlssReason::NvidiaDlss;
    }
    const auto& scale = Settings::values.resolution_info;
    if (scale.up_scale != 1 || scale.down_shift != 0) {
        return GameDlssReason::ResolutionScaling;
    }
    if (!rasterizer->SupportsGameDlssNetwork()) {
        return GameDlssReason::Unsupported;
    }
    return GameDlssReason::None;
}

// Checks the network's output against its input (on the GPU; the result arrives a few frames
// later) and drops the network for the session after DLSS_NETWORK_BAD_CHECKS failed checks in a
// row: a network that produces NaN, freezes on an old image or draws garbage would otherwise stay
// on screen.
void KeplerCompute::CheckDlssNetwork() {
    dlss_chain.checked = rasterizer->CanProbeGameDlssOutput();
    if (!dlss_chain.checked) {
        return;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x34c ||
        !memory_manager.IsFullyMappedRange(cbuf_address, 0x34c)) {
        return;
    }
    const auto surface = [&](u32 offset) -> std::optional<Fermi2D::Surface> {
        const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
        if (tic_index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{tic_index} * sizeof(Texture::TICEntry),
                sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        const auto tic = GetTICEntry(tic_index);
        using RTFormat = Tegra::RenderTargetFormat;
        Fermi2D::Surface result{};
        switch (tic.format) {
        case Texture::TextureFormat::B10G11R11:
            result.format = RTFormat::B10G11R11_FLOAT;
            break;
        case Texture::TextureFormat::R16G16B16A16:
            result.format = RTFormat::R16G16B16A16_FLOAT;
            break;
        case Texture::TextureFormat::A8B8G8R8:
            result.format = RTFormat::A8B8G8R8_UNORM;
            break;
        case Texture::TextureFormat::A2B10G10R10:
            result.format = RTFormat::A2B10G10R10_UNORM;
            break;
        default:
            return std::nullopt;
        }
        if (tic.Address() == 0 || !tic.IsBlockLinear()) {
            return std::nullopt;
        }
        result.linear = Fermi2D::MemoryLayout::BlockLinear;
        result.block_width.Assign(tic.block_width);
        result.block_height.Assign(tic.block_height);
        result.block_depth.Assign(tic.block_depth);
        result.depth = 1;
        result.width = tic.Width();
        result.height = tic.Height();
        result.addr_upper = static_cast<u32>(tic.Address() >> 32);
        result.addr_lower = static_cast<u32>(tic.Address());
        return result;
    };
    const auto input = surface(0x328);
    const auto output = surface(0x348);
    if (!input || !output) {
        return;
    }
    const auto probe = rasterizer->ProbeGameDlssOutput(*input, *output);
    if (!probe) {
        return;
    }
    const bool invalid = probe->invalid_samples != 0;
    const bool mismatched = probe->mismatched * 100 > probe->blocks * DLSS_CHECK_MISMATCH_PERCENT;
    if (!invalid && !mismatched) {
        ++dlss_chain.good_checks;
        dlss_chain.bad_checks = 0;
        return;
    }
    dlss_chain.good_checks = 0;
    if (++dlss_chain.bad_checks < DLSS_NETWORK_BAD_CHECKS) {
        return;
    }
    dlss_chain.failure = invalid ? VideoCore::GameDlssReason::InvalidOutput
                                 : VideoCore::GameDlssReason::OutputMismatch;
    LOG_WARNING(HW_GPU,
                "Game DLSS network: {} in {} checks in a row (last: {}/{} blocks mismatched, worst "
                "{:.3f}, {} invalid samples); the replacement upscale draws the output from now on",
                VideoCore::GameDlssReasonText(dlss_chain.failure), dlss_chain.bad_checks,
                probe->mismatched, probe->blocks, probe->worst, probe->invalid_samples);
}

// Logs what draws the game's DLSS evaluation, when that changes.
void KeplerCompute::UpdateGameDlssStatus(bool complete, VideoCore::GameDlssReason blocker) {
    using VideoCore::GameDlssReason;
    using VideoCore::GameDlssState;
    VideoCore::GameDlssStatus shown{.state = GameDlssState::Checking};
    if (dlss_chain.native) {
        shown.state = GameDlssState::Network;
    } else if (dlss_chain.failure != GameDlssReason::None) {
        shown = {.state = GameDlssState::Replacement, .reason = dlss_chain.failure};
    } else if (blocker != GameDlssReason::None) {
        shown = {.state = GameDlssState::Replacement, .reason = blocker};
    } else if (!complete) {
        shown = {.state = GameDlssState::Replacement,
                 .reason = GameDlssReason::PassesCannotRun,
                 .rejected_passes = dlss_chain.rejected};
    }
    if (shown.state == dlss_chain.status.state && shown.reason == dlss_chain.status.reason &&
        shown.rejected_passes == dlss_chain.status.rejected_passes) {
        return;
    }
    // A failed check was logged as it happened; the network's start is logged above.
    if (shown.state == GameDlssState::Checking) {
        LOG_INFO(HW_GPU, "Game DLSS network: checking its output; the replacement upscale draws "
                         "meanwhile");
    } else if (shown.reason == GameDlssReason::PassesCannotRun) {
        LOG_INFO(HW_GPU,
                 "Game DLSS network: {} pass(es) can't run, the replacement upscale "
                 "draws the output",
                 shown.rejected_passes);
    } else if (shown.state == GameDlssState::Replacement && shown.reason != dlss_chain.failure) {
        LOG_INFO(HW_GPU, "Game DLSS network: {}, the replacement upscale draws the output",
                 VideoCore::GameDlssReasonText(shown.reason));
    }
    dlss_chain.status = shown;
}

// Ends this frame's DLSS evaluation at its output pass: either the network's own result stays,
// or the replacement upscale draws over it. Returns whether the replacement ran.
bool KeplerCompute::FinishDlssEvaluation() {
    dlss_chain.active = false;
    dlss_chain.output_pass = false;
    if (dlss_chain.spatial_only) {
        dlss_chain.pending.clear();
        using namespace VideoCommon::FireEmblemUpscale;
        if (UseReplacement(StarFoxProgramId, StarFoxOutputProgramKey,
                           std::getenv("EDEN_NX2_DLSS_REPLACEMENT"))) {
            return TryFireEmblemUpscale(false);
        }
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        return false;
    }
    dlss_chain.programs.insert(dlss_chain.pending.begin(), dlss_chain.pending.end());
    dlss_chain.pending.clear();
    const bool complete = dlss_chain.rejected == 0;
    dlss_chain.clean_evaluations = complete ? dlss_chain.clean_evaluations + 1 : 0;
    ++dlss_chain.evaluations;
    using VideoCore::GameDlssReason;
    const GameDlssReason blocker = GameDlssBlocker();
    const bool can_show =
        blocker == GameDlssReason::None && dlss_chain.failure == GameDlssReason::None;
    // Checked before the replacement upscale draws over the output. Checks start over whenever
    // the network can't run.
    if (complete && can_show) {
        CheckDlssNetwork();
    } else {
        dlss_chain.good_checks = 0;
        dlss_chain.bad_checks = 0;
    }
    bool native = complete && can_show && dlss_chain.failure == GameDlssReason::None;
    if (!dlss_chain.native) {
        native = native && dlss_chain.clean_evaluations >= DLSS_NETWORK_CLEAN_EVALUATIONS &&
                 (!dlss_chain.checked || dlss_chain.good_checks >= DLSS_NETWORK_GOOD_CHECKS);
    }
    if (native != dlss_chain.native) {
        dlss_chain.native = native;
        if (native) {
            LOG_INFO(HW_GPU, "Game DLSS network: every pass runs{}, its own output is shown",
                     dlss_chain.checked ? " and its output checks out" : "");
        }
    }
    UpdateGameDlssStatus(complete, blocker);
    if (native) {
        dlss_motion_vectors.reset();
        dlss_depth.reset();
        (void)fe_upscale_jitter.Take();
        return false;
    }
    return TryFireEmblemUpscale();
}

// Logs the images a rejected program's handles (constant bank 0, bytes 0x80-0x11c) point to, so a
// substitute (e.g. an upscale blit for a tensor-core pass) can find its inputs and output.
void KeplerCompute::DescribeNativeResources(const NativeQmdScheduler::Descriptor& qmd) {
    static u32 described{};
    if (described++ >= 2) return;
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    LOG_INFO(HW_GPU, "Native resources: cbuf0 {:#x} size {:#x}, grid {}x{}x{}, block {}x{}x{}, tic pool {:#x}",
             cbuf_address, cbuf.size.Value(), qmd[12], qmd[13] & 0xffff, qmd[14] & 0xffff,
             launch_description.block_dim_x.Value(), launch_description.block_dim_y.Value(),
             launch_description.block_dim_z.Value(), regs.tic.Address());
    for (u32 offset = 0x80; offset < 0x120; offset += 4) {
        const u32 handle = memory_manager.Read<u32>(cbuf_address + offset);
        const u32 tic_index = handle & 0xfffff;
        if (handle == 0 || tic_index > regs.tic.limit) continue;
        const auto tic = GetTICEntry(tic_index);
        LOG_INFO(HW_GPU, "  handle @{:#x} = {:#x}: tic {} address {:#x} {}x{} format {:#x} types {}/{}/{}/{}",
                 offset, handle, tic_index, tic.Address(), tic.Width(), tic.Height(),
                 static_cast<u32>(tic.format.Value()), static_cast<u32>(tic.r_type.Value()),
                 static_cast<u32>(tic.g_type.Value()), static_cast<u32>(tic.b_type.Value()),
                 static_cast<u32>(tic.a_type.Value()));
    }
}

std::optional<Texture::TICEntry> KeplerCompute::ConstBuffer0Image(u32 offset) const {
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < offset + 4 ||
        !memory_manager.IsFullyMappedRange(cbuf_address + offset, 4)) {
        return std::nullopt;
    }
    const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
    if (tic_index > regs.tic.limit ||
        !memory_manager.IsFullyMappedRange(regs.tic.Address() +
                                               u64{tic_index} * sizeof(Texture::TICEntry),
                                           sizeof(Texture::TICEntry))) {
        return std::nullopt;
    }
    return GetTICEntry(tic_index);
}

bool KeplerCompute::BypassOotNetwork(u64 program_id, u64 native_key) {
    using namespace VideoCommon::OotNetwork;
    if (!BypassEnabled(program_id)) {
        return false;
    }
    auto* const bypass = rasterizer->OotNetworkBypass();
    if (!bypass) {
        return false;
    }
    // The tonemap pass draws before the network every frame; until it has been seen (menus, the
    // first frame) the network runs as usual. Its first passes are skipped only while the output
    // pass is being replaced, so a failed replacement brings the whole network back next frame.
    Maxwell3D::Regs::RenderTargetConfig tonemapped;
    {
        std::scoped_lock lock{bypass->mutex};
        tonemapped = bypass->tonemapped;
    }
    if (tonemapped.Address() == 0) {
        return false;
    }
    if (native_key == PrepassProgramKey || native_key == NetworkProgramKey) {
        return bypass->output_replaced.load(std::memory_order_relaxed);
    }
    if (native_key != OutputProgramKey) {
        return false;
    }
    // The copy needs the tonemapped image to be a B10G11R11 block-linear image of the output's size.
    const auto output = ConstBuffer0Image(OutputImageHandleOffset);
    if (!output || output->format != Texture::TextureFormat::B10G11R11 || output->Address() == 0 ||
        output->Address() == tonemapped.Address() || !output->IsBlockLinear() ||
        tonemapped.format != Tegra::RenderTargetFormat::B10G11R11_FLOAT ||
        tonemapped.tile_mode.is_pitch_linear || tonemapped.width != output->Width() ||
        tonemapped.height != output->Height()) {
        bypass->output_replaced.store(false, std::memory_order_relaxed);
        static bool warned = false;
        if (!std::exchange(warned, true)) {
            LOG_WARNING(HW_GPU,
                        "Ocarina of Time: unexpected network images; not replaced (tonemapped {}x{}, "
                        "output {})",
                        tonemapped.width, tonemapped.height,
                        output ? fmt::format("{}x{}", output->Width(), output->Height())
                               : std::string{"unreadable"});
        }
        return false;
    }
    Fermi2D::Surface destination{};
    destination.format = Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    destination.linear = Fermi2D::MemoryLayout::BlockLinear;
    destination.block_width.Assign(output->block_width);
    destination.block_height.Assign(output->block_height);
    destination.block_depth.Assign(output->block_depth);
    destination.depth = 1;
    destination.width = output->Width();
    destination.height = output->Height();
    destination.addr_upper = static_cast<u32>(output->Address() >> 32);
    destination.addr_lower = static_cast<u32>(output->Address());
    Fermi2D::Surface source = destination;
    source.block_width.Assign(tonemapped.tile_mode.block_width);
    source.block_height.Assign(tonemapped.tile_mode.block_height);
    source.block_depth.Assign(tonemapped.tile_mode.block_depth);
    source.addr_upper = tonemapped.address_high;
    source.addr_lower = tonemapped.address_low;
    const s32 width = static_cast<s32>(output->Width());
    const s32 height = static_cast<s32>(output->Height());
    const Fermi2D::Config config{
        .operation = Fermi2D::Operation::SrcCopy,
        .filter = Fermi2D::Filter::Point,
        .must_accelerate = true,
        .dst_x0 = 0,
        .dst_y0 = 0,
        .dst_x1 = width,
        .dst_y1 = height,
        .src_x0 = 0,
        .src_y0 = 0,
        .src_x1 = width,
        .src_y1 = height,
    };
    const bool copied = rasterizer->AccelerateSurfaceCopy(source, destination, config);
    bypass->output_replaced.store(copied, std::memory_order_relaxed);
    static bool logged = false;
    if (!std::exchange(logged, true)) {
        LOG_INFO(HW_GPU,
                 "Ocarina of Time: anti-aliasing network bypassed ({}): {:#x} -> {:#x} {}x{}",
                 copied ? "copy of the tonemapped image" : "copy failed", tonemapped.Address(),
                 output->Address(), width, height);
    }
    return copied;
}

std::atomic<u32> g_nx2_trace_draws_after_upscale{};

// The cbuf0 byte offsets of the texture handles a native program reads (TEX, TLD, TLD4, TXD) and
// the surface handles it stores to (SUST.P), from its code up to the first unconditional EXIT.
std::pair<std::vector<u32>, std::vector<u32>> KeplerCompute::NativeImageHandles(GPUVAddr program) {
    static std::unordered_map<GPUVAddr, std::pair<std::vector<u32>, std::vector<u32>>> cache;
    if (const auto it = cache.find(program); it != cache.end()) {
        return it->second;
    }
    std::pair<std::vector<u32>, std::vector<u32>> handles;
    const auto add = [](std::vector<u32>& list, u32 offset) {
        if (std::find(list.begin(), list.end(), offset) == list.end()) list.push_back(offset);
    };
    std::array<u32, 0x400> block{};
    bool exited = false;
    for (u64 offset = 0; offset < 0x40000 && !exited; offset += sizeof(block)) {
        if (!memory_manager.IsFullyMappedRange(program + offset, sizeof(block))) break;
        memory_manager.ReadBlock(program + offset, block.data(), sizeof(block));
        for (size_t word = 0; word < block.size() && !exited; word += 4) {
            const u32 opcode = block[word] & 0xfff;
            const u32 binding = (block[word + 1] >> 22) & 0x1f;
            const u32 handle = ((block[word + 1] >> 8) & 0x3fff) * 4;
            switch (opcode) {
            case 0xb60: case 0xb63: case 0xb66: case 0xb6d:
                if (binding == 0) add(handles.first, handle);
                break;
            case 0xb9b:
                if (binding == 0) add(handles.second, handle);
                break;
            case 0x94d: // EXIT, unconditional when guarded by PT
                exited = ((block[word] >> 12) & 0xf) == 7;
                break;
            default:
                break;
            }
        }
    }
    return cache.emplace(program, std::move(handles)).first->second;
}

void KeplerCompute::RememberNativeUpscaleInput() {
    native_upscale_input.reset();
    const auto [reads, stores] = NativeImageHandles(regs.code_loc.Address());
    // An image-fed matrix pass writes its intermediate results to buffers. Retain its colour
    // input: the final pass's sampled image is another intermediate that a skipped network never
    // produced (Ocarina's movie network reads its colour at +0x58).
    if (reads.size() != 1 || !stores.empty()) return;
    const auto& cbuf = launch_description.const_buffer_config[0];
    if (reads.front() + sizeof(u32) > cbuf.size.Value()) return;
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    const u32 handle = memory_manager.Read<u32>(address + reads.front());
    if (handle == 0 || (handle & 0xfffff) > regs.tic.limit) return;
    const auto input = GetTICEntry(handle & 0xfffff);
    if (input.Address() != 0 && input.IsBlockLinear() && input.Width() >= 64 &&
        input.Height() >= 64) {
        native_upscale_input = input;
    }
}

// Fire Emblem: Fortune's Weave's first DLSS pass (rejected, before the network) has the input colour
// at cbuf0+0x278, its motion vectors (R16G16 float, the input's size) at +0x290, its depth (Z32, the
// input's size) at +0x298 and this frame's camera jitter (x, y; input pixels, Halton bases 2 and 3
// less 0.5) at +0x1b0.
void KeplerCompute::RememberDlssMotionVectors() {
    dlss_motion_vectors.reset();
    dlss_depth.reset();
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr cbuf_address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x294 ||
        !memory_manager.IsFullyMappedRange(cbuf_address, cbuf.size.Value())) {
        return;
    }
    const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
        const u32 tic_index = memory_manager.Read<u32>(cbuf_address + offset) & 0xfffff;
        if (tic_index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{tic_index} * sizeof(Texture::TICEntry),
                sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        return GetTICEntry(tic_index);
    };
    const auto color = image(0x278), motion = image(0x290);
    if (!color || !motion || color->format != Texture::TextureFormat::B10G11R11 ||
        motion->format != Texture::TextureFormat::R16G16 || motion->Address() == 0 ||
        motion->Width() != color->Width() || motion->Height() != color->Height() ||
        !motion->IsBlockLinear())
        return;
    dlss_motion_vectors = *motion;
    dlss_motion_color = color->Address();
    const auto depth = cbuf.size.Value() >= 0x29c ? image(0x298) : std::nullopt;
    if (depth && depth->format == Texture::TextureFormat::Z32 && depth->Address() != 0 &&
        depth->Width() == color->Width() && depth->Height() == color->Height() &&
        depth->IsBlockLinear()) {
        dlss_depth = *depth;
    } else {
        dlss_depth.reset();
    }
}

void KeplerCompute::CaptureFireEmblemJitter() {
    using namespace VideoCommon::FireEmblemUpscale;
    (void)fe_upscale_jitter.Take();
    dlss_motion_vectors.reset();
    dlss_depth.reset();
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x27c ||
        !memory_manager.IsFullyMappedRange(address, 0x27c)) {
        return;
    }
    const u32 index = memory_manager.Read<u32>(address + PrepassColorOffset) & 0xfffff;
    if (index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
            regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry), sizeof(Texture::TICEntry))) {
        return;
    }
    const auto color = GetTICEntry(index);
    if (!ValidInput(color, dlss_chain.spatial_only)) {
        return;
    }
    fe_upscale_jitter.Capture(color.Address(), color.Width(), color.Height(),
        {std::bit_cast<f32>(memory_manager.Read<u32>(address + JitterOffset)),
         std::bit_cast<f32>(memory_manager.Read<u32>(address + JitterOffset + sizeof(f32)))});
    if (!dlss_chain.spatial_only) {
        RememberDlssMotionVectors();
    }
}

void KeplerCompute::TraceDlssContract(u64 native_key, bool warn_unvalidated) {
    static bool reported{};
    if (warn_unvalidated && !reported) {
        reported = true;
        LOG_WARNING(HW_GPU, "Fast Fusion temporal replacement is not selected: motion, depth, jitter and reset contract is unvalidated; native dispatches retained. EDEN_NX2_DLSS_CONTRACT_TRACE enables bounded evidence capture");
    }
    static const bool enabled = std::getenv("EDEN_NX2_DLSS_CONTRACT_TRACE") != nullptr;
    static const u32 skip = [] {
        const char* value = std::getenv("EDEN_NX2_DLSS_CONTRACT_SKIP");
        return value ? static_cast<u32>(std::min<unsigned long>(std::strtoul(value, nullptr, 10), 1000000UL)) : 0U;
    }();
    if (!enabled || dlss_contract_trace_count >= skip + 24) {
        return;
    }
    if (dlss_contract_trace_count++ < skip) {
        return;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    const u32 bytes = std::min<u32>(cbuf.size.Value(), 0x360U);
    if (!(launch_description.const_buffer_enable_mask & 1) || bytes < 0x100 ||
        !memory_manager.IsFullyMappedRange(address, bytes)) {
        return;
    }
    LOG_INFO(HW_GPU, "DLSS contract candidate {:016x}: cb0={:#x} bytes={:#x} grid={}x{}x{} block={}x{}x{} native dispatch retained",
             native_key, address, bytes, launch_description.grid_dim_x.Value(),
             launch_description.grid_dim_y.Value(), launch_description.grid_dim_z.Value(),
             launch_description.block_dim_x.Value(), launch_description.block_dim_y.Value(),
             launch_description.block_dim_z.Value());
    for (u32 offset = 0x100; offset + 16 <= bytes; offset += 16) {
        std::array<u32, 4> words{};
        memory_manager.ReadBlock(address + offset, words.data(), sizeof(words));
        LOG_INFO(HW_GPU, "DLSS contract cb0+{:03x}: {:08x}", offset, fmt::join(words, " "));
    }
    for (u32 offset = 0x278; offset + 4 <= bytes; offset += 4) {
        const u32 handle = memory_manager.Read<u32>(address + offset);
        const u32 index = handle & 0xfffff;
        const GPUVAddr tic_address = regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry);
        if (!handle || index > regs.tic.limit ||
            !memory_manager.IsFullyMappedRange(tic_address, sizeof(Texture::TICEntry))) {
            continue;
        }
        const auto tic = GetTICEntry(index);
        std::array<u32, sizeof(Texture::TICEntry) / sizeof(u32)> words{};
        std::memcpy(words.data(), &tic, sizeof(tic));
        LOG_INFO(HW_GPU, "DLSS contract handle+{:03x}={:08x}: tic={} addr={:#x} {}x{} raw={:08x}",
                 offset, handle, index, tic.Address(), tic.Width(), tic.Height(),
                 fmt::join(words, " "));
    }
}

bool KeplerCompute::TryFireEmblemUpscale(bool allow_history) {
    using namespace VideoCommon::FireEmblemUpscale;
    static const bool requested_history = [] {
        const char* const option = std::getenv("EDEN_NX2_DLSS_REPLACEMENT");
        return option && (std::string_view{option} == "fsr" || std::string_view{option} == "dlss");
    }();
    const bool use_history = allow_history && requested_history;
    const size_t game_index = dlss_chain.spatial_only ? 1 : 0;
    const char* const game = dlss_chain.spatial_only ? "Star Fox" : "Fire Emblem";
    const auto jitter = fe_upscale_jitter.Take();
    const auto motion = std::exchange(dlss_motion_vectors, std::nullopt);
    const auto depth = std::exchange(dlss_depth, std::nullopt);
    if (!jitter || (use_history && (!motion || !depth))) {
        static std::array<bool, 2> reported{};
        if (!reported[game_index]) {
            reported[game_index] = true;
            LOG_WARNING(HW_GPU, "{} replacement lacks current inputs: jitter={} motion={} depth={} temporal={}; retaining native dispatch",
                        game, jitter.has_value(), motion.has_value(), depth.has_value(), use_history);
        }
        return false;
    }
    const auto& cbuf = launch_description.const_buffer_config[0];
    const GPUVAddr address = (GPUVAddr{cbuf.address_high} << 32) | cbuf.address_low;
    if (!(launch_description.const_buffer_enable_mask & 1) || cbuf.size.Value() < 0x34c ||
        !memory_manager.IsFullyMappedRange(address, 0x34c)) {
        return false;
    }
    const auto image = [&](u32 offset) -> std::optional<Texture::TICEntry> {
        const u32 index = memory_manager.Read<u32>(address + offset) & 0xfffff;
        if (index > regs.tic.limit || !memory_manager.IsFullyMappedRange(
                regs.tic.Address() + u64{index} * sizeof(Texture::TICEntry), sizeof(Texture::TICEntry))) {
            return std::nullopt;
        }
        return GetTICEntry(index);
    };
    const auto input = image(OutputColorInputOffset);
    const auto output = image(OutputColorOffset);
    if (!input || !output || !ValidPair(*input, *output, dlss_chain.spatial_only) ||
        jitter->color != input->Address() || (use_history && dlss_motion_color != input->Address()) ||
        jitter->width != input->Width() || jitter->height != input->Height()) {
        return false;
    }
    const auto surface = [](const Texture::TICEntry& tic, Tegra::RenderTargetFormat format) {
        Fermi2D::Surface result{};
        result.format = format;
        result.linear = Fermi2D::MemoryLayout::BlockLinear;
        result.block_width.Assign(tic.block_width);
        result.block_height.Assign(tic.block_height);
        result.block_depth.Assign(tic.block_depth);
        result.depth = 1;
        result.width = tic.Width();
        result.height = tic.Height();
        result.addr_upper = static_cast<u32>(tic.Address() >> 32);
        result.addr_lower = static_cast<u32>(tic.Address());
        return result;
    };
    const auto input_format = dlss_chain.spatial_only ? Tegra::RenderTargetFormat::A8B8G8R8_SRGB
                                                     : Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    const auto output_format = dlss_chain.spatial_only ? Tegra::RenderTargetFormat::R16G16B16A16_FLOAT
                                                      : Tegra::RenderTargetFormat::B10G11R11_FLOAT;
    const bool done = rasterizer->AccelerateTemporalUpscale(
        surface(*input, input_format), surface(*output, output_format),
        use_history ? surface(*motion, Tegra::RenderTargetFormat::R16G16_FLOAT) : Fermi2D::Surface{},
        use_history ? &*depth : nullptr, jitter->offset,
        use_history ? std::array{1.0f, 1.0f} : std::array{0.0f, 0.0f}, use_history);
    static std::array<u32, 2> logged{};
    if (logged[game_index]++ < 8) {
        LOG_INFO(HW_GPU, "{} DLSS replacement: {} temporal={} input={:#x} {}x{} output={:#x} {}x{} jitter={},{}",
                 game, done ? "applied" : "unavailable; retaining native dispatch", use_history,
                 input->Address(), input->Width(), input->Height(), output->Address(),
                 output->Width(), output->Height(), jitter->offset[0], jitter->offset[1]);
    }
    return done;
}

void KeplerCompute::TableComputeMacroWrite(u32 method, u32 argument) {
    // EDEN_NX2_V2_REGSEM=1: the compute macro upload and bind registers through the table-driven
    // update function the front mirror uses (RegSem::ApplyCompute).
    namespace RegSem = VideoCommon::V2::RegSem;
    if (front_macros) {
        // The split's front loaded and bound them (ProcessNativeMethod stores only).
        RegSem::NoteWriteCompute(v2_groups, method, native_registers[method], argument);
        native_registers[method] = argument;
        return;
    }
    bool uploaded_without_context = false;
    struct BackState {
        std::array<u32, 0x1000>& registers;
        bool& compute_uploaded;
        u32& bind_pointer;
        VideoCommon::V2::ComputeGroupTracker& v2_groups;
    } back_state{native_registers,
                 native_context ? native_compute_uploaded : uploaded_without_context,
                 native_bind_pointer, v2_groups};
    const auto select = [this] {
        if (native_context) {
            native_macro = native_compute_uploaded ? &native_context->compute_engine
                                                   : &native_context->engine;
            native_macro_positions = native_compute_uploaded
                                         ? std::span<u32>{native_context->compute_positions}
                                         : std::span<u32>{native_context->positions};
        }
    };
    RegSem::ApplyCompute(back_state, method, argument, [&](RegSem::Effect effect, u32 a, u32 b) {
        select();
        switch (effect) {
        case RegSem::Effect::ClearCode:
            native_macro->ClearCode(a);
            return true;
        case RegSem::Effect::AddCode:
            native_macro->AddCode(a, b);
            return true;
        case RegSem::Effect::Bind:
            break;
        }
        if (a < native_macro_positions.size()) {
            native_macro_positions[a] = b;
            return true;
        }
        LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        return false;
    });
    select();
}

void KeplerCompute::VerifyComputeWrite(Core::System& system, u32 method, u32 argument,
                                       bool is_last_call) {
    // EDEN_NX2_V2_REGSEM=verify: the engine's own path decides; the table-driven update runs on
    // a copy of the registers and is compared.
    namespace RegSem = VideoCommon::V2::RegSem;
    if (!v2_compute_mirror) {
        v2_compute_mirror = std::make_unique<RegSem::MirrorCompute>();
        v2_compute_mirror->registers = native_registers;
        v2_compute_mirror->compute_uploaded = native_compute_uploaded;
        v2_compute_mirror->bind_pointer = native_bind_pointer;
    }
    RegSem::MirrorCompute& copy = *v2_compute_mirror;
    if (native_plain_methods[method]) {
        // CallMethod's plain store (the group tracker took it already).
        native_registers[method] = argument;
        if (method < Regs::NUM_REGS) {
            regs.reg_array[method] = argument;
        }
    } else if (!ProcessNativeMethod(system, method, argument, is_last_call)) {
        // The Kepler class's upload registers: the old path continues in CallMethod.
        CallMethodKepler(system, method, argument, is_last_call);
    }
    const bool macro_method = (RegSem::TableCompute()[method] &
                               (RegSem::ComputeClearCode | RegSem::ComputeAddCode |
                                RegSem::ComputeBindPointer | RegSem::ComputeBind)) != 0;
    if (front_macros && macro_method) {
        copy.registers[method] = argument;
    } else {
        const size_t positions = native_macro_positions.size();
        RegSem::ApplyCompute(copy, method, argument, [positions](RegSem::Effect effect, u32 a, u32) {
            return effect != RegSem::Effect::Bind || a < positions;
        });
        if (!native_context) {
            copy.compute_uploaded = native_compute_uploaded;
        }
    }
    if ((RegSem::TableCompute()[method] & RegSem::ComputeInlineQmd) != 0) {
        // The inline QMD's launch runs the engine's 0xb0 method, which stores its signal there
        // (not a write of the command stream; the table doesn't model it).
        copy.registers[0xb0] = native_registers[0xb0];
    }
    bool same = copy.registers[method] == native_registers[method] &&
                copy.compute_uploaded == native_compute_uploaded &&
                copy.bind_pointer == native_bind_pointer;
    // The register that differs: the one written, or the first one the full compare finds.
    u32 differs = method;
    if (same && (++v2_compute_writes & 4095) == 0) {
        const auto [table_it, engine_it] = std::ranges::mismatch(copy.registers, native_registers);
        if (table_it != copy.registers.end()) {
            same = false;
            differs = static_cast<u32>(table_it - copy.registers.begin());
        }
    }
    if (!same) {
        if (RegSem::ReportMismatch()) {
            LOG_WARNING(HW_GPU,
                        "V2 regsem compute back write {:#x} = {:#x}: register {:#x} differs, "
                        "register table {:#x} engine {:#x}, uploaded {} / {}, bind pointer {} / {}",
                        method, argument, differs, copy.registers[differs],
                        native_registers[differs], copy.compute_uploaded, native_compute_uploaded,
                        copy.bind_pointer, native_bind_pointer);
        }
        copy.registers = native_registers;
        copy.compute_uploaded = native_compute_uploaded;
        copy.bind_pointer = native_bind_pointer;
    }
}

void KeplerCompute::NoteV2Dispatch(bool native) {
    if (!native && VideoCommon::V2::IrCheck::g_active) [[unlikely]] {
        // Native launches are counted when the QMD launches (DispatchNativeCompute).
        VideoCommon::V2::IrCheck::NoteDispatch(0);
    }
    if (VideoCommon::V2::On(VideoCommon::V2::Switch::RegSem)) [[unlikely]] {
        if (native) {
            VideoCommon::V2::RegSem::NoteDispatchCompute(v2_groups, native_registers.data(),
                                                         static_cast<u32>(native_registers.size()));
        } else {
            VideoCommon::V2::RegSem::NoteDispatchCompute(v2_groups, regs.reg_array.data(),
                                                         Regs::NUM_REGS);
        }
    }
}

void KeplerCompute::ProcessLaunch() {
    const GPUVAddr launch_desc_loc = regs.launch_desc_loc.Address();
    memory_manager.ReadBlockUnsafe(launch_desc_loc, &launch_description,
                                   LaunchParams::NUM_LAUNCH_PARAMETERS * sizeof(u32));
    NoteV2Dispatch(false);
    rasterizer->DispatchCompute();
}

Texture::TICEntry KeplerCompute::GetTICEntry(u32 tic_index) const {
    const GPUVAddr tic_address_gpu{regs.tic.Address() + tic_index * sizeof(Texture::TICEntry)};

    Texture::TICEntry tic_entry;
    memory_manager.ReadBlockUnsafe(tic_address_gpu, &tic_entry, sizeof(Texture::TICEntry));
    return tic_entry;
}

Texture::TSCEntry KeplerCompute::GetTSCEntry(u32 tsc_index) const {
    const GPUVAddr tsc_address_gpu{regs.tsc.Address() + tsc_index * sizeof(Texture::TSCEntry)};

    Texture::TSCEntry tsc_entry;
    memory_manager.ReadBlockUnsafe(tsc_address_gpu, &tsc_entry, sizeof(Texture::TSCEntry));
    return tsc_entry;
}

// Split GPU thread with macros on its front half (gpu_split.h, EDEN_NX2_SPLIT=2): the front
// runs compute macros against its own copy of the native registers (which hold the last value
// written to each, on the back too) and loads and binds the compute macro code itself.
struct KeplerCompute::FrontState {
    std::array<u32, 0x1000> registers{};
    std::optional<u32> executing_macro;
    std::vector<u32> parameters;
    bool compute_uploaded{};
    u32 bind_pointer{};
    MacroEngine* macro{}; ///< Without a native context: the one macro engine
    /// EDEN_NX2_V2_REGSEM: the table-driven mirror (=1), or a copy compared with this (verify).
    VideoCommon::V2::Mode v2_mode{};
    std::unique_ptr<VideoCommon::V2::RegSem::MirrorCompute> v2_mirror;
    u64 v2_writes{};
    std::optional<std::pair<GPUVAddr, u64>> v2_upload; ///< The upload being sent (P2)
    /// EDEN_NX2_V2_REGSEM=1: compute register groups and fingerprints of `registers`.
    VideoCommon::V2::ComputeGroupTracker v2_groups;
    /// EDEN_NX2_V2_QMDSCHED: on (1) or checked (verify: exact Dispatch records, old hazard
    /// notes). Needs the front's write tracking (EDEN_NX2_V2_ACQUIRE or _HAZARD).
    VideoCommon::V2::Mode qmd_model{};
    struct QmdSignal {
        bool valid{};
        u32 dispatches{};
        bool unbounded{}; ///< a launched QMD writes where the front can't bound it
        std::vector<std::pair<GPUVAddr, u64>> writes;   ///< storage buffer ranges
        std::vector<std::pair<GPUVAddr, u64>> releases; ///< release words
    };
    std::deque<QmdSignal> qmd_signals; ///< this write's signals, consumed by the notes
    /// S1 single owner (GpuSplit::S1Exec, REGSEM=1, the native class): classified entries
    /// instead of method words (F1-05).
    bool s1{};
    /// EDEN_NX2_V2_MACROMEMO: replayed macro calls; EDEN_NX2_V2_STATS: calls per macro (F5).
    std::unique_ptr<VideoCommon::V2::MacroMemo::Memo> v2_memo;
    std::unique_ptr<VideoCommon::V2::MacroMemo::CallStats> v2_call_stats;
};

KeplerCompute::FrontState& KeplerCompute::Front(VideoCommon::GpuSplit& split) {
    if (!front) [[unlikely]] {
        // Copy the back's state and take the macros over while it is idle.
        split.Drain(VideoCommon::DrainSite::FrontStart);
        front = std::make_unique<FrontState>();
        front->registers = native_registers;
        front->executing_macro = native_executing_macro;
        front->parameters = std::move(native_parameters);
        native_parameters.clear();
        native_executing_macro.reset();
        front->compute_uploaded = native_compute_uploaded;
        front->bind_pointer = native_bind_pointer;
        front->macro = native_macro;
        front_macros = true;
        front->qmd_model = split.TracksWrites()
                               ? VideoCommon::V2::Get(VideoCommon::V2::Switch::QmdSched)
                               : VideoCommon::V2::Mode::Off;
        if (front->qmd_model != VideoCommon::V2::Mode::Off) {
            // Drained above: the back's table is current.
            VideoCommon::V2::QmdFront::FrontModel().Resync(Tegra::NativeQmdScheduler::BackTable());
        }
        front->v2_mode = VideoCommon::V2::Get(VideoCommon::V2::Switch::RegSem);
        if (front->v2_mode == VideoCommon::V2::Mode::Verify) [[unlikely]] {
            front->v2_mirror = std::make_unique<VideoCommon::V2::RegSem::MirrorCompute>();
            front->v2_mirror->registers = front->registers;
            front->v2_mirror->compute_uploaded = front->compute_uploaded;
            front->v2_mirror->bind_pointer = front->bind_pointer;
            VideoCommon::V2::RegSem::RehashCompute(front->v2_mirror->v2_groups,
                                                   front->v2_mirror->registers.data());
        } else if (front->v2_mode == VideoCommon::V2::Mode::On) [[unlikely]] {
            VideoCommon::V2::RegSem::RehashCompute(front->v2_groups, front->registers.data());
            // Drained above: the back's groups equal these, and it hashes nothing from now on.
            front->s1 = split.S1Exec() && native_mode;
            s1_compute = front->s1;
            if (front->s1) {
                v2_groups = front->v2_groups;
            }
        }
    }
    return *front;
}

// As Maxwell3D::ReleaseFront: the split stopped, the engine runs its macros again.
void KeplerCompute::ReleaseFront() {
    native_executing_macro = front->executing_macro;
    native_parameters = std::move(front->parameters);
    native_compute_uploaded = front->compute_uploaded;
    native_bind_pointer = front->bind_pointer;
    native_macro = front->macro;
    front.reset();
    front_macros = false;
    s1_compute = false;
    LOG_INFO(HW_GPU, "GPU thread split stopped: compute macros back on the engine");
}

void KeplerCompute::FrontApply(FrontState& fs, u32 method, u32 argument) {
    if (fs.v2_mode != VideoCommon::V2::Mode::Off) [[unlikely]] {
        return FrontApplyRegSem(fs, method, argument);
    }
    FrontApplyV1(fs, method, argument);
}

void KeplerCompute::FrontApplyRegSem(FrontState& fs, u32 method, u32 argument) {
    namespace RegSem = VideoCommon::V2::RegSem;
    if (fs.v2_mode == VideoCommon::V2::Mode::On) {
        RegSem::ApplyCompute(fs, method, argument, [this, &fs](RegSem::Effect effect, u32 a, u32 b) {
            switch (effect) {
            case RegSem::Effect::ClearCode:
                FrontMacroEngine(fs).ClearCode(a);
                return true;
            case RegSem::Effect::AddCode:
                FrontMacroEngine(fs).AddCode(a, b);
                return true;
            case RegSem::Effect::Bind:
                break;
            }
            const auto positions = FrontMacroPositions(fs);
            if (a < positions.size()) {
                positions[a] = b;
                return true;
            }
            LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
            return false;
        });
        return;
    }
    // Verify: the hand-written mirror decides, the table-driven copy follows.
    FrontApplyV1(fs, method, argument);
    RegSem::MirrorCompute& copy = *fs.v2_mirror;
    RegSem::ApplyCompute(copy, method, argument, [this, &copy](RegSem::Effect effect, u32 a, u32) {
        if (effect != RegSem::Effect::Bind) {
            return true;
        }
        const size_t size = native_context ? (copy.compute_uploaded
                                                  ? native_context->compute_positions.size()
                                                  : native_context->positions.size())
                                           : native_macro_positions.size();
        return a < size;
    });
    bool same = copy.registers[method] == fs.registers[method] &&
                copy.compute_uploaded == fs.compute_uploaded &&
                copy.bind_pointer == fs.bind_pointer;
    if (same && (++fs.v2_writes & 4095) == 0) {
        same = copy.registers == fs.registers;
    }
    if (!same) {
        if (RegSem::ReportMismatch()) {
            LOG_WARNING(HW_GPU,
                        "V2 regsem compute write {:#x} = {:#x}: register table {:#x} hand {:#x}, "
                        "uploaded {} / {}, bind pointer {} / {}",
                        method, argument, copy.registers[method], fs.registers[method],
                        copy.compute_uploaded, fs.compute_uploaded, copy.bind_pointer,
                        fs.bind_pointer);
        }
        copy.registers = fs.registers;
        copy.compute_uploaded = fs.compute_uploaded;
        copy.bind_pointer = fs.bind_pointer;
        VideoCommon::V2::RegSem::RehashCompute(copy.v2_groups, copy.registers.data());
    }
}

void KeplerCompute::FrontApplyV1(FrontState& fs, u32 method, u32 argument) {
    // As ProcessNativeMethod does with the macro code and bindings.

    if (method == 0x45 || method == 0x47) {
        fs.compute_uploaded = true;
    }
    fs.registers[method] = argument;
    switch (method) {
    case 0x45:
        FrontMacroEngine(fs).ClearCode(argument);
        break;
    case 0x46:
        FrontMacroEngine(fs).AddCode(0, argument);
        break;
    case 0x47:
        fs.bind_pointer = argument;
        break;
    case 0x48: {
        const auto positions = FrontMacroPositions(fs);
        if (fs.bind_pointer < positions.size()) {
            positions[fs.bind_pointer++] = argument;
        } else {
            LOG_ERROR(HW_GPU, "Native compute macro bind out of bounds");
        }
        break;
    }
    default:
        break;
    }
}

MacroEngine& KeplerCompute::FrontMacroEngine(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? native_context->compute_engine : native_context->engine;
    }
    return *fs.macro;
}

std::span<u32> KeplerCompute::FrontMacroPositions(FrontState& fs) {
    if (native_context) {
        return fs.compute_uploaded ? std::span<u32>{native_context->compute_positions}
                                      : std::span<u32>{native_context->positions};
    }
    return native_macro_positions;
}

void KeplerCompute::FrontCallMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                    u32 method, u32 argument, bool is_last_call,
                                    const GPUVAddr* segment) {
    auto& fs = Front(split);
    if (method >= fs.registers.size()) {
        split.Method(this, method, argument, is_last_call, segment);
        return;
    }
    if (method >= 0xe00) {
        if (!fs.executing_macro) {
            if (method & 1) {
                LOG_ERROR(HW_GPU, "Native compute macro starts on argument method");
                return;
            }
            fs.executing_macro = method;
        }
        fs.parameters.push_back(argument);
        if (is_last_call) {
            FrontRunMacro(system, split, fs);
        }
        return;
    }
    const u32 old_value = fs.registers[method];
    FrontApply(fs, method, argument);
    if (VideoCommon::V2::Resolve::MultiFront::g_recording) [[unlikely]] {
        NoteQmdUse(method);
    }
    if (fs.qmd_model != VideoCommon::V2::Mode::Off) [[unlikely]] {
        FrontModelQmd(split, fs, method, &argument, 1);
    }
    // A data upload word needs its pushbuffer address on the back (the upload source): it
    // stays a method word.
    if (fs.s1 && method != KEPLER_COMPUTE_REG_INDEX(data_upload)) [[likely]] {
        FrontS1Write(split, fs, method, old_value, is_last_call);
    } else {
        if (split.DrawRecords()) [[unlikely]] {
            FrontDispatchRecord(split, fs, method, &argument, 1);
        }
        split.Method(this, method, argument, is_last_call, segment);
    }
    if (split.TracksWrites()) [[unlikely]] {
        FrontNoteWrites(split, fs, method, argument);
    }
}

void KeplerCompute::FrontS1Write(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                 u32 old_value, bool is_last) {
    namespace S1 = VideoCommon::V2::S1;
    const u32 value = fs.registers[method];
    if (method >= 0x45 && method <= 0x48) {
        return; // macro code and bindings live on the front (FrontApply)
    }
    if (native_plain_methods[method]) {
        if (value != old_value) {
            split.S1Word(this, S1::MakeRegister(S1::Tag::Delta, method, value));
        }
        return;
    }
    switch (method) {
    case 0x107: {
        // As ProcessNativeMethod: the QMD upload, then a launch unless it waits for a
        // dependency, with the launch address in 0xad for the signal and restored after it.
        // The register writes are the front's deltas; the back only uploads and signals.
        split.S1Word(this, S1::MakeRegister(S1::Tag::Exec, method, value, S1::ComputeNoLaunch));
        const u32* const r = fs.registers.data();
        if ((r[0xa7] & (1U << 24)) == 0 && (r[0xcc] & (1U << 12)) != 0) {
            return;
        }
        const u32 previous = r[0xad];
        const GPUVAddr address = ((GPUVAddr{r[0xc6]} << 32) | r[0xc7]) << 8;
        FrontS1Store(split, fs, 0xad, static_cast<u32>(address >> 8));
        FrontS1Groups(split, fs);
        FrontApply(fs, 0xb0, 3);
        split.S1Word(this, S1::MakeRegister(S1::Tag::Exec, 0xb0, 3,
                                            S1::ComputeInlineLaunch | S1::IsLast));
        FrontS1Store(split, fs, 0xad, previous);
        return;
    }
    case 0xae:
    case 0xaf:
    case 0xb0:
        FrontS1Groups(split, fs); // a QMD signal may launch work
        break;
    default:
        break;
    }
    split.S1Word(this, S1::MakeRegister(S1::Tag::Exec, method, value, is_last ? S1::IsLast : 0));
}

void KeplerCompute::FrontS1Store(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                 u32 value) {
    namespace S1 = VideoCommon::V2::S1;
    const u32 old_value = fs.registers[method];
    FrontApply(fs, method, value);
    if (fs.registers[method] != old_value) {
        split.S1Word(this, S1::MakeRegister(S1::Tag::Delta, method, fs.registers[method]));
    }
}

void KeplerCompute::FrontS1Groups(VideoCommon::GpuSplit& split, FrontState& fs) {
    namespace S1 = VideoCommon::V2::S1;
    VideoCommon::V2::ComputeGroupTracker& groups = fs.v2_groups;
    const u32 changed = groups.dirty;
    if (changed == 0) {
        return; // the back's groups hold (its dispatches see nothing changed)
    }
    const u32 hashed = changed & VideoCommon::V2::HashedComputeGroups;
    const size_t size = 1 + static_cast<size_t>(std::popcount(hashed));
    u64* const entry = split.S1Begin(this, size);
    entry[0] = static_cast<u64>(S1::Tag::ComputeGroups) | (u64{changed & 0xffff} << 8);
    size_t slot = 1;
    for (u32 bits = hashed; bits != 0; bits &= bits - 1) {
        entry[slot++] = groups.hashes[std::countr_zero(bits)];
    }
    split.S1Commit(size);
    groups.BeginDraw();
}

void KeplerCompute::FrontDispatchRecord(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                        const u32* arguments, u32 amount) {
    // As ProcessNativeMethod: which writes can launch QMDs (EDEN_NX2_V2_IR_DRAWS). The inline
    // QMD (0x107) launches at once unless it waits for a dependency; QMD signals (0xaf, 0xb0)
    // may launch it and the QMDs that depend on it, a number only the back's scheduler knows.
    namespace Ir = VideoCommon::V2::Ir;
    const u32* const r = fs.registers.data();
    Ir::DispatchBody body{};
    body.method = method;
    body.max_dispatches = 0;
    size_t model_index = 0;
    u32 exact_sum = 0;
    bool inexact = false;
    bool any_signal = false;
    for (u32 i = 0; i < amount; ++i) {
        const u32 argument = arguments[i];
        body.argument = argument;
        switch (method) {
        case 0x107:
            if ((r[0xa7] & (1U << 24)) == 0 && (r[0xcc] & (1U << 12)) != 0) {
                continue; // Waits for a dependency: launched by a later signal
            }
            body.qmd_address = ((GPUVAddr{r[0xc6]} << 32) | r[0xc7]) << 8;
            body.program = (GPUVAddr{r[0xc8 + 49] & 0x1ffff} << 32) | r[0xc8 + 48];
            ++body.min_dispatches;
            body.max_dispatches = ~0U;
            break;
        case 0xaf:
        case 0xb0: {
            if (method == 0xaf ? (argument & 2) == 0
                               : ((argument & 15) == 0 || (argument & 15) == 1)) {
                continue;
            }
            body.qmd_address = GPUVAddr{r[0xad]} << 8;
            // EDEN_NX2_V2_QMDSCHED: the front's model knows how many QMDs launch.
            if (model_index < fs.qmd_signals.size() && fs.qmd_signals[model_index].valid) {
                exact_sum += fs.qmd_signals[model_index].dispatches;
            } else {
                inexact = true;
            }
            any_signal = true;
            body.max_dispatches = ~0U;
            ++model_index;
            break;
        }
        default:
            continue;
        }
    }
    if (any_signal && !inexact && fs.qmd_model != VideoCommon::V2::Mode::Off) {
        // Exact (EDEN_NX2_V2_QMDSCHED): a signal that launches nothing needs no record.
        body.min_dispatches = exact_sum;
        body.max_dispatches = exact_sum;
    }
    if (body.max_dispatches == 0) {
        return;
    }
    body.engine = split.Ref(this);
    body.write_begin = Ir::Unknown;
    body.write_end = Ir::Unknown;
    VideoCommon::V2::ComputeGroupTracker* const groups =
        fs.v2_mode == VideoCommon::V2::Mode::On
            ? &fs.v2_groups
            : (fs.v2_mirror ? &fs.v2_mirror->v2_groups : nullptr);
    if (!split.FullDrawRecords()) {
        // Nothing reads the full record (MASTER_PLAN §5.2's compact form).
        Ir::CompactBody compact{};
        compact.full_kind = Ir::Kind::Dispatch;
        compact.count = static_cast<u16>(std::min<u32>(body.max_dispatches, 0xffff));
        compact.engine = body.engine;
        if (groups != nullptr && groups->valid) {
            compact.dirty_groups = groups->dirty;
            compact.fingerprint = groups->Fingerprint(VideoCommon::V2::HashedComputeGroups);
            groups->BeginDraw();
        }
        split.Note(Ir::Kind::Compact, compact);
        return;
    }
    if (groups != nullptr && groups->valid) {
        body.dirty_groups = groups->dirty;
        size_t slot = 0;
        for (u32 bits = VideoCommon::V2::HashedComputeGroups; bits != 0; bits &= bits - 1) {
            body.group_hashes[slot++] = groups->hashes[std::countr_zero(bits)];
        }
        body.fingerprint = groups->Fingerprint(VideoCommon::V2::HashedComputeGroups);
        groups->BeginDraw();
    }
    split.Note(Ir::Kind::Dispatch, body);
}

void KeplerCompute::FrontNoteWrites(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                    u32 argument) {
    using VideoCommon::V2::Producer;
    switch (method) {
    case 0x6c3:
        VideoCommon::V2::NoteReportCompute(split, memory_manager, fs.registers.data(), argument);
        return;
    case 0x107:
        FrontNoteInlineQmd(split, fs);
        return;
    case 0x6c:
    case 0x6d:
        VideoCommon::V2::NoteUploadMethod(split, memory_manager, fs.registers.data(), method,
                                          argument, fs.v2_upload);
        return;
    case 0xae:
    case 0xaf:
    case 0xb0: {
        const bool runs = !(method == 0xae || (method == 0xaf && !(argument & 2)) ||
                            (method == 0xb0 && ((argument & 15) == 0 || (argument & 15) == 1)));
        if (runs && fs.qmd_model == VideoCommon::V2::Mode::On && !fs.qmd_signals.empty())
            [[unlikely]] {
            // EDEN_NX2_V2_QMDSCHED: the QMDs this signal launches, as the front's model knows
            // them: their storage ranges and release words instead of unbounded writes.
            FrontState::QmdSignal signal = std::move(fs.qmd_signals.front());
            fs.qmd_signals.pop_front();
            if (signal.valid) {
                if (signal.unbounded) {
                    split.NoteUnboundedWrite(Producer::Dispatch);
                } else {
                    for (const auto& [address, size] : signal.writes) {
                        VideoCommon::V2::NoteRangeWriteGpu(split, memory_manager, address, size,
                                                           Producer::Dispatch);
                    }
                }
                for (const auto& [address, size] : signal.releases) {
                    VideoCommon::V2::NoteRangeWriteGpu(split, memory_manager, address, size,
                                                       Producer::QmdRelease);
                }
                VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::QmdBoundedWrites,
                                               signal.writes.size() + signal.releases.size());
                return;
            }
        }
        // QMD signals: the scheduler launches QMDs from memory the front doesn't read.
        split.NoteUnboundedWrite(Producer::Dispatch);
        split.NoteUnboundedWrite(Producer::QmdRelease);
        return;
    }
    case 0x45: // Macro code and bindings
    case 0x46:
    case 0x47:
    case 0x48:
        return;
    default:
        if (MayWriteMemory(method)) {
            split.NoteUnboundedWrite(Producer::Dispatch);
            split.NoteUnboundedWrite(Producer::QmdRelease);
        }
        return;
    }
}

void KeplerCompute::FrontModelQmd(VideoCommon::GpuSplit& split, FrontState& fs, u32 method,
                                  const u32* arguments, u32 amount) {
    namespace QmdFront = VideoCommon::V2::QmdFront;
    namespace P5b = VideoCommon::V2::P5bStats;
    using Descriptor = Tegra::NativeQmdScheduler::Descriptor;
    const u32* const r = fs.registers.data();
    if (method == 0x107) {
        // An inline QMD stored for a later signal (or launched now: then its words are in memory
        // before any signal names it, the same either way).
        Descriptor words{};
        std::copy_n(r + 0xc8, words.size(), words.begin());
        QmdFront::FrontInline().Note(((GPUVAddr{r[0xc6]} << 32) | r[0xc7]) << 8, words);
        return;
    }
    if (method != 0xae && method != 0xaf && method != 0xb0) {
        return;
    }
    fs.qmd_signals.clear();
    auto& model = QmdFront::FrontModel();
    for (u32 i = 0; i < amount; ++i) {
        const u32 argument = arguments[i];
        if (method == 0xae || (method == 0xaf && !(argument & 2)) ||
            (method == 0xb0 && ((argument & 15) == 0 || (argument & 15) == 1))) {
            continue; // The back skips these too
        }
        FrontState::QmdSignal& signal = fs.qmd_signals.emplace_back();
        const GPUVAddr qmd_address = GPUVAddr{r[0xad]} << 8;
        const auto read = [&](u64 address, Descriptor& descriptor) {
            if (const Descriptor* forwarded = QmdFront::FrontInline().Find(address)) {
                descriptor = *forwarded;
                return QmdFront::ReadResult::Ok;
            }
            if (!memory_manager.IsFullyMappedRange(address, sizeof(descriptor))) {
                return QmdFront::ReadResult::Unmapped;
            }
            const auto device = memory_manager.GpuToCpuAddress(address);
            if (!device || !memory_manager.IsContinuousRange(address, sizeof(descriptor)) ||
                Tegra::Nx2MayBeGpuWritten(*device, sizeof(descriptor))) {
                return QmdFront::ReadResult::Unknown;
            }
            memory_manager.ReadBlockUnsafe(address, descriptor.data(), sizeof(descriptor));
            // Queued writes (copies, uploads, dispatches) that may change these words: unknown.
            const auto pending =
                split.FindWrites(*device, sizeof(descriptor), VideoCommon::V2::HazardFilter{},
                                 reinterpret_cast<u8*>(descriptor.data()));
            // A release still to land: the back reads memory, which doesn't hold it yet.
            return pending.kind == VideoCommon::V2::HazardResult::Kind::Wait || pending.landing
                       ? QmdFront::ReadResult::Unknown
                       : QmdFront::ReadResult::Ok;
        };
        const auto mapped = [&](u64 address, u64 size) {
            return memory_manager.IsFullyMappedRange(address, size);
        };
        const auto on_dispatch = [&](u64, const Descriptor& qmd) {
            if (!signal.unbounded && !FrontQmdWriteRanges(split, qmd.data(), signal.writes)) {
                signal.unbounded = true;
            }
        };
        const auto on_release = [&](u64 address, u32, u32 flags) {
            signal.releases.emplace_back(address, (flags >> 30) == 0 ? 16U : 4U);
        };
        const auto result = model.Signal(qmd_address, argument, read, mapped, on_dispatch,
                                         on_release);
        signal.valid = result.valid;
        signal.dispatches = result.dispatches;
        if (P5b::Enabled()) {
            P5b::Add(P5b::Counter::QmdSignals);
            P5b::Add(result.valid ? P5b::Counter::QmdExact : P5b::Counter::QmdUnknown);
            P5b::Add(P5b::Counter::QmdDispatches, result.dispatches);
            if (!result.valid && model.Invalidations() != 0) {
                P5b::Add(P5b::Counter::QmdInvalidations, model.Valid() ? 0 : 1);
            }
        }
    }
}

bool KeplerCompute::FrontQmdWriteRanges(VideoCommon::GpuSplit& split, const u32* qmd,
                                        std::vector<std::pair<GPUVAddr, u64>>& out) {
    // As FrontNoteInlineQmd: the program's storage buffer stores, bounded by the descriptors in
    // the QMD's constant buffers when no queued write can change them.
    namespace V2 = VideoCommon::V2;
    using V2::Producer;
    const GPUVAddr program = (GPUVAddr{qmd[49] & 0x1ffff} << 32) | qmd[48];
    const auto program_device = memory_manager.GpuToCpuAddress(program);
    const auto summary = program_device ? V2::ShaderWrites::Lookup(*program_device)
                                        : std::optional<V2::ShaderWrites::Summary>{};
    if (!summary || summary->unbounded) {
        return false;
    }
    V2::HazardFilter descriptor_filter;
    descriptor_filter.unbounded = V2::ProducerBit(Producer::Upload) |
                                  V2::ProducerBit(Producer::Copy) |
                                  V2::ProducerBit(Producer::Dispatch) |
                                  V2::ProducerBit(Producer::Draw);
    for (u32 i = 0; i < summary->num_storage; ++i) {
        const auto& storage = summary->storage[i];
        if (storage.cbuf_index >= NumConstBuffers || ((qmd[20] >> storage.cbuf_index) & 1) == 0) {
            return false;
        }
        const u32 high = qmd[33 + storage.cbuf_index * 2];
        const GPUVAddr cbuf = (GPUVAddr{high & 0xff} << 32) | qmd[32 + storage.cbuf_index * 2];
        const u32 cbuf_size = ((high >> 19) & 0x1fff) << 4;
        for (u32 element = 0; element < storage.count; ++element) {
            const u32 offset = storage.cbuf_offset + element * 16;
            const auto device = memory_manager.GpuToCpuAddress(cbuf + offset);
            if (offset + 16 > cbuf_size || !device || out.size() >= 64) {
                return false;
            }
            std::array<u32, 4> descriptor;
            memory_manager.ReadBlockUnsafe(cbuf + offset, descriptor.data(), sizeof(descriptor));
            const V2::HazardResult pending =
                split.FindWrites(*device, sizeof(descriptor), descriptor_filter,
                                 reinterpret_cast<u8*>(descriptor.data()));
            if (pending.kind == V2::HazardResult::Kind::Wait || pending.landing) {
                return false;
            }
            out.emplace_back((GPUVAddr{descriptor[1]} << 32) | descriptor[0], descriptor[2]);
        }
    }
    return true;
}

void KeplerCompute::FrontNoteInlineQmd(VideoCommon::GpuSplit& split, FrontState& fs) {
    namespace V2 = VideoCommon::V2;
    using V2::Producer;
    const u32* const registers = fs.registers.data();
    const u32* const qmd = registers + 0xc8;
    // As ProcessNativeMethod(0x107): the QMD is stored, then launched unless it waits for a
    // dependency (then the scheduler launches it from memory later).
    const GPUVAddr qmd_address = ((GPUVAddr{registers[0xc6]} << 32) | registers[0xc7]) << 8;
    V2::NoteRangeWriteGpu(split, memory_manager, qmd_address, 256, Producer::Upload);
    if ((registers[0xa7] & (1U << 24)) == 0 && (registers[0xcc] & (1U << 12)) != 0) {
        split.NoteUnboundedWrite(Producer::Dispatch);
        split.NoteUnboundedWrite(Producer::QmdRelease);
        return;
    }
    split.NoteUnboundedWrite(Producer::QmdRelease);
    const GPUVAddr program = (GPUVAddr{qmd[49] & 0x1ffff} << 32) | qmd[48];
    const auto program_device = memory_manager.GpuToCpuAddress(program);
    const auto summary = program_device ? V2::ShaderWrites::Lookup(*program_device)
                                        : std::optional<V2::ShaderWrites::Summary>{};
    if (!summary || summary->unbounded) {
        split.NoteUnboundedWrite(Producer::Dispatch);
        return;
    }
    // Storage buffers: their descriptors (address, size) in the QMD's constant buffers, as the
    // buffer cache binds them. A descriptor a queued write may still change can't be trusted.
    V2::HazardFilter descriptor_filter;
    descriptor_filter.unbounded = V2::ProducerBit(Producer::Upload) |
                                  V2::ProducerBit(Producer::Copy) |
                                  V2::ProducerBit(Producer::Dispatch) |
                                  V2::ProducerBit(Producer::Draw);
    std::array<V2::GpuRange, 16> writes;
    size_t num_writes = 0;
    for (u32 i = 0; i < summary->num_storage; ++i) {
        const auto& storage = summary->storage[i];
        if (storage.cbuf_index >= NumConstBuffers || ((qmd[20] >> storage.cbuf_index) & 1) == 0) {
            split.NoteUnboundedWrite(Producer::Dispatch);
            return;
        }
        const u32 high = qmd[33 + storage.cbuf_index * 2];
        const GPUVAddr cbuf = (GPUVAddr{high & 0xff} << 32) | qmd[32 + storage.cbuf_index * 2];
        const u32 cbuf_size = ((high >> 19) & 0x1fff) << 4;
        for (u32 element = 0; element < storage.count; ++element) {
            const u32 offset = storage.cbuf_offset + element * 16;
            const auto device = memory_manager.GpuToCpuAddress(cbuf + offset);
            if (offset + 16 > cbuf_size || !device || num_writes == writes.size()) {
                split.NoteUnboundedWrite(Producer::Dispatch);
                return;
            }
            std::array<u32, 4> descriptor;
            memory_manager.ReadBlockUnsafe(cbuf + offset, descriptor.data(), sizeof(descriptor));
            const V2::HazardResult pending =
                split.FindWrites(*device, sizeof(descriptor), descriptor_filter,
                                 reinterpret_cast<u8*>(descriptor.data()));
            if (pending.kind == V2::HazardResult::Kind::Wait || pending.landing) {
                split.NoteUnboundedWrite(Producer::Dispatch);
                return;
            }
            const GPUVAddr address = (GPUVAddr{descriptor[1]} << 32) | descriptor[0];
            writes[num_writes++] = {address, descriptor[2]};
        }
    }
    for (size_t i = 0; i < num_writes; ++i) {
        V2::NoteRangeWriteGpu(split, memory_manager, writes[i].first, writes[i].second,
                              Producer::Dispatch);
    }
}

void KeplerCompute::FrontCallMultiMethod(Core::System& system, VideoCommon::GpuSplit& split,
                                         u32 method, const u32* base_start, u32 amount,
                                         u32 methods_pending, const GPUVAddr* segment) {
    auto& fs = Front(split);
    // S1: every word is classified on its own (an upload stays one MethodData record).
    const bool s1_words = fs.s1 && method < 0xe00 && method != KEPLER_COMPUTE_REG_INDEX(data_upload);
    if ((method >= 0xe00 && method < fs.registers.size()) || s1_words) {
        for (u32 i = 0; i < amount; ++i) {
            FrontCallMethod(system, split, method, base_start[i], methods_pending - i <= 1, segment);
        }
        return;
    }
    if (method < fs.registers.size()) {
        for (u32 i = 0; i < amount; ++i) {
            FrontApply(fs, method, base_start[i]);
        }
    }
    if (VideoCommon::V2::Resolve::MultiFront::g_recording) [[unlikely]] {
        NoteQmdUse(method);
    }
    if (fs.qmd_model != VideoCommon::V2::Mode::Off) [[unlikely]] {
        FrontModelQmd(split, fs, method, base_start, amount);
    }
    if (split.DrawRecords()) [[unlikely]] {
        FrontDispatchRecord(split, fs, method, base_start, amount);
    }
    split.MultiMethod(this, method, base_start, amount, methods_pending, segment);
    if (split.TracksWrites()) [[unlikely]] {
        for (u32 i = 0; i < amount; ++i) {
            FrontNoteWrites(split, fs, method, base_start[i]);
        }
    }
}

void KeplerCompute::FrontRunMacro(Core::System& system, VideoCommon::GpuSplit& split,
                                  FrontState& fs) {
    const auto positions = FrontMacroPositions(fs);
    const u32 macro_method = *fs.executing_macro;
    const u32 start = positions[((*fs.executing_macro - 0xe00) / 2) % positions.size()];
    fs.executing_macro.reset();
    auto& macro = FrontMacroEngine(fs);
    // EDEN_NX2_V2_MACRO_DUMP: each distinct compute macro once, with what it wrote.
    namespace MacroHle = VideoCommon::V2::MacroHle;
    std::optional<MacroHle::Identity> dump_identity;
    std::vector<u32> dump_parameters;
    MacroHle::Observation observation;
    if (VideoCommon::V2::On(VideoCommon::V2::Switch::MacroDump)) [[unlikely]] {
        const MacroHle::Identity identity = MacroHle::Identify(macro.native_code, start);
        if (MacroHle::ShouldDump("compute", identity.hash)) {
            dump_identity = identity;
            dump_parameters = fs.parameters;
        }
    }
    SCOPE_EXIT {
        if (dump_identity) {
            MacroHle::Dump("compute", macro_method, start, *dump_identity, macro.native_code,
                           dump_parameters, observation);
        }
    };
    if (native_data_ram) {
        macro.native_interpreter.active_dram = native_data_ram;
    }
    if (VideoCommon::V2::Stats::Enabled()) [[unlikely]] {
        if (!fs.v2_call_stats) {
            fs.v2_call_stats = std::make_unique<VideoCommon::V2::MacroMemo::CallStats>();
        }
        fs.v2_call_stats->Note("compute", start, fs.parameters, macro.native_code,
                               macro.native_code_generation,
                               fs.v2_memo ? &fs.v2_memo->statistics : nullptr);
    }
    VideoCommon::Nx2Profile::Scope profile_scope{VideoCommon::Nx2Profile::Zone::Macro};
    const auto read = [&fs](u32 address) {
        return address < fs.registers.size() ? fs.registers[address] : 0U;
    };
    const auto emit = [&](u32 address, u32 value) {
        if (address >= 0xe00) {
            return false;
        }
        const u32 old_value = fs.registers[address];
        FrontApply(fs, address, value);
        if (fs.qmd_model != VideoCommon::V2::Mode::Off) [[unlikely]] {
            FrontModelQmd(split, fs, address, &value, 1);
        }
        if (fs.s1 && address != KEPLER_COMPUTE_REG_INDEX(data_upload)) [[likely]] {
            FrontS1Write(split, fs, address, old_value, true);
        } else {
            if (split.DrawRecords()) [[unlikely]] {
                FrontDispatchRecord(split, fs, address, &value, 1);
            }
            split.MacroWrite(this, address, value);
        }
        if (dump_identity && observation.writes.size() < 64) [[unlikely]] {
            observation.writes.emplace_back(address, value);
        }
        if (split.TracksWrites()) [[unlikely]] {
            FrontNoteWrites(split, fs, address, value);
        }
        return true;
    };
    const auto live = [&](auto& live_emit) {
#ifdef ARCHITECTURE_x86_64
        if (!macro.native_jit) {
            macro.native_jit = std::make_shared<TuringMacroJit>();
        }
        macro.native_jit->Prepare(macro.native_code, macro.native_code_generation);
        return macro.native_jit->Execute(start, fs.parameters,
                                         std::span<u32>{*macro.native_interpreter.active_dram},
                                         read, live_emit);
#else
        return macro.native_interpreter.Execute(macro.native_code, start, fs.parameters, read,
                                                live_emit);
#endif
    };
    const VideoCommon::V2::Mode memo = VideoCommon::V2::Get(VideoCommon::V2::Switch::MacroMemo);
    Tegra::TuringMacroInterpreter::Result result;
    if (memo != VideoCommon::V2::Mode::Off) [[unlikely]] {
        // Fix F5-02, as for 3D macros: compute macros make no MME DMA reads, and a compute
        // register write changes no other register.
        namespace MacroMemo = VideoCommon::V2::MacroMemo;
        if (!fs.v2_memo) {
            fs.v2_memo = std::make_unique<MacroMemo::Memo>();
        }
        // The front switches between two macro engines (FrontMacroEngine), each with its own
        // generation count.
        fs.v2_memo->Synchronize(macro.native_code_generation ^
                                (static_cast<u64>(reinterpret_cast<uintptr_t>(&macro)) << 20));
        const auto no_dma = [](u32, u32) {};
        const MacroMemo::CallInfo call{
            .start = start,
            .fifo = &fs.parameters,
            .dram = *macro.native_interpreter.active_dram,
            .code = macro.native_code,
            .generation = &macro.native_code_generation,
            .regs = fs.registers.data(),
            .num_regs = fs.registers.size(),
            .side_effects = {},
            .dma_fifoed = ~0U,
            .dma_ram = ~0U,
            .verify = memo == VideoCommon::V2::Mode::Verify,
        };
        result = MacroMemo::Call(*fs.v2_memo, macro.native_interpreter, call, live, read, emit,
                                 no_dma);
    } else {
        result = live(emit);
    }
    if (!result.success) {
        macro.native_execution_failed = true;
        LOG_ERROR(HW_GPU, "Native MME (split front) failed: compute start={:#x}, consumed={}, error={}",
                  start, result.consumed, result.error);
    }
    fs.parameters.clear();
}

} // namespace Tegra::Engines
