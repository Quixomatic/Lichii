// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>
#include <span>
#include <utility>

#include "common/common_types.h"
#include "video_core/v2/hazard.h"
#include "video_core/v2/ir.h"

namespace Tegra {
class MemoryManager;
}

namespace VideoCommon {
class GpuSplit;
}

// Pipeline v2 P2: the split front's records of the guest memory writes it sends to the back
// (GpuSplit::TracksWrites, the range hazard table in v2/hazard.h). Each is called right after the
// write was sent, so the ring position covers it. GPU addresses are translated to device
// addresses here: channels with different address spaces see the same memory.
namespace VideoCommon::V2 {

/// A release the front knows the payload of: 4 bytes, or `bytes` 16 (the payload as 64 bits,
/// then a timestamp). `known` false: a counter report or a reduction, whose value is unknown.
/// `fenced`: the query cache writes it when its fence runs (QueryPropertiesFlags::IsAFence), so it
/// is also kept until it lands (GpuSplit::NoteUnlandedWrite).
void NoteReleaseWrite(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                      u32 payload, u32 bytes, bool known, Producer producer, bool fenced);

/// A write of [address, address + size) whose bytes the front doesn't know. Long or unmapped
/// ranges become a write the front can't bound.
void NoteRangeWriteGpu(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                       u64 size, Producer producer);

/// A 3D report semaphore was triggered (report_semaphore.query sent), with the front's copy of
/// the 3D registers.
void NoteReport3D(GpuSplit& split, const Tegra::MemoryManager& memory, const u32* regs);

/// A compute report semaphore was triggered (native method 0x6c3 = `control`), with the front's
/// copy of the compute registers.
void NoteReportCompute(GpuSplit& split, const Tegra::MemoryManager& memory, const u32* registers,
                       u32 control);

/// A GPU range (address, bytes).
using GpuRange = std::pair<GPUVAddr, u64>;

/// What an inline upload (3D launch_dma/inline_data, compute exec_upload/data_upload,
/// inline-to-memory exec/data: all at methods 0x60-0x6d) writes: `upload` is the 12 words of
/// Upload::Registers (method 0x60 on), `exec` the exec/launch word (bit 0: pitch linear).
[[nodiscard]] std::optional<GpuRange> UploadRange(const u32* upload, u32 exec);

/// An upload method was sent to an engine whose registers are `registers` (indexed by method):
/// the exec method computes the range into `pending`, each data method records it (the data
/// words are what write it, so their positions are the ones to wait for).
void NoteUploadMethod(GpuSplit& split, const Tegra::MemoryManager& memory, const u32* registers,
                      u32 method, u32 argument, std::optional<GpuRange>& pending);

/// The inline-to-memory engine's upload registers as the front sends them.
struct UploadWriteTracker {
    std::array<u32, 0x6e> registers{};
    std::optional<GpuRange> pending;

    void Note(GpuSplit& split, const Tegra::MemoryManager& memory, u32 method, u32 argument) {
        if (method < registers.size()) {
            registers[method] = argument;
        }
        NoteUploadMethod(split, memory, registers.data(), method, argument, pending);
    }
};

/// The device ranges a GPU range maps to, as one bounding range per contiguous run (merged into
/// one when there are more than `max_ranges`). Empty if unmapped.
size_t DeviceRanges(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                    std::span<std::pair<u64, u64>> out);

/// The copy engine's methods as the front sends them: it keeps the registers a launch reads.
struct DmaWriteTracker {
    std::array<u32, 3> semaphore{}; ///< address high, low, payload
    u32 offset_out_high{};
    u32 offset_out_low{};
    s32 pitch_out{};
    u32 line_length_in{};
    u32 line_count{};
    u32 remap_components{};

    /// MaxwellDMA's launch_dma method.
    static constexpr u32 LaunchMethod = 0xc0;

    /// A copy engine method was sent.
    void Note(GpuSplit& split, const Tegra::MemoryManager& memory, u32 method, u32 argument);
};

/// EDEN_NX2_V2_IR_DRAWS: Copy records (informational) for copy engine launches and 2D blits,
/// from the registers the front saw sent.
struct CopyRecordTracker {
    std::array<u32, 8> dma{};  ///< MaxwellDMA 0x100-0x107: offsets in/out, pitches, line size
    std::array<u32, 4> blit{}; ///< Fermi2D destination and source surface addresses
    u32 remap_components{};

    /// A copy engine method was sent: true and the body for a launch.
    bool NoteDma(u32 method, u32 argument, Ir::CopyBody& body);
    /// A 2D engine method was sent: true and the body for a blit.
    bool NoteBlit(u32 method, u32 argument, Ir::CopyBody& body);
};

} // namespace VideoCommon::V2
