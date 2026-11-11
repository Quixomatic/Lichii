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
/// `fenced`: the query cache writes it when its fence runs (QueryPropertiesFlags::IsAFence).
/// `reduction`: a compute reduction (KeplerCompute writes it from a fence of its own).
/// A write that lands at an operation (fenced, a counter, High accuracy, or a write the back
/// defers behind an earlier one still to land: v2/landing.h) is noted once, by landing ticket,
/// in the table of writes still to land; any other in the queued table only.
/// SYNC_SUBMIT_SPEC §2.1: each word also goes to S1's semaphore view (GpuSplit::Sem), a
/// reduction (`reduction_op`, ReduceSemaphore) with the value it leaves when S1 can tell it.
void NoteReleaseWrite(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                      u32 payload, u32 bytes, bool known, Producer producer, bool fenced,
                      bool reduction = false, u32 reduction_op = 0);

/// SYNC_SUBMIT_SPEC §2.1: the newest unbounded unknown writer still queued or a GPU-written page
/// makes memory an incomplete base for a reduction S1 computes.
[[nodiscard]] bool SemMemoryComplete(GpuSplit& split, u64 device_address);

/// Whether fence operations wait for the host GPU (as FenceManager::SignalFence's delay_fence
/// without a report): accurate GPU, or Balanced/Accurate fences.
[[nodiscard]] bool FencesDelayed();

/// A write of [address, address + size) whose bytes the front doesn't know. Long or unmapped
/// ranges become a write the front can't bound. `position`: GpuSplit::NoteRangeWrite's (0: the
/// write just sent).
void NoteRangeWriteGpu(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                       u64 size, Producer producer, u64 position = 0);

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
    /// dst_params (methods 0x1c3-0x1c8): block size, width, height, depth, layer, origin.
    std::array<u32, 6> dst_params{};
    /// A copy engine macro ran: registers it set are not tracked, so launches stay unbounded.
    bool macros = false;

    /// MaxwellDMA's launch_dma method.
    static constexpr u32 LaunchMethod = 0xc0;
    static constexpr u32 DstParamsMethod = 0x1c3;

    /// The bytes a copy into block linear memory rewrites (MaxwellDMA's CopyPitchToBlockLinear
    /// and CopyBlockLinearToBlockLinear: the whole swizzled surface at offset_out), or nothing
    /// when the front can't bound it (layers, block width).
    [[nodiscard]] std::optional<u64> BlockLinearBytes(bool remap) const;

    /// A copy engine method was sent.
    void Note(GpuSplit& split, const Tegra::MemoryManager& memory, u32 method, u32 argument);
};

/// The 2D engine's destination surface as the front sends it (F3-X5): a blit rewrites at most
/// that surface (SoftwareBlitEngine::Blit's size; the accelerated blit writes the same image).
struct BlitWriteTracker {
    /// Fermi2D::Regs::dst (methods 0x80-0x89): format, layout, block, depth, layer, pitch,
    /// width, height, address high, low.
    std::array<u32, 10> dst{};
    /// A 2D macro ran: registers it set are not in `dst`, so blits stay unbounded.
    bool macros = false;

    static constexpr u32 DstMethod = 0x80;

    /// A 2D engine method was sent (`exec`: the blit's launch method).
    void Note(GpuSplit& split, const Tegra::MemoryManager& memory, u32 method, u32 argument,
              bool exec);
    /// The destination surface's bytes, or nothing when the front can't bound them.
    [[nodiscard]] std::optional<GpuRange> DestinationRange() const;
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
