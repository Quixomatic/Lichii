// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>

#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/v2/front_writes.h"

namespace VideoCommon::V2 {

namespace {

constexpr u64 PageSize = 0x1000;
/// Longer writes are not split into pages: the front can't bound them cheaply.
constexpr u64 MaxRangePages = 64;

} // Anonymous namespace

void NoteReleaseWrite(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                      u32 payload, u32 bytes, bool known, Producer producer, bool fenced) {
    const auto device = memory.GpuToCpuAddress(address);
    if (!device) {
        return; // The back drops it too (the query cache skips unmapped reports)
    }
    if (!known) {
        split.NoteRangeWrite(*device, bytes, producer);
        return;
    }
    if (bytes <= 4) {
        split.NoteKnownWrite(*device, &payload, sizeof(payload), producer);
        if (fenced) {
            split.NoteUnlandedWrite(*device, &payload, sizeof(payload));
        }
        return;
    }
    // The payload as 64 bits, then a timestamp the front can't tell.
    const std::array<u32, 2> value{payload, 0};
    split.NoteKnownWrite(*device, value.data(), sizeof(value), producer);
    split.NoteRangeWrite(*device + 8, bytes - 8, producer);
    if (fenced) {
        split.NoteUnlandedWrite(*device, value.data(), sizeof(value));
    }
}

void NoteRangeWriteGpu(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                       u64 size, Producer producer) {
    if (size == 0) {
        return;
    }
    const GPUVAddr end = address + size;
    if (end < address || (end - 1) / PageSize - address / PageSize >= MaxRangePages) {
        split.NoteUnboundedWrite(producer);
        return;
    }
    // Page by page: GPU pages may map anywhere in device memory.
    for (GPUVAddr begin = address; begin < end;) {
        const GPUVAddr page_end = std::min<GPUVAddr>((begin & ~(PageSize - 1)) + PageSize, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            split.NoteRangeWrite(*device, page_end - begin, producer);
        }
        begin = page_end;
    }
}

void NoteReport3D(GpuSplit& split, const Tegra::MemoryManager& memory, const u32* regs) {
    using Tegra::Engines::Maxwell3D;
    const u32 query = regs[MAXWELL3D_REG_INDEX(report_semaphore.query)];
    const u32 operation = query & 3;
    // Release (0) and ReportOnly (2) write (Maxwell3D::ProcessQueryGet); Acquire and Trap don't.
    if (operation != 0 && operation != 2) {
        return;
    }
    const GPUVAddr address =
        (GPUVAddr{regs[MAXWELL3D_REG_INDEX(report_semaphore.address_high)]} << 32) |
        regs[MAXWELL3D_REG_INDEX(report_semaphore.address_low)];
    const bool payload = ((query >> 23) & 0x1f) == 0; // Report::Payload
    const bool short_query = ((query >> 28) & 1) != 0;
    // ProcessQueryGet: a short Release is fence-flagged; ReportOnly and long ones are not.
    NoteReleaseWrite(split, memory, address, regs[MAXWELL3D_REG_INDEX(report_semaphore.payload)],
                     short_query ? 4 : 16, payload, Producer::Report,
                     short_query && operation == 0);
}

void NoteReportCompute(GpuSplit& split, const Tegra::MemoryManager& memory, const u32* registers,
                       u32 control) {
    // As KeplerCompute::ProcessNativeReportSemaphore writes it.
    const u32 operation = control & 3;
    if (operation != 0 && operation != 2) {
        return;
    }
    const GPUVAddr address = (GPUVAddr{registers[0x6c0] & 0x1ffffff} << 32) | registers[0x6c1];
    const bool reduction = ((control >> 3) & 1) != 0;
    const bool one_word = ((control >> 28) & 1) != 0;
    NoteReleaseWrite(split, memory, address, registers[0x6c2], reduction || one_word ? 4 : 16,
                     !reduction, Producer::Report, one_word);
}

void DmaWriteTracker::Note(GpuSplit& split, const Tegra::MemoryManager& memory, u32 method,
                           u32 argument) {
    using Tegra::Engines::MaxwellDMA;
    switch (method) {
    case 0x90: // semaphore.address.upper
    case 0x91: // semaphore.address.lower
    case 0x92: // semaphore.payload
        semaphore[method - 0x90] = argument;
        return;
    case 0x102: // offset_out.upper
        offset_out_high = argument;
        return;
    case 0x103: // offset_out.lower
        offset_out_low = argument;
        return;
    case 0x105: // pitch_out
        pitch_out = static_cast<s32>(argument);
        return;
    case 0x106: // line_length_in
        line_length_in = argument;
        return;
    case 0x107: // line_count
        line_count = argument;
        return;
    case 0x1c2: // remap_const components
        remap_components = argument;
        return;
    case LaunchMethod:
        break;
    default:
        return;
    }
    MaxwellDMA::LaunchDMA launch;
    static_assert(sizeof(launch) == sizeof(argument));
    std::memcpy(&launch, &argument, sizeof(launch));
    using LaunchDMA = MaxwellDMA::LaunchDMA;
    if (launch.data_transfer_type.Value() != LaunchDMA::DataTransferType::NONE) {
        // A pitch destination's bytes are bounded; block linear ones (swizzled) are not here.
        if (launch.dst_memory_layout.Value() == LaunchDMA::MemoryLayout::PITCH && pitch_out >= 0) {
            u64 line_bytes = line_length_in;
            if (launch.remap_enable != 0) {
                const u64 component_size = ((remap_components >> 16) & 3) + 1;
                const u64 components = ((remap_components >> 24) & 3) + 1;
                line_bytes *= component_size * components;
            }
            const u64 lines = launch.multi_line_enable != 0 ? std::max<u32>(line_count, 1) : 1;
            const GPUVAddr destination =
                MaxwellDMA::PackedGPUVAddr{offset_out_high, offset_out_low};
            NoteRangeWriteGpu(split, memory, destination,
                              (lines - 1) * static_cast<u64>(pitch_out) + line_bytes,
                              Producer::Copy);
        } else {
            split.NoteUnboundedWrite(Producer::Copy);
        }
    }
    // As MaxwellDMA::ReleaseSemaphore writes it (a reduction is written as the payload too).
    const auto type = launch.semaphore_type.Value();
    if (type == LaunchDMA::SemaphoreType::RELEASE_ONE_WORD_SEMAPHORE ||
        type == LaunchDMA::SemaphoreType::RELEASE_FOUR_WORD_SEMAPHORE) {
        const GPUVAddr address = MaxwellDMA::PackedGPUVAddr{semaphore[0], semaphore[1]};
        NoteReleaseWrite(split, memory, address, semaphore[2],
                         type == LaunchDMA::SemaphoreType::RELEASE_ONE_WORD_SEMAPHORE ? 4 : 16,
                         true, Producer::Release, true);
    }
}

} // namespace VideoCommon::V2
