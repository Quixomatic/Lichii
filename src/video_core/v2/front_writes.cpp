// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstring>

#include "common/settings.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/native_qmd.h"
#include "video_core/nx2.h"
#include "video_core/surface.h"
#include "video_core/textures/decoders.h"
#include "video_core/v2/front_writes.h"
#include "video_core/v2/landing.h"

namespace VideoCommon::V2 {

namespace {

constexpr u64 PageSize = 0x1000;
/// Longer writes are not split into pages: the front can't bound them cheaply.
constexpr u64 MaxRangePages = 64;

} // Anonymous namespace

bool FencesDelayed() {
    return Settings::IsGPUFenceBehaviorDefault()
               ? Settings::IsGPULevelHigh()
               : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
}

bool SemMemoryComplete(GpuSplit& split, u64 device_address) {
    return split.Sem().UnboundedUnknown(split.ConsumedNow()) == 0 &&
           !Tegra::Nx2MayBeGpuWritten(device_address, sizeof(u32));
}

namespace {

/// SYNC_SUBMIT_SPEC §2.1: the words a release writes, into S1's semaphore view. `ticket` 0: made
/// when the back runs it, else at its operation by landing ticket.
void NoteSemRelease(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                    u64 device, u32 payload, u32 bytes, bool known, Producer producer,
                    bool reduction, u32 reduction_op, u64 ticket) {
    SemView& view = split.Sem();
    const SemProgress progress = split.SemNow();
    const u64 position = split.WritePosition();
    const auto retire = ticket != 0 ? SemView::Retire::Landing : SemView::Retire::Consumed;
    using Kind = SemView::Kind;
    const Kind kind = !known && !reduction        ? Kind::Counter
                      : producer == Producer::Report ? Kind::Report
                                                     : Kind::Release;
    if (reduction) {
        view.NoteReduction(
            device, false, reduction_op, payload, position, retire, ticket, kind, progress,
            SemMemoryComplete(split, device),
            [](u32 op, u32 base, u32 operand) {
                return Tegra::ReduceSemaphore(op, false, base, operand);
            },
            [&memory, address] { return memory.Read<u32>(address); });
        return;
    }
    view.Note(device, payload, known, position, retire, ticket, kind, progress);
    if (bytes > 4) {
        // The payload as 64 bits, then a timestamp only the operation knows.
        view.Note(device + 4, 0, known, position, retire, ticket, kind, progress);
        for (u64 offset = 8; offset < bytes; offset += 4) {
            view.Note(device + offset, 0, false, position, retire, ticket, Kind::Counter,
                      progress);
        }
    }
}

} // Anonymous namespace

void NoteReleaseWrite(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                      u32 payload, u32 bytes, bool known, Producer producer, bool fenced,
                      bool reduction, u32 reduction_op) {
    const auto device = memory.GpuToCpuAddress(address);
    if (!device) {
        return; // The back drops it too (the query cache skips unmapped reports)
    }
    // Where the back writes it (QueryCacheBase::CounterReport, KeplerCompute's reduction): at an
    // operation, or at once. A write that could be made at once lands at an operation while an
    // earlier one may still land (the landing rule); the front decides which (LandingPending)
    // and the back obeys (Fix F3h, F3-X6b: Landing::PushVerdict / DeferAtOnceWrite).
    const bool may_land_at_once = reduction ? !FencesDelayed()
                                            : known && !fenced && !Settings::IsGPULevelHigh();
    const bool at_once = may_land_at_once && !split.LandingPending();
    if (may_land_at_once) {
        Landing::PushVerdict(split.WritePosition(), at_once);
    }
    if (at_once) {
        NoteSemRelease(split, memory, address, *device, payload, bytes, known, producer,
                       reduction, reduction_op, 0);
        if (!known) {
            split.NoteRangeWrite(*device, bytes, producer);
        } else if (bytes <= 4) {
            split.NoteKnownWrite(*device, &payload, sizeof(payload), producer);
        } else {
            const std::array<u32, 2> value{payload, 0};
            split.NoteKnownWrite(*device, value.data(), sizeof(value), producer);
            split.NoteRangeWrite(*device + 8, bytes - 8, producer);
        }
        return;
    }
    const u64 ticket = split.NextLandingTicket();
    NoteSemRelease(split, memory, address, *device, payload, bytes, known, producer, reduction,
                   reduction_op, ticket);
    if (!known) {
        split.NoteUnlandedRange(*device, bytes, ticket);
    } else if (bytes <= 4) {
        split.NoteUnlandedWrite(*device, &payload, sizeof(payload), ticket);
    } else {
        // The payload as 64 bits, then a timestamp only the operation knows.
        const std::array<u32, 2> value{payload, 0};
        split.NoteUnlandedWrite(*device, value.data(), sizeof(value), ticket);
        split.NoteUnlandedRange(*device + 8, bytes - 8, ticket);
    }
}

size_t DeviceRanges(const Tegra::MemoryManager& memory, GPUVAddr address, u64 size,
                    std::span<std::pair<u64, u64>> out) {
    if (out.empty() || size == 0) {
        return 0;
    }
    const auto segments = memory.GetSubmappedRange(address, static_cast<size_t>(size));
    size_t count = 0;
    for (const auto& [device, length] : segments) {
        const std::pair<u64, u64> range{device, device + length};
        if (count != 0 && out[count - 1].second == range.first) {
            out[count - 1].second = range.second;
        } else if (count < out.size()) {
            out[count++] = range;
        } else {
            // Too many pieces: one range around all of them (it may cover more, never less).
            out[count - 1].first = std::min(out[count - 1].first, range.first);
            out[count - 1].second = std::max(out[count - 1].second, range.second);
            for (size_t i = 0; i + 1 < count; ++i) {
                out[count - 1].first = std::min(out[count - 1].first, out[i].first);
                out[count - 1].second = std::max(out[count - 1].second, out[i].second);
            }
            out[0] = out[count - 1];
            count = 1;
        }
    }
    return count;
}

void NoteRangeWriteGpu(GpuSplit& split, const Tegra::MemoryManager& memory, GPUVAddr address,
                       u64 size, Producer producer, u64 position) {
    if (size == 0) {
        return;
    }
    const GPUVAddr end = address + size;
    if (end < address) {
        split.NoteUnboundedWrite(producer);
        return;
    }
    if ((end - 1) / PageSize - address / PageSize >= MaxRangePages) {
        // Long writes (storage buffers, copies): their device runs, as wide entries.
        std::array<std::pair<u64, u64>, 8> ranges;
        const size_t count = DeviceRanges(memory, address, size, ranges);
        for (size_t i = 0; i < count; ++i) {
            split.NoteRangeWrite(ranges[i].first, ranges[i].second - ranges[i].first, producer,
                                 position);
        }
        return;
    }
    // Page by page: GPU pages may map anywhere in device memory.
    for (GPUVAddr begin = address; begin < end;) {
        const GPUVAddr page_end = std::min<GPUVAddr>((begin & ~(PageSize - 1)) + PageSize, end);
        if (const auto device = memory.GpuToCpuAddress(begin)) {
            split.NoteRangeWrite(*device, page_end - begin, producer, position);
        }
        begin = page_end;
    }
}

std::optional<GpuRange> UploadRange(const u32* upload, u32 exec) {
    const u32 line_length = upload[0];
    const u32 line_count = upload[1];
    const GPUVAddr address = (GPUVAddr{upload[2]} << 32) | upload[3];
    const u32 pitch = upload[4];
    if (line_length == 0 || line_count == 0 || address == 0) {
        return GpuRange{address, 0};
    }
    if ((exec & 1) != 0) {
        // As Upload::State::ProcessData: one line per pitch.
        return GpuRange{address, u64{line_count - 1} * pitch + line_length};
    }
    // Block linear: the whole swizzled surface it rewrites (Upload::State::ProcessData).
    const u32 block = upload[5];
    u32 width = upload[6];
    u32 x_elements = line_length;
    u32 x_offset = upload[10];
    u32 shift = 4;
    for (const u32 value : {width, x_elements, x_offset, static_cast<u32>(address)}) {
        if (value != 0) {
            shift = std::min(shift, static_cast<u32>(std::countr_zero(value)));
        }
    }
    width >>= shift;
    const u64 size = Tegra::Texture::CalculateSize(true, 1U << shift, width, upload[7], upload[8],
                                                   (block >> 4) & 0xf, (block >> 8) & 0xf);
    return GpuRange{address, size};
}

void NoteUploadMethod(GpuSplit& split, const Tegra::MemoryManager& memory, const u32* registers,
                      u32 method, u32 argument, std::optional<GpuRange>& pending) {
    if (method == 0x6c) {
        pending = UploadRange(registers + 0x60, argument);
        return;
    }
    if (method != 0x6d) {
        return;
    }
    if (!pending) {
        split.NoteUnboundedWrite(Producer::Upload);
        return;
    }
    // SYNC §15.5: the words of one upload (and contiguous uploads) make one note.
    split.NoteUploadWrite(memory, pending->first, pending->second);
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
                     !reduction, Producer::Report, one_word, reduction, (control >> 9) & 7);
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
        if (method >= DstParamsMethod && method < DstParamsMethod + dst_params.size()) {
            dst_params[method - DstParamsMethod] = argument;
        }
        return;
    }
    MaxwellDMA::LaunchDMA launch;
    static_assert(sizeof(launch) == sizeof(argument));
    std::memcpy(&launch, &argument, sizeof(launch));
    using LaunchDMA = MaxwellDMA::LaunchDMA;
    if (launch.data_transfer_type.Value() != LaunchDMA::DataTransferType::NONE) {
        u64 line_bytes = line_length_in;
        if (launch.remap_enable != 0) {
            const u64 component_size = ((remap_components >> 16) & 3) + 1;
            const u64 components = ((remap_components >> 24) & 3) + 1;
            line_bytes *= component_size * components;
        }
        const GPUVAddr destination = MaxwellDMA::PackedGPUVAddr{offset_out_high, offset_out_low};
        const bool pitch = launch.dst_memory_layout.Value() == LaunchDMA::MemoryLayout::PITCH;
        if (macros) {
            split.NoteUnboundedWrite(Producer::Copy);
        } else if (launch.multi_line_enable == 0) {
            // A one-line copy (MaxwellDMA::Launch) writes its line, moved in 16-byte pieces
            // within the line's 512-byte GOBs when the destination pages are block linear,
            // whatever the launch says (F3-X5: bounded either way).
            const GPUVAddr begin = destination & ~GPUVAddr{0x1ff};
            const GPUVAddr end = (destination + line_bytes + 0x1ff) & ~GPUVAddr{0x1ff};
            NoteRangeWriteGpu(split, memory, begin, end - begin, Producer::Copy);
        } else if (pitch && pitch_out >= 0) {
            NoteRangeWriteGpu(split, memory, destination,
                              (std::max<u32>(line_count, 1) - 1) * static_cast<u64>(pitch_out) +
                                  line_bytes,
                              Producer::Copy);
        } else if (const auto bytes =
                       pitch ? std::nullopt : BlockLinearBytes(launch.remap_enable != 0)) {
            // F3-X5: a block linear destination rewrites its swizzled surface, no more.
            NoteRangeWriteGpu(split, memory, destination, *bytes, Producer::Copy);
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

std::optional<u64> DmaWriteTracker::BlockLinearBytes(bool remap) const {
    const u32 block = dst_params[0];
    const u32 width = dst_params[1];
    const u32 height = dst_params[2];
    const u32 depth = dst_params[3];
    const u32 layer = dst_params[4];
    if ((block & 0xf) != 0 || layer != 0 || width == 0 || height == 0) {
        return std::nullopt; // As the copies assume (block width one, layer 0)
    }
    // width * bytes per pixel is the same for every bpp_shift the copies pick.
    const u32 bytes_per_pixel =
        remap ? (((remap_components >> 16) & 3) + 1) * (((remap_components >> 24) & 3) + 1) : 1;
    return Tegra::Texture::CalculateSize(true, bytes_per_pixel, width, height,
                                         std::max<u32>(depth, 1), (block >> 4) & 0xf,
                                         (block >> 8) & 0xf);
}

void BlitWriteTracker::Note(GpuSplit& split, const Tegra::MemoryManager& memory, u32 method,
                            u32 argument, bool exec) {
    if (method >= DstMethod && method < DstMethod + dst.size()) {
        dst[method - DstMethod] = argument;
        return;
    }
    if (!exec) {
        return;
    }
    if (const auto range = DestinationRange()) {
        NoteRangeWriteGpu(split, memory, range->first, range->second, Producer::Copy);
    } else {
        split.NoteUnboundedWrite(Producer::Copy);
    }
}

std::optional<GpuRange> BlitWriteTracker::DestinationRange() const {
    using Tegra::Engines::Fermi2D;
    const auto format = VideoCore::Surface::TryPixelFormatFromRenderTargetFormat(
        static_cast<Tegra::RenderTargetFormat>(dst[0]));
    const u32 layer = dst[4];
    const GPUVAddr address = (GPUVAddr{dst[8]} << 32) | dst[9];
    if (macros || !format || layer != 0 || address == 0) {
        return std::nullopt;
    }
    const u32 block = dst[2];
    const u32 width = dst[6];
    const u32 height = dst[7];
    if (static_cast<Fermi2D::MemoryLayout>(dst[1]) == Fermi2D::MemoryLayout::BlockLinear) {
        return GpuRange{address, Tegra::Texture::CalculateSize(
                                     true, VideoCore::Surface::BytesPerBlock(*format), width,
                                     height, std::max<u32>(dst[3], 1), (block >> 4) & 0xf,
                                     (block >> 8) & 0xf)};
    }
    return GpuRange{address, u64{dst[5]} * height};
}

bool CopyRecordTracker::NoteDma(u32 method, u32 argument, Ir::CopyBody& body) {
    using Tegra::Engines::MaxwellDMA;
    if (method >= 0x100 && method < 0x108) {
        dma[method - 0x100] = argument;
        return false;
    }
    if (method == 0x1c2) {
        remap_components = argument;
        return false;
    }
    if (method != DmaWriteTracker::LaunchMethod) {
        return false;
    }
    MaxwellDMA::LaunchDMA launch;
    std::memcpy(&launch, &argument, sizeof(launch));
    using LaunchDMA = MaxwellDMA::LaunchDMA;
    if (launch.data_transfer_type.Value() == LaunchDMA::DataTransferType::NONE) {
        return false;
    }
    body.op = Ir::CopyOp::Dma;
    body.source = MaxwellDMA::PackedGPUVAddr{dma[0], dma[1]};
    body.destination = MaxwellDMA::PackedGPUVAddr{dma[2], dma[3]};
    const bool pitch = launch.src_memory_layout.Value() == LaunchDMA::MemoryLayout::PITCH &&
                       launch.dst_memory_layout.Value() == LaunchDMA::MemoryLayout::PITCH;
    body.flags = launch.dst_memory_layout.Value() == LaunchDMA::MemoryLayout::PITCH ? 0 : 1;
    if (pitch && launch.remap_enable == 0) {
        const u64 lines = launch.multi_line_enable != 0 ? std::max<u32>(dma[7], 1) : 1;
        body.bytes = (lines - 1) * static_cast<u64>(static_cast<s32>(dma[5])) + dma[6];
    } else {
        body.bytes = Ir::Unknown;
    }
    return true;
}

bool CopyRecordTracker::NoteBlit(u32 method, u32 argument, Ir::CopyBody& body) {
    using Tegra::Engines::Fermi2D;
    switch (method) {
    case FERMI2D_REG_INDEX(dst.addr_upper):
        blit[0] = argument;
        return false;
    case FERMI2D_REG_INDEX(dst.addr_lower):
        blit[1] = argument;
        return false;
    case FERMI2D_REG_INDEX(src.addr_upper):
        blit[2] = argument;
        return false;
    case FERMI2D_REG_INDEX(src.addr_lower):
        blit[3] = argument;
        return false;
    case FERMI2D_REG_INDEX(pixels_from_memory.src_y0) + 1:
        body.op = Ir::CopyOp::Blit2D;
        body.flags = 1;
        body.destination = (GPUVAddr{blit[0]} << 32) | blit[1];
        body.source = (GPUVAddr{blit[2]} << 32) | blit[3];
        body.bytes = Ir::Unknown;
        return true;
    default:
        return false;
    }
}

} // namespace VideoCommon::V2
