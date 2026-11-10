// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <span>

#include "common/logging.h"
#include "core/core.h"
#include "video_core/cache_types.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/engine_interface.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/memory_manager.h"
#include "video_core/query_cache/types.h"
#include "video_core/query_dma.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/v2/ir_check.h"
#include "video_core/v2/ir_exec.h"
#include "video_core/v2/macro_hle.h"
#include "video_core/v2/word_readback.h"

namespace VideoCommon::V2::IrExec {

namespace {

using VideoCore::RasterizerInterface;

void Barrier(const Ir::BarrierBody& body, const Context& context) {
    void* const object = context.objects.Get(body.object);
    switch (body.op) {
    case Ir::BarrierOp::WaitForIdle:
        static_cast<RasterizerInterface*>(object)->WaitForIdle();
        return;
    case Ir::BarrierOp::InvalidateCaches:
        static_cast<RasterizerInterface*>(object)->InvalidateGPUCache();
        return;
    case Ir::BarrierOp::ReleaseFences:
        static_cast<RasterizerInterface*>(object)->ReleaseFences(body.a !=
                                                                 Ir::ReleaseFencesUnforced);
        return;
    case Ir::BarrierOp::SignalReference:
        static_cast<RasterizerInterface*>(object)->SignalReference();
        return;
    case Ir::BarrierOp::TickWork:
        static_cast<Tegra::GPU*>(object)->TickWork();
        return;
    case Ir::BarrierOp::Invalidate:
        static_cast<RasterizerInterface*>(object)->OnCacheInvalidation(body.a, body.b);
        return;
    case Ir::BarrierOp::CommandListEnd: {
        auto* const gpu = static_cast<Tegra::GPU*>(object);
        gpu->FlushCommands();
        gpu->OnCommandListEnd();
        return;
    }
    case Ir::BarrierOp::SyncFence:
        static_cast<Tegra::DmaPusher*>(object)->SignalSyncFence();
        return;
    case Ir::BarrierOp::Count:
        break;
    }
    LOG_ERROR(HW_GPU, "V2 IR: unknown barrier op {}", static_cast<u32>(body.op));
}

void Readback(const Ir::RecordView& record, const Ir::ReadbackBody& body,
              const Context& context) {
    void* const object = context.objects.Get(body.object);
    switch (body.op) {
    case Ir::ReadbackOp::Flush:
        static_cast<RasterizerInterface*>(object)->FlushRegion(body.address, body.bytes);
        return;
    case Ir::ReadbackOp::FlushCache:
        static_cast<RasterizerInterface*>(object)->FlushRegion(
            body.address, body.bytes, static_cast<VideoCommon::CacheType>(body.cache));
        return;
    case Ir::ReadbackOp::MemoryFlush:
        static_cast<Tegra::MemoryManager*>(object)->FlushRegion(
            body.address, body.bytes, VideoCommon::CacheType::BufferCache);
        return;
    case Ir::ReadbackOp::MacroDmaFlush:
        MacroDmaFlush(*static_cast<Tegra::MemoryManager*>(object),
                      *context.objects.As<RasterizerInterface>(body.object2), body.address,
                      body.bytes);
        return;
    case Ir::ReadbackOp::QueryFlush:
        QueryDma::FlushPending(*static_cast<Tegra::MemoryManager*>(object),
                               *context.objects.As<RasterizerInterface>(body.object2),
                               body.address, body.bytes);
        return;
    case Ir::ReadbackOp::SegmentDirty:
        static_cast<Tegra::Engines::EngineInterface*>(object)->current_dirty =
            context.objects.As<Tegra::MemoryManager>(body.object2)
                ->IsMemoryDirty(body.address, body.bytes);
        return;
    case Ir::ReadbackOp::ListPages: {
        const auto* const pages = reinterpret_cast<const DAddr*>(
            context.arena.At(record.payload.index));
        static_cast<RasterizerInterface*>(object)->FlushGpuWrittenCommandListPages(
            std::span<const DAddr>{pages, record.payload.words});
        return;
    }
    case Ir::ReadbackOp::WordReadback:
        context.objects.As<WordReadback>(body.object2)
            ->Run(*static_cast<RasterizerInterface*>(object), body.address, body.bytes);
        return;
    case Ir::ReadbackOp::MacroRenderEnable:
        static_cast<Tegra::Engines::Maxwell3D*>(object)->RunMacroHleRenderEnable(
            context.system, body.address, static_cast<u32>(body.bytes), (body.cache & 1) != 0,
            (body.cache & 2) != 0);
        return;
    case Ir::ReadbackOp::Count:
        break;
    }
    LOG_ERROR(HW_GPU, "V2 IR: unknown readback op {}", static_cast<u32>(body.op));
}

void SyncPoint(const Ir::SyncPointBody& body, const Context& context) {
    void* const object = context.objects.Get(body.object);
    switch (body.op) {
    case Ir::SyncPointOp::Increment:
        static_cast<RasterizerInterface*>(object)->SignalSyncPoint(body.id);
        return;
    case Ir::SyncPointOp::CommandSignal:
        GPUThread::SignalCommandFence(*static_cast<GPUThread::SynchState*>(object), body.a,
                                      body.b != 0);
        return;
    case Ir::SyncPointOp::SignalEvent:
        context.system.GPU().SignalEventInOrder(object);
        return;
    case Ir::SyncPointOp::Count:
        break;
    }
    LOG_ERROR(HW_GPU, "V2 IR: unknown sync point op {}", static_cast<u32>(body.op));
}

void Draw(const Ir::RecordView& record, const Ir::DrawBody& body, const Context& context) {
    switch (body.op) {
    case Ir::DrawOp::Predicted:
        if (IrCheck::g_active) {
            IrCheck::NoteDrawRecord(body);
        }
        return;
    case Ir::DrawOp::MacroEnd:
        if (IrCheck::g_active) {
            IrCheck::NoteDrawRecord(body);
        }
        context.objects.As<Tegra::Engines::Maxwell3D>(body.engine)
            ->FinishFrontMacro(context.system);
        return;
    case Ir::DrawOp::HostIndirect: {
        if (IrCheck::g_active) {
            IrCheck::NoteDrawRecord(body);
        }
        const MacroHle::Draw draw{
            .kind = static_cast<MacroHle::Kind>(body.hle.kind),
            .topology = body.hle.topology,
            .address = body.hle.address,
            .count_address = body.hle.count_address,
            .max_draws = body.hle.max_draws,
            .stride = body.hle.stride,
            .base_vertex_attribute = body.hle.base_vertex_attribute != 0,
        };
        context.objects.As<Tegra::Engines::Maxwell3D>(body.engine)
            ->RunMacroHleDraw(context.system, draw);
        return;
    }
    }
    (void)record;
}

} // Anonymous namespace

bool IsInformational(const Ir::RecordView& record) {
    switch (record.kind) {
    case Ir::Kind::Dispatch:
    case Ir::Kind::Copy:
    case Ir::Kind::Acquire:
        return true;
    case Ir::Kind::Draw:
        return record.As<Ir::DrawBody>().op == Ir::DrawOp::Predicted;
    default:
        return false;
    }
}

void MacroDmaFlush(Tegra::MemoryManager& memory, VideoCore::RasterizerInterface& rasterizer,
                   u64 address, u64 bytes) {
    if (memory.IsMemoryDirty(address, bytes, VideoCommon::CacheType::BufferCache)) {
        memory.FlushRegion(address, bytes, VideoCommon::CacheType::BufferCache);
    }
    QueryDma::FlushPending(memory, rasterizer, address, bytes);
}

bool Execute(const Ir::RecordView& record, const Context& context) {
    switch (record.kind) {
    case Ir::Kind::Release: {
        const auto& body = record.As<Ir::ReleaseBody>();
        context.objects.As<RasterizerInterface>(body.rasterizer)
            ->Query(body.address, VideoCommon::QueryType::Payload,
                    body.bytes == 4 ? VideoCommon::QueryPropertiesFlags::IsAFence
                                    : VideoCommon::QueryPropertiesFlags::HasTimeout,
                    static_cast<u32>(body.payload), 0);
        return true;
    }
    case Ir::Kind::Barrier:
        Barrier(record.As<Ir::BarrierBody>(), context);
        return true;
    case Ir::Kind::Readback:
        Readback(record, record.As<Ir::ReadbackBody>(), context);
        return true;
    case Ir::Kind::SyncPoint:
        SyncPoint(record.As<Ir::SyncPointBody>(), context);
        return true;
    case Ir::Kind::Channel: {
        const auto& body = record.As<Ir::ChannelBody>();
        context.objects.As<Tegra::Control::Scheduler>(body.scheduler)
            ->BindChannelOnBack(*context.objects.As<Tegra::GPU>(body.gpu), body.bind_id);
        return true;
    }
    case Ir::Kind::Map: {
        // The CPU queue's map with its objects back as pointers.
        Ir::MapBody body = record.As<Ir::MapBody>();
        body.object = reinterpret_cast<u64>(context.objects.Get(static_cast<u32>(body.object)));
        body.object2 =
            reinterpret_cast<u64>(context.objects.Get(static_cast<u32>(body.object2)));
        context.system.GPU().ExecuteMap(body);
        return true;
    }
    case Ir::Kind::Draw:
        Draw(record, record.As<Ir::DrawBody>(), context);
        return true;
    case Ir::Kind::Compact: {
        // Only a macro end is executed; predicted draws and dispatches are informational.
        const auto& body = record.As<Ir::CompactBody>();
        if (body.full_kind == Ir::Kind::Draw &&
            body.op == static_cast<u8>(Ir::DrawOp::MacroEnd)) {
            context.objects.As<Tegra::Engines::Maxwell3D>(body.engine)
                ->FinishFrontMacro(context.system);
        }
        return true;
    }
    case Ir::Kind::Dispatch:
        // EDEN_NX2_V2_IR_DRAWS verify: the window the next records' dispatches count in (Fix
        // pass 1: only the never-called IsInformational noted it, so no window ever opened and
        // every dispatch the back performed counted as unexpected).
        if (IrCheck::g_active) {
            IrCheck::NoteDispatchRecord(record.As<Ir::DispatchBody>());
        }
        return true;
    case Ir::Kind::Copy:
    case Ir::Kind::Acquire:
    case Ir::Kind::Present: // the position of a composite the next TickWork runs
        return true;
    default:
        return false;
    }
}

} // namespace VideoCommon::V2::IrExec
