// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>

#ifdef __linux__
#include <sys/prctl.h>
#elif defined(_WIN32)
#include "common/windows/timer_resolution.h"
#endif

#include "common/assert.h"
#include "common/dkb_cpu_trace.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/thread.h"
#include "core/core.h"
#include "core/frontend/graphics_context.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/gpu.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_profile.h"
#include "video_core/host1x/host1x.h"
#include "video_core/renderer_base.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/v2_stats.h"

namespace VideoCommon::GPUThread {

void SignalCommandFence(SynchState& state, u64 fence, bool block) {
    state.signaled_fence.store(fence);
    if (block) {
        // We have to lock the write_lock to ensure that the condition_variable wait not get a
        // race between the check and the lock itself.
        std::scoped_lock lk{state.write_lock};
        state.cv.notify_all();
    }
}

ThreadManager::ThreadManager(Core::System& system_)
    : system{system_}
{}

ThreadManager::~ThreadManager() = default;

void ThreadManager::StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context, Tegra::Control::Scheduler& scheduler) {
    rasterizer = renderer.ReadRasterizer();
    thread = std::jthread([&](std::stop_token stop_token) {
        const Common::DkbCpuTrace::Registration cpu_trace{Common::DkbCpuTrace::Role::GpuCommand};
        Common::SetCurrentThreadName("GPU");
        // Pipeline v2 P5a: this thread owns the caches while the split doesn't run.
        VideoCommon::V2::Commit::t_gpu_thread = true;
#ifdef __linux__
        // The pending-channel polls below sleep 50 us; the default timer slack adds up to 50 more.
        prctl(PR_SET_TIMERSLACK, 1UL, 0, 0, 0);
#endif
        Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
        Common::SetCurrentThreadToPerformanceCores();
        if (Settings::values.pin_gpu_threads.GetValue()) {
            Common::PinCurrentThread(Common::PinnedRole::GpuFront);
        }
        system.RegisterHostThread();

        auto current_context = context.Acquire();
        CpuRecord next;
        // When a pending channel last moved or a command arrived (see the polling below).
        auto last_progress = std::chrono::steady_clock::now();
        namespace Profile = VideoCommon::Nx2Profile;
        auto& split = VideoCommon::GpuSplit::Get();
        while (!stop_token.stop_requested()) {
            Profile::Report();
            V2::Stats::Report("GPU thread");
            split.MaybeStart(system);
            if (scheduler.HasPendingChannels()) {
                // Suspended channels wait on semaphores the CPU releases: poll them meanwhile.
                bool resumed = false;
                {
                    Profile::Scope scope{Profile::Zone::ResumePending};
                    resumed = scheduler.ResumePending(system.GPU());
                }
                if (resumed) {
                    last_progress = std::chrono::steady_clock::now();
                }
                if (!state.queue.TryPop(next)) {
                    if (!scheduler.HasPendingChannels()) {
                        continue;
                    }
                    // The CPU usually releases the semaphore soon after: poll for half a
                    // millisecond, then in short sleeps. Ocarina of Time waits here ~330 times a
                    // second and its present waits for the work behind them: with 1 ms sleeps the
                    // present waited ~17 ms a frame in Kokiri village, with 50 us ~1 ms. A channel
                    // still waiting after 2 s (menus, loading) goes back to 1 ms sleeps.
                    Profile::Scope scope{Profile::Zone::PendingSleep};
                    if (split.AcquireMode() != V2::Mode::Off) [[unlikely]] {
                        // Polls no longer drain: hand the back what the front holds before
                        // sleeping (0758309bcc), or it waits out the sleep without work.
                        split.Publish();
                    }
                    const auto waited = std::chrono::steady_clock::now() - last_progress;
                    if (waited < std::chrono::microseconds{500}) {
                        std::this_thread::yield();
                    } else if (waited < std::chrono::seconds{2}) {
#ifdef _WIN32
                        // sleep_for(50us) rounds up to Sleep(1) (>= 1 ms) on Windows;
                        // SleepForOneTick uses NtDelayExecution(-1) for one high-res timer tick.
                        Common::Windows::SleepForOneTick();
#else
                        std::this_thread::sleep_for(std::chrono::microseconds{50});
#endif
                    } else {
                        std::this_thread::sleep_for(std::chrono::milliseconds{1});
                    }
                    continue;
                }
            } else {
                Profile::Scope scope{Profile::Zone::Idle};
                state.queue.PopWait(next, stop_token);
            }
            if (stop_token.stop_requested()) {
                break;
            }
            if (split.Active()) {
                // Only command lists are parsed here; everything else runs on the back thread
                // in order, and so does the signal that the command ran.
                if (next.op == CpuOp::Submit) {
                    Profile::Scope scope{Profile::Zone::Push};
                    scheduler.Push(system.GPU(), next.channel, std::move(next.list));
                } else {
                    SendToBack(next, renderer.ReadRasterizer());
                }
                const V2::Ir::SyncPointBody signal{.id = 0,
                                                   .value = 0,
                                                   .op = V2::Ir::SyncPointOp::CommandSignal,
                                                   .object = split.Ref(&state),
                                                   .a = next.sequence,
                                                   .b = next.block ? 1U : 0U};
                split.Emit(V2::Ir::Kind::SyncPoint, signal,
                           [state_ptr = &state, fence = next.sequence, block = next.block] {
                    SignalCommandFence(*state_ptr, fence, block);
                }, VideoCommon::GpuSplit::FastDrawNeutral);

                split.Publish();
                last_progress = std::chrono::steady_clock::now();
                continue;
            }
            switch (next.op) {
            case CpuOp::Submit: {
                Profile::Scope scope{Profile::Zone::Push};
                scheduler.Push(system.GPU(), next.channel, std::move(next.list));
                break;
            }
            case CpuOp::Tick: {
                Profile::Scope scope{Profile::Zone::Tick};
                system.GPU().TickWork();
                break;
            }
            case CpuOp::Map: {
                Profile::Scope scope{Profile::Zone::Run};
                system.GPU().ExecuteMap(next.map);
                break;
            }
            case CpuOp::Notify:
                break;
            case CpuOp::SignalEvent: {
                Profile::Scope scope{Profile::Zone::Run};
                system.GPU().SignalEventInOrder(reinterpret_cast<void*>(next.object));
                break;
            }
            case CpuOp::Flush: {
                Profile::Scope scope{Profile::Zone::FlushInvalidate};
                renderer.ReadRasterizer()->FlushRegion(next.address, next.size);
                break;
            }
            }
            SignalCommandFence(state, next.sequence, next.block);
            last_progress = std::chrono::steady_clock::now();
        }
    });
}

void ThreadManager::SendToBack(CpuRecord& record, VideoCore::RasterizerInterface* rasterizer_ptr) {
    auto& split = VideoCommon::GpuSplit::Get();
    switch (record.op) {
    case CpuOp::Submit:
        break;
    case CpuOp::Tick: {
        if (split.TypedIr()) {
            // Composites at their position in the stream (MASTER_PLAN §5.2 Present records),
            // before the TickWork that counts their placeholders' fences.
            for (const V2::Ir::PresentBody& present : system.GPU().TakeQueuedPresents()) {
                split.Emit(V2::Ir::Kind::Present, present,
                           [gpu = &system.GPU(), present] { gpu->RunPresent(present); });
            }
        }
        // TickWork ends fast draw reuse itself when it runs a request.
        const V2::Ir::BarrierBody body{.op = V2::Ir::BarrierOp::TickWork,
                                       .object = split.Ref(&system.GPU()),
                                       .a = 0,
                                       .b = 0};
        split.Emit(V2::Ir::Kind::Barrier, body, [gpu = &system.GPU()] { gpu->TickWork(); },
                   VideoCommon::GpuSplit::FastDrawNeutral);
        break;
    }
    case CpuOp::Map: {
        // GPU memory maps unmap caches (UnmapMemory downloads): the back's state.
        V2::Ir::MapBody body = record.map;
        body.object = split.Ref(reinterpret_cast<const void*>(record.map.object));
        body.object2 = split.Ref(reinterpret_cast<const void*>(record.map.object2));
        split.Emit(V2::Ir::Kind::Map, body,
                   [gpu = &system.GPU(), map = record.map] { gpu->ExecuteMap(map); });
        if ((record.map.flags & V2::Ir::MapBeforeLaterReads) != 0) {
            // The front reads later command lists through the GPU page table it changes.
            split.Drain(VideoCommon::DrainSite::Map);
        }
        break;
    }
    case CpuOp::Notify:
        break; // Only the command signal
    case CpuOp::SignalEvent: {
        const V2::Ir::SyncPointBody body{
            .id = 0,
            .value = 0,
            .op = V2::Ir::SyncPointOp::SignalEvent,
            .object = split.Ref(reinterpret_cast<const void*>(record.object)),
            .a = 0,
            .b = 0};
        split.Emit(V2::Ir::Kind::SyncPoint, body,
                   [gpu = &system.GPU(), event = record.object] {
            gpu->SignalEventInOrder(reinterpret_cast<void*>(event));
        });
        break;
    }
    case CpuOp::Flush: {
        const V2::Ir::ReadbackBody body{.op = V2::Ir::ReadbackOp::Flush,
                                        .object = split.Ref(rasterizer_ptr),
                                        .address = record.address,
                                        .bytes = record.size};
        split.Emit(V2::Ir::Kind::Readback, body,
                   [rasterizer_ptr, addr = record.address, size = record.size] {
            rasterizer_ptr->FlushRegion(addr, size);
        });
        break;
    }
    }
}

void ThreadManager::SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async) {
    CpuRecord record{.op = CpuOp::Submit, .channel = channel};
    record.list = std::move(entries);
    PushCommand(std::move(record), false, is_async);
}

void ThreadManager::FlushRegion(DAddr addr, u64 size, bool is_async) {
    if (!is_async) {
        // Always flush with synchronous GPU mode
        PushCommand(CpuRecord{.op = CpuOp::Flush, .address = addr, .size = size}, false, is_async);
    }
}

void ThreadManager::TickGPU(bool is_async) {
    PushCommand(CpuRecord{.op = CpuOp::Tick}, false, is_async);
}

void ThreadManager::QueueMap(const V2::Ir::MapBody& map, bool is_async) {
    PushCommand(CpuRecord{.op = CpuOp::Map, .map = map}, false, is_async);
}

u64 ThreadManager::Notify(bool is_async) {
    return PushCommand(CpuRecord{.op = CpuOp::Notify}, false, is_async);
}

void ThreadManager::SignalEvent(void* event, bool is_async) {
    PushCommand(CpuRecord{.op = CpuOp::SignalEvent, .object = reinterpret_cast<u64>(event)}, false,
                is_async);
}

void ThreadManager::InvalidateRegion(DAddr addr, u64 size) {
    rasterizer->OnCacheInvalidation(addr, size);
}

void ThreadManager::FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async) {
    if (Settings::IsGPULevelHigh()) {
        if (!is_async) {
            PushCommand(CpuRecord{.op = CpuOp::Flush, .address = addr, .size = size}, false,
                        is_async);
        } else {
            auto& gpu = system.GPU();
            const u64 fence = gpu.RequestFlush(addr, size);
            TickGPU(is_async);
            gpu.WaitForSyncOperation(fence);
        }
    }
    rasterizer->OnCacheInvalidation(addr, size);
}

u64 ThreadManager::PushCommand(CpuRecord&& record, bool block, bool is_async) {
    if (!is_async) {
        // In synchronous GPU mode, block the caller until the command has executed
        block = true;
    }

    std::unique_lock lk(state.write_lock);
    const u64 fence{++state.last_fence};
    record.sequence = fence;
    record.block = block;
    state.queue.EmplaceWait(std::move(record));

    if (block) {
        state.cv.wait(lk, thread.get_stop_token(), [this, fence] {
            return fence <= state.signaled_fence.load(std::memory_order_relaxed);
        });
    }

    return fence;
}

void ThreadManager::NotifyShutdown() {
    if (thread.joinable()) {
        thread.request_stop();
        thread.join();
    }
    VideoCommon::GpuSplit::Get().Stop();
}

} // namespace VideoCommon::GPUThread
