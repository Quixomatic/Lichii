// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <memory>
#include <utility>

#include "common/assert.h"
#include "common/settings.h"
#include "common/settings_enums.h"
#include "core/core.h"
#include "core/core_timing.h"
#include "core/gpu_dirty_memory_manager.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/graphics_context.h"
#include "core/hle/kernel/k_event.h"
#include "core/hle/service/nvdrv/core/heap_mapper.h"
#include "core/hle/service/nvdrv/nvdata.h"
#include "core/perf_stats.h"
#include "video_core/cdma_pusher.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/gpu.h"
#include "video_core/pending_maps.h"
#include "video_core/gpu_thread.h"
#include "video_core/gpu_timestamp.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/renderer_base.h"
#include "video_core/shader_notify.h"
#include "video_core/gpu_split.h"
#include "video_core/v2/v2_switches.h"

namespace Tegra {

namespace {

/// What a sync request asks of the GPU's owner thread (TickWork, or the split's back for the
/// CPU lane). Pipeline v2 P1b: typed requests instead of std::function closures.
enum class SyncOp : u8 {
    Idle,               ///< Nothing: WaitForIdle's ordering point
    Flush,              ///< FlushRegion(address, size) (RequestFlush)
    CpuReadFlush,       ///< InvalidateGPUCache, then FlushRegion (reactive flush on a CPU read)
    CpuReadback,        ///< InvalidateGPUCache, then FlushRegionForRead into *out (READBACK)
    PreserveInvalidate, ///< FlushRegion(BufferCache), then OnCacheInvalidation
    Present,            ///< Composite the present slot `present.slot` (typed Present record)
    Capture,            ///< *out (std::vector<u8>) = GetAppletCaptureBuffer()
    Debug,              ///< debug(): debug hooks only (EDEN_NX2_FRAME_CAPTURE dumps)
};

struct SyncRequest {
    SyncOp op{};
    u64 sequence{}; ///< The request's sync fence
    u64 address{};
    u64 size{};
    void* out{};
    VideoCommon::V2::Ir::PresentBody present{};
    bool recorded{}; ///< A Present record marks its position (TakeQueuedPresents)
    std::function<void()> debug;
};

/// The layers and fences of a queued composite (PresentBody::slot).
struct PresentSlot {
    std::vector<Tegra::FramebufferConfig> layers;
    std::vector<Service::Nvidia::NvFence> fences;
    size_t counter{};
};

} // Anonymous namespace

struct GPU::Impl {
    explicit Impl(Core::System& system_, bool is_async_, bool use_nvdec_)
        : system{system_}
        , gpu_thread{system_}
        , use_nvdec{use_nvdec_}
        , shader_notify()
        , is_async{is_async_}
    {}

    ~Impl() {
        NotifyShutdown();
        // Rasterizer teardown flushes commands and invalidates the bound engine's dirty flags.
        // Keep channels, their scheduler references, and the graphics context alive until then.
        renderer.reset();
    }

    std::shared_ptr<Control::ChannelState> CreateChannel(s32 channel_id) {
        auto channel_state = std::make_shared<Tegra::Control::ChannelState>(channel_id);
        channels.emplace(channel_id, channel_state);
        scheduler.DeclareChannel(channel_state);
        return channel_state;
    }

    void BindChannel(s32 channel_id) {
        if (bound_channel != channel_id) {
            auto it = channels.find(channel_id);
            ASSERT(it != channels.end());
            bound_channel = channel_id;
            current_channel = it->second.get();
            renderer->ReadRasterizer()->BindChannel(*current_channel);
        }
    }

    std::shared_ptr<Control::ChannelState> AllocateChannel() {
        return CreateChannel(new_channel_id++);
    }

    void InitChannel(Control::ChannelState& to_init, u64 program_id) {
        to_init.Init(system, program_id);
        to_init.BindRasterizer(renderer->ReadRasterizer());
        renderer->ReadRasterizer()->InitializeChannel(to_init);
    }

    void InitAddressSpace(Tegra::MemoryManager& memory_manager) {
        memory_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    void ReleaseChannel(Control::ChannelState& to_release) {
        UNIMPLEMENTED();
    }

    /// Binds a renderer to the GPU.
    void BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer_) {
        renderer = std::move(renderer_);
        system.Host1x().memory_manager.BindInterface(renderer->ReadRasterizer());
        system.Host1x().gmmu_manager.BindRasterizer(renderer->ReadRasterizer());
    }

    /// Flush all current written commands into the host GPU for execution.
    void FlushCommands() {
        renderer->ReadRasterizer()->FlushCommands();
    }

    /// Synchronizes CPU writes with Host GPU memory.
    void InvalidateGPUCache() {
        std::vector<std::pair<DAddr, u64>> writes;
        std::function<void(PAddr, size_t)> callback_writes(
            [&writes](PAddr address, size_t size) { writes.emplace_back(address, size); });
        system.GatherGPUDirtyMemory(callback_writes);
        if (!writes.empty()) {
            renderer->ReadRasterizer()->OnCacheInvalidations(writes);
        }
    }

    [[nodiscard]] u64 CollectedCpuWrites() const {
        u64 total = 0;
        for (const auto& manager : system.GetGPUDirtyMemoryManager()) {
            total += manager.Collected();
        }
        return total;
    }

    void ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
        for (auto& manager : system.GetGPUDirtyMemoryManager()) {
            manager.ForEachPending(callback);
        }
    }

    /// Signal the ending of command list.
    void OnCommandListEnd() {
        renderer->ReadRasterizer()->ReleaseFences(false);
        Settings::UpdateGPUAccuracy();
    }

    /// Queues a typed request for the next TickWork; returns its sync fence.
    [[nodiscard]] u64 RequestSyncOperation(SyncRequest&& request) {
        std::unique_lock lck{sync_request_mutex};
        const u64 fence = ++last_sync_fence;
        request.sequence = fence;
        sync_requests.push_back(std::move(request));
        return fence;
    }

    /// Runs a sync request (the owner of the caches: TickWork or the CPU lane).
    void RunSyncRequest(SyncRequest& request) {
        auto* const rasterizer = renderer->ReadRasterizer();
        switch (request.op) {
        case SyncOp::Idle:
            return;
        case SyncOp::Flush:
            rasterizer->FlushRegion(request.address, request.size);
            return;
        case SyncOp::CpuReadFlush:
            // Apply collected guest writes before downloading GPU data over guest memory.
            InvalidateGPUCache();
            rasterizer->FlushRegion(request.address, request.size);
            return;
        case SyncOp::CpuReadback:
            InvalidateGPUCache();
            rasterizer->FlushRegionForRead(request.address, request.size,
                                           *static_cast<std::function<void()>*>(request.out));
            return;
        case SyncOp::PreserveInvalidate:
            rasterizer->FlushRegion(request.address, request.size,
                                    VideoCommon::CacheType::BufferCache);
            rasterizer->OnCacheInvalidation(request.address, request.size);
            return;
        case SyncOp::Present:
            RunPresent(request.present);
            return;
        case SyncOp::Capture:
            *static_cast<std::vector<u8>*>(request.out) = renderer->GetAppletCaptureBuffer();
            return;
        case SyncOp::Debug:
            request.debug();
            return;
        }
    }

    /// Obtains current flush request fence id.
    [[nodiscard]] u64 CurrentSyncRequestFence() const {
        return current_sync_fence.load(std::memory_order_relaxed);
    }

    /// Returns once the request of `fence` ran, or once the GPU threads stopped for good
    /// (NotifyShutdown): then it never runs.
    void WaitForSyncOperation(const u64 fence) {
        std::unique_lock lck{sync_request_mutex};
        sync_request_cv.wait(lck, [this, fence] {
            return CurrentSyncRequestFence() >= fence || sync_requests_dropped;
        });
    }

    // EDEN_NX2_V2_CPULANE (MASTER_PLAN.md section 6): a guest CPU flush (a read of GPU-written
    // memory, an invalidation that keeps GPU writes) is served by the split's back after the
    // entry it runs now, instead of after everything queued. The ordering point is that entry:
    // GPU work the guest waited for (its fence signalled) has run by then, and work still queued
    // is work the guest hasn't synchronised with, which hardware could also be partway through.
    // Requests that need everything before them (WaitForIdle, composites, captures) keep the
    // ordered TickWork path. The lane has its own fences, so the two never reorder each other.
    [[nodiscard]] bool UseCpuLane() const {
        return is_async && VideoCommon::V2::On(VideoCommon::V2::Switch::CpuLane) &&
               VideoCommon::GpuSplit::Get().BackRunning();
    }

    void RunOnCpuLane(SyncRequest&& request) {
        u64 fence;
        {
            std::scoped_lock lck{lane_mutex};
            if (lane_dropped) [[unlikely]] {
                return;
            }
            fence = ++lane_last;
            request.sequence = fence;
            lane_requests.push_back(std::move(request));
            lane_pending.store(true, std::memory_order_release);
        }
        VideoCommon::GpuSplit::Get().RequestCpuLane();
        // Also through the ordered path: TickWork serves the lane too, should the back not.
        gpu_thread.TickGPU(is_async);
        std::unique_lock lck{lane_mutex};
        lane_cv.wait(lck, [this, fence] { return lane_done >= fence || lane_dropped; });
    }

    void TickCpuLane() {
        // EDEN_NX2_V2_COMMIT: the lane hook also serves work handed to the caches' owner.
        renderer->ReadRasterizer()->ServeCommitRequests();
        if (!lane_pending.load(std::memory_order_acquire)) {
            return;
        }
        std::unique_lock lck{lane_mutex};
        while (!lane_requests.empty()) {
            auto request = std::move(lane_requests.front());
            lane_requests.pop_front();
            lck.unlock();
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::TickWork);
            RunSyncRequest(request);
            lck.lock();
            ++lane_done;
            lane_cv.notify_all();
        }
        lane_pending.store(false, std::memory_order_release);
    }

    void WaitForIdle() {
        const u64 fence = RequestSyncOperation(SyncRequest{.op = SyncOp::Idle});
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
    }

    /// Gets the caches' owner to serve handed-over work: the split's back through its CPU lane
    /// hook (served after its current entry, or at once when it waits for work), else the GPU
    /// thread through a Tick (TickWork serves it).
    void WakeCacheOwner() {
        auto& split = VideoCommon::GpuSplit::Get();
        if (split.BackRunning()) {
            split.RequestCpuLane();
            return;
        }
        gpu_thread.TickGPU(is_async);
    }

    /// Tick pending requests within the GPU.
    void TickWork() {
        renderer->ReadRasterizer()->ServeCommitRequests();
        if (lane_used) [[unlikely]] {
            TickCpuLane();
        }
        std::unique_lock lck{sync_request_mutex};
        while (!sync_requests.empty()) {
            auto request = std::move(sync_requests.front());
            sync_requests.pop_front();
            sync_request_mutex.unlock();
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::TickWork);
            RunSyncRequest(request);
            current_sync_fence.fetch_add(1, std::memory_order_release);
            sync_request_mutex.lock();
            sync_request_cv.notify_all();
        }
    }

    [[nodiscard]] u64 GetTicks() const {
        return ReadGpuTimestamp(system.CoreTiming(), g_nx2_ampere.load(std::memory_order_relaxed),
                                Settings::values.gpu_clock.GetValue(),
                                ampere_anchor.load(std::memory_order_relaxed));
    }

    void RendererFrameEndNotify() {
        ampere_anchor.store(
            AmpereAnchor::At(static_cast<u64>(system.CoreTiming().GetGlobalTimeNs().count()),
                             Settings::values.gpu_clock.GetValue()),
            std::memory_order_relaxed);
        system.GetPerfStats().EndGameFrame();
    }

    /// Performs any additional setup necessary in order to begin GPU emulation.
    /// This can be used to launch any necessary threads and register any necessary
    /// core timing events.
    void Start() {
        Settings::UpdateGPUAccuracy();
        gpu_thread.StartThread(*renderer, renderer->Context(), scheduler);
    }

    void NotifyShutdown() {
        gpu_thread.NotifyShutdown();
        {
            // The GPU threads stopped: no sync operation runs anymore. Guest threads still run
            // until the kernel suspends them, and one waiting for a read's flush (OnCPURead)
            // would wait forever, in host code where the kernel cannot suspend it
            // (KernelCore::SuspendEmulation then waits for it forever). Drop the requests, which
            // may refer to their waiters' frames, and release the waiters.
            std::scoped_lock requests_lock{sync_request_mutex};
            sync_requests.clear();
            sync_requests_dropped = true;
        }
        sync_request_cv.notify_all();
        // Pipeline v2: the same for the CPU lane (EDEN_NX2_V2_CPULANE) and the caches' owner
        // calls (EDEN_NX2_V2_COMMIT).
        {
            std::scoped_lock lane_lock{lane_mutex};
            lane_requests.clear();
            lane_dropped = true;
        }
        lane_cv.notify_all();
        if (renderer) {
            renderer->ReadRasterizer()->OnCacheOwnerStopped();
        }
        std::unique_lock lk{sync_mutex};
        shutting_down.store(true, std::memory_order::relaxed);
        sync_cv.notify_all();
    }

    /// Obtain the CPU Context
    void ObtainContext() {
        if (!cpu_context) {
            cpu_context = renderer->GetRenderWindow().CreateSharedContext();
        }
        cpu_context->MakeCurrent();
    }

    /// Release the CPU Context
    void ReleaseContext() {
        cpu_context->DoneCurrent();
    }

    /// Push GPU command entries to be processed
    void PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
        gpu_thread.SubmitList(channel, std::move(entries), is_async);
    }

    void MapInOrder(VideoCommon::V2::Ir::MapBody map) {
        if ((map.flags & VideoCommon::V2::Ir::MapBeforeLaterReads) != 0) {
            map.pending_id = pending_maps.Add(map.gpu_address, map.size);
        }
        gpu_thread.QueueMap(map, is_async);
    }

    void ExecuteMap(const VideoCommon::V2::Ir::MapBody& map) {
        using VideoCommon::V2::Ir::MapOp;
        auto* const gmmu = reinterpret_cast<Tegra::MemoryManager*>(map.object);
        switch (map.op) {
        case MapOp::GmmuMap:
            gmmu->Map(map.gpu_address, map.device_address, map.size, Tegra::PTEKind::PITCH, false);
            break;
        case MapOp::GmmuUnmap:
            gmmu->Unmap(map.gpu_address, map.size);
            break;
        case MapOp::SmmuRelease: {
            auto* const smmu = reinterpret_cast<Tegra::MaxwellDeviceMemoryManager*>(map.object);
            smmu->Unmap(map.device_address, map.size);
            smmu->Free(map.device_address, map.extra);
            break;
        }
        case MapOp::HeapRelease:
            if (map.object != 0) {
                reinterpret_cast<Service::Nvidia::NvCore::HeapMapper*>(map.object)
                    ->Unmap(map.cpu_address, map.size);
            }
            break;
        case MapOp::SparseRelease: {
            auto* const smmu = reinterpret_cast<Tegra::MaxwellDeviceMemoryManager*>(map.object2);
            gmmu->Unmap(map.gpu_address, map.size);
            smmu->Unmap(map.device_address, map.size);
            smmu->Free(map.device_address, map.extra);
            break;
        }
        case MapOp::SparseTile: {
            auto* const smmu = reinterpret_cast<Tegra::MaxwellDeviceMemoryManager*>(map.object);
            smmu->Unmap(map.device_address, map.size);
            if (map.cpu_address != 0) {
                smmu->Map(map.device_address, map.cpu_address, map.size,
                          Core::Asid{static_cast<size_t>(map.extra)}, true);
            }
            break;
        }
        case MapOp::Count:
            LOG_ERROR(HW_GPU, "Unknown map op {}", static_cast<u32>(map.op));
            break;
        }
        if (map.pending_id != 0) {
            pending_maps.Remove(map.pending_id);
        }
    }

    u64 MarkInOrder() {
        return gpu_thread.Notify(is_async);
    }

    bool InOrderDone(u64 marker) const {
        return gpu_thread.SignaledSequence() >= marker;
    }

    void SignalInOrder(Kernel::KEvent* event) {
        gpu_thread.SignalEvent(event, is_async);
    }

    void SignalEventInOrder(void* event) {
        static_cast<Kernel::KEvent*>(event)->Signal(system.Kernel());
    }

    bool IsMapPending(GPUVAddr gpu_addr, u64 size) const {
        return pending_maps.Overlaps(gpu_addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(DAddr addr, u64 size) {
        gpu_thread.FlushRegion(addr, size, is_async);
    }

    VideoCore::RasterizerDownloadArea OnCPURead(DAddr addr, u64 size) {
        auto raster_area = renderer->ReadRasterizer()->GetFlushArea(addr, size);
        if (raster_area.preemtive) {
            return raster_area;
        }
        raster_area.preemtive = true;
        // A guest CPU read of GPU-written memory (reactive flushing): count them, they decide
        // whether buffer data must be copied back at every fence (EDEN_NX2_BUFFER_DOWNLOADS).
        if (static std::atomic<u64> reads{}; (reads.fetch_add(1, std::memory_order_relaxed) & 63) == 0) {
            LOG_INFO(HW_GPU, "CPU read of GPU-written memory [{:#x}, {:#x}) ({} so far)",
                     raster_area.start_address, raster_area.end_address,
                     reads.load(std::memory_order_relaxed));
        }
        if (VideoCommon::V2::On(VideoCommon::V2::Switch::Readback) && is_async) [[unlikely]] {
            // EDEN_NX2_V2_READBACK: the caches' owner records and submits the download; this
            // thread waits for the GPU and writes guest memory itself (Ryujinx's readback on the
            // reading thread), so the owner never blocks on the GPU for it.
            std::function<void()> land;
            const SyncRequest record{
                .op = SyncOp::CpuReadback,
                .address = raster_area.start_address,
                .size = raster_area.end_address - raster_area.start_address,
                .out = &land,
            };
            if (UseCpuLane()) {
                lane_used = true;
                RunOnCpuLane(SyncRequest{record});
            } else {
                const u64 fence = RequestSyncOperation(SyncRequest{record});
                gpu_thread.TickGPU(is_async);
                WaitForSyncOperation(fence);
            }
            if (land) {
                land();
            }
            return raster_area;
        }
        const SyncRequest flush{
            .op = SyncOp::CpuReadFlush,
            .address = raster_area.start_address,
            .size = raster_area.end_address - raster_area.start_address,
        };
        if (UseCpuLane()) [[unlikely]] {
            lane_used = true;
            RunOnCpuLane(SyncRequest{flush});
            return raster_area;
        }
        const u64 fence = RequestSyncOperation(SyncRequest{flush});
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(fence);
        return raster_area;
    }

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
        if (preserve_gpu_writes) {
            VideoCore::RasterizerInterface* rasterizer = renderer->ReadRasterizer();
            if (!rasterizer->TryCacheInvalidation(addr, size)) {
                const SyncRequest flush{
                    .op = SyncOp::PreserveInvalidate, .address = addr, .size = size};
                if (UseCpuLane()) [[unlikely]] {
                    lane_used = true;
                    RunOnCpuLane(SyncRequest{flush});
                    return;
                }
                const u64 fence = RequestSyncOperation(SyncRequest{flush});
                gpu_thread.TickGPU(is_async);
                WaitForSyncOperation(fence);
            }
            return;
        }
        gpu_thread.InvalidateRegion(addr, size);
    }

    bool OnCPUWrite(DAddr addr, u64 size) {
        return renderer->ReadRasterizer()->OnCPUWrite(addr, size);
    }

    /// Notify rasterizer that any caches of the specified region should be flushed and invalidated
    void FlushAndInvalidateRegion(DAddr addr, u64 size) {
        gpu_thread.FlushAndInvalidateRegion(addr, size, is_async);
    }

    void RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers, std::vector<Service::Nvidia::NvFence>&& fences) {
        const size_t num_fences{fences.size()};
        size_t current_request_counter{};
        if (num_fences != 0) {
            std::unique_lock<std::mutex> lk(request_swap_mutex);
            if (free_swap_counters.empty()) {
                current_request_counter = request_swap_counters.size();
                request_swap_counters.emplace_back(num_fences);
            } else {
                current_request_counter = free_swap_counters.front();
                request_swap_counters[current_request_counter] = num_fences;
                free_swap_counters.pop_front();
            }
        }
        // A typed Present request: its layers and fences wait in a present slot.
        VideoCommon::V2::Ir::PresentBody present{
            .frame = 0,
            .slot = 0,
            .num_layers = static_cast<u32>(layers.size()),
            .num_fences = static_cast<u32>(num_fences),
            .reserved = 0,
        };
        {
            std::unique_lock<std::mutex> lk(request_swap_mutex);
            present.frame = ++present_count;
            if (free_present_slots.empty()) {
                present.slot = static_cast<u32>(present_slots.size());
                present_slots.emplace_back();
            } else {
                present.slot = free_present_slots.back();
                free_present_slots.pop_back();
            }
            present_slots[present.slot] = {std::move(layers), std::move(fences),
                                           current_request_counter};
        }
        pending_composite_fence =
            RequestSyncOperation(SyncRequest{.op = SyncOp::Present, .present = present});
        gpu_thread.TickGPU(is_async);
    }

    /// The queued composites not recorded yet, for Present records. The request stays queued:
    /// the TickWork after the record composites it, exactly where and when it did without the
    /// records (vi2's composite order and pacing are unchanged).
    std::vector<VideoCommon::V2::Ir::PresentBody> TakeQueuedPresents() {
        std::vector<VideoCommon::V2::Ir::PresentBody> presents;
        std::unique_lock lck{sync_request_mutex};
        for (SyncRequest& request : sync_requests) {
            if (request.op == SyncOp::Present && !request.recorded) {
                presents.push_back(request.present);
                request.recorded = true;
            }
        }
        return presents;
    }

    /// Runs a Present request (TickWork): composites now, or once its fences signal.
    void RunPresent(const VideoCommon::V2::Ir::PresentBody& present) {
        PresentSlot slot;
        {
            std::unique_lock<std::mutex> lk(request_swap_mutex);
            slot = std::move(present_slots[present.slot]);
            present_slots[present.slot] = {};
            free_present_slots.push_back(present.slot);
        }
        if (slot.fences.empty()) {
            renderer->Composite(slot.layers);
            return;
        }
        auto& syncpoint_manager = system.Host1x().GetSyncpointManager();
        const auto executer = [this, current_request_counter = slot.counter,
                               composite_layers = slot.layers]() {
            {
                std::unique_lock<std::mutex> lk(request_swap_mutex);
                if (--request_swap_counters[current_request_counter] != 0) {
                    return;
                }
                free_swap_counters.push_back(current_request_counter);
            }
            renderer->Composite(composite_layers);
        };
        for (const auto& fence : slot.fences) {
            syncpoint_manager.RegisterGuestAction(fence.id, fence.value, executer);
        }
    }

    void WaitForComposite() {
        const u64 fence = pending_composite_fence;
        if (fence == 0) {
            return;
        }
        pending_composite_fence = 0;
        if (shutting_down.load(std::memory_order_relaxed)) {
            return;
        }
        WaitForSyncOperation(fence);
    }

    std::vector<u8> GetAppletCaptureBuffer() {
        std::vector<u8> out;

        const auto wait_fence =
            RequestSyncOperation(SyncRequest{.op = SyncOp::Capture, .out = &out});
        gpu_thread.TickGPU(is_async);
        WaitForSyncOperation(wait_fence);

        return out;
    }

    Core::System& system;

    // Destruction of thread must be done before all (non trivial)
    // previous members has been destroyed
    VideoCommon::GPUThread::ThreadManager gpu_thread;

    std::unique_ptr<VideoCore::RendererBase> renderer;
    const bool use_nvdec;

    s32 new_channel_id{1};
    /// Shader build notifier
    VideoCore::ShaderNotify shader_notify;
    /// When true, we are about to shut down emulation session, so terminate outstanding tasks
    std::atomic_bool shutting_down{};

    std::array<std::atomic<u32>, Service::Nvidia::MaxSyncPoints> syncpoints{};

    std::array<std::list<u32>, Service::Nvidia::MaxSyncPoints> syncpt_interrupts;

    std::mutex sync_mutex;
    std::mutex device_mutex;

    std::condition_variable sync_cv;

    std::deque<SyncRequest> sync_requests;
    std::atomic<u64> current_sync_fence{};
    std::atomic<AmpereAnchor> ampere_anchor{};
    u64 last_sync_fence{};

    VideoCommon::PendingMaps pending_maps;
    std::mutex sync_request_mutex;
    std::condition_variable sync_request_cv;
    /// The GPU threads stopped and the requests were dropped (NotifyShutdown, under the mutex).
    bool sync_requests_dropped{};

    // EDEN_NX2_V2_CPULANE (RunOnCpuLane).
    std::mutex lane_mutex;
    std::condition_variable lane_cv;
    std::deque<SyncRequest> lane_requests;
    u64 lane_last{};
    u64 lane_done{};
    bool lane_dropped{}; ///< NotifyShutdown dropped the lane (under lane_mutex)
    std::atomic<bool> lane_pending{};
    std::atomic<bool> lane_used{}; ///< Once set, TickWork serves the lane too

    const bool is_async;

    std::unique_ptr<Core::Frontend::GraphicsContext> cpu_context;

    Tegra::Control::Scheduler scheduler;
    ::Common::unordered_map<s32, std::shared_ptr<Tegra::Control::ChannelState>> channels;
    Tegra::Control::ChannelState* current_channel;
    s32 bound_channel{-1};

    std::deque<size_t> free_swap_counters;
    std::deque<size_t> request_swap_counters;
    std::mutex request_swap_mutex;
    std::vector<PresentSlot> present_slots;  ///< Under request_swap_mutex
    std::vector<u32> free_present_slots;
    u64 present_count{};
    u64 pending_composite_fence{};
};

GPU::GPU(Core::System& system, bool is_async, bool use_nvdec)
    : impl{std::make_unique<Impl>(system, is_async, use_nvdec)}
{}

GPU::~GPU() = default;

std::shared_ptr<Control::ChannelState> GPU::AllocateChannel() {
    return impl->AllocateChannel();
}

void GPU::InitChannel(Control::ChannelState& to_init, u64 program_id) {
    impl->InitChannel(to_init, program_id);
}

void GPU::BindChannel(s32 channel_id) {
    impl->BindChannel(channel_id);
}

void GPU::ReleaseChannel(Control::ChannelState& to_release) {
    impl->ReleaseChannel(to_release);
}

void GPU::InitAddressSpace(Tegra::MemoryManager& memory_manager) {
    impl->InitAddressSpace(memory_manager);
}

void GPU::BindRenderer(std::unique_ptr<VideoCore::RendererBase> renderer) {
    impl->BindRenderer(std::move(renderer));
}

void GPU::FlushCommands() {
    impl->FlushCommands();
}

void GPU::InvalidateGPUCache() {
    impl->InvalidateGPUCache();
}

bool GPU::HasPendingCPUWrites(DAddr addr, u64 size) {
    for (auto& manager : impl->system.GetGPUDirtyMemoryManager()) {
        if (manager.Overlaps(addr, size)) {
            return true;
        }
    }
    return false;
}

void GPU::ForEachPendingCpuWrite(const std::function<void(DAddr, size_t)>& callback) {
    impl->ForEachPendingCpuWrite(callback);
}

void GPU::OnCommandListEnd() {
    impl->OnCommandListEnd();
}

u64 GPU::RequestFlush(DAddr addr, std::size_t size) {
    return impl->RequestSyncOperation(
        SyncRequest{.op = SyncOp::Flush, .address = addr, .size = size});
}

u64 GPU::RunOnGpuThread(std::function<void()>&& action) {
    return impl->RequestSyncOperation(
        SyncRequest{.op = SyncOp::Debug, .debug = std::move(action)});
}

u64 GPU::CurrentSyncRequestFence() const {
    return impl->CurrentSyncRequestFence();
}

void GPU::WaitForSyncOperation(u64 fence) {
    return impl->WaitForSyncOperation(fence);
}

void GPU::TickWork() {
    impl->TickWork();
}

void GPU::TickCpuLane() {
    impl->TickCpuLane();
}

void GPU::WakeCacheOwner() {
    impl->WakeCacheOwner();
}

u64 GPU::CollectedCpuWrites() const {
    return impl->CollectedCpuWrites();
}

/// Gets a mutable reference to the Host1x interface
Host1x::Host1x& GPU::Host1x() {
    return impl->system.Host1x();
}

/// Gets an immutable reference to the Host1x interface.
const Host1x::Host1x& GPU::Host1x() const {
    return impl->system.Host1x();
}

Engines::Maxwell3D& GPU::Maxwell3D() {
    return impl->current_channel->payload->maxwell_3d;
}

const Engines::Maxwell3D& GPU::Maxwell3D() const {
    return impl->current_channel->payload->maxwell_3d;
}

Engines::KeplerCompute& GPU::KeplerCompute() {
    return impl->current_channel->payload->kepler_compute;
}

const Engines::KeplerCompute& GPU::KeplerCompute() const {
    return impl->current_channel->payload->kepler_compute;
}

Tegra::DmaPusher& GPU::DmaPusher() {
    return impl->current_channel->payload->dma_pusher;
}

const Tegra::DmaPusher& GPU::DmaPusher() const {
    return impl->current_channel->payload->dma_pusher;
}

VideoCore::RendererBase& GPU::Renderer() {
    return *impl->renderer;
}

const VideoCore::RendererBase& GPU::Renderer() const {
    return *impl->renderer;
}

VideoCore::ShaderNotify& GPU::ShaderNotify() {
    return impl->shader_notify;
}

const VideoCore::ShaderNotify& GPU::ShaderNotify() const {
    return impl->shader_notify;
}

void GPU::RequestComposite(std::vector<Tegra::FramebufferConfig>&& layers,
                           std::vector<Service::Nvidia::NvFence>&& fences) {
    impl->RequestComposite(std::move(layers), std::move(fences));
}

void GPU::WaitForComposite() {
    impl->WaitForComposite();
}

std::vector<VideoCommon::V2::Ir::PresentBody> GPU::TakeQueuedPresents() {
    return impl->TakeQueuedPresents();
}

std::vector<u8> GPU::GetAppletCaptureBuffer() {
    return impl->GetAppletCaptureBuffer();
}

u64 GPU::GetTicks() const {
    return impl->GetTicks();
}

bool GPU::IsAsync() const {
    return impl->is_async;
}

bool GPU::HasSuspendedChannels() const {
    return impl->scheduler.HasPendingChannelsAnyThread();
}

bool GPU::UseNvdec() const {
    return impl->use_nvdec;
}

void GPU::RendererFrameEndNotify() {
    impl->RendererFrameEndNotify();
}

void GPU::Start() {
    impl->Start();
}

void GPU::NotifyShutdown() {
    impl->NotifyShutdown();
}

void GPU::WaitForIdle() {
    impl->WaitForIdle();
}

void GPU::ObtainContext() {
    impl->ObtainContext();
}

void GPU::ReleaseContext() {
    impl->ReleaseContext();
}

void GPU::PushGPUEntries(s32 channel, Tegra::CommandList&& entries) {
    impl->PushGPUEntries(channel, std::move(entries));
}

VideoCore::RasterizerDownloadArea GPU::OnCPURead(PAddr addr, u64 size) {
    return impl->OnCPURead(addr, size);
}

void GPU::FlushRegion(DAddr addr, u64 size) {
    impl->FlushRegion(addr, size);
}

void GPU::MapInOrder(const VideoCommon::V2::Ir::MapBody& map) {
    impl->MapInOrder(map);
}

void GPU::ExecuteMap(const VideoCommon::V2::Ir::MapBody& map) {
    impl->ExecuteMap(map);
}

u64 GPU::MarkInOrder() {
    return impl->MarkInOrder();
}

bool GPU::InOrderDone(u64 marker) const {
    return impl->InOrderDone(marker);
}

void GPU::SignalInOrder(Kernel::KEvent* event) {
    impl->SignalInOrder(event);
}

void GPU::SignalEventInOrder(void* event) {
    impl->SignalEventInOrder(event);
}

bool GPU::IsMapPending(GPUVAddr gpu_addr, u64 size) const {
    return impl->IsMapPending(gpu_addr, size);
}

void GPU::InvalidateRegion(DAddr addr, u64 size, bool preserve_gpu_writes) {
    impl->InvalidateRegion(addr, size, preserve_gpu_writes);
}

bool GPU::OnCPUWrite(DAddr addr, u64 size) {
    return impl->OnCPUWrite(addr, size);
}

void GPU::FlushAndInvalidateRegion(DAddr addr, u64 size) {
    impl->FlushAndInvalidateRegion(addr, size);
}

} // namespace Tegra
