// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <cstdlib>
#include <memory>
#include <algorithm>
#include <vector>
#include <string_view>
#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>

#include "video_core/renderer_vulkan/vk_query_cache.h"

#include "common/logging.h"
#include "common/settings.h"
#include "common/dkb_cpu_trace.h"
#include "common/thread.h"
#include "video_core/gpu_logging/gpu_logging.h"
#include "video_core/nx2_profile.h"
#include "video_core/renderer_vulkan/vk_command_pool.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_state_tracker.h"
#include "video_core/renderer_vulkan/vk_background_readback.h"
#include "video_core/renderer_vulkan/vk_submit_engine.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/v2/front_wake.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/v2_switches.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/v2/versions.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

namespace {
std::atomic<u64> g_profile_submits{};
std::atomic<u64> g_profile_renderpasses{};
std::atomic<u64> g_profile_finishes{};
std::atomic<u64> g_profile_finish_ns{};
std::atomic<u64> g_profile_window_start{};
std::mutex g_profile_end_mutex;
std::map<std::string, u64> g_profile_ends;

void CountRenderPassEnd(const char* file, u32 line) {
    if (!VideoCommon::Nx2Profile::Enabled()) {
        return;
    }
    std::string_view name{file};
    if (const auto slash = name.find_last_of('/'); slash != std::string_view::npos) {
        name.remove_prefix(slash + 1);
    }
    std::scoped_lock lock{g_profile_end_mutex};
    ++g_profile_ends[fmt::format("{}:{}", name, line)];
}

void ReportSubmitCounts() {
    const u64 now = VideoCommon::Nx2Profile::Now();
    u64 start = g_profile_window_start.load(std::memory_order_relaxed);
    if (start == 0) {
        g_profile_window_start.compare_exchange_strong(start, now);
        return;
    }
    if (now - start < 10'000'000'000ULL ||
        !g_profile_window_start.compare_exchange_strong(start, now)) {
        return;
    }
    const double seconds = static_cast<double>(now - start) / 1e9;
    LOG_INFO(Render_Vulkan,
             "Scheduler per second: {:.0f} submits, {:.0f} render passes, {:.0f} finishes "
             "waiting {:.1f} ms",
             static_cast<double>(g_profile_submits.exchange(0)) / seconds,
             static_cast<double>(g_profile_renderpasses.exchange(0)) / seconds,
             static_cast<double>(g_profile_finishes.exchange(0)) / seconds,
             static_cast<double>(g_profile_finish_ns.exchange(0)) / 1e6 / seconds);
    std::vector<std::pair<u64, std::string>> ends;
    {
        std::scoped_lock lock{g_profile_end_mutex};
        for (const auto& [site, count] : g_profile_ends) {
            ends.emplace_back(count, site);
        }
        g_profile_ends.clear();
    }
    std::sort(ends.rbegin(), ends.rend());
    for (size_t i = 0; i < ends.size() && i < 16; ++i) {
        LOG_INFO(Render_Vulkan, "Render pass ends {:.0f}/s at {}",
                 static_cast<double>(ends[i].first) / seconds, ends[i].second);
    }
}
}

void Scheduler::CommandChunk::ExecuteAll(vk::CommandBuffer cmdbuf,
                                         vk::CommandBuffer upload_cmdbuf) {
    auto command = first;
    while (command != nullptr) {
        auto next = command->GetNext();
        command->Execute(cmdbuf, upload_cmdbuf);
        command->~Command();
        command = next;
    }
    submit = false;
    command_offset = 0;
    first = nullptr;
    last = nullptr;
}

Scheduler::Scheduler(const Device& device_, StateTracker& state_tracker_)
    : device{device_}, state_tracker{state_tracker_},
      master_semaphore{std::make_unique<MasterSemaphore>(device)},
      command_pool{std::make_unique<CommandPool>(*master_semaphore, device)} {

    if (Common::DkbFrameTrace::Enabled()) {
        const auto physical = device.GetPhysical();
        trace_timestamp_bits =
            physical.GetQueueFamilyProperties()[device.GetGraphicsFamily()].timestampValidBits;
        trace_timestamp_period = physical.GetProperties().limits.timestampPeriod;
        if (trace_timestamp_bits != 0 && trace_timestamp_bits <= 64 && trace_timestamp_period > 0) {
            try {
                trace_query_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxSubmissions * 2,
                    .pipelineStatistics = 0,
                });
                trace_submissions.resize(TraceMaxSubmissions);
                trace_work_pool = device.GetLogical().CreateQueryPool({
                    .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queryType = VK_QUERY_TYPE_TIMESTAMP,
                    .queryCount = TraceMaxWork * 2,
                    .pipelineStatistics = 0,
                });
                trace_work_items.resize(TraceMaxWork);
            } catch (const vk::Exception& error) {
                LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp unavailable: {}", error.what());
            }
        } else {
            LOG_INFO(Render_Vulkan, "DKB gpu_timestamp unsupported: valid_bits={} period_ns={}",
                     trace_timestamp_bits, trace_timestamp_period);
        }
    }

    // P5b: the transfer queue (EDEN_NX2_V2_XFERQ) and the submit/finish threads
    // (EDEN_NX2_V2_SUBMITTHREAD) share one submission path; with both off nothing changes.
    if (device.HasTransferQueue()) {
        transfer_queue = std::make_unique<TransferQueue>(device);
    }
    const bool submit_thread = VideoCommon::V2::On(VideoCommon::V2::Switch::SubmitThread) &&
                               device.HasTimelineSemaphore() && device.HasSynchronization2();
    if (VideoCommon::V2::On(VideoCommon::V2::Switch::SubmitThread) && !submit_thread) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_V2_SUBMITTHREAD needs timeline semaphores and "
                                "synchronization2; the worker keeps submitting");
    }
    dynamic_rendering = device.IsDynamicRenderingEnabled();
    track_barriers = VideoCommon::V2::On(VideoCommon::V2::Switch::DynRender);
    if (track_barriers) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_V2_DYNRENDER: {} rendering, range barrier tracker",
                 dynamic_rendering ? "dynamic" : "render pass (fallback)");
    }
    if (submit_thread || transfer_queue) {
        submit_engine = std::make_unique<SubmitEngine>(device, *master_semaphore, submit_mutex,
                                                       submit_thread, transfer_queue.get());
        finish_publishes_tick = submit_thread;
        if (submit_thread) {
            submit_engine->SetProgressHook(
                [](void* self) { static_cast<Scheduler*>(self)->OnGpuProgress(); }, this);
        }
    }

    if (transfer_queue && VideoCommon::V2::On(VideoCommon::V2::Switch::XReadback)) {
        background_readback = std::make_unique<BackgroundReadback>(device, *this);
    } else if (VideoCommon::V2::On(VideoCommon::V2::Switch::XReadback)) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_V2_XREADBACK needs the transfer queue "
                                "(EDEN_NX2_V2_XFERQ and a dedicated transfer family)");
    }

    AcquireNewChunk();
    AllocateWorkerCommandBuffer();
    worker_thread = std::jthread([this](std::stop_token token) { WorkerThread(token); });
}

Scheduler::~Scheduler() {
    if (trace_query_pool) {
        WaitWorker();
        worker_thread.request_stop();
        worker_thread.join();
        if (trace_gpu_tick != 0) {
            master_semaphore->Wait(trace_gpu_tick);
        }
    }
}

u64 Scheduler::Flush(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                     std::source_location location) {
    // EDEN_NX2_V2_DRAWCACHE: a submit is an ordering point (collect epoch). F4-08 (§8): with
    // lazy fences, only a submission carrying something the guest observes is one (a fence, a
    // waiter's request, deferred or wanted work); a heuristic submit orders nothing.
    if (!lazy_fences || flush_pending || submit_wanted || SubmitRequested() ||
        lazy_outstanding_tick == CurrentTick()) {
        VideoCommon::V2::Versions::HookEpoch(VideoCommon::V2::Versions::EpochReason::Submit);
    }
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    if (VideoCommon::V2::P5bStats::Enabled()) [[unlikely]] {
        VideoCommon::V2::P5bStats::NoteSubmitSite(location.file_name(), location.line());
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Flush");
    }
    if (lazy_fences) [[unlikely]] {
        NoteBlockedWait();
    }
    // When flushing, we only send data to the worker thread; no waiting is necessary.
    const u64 signal_value = SubmitExecution(signal_semaphore, wait_semaphore);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    submit_wanted = false;
    return signal_value;
}

void Scheduler::NoteBlockedWait() {
    if (blocked_since_us.load(std::memory_order_relaxed) == 0) {
        return;
    }
    const u64 since = blocked_since_us.exchange(0, std::memory_order_relaxed);
    if (since != 0 && submit_requests.Urgent(CurrentTick())) {
        blocked_wait_us.fetch_add(NowUs() - since, std::memory_order_relaxed);
    }
}

bool Scheduler::LazyFencesHere() const {
    return lazy_fences && VideoCommon::GpuSplit::OnBackThread();
}

void Scheduler::ServeIdleHook() {
    if (!lazy_fences) {
        FlushDeferred();
        return;
    }
    // F4-02: the back is going idle (nothing else will submit) or a waiter asked. Replaces
    // DeferFlush's own submit for lazy fences and command list ends.
    if (submit_requests.IdleSubmits(flush_pending, lazy_fences, submit_wanted,
                                    lazy_outstanding_tick, CurrentTick())) {
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::FlushLazy);
        Flush();
    }
}

void Scheduler::ServeSplitIdleHook(bool (*sync_due)(void*), void* argument) {
    if (!lazy_fences) {
        FlushDeferred();
        return;
    }
    namespace P5b = VideoCommon::V2::P5bStats;
    using IdleAction = VideoCommon::V2::SubmitRequests::IdleAction;
    // F4-X2 (MASTER_PLAN §5.5, DXVK): DeferFlush's work and an urgent request (a thread blocks on
    // the open tick: a landing wait, a front list read, a readback, a fence wait past its
    // escalation) submit now, whether the GPU is busy or not. Everything else (lazy fences and
    // their bounded-time requests, list ends, list copies) goes out here only by DXVK's
    // synchronization rule: the GPU is about to go idle (fewer than two submissions in flight)
    // and the work is worth a submission, or the pending threshold. A wake alone (RequestIdleHook)
    // is not a reason. Work left here rides the next pass boundary or chunk; no thread waits on
    // it, and one that starts to files its own urgent request.
    const u64 open = CurrentTick();
    if (!submit_requests.IdleSubmits(flush_pending, lazy_fences, submit_wanted,
                                     lazy_outstanding_tick, open)) {
        return;
    }
    // The heuristic is read only when nothing else decides.
    const bool due = !flush_pending && !submit_requests.Urgent(open) && sync_due(argument);
    switch (submit_requests.AtIdleHook(flush_pending, lazy_fences, submit_wanted,
                                       lazy_outstanding_tick, open, due)) {
    case IdleAction::None:
        return;
    case IdleAction::Leave:
        P5b::Add(P5b::Counter::IdleLeft);
        return;
    case IdleAction::Deferred:
        P5b::Add(P5b::Counter::IdleDeferredWork);
        break;
    case IdleAction::Urgent:
        P5b::Add(P5b::Counter::IdleUrgent);
        break;
    case IdleAction::Sync:
        P5b::Add(P5b::Counter::IdleSync);
        break;
    }
    P5b::Add(P5b::Counter::FlushLazy);
    Flush();
}

void Scheduler::OnGpuProgress() {
    // Finish thread, a completion published. A front channel suspended on a command list copy
    // (DmaPusher's pending list tick) may go on: end the front's pending sleep (W-01b).
    VideoCommon::V2::FrontWake::Ring();
}

bool Scheduler::DeferFlush() {
    // Fences come one per compute release: ~11,500 a second in Hyrule Warriors: Age of
    // Imprisonment fights, each a queue submission (~22,000 a second with the others), which
    // kept the Vulkan worker busy submitting. Batched within 500 us: ~10-15% more fps there.
    // Command list copies after dispatches were ~9,000 more a second, of which the front waited
    // for about 5%.
    static const s64 interval_us = [] {
        const char* value = std::getenv("EDEN_NX2_FENCE_FLUSH_US");
        if (value != nullptr) {
            return static_cast<s64>(std::atoll(value));
        }
        return s64{500};
    }();
    if (interval_us <= 0 || !VideoCommon::GpuSplit::OnBackThread() ||
        std::chrono::steady_clock::now() - last_flush >= std::chrono::microseconds{interval_us}) {
        return false;
    }
    flush_pending = true;
    return true;
}

void Scheduler::FlushDeferred() {
    if (flush_pending) {
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::FlushDeferred);
        Flush();
        return;
    }
    if (submit_requests.Open(CurrentTick())) [[unlikely]] {
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::FlushLazy);
        Flush();
    }
}

void Scheduler::EnsureSubmitted(u64 tick) {
    if (IsSubmitted(tick)) {
        return;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    if (!split.Active() || VideoCommon::GpuSplit::OnBackThread()) {
        Flush();
        return;
    }
    RequestLazyFlush(tick);
    if (!WaitSubmitted(tick) && !IsSubmitted(tick)) {
        Flush(); // The back stopped: nothing else uses the scheduler
    }
}

bool Scheduler::WaitSubmitted(u64 tick) {
    if (IsSubmitted(tick)) {
        return true;
    }
    auto& split = VideoCommon::GpuSplit::Get();
    // F4-X2 (R2-F4-01): this thread blocks on the tick: an urgent request, which the back's next
    // idle hook submits whether the GPU is busy or not (not after the GPU's queue drained).
    if (lazy_fences) [[unlikely]] {
        RequestLazyFlush(tick, true);
    }
    split.RequestIdleHook();
    // No time limit: the back can spend seconds on one entry (a large shader's translation),
    // and a fence reported as submitted before it is would release its downloads too early.
    const auto start = std::chrono::steady_clock::now();
    auto next_log = start + std::chrono::seconds{2};
    for (u32 spins = 0; !IsSubmitted(tick); ++spins) {
        if (!split.BackRunning()) {
            return IsSubmitted(tick);
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_log) {
            next_log = now + std::chrono::seconds{2};
            LOG_WARNING(Render_Vulkan, "Waited {} s for the GPU thread to submit tick {}",
                        std::chrono::duration_cast<std::chrono::seconds>(now - start).count(),
                        tick);
            split.RequestIdleHook();
        }
        if (spins < 64) {
            std::this_thread::yield();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds{50});
        }
    }
    return true;
}

void Scheduler::Finish(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                       std::source_location location) {
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    if (VideoCommon::V2::P5bStats::Enabled()) [[unlikely]] {
        VideoCommon::V2::P5bStats::NoteSubmitSite(location.file_name(), location.line());
    }
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::Finish");
    }
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    const bool profile = VideoCommon::Nx2Profile::Enabled();
    const u64 start = profile ? VideoCommon::Nx2Profile::Now() : 0;
    if (lazy_fences) [[unlikely]] {
        NoteBlockedWait();
    }
    SubmitExecution(signal_semaphore, wait_semaphore);
    Wait(presubmit_tick);
    AllocateNewContext();
    last_flush = std::chrono::steady_clock::now();
    flush_pending = false;
    submit_wanted = false;
    if (profile) {
        g_profile_finishes.fetch_add(1, std::memory_order_relaxed);
        g_profile_finish_ns.fetch_add(VideoCommon::Nx2Profile::Now() - start,
                                      std::memory_order_relaxed);
    }
}

void Scheduler::WaitWorker() {
    DispatchWork();

    // Ensure the queue is drained.
    {
        std::unique_lock ql{queue_mutex};
        event_cv.wait(ql, [this] { return work_queue.empty(); });
    }

    // Now wait for execution to finish.
    std::scoped_lock el{execution_mutex};
    WaitSubmitThread();
}

void Scheduler::WaitSubmitThread() {
    if (submit_engine) [[unlikely]] {
        submit_engine->WaitSubmitted();
    }
}

void Scheduler::RetireAfter(u64 tick, std::function<void()>&& func) {
    if (submit_engine) {
        submit_engine->RetireAfter(tick, std::move(func));
        return;
    }
    master_semaphore->Wait(tick);
    func();
}

bool Scheduler::TryHoistTransfer(u64 track) {
    if (!transfer_queue) {
        return false;
    }
    const u64 open_tick = master_semaphore->CurrentTick();
    if (!timeline_join.CanHoist(track, open_tick)) {
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::TransferFallbacks);
        return false;
    }
    timeline_join.Hoist(track, open_tick);
    return true;
}

bool Scheduler::TryHoistBufferUpload() {
    if (!transfer_queue) {
        return false;
    }
    if (!timeline_join.CanHoistUnordered()) {
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::TransferFallbacks);
        return false;
    }
    timeline_join.HoistUnordered();
    return true;
}

void Scheduler::NoteAcquireJoin() {
    if (transfer_queue) {
        timeline_join.NoteAcquire(master_semaphore->CurrentTick());
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::AcquireJoins);
    }
}

bool Scheduler::AddTransferJob(VkCommandBuffer cmdbuf, VkSemaphore semaphore, u64 value) {
    if (!submit_engine || !transfer_queue) {
        return false;
    }
    return timeline_join.AddPostJob(reinterpret_cast<u64>(cmdbuf),
                                    reinterpret_cast<u64>(semaphore), value);
}

bool Scheduler::AddGraphicsWait(VkSemaphore semaphore, u64 value) {
    if (!submit_engine) {
        return false;
    }
    return timeline_join.AddGraphicsWait(reinterpret_cast<u64>(semaphore), value);
}

vk::CommandBuffer Scheduler::WorkerTransferCmdbuf() {
    const vk::CommandBuffer cmdbuf{current_context->Transfer(), device.GetDispatchLoader()};
    if (!current_context->xfer_begun) {
        current_context->xfer_begun = true;
        cmdbuf.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        // Perf fix C: batches of unused-region uploads wait for no graphics tick; earlier batches
        // on this queue are ordered by this barrier instead (its first scope covers them).
        static constexpr VkMemoryBarrier TRANSFER_ORDER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                               TRANSFER_ORDER);
    }
    return cmdbuf;
}

void Scheduler::MarkFrameEnd(u64 frame) {
    if (Common::DkbFrameTrace::Enabled()) {
        Record([this, frame](vk::CommandBuffer) { trace_frame_end = frame; });
    }
}

void Scheduler::DispatchWork() {
    if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
        VideoCommon::GpuSplit::CheckOwner("Scheduler::DispatchWork");
    }
    if (chunk && !chunk->Empty()) {
        // Waking the worker is a system call (~2% of the GPU thread at one dispatch per 8
        // draws): skip it while the worker is still busy, it pops the queue before waiting.
        bool wake;
        {
            std::scoped_lock ql{queue_mutex};
            work_queue.push(std::move(chunk));
            wake = worker_waiting;
        }
        if (wake) {
            event_cv.notify_all();
        }
        AcquireNewChunk();
    }
}

void Scheduler::BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                                    const VkClearValue* clear_values, u32 clear_value_count) {
    if (VideoCommon::Nx2Profile::Enabled()) {
        g_profile_renderpasses.fetch_add(1, std::memory_order_relaxed);
    }
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    if (track_barriers) [[unlikely]] {
        // Only the pending writes this pass's attachments overlap need a barrier first; the
        // read hazards its first draw noted go with it.
        for (u32 i = 0; i < framebuffer->NumImages(); ++i) {
            const auto [begin, end] = framebuffer->GuestRanges()[i];
            const auto hazard = barrier_tracker.Take(reinterpret_cast<u64>(framebuffer->Images()[i]),
                                                     begin, end);
            read_hazard.stages |= hazard.stages;
            read_hazard.access |= hazard.access;
            read_hazard.count += hazard.count;
        }
        if (read_hazard.Any()) {
            const auto hazard = std::exchange(read_hazard, {});
            RecordHazardBarrier(hazard.stages, hazard.access, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        }
        renderpass_guest_ranges = framebuffer->GuestRanges();
    }
    VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::Passes);
    ++render_pass_serial;
    state.renderpass = renderpass;
    state.framebuffer = framebuffer_handle;
    state.render_area = render_area;
    state.dynamic = next_pass_dynamic;
    auto_flush = false;

    if (GPU::Logging::IsActive() && Settings::values.gpu_log_vulkan_calls.GetValue()) {
        const std::string render_pass_info =
            fmt::format("renderArea={}x{}, numImages={}", render_area.width, render_area.height,
                        framebuffer->NumImages());
        GPU::Logging::GPULogger::GetInstance().LogRenderPassBegin(render_pass_info);
    }

    std::array<VkClearValue, 9> values{};
    for (u32 i = 0; i < clear_value_count && i < values.size(); ++i) {
        values[i] = clear_values[i];
    }
    const bool after_compute = std::exchange(compute_writes_pending, false);
    Record([renderpass, framebuffer_handle, render_area, values, clear_value_count,
            after_compute, dynamic = next_pass_dynamic](vk::CommandBuffer cmdbuf) {
        const VkRenderPassBeginInfo renderpass_bi{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .pNext = nullptr,
            .renderPass = renderpass,
            .framebuffer = framebuffer_handle,
            .renderArea =
                {
                    .offset = {.x = 0, .y = 0},
                    .extent = render_area,
                },
            .clearValueCount = clear_value_count,
            .pClearValues = clear_value_count != 0 ? values.data() : nullptr,
        };
        if (after_compute) {
            // Draws read compute output as indirect arguments, vertices and resources; nothing
            // else orders those reads after the dispatch.
            static constexpr VkMemoryBarrier COMPUTE_TO_GRAPHICS{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, COMPUTE_TO_GRAPHICS);
        }
        if (!dynamic) {
            cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
        }
    });
    num_renderpass_images = framebuffer->NumImages();
    renderpass_images = framebuffer->Images();
    renderpass_image_ranges = framebuffer->ImageRanges();
    framebuffer->MarkResolveShadowsUpToDate();
}

void Scheduler::RealizeDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    const DeferredClear dc = deferred_clear;
    deferred_clear = {};

    std::array<VkClearValue, 9> clear_values{};
    u32 count = 0;
    const RenderPassKey& base = dc.framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (base.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            continue;
        }
        clear_values[count++] = dc.color_values[slot];
    }
    if (base.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        clear_values[count++] = dc.depth_stencil_value;
    }
    const u32 color_discard_mask =
        dc.framebuffer->DiscardsMsaaColor() ? dc.color_clear_mask : 0u;
    const bool depth_stencil_discard =
        dc.depth_stencil && dc.framebuffer->DiscardsMsaaDepthStencil();
    const VkRenderPass renderpass = dc.framebuffer->RenderPassVariant(
        dc.color_clear_mask, dc.depth_stencil, color_discard_mask, depth_stencil_discard);
    EndRenderPass();
    BeginRenderPassImpl(dc.framebuffer, renderpass, clear_values.data(), count);
}

bool Scheduler::DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot,
                                const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.color_clear_mask |= 1u << rt_slot;
    deferred_clear.color_values[rt_slot] = value;
    return true;
}

bool Scheduler::DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value) {
    if (IsRenderPassActive()) {
        return false;
    }
    if (deferred_clear.framebuffer != nullptr && deferred_clear.framebuffer != framebuffer) {
        RealizeDeferredClear();
        EndRenderPass();
    }
    deferred_clear.framebuffer = framebuffer;
    deferred_clear.depth_stencil = true;
    deferred_clear.depth_stencil_value = value;
    return true;
}

void Scheduler::FlushDeferredClear() {
    if (deferred_clear.framebuffer == nullptr) {
        return;
    }
    RealizeDeferredClear();
    EndRenderPass();
}

void Scheduler::RequestRenderpass(const Framebuffer* framebuffer, std::source_location location) {
    if (deferred_clear.framebuffer == framebuffer) {
        RealizeDeferredClear();
        return;
    }
    const VkRenderPass renderpass = framebuffer->RenderPass();
    const VkFramebuffer framebuffer_handle = framebuffer->Handle();
    const VkExtent2D render_area = framebuffer->RenderArea();
    // A draw continues the active pass whenever the same framebuffer and area are bound. The
    // render-pass *variant* (a deferred clear's loadOp=CLEAR form) is only meaningful at begin -
    // loadOps are applied once, and the base/variant passes are render-pass-compatible (only
    // loadOp/storeOp differ). Comparing the exact handle here made every draw after a deferred
    // clear restart the pass and re-LOAD what was just CLEARed (~118 MB of attachment traffic per
    // world-target turn). The framebuffer handle already fixes the attachment structure and key,
    // so it is the correct identity to compare.
    if (state.renderpass && !state.dynamic && framebuffer_handle == state.framebuffer &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        return;
    }
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    // Ends any active pass and realizes a deferred clear
    EndRenderPass();
    BeginRenderPassImpl(framebuffer, renderpass, nullptr, 0);
}

void Scheduler::RequestRendering(const Framebuffer* framebuffer, u32 color_count,
                                 std::source_location location) {
    if (!dynamic_rendering || !framebuffer->DynamicRenderingCapable()) {
        if (state.renderpass && state.dynamic) {
            EndRenderPass();
        }
        RequestRenderpass(framebuffer, location);
        return;
    }
    if (read_hazard.Any() && state.renderpass) [[unlikely]] {
        // A draw of the active pass reads a pending write: end the pass for its barrier.
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::HazardBreaks);
        EndRenderPass();
    }
    if (deferred_clear.framebuffer == framebuffer) {
        // The deferred clear becomes the pass's clear load ops.
        const DeferredClear dc = std::exchange(deferred_clear, DeferredClear{});
        EndRenderPass();
        BeginRenderingImpl(framebuffer, color_count, dc.color_clear_mask, dc.color_values,
                           dc.depth_stencil,
                           dc.depth_stencil_value);
        return;
    }
    const VkExtent2D render_area = framebuffer->RenderArea();
    if (state.renderpass && state.dynamic && framebuffer->Handle() == state.framebuffer &&
        (color_count == 0 || color_count == dynamic_color_count) &&
        render_area.width == state.render_area.width &&
        render_area.height == state.render_area.height) {
        ++pass_draws;
        return;
    }
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    EndRenderPass();
    BeginRenderingImpl(framebuffer, color_count, 0, {}, false, {});
}

void Scheduler::BeginRenderingImpl(const Framebuffer* framebuffer, u32 color_count,
                                   u32 color_clear_mask,
                                   const std::array<VkClearValue, 8>& color_values,
                                   bool depth_clear, const VkClearValue& depth_value) {
    // The same bookkeeping as a render pass (tracker, serials, queries), then BeginRendering.
    next_pass_dynamic = true;
    BeginRenderPassImpl(framebuffer, framebuffer->RenderPass(), nullptr, 0);
    next_pass_dynamic = false;
    VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::DynamicPasses);
    ++pass_draws;
    if (color_clear_mask == 0 && !depth_clear) {
        // U-13: a pass without clear load ops (nearly all of them) records only the views; the
        // worker builds the attachment infos (no ~700-byte struct built and copied here).
        struct Views {
            std::array<VkImageView, 8> colors{};
            VkImageView depth{};
            VkExtent2D render_area{};
            u32 layers{};
            u32 num_colors{};
            bool has_depth{};
            bool has_stencil{};
        } compact;
        const auto& color_views = framebuffer->ColorViews();
        const RenderPassKey& base_key = framebuffer->RenderPassKeyBase();
        for (u32 slot = 0; slot < 8; ++slot) {
            if (base_key.color_formats[slot] != VideoCore::Surface::PixelFormat::Invalid) {
                compact.colors[slot] = color_views[slot];
                compact.num_colors = slot + 1;
            }
        }
        if (color_count != 0) {
            compact.num_colors = (std::min)(color_count, 8U);
        }
        dynamic_color_count = compact.num_colors;
        if (base_key.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
            compact.depth = framebuffer->DepthView();
            compact.has_depth = framebuffer->HasAspectDepthBit();
            compact.has_stencil = framebuffer->HasAspectStencilBit();
        }
        compact.render_area = framebuffer->RenderArea();
        compact.layers = framebuffer->Layers();
        Record([compact](vk::CommandBuffer cmdbuf) {
            const auto info = [](VkImageView view) {
                return VkRenderingAttachmentInfo{
                    .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .pNext = nullptr,
                    .imageView = view,
                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .resolveMode = VK_RESOLVE_MODE_NONE,
                    .resolveImageView = VK_NULL_HANDLE,
                    .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                    .clearValue = {},
                };
            };
            std::array<VkRenderingAttachmentInfo, 8> colors;
            for (u32 slot = 0; slot < compact.num_colors; ++slot) {
                colors[slot] = info(compact.colors[slot]);
            }
            const VkRenderingAttachmentInfo depth = info(compact.depth);
            const VkRenderingInfo rendering_info{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .pNext = nullptr,
                .flags = 0,
                .renderArea = {.offset = {0, 0}, .extent = compact.render_area},
                .layerCount = compact.layers,
                .viewMask = 0,
                .colorAttachmentCount = compact.num_colors,
                .pColorAttachments = colors.data(),
                .pDepthAttachment = compact.has_depth ? &depth : nullptr,
                .pStencilAttachment = compact.has_stencil ? &depth : nullptr,
            };
            cmdbuf.BeginRendering(rendering_info);
        });
        return;
    }
    struct Attachments {
        std::array<VkRenderingAttachmentInfo, 8> colors{};
        u32 num_colors = 0;
        VkRenderingAttachmentInfo depth{};
        bool has_depth = false;
        bool has_stencil = false;
    } attachments;
    const auto& views = framebuffer->ColorViews();
    const RenderPassKey& key = framebuffer->RenderPassKeyBase();
    for (u32 slot = 0; slot < 8; ++slot) {
        if (key.color_formats[slot] == VideoCore::Surface::PixelFormat::Invalid) {
            attachments.colors[slot] = VkRenderingAttachmentInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = VK_NULL_HANDLE,
            };
            continue;
        }
        const bool clear = (color_clear_mask & (1u << slot)) != 0;
        attachments.colors[slot] = VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = views[slot],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = color_values[slot],
        };
        attachments.num_colors = slot + 1;
    }
    // The bound pipeline's count (its blend attachments) decides, unused slots are null views.
    if (color_count != 0) {
        attachments.num_colors = (std::min)(color_count, 8U);
    }
    dynamic_color_count = attachments.num_colors;
    if (key.depth_format != VideoCore::Surface::PixelFormat::Invalid) {
        attachments.depth = VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = framebuffer->DepthView(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = depth_clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = depth_value,
        };
        attachments.has_depth = framebuffer->HasAspectDepthBit();
        attachments.has_stencil = framebuffer->HasAspectStencilBit();
    }
    const VkExtent2D render_area = framebuffer->RenderArea();
    const u32 layers = framebuffer->Layers();
    Record([attachments, render_area, layers](vk::CommandBuffer cmdbuf) {
        const VkRenderingInfo rendering_info{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderArea = {.offset = {0, 0}, .extent = render_area},
            .layerCount = layers,
            .viewMask = 0,
            .colorAttachmentCount = attachments.num_colors,
            .pColorAttachments = attachments.colors.data(),
            .pDepthAttachment = attachments.has_depth ? &attachments.depth : nullptr,
            .pStencilAttachment = attachments.has_stencil ? &attachments.depth : nullptr,
        };
        cmdbuf.BeginRendering(rendering_info);
    });
}

void Scheduler::NoteImageReadTracked(VkImage image, u64 begin, u64 end) {
    const auto hazard = barrier_tracker.Take(reinterpret_cast<u64>(image), begin, end);
    if (hazard.Any()) {
        read_hazard.stages |= hazard.stages;
        read_hazard.access |= hazard.access;
        read_hazard.count += hazard.count;
        UpdateAutoFlush();
    }
}

void Scheduler::FlushReadHazards() {
    if (!read_hazard.Any()) {
        return;
    }
    if (state.renderpass) {
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::HazardBreaks);
        EndRenderPass();
    }
    const auto hazard = std::exchange(read_hazard, {});
    RecordHazardBarrier(hazard.stages, hazard.access,
                        VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

void Scheduler::FlushAllPendingWrites(std::source_location location) {
    const auto hazard = barrier_tracker.TakeAll();
    auto_flush = false;
    const auto reads = std::exchange(read_hazard, {});
    const u32 stages = hazard.stages | reads.stages;
    if (stages == 0) {
        return; // Nothing pending: no barrier (Fix pass 1: this was still counted as one)
    }
    VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::FullBarriers);
    if (VideoCommon::V2::P5bStats::Enabled()) [[unlikely]] {
        VideoCommon::V2::P5bStats::NoteFullBarrierSite(location.file_name(), location.line());
    }
    RecordHazardBarrierUncounted(stages, hazard.access | reads.access,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

void Scheduler::RecordHazardBarrier(u32 src_stages, u32 src_access, u32 dst_stages,
                                    u32 dst_access) {
    // Recorded straight into the chunk: Record's auto flush must not run again for it.
    if (src_stages == 0) {
        return;
    }
    VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::TargetedBarriers);
    RecordHazardBarrierUncounted(src_stages, src_access, dst_stages, dst_access);
}

void Scheduler::RecordHazardBarrierUncounted(u32 src_stages, u32 src_access, u32 dst_stages,
                                             u32 dst_access) {
    const bool saved = std::exchange(auto_flush, false);
    Record([src_stages, src_access, dst_stages, dst_access](vk::CommandBuffer cmdbuf) {
        const VkMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = src_access,
            .dstAccessMask = dst_access,
        };
        cmdbuf.PipelineBarrier(src_stages, dst_stages, 0, barrier);
    });
    auto_flush = saved && !barrier_tracker.Empty();
}

void Scheduler::RequestOutsideRenderPassOperationContext(std::source_location location) {
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    EndRenderPass();
    if (track_barriers && (!barrier_tracker.Empty() || read_hazard.Any())) [[unlikely]] {
        // An operation the tracker doesn't follow: every pending write before it, as the old
        // pass-end barrier gave it (one barrier for every pass that ended since).
        FlushAllPendingWrites(location);
    }
}

bool Scheduler::RequestOutsideRenderPassForImages(std::initializer_list<TrackedImage> images,
                                                  std::source_location location) {
    if (!track_barriers) {
        RequestOutsideRenderPassOperationContext(location);
        return false;
    }
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    EndRenderPass();
    // Reads noted for draws that never came keep their hazards (the next pass begin takes them).
    VideoCommon::V2::RangeBarrierTracker::Hazard hazard{};
    for (const TrackedImage& image : images) {
        if (image.image == VK_NULL_HANDLE) {
            continue;
        }
        const auto taken =
            barrier_tracker.Take(reinterpret_cast<u64>(image.image), image.begin, image.end);
        hazard.stages |= taken.stages;
        hazard.access |= taken.access;
        hazard.count += taken.count;
    }
    const bool saved = std::exchange(targeted_scope, true);
    UpdateAutoFlush();
    RecordHazardBarrier(hazard.stages, hazard.access,
                        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT |
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    return saved;
}

bool Scheduler::BeginBufferOperation(std::source_location location) {
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    EndRenderPass();
    const bool saved = std::exchange(targeted_scope, true);
    UpdateAutoFlush();
    return saved;
}

void Scheduler::RequestOutsideRenderPassForDispatch(std::source_location location) {
    if (!track_barriers) {
        RequestOutsideRenderPassOperationContext(location);
        return;
    }
    if (state.renderpass) {
        CountRenderPassEnd(location.file_name(), location.line());
    }
    EndRenderPass();
    // The dispatch's own read barrier (graphics, compute and transfer writes -> compute reads)
    // covers its reads of pending attachment writes; they stay pending for graphics readers.
    read_hazard = {};
}

void Scheduler::WaitForIdleTracked(u32 stages, std::source_location location) {
    // F4-X3: the old WFI ended the pass, flushed every pending write with a full barrier and
    // recorded its own barrier after it (~90 full barriers and ~70 pass ends a frame at Castle
    // Town). Now one barrier, and the pass goes on while its draws wrote only attachments.
    using VideoCommon::V2::WfiAction;
    namespace P5b = VideoCommon::V2::P5bStats;
    switch (VideoCommon::V2::DecideWaitForIdle(state.renderpass != VK_NULL_HANDLE,
                                                render_pass_has_stores,
                                                pass_transform_feedback)) {
    case WfiAction::AtPassEnd:
        // Draws before it wrote only attachments, which the pass's later draws see in
        // rasterization order unless they sample them; work recorded before the pass is ordered
        // for its draws by that work's own barriers (compute: the pass begin's). A later draw that
        // samples a render target or stores ends the pass (the fragment barrier rule); the pass
        // end records the barrier.
        wfi_at_pass_end |= stages;
        fragment_barrier.Defer(true);
        P5b::Add(P5b::Counter::WfiDeferred);
        return;
    case WfiAction::EndPass:
        CountRenderPassEnd(location.file_name(), location.line());
        P5b::Add(P5b::Counter::WfiPassEnds);
        wfi_at_pass_end |= stages;
        EndRenderPass();
        return;
    case WfiAction::Barrier:
        break;
    }
    // Realizes a deferred clear as the old pass end did (a deferred WFI is recorded with it).
    EndRenderPass();
    RecordWaitForIdleBarrier(stages);
}

void Scheduler::RecordWaitForIdleBarrier(u32 stages) {
    // The global barrier (MASTER_PLAN §5.5: the WFI is the only one): every pending write the
    // tracker holds folds into it, so later readers find nothing pending, as after the old
    // full flush, without a second barrier.
    const auto pending = barrier_tracker.TakeAll();
    const auto reads = std::exchange(read_hazard, {});
    VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::WfiBarriers);
    RecordHazardBarrierUncounted(stages | pending.stages | reads.stages,
                                 VK_ACCESS_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

bool Scheduler::UpdateGraphicsPipeline(GraphicsPipeline* pipeline) {
    if (state.graphics_pipeline == pipeline) {
        if (pipeline && pipeline->UsesExtendedDynamicState() &&
            state.needs_state_enable_refresh) {
            state_tracker.InvalidateStateEnableFlag();
            state.needs_state_enable_refresh = false;
        }
        return false;
    }

    state.graphics_pipeline = pipeline;

    if (!pipeline) {
        return true;
    }

    if (!pipeline->UsesExtendedDynamicState()) {
        state.needs_state_enable_refresh = true;
    } else if (state.needs_state_enable_refresh) {
        state_tracker.InvalidateStateEnableFlag();
        state.needs_state_enable_refresh = false;
    }

    return true;
}

bool Scheduler::UpdateRescaling(bool is_rescaling, bool mesh_layout) {
    if (state.rescaling_defined && is_rescaling == state.is_rescaling &&
        mesh_layout == state.mesh_layout) {
        return false;
    }
    state.rescaling_defined = true;
    state.is_rescaling = is_rescaling;
    state.mesh_layout = mesh_layout;
    return true;
}

bool Scheduler::UpdateDescriptorBufferChunk(u32 descriptor_chunk) {
    if (state.descriptor_buffer_bound && descriptor_chunk == state.descriptor_buffer_chunk) {
        return false;
    }
    state.descriptor_buffer_bound = true;
    state.descriptor_buffer_chunk = descriptor_chunk;
    return true;
}

void Scheduler::WorkerThread(std::stop_token stop_token) {
    Common::SetCurrentThreadName("VulkanWorker");
    const Common::DkbCpuTrace::Registration trace_thread{Common::DkbCpuTrace::Role::VulkanWorker};
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    Common::SetCurrentThreadToPerformanceCores();

    const auto TryPopQueue{[this](auto& work) -> bool {
        if (work_queue.empty()) {
            return false;
        }

        work = std::move(work_queue.front());
        work_queue.pop();
        event_cv.notify_all();
        return true;
    }};

    while (!stop_token.stop_requested()) {
        std::unique_ptr<CommandChunk> work;

        {
            std::unique_lock lk{queue_mutex};

            // Wait for work.
            event_cv.wait(lk, stop_token, [&] {
                worker_waiting = !TryPopQueue(work);
                return !worker_waiting;
            });
            worker_waiting = false;

            // If we've been asked to stop, we're done.
            if (stop_token.stop_requested()) {
                return;
            }

            // Exchange lock ownership so that we take the execution lock before
            // the queue lock goes out of scope. This allows us to force execution
            // to complete in the next step.
            void(std::exchange(lk, std::unique_lock{execution_mutex}));

            // Perform the work, tracking whether the chunk was a submission
            // before executing.
            const bool has_submit = work->HasSubmit();
            work->ExecuteAll(current_cmdbuf, current_upload_cmdbuf);

            // If the chunk was a submission, reallocate the command buffer.
            if (has_submit) {
                AllocateWorkerCommandBuffer();
            }
        }

        {
            std::scoped_lock rl{reserve_mutex};

            // Recycle the chunk back to the reserve.
            chunk_reserve.emplace_back(std::move(work));
        }
    }
}

void Scheduler::AllocateWorkerCommandBuffer() {
    if (submit_engine) {
        current_context = submit_engine->Acquire();
        current_cmdbuf = vk::CommandBuffer(current_context->Main(), device.GetDispatchLoader());
        current_upload_cmdbuf =
            vk::CommandBuffer(current_context->Upload(), device.GetDispatchLoader());
        current_cmdbuf.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        current_upload_cmdbuf.Begin({
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });
        if (trace_query_pool && !trace_gpu_failed) {
            TraceGpuSubmission();
        }
        return;
    }
    current_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    current_upload_cmdbuf = vk::CommandBuffer(command_pool->Commit(), device.GetDispatchLoader());
    current_upload_cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    if (trace_query_pool && !trace_gpu_failed) {
        TraceGpuSubmission();
    }
}

void Scheduler::TraceShader(u64 hash, u32 stage) {
    if (hash == 0) {
        return;
    }
    for (u32 i = 0; i < trace_work.num_shaders; ++i) {
        if (trace_work.shaders[i].hash == hash && trace_work.shaders[i].stage == stage) {
            return;
        }
    }
    if (trace_work.num_shaders == trace_work.shaders.size()) {
        ++trace_work.untracked_uses;
        return;
    }
    trace_work.shaders[trace_work.num_shaders++] = {hash, stage};
}

void Scheduler::TraceGraphicsPipeline(const std::array<u64, 6>& hashes) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hashes](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.draw_attempts;
        for (u32 stage = 0; stage < hashes.size(); ++stage) {
            TraceShader(hashes[stage], stage);
        }
    });
}

void Scheduler::TraceComputePipeline(u64 hash) {
    if (!trace_query_pool) {
        return;
    }
    Record([this, hash](vk::CommandBuffer) {
        if (!trace_gpu_recording) {
            return;
        }
        ++trace_work.dispatch_attempts;
        TraceShader(hash, 6);
    });
}

void Scheduler::ReportWorkTimes() {
    const u32 count = trace_work_count;
    trace_work_count = 0;
    if (count == 0) {
        return;
    }
    struct Result {
        u64 timestamp;
        u64 available;
    };
    std::vector<Result> results(count * 2);
    const auto result = device.GetLogical().GetQueryResults(
        *trace_work_pool, 0, count * 2, results.size() * sizeof(Result), results.data(),
        sizeof(Result), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (result != VK_SUCCESS) {
        return;
    }
    struct Totals {
        double total_ms{};
        double max_ms{};
        u32 count{};
        std::array<u32, 3> max_size{};
    };
    std::map<std::pair<u64, u64>, Totals> by_shaders;
    double total_ms{};
    for (u32 i = 0; i < count; ++i) {
        if (results[i * 2].available == 0 || results[i * 2 + 1].available == 0) {
            continue;
        }
        const double ms = static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                              results[i * 2].timestamp, results[i * 2 + 1].timestamp,
                              trace_timestamp_bits)) *
                          trace_timestamp_period / 1e6;
        const WorkTrace& item = trace_work_items[i];
        Totals& totals = by_shaders[{item.hash, item.fragment_hash}];
        totals.total_ms += ms;
        ++totals.count;
        if (ms > totals.max_ms) {
            totals.max_ms = ms;
            totals.max_size = item.size;
        }
        total_ms += ms;
    }
    std::vector<std::pair<std::pair<u64, u64>, Totals>> sorted(by_shaders.begin(),
                                                               by_shaders.end());
    std::ranges::sort(sorted, std::greater<>{},
                      [](const auto& entry) { return entry.second.total_ms; });
    if (sorted.size() > 8) {
        sorted.resize(8);
    }
    std::string slowest;
    for (const auto& [shaders, totals] : sorted) {
        const auto& [hash, fragment_hash] = shaders;
        slowest += fragment_hash == 0
                       ? fmt::format(" s6:{:016x}", hash)
                       : fmt::format(" s1:{:016x}+s5:{:016x}", hash, fragment_hash);
        slowest += fmt::format("={:.3f}ms/{}x(max {:.3f}ms size {}x{}x{})", totals.total_ms,
                               totals.count, totals.max_ms, totals.max_size[0],
                               totals.max_size[1], totals.max_size[2]);
    }
    LOG_INFO(Render_Vulkan,
             "DKB gpu_work renderer_frame={} timed={} timed_sum_ms={:.3f} slowest:{}",
             trace_sample_frame, count, total_ms, slowest);
}

void Scheduler::TraceGpuSubmission() {
    if (trace_gpu_pending) {
        master_semaphore->Refresh();
        if (!master_semaphore->IsFree(trace_gpu_tick)) {
            return;
        }
        struct Result {
            u64 timestamp;
            u64 available;
        };
        std::array<Result, TraceMaxSubmissions * 2> results{};
        const auto result = device.GetLogical().GetQueryResults(
            *trace_query_pool, 0, trace_gpu_count * 2, trace_gpu_count * 2 * sizeof(Result),
            results.data(), sizeof(Result),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (result == VK_NOT_READY) {
            return;
        }
        if (result != VK_SUCCESS) {
            LOG_WARNING(Render_Vulkan, "DKB gpu_timestamp disabled: query result {}",
                        static_cast<int>(result));
            trace_gpu_failed = true;
            return;
        }
        double total_ms{};
        std::array<std::pair<double, u32>, 4> slowest{};
        for (u32 i = 0; i < trace_gpu_count * 2; i += 2) {
            if (results[i].available == 0 || results[i + 1].available == 0) {
                return;
            }
            const auto ticks = Common::DkbFrameTrace::TimestampDelta(
                results[i].timestamp, results[i + 1].timestamp, trace_timestamp_bits);
            const double duration_ms = static_cast<double>(ticks) * trace_timestamp_period / 1e6;
            total_ms += duration_ms;
            if (duration_ms >= 10.0 && duration_ms > slowest.back().first) {
                slowest.back() = {duration_ms, i / 2};
                std::ranges::sort(slowest, std::greater<>{});
            }
        }
        const double span_ms = trace_gpu_skipped == 0
            ? static_cast<double>(Common::DkbFrameTrace::TimestampDelta(
                  results[0].timestamp, results[trace_gpu_count * 2 - 1].timestamp,
                  trace_timestamp_bits)) * trace_timestamp_period / 1e6
            : -1.0;
        LOG_INFO(Render_Vulkan,
                 "DKB gpu_frame renderer_frame={} last_submission_tick={} "
                 "gpu_frame_span_ms={:.3f} gpu_submission_sum_ms={:.3f} "
                 "measured_submissions={} dropped_submissions={} "
                 "valid_bits={} period_ns={}",
                 trace_sample_frame, trace_gpu_tick, span_ms, total_ms, trace_gpu_count,
                 trace_gpu_skipped, trace_timestamp_bits, trace_timestamp_period);
        for (const auto& [duration_ms, index] : slowest) {
            if (duration_ms < 10.0) {
                continue;
            }
            const auto& work = trace_submissions[index];
            std::string shaders;
            for (u32 i = 0; i < work.num_shaders; ++i) {
                const auto& shader = work.shaders[i];
                shaders += fmt::format("{}s{}:{:016x}", shaders.empty() ? "" : ",",
                                       shader.stage, shader.hash);
            }
            LOG_INFO(Render_Vulkan,
                     "DKB gpu_submission renderer_frame={} tick={} gpu_ms={:.3f} "
                     "draw_attempts={} dispatch_attempts={} shaders={} untracked_uses={}",
                     trace_sample_frame, work.tick, duration_ms, work.draw_attempts,
                     work.dispatch_attempts, shaders, work.untracked_uses);
        }
        ReportWorkTimes();
        trace_gpu_tick = 0;
        trace_gpu_count = 0;
        trace_gpu_skipped = 0;
        trace_gpu_pending = false;
        trace_gpu_rate.next = std::chrono::steady_clock::now() + std::chrono::seconds{1};
    }
    if (!trace_gpu_sampling || trace_gpu_count == TraceMaxSubmissions) {
        return;
    }
    current_upload_cmdbuf.ResetQueryPool(*trace_query_pool, trace_gpu_count * 2, 2);
    if (trace_gpu_count == 0 && trace_work_pool) {
        // Draws are timed inside render passes, where queries cannot be reset.
        current_upload_cmdbuf.ResetQueryPool(*trace_work_pool, 0, TraceMaxWork * 2);
    }
    current_upload_cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, *trace_query_pool,
                                        trace_gpu_count * 2);
    trace_gpu_recording = true;
}

u64 Scheduler::SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore) {
    if (VideoCommon::Nx2Profile::Enabled()) {
        g_profile_submits.fetch_add(1, std::memory_order_relaxed);
        ReportSubmitCounts();
    }
    if (pre_submit) [[unlikely]] {
        pre_submit();
    }
    EndPendingOperations();
    InvalidateState();
    if (host_read_wanted) [[unlikely]] {
        // R2-F4-07 (MASTER_PLAN §6 class C): a mapped buffer was written in this submission;
        // its end makes the writes host-visible, so a reader waits for this tick only
        // (HostVisibleThrough) instead of a Finish of whatever is open then.
        host_read_wanted = false;
        host_visible_tick = master_semaphore->CurrentTick();
        const bool saved = std::exchange(auto_flush, false);
        Record([](vk::CommandBuffer cmdbuf) {
            static constexpr VkMemoryBarrier HOST_READ_BARRIER{
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
            };
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                   0, HOST_READ_BARRIER);
        });
        auto_flush = saved;
    }

    const u64 signal_value = master_semaphore->NextTick();
    if (VideoCommon::V2::P5bStats::Enabled()) {
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::Submits);
    }
    if (submit_engine) [[unlikely]] {
        SubmitExecutionEngine(signal_semaphore, wait_semaphore, signal_value);
        return signal_value;
    }
    auto_flush = false; // The submit closure touches no image: pending writes stay pending
    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value,
                            this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.End();
        if (trace_gpu_recording) {
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_query_pool,
                                 trace_gpu_count * 2 + 1);
            trace_gpu_tick = signal_value;
            trace_work.tick = signal_value;
            trace_submissions[trace_gpu_count] = trace_work;
            ++trace_gpu_count;
            trace_gpu_recording = false;
        } else if (trace_gpu_sampling) {
            ++trace_gpu_skipped;
        }
        if (trace_query_pool) {
            trace_work = {};
        }
        cmdbuf.End();

        if (on_submit) {
            on_submit();
        }

        std::scoped_lock lock{submit_mutex};
        switch (const VkResult result = master_semaphore->SubmitQueue(
                    cmdbuf, upload_cmdbuf, signal_semaphore, wait_semaphore, signal_value)) {
        case VK_SUCCESS:
            // Log successful queue submission
            if (GPU::Logging::IsActive() &&
                Settings::values.gpu_log_vulkan_calls.GetValue()) {
                GPU::Logging::GPULogger::GetInstance().LogVulkanCall(
                    "vkQueueSubmit", "", VK_SUCCESS);
            }
            break;
        case VK_ERROR_DEVICE_LOST:
            device.ReportLoss();
            [[fallthrough]];
        default:
            vk::Check(result);
            break;
        }
        if (trace_frame_end != 0) {
            if (trace_gpu_sampling) {
                trace_gpu_pending = true;
                trace_gpu_sampling = false;
            }
            if (trace_query_pool && !trace_gpu_failed && !trace_gpu_pending &&
                trace_gpu_rate.Ready(std::chrono::steady_clock::now())) {
                trace_sample_frame = trace_frame_end + 1;
                trace_gpu_sampling = true;
            }
            trace_frame_end = 0;
        }
    });
    chunk->MarkSubmit();
    DispatchWork();
    UpdateAutoFlush();
    return signal_value;
}

void Scheduler::SubmitExecutionEngine(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                                      u64 signal_value) {
    // P5b: as the closure above, but the submission (with the transfer batch and the joins the
    // recording thread decided) goes to the submit engine: the submit thread, or inline.
    const VideoCommon::V2::TimelineJoin::Batch batch = timeline_join.Close(signal_value);
    auto_flush = false; // The submit closure touches no image: pending writes stay pending
    RecordWithUploadBuffer([signal_semaphore, wait_semaphore, signal_value, batch,
                            this](vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) {
        static constexpr VkMemoryBarrier WRITE_BARRIER{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        };
        upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, WRITE_BARRIER);
        upload_cmdbuf.End();
        if (trace_gpu_recording) {
            cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_query_pool,
                                  trace_gpu_count * 2 + 1);
            trace_gpu_tick = signal_value;
            trace_work.tick = signal_value;
            trace_submissions[trace_gpu_count] = trace_work;
            ++trace_gpu_count;
            trace_gpu_recording = false;
        } else if (trace_gpu_sampling) {
            ++trace_gpu_skipped;
        }
        if (trace_query_pool) {
            trace_work = {};
        }
        cmdbuf.End();
        const bool transfer = batch.transfer && current_context->xfer_begun;
        if (current_context->xfer_begun) {
            WorkerTransferCmdbuf().End();
        }
        if (on_submit) {
            on_submit();
        }
        SubmitWork work{};
        work.context = current_context;
        work.signal_semaphore = signal_semaphore;
        work.wait_semaphore = wait_semaphore;
        work.tick = signal_value;
        work.transfer = transfer;
        work.transfer_wait = batch.transfer_wait;
        work.graphics_waits_transfer = transfer && batch.graphics_waits_transfer;
        work.transfer_images = batch.transfer_images;
        work.num_extra_waits = batch.num_extra_waits;
        for (u32 i = 0; i < batch.num_extra_waits; ++i) {
            work.extra_waits[i] = {reinterpret_cast<VkSemaphore>(batch.extra_waits[i].semaphore),
                                   batch.extra_waits[i].value};
        }
        work.num_post_jobs = (std::min<u32>)(batch.num_post_jobs,
                                             static_cast<u32>(work.post_jobs.size()));
        for (u32 i = 0; i < work.num_post_jobs; ++i) {
            work.post_jobs[i] = {
                reinterpret_cast<VkCommandBuffer>(batch.post_jobs[i].cmdbuf),
                reinterpret_cast<VkSemaphore>(batch.post_jobs[i].semaphore),
                batch.post_jobs[i].value,
            };
        }
        submit_engine->Submit(std::move(work));
        if (trace_frame_end != 0) {
            if (trace_gpu_sampling) {
                trace_gpu_pending = true;
                trace_gpu_sampling = false;
            }
            if (trace_query_pool && !trace_gpu_failed && !trace_gpu_pending &&
                trace_gpu_rate.Ready(std::chrono::steady_clock::now())) {
                trace_sample_frame = trace_frame_end + 1;
                trace_gpu_sampling = true;
            }
            trace_frame_end = 0;
        }
    });
    chunk->MarkSubmit();
    DispatchWork();
    UpdateAutoFlush();
}

void Scheduler::AllocateNewContext() {
    // Enable counters once again. These are disabled when a command buffer is finished.
}

void Scheduler::InvalidateState() {
    ++recording_epoch;
    state.graphics_pipeline = nullptr;
    state.rescaling_defined = false;
    state.descriptor_buffer_bound = false;
    state_tracker.InvalidateCommandBufferState();
}

void Scheduler::EndPendingOperations() {
    // Pause, don't reset: a submission can land between a guest's counter reset and its report,
    // and resetting here would drop the samples counted so far (the report then reads too low).
    query_cache->CounterClose(VideoCommon::QueryType::ZPassPixelCount64);
    EndRenderPass();
}

void Scheduler::EndRenderPass()
    {
        RealizeDeferredClear();
        if (!state.renderpass) {
            return;
        }
        ++recording_epoch;
        fragment_barrier.Reset();
        render_pass_has_stores = false;
        ++render_pass_serial;

        query_cache->CounterClose(VideoCommon::QueryType::StreamingByteCount);

        // Log render pass end
        if (GPU::Logging::IsActive() &&
            Settings::values.gpu_log_vulkan_calls.GetValue()) {
            GPU::Logging::GPULogger::GetInstance().LogRenderPassEnd();
        }

        query_cache->CounterEnable(VideoCommon::QueryType::ZPassPixelCount64, false);
        query_cache->NotifySegment(false);
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::PassDraws,
                                       std::exchange(pass_draws, 0));

        if (track_barriers) [[unlikely]] {
            // No barrier now: the attachments' writes stay pending, keyed by image and guest
            // range, until something overlapping reads or writes them (or an untracked
            // operation flushes them all).
            for (u32 i = 0; i < num_renderpass_images; ++i) {
                const bool is_color =
                    (renderpass_image_ranges[i].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const u32 stages = is_color ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                            : VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                const u32 access = is_color ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                            : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                barrier_tracker.NoteWrite(reinterpret_cast<u64>(renderpass_images[i]),
                                          renderpass_guest_ranges[i].first,
                                          renderpass_guest_ranges[i].second,
                                          stages | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                          access | VK_ACCESS_SHADER_WRITE_BIT);
            }
            const bool dynamic = state.dynamic;
            // F4-07: the XFB barrier only after a pass that ran transform feedback.
            const bool has_transform_feedback = device.IsExtTransformFeedbackSupported() &&
                                                std::exchange(pass_transform_feedback, false);
            Record([dynamic, has_transform_feedback](vk::CommandBuffer cmdbuf) {
                if (dynamic) {
                    cmdbuf.EndRendering();
                } else {
                    cmdbuf.EndRenderPass();
                }
                if (has_transform_feedback) {
                    static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                        .dstAccessMask =
                            VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                    };
                    cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                           VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                               VK_PIPELINE_STAGE_TRANSFER_BIT,
                                           0, XFB_OUTPUT_BARRIER);
                }
            });
            VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::SkippedBarriers);
            state.renderpass = VkRenderPass{};
            state.dynamic = false;
            num_renderpass_images = 0;
            UpdateAutoFlush();
            if (wfi_at_pass_end != 0) [[unlikely]] {
                // F4-X3: a wait-for-idle recorded inside the pass.
                RecordWaitForIdleBarrier(std::exchange(wfi_at_pass_end, 0));
            }
            return;
        }
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::PassEndBarriers);

        Record([num_images = num_renderpass_images,
                       images = renderpass_images,
                       ranges = renderpass_image_ranges,
                       has_transform_feedback = device.IsExtTransformFeedbackSupported()](
                          vk::CommandBuffer cmdbuf) {
            std::array<VkImageMemoryBarrier, 9> barriers;
            for (size_t i = 0; i < num_images; ++i) {
                const VkImageSubresourceRange& range = ranges[i];
                const bool is_color = (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0;
                const bool is_depth_stencil = (range.aspectMask
                                              & (VK_IMAGE_ASPECT_DEPTH_BIT
                                                 | VK_IMAGE_ASPECT_STENCIL_BIT)) !=0;

                VkAccessFlags src_access = 0;

                if (is_color)
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                else if (is_depth_stencil)
                    src_access |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                else
                    src_access |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                  | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

                barriers[i] = VkImageMemoryBarrier{
                        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        .pNext = nullptr,
                        .srcAccessMask = src_access,
                        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                         | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                        .image = images[i],
                        .subresourceRange = range,
                };
            }
            cmdbuf.EndRenderPass();
            cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, vk::PIPELINE_STAGE_GRAPHICS_COMPUTE,
                                   0, nullptr, nullptr, vk::Span(barriers.data(), num_images));
            if (has_transform_feedback) {
                static constexpr VkMemoryBarrier XFB_OUTPUT_BARRIER{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = nullptr,
                    .srcAccessMask = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
                    .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT,
                };
                cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
                                       VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       0, XFB_OUTPUT_BARRIER);
            }
        });

        state.renderpass = VkRenderPass{};
        num_renderpass_images = 0;
    }


void Scheduler::AcquireNewChunk() {
    std::scoped_lock rl{reserve_mutex};

    if (chunk_reserve.empty()) {
        // If we don't have anything reserved, we need to make a new chunk.
        chunk = std::make_unique<CommandChunk>();
    } else {
        // Otherwise, we can just take from the reserve.
        chunk = std::move(chunk_reserve.back());
        chunk_reserve.pop_back();
    }
}

} // namespace Vulkan
