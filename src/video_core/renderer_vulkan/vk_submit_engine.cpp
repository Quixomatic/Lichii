// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <utility>

#include "common/logging.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_submit_engine.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

namespace {
namespace P5bStats = VideoCommon::V2::P5bStats;
using P5bStats::Counter;

/// Contexts in flight before the worker waits for one to come back (a deep queue already).
constexpr size_t MaxContexts = 64;

u64 MicrosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}
} // Anonymous namespace

TransferQueue::TransferQueue(const Device& device_)
    : device{device_}, family{device_.GetTransferFamily()}, queue{device_.GetTransferQueue()} {
    timeline = CreateTimeline();
}

TransferQueue::~TransferQueue() = default;

vk::Semaphore TransferQueue::CreateTimeline() const {
    static constexpr VkSemaphoreTypeCreateInfo semaphore_type_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .pNext = nullptr,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    static constexpr VkSemaphoreCreateInfo semaphore_ci{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &semaphore_type_ci,
        .flags = 0,
    };
    return device.GetLogical().CreateSemaphore(semaphore_ci);
}

VkResult TransferQueue::Submit(VkCommandBuffer cmdbuf, VkSemaphore wait_semaphore, u64 wait_value,
                               VkSemaphore signal_semaphore, u64 signal_value) {
    const VkCommandBufferSubmitInfo cmdbuf_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .pNext = nullptr,
        .commandBuffer = cmdbuf,
        .deviceMask = 0,
    };
    const VkSemaphoreSubmitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = wait_semaphore,
        .value = wait_value,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0,
    };
    const VkSemaphoreSubmitInfo signal_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .pNext = nullptr,
        .semaphore = signal_semaphore,
        .value = signal_value,
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .deviceIndex = 0,
    };
    const bool waits = wait_semaphore != VK_NULL_HANDLE && wait_value != 0;
    const VkSubmitInfo2 submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .pNext = nullptr,
        .flags = 0,
        .waitSemaphoreInfoCount = waits ? 1U : 0U,
        .pWaitSemaphoreInfos = waits ? &wait_info : nullptr,
        .commandBufferInfoCount = cmdbuf != VK_NULL_HANDLE ? 1U : 0U,
        .pCommandBufferInfos = cmdbuf != VK_NULL_HANDLE ? &cmdbuf_info : nullptr,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signal_info,
    };
    std::scoped_lock lock{mutex};
    return queue.Submit2(submit_info);
}

SubmitEngine::SubmitEngine(const Device& device_, MasterSemaphore& master_,
                           std::mutex& submit_mutex_, bool threaded_, TransferQueue* transfer_)
    : device{device_}, master{master_}, submit_mutex{submit_mutex_}, transfer{transfer_},
      threaded{threaded_} {
    buffer_consumer_stages = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                             VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                             VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                             VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    if (device.IsExtTransformFeedbackSupported()) {
        buffer_consumer_stages |= VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT;
    }
    if (device.IsExtConditionalRendering()) {
        buffer_consumer_stages |= VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT;
    }
    if (threaded) {
        submit_thread =std::jthread([this](std::stop_token token) { SubmitThread(token); });
        finish_thread = std::jthread([this](std::stop_token token) { FinishThread(token); });
        LOG_INFO(Render_Vulkan, "EDEN_NX2_V2_SUBMITTHREAD: submit and finish threads started");
    }
}

SubmitEngine::~SubmitEngine() {
    if (threaded) {
        submit_thread.request_stop();
        submit_thread.join();
        finish_thread.request_stop();
        finish_thread.join();
    }
}

SubmitContext* SubmitEngine::CreateContext() {
    auto context = std::make_unique<SubmitContext>();
    const auto& logical = device.GetLogical();
    context->pool = logical.CreateCommandPool({
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = device.GetGraphicsFamily(),
    });
    context->buffers = context->pool.Allocate(2);
    if (transfer != nullptr) {
        context->xfer_pool = logical.CreateCommandPool({
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = transfer->Family(),
        });
        context->xfer_buffers = context->xfer_pool.Allocate(1);
    }
    SubmitContext* const raw = context.get();
    contexts.push_back(std::move(context));
    return raw;
}

void SubmitEngine::ResetContext(SubmitContext& context) {
    const auto& dld = device.GetDispatchLoader();
    const VkDevice handle = *device.GetLogical();
    vk::Check(dld.vkResetCommandPool(handle, *context.pool, 0));
    if (context.xfer_pool) {
        vk::Check(dld.vkResetCommandPool(handle, *context.xfer_pool, 0));
    }
    context.xfer_begun = false;
}

void SubmitEngine::RecycleCompleted() {
    // Inline mode (the worker owns everything): contexts whose submission the GPU finished.
    if (in_flight.empty()) {
        return;
    }
    master.Refresh();
    const u64 gpu_tick = master.KnownGpuTick();
    while (!in_flight.empty() && in_flight.front()->tick <= gpu_tick) {
        SubmitContext* const context = in_flight.front();
        in_flight.pop_front();
        ResetContext(*context);
        free_contexts.push_back(context);
        P5bStats::Add(Counter::RecycledContexts);
    }
    RunRetired(gpu_tick);
}

SubmitContext* SubmitEngine::Acquire() {
    if (!threaded) {
        RecycleCompleted();
        if (free_contexts.empty() && contexts.size() >= MaxContexts && !in_flight.empty()) {
            master.Wait(in_flight.front()->tick);
            RecycleCompleted();
        }
        if (!free_contexts.empty()) {
            SubmitContext* const context = free_contexts.front();
            free_contexts.pop_front();
            return context;
        }
        return CreateContext();
    }
    std::unique_lock lock{free_mutex};
    if (free_contexts.empty() && contexts.size() >= MaxContexts) {
        free_cv.wait(lock, [this] { return !free_contexts.empty(); });
    }
    if (!free_contexts.empty()) {
        SubmitContext* const context = free_contexts.front();
        free_contexts.pop_front();
        return context;
    }
    lock.unlock();
    return CreateContext();
}

void SubmitEngine::DoSubmit(SubmitWork& work) {
    SubmitContext& context = *work.context;
    const auto& dld = device.GetDispatchLoader();
    std::array<VkSemaphoreSubmitInfo, 10> waits{};
    u32 num_waits = 0;
    if (work.transfer && transfer != nullptr) {
        // The transfer batch first: it waits for the graphics tick its resources were last used
        // by (already finished ones need no wait) and signals the transfer timeline with this
        // submission's tick, which the graphics submission waits for.
        const u64 wait_value = work.transfer_wait > master.KnownGpuTick() ? work.transfer_wait : 0;
        const VkResult result = transfer->Submit(context.Transfer(), master.TimelineHandle(),
                                                 wait_value, transfer->Timeline(), work.tick);
        if (result == VK_ERROR_DEVICE_LOST) {
            device.ReportLoss();
        }
        vk::Check(result);
        P5bStats::Add(Counter::TransferBatches);
    }
    if (work.graphics_waits_transfer && transfer != nullptr) {
        // U-09 (§8 joins): buffer uploads are read only by copies, vertex/index fetch, indirect
        // reads, shaders and the optional buffer-reading stages; attachment work of the
        // submission need not wait. Image uploads are acquired at its start: every stage.
        waits[num_waits++] = VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = transfer->Timeline(),
            .value = work.tick,
            .stageMask = work.transfer_images ? VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
                                              : buffer_consumer_stages,
            .deviceIndex = 0,
        };
    }
    for (u32 i = 0; i < work.num_extra_waits && num_waits < waits.size(); ++i) {
        waits[num_waits++] = VkSemaphoreSubmitInfo{
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
            .pNext = nullptr,
            .semaphore = work.extra_waits[i].first,
            .value = work.extra_waits[i].second,
            .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .deviceIndex = 0,
        };
    }
    vk::CommandBuffer cmdbuf{context.Main(), dld};
    vk::CommandBuffer upload_cmdbuf{context.Upload(), dld};
    VkResult result;
    {
        std::scoped_lock lock{submit_mutex};
        result = master.SubmitQueue(cmdbuf, upload_cmdbuf, work.signal_semaphore,
                                    work.wait_semaphore, work.tick,
                                    std::span<const VkSemaphoreSubmitInfo>(waits.data(), num_waits));
    }
    switch (result) {
    case VK_SUCCESS:
        break;
    case VK_ERROR_DEVICE_LOST:
        device.ReportLoss();
        [[fallthrough]];
    default:
        vk::Check(result);
        break;
    }
    // Perf fix C: readback jobs of this tick go to the transfer queue behind this tick's own
    // transfer batch (submitted above), so queue order needs no CPU wait on the recorder.
    for (u32 i = 0; i < work.num_post_jobs && transfer != nullptr; ++i) {
        const SubmitWork::PostJob& job = work.post_jobs[i];
        const VkResult job_result = transfer->Submit(job.cmdbuf, master.TimelineHandle(),
                                                     work.tick, job.semaphore, job.value);
        if (job_result == VK_ERROR_DEVICE_LOST) {
            device.ReportLoss();
        }
        vk::Check(job_result);
    }
    queued_tick.store(work.tick, std::memory_order_release);
    context.submitted = std::chrono::steady_clock::now();
    if (P5bStats::Enabled()) {
        P5bStats::Add(Counter::SubmitThreadSubmits, threaded ? 1 : 0);
        if (threaded) {
            P5bStats::Add(Counter::SubmitLatencyUs, MicrosecondsSince(work.handed));
        }
    }
}

void SubmitEngine::Submit(SubmitWork&& work) {
    work.context->tick = work.tick;
    if (!threaded) {
        DoSubmit(work);
        in_flight.push_back(work.context);
        return;
    }
    work.handed = std::chrono::steady_clock::now();
    // Perf fix C: wake the submit thread only when it sleeps (it pops the queue before it does).
    bool wake;
    {
        std::scoped_lock lock{submit_queue_mutex};
        submit_queue.push_back(std::move(work));
        handed.fetch_add(1, std::memory_order_release);
        wake = submit_waiting;
    }
    if (wake) {
        submit_cv.notify_one();
    }
}

void SubmitEngine::WaitSubmitted() {
    if (!threaded) {
        return;
    }
    const u64 target = handed.load(std::memory_order_acquire);
    if (submitted.load(std::memory_order_acquire) >= target) {
        return;
    }
    std::unique_lock lock{submitted_mutex};
    ++submitted_waiters;
    submitted_cv.wait(lock,
                      [&] { return submitted.load(std::memory_order_acquire) >= target; });
    --submitted_waiters;
}

void SubmitEngine::RetireAfter(u64 tick, std::function<void()>&& func) {
    if (master.IsFree(tick)) {
        func();
        return;
    }
    std::scoped_lock lock{retire_mutex};
    retires.push_back({tick, std::move(func)});
}

void SubmitEngine::RunRetired(u64 completed_tick) {
    std::vector<std::function<void()>> ready;
    {
        std::scoped_lock lock{retire_mutex};
        if (retires.empty()) {
            return;
        }
        std::erase_if(retires, [&](Retire& retire) {
            if (retire.tick > completed_tick) {
                return false;
            }
            ready.push_back(std::move(retire.func));
            return true;
        });
    }
    for (auto& func : ready) {
        func();
    }
}

void SubmitEngine::SubmitThread(std::stop_token token) {
    Common::SetCurrentThreadName("VulkanSubmit");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Critical);
    while (true) {
        SubmitWork work;
        {
            std::unique_lock lock{submit_queue_mutex};
            submit_cv.wait(lock, token, [this] {
                submit_waiting = submit_queue.empty();
                return !submit_waiting;
            });
            submit_waiting = false;
            if (submit_queue.empty()) {
                return; // Stop requested with nothing left to submit
            }
            work = std::move(submit_queue.front());
            submit_queue.pop_front();
        }
        DoSubmit(work);
        bool wake_finish;
        {
            std::scoped_lock lock{finish_mutex};
            finish_queue.push_back(work.context);
            wake_finish = finish_waiting;
        }
        if (wake_finish) {
            finish_cv.notify_one();
        }
        bool wake_waiters;
        {
            std::scoped_lock lock{submitted_mutex};
            submitted.fetch_add(1, std::memory_order_release);
            wake_waiters = submitted_waiters != 0;
        }
        if (wake_waiters) {
            submitted_cv.notify_all();
        }
    }
}

void SubmitEngine::FinishThread(std::stop_token token) {
    Common::SetCurrentThreadName("VulkanFinish");
    // F4-10: this thread is the timeline's one waiter; others sleep on what it publishes.
    master.SetFinishPublisher(true);
    SCOPE_EXIT {
        master.SetFinishPublisher(false);
    };
    while (true) {
        SubmitContext* context = nullptr;
        {
            std::unique_lock lock{finish_mutex};
            finish_cv.wait(lock, token, [this] {
                finish_waiting = finish_queue.empty();
                return !finish_waiting;
            });
            finish_waiting = false;
            if (finish_queue.empty()) {
                return;
            }
            context = finish_queue.front();
        }
        // Submissions complete in order: wait for this one (with a timeout so a stop request at
        // shutdown is seen; then the remaining ones get two seconds).
        const auto stop_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
        bool done = false;
        while (!done) {
            done = master.WaitFor(context->tick, 5'000'000);
            if (!done && token.stop_requested() &&
                std::chrono::steady_clock::now() >= stop_deadline) {
                return;
            }
        }
        master.PublishFromFinish();
        {
            std::scoped_lock lock{finish_mutex};
            finish_queue.pop_front();
        }
        if (P5bStats::Enabled()) {
            P5bStats::Add(Counter::FinishCompletions);
            P5bStats::Add(Counter::FinishLatencyUs, MicrosecondsSince(context->submitted));
            P5bStats::Add(Counter::RecycledContexts);
        }
        ResetContext(*context);
        {
            std::scoped_lock lock{free_mutex};
            free_contexts.push_back(context);
        }
        free_cv.notify_one();
        RunRetired(master.KnownGpuTick());
    }
}

} // namespace Vulkan
