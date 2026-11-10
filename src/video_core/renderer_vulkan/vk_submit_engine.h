// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class MasterSemaphore;

/// EDEN_NX2_V2_XFERQ: the device's dedicated transfer queue, its timeline (the transfer batch of
/// graphics submission T signals T) and the lock that serialises the threads submitting to it
/// (the submission owner for batches, readers' background resources for readback jobs).
class TransferQueue {
public:
    explicit TransferQueue(const Device& device);
    ~TransferQueue();

    [[nodiscard]] u32 Family() const noexcept {
        return family;
    }
    [[nodiscard]] VkSemaphore Timeline() const noexcept {
        return *timeline;
    }

    /// Submits `cmdbuf`, waiting for `wait_value` on `wait_semaphore` (0: none) and signalling
    /// `signal_value` on `signal_semaphore`.
    VkResult Submit(VkCommandBuffer cmdbuf, VkSemaphore wait_semaphore, u64 wait_value,
                    VkSemaphore signal_semaphore, u64 signal_value);

    /// Creates a timeline semaphore (background resources).
    [[nodiscard]] vk::Semaphore CreateTimeline() const;

private:
    const Device& device;
    u32 family{};
    vk::Queue queue;
    vk::Semaphore timeline;
    std::mutex mutex;
};

/// The command buffers of one graphics submission, each from a pool of its own so a finished
/// submission's pool can be reset in one call by whichever thread retires it.
struct SubmitContext {
    vk::CommandPool pool;
    vk::CommandBuffers buffers;  ///< [0] main, [1] upload
    vk::CommandPool xfer_pool;   ///< transfer family (XFERQ)
    vk::CommandBuffers xfer_buffers;
    bool xfer_begun{};
    u64 tick{};
    std::chrono::steady_clock::time_point submitted{};

    [[nodiscard]] VkCommandBuffer Main() const {
        return buffers[0];
    }
    [[nodiscard]] VkCommandBuffer Upload() const {
        return buffers[1];
    }
    [[nodiscard]] VkCommandBuffer Transfer() const {
        return xfer_pool ? xfer_buffers[0] : VkCommandBuffer{};
    }
};

/// One submission handed by the Vulkan worker (P5b).
struct SubmitWork {
    SubmitContext* context{};
    VkSemaphore signal_semaphore{};
    VkSemaphore wait_semaphore{};
    u64 tick{};
    bool transfer{};          ///< context's transfer command buffer is recorded
    u64 transfer_wait{};      ///< the transfer batch waits for graphics >= this (0: none)
    bool graphics_waits_transfer{};
    bool transfer_images{};   ///< U-09: the batch holds image uploads (wait at every stage)
    u32 num_extra_waits{};
    std::array<std::pair<VkSemaphore, u64>, 8> extra_waits{};
    /// Perf fix C (EDEN_NX2_V2_XREADBACK): readback jobs submitted on the transfer queue right
    /// after this submission; each waits for this tick and signals its own timeline value.
    struct PostJob {
        VkCommandBuffer cmdbuf{};
        VkSemaphore semaphore{};
        u64 value{};
    };
    u32 num_post_jobs{};
    std::array<PostJob, 4> post_jobs{};
    std::chrono::steady_clock::time_point handed{};
};

/// P5b submission engine.
/// - EDEN_NX2_V2_SUBMITTHREAD: a submit thread does every vkQueueSubmit2 of the scheduler (it is
///   the single submission owner of the graphics queue's scheduler work) and a finish thread waits
///   on the timeline, publishes the GPU tick, resets and recycles the submissions' command pools
///   and runs completion callbacks (readback completion, lifetimes). The Vulkan worker only
///   encodes.
/// - Without it (EDEN_NX2_V2_XFERQ alone) the worker submits inline as before, and contexts are
///   recycled when the worker finds their tick complete.
class SubmitEngine {
public:
    explicit SubmitEngine(const Device& device, MasterSemaphore& master, std::mutex& submit_mutex,
                          bool threaded, TransferQueue* transfer);
    ~SubmitEngine();

    SubmitEngine(const SubmitEngine&) = delete;
    SubmitEngine& operator=(const SubmitEngine&) = delete;

    [[nodiscard]] bool Threaded() const noexcept {
        return threaded;
    }

    /// Worker: a context whose command buffers are free (reset), growing the pool if needed.
    SubmitContext* Acquire();

    /// Worker: submits (inline) or hands the work to the submit thread.
    void Submit(SubmitWork&& work);

    /// Any thread: waits until every submission handed so far reached the queue. Needed before
    /// waiting on a binary semaphore a handed submission signals (present).
    void WaitSubmitted();

    /// Any thread: the newest tick whose submission (and transfer batch) reached its queue.
    [[nodiscard]] u64 QueuedTick() const noexcept {
        return queued_tick.load(std::memory_order_acquire);
    }

    /// Any thread: runs `func` on the finish thread (or the worker without one) once the GPU
    /// finished `tick`.
    void RetireAfter(u64 tick, std::function<void()>&& func);

private:
    struct Retire {
        u64 tick;
        std::function<void()> func;
    };

    SubmitContext* CreateContext();
    void ResetContext(SubmitContext& context);
    void DoSubmit(SubmitWork& work);
    void RecycleCompleted();
    void RunRetired(u64 completed_tick);
    void SubmitThread(std::stop_token token);
    void FinishThread(std::stop_token token);

    const Device& device;
    MasterSemaphore& master;
    std::mutex& submit_mutex;
    TransferQueue* transfer;
    const bool threaded;
    VkPipelineStageFlags2 buffer_consumer_stages{}; ///< U-09

    std::vector<std::unique_ptr<SubmitContext>> contexts;
    std::mutex free_mutex;
    std::condition_variable free_cv;
    std::deque<SubmitContext*> free_contexts;
    std::deque<SubmitContext*> in_flight; ///< inline mode: the worker's contexts not yet free

    std::mutex submit_queue_mutex;
    std::condition_variable_any submit_cv;
    std::deque<SubmitWork> submit_queue;
    bool submit_waiting{};   ///< under submit_queue_mutex: the submit thread sleeps
    std::atomic<u64> handed{};
    std::atomic<u64> submitted{};
    std::atomic<u64> queued_tick{};
    std::condition_variable submitted_cv;
    std::mutex submitted_mutex;
    u32 submitted_waiters{}; ///< under submitted_mutex

    std::mutex finish_mutex;
    std::condition_variable_any finish_cv;
    std::deque<SubmitContext*> finish_queue;
    bool finish_waiting{};   ///< under finish_mutex: the finish thread sleeps

    std::mutex retire_mutex;
    std::deque<Retire> retires;

    std::jthread submit_thread;
    std::jthread finish_thread;
};

} // namespace Vulkan
