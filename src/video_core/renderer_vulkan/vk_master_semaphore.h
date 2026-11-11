// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <queue>
#include <span>

#include "common/common_types.h"
#include "common/polyfill_thread.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;

class MasterSemaphore {
    using Waitable = std::pair<u64, vk::Fence>;

public:
    explicit MasterSemaphore(const Device& device);
    ~MasterSemaphore();

    /// Returns the current logical tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }

    /// Returns the last known GPU tick.
    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }

    /// Returns true when a tick has been hit by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }

    /// Advance to the logical tick and return the old one
    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }

    /// Refresh the known GPU tick
    void Refresh();

    /// SYNC §14 #1 (MP §5.5 "pending GPU work", "timeline semaphores"): the GPU's progress as
    /// the timeline semaphore holds it now (one counter read), without publishing it (the finish
    /// thread's KnownGpuTick lags it by its wake-up, and its publish keeps the §11.2 handshake).
    [[nodiscard]] u64 GpuValueNow() const {
        const u64 known = KnownGpuTick();
        if (!semaphore) {
            return known;
        }
        const u64 counter = semaphore.GetCounter();
        return counter > known ? counter : known;
    }

    /// Waits for a tick to be hit on the GPU
    void Wait(u64 tick);

    /// Submits the device graphics queue, updating the tick as necessary. `extra_waits`
    /// (P5b: the transfer timeline and readback jobs) needs timeline semaphores and
    /// synchronization2.
    VkResult SubmitQueue(vk::CommandBuffer& cmdbuf, vk::CommandBuffer& upload_cmdbuf,
                         VkSemaphore signal_semaphore, VkSemaphore wait_semaphore, u64 host_tick,
                         std::span<const VkSemaphoreSubmitInfo> extra_waits = {});

    /// The timeline semaphore (null without timeline semaphore support).
    [[nodiscard]] VkSemaphore TimelineHandle() const noexcept {
        return semaphore ? *semaphore : VkSemaphore{};
    }

    /// Whether the GPU reached `tick`, asking the driver (finish thread).
    bool WaitFor(u64 tick, u64 timeout_ns) {
        return semaphore.Wait(tick, timeout_ns);
    }

    /// F4-10 (MASTER_PLAN §5.5: one thread waits on the timeline): while the finish thread runs
    /// (EDEN_NX2_V2_SUBMITTHREAD), Wait sleeps on the ticks it publishes instead of waiting on the
    /// driver itself. Off again when the finish thread stops.
    void SetFinishPublisher(bool on);
    /// Finish thread: refreshes the known tick and wakes Wait's sleepers.
    void PublishFromFinish();

private:
    VkResult SubmitQueueTimeline(vk::CommandBuffer& cmdbuf, vk::CommandBuffer& upload_cmdbuf,
                                 VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                                 u64 host_tick, std::span<const VkSemaphoreSubmitInfo> extra_waits);
    VkResult SubmitQueueFence(vk::CommandBuffer& cmdbuf, vk::CommandBuffer& upload_cmdbuf,
                              VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                              u64 host_tick);

    void WaitThread(std::stop_token token);

    vk::Fence GetFreeFence();

private:
    const Device& device;             ///< Device.
    vk::Semaphore semaphore;          ///< Timeline semaphore.
    std::atomic<u64> gpu_tick{0};     ///< Current known GPU tick.
    std::atomic<u64> current_tick{1}; ///< Current logical tick.
    std::mutex wait_mutex;
    std::mutex free_mutex;
    std::condition_variable_any wait_cv;
    std::queue<Waitable> wait_queue;  ///< Queue for the fences to be waited on by the wait thread.
    std::deque<vk::Fence> free_queue; ///< Holds available fences for submission.
    std::atomic<bool> finish_publisher{}; ///< F4-10: SetFinishPublisher
    std::atomic<u32> publish_waiters{};
    std::mutex publish_mutex;
    std::condition_variable publish_cv;
    std::jthread debug_thread;        ///< Debug thread to workaround validation layer bugs.
    std::jthread wait_thread;         ///< Helper thread that waits for submitted fences.
};

} // namespace Vulkan
