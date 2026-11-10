// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/timeline_join.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {

class Device;
class Scheduler;
class TransferQueue;

/// EDEN_NX2_V2_XREADBACK: guest readbacks (buffers, mapped buffers, textures) copied on the
/// transfer queue with per-thread background resources (Ryujinx's BackgroundResources): each
/// thread that submits readback jobs has its own transfer command pool and timeline semaphore,
/// so jobs from different threads never need one ordering. A job waits for the graphics tick
/// that produced the data; the next graphics submission waits for the job (graphics work after
/// the readback must not overwrite what it reads). Images are exclusive to the graphics family:
/// the job's copy is wrapped in queue family ownership transfers (release at the end of the
/// producing submission, acquire on the transfer queue, release back, acquire at the start of
/// the next graphics submission).
class BackgroundReadback {
public:
    struct Job {
        VkSemaphore semaphore{};
        u64 value{};
        std::chrono::steady_clock::time_point submitted{};
        [[nodiscard]] bool Valid() const noexcept {
            return semaphore != VK_NULL_HANDLE && value != 0;
        }
    };

    explicit BackgroundReadback(const Device& device, Scheduler& scheduler);
    ~BackgroundReadback();

    /// EDEN_NX2_V2_XREADBACK with a transfer queue.
    [[nodiscard]] bool Active() const noexcept {
        return active;
    }

    /// Recording thread (the caches' owner): copies for the job being built.
    void AddBufferCopy(VkBuffer src, VkBuffer dst, const VkBufferCopy& copy);
    /// Records the graphics-side release of `image` in the open submission.
    void AddImageCopy(VkImage image, VkImageAspectFlags aspect, VkBuffer dst,
                      std::span<const VkBufferImageCopy> copies);
    [[nodiscard]] bool Empty() const noexcept {
        return buffer_copies.empty() && image_copies.empty();
    }

    /// Submits the job on this thread's background resources. An invalid job means the caller
    /// must take the graphics path (nothing queued). F4-04: `producer` is the newest graphics
    /// tick that used the buffers read (UINT64_MAX: unknown, treated as the open one). Already on
    /// the graphics queue: the job goes to the transfer queue now, waiting on it, without
    /// flushing. Still open: the job rides the open submission, which is flushed at once.
    /// `ride_tick` (if given) is set to 0: no job is left riding an unsubmitted tick.
    Job Submit(u64 producer = (std::numeric_limits<u64>::max)(), u64* ride_tick = nullptr);

    /// Any thread: waits for the job (and counts its latency). `ride_tick`: the open submission
    /// the job rode (Submit), asked for first if still unsubmitted.
    void Wait(const Job& job, u64 ride_tick = 0);

private:
    struct BufferCopy {
        VkBuffer src;
        VkBuffer dst;
        VkBufferCopy copy;
    };
    struct ImageCopy {
        VkImage image;
        VkImageAspectFlags aspect;
        VkBuffer dst;
        std::vector<VkBufferImageCopy> copies;
    };
    struct Resources {
        vk::CommandPool pool;
        vk::Semaphore timeline;
        VideoCommon::V2::JobTimeline values;
        u64 last_post_tick = 0; ///< tick the newest post job rides (queued behind it)
        struct InFlight {
            u64 value;
            VkCommandBuffer cmdbuf;
        };
        std::vector<InFlight> in_flight;
        std::vector<VkCommandBuffer> free_cmdbufs;
        std::vector<vk::CommandBuffers> allocations;
    };

    Resources& ThreadResources();
    VkCommandBuffer TakeCommandBuffer(Resources& resources);

    const Device& device;
    Scheduler& scheduler;
    TransferQueue* transfer{};
    bool active{};
    std::vector<BufferCopy> buffer_copies;
    std::vector<ImageCopy> image_copies;
    std::mutex resources_mutex;
    std::unordered_map<std::thread::id, std::unique_ptr<Resources>> resources;
};

} // namespace Vulkan
