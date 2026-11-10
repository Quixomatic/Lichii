// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include "common/logging.h"
#include "video_core/renderer_vulkan/vk_background_readback.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_submit_engine.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/v2_switches.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {
namespace P5bStats = VideoCommon::V2::P5bStats;
using P5bStats::Counter;

constexpr VkAccessFlags ALL_WRITES =
    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
constexpr VkAccessFlags ALL_ACCESS = ALL_WRITES | VK_ACCESS_SHADER_READ_BIT |
                                     VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                     VK_ACCESS_TRANSFER_READ_BIT;

VkImageMemoryBarrier OwnershipBarrier(VkImage image, VkImageAspectFlags aspect, u32 src_family,
                                      u32 dst_family, VkAccessFlags src_access,
                                      VkAccessFlags dst_access) {
    return VkImageMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = src_access,
        .dstAccessMask = dst_access,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = src_family,
        .dstQueueFamilyIndex = dst_family,
        .image = image,
        .subresourceRange =
            {
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
    };
}
} // Anonymous namespace

BackgroundReadback::BackgroundReadback(const Device& device_, Scheduler& scheduler_)
    : device{device_}, scheduler{scheduler_}, transfer{scheduler_.GetTransferQueue()} {
    active = transfer != nullptr && VideoCommon::V2::On(VideoCommon::V2::Switch::XReadback);
    if (VideoCommon::V2::On(VideoCommon::V2::Switch::XReadback) && transfer == nullptr) {
        LOG_INFO(Render_Vulkan, "EDEN_NX2_V2_XREADBACK needs the transfer queue "
                                "(EDEN_NX2_V2_XFERQ and a dedicated transfer family)");
    }
}

BackgroundReadback::~BackgroundReadback() {
    // Jobs still in flight own command buffers of these pools: let them finish first.
    std::scoped_lock lock{resources_mutex};
    for (auto& [id, entry] : resources) {
        if (entry->values.LastIssued() != 0) {
            entry->timeline.Wait(entry->values.LastIssued(), 2'000'000'000);
        }
    }
}

BackgroundReadback::Resources& BackgroundReadback::ThreadResources() {
    std::scoped_lock lock{resources_mutex};
    auto& entry = resources[std::this_thread::get_id()];
    if (!entry) {
        entry = std::make_unique<Resources>();
        entry->pool = device.GetLogical().CreateCommandPool({
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = transfer->Family(),
        });
        entry->timeline = transfer->CreateTimeline();
    }
    return *entry;
}

VkCommandBuffer BackgroundReadback::TakeCommandBuffer(Resources& r) {
    if (!r.in_flight.empty()) {
        const u64 completed = r.timeline.GetCounter();
        r.values.NoteCompleted(completed);
        std::erase_if(r.in_flight, [&](const Resources::InFlight& entry) {
            if (entry.value > completed) {
                return false;
            }
            r.free_cmdbufs.push_back(entry.cmdbuf);
            return true;
        });
    }
    if (r.free_cmdbufs.empty()) {
        vk::CommandBuffers& allocation = r.allocations.emplace_back(r.pool.Allocate(4));
        for (size_t i = 0; i < 4; ++i) {
            r.free_cmdbufs.push_back(allocation[i]);
        }
    }
    const VkCommandBuffer cmdbuf = r.free_cmdbufs.back();
    r.free_cmdbufs.pop_back();
    return cmdbuf;
}

void BackgroundReadback::AddBufferCopy(VkBuffer src, VkBuffer dst, const VkBufferCopy& copy) {
    buffer_copies.push_back({src, dst, copy});
}

void BackgroundReadback::AddImageCopy(VkImage image, VkImageAspectFlags aspect, VkBuffer dst,
                                      std::span<const VkBufferImageCopy> copies) {
    // The graphics half of the ownership transfer: the last use of the image in the open
    // submission, which Submit flushes right after.
    const u32 graphics_family = device.GetGraphicsFamily();
    const u32 transfer_family = transfer->Family();
    scheduler.RequestOutsideRenderPassOperationContext();
    scheduler.Record([image, aspect, graphics_family, transfer_family](vk::CommandBuffer cmdbuf) {
        const VkImageMemoryBarrier release = OwnershipBarrier(
            image, aspect, graphics_family, transfer_family, ALL_WRITES, 0);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, release);
    });
    image_copies.push_back({image, aspect, dst, {copies.begin(), copies.end()}});
}

BackgroundReadback::Job BackgroundReadback::Submit(u64 producer, u64* ride_tick) {
    if (Empty()) {
        return {};
    }
    const u64 open_tick = scheduler.CurrentTick();
    const SubmitEngine* const engine = scheduler.GetSubmitEngine();
    const auto route = VideoCommon::V2::ChooseReadbackRoute(
        !image_copies.empty(), ride_tick != nullptr && scheduler.LazyFencesHere(),
        (std::min)(producer, open_tick), open_tick, engine != nullptr ? engine->QueuedTick() : 0,
        scheduler.GetMasterSemaphore().KnownGpuTick());
    // The producers (and the releases) go to the GPU; the job waits for that tick.
    const u64 producer_tick =
        route == VideoCommon::V2::ReadbackRoute::Direct ? producer : open_tick;
    Resources& r = ThreadResources();
    const vk::CommandBuffer cmdbuf{TakeCommandBuffer(r), device.GetDispatchLoader()};
    cmdbuf.Begin({
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    });
    const u32 graphics_family = device.GetGraphicsFamily();
    const u32 transfer_family = transfer->Family();
    for (const ImageCopy& copy : image_copies) {
        const VkImageMemoryBarrier acquire = OwnershipBarrier(
            copy.image, copy.aspect, graphics_family, transfer_family, 0,
            VK_ACCESS_TRANSFER_READ_BIT);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                               0, acquire);
        cmdbuf.CopyImageToBuffer(copy.image, VK_IMAGE_LAYOUT_GENERAL, copy.dst, copy.copies);
        const VkImageMemoryBarrier release = OwnershipBarrier(
            copy.image, copy.aspect, transfer_family, graphics_family, 0, 0);
        cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                               VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, release);
    }
    for (const BufferCopy& copy : buffer_copies) {
        cmdbuf.CopyBuffer(copy.src, copy.dst, copy.copy);
    }
    cmdbuf.End();
    const u64 value = r.values.Next();
    if (route == VideoCommon::V2::ReadbackRoute::Direct) {
        // F4-04: the producer and its transfer batch are already on their queues (QueuedTick),
        // so B1's cycle can't form: no flush, no worker wait. A producer of 0 needs no wait.
        const VkResult result =
            transfer->Submit(*cmdbuf, scheduler.GetMasterSemaphore().TimelineHandle(),
                             producer_tick, *r.timeline, value);
        if (result == VK_ERROR_DEVICE_LOST) {
            device.ReportLoss();
        }
        vk::Check(result);
        P5bStats::Add(Counter::ReadbacksDirect);
    } else if (route == VideoCommon::V2::ReadbackRoute::Ride &&
               scheduler.AddTransferJob(*cmdbuf, *r.timeline, value)) {
        // F4-04: the producer is open: the job rides it with no submit of its own. The open
        // submission goes out at the next heuristic submit, the idle hook, or when the reader
        // asks for it (Wait with ride_tick).
        scheduler.WantSubmit();
        *ride_tick = open_tick;
        P5bStats::Add(Counter::ReadbacksRiding);
    } else if (scheduler.AddTransferJob(*cmdbuf, *r.timeline, value)) {
        // Perf fix C (MASTER_PLAN §6 class C): the job rides the producing submission: the submit
        // path queues it on the transfer queue right behind that tick's own transfer batch, so
        // the recorder never waits for the worker. Fix pass 1's B1 order (job after the batch
        // graphics `producer_tick` waits for) holds by construction.
        scheduler.Flush();
    } else {
        scheduler.Flush();
        // Fix pass 1 (B1, the save-load deadlock): Flush only hands the tick to the Vulkan
        // worker, and its transfer batch reaches the queue when the worker submits it; a job
        // submitted before that waits on a tick that waits for the batch behind it.
        scheduler.WaitWorker();
        const VkResult result =
            transfer->Submit(*cmdbuf, scheduler.GetMasterSemaphore().TimelineHandle(),
                             producer_tick, *r.timeline, value);
        if (result == VK_ERROR_DEVICE_LOST) {
            device.ReportLoss();
        }
        vk::Check(result);
    }
    r.in_flight.push_back({value, *cmdbuf});
    const Job job{*r.timeline, value, std::chrono::steady_clock::now()};
    // Graphics work after the readback must not overwrite what it reads; the images come back
    // to the graphics family at the start of the next submission.
    const bool joined = scheduler.AddGraphicsWait(*r.timeline, value);
    if (!image_copies.empty()) {
        std::vector<std::pair<VkImage, VkImageAspectFlags>> images;
        for (const ImageCopy& copy : image_copies) {
            images.emplace_back(copy.image, copy.aspect);
        }
        scheduler.RecordWithUploadBuffer([images = std::move(images), graphics_family,
                                          transfer_family](vk::CommandBuffer,
                                                           vk::CommandBuffer upload_cmdbuf) {
            for (const auto& [image, aspect] : images) {
                const VkImageMemoryBarrier acquire = OwnershipBarrier(
                    image, aspect, transfer_family, graphics_family, 0, ALL_ACCESS);
                upload_cmdbuf.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, acquire);
            }
        });
        scheduler.BlockTransferHoisting();
    }
    if (P5bStats::Enabled()) {
        P5bStats::Add(Counter::Readbacks);
        P5bStats::Add(Counter::TextureReadbacks, image_copies.size());
    }
    buffer_copies.clear();
    image_copies.clear();
    if (!joined) {
        // Every wait slot of the next submission holds another thread's job: wait here instead.
        Wait(job, ride_tick != nullptr ? *ride_tick : 0);
    }
    return job;
}

void BackgroundReadback::Wait(const Job& job, u64 ride_tick) {
    if (!job.Valid()) {
        return;
    }
    if (ride_tick != 0) [[unlikely]] {
        // F4-04: the job rides an open submission: a reader waiting on it is the "interrupt" that
        // forces the submit (MASTER_PLAN §6).
        scheduler.EnsureSubmitted(ride_tick);
    }
    const VkSemaphoreWaitInfo wait_info{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .pNext = nullptr,
        .flags = 0,
        .semaphoreCount = 1,
        .pSemaphores = &job.semaphore,
        .pValues = &job.value,
    };
    const VkResult result = device.GetDispatchLoader().vkWaitSemaphores(
        *device.GetLogical(), &wait_info, (std::numeric_limits<u64>::max)());
    if (result == VK_ERROR_DEVICE_LOST) {
        device.ReportLoss();
    }
    vk::Check(result);
    if (P5bStats::Enabled()) {
        P5bStats::Add(Counter::ReadbackLanded);
        P5bStats::Add(Counter::ReadbackLatencyUs,
                      static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                           std::chrono::steady_clock::now() - job.submitted)
                                           .count()));
    }
}

} // namespace Vulkan
