// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <memory>
#include <thread>

#include "common/logging.h"
#include "core/core.h"
#include "video_core/gpu_split.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_fence_manager.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_} {}

InnerFence::~InnerFence() = default;

void InnerFence::Queue(bool lazy_) {
    if (is_stubbed) {
        return;
    }
    // Get the current tick so we can wait for it
    wait_tick = scheduler.CurrentTick();
    if (lazy_) {
        lazy = true;
        queued_at = std::chrono::steady_clock::now();
        return;
    }
    if (!scheduler.DeferFlush()) {
        scheduler.Flush();
    }
}

bool InnerFence::IsSignaled() const {
    if (is_stubbed) {
        return true;
    }
    return scheduler.IsFree(wait_tick);
}

void InnerFence::Wait() {
    if (is_stubbed) {
        return;
    }
    // Perf fix C (EDEN_NX2_V2_ADAPTSUBMIT's fast-flush input): the time spent here on a tick that
    // isn't submitted while the guest waits on a sync. Fences pop in order, so a guest still
    // waiting waits on this fence or a later one: it is the guest waiting on unsubmitted work.
    const bool guest_waits = Core::g_nx2_guest_sync_waiters.load(std::memory_order_relaxed) != 0;
    std::chrono::steady_clock::time_point unsubmitted_since{};
    if (guest_waits && !scheduler.IsSubmitted(wait_tick)) [[unlikely]] {
        unsubmitted_since = std::chrono::steady_clock::now();
    }
    const auto note_submitted = [&] {
        if (unsubmitted_since != std::chrono::steady_clock::time_point{}) [[unlikely]] {
            Core::g_nx2_guest_sync_wait_us.fetch_add(
                static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                     std::chrono::steady_clock::now() - unsubmitted_since)
                                     .count()),
                std::memory_order_relaxed);
        }
    };
    // A deferred fence flush (Scheduler::DeferFlush) may not be submitted yet. Only the
    // split's back may submit; other threads (the fencing thread) ask it to and wait. If the back
    // has stopped, nothing else uses the scheduler and Wait below submits it.
    if (lazy && !scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread()) {
        // A lazy fence (early downloads) goes with the next submission the back makes anyway
        // (~1 per ms in heavy scenes). Only if none came within LazyWindow, or the guest is
        // waiting, does it ask for one, so the fence operations behind it never wait long.
        constexpr auto LazyWindow = std::chrono::milliseconds{1};
        for (u32 spins = 0; !scheduler.IsSubmitted(wait_tick); ++spins) {
            if (std::chrono::steady_clock::now() - queued_at >= LazyWindow ||
                Core::g_nx2_guest_sync_waiters.load(std::memory_order_relaxed) != 0 ||
                !VideoCommon::GpuSplit::Get().BackRunning()) {
                scheduler.RequestLazyFlush(wait_tick);
                break;
            }
            if (spins < 16) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds{50});
            }
        }
    }
    if (!scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread() && scheduler.WaitSubmitted(wait_tick)) {
        note_submitted();
        scheduler.GetMasterSemaphore().Wait(wait_tick);
        return;
    }
    note_submitted();
    scheduler.Wait(wait_tick);
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, const Device& device_, Scheduler& scheduler_)
    : GenericFenceManager{rasterizer_, gpu_, texture_cache_, buffer_cache_, query_cache_},
      scheduler{scheduler_} {}

Fence FenceManager::CreateFence(bool is_stubbed) {
    return std::make_shared<InnerFence>(scheduler, is_stubbed);
}

void FenceManager::QueueFence(Fence& fence) {
    fence->Queue(queue_lazily);
}

bool FenceManager::IsFenceSignaled(Fence& fence) const {
    return fence->IsSignaled();
}

void FenceManager::WaitFence(Fence& fence) {
    fence->Wait();
}

void FenceManager::SubmitDeferredWork() {
    if (VideoCommon::GpuSplit::OnBackThread()) {
        scheduler.FlushDeferred();
    }
}

} // namespace Vulkan
