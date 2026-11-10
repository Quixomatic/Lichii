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
#include "video_core/v2/landing_fence.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

InnerFence::InnerFence(Scheduler& scheduler_, bool is_stubbed_,
                       const Tegra::Host1x::SyncpointManager* syncpoints_)
    : FenceBase{is_stubbed_}, scheduler{scheduler_}, syncpoints{syncpoints_} {}

InnerFence::~InnerFence() = default;

void InnerFence::Queue(bool lazy_) {
    if (is_stubbed) {
        return;
    }
    // Get the current tick so we can wait for it
    wait_tick = scheduler.CurrentTick();
    if (scheduler.LazyFencesHere()) [[unlikely]] {
        // F4-02: every fence rides the open submission; a waiter asks for it (Wait below), the
        // back's idle hook or the next heuristic submission takes it.
        lazy = true;
        adaptive = true;
        queued_at = std::chrono::steady_clock::now();
        scheduler.NoteLazyFence(wait_tick);
        // F4-X2: a thread blocking on an operation of this fence asks for this tick.
        VideoCommon::V2::LandingFence::NoteFence(wait_tick);
        return;
    }
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
    // R2-F4-02 (MASTER_PLAN §6: urgency belongs to the sync waited on): a guest thread waiting on
    // a host syncpoint value (nvhost_ctrl events, WaitHost) waits on what fence completions
    // advance, so this fence is urgent then. vi2's present wait reads the guest syncpoint, which
    // the back sets as it runs: it never needs a submit. Fast-flush mode's input is the time
    // blocked threads waited for submissions (Scheduler::TakeBlockedWaitUs).
    const auto host_waits = [this] {
        return syncpoints != nullptr && syncpoints->HasHostWaiters();
    };
    // A deferred fence flush (Scheduler::DeferFlush) may not be submitted yet. Only the
    // split's back may submit; other threads (the fencing thread) ask it to and wait. If the back
    // has stopped, nothing else uses the scheduler and Wait below submits it.
    if (adaptive && !scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread()) [[unlikely]] {
        // F4-02: the request is made at once while a host syncpoint wait is open, after
        // LazyWindow otherwise; the back serves it at its next pass boundary or chunk (RequestGap
        // after its last submission), an urgent one at its next idle hook, so requests made close
        // together share one submission. Past RequestGap (host wait) or EscalateWindow it is
        // forced through WaitSubmitted's urgent request below.
        constexpr auto LazyWindow = std::chrono::milliseconds{1};
        constexpr auto RequestGap = std::chrono::microseconds{500};
        constexpr auto EscalateWindow = std::chrono::milliseconds{4};
        std::chrono::steady_clock::time_point guest_since{};
        bool requested = false;
        bool urgent = false;
        for (u32 spins = 0; !scheduler.IsSubmitted(wait_tick); ++spins) {
            const auto now = std::chrono::steady_clock::now();
            if (!VideoCommon::GpuSplit::Get().BackRunning() || now - queued_at >= EscalateWindow) {
                break;
            }
            const bool guest = host_waits();
            if ((!requested && now - queued_at >= LazyWindow) || (guest && !urgent)) {
                scheduler.RequestLazyFlush(wait_tick, guest);
                requested = true;
                urgent = guest;
            }
            if (guest) {
                if (guest_since == std::chrono::steady_clock::time_point{}) {
                    guest_since = now;
                } else if (now - guest_since >= RequestGap) {
                    break;
                }
            }
            if (spins < 16) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds{50});
            }
        }
        if (!requested) {
            scheduler.RequestLazyFlush(wait_tick, false);
        }
    } else if (lazy && !scheduler.IsSubmitted(wait_tick) && VideoCommon::GpuSplit::Get().Active() &&
        !VideoCommon::GpuSplit::OnBackThread()) {
        // A lazy fence (early downloads) goes with the next submission the back makes anyway
        // (~1 per ms in heavy scenes). Only if none came within LazyWindow, or a guest thread
        // waits on a host syncpoint, does it ask for one, so the operations behind it never wait
        // long.
        constexpr auto LazyWindow = std::chrono::milliseconds{1};
        for (u32 spins = 0; !scheduler.IsSubmitted(wait_tick); ++spins) {
            const bool guest = host_waits();
            if (std::chrono::steady_clock::now() - queued_at >= LazyWindow || guest ||
                !VideoCommon::GpuSplit::Get().BackRunning()) {
                scheduler.RequestLazyFlush(wait_tick, guest);
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
        scheduler.GetMasterSemaphore().Wait(wait_tick);
        return;
    }
    scheduler.Wait(wait_tick);
}

FenceManager::FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                           TextureCache& texture_cache_, BufferCache& buffer_cache_,
                           QueryCache& query_cache_, const Device& device_, Scheduler& scheduler_)
    : GenericFenceManager{rasterizer_, gpu_, texture_cache_, buffer_cache_, query_cache_},
      scheduler{scheduler_} {}

Fence FenceManager::CreateFence(bool is_stubbed) {
    return std::make_shared<InnerFence>(scheduler, is_stubbed, &syncpoint_manager);
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

void FenceManager::SubmitBeforeWait() {
    if (VideoCommon::GpuSplit::OnBackThread()) {
        scheduler.ServeIdleHook();
    }
}

} // namespace Vulkan
