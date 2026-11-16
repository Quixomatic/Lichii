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
        if (scheduler.LazyFencesHere()) [[unlikely]] {
            // F4-X2 (chunk B2): a stubbed fence (a delayed fence with nothing to flush: deferred
            // QMD reductions and max releases) needs no GPU work of its own, but its operations
            // land only after every earlier fence popped, i.e. once the newest lazy fence's tick
            // is done: a waiter for them asks for that tick (LandingFence::TickFor). SYNC Step 4:
            // not urgent at queue time.
            const u64 tick = scheduler.LazyOutstandingTick();
            if (tick != 0) {
                VideoCommon::V2::LandingFence::NoteFence(tick);
            }
        }
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
        // F4-X2: a thread blocking on an operation of this fence asks for this tick (SYNC Step 4:
        // none is made urgent here).
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
    if ((adaptive || lazy) && !scheduler.IsSubmitted(wait_tick) &&
        VideoCommon::GpuSplit::Get().Active() && !VideoCommon::GpuSplit::OnBackThread())
        [[unlikely]] {
        // SYNC_SUBMIT_SPEC §2.5: the fencing thread is not a guest waiter. It waits for the
        // submission the back makes anyway (rule B at a pass boundary, C at a chunk, I when the
        // back goes idle with the GPU about to idle; a fence carrying captures counts as work
        // then), and files rule W (one urgent request, served at the back's next entry boundary)
        // only while a guest thread waits on a host syncpoint value (46e10b8289). No LazyWindow
        // / RequestGap / EscalateWindow timers (SESSION_HANDOFF §5 "lazy fences must request a
        // flush after a bounded time, and immediately while the guest waits on them": the bound
        // is the next boundary, chunk or idle hook).
        auto& split = VideoCommon::GpuSplit::Get();
        bool requested = false;
        // SYNC_SUBMIT_SPEC §7.3 fix 4 (MP §6 interrupt rule; SESSION_HANDOFF §5 lesson 2): the
        // blocked fencing thread is a real waiter for every later fence's operations. Once the
        // back has run entries past this fence's turn twice (two `consumed` stores: its batch
        // ends, where fix 1 already ran rule I) without submitting it, it files rule W once for
        // this tick. With the back asleep it does so at once. No timer; ≈ 0/s in steady state.
        u64 last_consumed = split.ConsumedPosition();
        u32 back_moves = 0;
        VideoCommon::V2::LandingFence::diag_fence_wait_tick.store(wait_tick,
                                                                  std::memory_order_relaxed);
        for (u32 spins = 0; !scheduler.IsSubmitted(wait_tick); ++spins) {
            if (!split.BackRunning()) {
                break;
            }
            if (const u64 consumed = split.ConsumedPosition(); consumed != last_consumed) {
                last_consumed = consumed;
                ++back_moves;
            }
            if (!requested && (host_waits() || back_moves >= 2 || split.BackIdle())) {
                scheduler.RequestLazyFlush(wait_tick, true);
                split.RequestIdleHook();
                requested = true;
                VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::FencingRequests);
            }
            if (spins < 16) {
                std::this_thread::yield();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds{50});
            }
        }
        VideoCommon::V2::LandingFence::diag_fence_wait_tick.store(0, std::memory_order_relaxed);
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
