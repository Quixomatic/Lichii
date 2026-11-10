// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <memory>
#include <vector>

#include "common/assert.h"
#include "core/core.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/control/channel_state.h"
#include "video_core/control/scheduler.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"

namespace Tegra::Control {

void Scheduler::BindChannel(GPU& gpu, s32 bind_id) {
    // EDEN_NX2_V2_MULTIFRONT: the channel the front dispatches next.
    VideoCommon::V2::Resolve::MultiFront::t_channel = static_cast<u32>(bind_id);
    // With the GPU thread split the rasterizer's channel switches with the methods around it, on
    // the back thread (deferred outside the lock: the back takes it to bind).
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        split.SetFrontChannel(static_cast<u32>(bind_id));
        const VideoCommon::V2::Ir::ChannelBody body{
            .scheduler = split.Ref(this),
            .gpu = split.Ref(&gpu),
            .bind_id = bind_id,
            .reserved = 0,
        };
        split.Emit(VideoCommon::V2::Ir::Kind::Channel, body,
                   [this, gpu_ptr = &gpu, bind_id] { BindChannelOnBack(*gpu_ptr, bind_id); });
        return;
    }
    BindChannelOnBack(gpu, bind_id);
}

void Scheduler::BindChannelOnBack(GPU& gpu, s32 bind_id) {
    std::scoped_lock lk{scheduling_guard};
    gpu.BindChannel(bind_id);
}

void Scheduler::Push(GPU& gpu, s32 channel, CommandList&& entries) {
    std::shared_ptr<ChannelState> channel_state;
    {
        std::unique_lock lk(scheduling_guard);
        auto it = channels.find(channel);
        ASSERT(it != channels.end());
        channel_state = it->second;
    }
    BindChannel(gpu, channel_state->bind_id);
    VideoCommon::V2::Resolve::MultiFront::Note(VideoCommon::V2::Resolve::MultiFront::EventKind::Work);
    // Process commands outside the lock to reduce contention.
    // Multiple channels can prepare their commands in parallel.
    channel_state->payload->dma_pusher.Push(std::move(entries));
    channel_state->payload->dma_pusher.DispatchCalls();
    if (channel_state->payload->dma_pusher.HasPendingAcquire() &&
        std::find(pending_channels.begin(), pending_channels.end(), channel_state) ==
            pending_channels.end()) {
        pending_channels.push_back(channel_state);
        any_pending.store(true, std::memory_order_release);
    }
    if (!pending_channels.empty()) {
        ResumePending(gpu);
    }
}

bool Scheduler::ResumePending(GPU& gpu) {
    bool any_progress = false;
    // Running one channel can release the semaphore another waits for: retry until none moves.
    for (bool progress = true; progress;) {
        progress = false;
        for (const auto& state : pending_channels) {
            auto& dma_pusher = state->payload->dma_pusher;
            // Powered off (shutting down), DispatchCalls returns at once and the acquire stays
            // pending: resuming it again counted as progress forever and the GPU thread never
            // reached its stop request (closing the game hung).
            if (!dma_pusher.system.IsPoweredOn()) {
                return any_progress;
            }
            if (!dma_pusher.HasPendingAcquire() || !dma_pusher.CanResumeAcquire()) {
                continue;
            }
            BindChannel(gpu, state->bind_id);
            dma_pusher.DispatchCalls();
            progress = true;
            any_progress = true;
        }
    }
    std::erase_if(pending_channels, [](const auto& state) {
        return !state->payload->dma_pusher.HasPendingAcquire();
    });
    any_pending.store(!pending_channels.empty(), std::memory_order_release);
    return any_progress;
}

void Scheduler::DeclareChannel(std::shared_ptr<ChannelState> new_channel) {
    s32 channel = new_channel->bind_id;
    std::unique_lock lk(scheduling_guard);
    channels.emplace(channel, new_channel);
}

} // namespace Tegra::Control
