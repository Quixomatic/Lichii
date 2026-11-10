// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include <memory>

#include "video_core/fence_manager.h"
#include "video_core/renderer_vulkan/vk_buffer_cache.h"
#include "video_core/renderer_vulkan/vk_query_cache.h"
#include "video_core/renderer_vulkan/vk_texture_cache.h"

namespace Core {
class System;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Vulkan {

class Device;
class Scheduler;

class InnerFence : public VideoCommon::FenceBase {
public:
    explicit InnerFence(Scheduler& scheduler_, bool is_stubbed_,
                        const Tegra::Host1x::SyncpointManager* syncpoints_ = nullptr);
    ~InnerFence();

    /// `lazy`: only record the wait point; the next submission takes the fence's work.
    void Queue(bool lazy = false);

    bool IsSignaled() const;

    void Wait();

private:
    Scheduler& scheduler;
    /// R2-F4-02: a wait on a host syncpoint value makes this fence's submission urgent.
    const Tegra::Host1x::SyncpointManager* syncpoints;
    u64 wait_tick = 0;
    bool lazy = false;
    bool adaptive = false; ///< F4-02: lazy because fences ride submissions (Scheduler::SetLazyFences)
    std::chrono::steady_clock::time_point queued_at{};
};
using Fence = std::shared_ptr<InnerFence>;

struct FenceManagerParams {
    using FenceType = Fence;
    using BufferCacheType = BufferCache;
    using TextureCacheType = TextureCache;
    using QueryCacheType = QueryCache;

    static constexpr bool HAS_ASYNC_CHECK = true;
};

using GenericFenceManager = VideoCommon::FenceManager<FenceManagerParams>;

class FenceManager final : public GenericFenceManager {
public:
    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer, Tegra::GPU& gpu,
                          TextureCache& texture_cache, BufferCache& buffer_cache,
                          QueryCache& query_cache, const Device& device, Scheduler& scheduler);

protected:
    Fence CreateFence(bool is_stubbed) override;
    void QueueFence(Fence& fence) override;
    bool IsFenceSignaled(Fence& fence) const override;
    void WaitFence(Fence& fence) override;
    void SubmitDeferredWork() override;
    void SubmitBeforeWait() override;

private:
    Scheduler& scheduler;
};

} // namespace Vulkan
