// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include "common/bounded_threadsafe_queue.h"
#include "common/polyfill_thread.h"
#include "video_core/dma_pusher.h"
#include "video_core/framebuffer_config.h"
#include "video_core/v2/ir.h"

namespace Tegra {
struct FramebufferConfig;
namespace Control {
class Scheduler;
}
} // namespace Tegra

namespace Core {
namespace Frontend {
class GraphicsContext;
}
class System;
} // namespace Core

namespace VideoCore {
class RasterizerInterface;
class RendererBase;
} // namespace VideoCore

namespace VideoCommon::GPUThread {

/// What a CpuRecord asks the GPU thread to do (Pipeline v2 P1b: typed records instead of a
/// variant of commands and std::function closures).
enum class CpuOp : u8 {
    Submit,      ///< Parse a command list of `channel` (`list`)
    Flush,       ///< Flush [address, address + size) of the caches (synchronous GPU mode)
    Tick,        ///< Run the GPU's TickWork: its typed sync requests (GPU::Impl)
    Map,         ///< A GPU/SMMU memory map or unmap (`map`, objects as pointers)
    Notify,      ///< Nothing to run: its sequence tells a waiter the GPU got past it
    SignalEvent, ///< Signal the kernel event `object` (owned by its service)
};

/// A record of the CPU->GPU-thread queue. `sequence` is its command fence: +1 per record and
/// signalled in queue order once the record ran (SignalCommandFence), on the split's back for
/// everything but command lists.
struct CpuRecord {
    CpuOp op{};
    bool block{};
    s32 channel{};
    u64 sequence{};
    u64 address{};
    u64 size{};
    u64 object{};
    VideoCommon::V2::Ir::MapBody map{};
    Tegra::CommandList list;
};

/// Struct used to synchronize the GPU thread
struct SynchState final {
    using CommandQueue = Common::SPSCQueue<CpuRecord>;
    std::mutex write_lock;
    CommandQueue queue;
    u64 last_fence{};
    std::atomic<u64> signaled_fence{};
    std::condition_variable_any cv;
};

/// Tells a CPU thread waiting in PushCommand that the command with `fence` ran (on the split's
/// back: a CommandSignal SyncPoint IR record or a deferred call).
void SignalCommandFence(SynchState& state, u64 fence, bool block);

/// Class used to manage the GPU thread
class ThreadManager final {
public:
    explicit ThreadManager(Core::System& system_);
    ~ThreadManager();

    /// Creates and starts the GPU thread.
    void StartThread(VideoCore::RendererBase& renderer, Core::Frontend::GraphicsContext& context,
                     Tegra::Control::Scheduler& scheduler);

    /// Push GPU command entries to be processed
    void SubmitList(s32 channel, Tegra::CommandList&& entries, bool is_async);

    /// Notify rasterizer that any caches of the specified region should be flushed to Switch memory
    void FlushRegion(DAddr addr, u64 size, bool is_async);

    /// Notify rasterizer that any caches of the specified region should be invalidated
    void InvalidateRegion(DAddr addr, u64 size);

    /// Notify rasterizer that any caches of the specified region should be flushed and invalidated
    void FlushAndInvalidateRegion(DAddr addr, u64 size, bool is_async);

    void TickGPU(bool is_async);

    /// A memory map or unmap, run once the commands queued before it have been processed
    /// (MapBeforeLaterReads: also before the GPU thread reads later command lists).
    void QueueMap(const VideoCommon::V2::Ir::MapBody& map, bool is_async);

    /// A record that runs nothing: returns its sequence, which SignaledSequence reaches once the
    /// commands queued before it have been processed.
    u64 Notify(bool is_async);

    /// Signals a kernel event once the commands queued before it have been processed.
    void SignalEvent(void* event, bool is_async);

    /// The sequence of the last record that ran.
    [[nodiscard]] u64 SignaledSequence() const {
        return state.signaled_fence.load(std::memory_order_acquire);
    }

    void NotifyShutdown();

private:
    /// Pushes a record to be executed by the GPU thread; returns its sequence.
    u64 PushCommand(CpuRecord&& record, bool block, bool is_async);
    /// With the split: the front sends a record that isn't a command list to the back.
    void SendToBack(CpuRecord& record, VideoCore::RasterizerInterface* rasterizer_ptr);

    Core::System& system;
    VideoCore::RasterizerInterface* rasterizer = nullptr;
    SynchState state;
    std::jthread thread;
};

} // namespace VideoCommon::GPUThread
