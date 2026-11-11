// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <queue>
#include <initializer_list>
#include <source_location>
#include <vector>

#include "common/alignment.h"
#include "common/common_types.h"
#include "common/dkb_frame_trace.h"
#include "common/logging.h"
#include "common/polyfill_thread.h"
#include "common/settings.h"
#include "video_core/gpu_split.h"
#include "video_core/v2/frame_causes.h"
#include "video_core/renderer_vulkan/fragment_barrier_state.h"
#include "video_core/renderer_vulkan/vk_master_semaphore.h"
#include "video_core/v2/barrier_tracker.h"
#include "video_core/v2/after_pass.h"
#include "video_core/v2/landing_fence.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/submit_requests.h"
#include "video_core/v2/timeline_join.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace VideoCommon {
template <typename Trait>
class QueryCacheBase;
}

namespace Vulkan {

class CommandPool;
class Device;
class Framebuffer;
class GraphicsPipeline;
class StateTracker;
class BackgroundReadback;
class SubmitEngine;
class TransferQueue;
struct SubmitContext;

struct QueryCacheParams;

/// The scheduler abstracts command buffer and fence management with an interface that's able to do
/// OpenGL-like operations on Vulkan command buffers.
class Scheduler {
public:
    explicit Scheduler(const Device& device, StateTracker& state_tracker);
    ~Scheduler();

    /// Sends the current execution context to the GPU.
    u64 Flush(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
              std::source_location location = std::source_location::current());

    /// Sends the current execution context to the GPU and waits for it to complete.
    void Finish(VkSemaphore signal_semaphore = {}, VkSemaphore wait_semaphore = {},
                std::source_location location = std::source_location::current());

    /// On the split GPU thread, a fence or a command list copy submits only if the last
    /// submission is older than a window (EDEN_NX2_FENCE_FLUSH_US, default 500 us, 0 submits at
    /// once); otherwise it goes with the next one. The back submits deferred work before it
    /// sleeps and when another thread asks (WaitSubmitted). True when the flush was deferred.
    bool DeferFlush();
    /// Submits deferred work (the split's back, from its idle hook), and a lazy fence's tick
    /// another thread asked for (RequestLazyFlush).
    void FlushDeferred();
    /// Any thread: a lazy fence (early downloads) waited long enough for `tick` to be submitted;
    /// the back submits it at its next idle hook. Follow with WaitSubmitted.
    /// `urgent` (F4-X2): the caller blocks on the tick now, and the back's next idle hook submits
    /// it; a lazy fence's bounded-time request passes false and rides the next pass boundary or
    /// chunk.
    void RequestLazyFlush(u64 tick, bool urgent = true) {
        submit_requests.Request(tick, urgent);
        if (urgent && lazy_fences) [[unlikely]] {
            // Ryujinx's fast-flush input: how long blocked threads wait for submissions.
            u64 none = 0;
            blocked_since_us.compare_exchange_strong(none, NowUs(), std::memory_order_relaxed);
        }
    }
    /// F4-X2 (R2-F4-02): microseconds threads blocked on unsubmitted ticks since the last call
    /// (FlushHeuristic::NoteGuestSyncWait).
    [[nodiscard]] u64 TakeBlockedWaitUs() noexcept {
        return blocked_wait_us.exchange(0, std::memory_order_relaxed);
    }

    /// F4-02 (EDEN_NX2_V2_ADAPTSUBMIT, MASTER_PLAN §6 classes A/B, Ryujinx SyncManager non-strict
    /// syncs): fences queued on the split's back ride the open submission instead of submitting
    /// within DeferFlush's window. Their tick goes out with the next heuristic submission (pass
    /// boundaries), a waiter's request (RequestLazyFlush), or the back's idle hook.
    void SetLazyFences(bool on) {
        lazy_fences = on;
    }
    /// Whether a fence queued here now is lazy (F4-02 mode on the split's back).
    [[nodiscard]] bool LazyFencesHere() const;
    /// SYNC_SUBMIT_SPEC Step 8 (U-01, MASTER_PLAN §6 "both started early"): runs on the recording
    /// thread at the start of every submission, so early downloads of the work in it are issued
    /// with the producer's submit rather than at the next fence.
    void SetPreSubmitHook(void (*hook)(void*), void* argument) {
        pre_submit_hook = hook;
        pre_submit_argument = argument;
    }
    /// SYNC_SUBMIT_SPEC Step 9 (MP §6 class D: results land at a host fence, so a copy that waits
    /// for the pass end changes nothing a guest sees before it): with the range tracker, runs
    /// `job(argument)` at the open pass's next natural end or at the submission, never ending
    /// the pass itself. False with no pass open (the caller runs it now). Deferring the same job
    /// again before it ran adds nothing (it then covers both).
    bool DeferToPassEnd(void (*job)(void*), void* argument) {
        if (!track_barriers || !state.renderpass) {
            return false;
        }
        after_pass.Defer(job, argument);
        return true;
    }
    /// Recording thread: a lazy fence waits for `tick`.
    void NoteLazyFence(u64 tick) {
        lazy_outstanding_tick = tick;
        if (lazy_fences && tick == CurrentTick()) {
            NoteOpenTickWork(); // §11.2: a lazy fence alone makes the open tick worth a submit
        }
        VideoCommon::V2::LandingFence::diag_lazy_tick.store(tick, std::memory_order_relaxed);
        if (open_since_us == 0) {
            open_since_us = NowUs(); // SYNC §7.3: the open tick's age at its submit
        }
    }
    /// Recording thread: the newest lazy fence's tick (0: none yet).
    [[nodiscard]] u64 LazyOutstandingTick() const noexcept {
        return lazy_outstanding_tick;
    }
    /// F4-02, recording thread: work the guest may wait on was recorded (a fence's downloads or
    /// queries, a command list end); it goes out at the next idle hook or heuristic submission
    /// instead of a submission of its own.
    void WantSubmit() {
        submit_wanted = true;
        if (lazy_fences) [[unlikely]] {
            VideoCommon::V2::LandingFence::diag_submit_wanted.store(true,
                                                                    std::memory_order_relaxed);
        }
    }
    /// Recording thread (SYNC §11.6 item 7): DeferFlush's work under lazy fences, taken once as
    /// a strong hint by the flush heuristic's next pass boundary.
    [[nodiscard]] bool TakeDeferredHint() noexcept {
        return std::exchange(deferred_hint, false);
    }
    /// Recording thread (SYNC §7.3 fix 2): work the guest waits on (a list end) is open.
    [[nodiscard]] bool SubmitWantedNow() const noexcept {
        return submit_wanted;
    }
    /// Recording thread (SYNC_SUBMIT_SPEC §10): a command was recorded since the last submission
    /// (lazy-fence mode): the open tick is not empty.
    [[nodiscard]] bool RecordedSinceSubmit() const noexcept {
        return recorded_since_submit;
    }
    /// Recording thread (SYNC §14 #2/#3, MP §5.5 "strong hints after work the guest will read"):
    /// the open tick holds a fence carrying a guest-visible result (an operation landing at it,
    /// committed downloads). Cleared when the tick is submitted.
    [[nodiscard]] bool OpenTickHasGuestVisibleFence() const noexcept {
        return guest_visible_fence;
    }
    void NoteGuestVisibleFence() noexcept {
        guest_visible_fence = true;
    }
    /// The idle hook (the back goes idle, or another thread asked with RequestIdleHook): what
    /// FlushDeferred submits, plus with lazy fences any work they or WantSubmit wait for.
    void ServeIdleHook();
    /// The split's idle hook (RasterizerVulkan registers it): submits DeferFlush's work and an
    /// urgent request (rule W); open work no thread blocks on is left for rules B/C and the
    /// back's sleep point (SYNC_SUBMIT_SPEC §11, SubmitRequests::AtIdleHook).
    void ServeSplitIdleHook();
    /// Any thread, F4-02/F4-04: `tick` is submitted when this returns. On the split's back it
    /// submits; other threads ask the back (RequestLazyFlush + WaitSubmitted), or submit
    /// themselves once the back stopped.
    void EnsureSubmitted(u64 tick);
    /// Recording thread, F4-02: a thread asked for a tick that is still open.
    [[nodiscard]] bool SubmitRequested() const noexcept {
        return submit_requests.Open(master_semaphore->CurrentTick());
    }
    /// Recording thread, F4-X2: a thread blocks on the open tick (urgent request).
    [[nodiscard]] bool SubmitUrgent() const noexcept {
        return submit_requests.Urgent(master_semaphore->CurrentTick());
    }
    /// Threads other than the split's back: waits until tick is submitted, asking the back to
    /// submit deferred work (logged every 2 s while it waits). False only when the back thread
    /// is not running, so nothing will submit it: the caller then submits it itself.
    bool WaitSubmitted(u64 tick);
    /// Whether everything up to tick has been handed to the worker for submission.
    [[nodiscard]] bool IsSubmitted(u64 tick) const noexcept {
        return tick < master_semaphore->CurrentTick();
    }

    /// R2-F4-07, recording thread: a mapped (host-coherent) buffer was written by the GPU in the
    /// open submission; it ends with a host-read barrier.
    void WantHostReadBarrier() noexcept {
        host_read_wanted = true;
    }
    /// R2-F4-07: whether GPU writes recorded up to `tick` are host-visible once it completes (a
    /// later submission ended with the host-read barrier).
    [[nodiscard]] bool HostVisibleThrough(u64 tick) const noexcept {
        return tick != 0 && tick <= host_visible_tick;
    }

    /// Waits for the worker thread to finish executing everything. After this function returns it's
    /// safe to touch worker resources.
    void WaitWorker();

    void MarkFrameEnd(u64 frame);
    void TraceGraphicsPipeline(const std::array<u64, 6>& hashes);
    void TraceComputePipeline(u64 hash);
    /// Worker thread, inside a recorded command: timestamps one draw or dispatch of a sampled
    /// submission. A draw passes its vertex and fragment hashes, a dispatch its compute hash.
    template <typename Func>
    void TraceGpuWork(vk::CommandBuffer cmdbuf, u64 hash, u64 fragment_hash,
                      std::array<u32, 3> size, Func&& work) {
        if (!trace_gpu_recording || !trace_work_pool || trace_work_count == TraceMaxWork) {
            work();
            return;
        }
        const u32 query = trace_work_count * 2;
        trace_work_items[trace_work_count++] = {hash, fragment_hash, size};
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query);
        work();
        cmdbuf.WriteTimestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, *trace_work_pool, query + 1);
    }

    /// Sends currently recorded work to the worker thread.
    void DispatchWork();

    /// Requests to begin a renderpass.
    void RequestRenderpass(const Framebuffer* framebuffer,
                           std::source_location location = std::source_location::current());

    /// Game draws and clears: dynamic rendering with EDEN_NX2_V2_DYNRENDER (when the
    /// framebuffer has no resolve attachments), else RequestRenderpass. Pipelines made for
    /// dynamic rendering only run in such passes.
    void RequestRendering(const Framebuffer* framebuffer, u32 color_count = 0,
                          std::source_location location = std::source_location::current());

    /// EDEN_NX2_V2_DYNRENDER: a draw or dispatch is about to read `image` (guest range
    /// [begin, end)): pending attachment writes it overlaps get a barrier before its pass or
    /// dispatch.
    void NoteImageRead(VkImage image, u64 begin, u64 end) {
        // F4-12: nothing pending (the common case) needs no scan.
        if (track_barriers && !barrier_tracker.Empty()) [[unlikely]] {
            NoteImageReadTracked(image, begin, end);
        }
    }
    /// Records the read hazards noted so far (ends an active pass first). Dispatches call it
    /// before recording; render pass begins do it themselves.
    void FlushReadHazards();
    /// While set, recording outside a pass doesn't flush every pending write (the dispatch the
    /// caller checked with FlushReadHazards).
    void SetTargetedScope(bool active) {
        targeted_scope = active;
        UpdateAutoFlush();
    }
    /// Perf fix C (EDEN_NX2_V2_DYNRENDER, MASTER_PLAN §5.5 targeted barriers): records an
    /// operation outside any pass that touches only buffers and query pools (copies, fills,
    /// their own barriers). The tracker's pending writes are attachment writes to images, which
    /// such an operation can't read or overwrite (host buffers never alias host images; buffer
    /// hazards keep their own barriers), so it flushes none of them. Without the tracker: the
    /// old RequestOutsideRenderPassOperationContext + Record.
    template <typename T>
    void RecordBufferOperation(T&& command,
                               std::source_location location = std::source_location::current()) {
        if (!track_barriers) {
            RequestOutsideRenderPassOperationContext(location);
            Record(std::forward<T>(command));
            return;
        }
        const bool saved = BeginBufferOperation(location);
        Record(std::forward<T>(command));
        targeted_scope = saved;
        UpdateAutoFlush();
    }
    /// F4-06: an image the next out-of-pass operation reads or writes (guest range for aliasing).
    struct TrackedImage {
        VkImage image{};
        u64 begin{};
        u64 end{};
    };
    /// F4-06 (MASTER_PLAN §5.5 targeted barriers): the operation recorded next, outside any pass,
    /// touches only `images` (and buffers). Pending attachment writes overlapping them get one
    /// barrier; the others stay pending. The operation's own records run in the targeted scope
    /// until EndImageOperation (pass the returned value). Without the tracker: the old
    /// RequestOutsideRenderPassOperationContext.
    [[nodiscard]] bool RequestOutsideRenderPassForImages(
        std::initializer_list<TrackedImage> images,
        std::source_location location = std::source_location::current());
    void EndImageOperation(bool saved) {
        if (track_barriers) {
            targeted_scope = saved;
            UpdateAutoFlush();
        }
    }
    /// RecordBufferOperation's start with the tracker: ends the pass, opens the targeted scope
    /// and returns the scope's previous state.
    bool BeginBufferOperation(std::source_location location);
    /// Dispatches: ends a pass without flushing the tracked writes (the dispatch's own barrier
    /// orders its reads).
    void RequestOutsideRenderPassForDispatch(
        std::source_location location = std::source_location::current());
    /// F4-X3 (tracker on): the guest's wait-for-idle (V2::DecideWaitForIdle). `stages`: the
    /// source stages of its barrier (shader and transfer writes -> every later command).
    void WaitForIdleTracked(u32 stages,
                            std::source_location location = std::source_location::current());
    [[nodiscard]] bool TracksBarriers() const noexcept {
        return track_barriers;
    }
    /// F2-09: 0 while image reads can hazard with nothing; else the tracker's write serial.
    [[nodiscard]] u64 ReadHazardSerial() const noexcept {
        return !track_barriers || barrier_tracker.Empty() ? 0 : barrier_tracker.WriteSerial();
    }

    /// Defers a full-attachment color clear so it becomes the next render pass.
    bool DeferColorClear(const Framebuffer* framebuffer, u32 rt_slot, const VkClearValue& value);

    /// Defers a full depth/stencil clear so it becomes the next render pass.
    bool DeferDepthStencilClear(const Framebuffer* framebuffer, const VkClearValue& value);

    /// Realizes any pending deferred clear before its framebuffer can be moved or freed.
    void FlushDeferredClear();

    /// Requests the current execution context to be able to execute operations only allowed outside
    /// of a renderpass.
    void RequestOutsideRenderPassOperationContext(
        std::source_location location = std::source_location::current());

    void DeferFragmentBarrier() {
        fragment_barrier.Defer(state.renderpass != VK_NULL_HANDLE);
    }

    bool HasFragmentBarrier() const {
        return fragment_barrier.Pending();
    }

    void NoteRenderPassStores() {
        render_pass_has_stores = true;
    }

    bool RenderPassHasStores() const {
        return render_pass_has_stores;
    }

    /// A dispatch was recorded: the next render pass must see what it wrote.
    void NoteComputeWrites() {
        compute_writes_pending = true;
    }

    /// F4-07: transform feedback began; the pass that ends next records the XFB barrier.
    void NoteTransformFeedback() {
        pass_transform_feedback = true;
    }

    /// Returns true when a render pass is currently active in the scheduler state.
    bool IsRenderPassActive() const {
        return state.renderpass != VK_NULL_HANDLE;
    }

    /// Chunk B2 (EDEN_NX2_V2_DYNRENDER): whether a clear of `framebuffer`'s colour slot and/or
    /// depth-stencil can be a vkCmdClearAttachments in the active pass: those views are the
    /// pass's own attachments at the same slots and the rect and layers lie inside it.
    [[nodiscard]] bool CanClearInPass(const Framebuffer* framebuffer,
                                      std::optional<u32> color_slot, bool depth_stencil,
                                      const VkClearRect& rect) const;

    /// Chunk B3: a full clear of attachments the open pass doesn't hold ends it (its own pass
    /// begin would), so that the clear becomes the next draw's load op.
    void EndRenderPassForClear(std::source_location location = std::source_location::current());

    /// Changes whenever a render pass begins or ends.
    u64 RenderPassSerial() const {
        return render_pass_serial;
    }

    /// The graphics pipeline bound in the current command buffer, if the scheduler knows it.
    const GraphicsPipeline* BoundGraphicsPipeline() const {
        return state.graphics_pipeline;
    }

    /// Update the pipeline to the current execution context.
    bool UpdateGraphicsPipeline(GraphicsPipeline* pipeline);

    /// Worker thread only: whether the bound graphics pipeline failed to build, so the draws
    /// recorded after it must be dropped instead of reaching the driver without a pipeline.
    void SetGraphicsPipelineMissing(bool missing) {
        graphics_pipeline_missing = missing;
    }
    [[nodiscard]] bool GraphicsPipelineMissing() const {
        return graphics_pipeline_missing;
    }

    /// Update the rescaling state. Returns true if the state has to be updated.
    bool UpdateRescaling(bool is_rescaling, bool mesh_layout);

    /// Returns true when the descriptor buffer chunk has to be bound into the command buffer.
    bool UpdateDescriptorBufferChunk(u32 descriptor_chunk);

    /// Invalidates current command buffer state except for render passes
    void InvalidateState();

    /// Assigns the query cache.
    void SetQueryCache(VideoCommon::QueryCacheBase<QueryCacheParams>& query_cache_) {
        query_cache = &query_cache_;
    }

    /// Recording thread: runs at the start of every submission's closing (P5b: the GPU decode
    /// batch goes into the closing submission's upload command buffer).
    void RegisterPreSubmit(std::function<void()>&& func) {
        pre_submit = std::move(func);
    }

    // Registers a callback to perform on queue submission.
    void RegisterOnSubmit(std::function<void()>&& func) {
        on_submit = std::move(func);
    }

    /// Send work to a separate thread.
    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer, vk::CommandBuffer>
    void RecordWithUploadBuffer(T&& command,
                                std::source_location location = std::source_location::current()) {
        if (VideoCommon::GpuSplit::check_owner) [[unlikely]] {
            VideoCommon::GpuSplit::CheckOwner("Scheduler::Record");
        }
        if (auto_flush) [[unlikely]] {
            // EDEN_NX2_V2_DYNRENDER: an untracked operation outside a pass sees every pending
            // attachment write (what the old pass-end barrier gave it). F4-06: counted at the
            // recording call site (STATS full barriers by site).
            FlushAllPendingWrites(location);
        }
        if (lazy_fences && !recorded_since_submit) [[unlikely]] {
            // SYNC_SUBMIT_SPEC §11 invariant S: the open tick holds work (the back may not sleep
            // on it with fewer than two submissions in flight).
            NoteOpenTickWork();
        }
        if (chunk->Record(command)) {
            return;
        }
        DispatchWork();
        (void)chunk->Record(command);
    }

    /// A command that touches no memory (dynamic state): pending attachment writes need no
    /// barrier for it, inside a pass or not (no auto flush).
    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer>
    void RecordState(T&& c) {
        const bool saved = std::exchange(auto_flush, false);
        Record(std::forward<T>(c));
        auto_flush = saved;
    }

    /// A command recorded only into the upload command buffer, which runs before the whole
    /// submission: no pending write of the main stream can be ordered before it (no auto flush).
    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer, vk::CommandBuffer>
    void RecordUpload(T&& command) {
        const bool saved = std::exchange(auto_flush, false);
        RecordWithUploadBuffer(std::forward<T>(command));
        auto_flush = saved;
    }

    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer>
    void Record(T&& c, std::source_location location = std::source_location::current()) {
        this->RecordWithUploadBuffer(
            [command = std::move(c)](vk::CommandBuffer cmdbuf, vk::CommandBuffer) {
                command(cmdbuf);
            },
            location);
    }

    /// P6 Step 0 (DRAWREC=verify purity): moves with every recorded command.
    [[nodiscard]] u64 RecordMark() const noexcept {
        return (reinterpret_cast<std::uintptr_t>(chunk.get()) * 0x9E3779B97F4A7C15ULL) ^
               reinterpret_cast<std::uintptr_t>(chunk->LastCommand());
    }

    /// P5b (EDEN_NX2_V2_XFERQ): whether the device's dedicated transfer queue is in use.
    [[nodiscard]] bool HasTransferQueue() const noexcept {
        return transfer_queue != nullptr;
    }
    [[nodiscard]] TransferQueue* GetTransferQueue() const noexcept {
        return transfer_queue.get();
    }

    /// Recording thread: whether a transfer op on a resource last used by graphics tick `track`
    /// may run on the transfer queue ahead of the open submission (DXVK's rule). When true the
    /// op must be recorded with RecordTransfer before the next Flush.
    bool TryHoistTransfer(u64 track);
    /// Perf fix C: a buffer upload into a region no in-flight work uses: it reads no GPU data
    /// and writes nothing queued work touches, so its batch waits for no graphics tick (no
    /// class-B floor) and runs while earlier submissions execute.
    bool TryHoistBufferUpload();

    /// Records a closure into the open submission's transfer command buffer (worker thread).
    template <typename T>
        requires std::is_invocable_v<T, vk::CommandBuffer>
    void RecordTransfer(T&& c) {
        Record([this, command = std::move(c)](vk::CommandBuffer) {
            command(WorkerTransferCmdbuf());
        });
    }

    /// Recording thread: an acquire's position executed (class A/B with the transfer queue on):
    /// later transfer work must wait for the graphics work before it (MASTER_PLAN §6).
    void NoteAcquireJoin();

    /// Recording thread: the next graphics submission waits for `value` on `semaphore`.
    bool AddGraphicsWait(VkSemaphore semaphore, u64 value);
    /// Recording thread (Perf fix C): the submit path queues `cmdbuf` on the transfer queue right
    /// after the open submission, waiting for its tick and signalling `value` on `semaphore`.
    /// False without a submit engine or with every slot taken (the caller submits it itself).
    bool AddTransferJob(VkCommandBuffer cmdbuf, VkSemaphore semaphore, u64 value);

    /// Stops hoisting to the transfer queue until the next submission.
    void BlockTransferHoisting() {
        timeline_join.BlockHoisting();
    }

    /// Any thread: with the submit thread, waits until every handed submission reached the
    /// queue (before waiting on a binary semaphore such a submission signals).
    void WaitSubmitThread();

    /// Any thread: runs `func` once the GPU finished `tick` (the finish thread with
    /// EDEN_NX2_V2_SUBMITTHREAD, else the worker or the caller).
    void RetireAfter(u64 tick, std::function<void()>&& func);

    [[nodiscard]] SubmitEngine* GetSubmitEngine() const noexcept {
        return submit_engine.get();
    }

    /// F4-12: the finish thread (EDEN_NX2_V2_SUBMITTHREAD) publishes GPU progress as it sees
    /// it, so KnownGpuTick is current without asking the driver (MasterSemaphore::Refresh).
    [[nodiscard]] bool FinishThreadPublishesTick() const noexcept {
        return finish_publishes_tick;
    }

    /// EDEN_NX2_V2_XREADBACK: readbacks on the transfer queue (null when off or unavailable).
    [[nodiscard]] BackgroundReadback* GetBackgroundReadback() const noexcept {
        return background_readback.get();
    }

    /// Whether the recording thread submitted `tick` within 20 ms.
    [[nodiscard]] bool WaitForSubmission(u64 tick) const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{20};
        while (tick >= master_semaphore->CurrentTick()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds{100});
        }
        return true;
    }

    /// Returns the current command buffer tick.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return master_semaphore->CurrentTick();
    }

    /// Changes whenever recorded command buffer state may have been reset (InvalidateState, the
    /// end of a render pass), so a binding recorded at the same epoch is still bound.
    [[nodiscard]] u64 RecordingEpoch() const noexcept {
        return recording_epoch;
    }

    /// Returns true when a tick has been triggered by the GPU.
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return master_semaphore->IsFree(tick);
    }

    /// Waits for the given GPU tick, optionally pacing frames. Threads other than the one that
    /// records (the async present thread) pass may_flush = false: flushing from there records
    /// into the current chunk while that thread does (corrupt command buffers, crashes). They
    /// wait for the recording thread to submit the tick instead, and give up after 20 ms (it may
    /// be waiting for them).
    void Wait(u64 tick, double target_fps = 0.0, bool may_flush = true) {
        const bool trace = Common::DkbFrameTrace::Enabled() && target_fps > 0.0;
        const auto trace_start = trace ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        if (tick > 0 && (may_flush || WaitForSubmission(tick))) {
            if (tick >= master_semaphore->CurrentTick()) {
                if (lazy_fences) [[unlikely]] {
                    VideoCommon::V2::P5bStats::Add(
                        VideoCommon::V2::P5bStats::Counter::SubmitsWaiter); // blocks on it: W
                }
                Flush();
            }
            master_semaphore->Wait(tick);
        }
        const auto trace_waited = trace ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
        double trace_requested_ms{};
        double trace_sleep_ms{};
        double trace_slept_ms{};
        if (Settings::values.use_speed_limit.GetValue() && target_fps > 0.0) {
            auto now = std::chrono::steady_clock::now();
            if (last_target_fps != target_fps) {
                frame_interval = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / target_fps));
                max_frame_count = static_cast<int>(0.1 * target_fps);
                last_target_fps = target_fps;
                frame_counter = 0;
                start_time = now;
            }
            frame_counter++;
            auto target_time = start_time + frame_interval * frame_counter;
            if (target_time >= now) {
                constexpr auto spin_tail = std::chrono::milliseconds(1);
                auto sleep_time = target_time - now;
                if (trace) {
                    trace_requested_ms =
                        std::chrono::duration<double, std::milli>(sleep_time).count();
                }
                if (sleep_time > spin_tail * 2) {
                    if (trace) {
                        trace_sleep_ms =
                            std::chrono::duration<double, std::milli>(sleep_time - spin_tail).count();
                    }
                    const auto sleep_start = trace ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
                    std::this_thread::sleep_for(sleep_time - spin_tail);
                    if (trace) {
                        trace_slept_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - sleep_start).count();
                    }
                }
                while (std::chrono::steady_clock::now() < target_time) {
                    std::this_thread::yield();
                }
            } else if (frame_counter > max_frame_count) {
                frame_counter = 0;
                start_time = now;
            }
        }
        if (trace) {
            static thread_local Common::DkbFrameTrace::RateLimit rate;
            const auto end = std::chrono::steady_clock::now();
            if (rate.Ready(end)) {
                LOG_INFO(Render_Vulkan,
                         "DKB host_pace target_fps={} enabled={} may_flush={} "
                         "cpu_wait_wall_ms={:.3f} pace_requested_ms={:.3f} "
                         "sleep_requested_ms={:.3f} sleep_actual_wall_ms={:.3f} "
                         "pace_actual_wall_ms={:.3f}",
                         target_fps, Settings::values.use_speed_limit.GetValue(), may_flush,
                         std::chrono::duration<double, std::milli>(trace_waited - trace_start).count(),
                         trace_requested_ms, trace_sleep_ms, trace_slept_ms,
                         std::chrono::duration<double, std::milli>(end - trace_waited).count());
            }
        }
    }

    /// Returns the master timeline semaphore.
    [[nodiscard]] MasterSemaphore& GetMasterSemaphore() const noexcept {
        return *master_semaphore;
    }

    std::mutex submit_mutex;

private:
    [[nodiscard]] static u64 NowUs() noexcept {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }
    /// F4-X2, recording thread, before a submission: adds the time blocked threads waited for it.
    void NoteBlockedWait();

    std::chrono::steady_clock::time_point last_flush{};
    bool flush_pending{};
    VideoCommon::V2::SubmitRequests submit_requests; ///< RequestLazyFlush
    std::atomic<u64> blocked_since_us{}; ///< F4-X2: first urgent request since the last submit
    std::atomic<u64> blocked_wait_us{};  ///< F4-X2: TakeBlockedWaitUs
    bool host_read_wanted{};             ///< R2-F4-07: WantHostReadBarrier
    std::array<VkImageView, 8> active_color_views{}; ///< CanClearInPass: the open pass's views
    VkImageView active_depth_view{};
    u32 active_layers{};
    u64 host_visible_tick{};             ///< R2-F4-07: HostVisibleThrough
    bool lazy_fences{};                 ///< F4-02: SetLazyFences
    bool finish_publishes_tick{};       ///< F4-12: FinishThreadPublishesTick
    bool pass_transform_feedback{};     ///< F4-07: NoteTransformFeedback
    u32 wfi_at_pass_end{};              ///< F4-X3: barrier stages of a WFI in the active pass
    bool submit_wanted{};             ///< F4-02: work waits for the next idle hook or boundary
    u64 lazy_outstanding_tick{};        ///< F4-02: newest lazy fence's tick (recording thread)
    /// SYNC Step 9: DeferToPassEnd's jobs, in order.
    VideoCommon::V2::AfterPassList after_pass;
    void RunAfterPass() {
        after_pass.Run();
    }
    u64 open_since_us{}; ///< SYNC §7.3: when the open tick first held a lazy fence (0: none)
    bool recorded_since_submit{}; ///< SYNC §10: RecordedSinceSubmit (recording thread)
    bool guest_visible_fence{};   ///< SYNC §14: OpenTickHasGuestVisibleFence (recording thread)
    u64 open_work_since_ns{};     ///< SYNC §14 #5: the open tick's first recorded work (trace)
    [[nodiscard]] static u64 TraceNowNs() {
        return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count());
    }
    bool deferred_hint{};         ///< SYNC §11.6 item 7: TakeDeferredHint
    std::atomic<bool> open_tick_nonempty{}; ///< ... for the finish thread's progress wake
    /// SYNC §7.3 counters and the 2 s line's state, at a lazy-fence-mode submission.
    void NoteOpenTickSubmitted() {
        if (open_since_us != 0) {
            const u64 age_us = NowUs() - open_since_us;
            VideoCommon::V2::P5bStats::Max(VideoCommon::V2::P5bStats::Counter::OpenTickAgeMaxUs,
                                           age_us);
            if (VideoCommon::V2::FrameCauses::Enabled()) [[unlikely]] {
                VideoCommon::V2::FrameCauses::Note(VideoCommon::V2::FrameCauses::Cause::OpenTickAge,
                                                   age_us * 1000);
            }
            open_since_us = 0;
        }
        VideoCommon::V2::LandingFence::diag_submit_wanted.store(false, std::memory_order_relaxed);
    }
    /// SYNC §11.2: the open tick holds work (a command or a lazy fence).
    void NoteOpenTickWork() {
        if (!recorded_since_submit && trace_query_pool) [[unlikely]] {
            open_work_since_ns = TraceNowNs(); // SYNC §14 #5: the idle-gap split
        }
        recorded_since_submit = true;
        open_tick_nonempty.store(true, std::memory_order_seq_cst);
        VideoCommon::V2::LandingFence::diag_recorded.store(true, std::memory_order_relaxed);
    }
    /// SYNC §10: after SubmitExecution (whatever it recorded to close the command buffer went
    /// with it), the new open tick is empty.
    void NoteOpenTickEmpty() {
        recorded_since_submit = false;
        guest_visible_fence = false;
        open_tick_nonempty.store(false, std::memory_order_seq_cst);
        VideoCommon::V2::LandingFence::diag_recorded.store(false, std::memory_order_relaxed);
    }
    void (*pre_submit_hook)(void*){};   ///< SYNC Step 8: SetPreSubmitHook
    void* pre_submit_argument{};
    bool in_pre_submit{};               ///< The hook runs (it may record, never resubmit)
    void RunPreSubmitHook() {
        if (pre_submit_hook != nullptr && !in_pre_submit) [[unlikely]] {
            in_pre_submit = true;
            pre_submit_hook(pre_submit_argument);
            in_pre_submit = false;
        }
    }

    class Command {
    public:
        virtual ~Command() = default;

        virtual void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const = 0;

        Command* GetNext() const {
            return next;
        }

        void SetNext(Command* next_) {
            next = next_;
        }

    private:
        Command* next = nullptr;
    };

    template <typename T>
    class TypedCommand final : public Command {
    public:
        explicit TypedCommand(T&& command_) : command{std::move(command_)} {}
        ~TypedCommand() override = default;

        TypedCommand(TypedCommand&&) = delete;
        TypedCommand& operator=(TypedCommand&&) = delete;

        void Execute(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf) const override {

            command(cmdbuf, upload_cmdbuf);
        }

    private:
        T command;
    };

    class CommandChunk final {
    public:
        void ExecuteAll(vk::CommandBuffer cmdbuf, vk::CommandBuffer upload_cmdbuf);

        template <typename T>
        bool Record(T& command) {
            using FuncType = TypedCommand<T>;
            static_assert(sizeof(FuncType) < sizeof(data), "Lambda is too large");

            command_offset = Common::AlignUp(command_offset, alignof(FuncType));
            if (command_offset > sizeof(data) - sizeof(FuncType)) {
                return false;
            }
            Command* const current_last = last;
            last = new (data.data() + command_offset) FuncType(std::move(command));

            if (current_last) {
                current_last->SetNext(last);
            } else {
                first = last;
            }
            command_offset += sizeof(FuncType);
            return true;
        }

        void MarkSubmit() {
            submit = true;
        }

        bool Empty() const {
            return command_offset == 0;
        }

        /// The last recorded command (DRAWREC=verify: moves with every record).
        const void* LastCommand() const {
            return last;
        }

        bool HasSubmit() const {
            return submit;
        }

    private:
        Command* first = nullptr;
        Command* last = nullptr;

        size_t command_offset = 0;
        bool submit = false;
        alignas(std::max_align_t) std::array<u8, 0x8000> data{};
    };

    struct State {
        VkRenderPass renderpass{};
        VkFramebuffer framebuffer{};
        VkExtent2D render_area = {0, 0};
        GraphicsPipeline* graphics_pipeline = nullptr;
        bool is_rescaling = false;
        bool rescaling_defined = false;
        bool mesh_layout = false;
        bool needs_state_enable_refresh = false;
        u32 descriptor_buffer_chunk = 0;
        bool descriptor_buffer_bound = false;
        bool dynamic = false; ///< the active pass is dynamic rendering (EDEN_NX2_V2_DYNRENDER)
    };

    struct DeferredClear {
        const Framebuffer* framebuffer = nullptr;
        u32 color_clear_mask = 0;
        std::array<VkClearValue, 8> color_values{};
        bool depth_stencil = false;
        VkClearValue depth_stencil_value{};
    };

    /// EDEN_NX2_V2_DYNRENDER
    void BeginRenderingImpl(const Framebuffer* framebuffer, u32 color_count, u32 color_clear_mask,
                            const std::array<VkClearValue, 8>& color_values, bool depth_clear,
                            const VkClearValue& depth_value);
    void NoteImageReadTracked(VkImage image, u64 begin, u64 end);
    /// `location`: who asked (F4-06 stats: full barriers by site; Record's auto flush shows as
    /// the header line).
    void FlushAllPendingWrites(std::source_location location = std::source_location::current());
    void RecordHazardBarrier(u32 src_stages, u32 src_access, u32 dst_stages, u32 dst_access);
    /// The barrier without the targeted-barrier count (a full barrier counts as that).
    void RecordHazardBarrierUncounted(u32 src_stages, u32 src_access, u32 dst_stages,
                                      u32 dst_access);
    /// F4-X3: the WFI barrier, with every pending tracked write folded into it.
    void RecordWaitForIdleBarrier(u32 stages);
    /// F4-X2, finish thread (SubmitEngine progress hook): wakes the idle hook for deferred work.
    void OnGpuProgress();

    void UpdateAutoFlush() {
        auto_flush = track_barriers && !targeted_scope && !state.renderpass &&
                     !barrier_tracker.Empty();
    }

    /// Begins a render pass for the given framebuffer, optionally with clear values.
    void BeginRenderPassImpl(const Framebuffer* framebuffer, VkRenderPass renderpass,
                             const VkClearValue* clear_values, u32 clear_value_count);

    /// If a deferred clear is pending.
    void RealizeDeferredClear();

    void WorkerThread(std::stop_token stop_token);

    void AllocateWorkerCommandBuffer();

    void TraceGpuSubmission();
    void ReportWorkTimes();
    void TraceShader(u64 hash, u32 stage);

    u64 SubmitExecution(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore);

    void SubmitExecutionEngine(VkSemaphore signal_semaphore, VkSemaphore wait_semaphore,
                               u64 signal_value);

    /// Worker thread: the open submission's transfer command buffer, begun on first use.
    vk::CommandBuffer WorkerTransferCmdbuf();

    void AllocateNewContext();

    void EndPendingOperations();

    void EndRenderPass();

    void AcquireNewChunk();

    const Device& device;
    StateTracker& state_tracker;

    std::unique_ptr<MasterSemaphore> master_semaphore;
    FragmentBarrierState fragment_barrier;
    bool render_pass_has_stores{};
    bool compute_writes_pending{};
    bool graphics_pipeline_missing{};
    u64 render_pass_serial{};
    std::unique_ptr<CommandPool> command_pool;

    vk::QueryPool trace_query_pool;
    static constexpr u32 TraceMaxSubmissions = 512;
    struct SubmissionTrace {
        struct ShaderUse {
            u64 hash{};
            u32 stage{};
        };
        u64 tick{};
        u32 draw_attempts{};
        u32 dispatch_attempts{};
        u32 num_shaders{};
        u32 untracked_uses{};
        std::array<ShaderUse, 32> shaders{};
        /// SYNC §14 #5 (host steady clock, ns): when the tick first held recorded work
        /// (lazy fences; 0 unknown) and when it went to the queue.
        u64 recorded_ns{};
        u64 submitted_ns{};
    };
    SubmissionTrace trace_work;
    std::vector<SubmissionTrace> trace_submissions;
    static constexpr u32 TraceMaxWork = 4096;
    struct WorkTrace {
        u64 hash{};
        u64 fragment_hash{};
        /// Dispatch grid, or vertex and instance counts of a draw; zero when indirect.
        std::array<u32, 3> size{};
    };
    vk::QueryPool trace_work_pool;
    std::vector<WorkTrace> trace_work_items;
    u32 trace_work_count{};
    Common::DkbFrameTrace::RateLimit trace_gpu_rate;
    u64 trace_gpu_tick{};
    u64 trace_frame_end{};
    u64 trace_sample_frame{};
    u32 trace_gpu_count{};
    u64 trace_gpu_skipped{};
    u32 trace_timestamp_bits{};
    float trace_timestamp_period{};
    bool trace_gpu_recording{};
    bool trace_gpu_sampling{};
    bool trace_gpu_pending{};
    bool trace_gpu_failed{};

    VideoCommon::QueryCacheBase<QueryCacheParams>* query_cache = nullptr;

    vk::CommandBuffer current_cmdbuf;
    vk::CommandBuffer current_upload_cmdbuf;

    DeferredClear deferred_clear;

    std::unique_ptr<CommandChunk> chunk;
    std::function<void()> on_submit;
    std::function<void()> pre_submit;

    State state;
    u64 recording_epoch = 0;

    u32 num_renderpass_images = 0;
    std::array<VkImage, 9> renderpass_images{};
    std::array<VkImageSubresourceRange, 9> renderpass_image_ranges{};

    /// P5b: the transfer queue (EDEN_NX2_V2_XFERQ) and the submission engine
    /// (EDEN_NX2_V2_SUBMITTHREAD, or contexts for the transfer queue). Destroyed after the worker.
    std::unique_ptr<TransferQueue> transfer_queue;
    std::unique_ptr<SubmitEngine> submit_engine;
    std::unique_ptr<BackgroundReadback> background_readback;
    SubmitContext* current_context{}; ///< worker thread
    VideoCommon::V2::TimelineJoin timeline_join; ///< recording thread
    /// EDEN_NX2_V2_DYNRENDER: dynamic rendering for game passes, and pending attachment writes
    /// tracked by resource and guest range instead of the all-attachment pass-end barrier.
    bool dynamic_rendering = false;
    bool track_barriers = false;
    bool targeted_scope = false;
    bool auto_flush = false;
    VideoCommon::V2::RangeBarrierTracker barrier_tracker;
    VideoCommon::V2::RangeBarrierTracker::Hazard read_hazard{};
    std::array<std::pair<u64, u64>, 9> renderpass_guest_ranges{};
    u32 pass_draws = 0;
    bool next_pass_dynamic = false;
    u32 dynamic_color_count = 0; ///< colorAttachmentCount of the active dynamic pass

    std::queue<std::unique_ptr<CommandChunk>> work_queue;
    std::vector<std::unique_ptr<CommandChunk>> chunk_reserve;
    std::mutex execution_mutex;
    std::mutex reserve_mutex;
    std::mutex queue_mutex;
    std::condition_variable_any event_cv;
    bool worker_waiting{}; ///< The worker is waiting for work (guarded by queue_mutex)
    std::jthread worker_thread;

    std::chrono::steady_clock::duration frame_interval{};
    std::chrono::steady_clock::time_point start_time{};
    double last_target_fps{};
    u64 max_frame_count{};
    u64 frame_counter{};
};

} // namespace Vulkan
