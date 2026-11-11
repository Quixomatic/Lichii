// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>

#include "common/common_types.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/gpu.h"
#include "video_core/host1x/host1x.h"
#include "video_core/host1x/syncpoint_manager.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/nx2_profile.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/v2/commit.h"

namespace VideoCommon {

class FenceBase {
public:
    explicit FenceBase(bool is_stubbed_) : is_stubbed{is_stubbed_} {}

    bool IsStubbed() const {
        return is_stubbed;
    }

protected:
    bool is_stubbed;
};

template <typename Traits>
class FenceManager {
    using TFence = typename Traits::FenceType;
    using TTextureCache = typename Traits::TextureCacheType;
    using TBufferCache = typename Traits::BufferCacheType;
    using TQueryCache = typename Traits::QueryCacheType;
    static constexpr bool can_async_check = Traits::HAS_ASYNC_CHECK;

public:
    /// Notify the fence manager about a new frame
    void TickFrame() {
        std::unique_lock lock(ring_guard);
        delayed_destruction_ring.Tick();
    }

    // Unlike other fences, this one doesn't
    void SignalOrdering() {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        std::scoped_lock lock{buffer_cache.mutex};
        buffer_cache.AccumulateFlushes();
    }

    void SignalReference() {
        std::function<void()> do_nothing([] {});
        // No guest sync rides on it: with deferred writebacks it commits no downloads.
        SignalFence(std::move(do_nothing), false);
    }

    /// Pipeline v2 P5a (EDEN_NX2_V2_WRITEBACK): GPU data owed to guest memory (the caches'
    /// uncommitted downloads) is committed only at fences a guest sync rides on, or at frame end,
    /// instead of at every fence.
    void SetDeferredWritebacks(bool on) {
        defer_writebacks = on;
    }

    void SyncOperation(std::function<void()>&& func) {
        // Not only the GPU thread queues these (query cache work runs elsewhere too), while
        // SignalFence moves the list into pending_operations: unguarded, the list was corrupted
        // and the fencing thread crashed walking it. Its own lock, held for no callbacks.
        std::scoped_lock lock{uncommitted_guard};
        uncommitted_operations.emplace_back(std::move(func));
        has_uncommitted.store(true, std::memory_order_release);
    }

    /// SYNC_SUBMIT_SPEC §9.3: operations queued with no fence after them yet (one load).
    [[nodiscard]] bool HasUncommittedOperations() const {
        return has_uncommitted.load(std::memory_order_acquire);
    }

    /// `observable`: a guest sync rides on the fence (a syncpoint, a release, a report): GPU data
    /// owed to guest memory must land before it. Only deferred writebacks look at it.
    void SignalFence(std::function<void()>&& func, bool observable = true) {
        Nx2Profile::Scope profile_scope{Nx2Profile::Zone::Fence};
        // A fence that writes a fence-flagged query report is never skipped as empty: a skipped
        // fence runs its operation at once, before the host GPU has run the work the report
        // follows, and Ocarina of Time then reads a wrong report (white flashes on leaves and walls
        // at low GPU accuracy).
        const bool delay_fence = VideoCore::g_signaling_query_report_fence ||
                                 (Settings::IsGPUFenceBehaviorDefault()
                                      ? Settings::IsGPULevelHigh()
                                      : Settings::IsGPUFenceBehaviorBalanced() ||
                                            Settings::IsGPUFenceBehaviorAccurate());
        if constexpr (!can_async_check) {
            TryReleasePendingFences<false>();
        }
        // EDEN_NX2_V2_WRITEBACK: the owed downloads wait for a fence the guest can observe.
        const bool defer = defer_writebacks && !observable;
        if (defer &&
            (texture_cache.HasAsyncFlushWork() || buffer_cache.HasUncommittedFlushes()))
            [[unlikely]] {
            V2::Commit::Stats::Count(V2::Commit::Stats::Get().writeback_queued);
        }
        auto [should_flush, committed_flushes, early_only] =
            defer ? CacheFlushes{} : CommitCacheFlushes();
        if (Nx2Profile::Enabled()) {
            NoteFenceDecision(delay_fence, committed_flushes);
        }
        if (!delay_fence && committed_flushes == 0 && SkipEmptyFence()) {
            Nx2Profile::Scope skipped_scope{Nx2Profile::Zone::FenceSkipped};
            func();
            rasterizer.InvalidateGPUCache();
            return;
        }
        skipped_fences = 0;
        const bool downloads = should_flush;
        should_flush |= query_cache.HasUncommittedFlushes();
        query_cache.CommitAsyncFlushes();
        // EDEN_NX2_V2_READBACK early downloads alone (MASTER_PLAN §6 class C): recorded now,
        // they go to the GPU with the next submission; the fence waits for it instead of making
        // one (a lazy fence: no flush here or in Queue).
        const bool lazy = !should_flush && early_only;
        TFence new_fence = CreateFence(!should_flush && !lazy);
        queue_lazily = lazy;
        if constexpr (can_async_check) {
            guard.lock();
        }
        bool carries_results = delay_fence || committed_flushes != 0;
        {
            std::scoped_lock lock{uncommitted_guard};
            carries_results |= !uncommitted_operations.empty();
            if (delay_fence) {
                uncommitted_operations.emplace_back(std::move(func));
            }
            pending_operations.emplace_back(std::move(uncommitted_operations));
            uncommitted_operations.clear(); // moved-from: make it empty explicitly
            uncommitted_operations.reserve(64);
            has_uncommitted.store(false, std::memory_order_release);
        }
        pending_flushes.push_back(committed_flushes);
        QueueFence(new_fence);
        if (!delay_fence) {
            func();
        }
        fences.push(std::move(new_fence));
        if (should_flush) {
            rasterizer.FlushCommandsForFence(downloads);
        }
        if (carries_results) {
            rasterizer.NoteFenceResults();
        }
        if constexpr (can_async_check) {
            guard.unlock();
            cv.notify_all();
        }
        if (delay_fence) {
            SubmitDeferredWork();
        }
        rasterizer.InvalidateGPUCache();
    }

    void SignalSyncPoint(u32 value) {
        // The id comes from a command list (the 3D engine's sync method, the puller's fence
        // action). Past the syncpoint table, the increment would walk memory after it as an
        // action list; command words read wrong carry any id (Splatoon Raiders on Linux).
        if (value >= Tegra::Host1x::SyncpointManager::NUM_MAX_SYNCPOINTS) {
            LOG_ERROR(HW_GPU, "Signal of syncpoint {} past the table, ignored", value);
            return;
        }
        syncpoint_manager.IncrementGuest(value);
        std::function<void()> func([this, value] { syncpoint_manager.IncrementHost(value); });
        SignalFence(std::move(func));
    }

    /// Pipeline v2 P5a (EDEN_NX2_V2_COMMIT without the cache locks): the fencing thread no longer
    /// pops the caches' async flushes. The pops of a released batch go to the caches' owner,
    /// which lands them at its next commit point (`wake` gets an idle owner there); the fencing
    /// thread waits for that and runs the operations itself, in fence order (Perf fix C).
    void SetCommitBacklog(bool on, std::function<void()> wake) {
        commit_backlog = on;
        wake_owner = std::move(wake);
    }

    /// One relaxed load: whether the owner has fence work waiting.
    [[nodiscard]] bool BacklogPending() const {
        return backlog_pending.load(std::memory_order_acquire);
    }

    /// The caches' owner only: pops and runs, in fence order, what the fencing thread handed
    /// over (deferred writebacks land here, before the operations that make them guest-visible).
    /// Perf fix C: the owner lands only the cache pops (the cache structures are its own); the
    /// fencing thread waits for that and then runs the queries and operations itself, in fence
    /// order (I5: a fence's pops precede its operations). Batches without pops never come here.
    void ServeBacklog() {
        if (!backlog_pending.load(std::memory_order_acquire)) {
            return;
        }
        std::deque<BacklogEntry> taken;
        {
            std::scoped_lock lock{backlog_mutex};
            taken.swap(backlog);
            backlog_pending.store(false, std::memory_order_release);
        }
        if (!taken.empty()) {
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            for (BacklogEntry& entry : taken) {
                if ((entry.flushes & TextureFlushes) != 0) {
                    texture_cache.PopAsyncFlushes();
                }
                if ((entry.flushes & BufferFlushes) != 0) {
                    buffer_cache.PopAsyncFlushes();
                }
            }
        }
        bool wake;
        {
            std::scoped_lock lock{backlog_mutex};
            backlog_served.store(backlog_served.load(std::memory_order_relaxed) + taken.size(),
                                 std::memory_order_release);
            wake = backlog_waiting;
        }
        if (wake) {
            backlog_cv.notify_one();
        }
    }

    void WaitPendingFences([[maybe_unused]] bool force) {
        if constexpr (!can_async_check) {
            TryReleasePendingFences<true>();
        } else {
            if (!force) {
                return;
            }
            std::mutex wait_mutex;
            std::condition_variable wait_cv;
            std::atomic<bool> wait_finished{};
            std::function<void()> func([&] {
                std::scoped_lock lk(wait_mutex);
                wait_finished.store(true, std::memory_order_relaxed);
                wait_cv.notify_all();
            });
            SignalFence(std::move(func), false);
            // The fence's commands can be held back by a deferred flush that only this thread
            // submits. The fencing thread waits for that submission before releasing the fence,
            // so waiting here without submitting deadlocks (Ocarina of Time entering gameplay).
            SubmitBeforeWait();
            std::unique_lock lk(wait_mutex);
            if (commit_backlog) [[unlikely]] {
                // The fencing thread may be waiting for this owner to land a fence's downloads
                // before it runs the operations: serve it.
                while (!wait_cv.wait_for(lk, std::chrono::microseconds(100), [&wait_finished] {
                    return wait_finished.load(std::memory_order_relaxed);
                })) {
                    lk.unlock();
                    rasterizer.ServeCommitRequests();
                    lk.lock();
                }
                return;
            }
            wait_cv.wait(
                lk, [&wait_finished] { return wait_finished.load(std::memory_order_relaxed); });
        }
    }

protected:
    /// The fence being queued is lazy (see SignalFence): the backend records its wait point
    /// without submitting.
    bool queue_lazily = false;

    explicit FenceManager(VideoCore::RasterizerInterface& rasterizer_, Tegra::GPU& gpu_,
                          TTextureCache& texture_cache_, TBufferCache& buffer_cache_,
                          TQueryCache& query_cache_)
        : rasterizer{rasterizer_}, gpu{gpu_}, syncpoint_manager{gpu.Host1x().GetSyncpointManager()},
          texture_cache{texture_cache_}, buffer_cache{buffer_cache_}, query_cache{query_cache_} {
        if constexpr (can_async_check) {
            fence_thread =
                std::jthread([this](std::stop_token token) { ReleaseThreadFunc(token); });
        }
    }

    virtual ~FenceManager() {
        if constexpr (can_async_check) {
            fence_thread.request_stop();
            cv.notify_all();
            fence_thread.join();
        }
    }

    /// Creates a Fence Interface, does not create a backend fence if 'is_stubbed' is
    /// true
    virtual TFence CreateFence(bool is_stubbed) = 0;
    /// Queues a fence into the backend if the fence isn't stubbed.
    virtual void QueueFence(TFence& fence) = 0;
    /// Notifies that the backend fence has been signaled/reached in host GPU.
    virtual bool IsFenceSignaled(TFence& fence) const = 0;
    /// Waits until a fence has been signalled by the host GPU.
    virtual void WaitFence(TFence& fence) = 0;
    /// Submits work whose submission the backend deferred (called before blocking on a fence).
    virtual void SubmitDeferredWork() {}
    /// This thread is about to block until the fence just signalled runs: submits everything it
    /// waits for, lazy fences included (F4-02). Defaults to SubmitDeferredWork.
    virtual void SubmitBeforeWait() {
        SubmitDeferredWork();
    }

    VideoCore::RasterizerInterface& rasterizer;
    Tegra::GPU& gpu;
    Tegra::Host1x::SyncpointManager& syncpoint_manager;
    TTextureCache& texture_cache;
    TBufferCache& buffer_cache;
    TQueryCache& query_cache;

private:
    template <bool force_wait>
    void TryReleasePendingFences() {
        while (!fences.empty()) {
            TFence& current_fence = fences.front();
            if (ShouldWait() && !IsFenceSignaled(current_fence)) {
                if constexpr (force_wait) {
                    WaitFence(current_fence);
                } else {
                    return;
                }
            }
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            PopAsyncFlushes(pending_flushes.front());
            pending_flushes.pop_front();
            auto operations = std::move(pending_operations.front());
            pending_operations.pop_front();
            for (auto& operation : operations) {
                operation();
            }
            {
                std::unique_lock lock(ring_guard);
                delayed_destruction_ring.Push(std::move(current_fence));
            }
            fences.pop();
        }
    }

    void ReleaseThreadFunc(std::stop_token stop_token) {
        Common::SetCurrentThreadName("GPUFencingThread");
        Common::SetCurrentThreadPriority(Common::ThreadPriority::High);
        Common::SetCurrentThreadToPerformanceCores();

        // Every fence the host GPU has already passed is released together: Switch 2 titles
        // signal thousands a second and the host GPU completes them in bursts, so each cache's
        // lock is taken once per burst instead of once per fence (the GPU thread, which holds
        // them for nearly every command, spun, slept and woke this thread for each one).
        // EDEN_NX2_FENCE_BATCH=0: one fence at a time (for comparisons).
        static const size_t MaxBatch = [] {
            const char* value = std::getenv("EDEN_NX2_FENCE_BATCH");
            return value != nullptr && value[0] == '0' ? size_t{1} : size_t{64};
        }();
        struct Released {
            TFence fence;
            OperationList operations;
            u8 flushes;
        };
        std::vector<Released> batch;
        batch.reserve(MaxBatch);
        const auto take_front = [&] {
            batch.push_back({std::move(fences.front()), std::move(pending_operations.front()),
                             pending_flushes.front()});
            fences.pop();
            pending_operations.pop_front();
            pending_flushes.pop_front();
        };
        while (!stop_token.stop_requested()) {
            {
                std::unique_lock lock(guard);
                cv.wait(lock, [&] { return stop_token.stop_requested() || !fences.empty(); });
                if (stop_token.stop_requested()) [[unlikely]] {
                    return;
                }
                take_front();
            }
            if (!batch.front().fence->IsStubbed()) {
                WaitFence(batch.front().fence);
            }
            {
                std::unique_lock lock(guard);
                while (batch.size() < MaxBatch && !fences.empty() &&
                       (fences.front()->IsStubbed() || IsFenceSignaled(fences.front()))) {
                    take_front();
                }
            }
            if (!batch.empty()) {
                VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            }
            size_t texture_pops = 0;
            size_t buffer_pops = 0;
            for (const Released& released : batch) {
                texture_pops += (released.flushes & TextureFlushes) != 0 ? 1 : 0;
                buffer_pops += (released.flushes & BufferFlushes) != 0 ? 1 : 0;
            }
            if (commit_backlog && (texture_pops != 0 || buffer_pops != 0)) [[unlikely]] {
                // P5a: this thread touches no cache: the owner lands the pops. Perf fix C: only
                // the pops go to the owner (not the operations, and not every later batch); this
                // thread waits for them (spin, then sleep) and keeps the operations in fence
                // order below. The old locked path waited for the cache locks the same way.
                u64 target = 0;
                bool wake = false;
                {
                    std::scoped_lock lock{backlog_mutex};
                    for (const Released& released : batch) {
                        if ((released.flushes & (TextureFlushes | BufferFlushes)) != 0) {
                            backlog.push_back(BacklogEntry{{}, released.flushes});
                            ++backlog_requested;
                        }
                    }
                    target = backlog_requested;
                    wake = !backlog_pending.exchange(true, std::memory_order_acq_rel);
                }
                if (wake && wake_owner) {
                    wake_owner();
                }
                const auto spin_start = std::chrono::steady_clock::now();
                for (u32 spins = 0; backlog_served.load(std::memory_order_acquire) < target;
                     ++spins) {
                    if ((spins & 63) == 63 && std::chrono::steady_clock::now() - spin_start >=
                                                  std::chrono::microseconds{50}) {
                        break;
                    }
                    std::this_thread::yield();
                }
                if (backlog_served.load(std::memory_order_acquire) < target) {
                    std::unique_lock lock{backlog_mutex};
                    backlog_waiting = true;
                    while (!backlog_cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                        return backlog_served.load(std::memory_order_acquire) >= target ||
                               stop_token.stop_requested();
                    })) {
                        // The owner may have gone idle between the wake and the push.
                        lock.unlock();
                        if (wake_owner) {
                            wake_owner();
                        }
                        lock.lock();
                    }
                    backlog_waiting = false;
                }
                texture_pops = 0;
                buffer_pops = 0;
            }
            if (texture_pops != 0) {
                std::scoped_lock lock{texture_cache.mutex};
                for (size_t i = 0; i < texture_pops; ++i) {
                    texture_cache.PopAsyncFlushes();
                }
            }
            if (buffer_pops != 0) {
                std::scoped_lock lock{buffer_cache.mutex};
                for (size_t i = 0; i < buffer_pops; ++i) {
                    buffer_cache.PopAsyncFlushes();
                }
            }
            // Queries and operations keep their per-fence order.
            for (Released& released : batch) {
                query_cache.PopAsyncFlushes();
                for (auto& operation : released.operations) {
                    operation();
                }
            }
            if (!batch.empty()) {
                VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            }
            {
                std::unique_lock lock(ring_guard);
                for (Released& released : batch) {
                    delayed_destruction_ring.Push(std::move(released.fence));
                }
            }
            batch.clear();
        }
    }

    bool ShouldWait() const {
        std::scoped_lock lock{buffer_cache.mutex, texture_cache.mutex};
        return texture_cache.ShouldWaitAsyncFlushes() || buffer_cache.ShouldWaitAsyncFlushes() ||
               query_cache.ShouldWaitAsyncFlushes();
    }

    static constexpr u8 TextureFlushes = 1;
    static constexpr u8 BufferFlushes = 2;

    /// committed: what CommitAsyncFlushes returned for this fence.
    void PopAsyncFlushes(u8 committed) {
        if (committed & TextureFlushes) {
            std::scoped_lock lock{texture_cache.mutex};
            texture_cache.PopAsyncFlushes();
        }
        if (committed & BufferFlushes) {
            std::scoped_lock lock{buffer_cache.mutex};
            buffer_cache.PopAsyncFlushes();
        }
        query_cache.PopAsyncFlushes();
    }

    struct CacheFlushes {
        bool should_flush;  ///< Downloads to wait for: the fence must reach the host GPU
        u8 committed;       ///< TextureFlushes, BufferFlushes: the caches that committed downloads
        bool early_only;    ///< The buffer downloads are all learned early downloads
    };

    /// Commits the texture and buffer caches' downloads for a new fence (the query cache's are
    /// committed once a fence is made). Switch 2 titles signal a fence per compute dispatch
    /// (thousands a second), mostly with nothing or only buffer data to download: each cache's
    /// lock is taken once, alone, and the fencing thread later takes only the lock of the cache
    /// with work, instead of holding both while it copies (the GPU thread's draws and uploads
    /// waited for it).
    CacheFlushes CommitCacheFlushes() {
        CacheFlushes flushes{};
        {
            std::scoped_lock lock{texture_cache.mutex};
            flushes.should_flush = texture_cache.HasUncommittedFlushes();
            if (texture_cache.HasAsyncFlushWork()) {
                texture_cache.CommitAsyncFlushes();
                flushes.committed |= TextureFlushes;
            }
        }
        {
            std::scoped_lock lock{buffer_cache.mutex};
            if (buffer_cache.HasUncommittedFlushes()) {
                // Learned early downloads never make a submission of their own: they ride the
                // next one (a lazy fence), unless something else here needs the fence flushed.
                const bool early = buffer_cache.FlushesAreEarlyDownloads();
                buffer_cache.CommitAsyncFlushes();
                flushes.early_only = early;
                flushes.should_flush |= !early;
                flushes.committed |= BufferFlushes;
            }
        }
        return flushes;
    }

    /// Whether a fence with no downloads would do nothing, so need not be made: with fast GPU
    /// fences its callback runs at once anyway, and most of the per-dispatch fences of Switch 2
    /// titles have no queries or queued operations either. Making, queueing and retiring them
    /// cost the GPU thread ~5% and woke the fencing thread for each. One in 64 is still made, so
    /// the operations fences carry along (query slot cleanup) keep running.
    // With the GPU profile on: why fences are made rather than skipped, logged every 5 s.
    // Called before SkipEmptyFence, which it mirrors without changing its state.
    void NoteFenceDecision(bool delay_fence, size_t committed_flushes) {
        static std::array<u64, 6> counts{};
        static auto since = std::chrono::steady_clock::now();
        size_t reason = 5; // would be skipped
        if (delay_fence) {
            reason = 0;
        } else if (committed_flushes != 0) {
            reason = 1;
        } else if constexpr (requires { query_cache.HasPendingFenceWork(); }) {
            if (skipped_fences >= 64) {
                reason = 2;
            } else if (query_cache.HasPendingFenceWork()) {
                reason = 3;
            } else {
                std::scoped_lock lock{uncommitted_guard};
                reason = uncommitted_operations.empty() ? 5 : 4;
            }
        } else {
            reason = 2;
        }
        ++counts[reason];
        const auto now = std::chrono::steady_clock::now();
        const double window = std::chrono::duration<double>(now - since).count();
        if (window < 5.0) {
            return;
        }
        LOG_INFO(HW_GPU,
                 "Fence decisions/s: {:.0f} made (delayed fences {:.0f}, committed flushes {:.0f}, "
                 "64-skip cap {:.0f}, query cache fence work {:.0f}, uncommitted operations "
                 "{:.0f}), {:.0f} skipped",
                 (counts[0] + counts[1] + counts[2] + counts[3] + counts[4]) / window,
                 counts[0] / window, counts[1] / window, counts[2] / window, counts[3] / window,
                 counts[4] / window, counts[5] / window);
        counts = {};
        since = now;
    }

    bool SkipEmptyFence() {
        if constexpr (requires { query_cache.HasPendingFenceWork(); }) {
            if (skipped_fences >= 64 || query_cache.HasPendingFenceWork()) {
                return false;
            }
            {
                std::scoped_lock lock{uncommitted_guard};
                if (!uncommitted_operations.empty()) {
                    return false;
                }
            }
            ++skipped_fences;
            return true;
        } else {
            return false;
        }
    }

    std::queue<TFence> fences;
    // Vectors, not deques: MSVC's deque of std::function holds one element per block, so every
    // queued operation (one per query report, about 1000 a frame in busy scenes) allocated a block.
    using OperationList = std::vector<std::function<void()>>;
    OperationList uncommitted_operations;
    std::deque<OperationList> pending_operations;
    std::deque<u8> pending_flushes; ///< Per fence: the caches that committed downloads
    u32 skipped_fences{}; ///< Empty fences skipped since the last one made (GPU thread only)

    std::mutex guard;
    std::mutex ring_guard;
    std::mutex uncommitted_guard; ///< Guards uncommitted_operations
    std::atomic<bool> has_uncommitted{false}; ///< HasUncommittedOperations

    /// P5a: fence work handed to the caches' owner (SetCommitBacklog).
    struct BacklogEntry {
        OperationList operations;
        u8 flushes;
    };
    bool commit_backlog = false;
    bool defer_writebacks = false;
    std::function<void()> wake_owner;
    std::mutex backlog_mutex;
    std::deque<BacklogEntry> backlog;     ///< Under backlog_mutex
    std::atomic<bool> backlog_pending{false};
    /// Perf fix C: pop entries handed to the owner and landed by it (the fencing thread waits
    /// for served >= requested before it runs the batch's operations).
    u64 backlog_requested = 0;            ///< Under backlog_mutex
    std::atomic<u64> backlog_served{0};   ///< Stored under backlog_mutex
    bool backlog_waiting = false;         ///< Under backlog_mutex: the fencing thread sleeps
    std::condition_variable backlog_cv;
    std::condition_variable cv;

    std::jthread fence_thread;

    DelayedDestructionRing<TFence, 8> delayed_destruction_ring;
};

} // namespace VideoCommon
