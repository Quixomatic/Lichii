// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "video_core/v2/p5b_stats.h"
#include "video_core/buffer_cache/uniform_cache_history.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <fmt/ranges.h>

#include "common/range_sets.inc"
#include "video_core/buffer_cache/buffer_cache_base.h"
#include "video_core/buffer_cache/multi_range_refresh.h"
#include "video_core/buffer_cache/uniform_stream_reuse.h"
#include "video_core/guest_memory.h"
#include "video_core/host1x/gpu_device_memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_cmdpage_diag.h"
#include "video_core/nx2_perf.h"
#include "video_core/nx2_profile.h"
#include "video_core/texture_cache/util.h"
#include "video_core/v2/cbuf_version.h"
#include "video_core/v2/p3_stats.h"
#include "video_core/v2/v2_stats.h"
#include "video_core/v2/versions.h"

namespace VideoCommon {

using Core::DEVICE_PAGESIZE;

template <class P>
BufferCache<P>::BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_)
    : runtime{runtime_}, device_memory{device_memory_}, memory_tracker{device_memory} {
    // A new runtime has a new scheduler (its ticks restart) and a new stream buffer: whatever the
    // stream uniform reuse remembers refers to the old ones. Before the early return below.
    VideoCommon::ResetUniformStreamReuse<Runtime>();
    if constexpr (VideoCommon::StreamUniformReusable<Runtime>) {
        // EDEN_NX2_V2_CBUFVER: versions of the old runtime's stream buffer are gone.
        VideoCommon::V2::CbufVer::Table<typename Runtime::StreamUniformRef>().Reset();
    }
    // Ensure the first slot is used for the null buffer
    void(slot_buffers.insert(runtime, NullBufferParams{}));
    gpu_modified_ranges.Clear();
    inline_buffer_id = NULL_BUFFER_ID;
#ifdef YUZU_LEGACY
    immediately_free = (Settings::values.vram_usage_mode.GetValue() == Settings::VramUsageMode::Aggressive);
#endif
    if (!runtime.CanReportMemoryUsage()) {
        minimum_memory = DEFAULT_EXPECTED_MEMORY;
        critical_memory = DEFAULT_CRITICAL_MEMORY;
        return;
    }

    const s64 device_local_memory = static_cast<s64>(runtime.GetDeviceLocalMemory());
    const s64 min_spacing_expected = device_local_memory - 1_GiB;
    const s64 min_spacing_critical = device_local_memory - 512_MiB;
    const s64 mem_threshold = (std::min)(device_local_memory, TARGET_THRESHOLD);
    const s64 min_vacancy_expected = (6 * mem_threshold) / 10;
    const s64 min_vacancy_critical = (2 * mem_threshold) / 10;
    minimum_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_expected, min_spacing_expected),
                 DEFAULT_EXPECTED_MEMORY));
    critical_memory = static_cast<u64>(
        (std::max)((std::min)(device_local_memory - min_vacancy_critical, min_spacing_critical),
                 DEFAULT_CRITICAL_MEMORY));
}

template <class P>
BufferCache<P>::~BufferCache() {
    if constexpr (VideoCommon::StreamUniformReusable<Runtime>) {
        VideoCommon::V2::CbufVer::Table<typename Runtime::StreamUniformRef>().Reset();
    }
}

template <class P>
void BufferCache<P>::RunGarbageCollector(bool aggressive_gc) {
    const u64 ticks_to_destroy = aggressive_gc ? 60 : 120;
    int num_iterations = aggressive_gc ? 64 : 32;
    // Their GPU-written data is downloaded first, all with one wait for the GPU: one wait per
    // buffer stalled the GPU thread every frame on title 1 (an 8 GB GPU near its budget, no
    // per-fence downloads keeping the buffers clean), ~2.5% of its time.
    boost::container::small_vector<BufferId, 64> evicted;
    lru_cache.ForEachItemBelow(frame_tick - ticks_to_destroy, [&](BufferId buffer_id) {
        if (num_iterations == 0) {
            return true;
        }
        --num_iterations;
        // Below the critical threshold, buffers holding GPU-written data stay: evicting one
        // means downloading it first. The thresholds compare the whole device's use (mostly
        // textures) with fixed headroom, so on an 8 GB GPU the collector ran every frame once a
        // title 1 course was loaded, and the downloads were ~2.7% of the GPU thread.
        const Buffer& buffer = slot_buffers[buffer_id];
        if (!aggressive_gc && gpu_modified_ranges.Intersects(buffer.CpuAddr(), buffer.SizeBytes())) {
            return false;
        }
        evicted.push_back(buffer_id);
        return false;
    });
    if constexpr (USE_MEMORY_MAPS) {
        DownloadBuffersMemory(evicted);
    } else {
        for (const BufferId buffer_id : evicted) {
            DownloadBufferMemory(slot_buffers[buffer_id]);
        }
    }
    for (const BufferId buffer_id : evicted) {
        DeleteBuffer(buffer_id);
    }
}

template <class P>
void BufferCache<P>::TickFrame() {
    // Homebrew console apps don't create or bind any channels, so this will be nullptr.
    if (!channel_state) {
        return;
    }
    runtime.TickFrame(slot_buffers);
    ++learned_frame;
    if (!readbacks_in_flight.empty()) [[unlikely]] {
        FreeLandedReadbacks();
    }
    native_indirect_uniforms_this_frame.clear();
    if (const GeometryStats& stats = geometry_stats; frame_tick % 600 == 0 && stats.syncs != 0) {
        if (Nx2Profile::Enabled() || Nx2Perf::Verify()) {
            LOG_INFO(HW_GPU,
                     "Geometry upload scans in 600 frames: {} of {} skipped, {} verify mismatches",
                     stats.syncs_skipped, stats.syncs, stats.verify_mismatches);
        }
        geometry_stats = {};
    }
    if (frame_tick % 600 == 0 && !uploaded_buffer_ranges.empty()) {
        for (const auto& [address, size] : uploaded_buffer_ranges) {
            device_memory.HintCold(address, size);
        }
        uploaded_buffer_ranges.clear();
    }
    if (static u64 frames{}; ++frames % 600 == 0) {
        u64 total{};
        size_t count{};
        std::vector<std::pair<u64, VAddr>> largest;
        for (const auto [buffer_id, buffer] : slot_buffers) {
            total += buffer->SizeBytes();
            ++count;
            largest.emplace_back(buffer->SizeBytes(), buffer->CpuAddr());
        }
        std::partial_sort(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()),
                          largest.end(), std::greater<>{});
        LOG_INFO(HW_GPU, "Buffer cache: {} buffers {} MiB; largest {}", count, total >> 20,
                 fmt::join(largest.begin(), largest.begin() + std::min<size_t>(6, largest.size()), ", "));
    }

    // Calculate hits and shots and move hit bits to the right
    const u32 hits = std::reduce(channel_state->uniform_cache_hits.begin(),
                                 channel_state->uniform_cache_hits.end());
    const u32 shots = std::reduce(channel_state->uniform_cache_shots.begin(),
                                  channel_state->uniform_cache_shots.end());
    AdvanceUniformCacheHistory(channel_state->uniform_cache_hits);
    AdvanceUniformCacheHistory(channel_state->uniform_cache_shots);

    const bool skip_preferred = hits * 256 < shots * 251;
    channel_state->uniform_buffer_skip_cache_size = skip_preferred ? DEFAULT_SKIP_CACHE_SIZE : 0;

    // If we can obtain the memory info, use it instead of the estimate.
    if (runtime.CanReportMemoryUsage()) {
        total_used_memory = runtime.GetDeviceMemoryUsage();
    }
    if (gpu_memory != nullptr && gpu_memory->UsesNativeTiling()) {
        // Switch 2 titles fill the device with textures (their collector watches the budget);
        // title 1's buffers were ~200 MiB of an 8 GB GPU's ~6.8 GB in use, yet this collector
        // ran aggressively every frame on the device-wide figure, evicting up to 64 buffers (and
        // downloading the GPU-written ones, ~5% of the GPU thread) that were recreated later.
        // Collect when the buffers themselves are large. At 1 GiB it held geometry of scenes
        // long gone: title 1's title screen grew to ~0.9 GB of buffers in ten minutes.
        if (buffer_bytes >= 512_MiB) {
            RunGarbageCollector(buffer_bytes >= 1_GiB);
        }
    } else if (total_used_memory >= minimum_memory) {
        RunGarbageCollector(total_used_memory >= critical_memory);
    }
    ++frame_tick;
    delayed_destruction_ring.Tick();

    for (auto& buffer : async_buffers_death_ring) {
        runtime.FreeDeferredStagingBuffer(buffer);
    }
    async_buffers_death_ring.clear();
}

template <class P>
void BufferCache<P>::UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        virtual_ranges.Unmap(as_id, gpu_addr, size);
    }
}

template <class P>
void BufferCache<P>::WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data) {
    // EDEN_NX2_V2_DRAWCACHE: every path that marks memory CPU-modified moves its version (a
    // recipe's skipped SynchronizeBuffer relies on it), whoever called.
    V2::Versions::HookWrite(device_addr, size, V2::Versions::Source::Host);
    Nx2NoteListHostWrite(device_addr, size);
    // The written range lies within its pages: when they hold no GPU-written data, both lookups
    // below find nothing, so one lookup answers them.
    if (keep_page_gpu_data) {
        const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
        const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
        if (!gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
            if (!Nx2Perf::Verify()) {
                memory_tracker.MarkRegionAsCpuModified(device_addr, size);
                MarkVertexBuffersStale(device_addr, size);
                return;
            }
            static std::atomic<u32> logged{};
            if (IsRegionGpuModified(device_addr, size) &&
                logged.fetch_add(1, std::memory_order_relaxed) < 64) {
                LOG_ERROR(HW_GPU, "Nx2Perf texture-writes: GPU-written data in {:#x}+{:#x} outside its pages",
                          device_addr, size);
            }
        }
    }
    if (IsRegionGpuModified(device_addr, size)) {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
    }
    // The tracker marks whole pages CPU-modified, which drops their GPU-modified state: GPU-written
    // bytes elsewhere on these pages would never be downloaded, and the page's next upload would
    // send the stale guest copy over them. Hyrule Warriors: Age of Imprisonment writes small
    // structures next to the draw arguments its compute dispatches fill in; the draws then read
    // leftover bytes as vertex counts. Download those bytes first (the CPU's own were subtracted
    // above, so they aren't overwritten).
    const DAddr page_begin = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    const DAddr page_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (keep_page_gpu_data && gpu_modified_ranges.Intersects(page_begin, page_end - page_begin)) {
        // Switch 2 titles: upload the CPU's bytes into the buffer instead, as CachedWriteMemory
        // does, so the page stays GPU-owned and nothing waits for the host GPU.
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_INLINE_PAGE_WRITES");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        const bool inline_writes =
            forced >= 0 ? forced == 1 : Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
        if (inline_writes && IsRegionRegistered(device_addr, size)) {
            tmp_buffer.resize_destructive(size);
            device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);
            InlineMemoryImplementation(device_addr, size, tmp_buffer);
            return;
        }
        DownloadMemory(page_begin, page_end - page_begin);
    }
    memory_tracker.MarkRegionAsCpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
}

template <class P>
void BufferCache<P>::CachedWriteMemory(DAddr device_addr, u64 size) {
    V2::Versions::HookWrite(device_addr, size, V2::Versions::Source::Host);
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return;
    }
    DAddr aligned_start = Common::AlignDown(device_addr, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(device_addr + size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        WriteMemory(device_addr, size);
        return;
    }

    Nx2NoteListHostWrite(device_addr, size);
    tmp_buffer.resize_destructive(size);
    device_memory.ReadBlockUnsafe(device_addr, tmp_buffer.data(), size);

    InlineMemoryImplementation(device_addr, size, tmp_buffer);
}

template <class P>
bool BufferCache<P>::OnCPUWrite(DAddr device_addr, u64 size) {
    const bool is_dirty = IsRegionRegistered(device_addr, size);
    if (!is_dirty) {
        return false;
    }
    if (memory_tracker.IsRegionGpuModified(device_addr, size)) {
        return true;
    }
    WriteMemory(device_addr, size);
    return false;
}

template <class P>
std::optional<VideoCore::RasterizerDownloadArea> BufferCache<P>::GetFlushArea(DAddr device_addr,
                                                                              u64 size) {
    std::optional<VideoCore::RasterizerDownloadArea> area{};
    area.emplace();
    DAddr device_addr_start_aligned = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
    DAddr device_addr_end_aligned = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
    area->start_address = device_addr_start_aligned;
    area->end_address = device_addr_end_aligned;
    if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
        if (learned_marked != 0) [[unlikely]] {
            LearnReadback(device_addr_start_aligned, device_addr_end_aligned); // A read: not unread
        }
        area->preemtive = true;
        return area;
    };
    area->preemtive = !IsRegionGpuModified(device_addr_start_aligned,
                                           device_addr_end_aligned - device_addr_start_aligned);
    // A preflushable mark persists and downloads later GPU writes at every fence. Native titles
    // instead keep read traps and flush on demand.
    const bool preflush = !Tegra::g_nx2_ampere.load(std::memory_order_relaxed);
    if (!area->preemtive) {
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Marked(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned, preflush);
        }
        if (preflush) {
            memory_tracker.MarkRegionAsPreflushable(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        } else {
            if (VideoCommon::V2::On(VideoCommon::V2::Switch::Readback) &&
                !Settings::IsGPULevelHigh()) [[unlikely]] {
                // Early downloads: pages the guest keeps reading back become Preflushable.
                LearnReadback(device_addr_start_aligned, device_addr_end_aligned);
            }
            if constexpr (USE_MEMORY_MAPS) {
                cpu_read_ranges.Add(area->start_address, area->end_address - area->start_address);
                download_cpu_read_ranges = true;
            }
            // The core consumed this trap before GetFlushArea. Forget it so the next GPU write
            // really rearms the core instead of being skipped by the deduplication cache.
            read_trapped_ranges.Subtract(
                device_addr_start_aligned, device_addr_end_aligned - device_addr_start_aligned);
        }
    } else {
        // The core keeps this area and has no mark. The next GPU write here must arm the range
        // again, which ends the area and sets the trap.
        read_trapped_ranges.Subtract(device_addr_start_aligned,
                                     device_addr_end_aligned - device_addr_start_aligned);
    }
    return area;
}

template <class P>
void BufferCache<P>::DownloadMemory(DAddr device_addr, u64 size) {
    if constexpr (USE_MEMORY_MAPS) {
        if (download_cpu_read_ranges) {
            download_cpu_read_ranges = false;
            boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges{{device_addr, size}};
            cpu_read_ranges.ForEach([&](DAddr lower, DAddr upper) {
                gpu_modified_ranges.ForEachInRange(lower, upper - lower,
                                                   [&](DAddr start, DAddr end) {
                                                       ranges.emplace_back(start, end - start);
                                                   });
            });
            DownloadRangesMemory(ranges);
            return;
        }
    }
    ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
        const DAddr start = (std::max)(device_addr, buffer.CpuAddr());
        const DAddr end = (std::min)(device_addr + size, buffer.CpuAddr() + buffer.SizeBytes());
        if (start < end) {
            DownloadBufferMemory(buffer, start, end - start);
        }
    });
}

template <class P>
struct BufferCache<P>::ReadbackState {
    struct Copy {
        DAddr device_addr;
        u64 offset; ///< In the staging buffer's mapped span
        u64 size;
    };
    typename ReadbackStaging<typename P::Runtime>::type staging{};
    boost::container::small_vector<Copy, 8> copies;
    Common::RangeSet<DAddr> cpu_written;
    std::vector<u8> expected; ///< Verify: what the synchronous download wrote
    u64 tick{};
    u64 job_semaphore{}; ///< EDEN_NX2_V2_XREADBACK: the transfer queue job, else 0
    u64 job_value{};
    bool verify{};
    std::atomic<bool> landed{};
};

template <class P>
void BufferCache<P>::FreeLandedReadbacks() {
    if constexpr (requires(typename ReadbackStaging<Runtime>::type& ref) { runtime.FreeDeferredStagingBuffer(ref); }) {
        std::erase_if(readbacks_in_flight, [this](const std::shared_ptr<ReadbackState>& state) {
            if (!state->landed.load(std::memory_order_acquire)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(state->staging);
            return true;
        });
    }
}

template <class P>
bool BufferCache<P>::DownloadMemoryAsync(DAddr device_addr, u64 size,
                                         std::function<void()>& land, bool verify) {
    if constexpr (!USE_MEMORY_MAPS ||
                  !requires { runtime.SubmitForHostRead(); runtime.WaitForHostRead(u64{}); } ||
                  !requires(typename ReadbackStaging<Runtime>::type& ref) { runtime.FreeDeferredStagingBuffer(ref); }) {
        return false;
    } else {
        FreeLandedReadbacks();
        if (readbacks_in_flight.size() >= 64) {
            return false; // Readers that never landed theirs: don't pile up staging memory
        }
        // EDEN_NX2_V2_XREADBACK: the copies run on the transfer queue (per-thread background
        // resources); mapped buffers are snapshotted there too.
        bool transfer = false;
        if constexpr (requires { runtime.TransferReadbacks(); }) {
            transfer = !verify && runtime.TransferReadbacks();
        }
        // Buffers read in place (mapped) need no copy, but a later GPU write could reach them
        // before the reader does: they keep the synchronous path (unless copied on the transfer
        // queue).
        bool mapped = false;
        ForEachBufferInRange(device_addr, size, [&](BufferId, Buffer& buffer) {
            if constexpr (requires { buffer.Mapped(); }) {
                mapped |= !buffer.Mapped().empty();
            }
        });
        if (mapped && !transfer) {
            return false;
        }
        struct Download {
            BufferId buffer_id;
            BufferCopy copy;
        };
        boost::container::small_vector<Download, 16> staged;
        u64 total_size_bytes = 0;
        ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const DAddr begin = (std::max)(device_addr, buffer_addr);
            const DAddr end = (std::min)(device_addr + size, buffer_addr + buffer.SizeBytes());
            if (begin >= end) {
                return;
            }
            memory_tracker.ForEachDownloadRangeAndClear(
                begin, end - begin, [&](u64 device_addr_out, u64 range_size) {
                    gpu_modified_ranges.ForEachInRange(
                        device_addr_out, range_size, [&](DAddr start, DAddr range_end) {
                            staged.push_back({buffer_id, BufferCopy{
                                                             .src_offset = start - buffer_addr,
                                                             .dst_offset = total_size_bytes,
                                                             .size = range_end - start,
                                                         }});
                            total_size_bytes += Common::AlignUp(range_end - start, 64);
                        });
                    ClearDownload(device_addr_out, range_size);
                    gpu_modified_ranges.Subtract(device_addr_out, range_size);
                });
        });
        if (staged.empty()) {
            return false;
        }
        auto state = std::make_shared<ReadbackState>();
        state->staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
        for (Download& download : staged) {
            Buffer& buffer = slot_buffers[download.buffer_id];
            state->copies.push_back({buffer.CpuAddr() + download.copy.src_offset,
                                     download.copy.dst_offset, download.copy.size});
            download.copy.dst_offset += state->staging.offset;
            buffer.MarkUsage(download.copy.src_offset, download.copy.size);
            if constexpr (requires { runtime.TransferReadbacks(); }) {
                if (transfer) {
                    runtime.TransferReadbackCopy(state->staging.buffer, buffer, download.copy,
                                                 mapped);
                    continue;
                }
            }
            runtime.CopyBuffer(state->staging.buffer, buffer, std::span{&download.copy, 1}, true);
        }
        state->cpu_written = PendingCpuWriteRanges();
        state->verify = verify;
        if (verify) {
            // The old path: wait here and write; the reader compares its copy with this.
            runtime.FinishForHostRead();
            for (const auto& copy : state->copies) {
                const u8* const data = state->staging.mapped_span.data() + copy.offset;
                state->expected.insert(state->expected.end(), data, data + copy.size);
                WriteDownloadedMemory(state->cpu_written, copy.device_addr, data, copy.size);
            }
        } else if (transfer) {
            if constexpr (requires { runtime.TransferReadbacks(); }) {
                const auto job = runtime.SubmitTransferReadback();
                state->job_semaphore = job.first;
                state->job_value = job.second;
            }
        } else {
            state->tick = runtime.SubmitForHostRead();
        }
        readbacks_in_flight.push_back(state);
        land = [state, runtime_ptr = &runtime, memory = &device_memory] {
            if (state->job_value != 0) {
                if constexpr (requires { runtime_ptr->TransferReadbacks(); }) {
                    runtime_ptr->WaitTransferReadback(state->job_semaphore, state->job_value);
                }
            } else if (!state->verify) {
                runtime_ptr->WaitForHostRead(state->tick);
            }
            size_t expected_offset = 0;
            for (const auto& copy : state->copies) {
                const u8* const data = state->staging.mapped_span.data() + copy.offset;
                if (state->verify) {
                    if (std::memcmp(data, state->expected.data() + expected_offset, copy.size) !=
                        0) {
                        static std::atomic<u32> logged{};
                        if (logged.fetch_add(1) < 16) {
                            LOG_WARNING(HW_GPU,
                                        "V2 readback verify: the asynchronous download of {:#x} "
                                        "({} bytes) differs from the synchronous one",
                                        copy.device_addr, copy.size);
                        }
                    }
                    expected_offset += copy.size;
                    continue;
                }
                // As WriteDownloadedMemory: CPU writes since the download keep their bytes.
                V2::Versions::HookWrite(copy.device_addr, copy.size, V2::Versions::Source::Gpu);
                if (!state->cpu_written.Intersects(copy.device_addr, copy.size)) {
                    memory->WriteBlockUnsafe(copy.device_addr, data, copy.size);
                    continue;
                }
                Common::RangeSet<DAddr> parts;
                parts.Add(copy.device_addr, copy.size);
                state->cpu_written.ForEachInRange(copy.device_addr, copy.size,
                                                  [&parts](DAddr start, DAddr end) {
                                                      parts.Subtract(start, end - start);
                                                  });
                parts.ForEach([&](DAddr start, DAddr end) {
                    memory->WriteBlockUnsafe(start, data + (start - copy.device_addr),
                                             end - start);
                });
            }
            state->landed.store(true, std::memory_order_release);
        };
        return true;
    }
}

template <class P>
void BufferCache<P>::LearnReadback(DAddr begin, DAddr end) {
    // PR #805's learning rules: a page qualifies after 2 reads; at most 256 pages (1 MiB) at once.
    static constexpr size_t MaxMarked = 256;
    for (DAddr page = begin >> Core::DEVICE_PAGEBITS; page < (end + Core::DEVICE_PAGESIZE - 1) >>
                                                          Core::DEVICE_PAGEBITS;
         ++page) {
        LearnedPage& learned = learned_pages[page];
        learned.unread_writes = 0;
        if (learned.marked || ++learned.reads < 2 || learned_marked >= MaxMarked) {
            continue;
        }
        learned.marked = true;
        ++learned_marked;
        const DAddr address = page << Core::DEVICE_PAGEBITS;
        memory_tracker.MarkRegionAsPreflushable(address, Core::DEVICE_PAGESIZE);
        learned_lowest = (std::min)(learned_lowest, address);
        learned_highest = (std::max)(learned_highest, address + Core::DEVICE_PAGESIZE);
    }
    if (learned_pages.size() > 16384) {
        std::erase_if(learned_pages, [](const auto& pair) { return !pair.second.marked; });
    }
}

template <class P>
void BufferCache<P>::NoteLearnedWrite(DAddr device_addr, u64 size) {
    if (learned_marked == 0 || device_addr >= learned_highest ||
        device_addr + size <= learned_lowest) {
        return;
    }
    const DAddr first = (std::max)(device_addr, learned_lowest) >> Core::DEVICE_PAGEBITS;
    const DAddr last = ((std::min)(device_addr + size, learned_highest) - 1) >> Core::DEVICE_PAGEBITS;
    for (DAddr page = first; page <= last; ++page) {
        const auto it = learned_pages.find(page);
        if (it == learned_pages.end() || !it->second.marked) {
            continue;
        }
        // Counted once per frame: Ocarina of Time writes such pages several times a frame and
        // reads them once, which flipped them between learned and unlearned every frame.
        if (it->second.write_frame == learned_frame) {
            continue;
        }
        it->second.write_frame = learned_frame;
        if (++it->second.unread_writes >= 4) {
            // Written in four frames without a read: stop downloading it at fences.
            memory_tracker.UnmarkRegionAsPreflushable(page << Core::DEVICE_PAGEBITS,
                                                      Core::DEVICE_PAGESIZE);
            learned_pages.erase(it);
            --learned_marked;
        }
    }
}

template <class P>
void BufferCache<P>::DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges) {
    constexpr bool has_mapped = requires(Buffer& buffer) { buffer.Mapped(); };
    if constexpr (!USE_MEMORY_MAPS) {
        for (const auto& [device_addr, size] : ranges) {
            DownloadMemory(device_addr, size);
        }
    } else {
        struct Download {
            BufferId buffer_id;
            BufferCopy copy;
        };
        boost::container::small_vector<Download, 16> staged;
        boost::container::small_vector<Download, 16> mapped;
        u64 total_size_bytes = 0;
        for (const auto& [device_addr, size] : ranges) {
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(device_addr, buffer_addr);
                const DAddr end = (std::min)(device_addr + size, buffer_addr + buffer.SizeBytes());
                if (begin >= end) {
                    return;
                }
                bool is_mapped = false;
                if constexpr (has_mapped) {
                    is_mapped = !buffer.Mapped().empty();
                }
                memory_tracker.ForEachDownloadRangeAndClear(
                    begin, end - begin, [&](u64 device_addr_out, u64 range_size) {
                        gpu_modified_ranges.ForEachInRange(
                            device_addr_out, range_size, [&](DAddr start, DAddr range_end) {
                                const BufferCopy copy{
                                    .src_offset = start - buffer_addr,
                                    .dst_offset = total_size_bytes,
                                    .size = range_end - start,
                                };
                                if (is_mapped) {
                                    mapped.push_back({buffer_id, copy});
                                    return;
                                }
                                staged.push_back({buffer_id, copy});
                                total_size_bytes += Common::AlignUp(range_end - start, 64);
                            });
                        ClearDownload(device_addr_out, range_size);
                        gpu_modified_ranges.Subtract(device_addr_out, range_size);
                    });
            });
        }
        if (staged.empty() && mapped.empty()) {
            return;
        }
        const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
        if (staged.empty()) {
            if constexpr (has_mapped) {
                runtime.FinishForHostRead();
            }
        } else {
            auto staging = runtime.DownloadStagingBuffer(total_size_bytes);
            for (Download& download : staged) {
                download.copy.dst_offset += staging.offset;
                Buffer& buffer = slot_buffers[download.buffer_id];
                buffer.MarkUsage(download.copy.src_offset, download.copy.size);
                runtime.CopyBuffer(staging.buffer, buffer, std::span{&download.copy, 1}, true);
            }
            if constexpr (has_mapped) {
                runtime.FinishForHostRead();
            } else {
                runtime.Finish();
            }
            for (const Download& download : staged) {
                WriteDownloadedMemory(
                    cpu_written,
                    slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset,
                    staging.mapped_span.data() + (download.copy.dst_offset - staging.offset),
                    download.copy.size);
            }
        }
        if constexpr (has_mapped) {
            for (const Download& download : mapped) {
                Buffer& buffer = slot_buffers[download.buffer_id];
                WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + download.copy.src_offset,
                                      buffer.Mapped().data() + download.copy.src_offset,
                                      download.copy.size);
            }
        }
    }
}

template <class P>
void BufferCache<P>::ClearDownload(DAddr device_addr, u64 size) {
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    uncommitted_gpu_modified_ranges.Subtract(device_addr, size);
    for (auto& interval_set : committed_gpu_modified_ranges) {
        interval_set.Subtract(device_addr, size);
    }
}

template <class P>
bool BufferCache<P>::DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount) {
    const std::optional<DAddr> cpu_src_address = gpu_memory->GpuToCpuAddress(src_address);
    const std::optional<DAddr> cpu_dest_address = gpu_memory->GpuToCpuAddress(dest_address);
    if (!cpu_src_address || !cpu_dest_address) {
        return false;
    }
    const bool source_dirty = IsRegionRegistered(*cpu_src_address, amount);
    const bool dest_dirty = IsRegionRegistered(*cpu_dest_address, amount);
    if (!source_dirty && !dest_dirty) {
        return false;
    }

    ClearDownload(*cpu_dest_address, amount);
    MarkVertexBuffersStale(*cpu_dest_address, amount);
    V2::Versions::HookWrite(*cpu_dest_address, amount, V2::Versions::Source::Dma);

    BufferId buffer_a;
    BufferId buffer_b;
    do {
        channel_state->has_deleted_buffers = false;
        buffer_a = FindBuffer(*cpu_src_address, static_cast<u32>(amount), false);
        buffer_b = FindBuffer(*cpu_dest_address, static_cast<u32>(amount), false);
    } while (channel_state->has_deleted_buffers);
    auto& src_buffer = slot_buffers[buffer_a];
    auto& dest_buffer = slot_buffers[buffer_b];
    SynchronizeBuffer(src_buffer, *cpu_src_address, static_cast<u32>(amount));
    SynchronizeBuffer(dest_buffer, *cpu_dest_address, static_cast<u32>(amount));
    std::array copies{BufferCopy{
        .src_offset = src_buffer.Offset(*cpu_src_address),
        .dst_offset = dest_buffer.Offset(*cpu_dest_address),
        .size = amount,
    }};

    boost::container::small_vector<std::pair<DAddr, size_t>, 4> tmp_intervals;
    auto mirror = [&](DAddr base_address, DAddr base_address_end) {
        const u64 size = base_address_end - base_address;
        const DAddr diff = base_address - *cpu_src_address;
        const DAddr new_base_address = *cpu_dest_address + diff;
        tmp_intervals.push_back({new_base_address, size});
        QueueFenceDownload(new_base_address, size);
    };
    gpu_modified_ranges.ForEachInRange(*cpu_src_address, amount, mirror);
    // This subtraction in this order is important for overlapping copies.
    gpu_modified_ranges.Subtract(*cpu_dest_address, amount);
    const bool has_new_downloads = tmp_intervals.size() != 0;
    for (const auto& pair : tmp_intervals) {
        gpu_modified_ranges.Add(pair.first, pair.second);
        if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
            Tegra::Nx2MarkGpuWritten(pair.first, pair.second);
            Nx2NoteListWrite(buffer_b, pair.first, pair.second);
        } else if (commit != nullptr) [[unlikely]] {
            Tegra::Nx2MarkGpuWritten(pair.first, pair.second);
        }
    }
    const auto& copy = copies[0];
    src_buffer.MarkUsage(copy.src_offset, copy.size);
    dest_buffer.MarkUsage(copy.dst_offset, copy.size);
    runtime.CopyBuffer(dest_buffer, src_buffer, copies, true);
    if (has_new_downloads) {
        memory_tracker.MarkRegionAsGpuModified(*cpu_dest_address, amount);
        const bool should_sync = Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
        if (should_sync) {
            runtime.Finish();
        }
    }

    Tegra::Memory::DeviceGuestMemoryScoped<u8, Tegra::Memory::GuestMemoryFlags::UnsafeReadWrite>
        tmp(device_memory, *cpu_src_address, amount, &tmp_buffer);
    tmp.SetAddressAndSize(*cpu_dest_address, amount);
    return true;
}

template <class P>
bool BufferCache<P>::DMAClear(GPUVAddr dst_address, u64 amount, u32 value) {
    const std::optional<DAddr> cpu_dst_address = gpu_memory->GpuToCpuAddress(dst_address);
    if (!cpu_dst_address) {
        return false;
    }
    const bool dest_dirty = IsRegionRegistered(*cpu_dst_address, amount);
    if (!dest_dirty) {
        return false;
    }

    const size_t size = amount * sizeof(u32);
    ClearDownload(*cpu_dst_address, size);
    V2::Versions::HookWrite(*cpu_dst_address, size, V2::Versions::Source::Dma);
    gpu_modified_ranges.Subtract(*cpu_dst_address, size);

    const BufferId buffer = FindBuffer(*cpu_dst_address, static_cast<u32>(size), false);
    Buffer& dest_buffer = slot_buffers[buffer];
    const u32 offset = dest_buffer.Offset(*cpu_dst_address);
    runtime.ClearBuffer(dest_buffer, offset, size, value);
    dest_buffer.MarkUsage(offset, size);
    return true;
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                                 ObtainBufferSynchronize sync_info,
                                                                 ObtainBufferOperation post_op) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        return {&slot_buffers[NULL_BUFFER_ID], 0};
    }
    return ObtainCPUBuffer(*device_addr, size, sync_info, post_op);
}

template <class P>
std::pair<typename P::Buffer*, u32> BufferCache<P>::ObtainCPUBuffer(
    DAddr device_addr, u32 size, ObtainBufferSynchronize sync_info, ObtainBufferOperation post_op) {
    const BufferId buffer_id = FindBuffer(device_addr, size, false);
    Buffer& buffer = slot_buffers[buffer_id];

    // synchronize op
    switch (sync_info) {
    case ObtainBufferSynchronize::FullSynchronize:
        SynchronizeBuffer(buffer, device_addr, size);
        break;
    default:
        break;
    }

    switch (post_op) {
    case ObtainBufferOperation::MarkAsWritten:
        MarkWrittenBuffer(buffer_id, device_addr, size);
        buffer.MarkUsage(buffer.Offset(device_addr), size);
        break;
    case ObtainBufferOperation::DiscardWrite: {
        ClearDownload(device_addr, size);
        gpu_modified_ranges.Subtract(device_addr, size);
        break;
    }
    default:
        break;
    }

    return {&buffer, buffer.Offset(device_addr)};
}

template <class P>
void BufferCache<P>::BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr,
                                               u32 size) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr) {
        channel_state->uniform_buffers[stage][index] = NULL_BINDING;
        return;
    }
    const Binding binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = BufferId{},
    };
    channel_state->uniform_buffers[stage][index] = binding;
}

template <class P>
void BufferCache<P>::DisableGraphicsUniformBuffer(size_t stage, u32 index) {
    auto& current = channel_state->uniform_buffers[stage][index];
    if (current.device_addr == 0 && current.size == 0 && current.buffer_id == NULL_BUFFER_ID)
        [[likely]] {
        return;
    }
    current = NULL_BINDING;
}

template <class P>
void BufferCache<P>::UpdateGraphicsBuffers(bool is_indexed) {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateGraphicsBuffers(is_indexed);
    } while (channel_state->has_deleted_buffers);
}

template <class P>
void BufferCache<P>::UpdateComputeBuffers() {
    do {
        channel_state->has_deleted_buffers = false;
        DoUpdateComputeBuffers();
    } while (channel_state->has_deleted_buffers);
}

template <class P>
u64 BufferCache<P>::KeptGraphicsBindingsHash(bool vertex, bool storage, bool texture) const {
    u64 hash = 0xcbf29ce484222325ULL;
    const auto mix = [&](u64 value) { hash = (hash ^ value) * 0x100000001b3ULL; };
    const auto mix_binding = [&](const Binding& binding) {
        mix(binding.device_addr);
        mix(binding.size);
    };
    if (vertex) {
        for (const Binding& binding : channel_state->vertex_buffers) {
            mix_binding(binding);
        }
    }
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        if (storage) {
            mix(channel_state->enabled_storage_buffers[stage]);
            mix(channel_state->written_storage_buffers[stage]);
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                mix_binding(channel_state->storage_buffers[stage][index]);
            });
        }
        if (texture) {
            mix(channel_state->enabled_texture_buffers[stage]);
            ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
                const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
                mix_binding(binding);
                mix(static_cast<u64>(binding.format));
            });
        }
    }
    return hash;
}

template <class P>
void BufferCache<P>::BindHostGeometryBuffers(bool is_indexed) {
    if (is_indexed) {
        BindHostIndexBuffer();
    } else if constexpr (!HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
            draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
    }
    BindHostVertexBuffers();
    BindHostTransformFeedbackBuffers();
    if (current_draw_indirect) {
        BindHostDrawIndirectBuffers();
    }
}

template <class P>
void BufferCache<P>::BindHostStageBuffers(size_t stage) {
    BindHostGraphicsUniformBuffers(stage);
    if (channel_state->enabled_storage_buffers[stage] != 0) {
        BindHostGraphicsStorageBuffers(stage);
    }
    if (channel_state->enabled_texture_buffers[stage] != 0) {
        BindHostGraphicsTextureBuffers(stage);
    }
}

template <class P>
void BufferCache<P>::BindHostComputeBuffers() {
    BindHostComputeUniformBuffers();
    if (channel_state->enabled_compute_storage_buffers != 0) {
        BindHostComputeStorageBuffers();
    }
    if (channel_state->enabled_compute_texture_buffers != 0) {
        BindHostComputeTextureBuffers();
    }
    if (any_buffer_uploaded) {
        runtime.PostCopyBarrier();
        any_buffer_uploaded = false;
    }
}

template <class P>
void BufferCache<P>::SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                            const UniformBufferSizes* sizes) {
    if (channel_state->enabled_uniform_buffer_masks == mask &&
        channel_state->uniform_buffer_sizes == sizes) [[likely]] {
        return;
    }
    if (channel_state->enabled_uniform_buffer_masks != mask) {
        channel_state->fast_bound_uniform_buffers.fill(0);
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers.fill(~u32{0});
            channel_state->uniform_buffer_binding_sizes.fill({});
        }
        channel_state->enabled_uniform_buffer_masks = mask;
    }
    channel_state->uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::SetComputeUniformBufferState(u32 mask,
                                                  const ComputeUniformBufferSizes* sizes) {
    channel_state->enabled_compute_uniform_buffer_mask = mask;
    channel_state->compute_uniform_buffer_sizes = sizes;
}

template <class P>
void BufferCache<P>::UnbindGraphicsStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) {
        return;
    }
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_graphics_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_storage_buffers[stage]));
        }
    }
    channel_state->enabled_storage_buffers[stage] = 0;
    channel_state->written_storage_buffers[stage] = 0;
}

template <class P>
bool BufferCache<P>::BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index,
                                               u32 cbuf_offset, bool is_written,
                                               u32 indirect_offset) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        LOG_ERROR(HW_GPU, "Invalid graphics storage buffer binding: stage {}, index {}",
                  stage, ssbo_index);
        return false;
    }
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_graphics_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping graphics storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return false;
            }
        }
    }
    channel_state->enabled_storage_buffers[stage] |= u64{1} << ssbo_index;
    channel_state->written_storage_buffers[stage] |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_graphics_storage_buffers;
        }
    }

    const auto& cbufs = maxwell3d->state.shader_stages[stage];
    GPUVAddr ssbo_addr = cbufs.const_buffers[cbuf_index].address + cbuf_offset;
    GPUVAddr holder{};
    if (indirect_offset != ~0U) {
        // The pointer is an entry of a table the constant buffer points to.
        holder = gpu_memory->Read<u64>(ssbo_addr);
        if (holder == 0) {
            channel_state->storage_buffers[stage][ssbo_index] = NULL_BINDING;
            return false;
        }
        ssbo_addr = holder + indirect_offset;
    }
    channel_state->storage_buffers[stage][ssbo_index] =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written,
                             maxwell3d->GetNativeMacroEngine().native_turing ||
                                 indirect_offset != ~0U,
                             indirect_offset != ~0U);
    if (indirect_offset != ~0U) {
        TraceTablePointer(false, stage, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          channel_state->storage_buffers[stage][ssbo_index]);
    }
    return (channel_state->storage_buffers[stage][ssbo_index].buffer_id != NULL_BUFFER_ID);
}


template <class P>
void BufferCache<P>::RecordGraphicsBuffers(VideoCommon::V2::DrawCache::BufferRecipe& recipe,
                                           bool is_indexed, u64 stamp) {
    namespace DC = VideoCommon::V2::DrawCache;
    namespace Versions = VideoCommon::V2::Versions;
    recipe.Clear();
    if constexpr (!IS_OPENGL && !HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT &&
                  VideoCommon::StreamUniformReusable<Runtime>) {
        const Versions::Generations& gens = Versions::Gens();
        const auto make = [&](u32 stage, u32 index, const Binding& binding) {
            return DC::BufferBind{
                .stage = static_cast<u8>(stage),
                .index = static_cast<u8>(index),
                .buffer_id = binding.buffer_id.index,
                .generation = gens.Get(Versions::Kind::Buffer, binding.buffer_id.index),
                .device_addr = binding.device_addr,
                .size = binding.size,
            };
        };
        if (current_draw_indirect || runtime.AnyVertexRepacked()) {
            return;
        }
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (is_indexed) {
            if (!draw_state.inline_index_draw_indexes.empty() ||
                (draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
                 runtime.WidensUint8IndicesOnHost())) {
                // Widened on the host from guest memory every draw: no recipe.
                return;
            }
            recipe.has_index = true;
            recipe.index = make(0, 0, channel_state->index_buffer);
        }
        for (u32 mask = enabled_vertex_buffers_mask; mask != 0; mask &= mask - 1) {
            const u32 index = static_cast<u32>(std::countr_zero(mask));
            recipe.vertex.push_back(make(0, index, VertexBufferSlot(index)));
        }
        const u64 tick = runtime.CurrentTick();
        for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
                const Binding& binding = channel_state->uniform_buffers[stage][index];
                DC::BufferBind bind = make(stage, index, binding);
                const u32 size = (std::min)(binding.size,
                                            (*channel_state->uniform_buffer_sizes)[stage][index]);
                bind.size = size;
                const DC::BufferBind::Kind path = UniformBindKind(stage, index, binding);
                if (path == DC::BufferBind::Kind::Normal) {
                    bind.kind = DC::BufferBind::Kind::Normal;
                } else if (path == DC::BufferBind::Kind::Version) {
                    using Ref = typename Runtime::StreamUniformRef;
                    static_assert(sizeof(Ref) <= sizeof(bind.ref));
                    const Ref* version = nullptr;
                    if (VideoCommon::V2::On(VideoCommon::V2::Switch::CbufVer)) {
                        // The version the bind just referenced (or made) still holds the bytes
                        // (read and compared only when its write checks pass).
                        version = VideoCommon::V2::CbufVer::Table<Ref>().FindWith(
                            binding.device_addr, size, tick,
                            [&] { return V2GuestBytes(binding.device_addr, size); });
                    }
                    if (version != nullptr) {
                        bind.kind = DC::BufferBind::Kind::Version;
                        std::memcpy(bind.ref.data(), version, sizeof(Ref));
                        bind.tick = tick;
                    } else {
                        bind.kind = DC::BufferBind::Kind::Normal;
                    }
                } else {
                    bind.kind = DC::BufferBind::Kind::Cached;
                }
                recipe.uniform.push_back(bind);
            });
            bool multi_range = false;
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& binding = channel_state->storage_buffers[stage][index];
                multi_range |= binding.segment_count != 0;
                recipe.storage.push_back(make(stage, index, binding));
            });
            if (multi_range) {
                recipe.Clear();
                return;
            }
            ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
                const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
                DC::BufferBind bind = make(stage, index, binding);
                bind.format = static_cast<u32>(binding.format);
                if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
                    bind.flags = static_cast<u8>(
                        (channel_state->image_texture_buffers[stage] >> index) & 1);
                }
                recipe.texture.push_back(bind);
            });
        }
        recipe.stamp = stamp;
        recipe.valid = true;
        recipe.MergeRanges();
    } else {
        static_cast<void>(is_indexed);
        static_cast<void>(stamp);
    }
}

template <class P>
bool BufferCache<P>::ReplayGraphicsBuffers(const VideoCommon::V2::DrawCache::BufferRecipe& recipe,
                                           bool is_indexed,
                                           VideoCommon::V2::Versions::EpochMemo& memo) {
    namespace DC = VideoCommon::V2::DrawCache;
    namespace Versions = VideoCommon::V2::Versions;
    if (V2Defers()) [[unlikely]] {
        // The replay's binds defer their mutations like an update pass's (its fallback lookups
        // create at once: P6 treats a replay creation as serial).
        CommitBinds();
        V2BeginBindScope();
    }
    if constexpr (!IS_OPENGL && !HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT &&
                  VideoCommon::StreamUniformReusable<Runtime>) {
        if (!recipe.valid || recipe.has_index != is_indexed || current_draw_indirect ||
            runtime.AnyVertexRepacked() || maxwell3d->regs.transform_feedback_enabled != 0) {
            return false;
        }
        const auto& draw_state = maxwell3d->draw_manager.draw_state;
        if (!draw_state.inline_index_draw_indexes.empty() ||
            (is_indexed && draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
             runtime.WidensUint8IndicesOnHost())) {
            // The index format is not in the key: 8-bit indices widened on the host from guest
            // memory every draw take the existing path.
            return false;
        }
        const Versions::Generations& gens = Versions::Gens();
        // F2-07: one check of the recipe's merged ranges, not one memo query per binding.
        replay_all_unchanged = recipe.AllUnchanged(memo, Versions::Pages());
        // ...and one liveness check per buffer deletion, not one per binding per draw.
        const auto alive = [&](const DC::BufferBind& bind) {
            return gens.Get(Versions::Kind::Buffer, bind.buffer_id) == bind.generation &&
                   slot_buffers.contains(BufferId{bind.buffer_id});
        };
        bool all_alive = recipe.alive_deletions == buffer_deletions;
        if (!all_alive) {
            all_alive = !recipe.has_index || alive(recipe.index);
            for (const auto* list : {&recipe.vertex, &recipe.uniform, &recipe.storage,
                                     &recipe.texture}) {
                for (const DC::BufferBind& bind : *list) {
                    all_alive &= alive(bind);
                }
            }
            if (all_alive) {
                recipe.alive_deletions = buffer_deletions;
            }
        }
        replay_all_alive = all_alive;
        // A recorded buffer still bindable for [address, address + size): alive, the range within
        // the recorded one and unchanged since the recipe's stamp.
        const auto usable = [&](const DC::BufferBind* bind, DAddr address, u64 size) {
            return bind != nullptr && bind->Contains(address, size) &&
                   (all_alive || alive(*bind)) &&
                   (replay_all_unchanged || memo.Unchanged(address, size, recipe.stamp));
        };
        auto& flags = maxwell3d->dirty.flags;
        graphics_segments.clear();
        channel_state->has_deleted_buffers = false;
        replay_skip_index = false;
        replay_skip_vertex_mask = 0;

        // Index buffer (UpdateIndexBuffer without the lookup when the recipe holds it).
        if (is_indexed) {
            if (flags[Dirty::IndexBuffer]) {
                flags[Dirty::IndexBuffer] = false;
                const auto& index_buffer_ref = draw_state.index_buffer;
                const GPUVAddr gpu_addr_begin = index_buffer_ref.StartAddress();
                const GPUVAddr gpu_addr_end = index_buffer_ref.EndAddress();
                const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
                const u32 address_size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
                const u32 draw_size = (index_buffer_ref.count + index_buffer_ref.first) *
                                      u32(index_buffer_ref.FormatSizeInBytes());
                const u32 size = (std::min)(address_size, draw_size);
                if (size == 0 || !device_addr) {
                    channel_state->index_buffer = NULL_BINDING;
                } else if (usable(&recipe.index, *device_addr, size)) {
                    channel_state->index_buffer = Binding{
                        .device_addr = *device_addr,
                        .size = size,
                        .buffer_id = BufferId{recipe.index.buffer_id},
                    };
                    replay_skip_index = true;
                } else {
                    channel_state->index_buffer = Binding{
                        .device_addr = *device_addr,
                        .size = size,
                        .buffer_id = FindBuffer(*device_addr, size, false),
                    };
                }
            } else {
                const Binding& current = channel_state->index_buffer;
                replay_skip_index = current.buffer_id.index == recipe.index.buffer_id &&
                                    usable(&recipe.index, current.device_addr, current.size);
            }
        }
        // Vertex buffers (UpdateVertexBuffer without the lookup when the recipe holds it).
        if (flags[Dirty::VertexBuffers]) {
            flags[Dirty::VertexBuffers] = false;
            for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
                if (!flags[Dirty::VertexBuffer0 + index]) {
                    continue;
                }
                const auto& array = maxwell3d->regs.vertex_streams[index];
                const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
                const GPUVAddr gpu_addr_begin = array.Address();
                const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
                const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
                if (native && array.enable != 0 &&
                    native_size > (std::numeric_limits<u32>::max)()) {
                    throw std::runtime_error(
                        "Native vertex stream exceeds supported 32-bit buffer size");
                }
                const GPUVAddr gpu_addr_end =
                    native ? gpu_addr_begin + native_size : limit.Address() + 1;
                const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr_begin);
                u32 size = static_cast<u32>(gpu_addr_end - gpu_addr_begin);
                if (array.enable == 0 || size == 0 || !device_addr) {
                    channel_state->vertex_buffers[index] = NULL_BINDING;
                    UpdateVertexBufferSlot(index, NULL_BINDING);
                    continue;
                }
                if (!gpu_memory->IsWithinGPUAddressRange(gpu_addr_end) || size >= 64_MiB) {
                    size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr_begin, size));
                }
                const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.vertex, 0, index);
                const BufferId buffer_id = usable(kept, *device_addr, size)
                                               ? BufferId{kept->buffer_id}
                                               : FindBuffer(*device_addr, size, false);
                const Binding binding{
                    .device_addr = *device_addr,
                    .size = size,
                    .buffer_id = buffer_id,
                };
                channel_state->vertex_buffers[index] = binding;
                UpdateVertexBufferSlot(index, binding);
            }
        }
        for (u32 mask = enabled_vertex_buffers_mask; mask != 0; mask &= mask - 1) {
            const u32 index = static_cast<u32>(std::countr_zero(mask));
            const Binding& binding = VertexBufferSlot(index);
            const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.vertex, 0, index);
            if (kept != nullptr && binding.buffer_id.index == kept->buffer_id &&
                usable(kept, binding.device_addr, binding.size)) {
                replay_skip_vertex_mask |= 1U << index;
            }
        }
        // Uniform, storage and texture buffers (Update*Buffers).
        for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
                Binding& binding = channel_state->uniform_buffers[stage][index];
                if (binding.buffer_id) {
                    return;
                }
                const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.uniform, stage, index);
                const u32 size = (std::min)(binding.size,
                                            (*channel_state->uniform_buffer_sizes)[stage][index]);
                // FindBuffer would return a buffer holding the whole binding: the recorded one
                // still does when it covers it.
                const bool covers =
                    kept != nullptr && kept->kind != DC::BufferBind::Kind::Normal &&
                    usable(kept, binding.device_addr, size) && [&] {
                        const Buffer& buffer = slot_buffers[BufferId{kept->buffer_id}];
                        return buffer.CpuAddr() <= binding.device_addr &&
                               binding.device_addr + binding.size <=
                                   buffer.CpuAddr() + buffer.SizeBytes();
                    }();
                binding.buffer_id = covers ? BufferId{kept->buffer_id}
                                           : FindBuffer(binding.device_addr, binding.size, false);
            });
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                Binding& binding = channel_state->storage_buffers[stage][index];
                const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.storage, stage, index);
                if (kept != nullptr && kept->device_addr == binding.device_addr &&
                    kept->size == binding.size && usable(kept, binding.device_addr, binding.size)) {
                    binding.buffer_id = BufferId{kept->buffer_id};
                } else {
                    binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
                    const bool is_written =
                        ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
                    ResolveMultiRangeStorage(binding, is_written, graphics_segments);
                }
            });
            ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
                Binding& binding = channel_state->texture_buffers[stage][index];
                const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.texture, stage, index);
                binding.buffer_id = kept != nullptr && kept->device_addr == binding.device_addr &&
                                            kept->size == binding.size &&
                                            usable(kept, binding.device_addr, binding.size)
                                        ? BufferId{kept->buffer_id}
                                        : FindBuffer(binding.device_addr, binding.size, false);
            });
        }
        if (channel_state->has_deleted_buffers) {
            // A lookup joined or recreated buffers: the existing loop resolves everything again.
            UpdateGraphicsBuffers(is_indexed);
            replay_skip_index = false;
            replay_skip_vertex_mask = 0;
        }
        RefreshMultiRangeSegments(graphics_segments);

        // BindHostGeometryBuffers.
        if (is_indexed) {
            Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
            TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
            const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
            const u32 size = channel_state->index_buffer.size;
            if (replay_skip_index) {
                ++replay_binds_replayed;
            } else {
                ++replay_binds_synced;
                SynchronizeGeometryBuffer(NUM_VERTEX_BUFFERS, channel_state->index_buffer.buffer_id,
                                          channel_state->index_buffer.device_addr, size);
            }
            V2NoteUsage(buffer, offset, size);
            runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format,
                                    draw_state.index_buffer.first, draw_state.index_buffer.count,
                                    buffer, offset, size);
        } else if (draw_state.topology == Maxwell::PrimitiveTopology::Quads ||
                   draw_state.topology == Maxwell::PrimitiveTopology::QuadStrip) {
            runtime.BindQuadIndexBuffer(draw_state.topology, draw_state.vertex_buffer.first,
                                        draw_state.vertex_buffer.count);
        }
        RefreshRepackedVertexBuffers();
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = static_cast<u32>(std::countr_zero(enabled_mask));
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            if (((replay_skip_vertex_mask >> index) & 1) != 0) {
                ++replay_binds_replayed;
            } else {
                ++replay_binds_synced;
                SynchronizeGeometryBuffer(index, binding.buffer_id, binding.device_addr,
                                          binding.size);
            }
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            V2NoteUsage(buffer, offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
        return true;
    } else {
        static_cast<void>(recipe);
        static_cast<void>(is_indexed);
        static_cast<void>(memo);
        return false;
    }
}

template <class P>
void BufferCache<P>::ReplayHostStageBuffers(size_t stage,
                                            const VideoCommon::V2::DrawCache::BufferRecipe& recipe,
                                            VideoCommon::V2::Versions::EpochMemo& memo) {
    namespace DC = VideoCommon::V2::DrawCache;
    if constexpr (!IS_OPENGL && !HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT &&
                  !NEEDS_BIND_UNIFORM_INDEX && !NEEDS_BIND_STORAGE_INDEX &&
                  !HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS &&
                  VideoCommon::StreamUniformReusable<Runtime>) {
        const u32 stage_index = static_cast<u32>(stage);
        const auto unchanged = [&](DAddr address, u64 size) {
            return replay_all_unchanged || memo.Unchanged(address, size, recipe.stamp);
        };
        // The binding still uses the recorded buffer object (same slot and generation) for a
        // range within the recorded one, unchanged since the recipe's stamp.
        const auto replayable = [&](const DC::BufferBind* kept, const Binding& binding, u32 size) {
            return kept != nullptr && binding.buffer_id.index == kept->buffer_id &&
                   (replay_all_alive ||
                    VideoCommon::V2::Versions::Gens().Get(VideoCommon::V2::Versions::Kind::Buffer,
                                                          kept->buffer_id) == kept->generation) &&
                   kept->Contains(binding.device_addr, size) &&
                   unchanged(binding.device_addr, size);
        };
        // Uniform buffers (BindHostGraphicsUniformBuffers).
        ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
            const Binding& binding = channel_state->uniform_buffers[stage][index];
            const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.uniform, stage_index,
                                                                      index);
            const u32 size =
                (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
            // F2-X1: only on the path the bind takes now. A range the GPU wrote since the recipe
            // (a version holds guest memory, the host buffer is newer) or one the guest memory
            // caught up with (the stream path again) is bound as BindHostGraphicsUniformBuffer
            // binds it.
            const DC::BufferBind::Kind path =
                kept != nullptr ? UniformBindKind(stage, index, binding) : DC::BufferBind::Kind::Normal;
            if (kept != nullptr && kept->kind == DC::BufferBind::Kind::Version &&
                path == DC::BufferBind::Kind::Version &&
                kept->tick == runtime.CurrentTick() && kept->device_addr == binding.device_addr &&
                kept->size == size && binding.buffer_id.index == kept->buffer_id &&
                VideoCommon::V2::Versions::Gens().Get(VideoCommon::V2::Versions::Kind::Buffer,
                                                      kept->buffer_id) == kept->generation) {
                // The range's version in this command buffer, while it still holds the bytes
                // guest memory has (the version table compares them): no copy.
                using Ref = typename Runtime::StreamUniformRef;
                if (const Ref* const version = VideoCommon::V2::CbufVer::Table<Ref>().FindWith(
                        binding.device_addr, size, runtime.CurrentTick(),
                        [&] { return V2GuestBytes(binding.device_addr, size); })) {
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << index;
                    channel_state->uniform_buffer_binding_sizes[stage][index] = size;
                    runtime.RebindStreamUniform(*version, size);
                    ++replay_binds_replayed;
                    return;
                }
            }
            if (kept != nullptr && kept->kind == DC::BufferBind::Kind::Cached &&
                path == DC::BufferBind::Kind::Cached && replayable(kept, binding, size)) {
                Buffer& buffer = slot_buffers[binding.buffer_id];
                TouchBuffer(buffer, binding.buffer_id);
                const u32 offset = buffer.Offset(binding.device_addr);
                V2NoteUsage(buffer, offset, size);
                runtime.BindUniformBuffer(buffer, offset, size);
                channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << index);
                ++replay_binds_replayed;
                return;
            }
            ++replay_binds_synced;
            BindHostGraphicsUniformBuffer(stage, index, 0, true);
        });
        // Storage buffers (BindHostGraphicsStorageBuffers).
        if (channel_state->enabled_storage_buffers[stage] != 0) {
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& binding = channel_state->storage_buffers[stage][index];
                const bool is_written =
                    ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
                if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
                    ++replay_binds_synced;
                    return;
                }
                Buffer& buffer = slot_buffers[binding.buffer_id];
                TouchBuffer(buffer, binding.buffer_id);
                const u32 size = binding.size;
                const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.storage,
                                                                          stage_index, index);
                if (!is_written && replayable(kept, binding, size)) {
                    ++replay_binds_replayed;
                } else {
                    ++replay_binds_synced;
                    SynchronizeBuffer(buffer, binding.device_addr, size);
                }
                const u32 offset = buffer.Offset(binding.device_addr);
                V2NoteUsage(buffer, offset, size);
                if (is_written) {
                    MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
                }
                runtime.BindStorageBuffer(buffer, offset, size, is_written);
            });
        }
        // Texture buffers (BindHostGraphicsTextureBuffers).
        if (channel_state->enabled_texture_buffers[stage] != 0) {
            ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
                const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
                Buffer& buffer = slot_buffers[binding.buffer_id];
                const u32 size = binding.size;
                const bool is_written =
                    ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
                const DC::BufferBind* const kept = DC::BufferRecipe::Find(recipe.texture,
                                                                          stage_index, index);
                if (!is_written && replayable(kept, binding, size)) {
                    ++replay_binds_replayed;
                } else {
                    ++replay_binds_synced;
                    SynchronizeBuffer(buffer, binding.device_addr, size);
                }
                if (is_written) {
                    MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
                }
                const u32 offset = buffer.Offset(binding.device_addr);
                V2NoteUsage(buffer, offset, size);
                if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
                    if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                        runtime.BindImageBuffer(buffer, offset, size, binding.format);
                    } else {
                        runtime.BindTextureBuffer(buffer, offset, size, binding.format);
                    }
                } else {
                    runtime.BindTextureBuffer(buffer, offset, size, binding.format);
                }
            });
        }
    } else {
        static_cast<void>(recipe);
        static_cast<void>(memo);
        BindHostStageBuffers(stage);
    }
}

template <class P>
u32 BufferCache<P>::CountStaleRecipeRanges(const VideoCommon::V2::DrawCache::BufferRecipe& recipe,
                                           VideoCommon::V2::Versions::EpochMemo& memo) {
    if (!recipe.valid) {
        return 0;
    }
    u32 stale = 0;
    using BindKind = VideoCommon::V2::DrawCache::BufferBind::Kind;
    const auto& gens = VideoCommon::V2::Versions::Gens();
    const auto check = [&](const VideoCommon::V2::DrawCache::BufferBind& bind) {
        // Only bindings the replay takes from the host buffer: stream copies (Normal) and
        // constant buffer versions (Version) read guest memory, never sync the buffer, and keep
        // its CPU-modified bits set by design. A binding whose buffer slot moved on (generation)
        // is never replayed.
        if (bind.kind == BindKind::Normal || bind.kind == BindKind::Version || bind.size == 0 ||
            !slot_buffers.contains(BufferId{bind.buffer_id}) ||
            gens.Get(VideoCommon::V2::Versions::Kind::Buffer, bind.buffer_id) != bind.generation) {
            return;
        }
        if (memo.Unchanged(bind.device_addr, bind.size, recipe.stamp) &&
            memory_tracker.IsRegionCpuModified(bind.device_addr, bind.size)) {
            ++stale;
        }
    };
    if (recipe.has_index) {
        check(recipe.index);
    }
    for (const auto* binds : {&recipe.vertex, &recipe.uniform, &recipe.storage, &recipe.texture}) {
        for (const auto& bind : *binds) {
            check(bind);
        }
    }
    return stale;
}

template <class P>
VideoCommon::V2::DrawCache::BufferBind::Kind BufferCache<P>::UniformBindKind(
    size_t stage, u32 index, const Binding& binding) {
    using Kind = VideoCommon::V2::DrawCache::BufferBind::Kind;
    // BindHostGraphicsUniformBuffer's decisions, in its order.
    const u32 bank_size = (*channel_state->uniform_buffer_sizes)[stage][index];
    const u32 size = (std::min)(binding.size, bank_size);
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const bool native_bank = has_host_buffer && bank_size == 0x10000 && gpu_memory != nullptr &&
                             gpu_memory->UsesNativeTiling();
    if ((index == 17 && maxwell3d->root_tables.loaded && binding.size == 0) || native_bank ||
        !has_host_buffer) {
        return Kind::Normal;
    }
    bool needs_alignment_stream = false;
    if constexpr (!IS_OPENGL) {
        const u32 offset = slot_buffers[binding.buffer_id].Offset(binding.device_addr);
        const u32 alignment = runtime.GetUniformBufferAlignment();
        needs_alignment_stream = alignment > 1 && (offset % alignment) != 0;
    }
    const bool use_fast_buffer =
        needs_alignment_stream ||
        (size <= channel_state->uniform_buffer_skip_cache_size &&
         !(memory_tracker.IsRegionGpuModified(binding.device_addr, size) ||
           (v2_defer && V2PendingGpuWrite(binding.device_addr, size))));
    return use_fast_buffer ? Kind::Version : Kind::Cached;
}

// Pipeline v2 P6 Step 0 (EDEN_NX2_V2_DRAWREC): pure twins and record appliers.

template <class P>
BufferId BufferCache<P>::PeekBuffer(DAddr device_addr, u32 size, bool sparse_compatible) const {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    // F2-20: the owner's page table (FindBuffer's commit branch).
    const BufferId found = page_table[page];
    if (!found || !slot_buffers[found].IsInBounds(device_addr, size)) {
        return BufferId{};
    }
    if constexpr (requires { slot_buffers[found].IsSparseCompatible(); }) {
        if (sparse_compatible && !slot_buffers[found].IsSparseCompatible()) {
            return BufferId{};
        }
    }
    return found;
}

template <class P>
V2::Commit::Request BufferCache<P>::MakeCreateRequest(DAddr device_addr, u32 size,
                                                      bool sparse_compatible) const {
    V2::Commit::Request request{};
    request.kind = V2::Commit::Kind::BufferCreate;
    request.resource = V2::Commit::Resource::Buffer;
    request.address = device_addr;
    request.size = size;
    request.arg = sparse_compatible ? 1U : 0U;
    const u64 begin = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    const u64 end = Common::AlignUp(device_addr + (std::max)(size, 1U), CACHING_PAGESIZE);
    bool merges = false;
    for (u64 at = begin; at < end && !merges; at += CACHING_PAGESIZE) {
        merges = v2_pages->Load(at) != 0;
    }
    request.effects = V2::Commit::EffectCreate | (merges ? V2::Commit::EffectDelete : 0);
    return request;
}

template <class P>
template <class Memory>
auto BufferCache<P>::IndexRange(Memory& memory, GPUVAddr begin, GPUVAddr end, u32 first,
                                u32 count, u32 format_size) -> GeometryRange {
    const std::optional<DAddr> device_addr = memory.GpuToCpuAddress(begin);
    const u32 address_size = static_cast<u32>(end - begin);
    const u32 draw_size = (count + first) * format_size;
    const u32 size = (std::min)(address_size, draw_size);
    if (size == 0 || !device_addr) {
        return {};
    }
    return {*device_addr, size, true};
}

template <class P>
template <class Memory>
auto BufferCache<P>::VertexRange(Memory& memory, GPUVAddr begin, bool enabled,
                                 GPUVAddr limit_address, bool native, u64 native_size)
    -> GeometryRange {
    if (native && enabled && native_size > (std::numeric_limits<u32>::max)()) {
        throw std::runtime_error("Native vertex stream exceeds supported 32-bit buffer size");
    }
    const GPUVAddr end = native ? begin + native_size : limit_address + 1;
    const std::optional<DAddr> device_addr = memory.GpuToCpuAddress(begin);
    u32 size = static_cast<u32>(end - begin);
    if (!enabled || size == 0 || !device_addr) {
        return {};
    }
    if (!memory.IsWithinGPUAddressRange(end) || size >= 64_MiB) {
        size = static_cast<u32>(memory.MaxContinuousRange(begin, size));
    }
    return {*device_addr, size, true};
}

template <class P>
template <class View>
V2::Resolve::ResolvedIndex BufferCache<P>::ResolveIndexBinding(const View& view) const {
    // UpdateIndexBuffer: the draw state's index buffer is the registers' with the draw's
    // first and count (DrawManager).
    const auto& index_buffer = view.Regs().index_buffer;
    const V2::Resolve::ViewDraw draw = view.Draw();
    const GeometryRange range =
        IndexRange(*gpu_memory, index_buffer.StartAddress(), index_buffer.EndAddress(),
                   draw.first, draw.count, u32(index_buffer.FormatSizeInBytes()));
    V2::Resolve::ResolvedIndex index{};
    index.format = static_cast<u32>(index_buffer.format);
    index.first = draw.first;
    index.count = draw.count;
    if (!range.valid) {
        index.buffer = {NULL_BUFFER_ID.index, BufferGeneration(NULL_BUFFER_ID)};
        return index;
    }
    const BufferId id = PeekBuffer(range.device_addr, range.size, false);
    index.buffer = {id.index, BufferGeneration(id)};
    index.daddr = range.device_addr;
    index.size = range.size;
    index.valid = true;
    return index;
}

template <class P>
template <class View>
std::optional<V2::Resolve::ResolvedVertex> BufferCache<P>::ResolveVertexBinding(const View& view,
                                                                                u32 slot) const {
    const auto& regs = view.Regs();
    const auto& array = regs.vertex_streams[slot];
    const bool native = view.NativeTuring();
    const u64 native_size = native ? regs.NativeVertexStreamSize(slot) : 0;
    const GeometryRange range =
        VertexRange(*gpu_memory, array.Address(), array.enable != 0,
                    regs.vertex_stream_limits[slot].Address(), native, native_size);
    if (!range.valid) {
        return std::nullopt;
    }
    const BufferId id = PeekBuffer(range.device_addr, range.size, false);
    V2::Resolve::ResolvedVertex vertex{};
    vertex.slot = static_cast<u8>(slot);
    vertex.buffer = {id.index, BufferGeneration(id)};
    vertex.daddr = range.device_addr;
    vertex.size = range.size;
    vertex.stride = array.stride;
    return vertex;
}

template <class P>
template <class View>
V2::Resolve::ResolvedUbo BufferCache<P>::ResolveUniformBinding(const View& view, size_t stage,
                                                               u32 slot) const {
    // BindGraphicsUniformBuffer at the constant buffer bind (R-10: translated at the draw), then
    // UpdateUniformBuffers' lookup.
    V2::Resolve::ResolvedUbo ubo{};
    ubo.stage = static_cast<u8>(stage);
    ubo.slot = static_cast<u8>(slot);
    ubo.buffer = {NULL_BUFFER_ID.index, BufferGeneration(NULL_BUFFER_ID)};
    const V2::Resolve::CbufBinding cbuf = view.Cbuf(stage, slot);
    if (cbuf.enabled == 0) {
        return ubo;
    }
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf.address);
    if (!device_addr) {
        return ubo;
    }
    const BufferId id = PeekBuffer(*device_addr, cbuf.size, false);
    ubo.daddr = *device_addr;
    ubo.size = cbuf.size;
    ubo.buffer = {id.index, BufferGeneration(id)};
    return ubo;
}

template <class P>
template <class View>
V2::Resolve::ResolvedSsbo BufferCache<P>::ResolveStorageBinding(
    const View& view, size_t stage, u32 ssbo_index, u32 cbuf_index, u32 cbuf_offset,
    bool is_written, u32 indirect_offset, bool& multi_range) const {
    V2::Resolve::ResolvedSsbo ssbo{};
    ssbo.stage = static_cast<u8>(stage);
    ssbo.index = static_cast<u8>(ssbo_index);
    ssbo.written = is_written ? 1 : 0;
    ssbo.table_pointer = indirect_offset != ~0U ? 1 : 0;
    ssbo.buffer = {NULL_BUFFER_ID.index, BufferGeneration(NULL_BUFFER_ID)};
    multi_range = false;
    GPUVAddr ssbo_addr = view.Cbuf(stage, cbuf_index).address + cbuf_offset;
    if (indirect_offset != ~0U) {
        const GPUVAddr holder = gpu_memory->Read<u64>(ssbo_addr);
        if (holder == 0) {
            return ssbo;
        }
        ssbo_addr = holder + indirect_offset;
    }
    const Binding binding =
        StorageBufferBinding(ssbo_addr, cbuf_index, is_written,
                             view.NativeTuring() || indirect_offset != ~0U, indirect_offset != ~0U);
    const BufferId id = PeekBuffer(binding.device_addr, binding.size, false);
    ssbo.daddr = binding.device_addr;
    ssbo.gpu_addr = binding.gpu_addr;
    ssbo.size = binding.size;
    ssbo.buffer = {id.index, BufferGeneration(id)};
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        // ResolveMultiRangeStorage's test, through the range cache's pure peek: two or more
        // segments (or a range it never walked) can't be resolved without walking it.
        if (binding.gpu_addr != 0 && binding.size != 0 &&
            (!is_written || runtime.PrefersSparseSources())) {
            const VirtualSegments* const found =
                virtual_ranges.Peek(*gpu_memory, binding.gpu_addr, binding.size);
            multi_range = found == nullptr || found->size() >= 2;
        }
    }
    return ssbo;
}

template <class P>
V2::Resolve::ResolvedTbo BufferCache<P>::ResolveTextureBufferBinding(
    size_t stage, u32 index, GPUVAddr gpu_addr, u32 size, PixelFormat format, bool is_written,
    bool is_image) const {
    V2::Resolve::ResolvedTbo tbo{};
    tbo.stage = static_cast<u8>(stage);
    tbo.index = static_cast<u8>(index);
    tbo.written = is_written ? 1 : 0;
    // Kept only where the bindings keep it (BindGraphicsTextureBuffer).
    tbo.image = SEPARATE_IMAGE_BUFFERS_BINDINGS && is_image ? 1 : 0;
    tbo.gpu_addr = gpu_addr;
    // GetTextureBufferBinding, then UpdateTextureBuffers' lookup.
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (!device_addr || size == 0) {
        tbo.format = static_cast<u32>(PixelFormat::Invalid);
        tbo.buffer = {NULL_BUFFER_ID.index, BufferGeneration(NULL_BUFFER_ID)};
        return tbo;
    }
    const BufferId id = PeekBuffer(*device_addr, size, false);
    tbo.daddr = *device_addr;
    tbo.size = size;
    tbo.format = static_cast<u32>(format);
    tbo.buffer = {id.index, BufferGeneration(id)};
    return tbo;
}

template <class P>
u32 BufferCache<P>::ApplyDrawRequest(const V2::Commit::Request& request) {
    if (request.kind != V2::Commit::Kind::BufferCreate) {
        return request.slot;
    }
    // As V2ApplyCreates applies a creation: the lookup creates (joining what overlaps).
    const bool proxy_finds = std::exchange(v2_proxy_finds, false);
    const BufferId id =
        FindBuffer(request.address, static_cast<u32>(request.size), request.arg != 0);
    v2_proxy_finds = proxy_finds;
    return id.index;
}

template <class P>
void BufferCache<P>::EncodeGraphicsBindings(const V2::Resolve::ResolvedDraw& record) {
    namespace R = V2::Resolve;
    const auto binding_of = [](DAddr device_addr, u32 size, const V2::Versions::WeakRef& buffer) {
        return Binding{
            .device_addr = device_addr,
            .size = size,
            .buffer_id = BufferId{buffer.id},
        };
    };
    BufferOperations([&]() {
        graphics_segments.clear();
        const R::DrawHeader& h = record.h;
        if (h.indexed) {
            channel_state->index_buffer =
                h.index.valid ? binding_of(h.index.daddr, h.index.size, h.index.buffer)
                              : NULL_BINDING;
        }
        // Every slot as UpdateVertexBuffer leaves it: the record's, NULL for disabled ones.
        std::array<const R::ResolvedVertex*, NUM_VERTEX_BUFFERS> vertex{};
        for (const R::ResolvedVertex& entry : record.Span<R::ResolvedVertex>(h.vertex)) {
            if (entry.slot < NUM_VERTEX_BUFFERS) {
                vertex[entry.slot] = &entry;
            }
        }
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding binding = vertex[index] != nullptr
                                        ? binding_of(vertex[index]->daddr, vertex[index]->size,
                                                     vertex[index]->buffer)
                                        : NULL_BINDING;
            channel_state->vertex_buffers[index] = binding;
            UpdateVertexBufferSlot(index, binding);
        }
        for (const R::ResolvedUbo& ubo : record.Span<R::ResolvedUbo>(h.ubo)) {
            channel_state->uniform_buffers[ubo.stage][ubo.slot] =
                ubo.buffer.id == NULL_BUFFER_ID.index && ubo.size == 0
                    ? NULL_BINDING
                    : binding_of(ubo.daddr, ubo.size, ubo.buffer);
        }
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            if (stage < h.stages.size() && h.stages[stage]) {
                // UnbindGraphicsStorageBuffers / UnbindGraphicsTextureBuffers (no dynamic
                // storage buffer limit under DRAWREC).
                channel_state->enabled_storage_buffers[stage] = 0;
                channel_state->written_storage_buffers[stage] = 0;
                channel_state->enabled_texture_buffers[stage] = 0;
                channel_state->written_texture_buffers[stage] = 0;
                channel_state->image_texture_buffers[stage] = 0;
            }
        }
        for (const R::ResolvedSsbo& ssbo : record.Span<R::ResolvedSsbo>(h.ssbo)) {
            channel_state->enabled_storage_buffers[ssbo.stage] |= u64{1} << ssbo.index;
            channel_state->written_storage_buffers[ssbo.stage] |= u64{ssbo.written} << ssbo.index;
            Binding binding = binding_of(ssbo.daddr, ssbo.size, ssbo.buffer);
            binding.gpu_addr = ssbo.gpu_addr;
            if (ssbo.daddr == 0 && ssbo.size == 0) {
                binding = NULL_BINDING;
            }
            channel_state->storage_buffers[ssbo.stage][ssbo.index] = binding;
        }
        for (const R::ResolvedTbo& tbo : record.Span<R::ResolvedTbo>(h.tbo)) {
            channel_state->enabled_texture_buffers[tbo.stage] |= 1U << tbo.index;
            channel_state->written_texture_buffers[tbo.stage] |= u32{tbo.written} << tbo.index;
            if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
                channel_state->image_texture_buffers[tbo.stage] |= u32{tbo.image} << tbo.index;
            }
            TextureBufferBinding binding{};
            binding.device_addr = tbo.daddr;
            binding.size = tbo.size;
            binding.buffer_id = BufferId{tbo.buffer.id};
            binding.format = static_cast<PixelFormat>(tbo.format);
            channel_state->texture_buffers[tbo.stage][tbo.index] = binding;
        }
        auto& flags = maxwell3d->dirty.flags;
        flags[Dirty::VertexBuffers] = false;
        flags[Dirty::IndexBuffer] = false;
    });
}

template <class P>
void BufferCache<P>::ObserveGraphicsBindings(V2::Resolve::ResolvedDraw& observed, bool is_indexed,
                                             const std::array<bool, NUM_STAGES>& stages) const {
    namespace R = V2::Resolve;
    const auto ref_of = [&](BufferId id) {
        return V2::Versions::WeakRef{id.index, BufferGeneration(id)};
    };
    R::DrawHeader& h = observed.h;
    if (is_indexed) {
        const Binding& index = channel_state->index_buffer;
        h.index.valid = index.device_addr != 0 || index.size != 0;
        h.index.daddr = index.device_addr;
        h.index.size = index.size;
        h.index.buffer = ref_of(index.buffer_id);
    }
    boost::container::small_vector<R::ResolvedVertex, NUM_VERTEX_BUFFERS> vertex;
    for (u32 slot = 0; slot < NUM_VERTEX_BUFFERS; ++slot) {
        const Binding& binding = channel_state->vertex_buffers[slot];
        if (binding.device_addr == 0 && binding.size == 0) {
            continue;
        }
        vertex.push_back(R::ResolvedVertex{
            .slot = static_cast<u8>(slot),
            .buffer = ref_of(binding.buffer_id),
            .daddr = binding.device_addr,
            .size = binding.size,
            .stride = maxwell3d->regs.vertex_streams[slot].stride,
        });
    }
    h.vertex_mask = enabled_vertex_buffers_mask;
    boost::container::small_vector<R::ResolvedUbo, 32> ubos;
    boost::container::small_vector<R::ResolvedSsbo, 32> ssbos;
    boost::container::small_vector<R::ResolvedTbo, 16> tbos;
    for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
        h.ubo_masks[stage] = 0;
        if (!stages[stage]) {
            continue;
        }
        h.ubo_masks[stage] = channel_state->enabled_uniform_buffer_masks[stage];
        ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
            const Binding& binding = channel_state->uniform_buffers[stage][index];
            ubos.push_back(R::ResolvedUbo{
                .stage = static_cast<u8>(stage),
                .slot = static_cast<u8>(index),
                .daddr = binding.device_addr,
                .size = binding.size,
                .buffer = ref_of(binding.buffer_id),
            });
        });
        ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
            const Binding& binding = channel_state->storage_buffers[stage][index];
            ssbos.push_back(R::ResolvedSsbo{
                .stage = static_cast<u8>(stage),
                .index = static_cast<u8>(index),
                .written = static_cast<u8>(
                    (channel_state->written_storage_buffers[stage] >> index) & 1),
                .daddr = binding.device_addr,
                .gpu_addr = binding.gpu_addr,
                .size = binding.size,
                .buffer = ref_of(binding.buffer_id),
            });
        });
        ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
            const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
            tbos.push_back(R::ResolvedTbo{
                .stage = static_cast<u8>(stage),
                .index = static_cast<u8>(index),
                .written = static_cast<u8>(
                    (channel_state->written_texture_buffers[stage] >> index) & 1),
                .image = static_cast<u8>(
                    SEPARATE_IMAGE_BUFFERS_BINDINGS
                        ? (channel_state->image_texture_buffers[stage] >> index) & 1
                        : 0),
                .daddr = binding.device_addr,
                .size = binding.size,
                .format = static_cast<u32>(binding.format),
                .buffer = ref_of(binding.buffer_id),
            });
        });
    }
    h.vertex = observed.Push(std::span<const R::ResolvedVertex>(vertex.data(), vertex.size()));
    h.ubo = observed.Push(std::span<const R::ResolvedUbo>(ubos.data(), ubos.size()));
    h.ssbo = observed.Push(std::span<const R::ResolvedSsbo>(ssbos.data(), ssbos.size()));
    h.tbo = observed.Push(std::span<const R::ResolvedTbo>(tbos.data(), tbos.size()));
}

template <class P>
void BufferCache<P>::ReplayGraphicsStorageBuffer(size_t stage, size_t ssbo_index,
                                                 DAddr device_addr, GPUVAddr gpu_addr, u32 size,
                                                 u32 segment_first, u32 segment_count,
                                                 bool is_written) {
    if (stage >= NUM_STAGES || ssbo_index >= NUM_STORAGE_BUFFERS) {
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_storage_buffers[stage] >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            if (channel_state->total_graphics_storage_buffers >=
                runtime.GetMaxDynamicStorageBuffers()) {
                return;
            }
            ++channel_state->total_graphics_storage_buffers;
        }
    }
    channel_state->enabled_storage_buffers[stage] |= u64{1} << ssbo_index;
    channel_state->written_storage_buffers[stage] |= u64{is_written ? 1U : 0U} << ssbo_index;
    // The buffer is found again by UpdateGraphicsBuffers, as for any binding.
    channel_state->storage_buffers[stage][ssbo_index] = Binding{
        .device_addr = device_addr,
        .gpu_addr = gpu_addr,
        .size = size,
        .buffer_id = NULL_BUFFER_ID,
        .segment_first = segment_first,
        .segment_count = segment_count,
    };
}

template <class P>
void BufferCache<P>::UnbindGraphicsTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) {
        return;
    }
    channel_state->enabled_texture_buffers[stage] = 0;
    channel_state->written_texture_buffers[stage] = 0;
    channel_state->image_texture_buffers[stage] = 0;
}

template <class P>
void BufferCache<P>::BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr,
                                               u32 size, PixelFormat format, bool is_written,
                                               bool is_image) {
    channel_state->enabled_texture_buffers[stage] |= 1U << tbo_index;
    channel_state->written_texture_buffers[stage] |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_texture_buffers[stage] |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->texture_buffers[stage][tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::UnbindComputeStorageBuffers() {
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers()) {
            channel_state->total_compute_storage_buffers -=
                static_cast<u32>(std::popcount(channel_state->enabled_compute_storage_buffers));
        }
    }
    channel_state->enabled_compute_storage_buffers = 0;
    channel_state->written_compute_storage_buffers = 0;
    channel_state->native_compute_storage_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                              bool is_written, u32 indirect_offset,
                                              const Shader::StorageWriteExtent& write_extent) {
    if (ssbo_index >= channel_state->compute_storage_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Storage buffer index {} exceeds maximum storage buffer count",
                  ssbo_index);
        return;
    }
    const bool already_enabled =
        ((channel_state->enabled_compute_storage_buffers >> ssbo_index) & 1U) != 0;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            const u32 max_bindings = runtime.GetMaxDynamicStorageBuffers();
            if (channel_state->total_compute_storage_buffers >= max_bindings) {
                LOG_WARNING(HW_GPU,
                            "Skipping compute storage buffer {} due to driver limit {}",
                            ssbo_index, max_bindings);
                return;
            }
        }
    }
    channel_state->enabled_compute_storage_buffers |= u64{1} << ssbo_index;
    channel_state->written_compute_storage_buffers |= u64{is_written ? 1U : 0U} << ssbo_index;
    if constexpr (requires { runtime.ShouldLimitDynamicStorageBuffers(); }) {
        if (runtime.ShouldLimitDynamicStorageBuffers() && !already_enabled) {
            ++channel_state->total_compute_storage_buffers;
        }
    }

    const auto& launch_desc = kepler_compute->launch_description;
    if (((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) == 0) {
        LOG_WARNING(HW_GPU, "Skipped binding SSBO: cbuf index {} is not enabled", cbuf_index);
        return;
    }
    ASSERT(((launch_desc.const_buffer_enable_mask >> cbuf_index) & 1) != 0);

    const auto& cbufs = launch_desc.const_buffer_config;
    GPUVAddr ssbo_addr = cbufs[cbuf_index].Address() + cbuf_offset;
    GPUVAddr holder{};
    const bool table_pointer = indirect_offset != ~0U;
    if (table_pointer) {
        holder = gpu_memory->Read<u64>(ssbo_addr);
        ssbo_addr = holder + indirect_offset;
    }
    const bool native_pointer = kepler_compute->IsNativeMode();
    channel_state->native_compute_storage_buffers =
        (channel_state->native_compute_storage_buffers & ~(u64{1} << ssbo_index)) |
        (u64{native_pointer ? 1U : 0U} << ssbo_index);
    const Binding binding =
        table_pointer && holder == 0
            ? NULL_BINDING
            : StorageBufferBinding(ssbo_addr, cbuf_index, is_written, native_pointer,
                                   table_pointer);
    channel_state->compute_storage_buffers[ssbo_index] = binding;
    if (table_pointer && holder != 0) {
        TraceTablePointer(true, 0, ssbo_index, cbuf_index, cbuf_offset, indirect_offset, holder,
                          binding);
    }

    // How much of the binding the dispatch marks GPU-written. A native pointer has no size, so the
    // binding is up to 8 MiB past it, and marking all of it defers the CPU's later writes to those
    // pages until the next fence and has fences copy the over-marked range back.
    u64 mark = binding.size;
    if (native_pointer && is_written && binding.size != 0) {
        switch (Settings::values.native_compute_write_mark.GetValue()) {
        case Settings::NativeComputeWriteMark::Auto: {
            const auto& qmd = launch_desc;
            const std::array<u32, 3> block{qmd.block_dim_x.Value(), qmd.block_dim_y.Value(),
                                           qmd.block_dim_z.Value()};
            std::optional<std::array<u32, 3>> grid;
            if (!kepler_compute->GetIndirectComputeAddress()) {
                grid = std::array<u32, 3>{qmd.grid_dim_x.Value(), qmd.grid_dim_y.Value(),
                                          qmd.grid_dim_z.Value()};
            }
            if (const auto extent = Shader::ResolveStorageWriteExtent(write_extent, block, grid)) {
                // The extent counts from the pointer, the binding from the pointer aligned down.
                const GPUVAddr pointer = gpu_memory->Read<u64>(ssbo_addr);
                const u64 pad = pointer >= binding.gpu_addr ? pointer - binding.gpu_addr : 0;
                mark = std::min<u64>(binding.size, pad + *extent);
            }
            break;
        }
        case Settings::NativeComputeWriteMark::Full:
            break;
        }
    }
    channel_state->compute_storage_write_marks[ssbo_index] = static_cast<u32>(mark);
}

template <class P>
void BufferCache<P>::UnbindComputeTextureBuffers() {
    channel_state->enabled_compute_texture_buffers = 0;
    channel_state->written_compute_texture_buffers = 0;
    channel_state->image_compute_texture_buffers = 0;
}

template <class P>
void BufferCache<P>::BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                              PixelFormat format, bool is_written, bool is_image) {
    if (tbo_index >= channel_state->compute_texture_buffers.size()) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Texture buffer index {} exceeds maximum texture buffer count",
                  tbo_index);
        return;
    }
    channel_state->enabled_compute_texture_buffers |= 1U << tbo_index;
    channel_state->written_compute_texture_buffers |= (is_written ? 1U : 0U) << tbo_index;
    if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
        channel_state->image_compute_texture_buffers |= (is_image ? 1U : 0U) << tbo_index;
    }
    channel_state->compute_texture_buffers[tbo_index] =
        GetTextureBufferBinding(gpu_addr, size, format);
}

template <class P>
void BufferCache<P>::FlushCachedWrites() {
    memory_tracker.FlushCachedWrites();
}

template <class P>
bool BufferCache<P>::HasUncommittedFlushes() const noexcept {
    return !uncommitted_gpu_modified_ranges.Empty() || !committed_gpu_modified_ranges.empty();
}

template <class P>
bool BufferCache<P>::FlushesAreEarlyDownloads() const {
    return Tegra::g_nx2_ampere.load(std::memory_order_relaxed) && !AsyncBufferDownloads();
}

template <class P>
void BufferCache<P>::AccumulateFlushes() {
    if (uncommitted_gpu_modified_ranges.Empty()) {
        return;
    }
    committed_gpu_modified_ranges.emplace_back(std::move(uncommitted_gpu_modified_ranges));
}

template <class P>
bool BufferCache<P>::ShouldWaitAsyncFlushes() const noexcept {
    return (!async_buffers.empty() && async_buffers.front().has_value());
}

template <class P>
void BufferCache<P>::RecordWriterDownloads() {
    // EDEN_NX2_V2_XREADBACK: the fence downloads queued so far (learned pages the guest reads
    // back) are recorded now, right after the work that wrote them and in its submission, and
    // land with the next fence's operations.
    if constexpr (!USE_MEMORY_MAPS) {
        return;
    } else {
        if (uncommitted_gpu_modified_ranges.Empty()) {
            return;
        }
        Common::RangeSet<DAddr> ranges = std::move(uncommitted_gpu_modified_ranges);
        uncommitted_gpu_modified_ranges.Clear();
        boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
        u64 total_size_bytes = 0;
        ranges.ForEach([&](DAddr lower, DAddr upper) {
            ForEachBufferInRange(lower, upper - lower, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_addr = buffer.CpuAddr();
                const DAddr begin = (std::max)(buffer_addr, lower);
                const DAddr end = (std::min)(buffer_addr + buffer.SizeBytes(), upper);
                if (begin >= end) {
                    return;
                }
                memory_tracker.ForEachDownloadRange(
                    begin, end - begin, false, [&](u64 range_addr, u64 range_size) {
                        gpu_modified_ranges.ForEachInRange(
                            range_addr, range_size, [&](DAddr start, DAddr stop) {
                                downloads.push_back({BufferCopy{
                                                         .src_offset = start - buffer_addr,
                                                         .dst_offset = total_size_bytes,
                                                         .size = stop - start,
                                                     },
                                                     buffer_id});
                                total_size_bytes += Common::AlignUp(stop - start, 64);
                            });
                    });
            });
        });
        if (downloads.empty()) {
            return;
        }
        PreparedDownload prepared;
        prepared.staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
        runtime.PreCopyBarrier();
        for (auto& [copy, buffer_id] : downloads) {
            Buffer& buffer = slot_buffers[buffer_id];
            copy.dst_offset += prepared.staging.offset;
            BufferCopy normalized{copy};
            normalized.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
            prepared.ranges.Add(normalized.src_offset, copy.size);
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(prepared.staging.buffer, buffer, std::span{&copy, 1}, false);
            prepared.copies.push_back(normalized);
        }
        runtime.PostCopyBarrier();
        VideoCommon::V2::P5bStats::Add(VideoCommon::V2::P5bStats::Counter::EarlyDownloads,
                                       prepared.copies.size());
        writer_downloads_open.push_back(std::move(prepared));
    }
}

template <class P>
void BufferCache<P>::CommitAsyncFlushesHigh() {
    if (v2_writer_downloads) [[unlikely]] {
        // The downloads prepared behind their writers land with this fence.
        writer_downloads_fenced.push_back(std::move(writer_downloads_open));
        writer_downloads_open.clear();
    }
    AccumulateFlushes();

    if (committed_gpu_modified_ranges.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }

    auto it = committed_gpu_modified_ranges.begin();
    while (it != committed_gpu_modified_ranges.end()) {
        auto& current_intervals = *it;
        auto next_it = std::next(it);
        while (next_it != committed_gpu_modified_ranges.end()) {
            next_it->ForEach([&current_intervals](DAddr start, DAddr end) {
                current_intervals.Subtract(start, end - start);
            });
            next_it++;
        }
        it++;
    }

    boost::container::small_vector<std::pair<BufferCopy, BufferId>, 16> downloads;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    for (const Common::RangeSet<DAddr>& range_set : committed_gpu_modified_ranges) {
        range_set.ForEach([&](DAddr interval_lower, DAddr interval_upper) {
            const std::size_t size = interval_upper - interval_lower;
            const DAddr device_addr = interval_lower;
            ForEachBufferInRange(device_addr, size, [&](BufferId buffer_id, Buffer& buffer) {
                const DAddr buffer_start = buffer.CpuAddr();
                const DAddr buffer_end = buffer_start + buffer.SizeBytes();
                const DAddr new_start = (std::max)(buffer_start, device_addr);
                const DAddr new_end = (std::min)(buffer_end, device_addr + size);
                memory_tracker.ForEachDownloadRange(
                    new_start, new_end - new_start, false,
                    [&](u64 device_addr_out, u64 range_size) {
                        const DAddr buffer_addr = buffer.CpuAddr();
                        const auto add_download = [&](DAddr start, DAddr end) {
                            const u64 new_offset = start - buffer_addr;
                            const u64 new_size = end - start;
                            downloads.push_back({
                                BufferCopy{
                                    .src_offset = new_offset,
                                    .dst_offset = total_size_bytes,
                                    .size = new_size,
                                },
                                buffer_id,
                            });
                            // Align up to avoid cache conflicts
                            constexpr u64 align = 64ULL;
                            constexpr u64 mask = ~(align - 1ULL);
                            total_size_bytes += (new_size + align - 1) & mask;
                            largest_copy = (std::max)(largest_copy, new_size);
                        };

                        gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                           add_download);
                    });
            });
        });
    }
    committed_gpu_modified_ranges.clear();
    if (downloads.empty()) {
        async_buffers.emplace_back(std::optional<Async_Buffer>{});
        return;
    }
    Nx2Profile::Add(Nx2Profile::Counter::DownloadBytes, total_size_bytes);
    Nx2Profile::Add(Nx2Profile::Counter::DownloadCopies, downloads.size());
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes, true);
    boost::container::small_vector<BufferCopy, 4> normalized_copies;
    Common::RangeSet<DAddr> download_ranges;
    runtime.PreCopyBarrier();
    for (auto& [copy, buffer_id] : downloads) {
        copy.dst_offset += download_staging.offset;
        const std::array copies{copy};
        BufferCopy second_copy{copy};
        Buffer& buffer = slot_buffers[buffer_id];
        second_copy.src_offset = static_cast<size_t>(buffer.CpuAddr()) + copy.src_offset;
        const DAddr orig_device_addr = static_cast<DAddr>(second_copy.src_offset);
        if (Nx2Profile::Enabled()) {
            u64 on_dma_pages = 0;
            const auto& pages = Nx2Profile::DmaReadPages();
            for (DAddr page = orig_device_addr >> 12; page <= (orig_device_addr + copy.size - 1) >> 12;
                 ++page) {
                if (pages.contains(page)) {
                    const DAddr begin = (std::max)(orig_device_addr, page << 12);
                    const DAddr end = (std::min)(orig_device_addr + copy.size, (page + 1) << 12);
                    on_dma_pages += end - begin;
                }
            }
            Nx2Profile::Add(Nx2Profile::Counter::DownloadBytesOnDmaPages, on_dma_pages);
        }
        if (Nx2DownloadTrace::g_enabled) [[unlikely]] {
            Nx2DownloadTrace::Copied(orig_device_addr, copy.size);
        }
        download_ranges.Add(orig_device_addr, copy.size);
        buffer.MarkUsage(copy.src_offset, copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, copies, false);
        normalized_copies.push_back(second_copy);
    }
    runtime.PostCopyBarrier();
    pending_downloads.emplace_back(std::move(normalized_copies));
    pending_download_ranges.emplace_back(std::move(download_ranges));
    async_buffers.emplace_back(download_staging);
}

template <class P>
void BufferCache<P>::CommitAsyncFlushes() {
    CommitAsyncFlushesHigh();
}

template <class P>
void BufferCache<P>::PopAsyncFlushes() {
    PopAsyncBuffers();
}

template <class P>
void BufferCache<P>::ClaimCpuWrites(std::span<const std::pair<DAddr, u64>> ranges) {
    for (const auto& [device_addr, size] : ranges) {
        if (IsRegionGpuModified(device_addr, size)) {
            ClearDownload(device_addr, size);
            gpu_modified_ranges.Subtract(device_addr, size);
        }
    }
}

template <class P>
Common::RangeSet<DAddr> BufferCache<P>::PendingCpuWriteRanges() {
    Common::RangeSet<DAddr> ranges;
    if (pending_cpu_writes) {
        pending_cpu_writes(
            [&ranges](DAddr device_addr, size_t size) { ranges.Add(device_addr, size); });
    }
    return ranges;
}

template <class P>
void BufferCache<P>::WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written,
                                           DAddr device_addr, const u8* data, u64 size) {
    // A download changes guest memory (handle words, stream-copied uniforms read it).
    V2::Versions::HookWrite(device_addr, size, V2::Versions::Source::Gpu);
    if (!cpu_written.Intersects(device_addr, size)) {
        device_memory.WriteBlockUnsafe(device_addr, data, size);
        return;
    }
    Common::RangeSet<DAddr> parts;
    parts.Add(device_addr, size);
    cpu_written.ForEachInRange(device_addr, size, [&parts](DAddr start, DAddr end) {
        parts.Subtract(start, end - start);
    });
    parts.ForEach([&](DAddr start, DAddr end) {
        device_memory.WriteBlockUnsafe(start, data + (start - device_addr), end - start);
    });
}

template <class P>
void BufferCache<P>::PopAsyncBuffers() {
    if (v2_writer_downloads && !writer_downloads_fenced.empty() && !async_buffers.empty())
        [[unlikely]] {
        if constexpr (USE_MEMORY_MAPS) {
            const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
            for (auto& prepared : writer_downloads_fenced.front()) {
                u8* const base = prepared.staging.mapped_span.data();
                for (const BufferCopy& copy : prepared.copies) {
                    const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
                    const u8* const data = base + (copy.dst_offset - prepared.staging.offset);
                    prepared.ranges.ForEachInRange(device_addr, copy.size,
                                                   [&](DAddr start, DAddr end) {
                        WriteDownloadedMemory(cpu_written, start, &data[start - device_addr],
                                              end - start);
                        gpu_modified_ranges.Subtract(start, end - start);
                    });
                }
                async_buffers_death_ring.emplace_back(prepared.staging);
            }
        }
        writer_downloads_fenced.pop_front();
    }
    if (async_buffers.empty()) {
        return;
    }
    if (!async_buffers.front().has_value()) {
        async_buffers.pop_front();
        return;
    }
    auto& downloads = pending_downloads.front();
    auto& ranges = pending_download_ranges.front();
    auto& async_buffer = async_buffers.front();
    u8* base = async_buffer->mapped_span.data();
    const size_t base_offset = async_buffer->offset;
    // This runs on the fencing thread while the CPU keeps writing: CPU writes collected since the
    // copy was taken are newer than it, and the GPU thread hasn't dropped them from `ranges` yet.
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const auto& copy : downloads) {
        const DAddr device_addr = static_cast<DAddr>(copy.src_offset);
        const u64 dst_offset = copy.dst_offset - base_offset;
        const u8* read_mapped_memory = base + dst_offset;
        // Only what no write replaced since the copy was taken: the rest is older than guest
        // memory (CPU writes) or than the GPU's copy (GPU writes, still to be downloaded).
        ranges.ForEachInRange(device_addr, copy.size, [&](DAddr start, DAddr end) {
            WriteDownloadedMemory(cpu_written, start, &read_mapped_memory[start - device_addr],
                                  end - start);
            gpu_modified_ranges.Subtract(start, end - start);
        });
    }
    async_buffers_death_ring.emplace_back(*async_buffer);
    async_buffers.pop_front();
    pending_downloads.pop_front();
    pending_download_ranges.pop_front();
}

template <class P>
bool BufferCache<P>::IsRegionGpuModified(DAddr addr, size_t size) {
    // One lookup instead of visiting every range inside (~4% of the GPU thread for title 1's
    // inline uploads once GPU-written ranges are no longer cleared by downloads).
    return gpu_modified_ranges.Intersects(addr, size);
}

template <class P>
bool BufferCache<P>::IsRegionRegistered(DAddr addr, size_t size) {
    const DAddr end_addr = addr + size;
    const u64 page_end = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    for (u64 page = addr >> CACHING_PAGEBITS; page < page_end;) {
        const BufferId buffer_id = page_table[page];
        if (!buffer_id) {
            ++page;
            continue;
        }
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buf_start_addr = buffer.CpuAddr();
        const DAddr buf_end_addr = buf_start_addr + buffer.SizeBytes();
        if (buf_start_addr < end_addr && addr < buf_end_addr) {
            return true;
        }
        page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
    }
    return false;
}

template <class P>
bool BufferCache<P>::IsRegionCpuModified(DAddr addr, size_t size) {
    return memory_tracker.IsRegionCpuModified(addr, size);
}

template <class P>
void BufferCache<P>::BindHostIndexBuffer() {
    Buffer& buffer = slot_buffers[channel_state->index_buffer.buffer_id];
    TouchBuffer(buffer, channel_state->index_buffer.buffer_id);
    const u32 offset = buffer.Offset(channel_state->index_buffer.device_addr);
    const u32 size = channel_state->index_buffer.size;
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    if constexpr (requires { runtime.BindHostUint8IndexBuffer(std::span<const u8>{}); }) {
        if (draw_state.index_buffer.format == Maxwell::IndexFormat::UnsignedByte &&
            draw_state.topology != Maxwell::PrimitiveTopology::Quads &&
            draw_state.topology != Maxwell::PrimitiveTopology::QuadStrip &&
            runtime.WidensUint8IndicesOnHost()) {
            std::span<const u8> indices;
            const auto& index_buffer = draw_state.index_buffer;
            const u32 guest_size = (std::min)(
                static_cast<u32>(index_buffer.EndAddress() - index_buffer.StartAddress()),
                index_buffer.first + index_buffer.count);
            if (!draw_state.inline_index_draw_indexes.empty()) {
                indices = std::span{draw_state.inline_index_draw_indexes.data(), size};
            } else if (guest_size != 0 &&
                       !(IsRegionGpuModified(channel_state->index_buffer.device_addr, guest_size) ||
                         (v2_defer && V2PendingGpuWrite(channel_state->index_buffer.device_addr,
                                                        guest_size)))) {
                guest_uint8_indices.resize(guest_size);
                device_memory.ReadBlockUnsafe(channel_state->index_buffer.device_addr,
                                              guest_uint8_indices.data(), guest_size);
                indices = guest_uint8_indices;
            }
            if (!indices.empty()) {
                runtime.BindHostUint8IndexBuffer(indices);
                return;
            }
        }
    }
    if (draw_state.inline_index_draw_indexes.empty()) {
        SynchronizeGeometryBuffer(NUM_VERTEX_BUFFERS, channel_state->index_buffer.buffer_id,
                                  channel_state->index_buffer.device_addr, size);
    } else {
        if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
            auto upload_staging = runtime.UploadStagingBuffer(size);
            std::array<BufferCopy, 1> copies{{BufferCopy{.src_offset = upload_staging.offset, .dst_offset = 0, .size = size}}};
            std::memcpy(upload_staging.mapped_span.data(), draw_state.inline_index_draw_indexes.data(), size);
            runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true);
        } else {
            buffer.ImmediateUpload(0, draw_state.inline_index_draw_indexes);
        }
    }
    if constexpr (HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT) {
        const u32 new_offset = offset + draw_state.index_buffer.first * u32(draw_state.index_buffer.FormatSizeInBytes());
        runtime.BindIndexBuffer(buffer, new_offset, size);
    } else {
        V2NoteUsage(buffer, offset, size);
        runtime.BindIndexBuffer(draw_state.topology, draw_state.index_buffer.format, draw_state.index_buffer.first, draw_state.index_buffer.count, buffer, offset, size);
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size,
                                          u32 stride) {
    if constexpr (IS_OPENGL) {
        runtime.BindVertexBuffer(index, buffer, offset, size, stride);
    } else {
        runtime.BindVertexBuffer(index, buffer.Handle(), offset, size, stride);
    }
}

template <class P>
Binding& BufferCache<P>::VertexBufferSlot(u32 index) {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
const Binding& BufferCache<P>::VertexBufferSlot(u32 index) const {
    ASSERT(index < NUM_VERTEX_BUFFERS);
    return v_buffer[index];
}

template <class P>
void BufferCache<P>::UpdateVertexBufferSlot(u32 index, const Binding& binding) {
    Binding& slot = VertexBufferSlot(index);
    if (slot.device_addr != binding.device_addr || slot.size != binding.size) {
        ++vertex_buffers_serial;
    }
    slot = binding;
    if (binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
        enabled_vertex_buffers_mask |= (1u << index);
    } else {
        enabled_vertex_buffers_mask &= ~(1u << index);
    }
}

template <class P>
void BufferCache<P>::MarkVertexBuffersStale(DAddr device_addr, u64 size) {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = enabled_vertex_buffers_mask;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            const Binding& binding = v_buffer[index];
            if (device_addr < binding.device_addr + binding.size &&
                binding.device_addr < device_addr + size) {
                stale_vertex_buffers_mask |= 1U << index;
            }
        }
    }
}

template <class P>
void BufferCache<P>::RefreshRepackedVertexBuffers() {
    if constexpr (requires { runtime.IsVertexRepacked(u32{}); }) {
        u32 mask = std::exchange(stale_vertex_buffers_mask, 0) & enabled_vertex_buffers_mask;
        auto& flags = maxwell3d->dirty.flags;
        while (mask != 0) {
            const u32 index = std::countr_zero(mask);
            mask &= mask - 1;
            if (runtime.IsVertexRepacked(index)) {
                flags[Dirty::VertexBuffer0 + index] = true;
            }
        }
    }
}

template <class P>
void BufferCache<P>::BindHostVertexBuffers() {
    RefreshRepackedVertexBuffers();

#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        auto& flags = maxwell3d->dirty.flags;
        u32 enabled_mask = enabled_vertex_buffers_mask;
        HostBindings<Buffer> bindings{};
        u32 last_index = (std::numeric_limits<u32>::max)();
        const auto flush_bindings = [&]() {
            if (bindings.buffers.empty()) {
                return;
            }
            bindings.max_index = bindings.min_index + static_cast<u32>(bindings.buffers.size());
            runtime.BindVertexBuffers(bindings);
            bindings = HostBindings<Buffer>{};
            last_index = (std::numeric_limits<u32>::max)();
        };
        while (enabled_mask != 0) {
            const u32 index = std::countr_zero(enabled_mask);
            enabled_mask &= (enabled_mask - 1);
            const Binding& binding = VertexBufferSlot(index);
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeGeometryBuffer(index, binding.buffer_id, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                flush_bindings();
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;
            const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
            const u32 offset = buffer.Offset(binding.device_addr);
            V2NoteUsage(buffer, offset, binding.size);
            if (!bindings.buffers.empty() && index != last_index + 1) {
                flush_bindings();
            }
            if (bindings.buffers.empty()) {
                bindings.min_index = index;
            }
            bindings.buffers.push_back(&buffer);
            bindings.offsets.push_back(offset);
            bindings.sizes.push_back(binding.size);
            bindings.strides.push_back(stride);
            last_index = index;
        }
        flush_bindings();
    } else {
        HostBindings<typename P::Buffer> host_bindings;
        bool any_valid{false};
        auto& flags = maxwell3d->dirty.flags;
        for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
            const Binding& binding = channel_state->vertex_buffers[index];
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            SynchronizeBuffer(buffer, binding.device_addr, binding.size);
            if (!flags[Dirty::VertexBuffer0 + index]) {
                continue;
            }
            flags[Dirty::VertexBuffer0 + index] = false;

            host_bindings.min_index = (std::min)(host_bindings.min_index, index);
            host_bindings.max_index = (std::max)(host_bindings.max_index, index);
            any_valid = true;
        }

        if (any_valid) {
            host_bindings.max_index++;
            for (u32 index = host_bindings.min_index; index < host_bindings.max_index; index++) {
                flags[Dirty::VertexBuffer0 + index] = false;

                const Binding& binding = channel_state->vertex_buffers[index];
                Buffer& buffer = slot_buffers[binding.buffer_id];

                const u32 stride = maxwell3d->regs.vertex_streams[index].stride;
                const u32 offset = buffer.Offset(binding.device_addr);
                V2NoteUsage(buffer, offset, binding.size);

                host_bindings.buffers.push_back(&buffer);
                host_bindings.offsets.push_back(offset);
                host_bindings.sizes.push_back(binding.size);
                host_bindings.strides.push_back(stride);
            }
            runtime.BindVertexBuffers(host_bindings);
        }
    }
}

template <class P>
void BufferCache<P>::BindHostDrawIndirectBuffers() {
    const auto bind_buffer = [this](const Binding& binding) {
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        SynchronizeBuffer(buffer, binding.device_addr, binding.size);
    };
    if (current_draw_indirect->include_count) {
        bind_buffer(channel_state->count_buffer_binding);
    }
    bind_buffer(channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffers(size_t stage) {
    u32 dirty = ~0U;
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        dirty = std::exchange(channel_state->dirty_uniform_buffers[stage], 0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        const bool needs_bind = ((dirty >> index) & 1) != 0;
        BindHostGraphicsUniformBuffer(stage, index, binding_index, needs_bind);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            ++binding_index;
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind) {
    ++channel_state->uniform_cache_shots[0];
    // Bank 24 (root table 0, NVN2's push constants) is read from bank 17, which such games do not
    // bind: stream the table's current words as that buffer.
    if (index == 17 && maxwell3d->root_tables.loaded && channel_state->uniform_buffers[stage][index].size == 0) {
        const auto& words = maxwell3d->root_tables.tables[0];
        const u32 size = (*channel_state->uniform_buffer_sizes)[stage][index];
        const u32 copied = (std::min)(size, static_cast<u32>(sizeof(words)));
        if constexpr (IS_OPENGL) {
            std::array<u8, sizeof(words)> data{};
            std::memcpy(data.data(), words.data(), copied);
            runtime.BindFastUniformBuffer(stage, binding_index, copied);
            runtime.PushFastUniformBuffer(stage, binding_index, std::span<const u8>(data.data(), copied));
        } else {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            std::memset(span.data(), 0, span.size());
            std::memcpy(span.data(), words.data(), copied);
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        return;
    }
    const Binding& binding = channel_state->uniform_buffers[stage][index];
    const DAddr device_addr = binding.device_addr;
    const u32 size = (std::min)(binding.size, (*channel_state->uniform_buffer_sizes)[stage][index]);
    Buffer& buffer = slot_buffers[binding.buffer_id];
    TouchBuffer(buffer, binding.buffer_id);
    const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
    const u32 offset = has_host_buffer ? buffer.Offset(device_addr) : 0;
    const bool needs_alignment_stream = [&]() {
        if constexpr (IS_OPENGL) {
            return false;
        } else {
            if (!has_host_buffer) {
                return false;
            }
            const u32 alignment = runtime.GetUniformBufferAlignment();
            return alignment > 1 && (offset % alignment) != 0;
        }
    }();
    // Native programs keep tables next to their code in a 64 KiB bank read with indirect offsets
    // (Splatoon Raiders: the dialogue box corners in c1). Writes to that memory can go unnoticed by
    // the memory tracker, leaving stale host contents, so upload such a bank again once per frame.
    if (has_host_buffer && (*channel_state->uniform_buffer_sizes)[stage][index] == 0x10000 &&
        size != 0 && gpu_memory != nullptr && gpu_memory->UsesNativeTiling() &&
        (Nx2Perf::Verify() || !native_indirect_uniforms_this_frame.contains(device_addr)) &&
        !(memory_tracker.IsRegionGpuModified(device_addr, size) ||
          (v2_defer && V2PendingGpuWrite(device_addr, size))) &&
        native_indirect_uniforms_this_frame.insert(device_addr).second) {
        // The bank size is only an upper limit: stop at the end of the mapped memory.
        u64 mapped_size = 0;
        while (mapped_size < size &&
               device_memory.template GetPointer<u8>(device_addr + mapped_size) != nullptr) {
            mapped_size += Core::DEVICE_PAGESIZE - ((device_addr + mapped_size) & Core::DEVICE_PAGEMASK);
        }
        const u64 upload_size = (std::min<u64>)(mapped_size, size);
        if (upload_size != 0) {
            std::array<BufferCopy, 1> copies{BufferCopy{
                .src_offset = 0,
                .dst_offset = buffer.Offset(device_addr),
                .size = upload_size,
            }};
            if (!v2_defer || !V2DeferBind(V2::Commit::Kind::BufferUpload, buffer, device_addr,
                                          upload_size, V2::Commit::ArgDirect)) {
                UploadMemory(buffer, upload_size, upload_size, copies);
            }
        }
    }
    const bool use_fast_buffer = needs_alignment_stream
        || (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size
            && !(memory_tracker.IsRegionGpuModified(device_addr, size) ||
          (v2_defer && V2PendingGpuWrite(device_addr, size))));
    if (use_fast_buffer) {
        if constexpr (IS_OPENGL) {
            if (runtime.HasFastBufferSubData()) {
                // Fast path for Nvidia
                const bool should_fast_bind =
                    !HasFastUniformBufferBound(stage, binding_index) ||
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
                if (should_fast_bind) {
                    // We only have to bind when the currently bound buffer is not the fast version
                    channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
                    channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
                    runtime.BindFastUniformBuffer(stage, binding_index, size);
                }
                const auto span = ImmediateBufferWithData(device_addr, size);
                runtime.PushFastUniformBuffer(stage, binding_index, span);
                return;
            }
        }
        channel_state->fast_bound_uniform_buffers[stage] |= 1u << binding_index;
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
        // Stream buffer path to avoid stalling on non-Nvidia drivers or Vulkan
        bool streamed = false;
        if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
            // EDEN_NX2_V2_CBUFVER: a version of this range copied earlier in this command buffer
            // (by any stage or slot) that still holds what guest memory holds is referenced
            // again, without a stream allocation or copy (v2/cbuf_version.h).
            if (VideoCommon::V2::On(VideoCommon::V2::Switch::CbufVer)) [[unlikely]] {
                if (BindUniformVersion(stage, index, binding_index, device_addr, size)) {
                    return;
                }
                streamed = true;
            }
            // The same bytes as this slot's previous bind, in the same command buffer: bind that
            // stream allocation again instead of allocating and copying (uniform_stream_reuse.h).
            // EDEN_NX2_UBOREUSE=0 turns it off.
            if (!streamed && VideoCommon::UniformStreamReuseEnabled()) {
                streamed = VideoCommon::BindSmallUniformReusing(
                    runtime, device_memory, stage, index, binding_index, device_addr, size);
            }
        }
        if (!streamed) {
            const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
            device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        }
        return;
    }
    // Classic cached path
    if (SynchronizeBuffer(buffer, device_addr, size)) {
        ++channel_state->uniform_cache_hits[0];
    }
    // Skip binding if it's not needed and if the bound buffer is not the fast version
    // This exists to avoid instances where the fast buffer is bound and a GPU write happens
    needs_bind |= HasFastUniformBufferBound(stage, binding_index);
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        needs_bind |= channel_state->uniform_buffer_binding_sizes[stage][binding_index] != size;
    }
    if (!needs_bind) {
        return;
    }
    if constexpr (IS_OPENGL) {
        // Mark the index as dirty if offset doesn't match
        const bool is_copy_bind = offset != 0 && !runtime.SupportsNonZeroUniformOffset();
        channel_state->dirty_uniform_buffers[stage] |= (is_copy_bind ? 1U : 0U) << index;
    }
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->uniform_buffer_binding_sizes[stage][binding_index] = size;
    }
    V2NoteUsage(buffer, offset, size);
    if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
        runtime.BindUniformBuffer(stage, binding_index, buffer, offset, size);
    } else {
        runtime.BindUniformBuffer(buffer, offset, size);
    }
    channel_state->fast_bound_uniform_buffers[stage] &= ~(1u << binding_index);
}

template <class P>
const u8* BufferCache<P>::V2GuestBytes(DAddr device_addr, u32 size) {
    if (const u8* const span = device_memory.GetSpan(device_addr, size)) {
        return span;
    }
    version_scratch.resize(size);
    device_memory.ReadBlockUnsafe(device_addr, version_scratch.data(), size);
    return version_scratch.data();
}

template <class P>
bool BufferCache<P>::BindUniformVersion(size_t stage, u32 index, u32 binding_index,
                                        DAddr device_addr, u32 size) {
    if constexpr (!IS_OPENGL && VideoCommon::StreamUniformReusable<Runtime>) {
        namespace V2 = VideoCommon::V2;
        using Ref = typename Runtime::StreamUniformRef;
        auto& table = V2::CbufVer::Table<Ref>();
        const u64 tick = runtime.CurrentTick();
        const bool verify = V2::Verify(V2::Switch::CbufVer);
        const bool stats = V2::P3Stats::Enabled();
        if (stats) {
            ++V2::P3Stats::Get().cbuf_binds;
        }
        // Taken before the bytes are read: a host write after it ends the version.
        const u64 page_sum = table.Begin(device_addr, size);
        // What guest memory holds now: a version is referenced only while its bytes equal it, so
        // a guest CPU write ends it at the next bind with no page in the write tracking. Read and
        // compared only when the version's write checks pass (Perf fix D).
        const Ref* const version =
            table.MayHold(device_addr, size, tick, page_sum)
                ? table.Matching(device_addr, size, V2GuestBytes(device_addr, size))
                : nullptr;
        if (version != nullptr) {
            bool same = true;
            if (verify) {
                // The bytes the shader would read now (read again) against the version's.
                auto& data = V2::P3Stats::Get();
                ++data.cbuf_verify;
                const std::vector<u8>& kept = table.Bytes(device_addr, size);
                verify_scratch.resize(size);
                device_memory.ReadBlockUnsafe(device_addr, verify_scratch.data(), size);
                same = kept.size() == size &&
                       std::memcmp(kept.data(), verify_scratch.data(), size) == 0;
                if (!same) {
                    ++data.cbuf_mismatches;
                    static std::atomic<u32> logged{};
                    if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                        LOG_WARNING(HW_GPU,
                                    "V2 cbuf version verify: stage {} cbuf {} at {:#x}+{:#x} "
                                    "changed behind its version",
                                    stage, index, device_addr, size);
                    }
                }
            }
            if (same) {
                runtime.RebindStreamUniform(*version, size);
                if (stats) {
                    auto& data = V2::P3Stats::Get();
                    ++data.cbuf_hits;
                    data.cbuf_bytes_saved += size;
                }
                return true;
            }
        }
        // A new version: one read of guest memory into the version's own bytes feeds the stream
        // copy too, so the two can't differ when the CPU writes meanwhile (two copies, no
        // scratch).
        u8* const bytes = table.Prepare(device_addr, size, tick, page_sum);
        if (const u8* const guest = device_memory.GetSpan(device_addr, size)) {
            std::memcpy(bytes, guest, size);
        } else {
            device_memory.ReadBlockUnsafe(device_addr, bytes, size);
        }
        const std::span<u8> span = runtime.BindMappedUniformBuffer(stage, binding_index, size);
        std::memcpy(span.data(), bytes, size);
        table.Commit(device_addr, size, runtime.LastStreamUniformRef());
        if (stats) {
            ++V2::P3Stats::Get().cbuf_copies;
        }
    }
    return false;
}

template <class P>
void BufferCache<P>::RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        VideoCommon::RefreshMultiRangeSegments(
            std::span{pool}, [this](BufferId id) { return slot_buffers.contains(id); },
            [this](DAddr address, u32 size) {
                return FindBuffer(address, size, runtime.PrefersSparseSources());
            });
    }
}

template <class P>
void BufferCache<P>::ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                              std::vector<MultiRangeSegment>& pool) {
    // Segment lookups use their buffers at once: no proxies (P6: a serial segment).
    const bool proxy_finds = std::exchange(v2_proxy_finds, false);
    SCOPE_EXIT {
        v2_proxy_finds = proxy_finds;
    };
    binding.segment_first = 0;
    binding.segment_count = 0;
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.gpu_addr == 0 || binding.size == 0) {
            return;
        }
        if (is_written && !runtime.PrefersSparseSources()) {
            return;
        }
        const VirtualSegments* found =
            virtual_ranges.Query(*gpu_memory, binding.gpu_addr, binding.size);
        if (!found || found->size() < 2) {
            return;
        }
        const VirtualSegments segments = *found;
        const u32 first = static_cast<u32>(pool.size());
        const bool prefer_sparse = runtime.PrefersSparseSources();
        for (const VirtualSegment& segment : segments) {
            const BufferId buffer_id =
                FindBuffer(segment.device_addr, segment.size, prefer_sparse);
            if (!buffer_id) {
                pool.resize(first);
                return;
            }
            pool.push_back(MultiRangeSegment{
                .buffer_id = buffer_id,
                .device_addr = segment.device_addr,
                .size = segment.size,
            });
        }
        binding.segment_first = first;
        binding.segment_count = static_cast<u32>(segments.size());
    }
}

template <class P>
bool BufferCache<P>::BindMultiRangeStorage(const Binding& binding, bool is_written,
                                           std::span<const MultiRangeSegment> pool,
                                           u64 write_mark_limit) {
    if constexpr (requires { runtime.BindMultiRangeStorageBuffer(u64{}, bool{}); }) {
        if (binding.segment_count < 2) {
            return false;
        }
        if (binding.segment_first > pool.size() ||
            binding.segment_count > pool.size() - binding.segment_first) {
            return false;
        }
        const auto segments = pool.subspan(binding.segment_first, binding.segment_count);
        if (!MultiRangeSegmentsLive(segments,
                                    [this](BufferId id) { return slot_buffers.contains(id); })) {
            return false;
        }
        const u64 key = (static_cast<u64>(gpu_memory->GetID()) << 48) ^ binding.gpu_addr;
        runtime.ResetMultiRange();
        for (u32 index = 0; index < binding.segment_count; ++index) {
            const MultiRangeSegment& segment = pool[binding.segment_first + index];
            Buffer& buffer = slot_buffers[segment.buffer_id];
            TouchBuffer(buffer, segment.buffer_id);
            SynchronizeBuffer(buffer, segment.device_addr, segment.size);
            const u32 offset = buffer.Offset(segment.device_addr);
            V2NoteUsage(buffer, offset, segment.size);
            if (is_written && write_mark_limit != 0) {
                // write_mark_limit: the native write mark (BindComputeStorageBuffer), spread over
                // the segments in order.
                const u32 mark = static_cast<u32>(std::min<u64>(segment.size, write_mark_limit));
                MarkWrittenBuffer(segment.buffer_id, segment.device_addr, mark);
                if (write_mark_limit != ~u64{0}) {
                    write_mark_limit -= mark;
                }
            }
            runtime.PushMultiRangeSource(buffer, offset, segment.size);
        }
        return BindFreshMultiRangeStorage(runtime, key, is_written);
    } else {
        return false;
    }
}

template <class P>
void BufferCache<P>::BindHostGraphicsStorageBuffers(size_t stage) {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        const Binding& binding = channel_state->storage_buffers[stage][index];
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        if (BindMultiRangeStorage(binding, is_written, graphics_segments)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        V2NoteUsage(buffer, offset, size);

        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindStorageBuffer(stage, binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostGraphicsTextureBuffers(size_t stage) {
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->texture_buffers[stage][index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written = ((channel_state->written_texture_buffers[stage] >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        V2NoteUsage(buffer, offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_texture_buffers[stage] >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    HostBindings<typename P::Buffer> host_bindings;
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        const Binding& binding = channel_state->transform_feedback_buffers[index];
        const auto& control = maxwell3d->regs.transform_feedback.controls[index];
        const bool has_layout = control.varying_count != 0 || control.stride != 0;

        Buffer* host_buffer = &slot_buffers[NULL_BUFFER_ID];
        u32 offset = 0;
        u32 size = 0;

        if (has_layout && binding.buffer_id != NULL_BUFFER_ID && binding.size != 0) {
            Buffer& buffer = slot_buffers[binding.buffer_id];
            TouchBuffer(buffer, binding.buffer_id);
            size = binding.size;
            SynchronizeBuffer(buffer, binding.device_addr, size);
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
            offset = buffer.Offset(binding.device_addr);
            V2NoteUsage(buffer, offset, size);
            host_buffer = &buffer;
        }

        host_bindings.buffers.push_back(host_buffer);
        host_bindings.offsets.push_back(offset);
        host_bindings.sizes.push_back(size);
    }
    runtime.BindTransformFeedbackBuffers(host_bindings);
}

template <class P>
void BufferCache<P>::BindHostComputeUniformBuffers() {
    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        // Mark all uniform buffers as dirty
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->fast_bound_uniform_buffers.fill(0);
    }
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        const Binding& binding = channel_state->compute_uniform_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size =
            (std::min)(binding.size, (*channel_state->compute_uniform_buffer_sizes)[index]);
        const bool has_host_buffer = binding.buffer_id != NULL_BUFFER_ID;
        const u32 offset = has_host_buffer ? buffer.Offset(binding.device_addr) : 0;
        const bool needs_alignment_stream = [&]() {
            if constexpr (IS_OPENGL) {
                return false;
            } else {
                if (!has_host_buffer) {
                    return false;
                }
                const u32 alignment = runtime.GetUniformBufferAlignment();
                return alignment > 1 && (offset % alignment) != 0;
            }
        }();
        // Small buffers the GPU has not written are streamed, as graphics ones are: synchronizing
        // them made every dispatch re-protect pages an inline upload had just unprotected (two
        // VirtualProtect calls per dispatch, ~4,000 dispatches a second in title 1).
        const bool use_fast_buffer =
            needs_alignment_stream ||
            (has_host_buffer && size <= channel_state->uniform_buffer_skip_cache_size &&
             !(memory_tracker.IsRegionGpuModified(binding.device_addr, size) ||
               (v2_defer && V2PendingGpuWrite(binding.device_addr, size))));
        if constexpr (!IS_OPENGL) {
            if (use_fast_buffer) {
                const std::span<u8> span =
                    runtime.BindMappedUniformBuffer(0, binding_index, size);
                device_memory.ReadBlockUnsafe(binding.device_addr, span.data(), size);
                return;
            }
        }

        SynchronizeBuffer(buffer, binding.device_addr, size);

        V2NoteUsage(buffer, offset, size);
        if constexpr (NEEDS_BIND_UNIFORM_INDEX) {
            runtime.BindComputeUniformBuffer(binding_index, buffer, offset, size);
            ++binding_index;
        } else {
            runtime.BindUniformBuffer(buffer, offset, size);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeStorageBuffers() {
    u32 binding_index = 0;
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        const Binding& binding = channel_state->compute_storage_buffers[index];
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        // A native program's binding is min(layout, 8 MiB) because its pointer has no size, and
        // marking all of it GPU-written defers the CPU's later writes to those pages. Mark only
        // what BindComputeStorageBuffer worked out the dispatch can write; the binding keeps its
        // size so the shader can still read all of it.
        const bool native =
            ((channel_state->native_compute_storage_buffers >> index) & 1) != 0;
        const u64 write_mark_limit =
            native ? u64{channel_state->compute_storage_write_marks[index]} : ~u64{0};
        if (BindMultiRangeStorage(binding, is_written, compute_segments, write_mark_limit)) {
            return;
        }
        Buffer& buffer = slot_buffers[binding.buffer_id];
        TouchBuffer(buffer, binding.buffer_id);
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const u32 offset = buffer.Offset(binding.device_addr);
        V2NoteUsage(buffer, offset, size);

        const u32 mark =
            is_written ? static_cast<u32>(std::min<u64>(size, write_mark_limit)) : 0;
        if (mark != 0) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, mark, true);
        }

        if constexpr (NEEDS_BIND_STORAGE_INDEX) {
            runtime.BindComputeStorageBuffer(binding_index, buffer, offset, size, is_written);
            ++binding_index;
        } else {
            runtime.BindStorageBuffer(buffer, offset, size, is_written);
        }
    });
}

template <class P>
void BufferCache<P>::BindHostComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        const TextureBufferBinding& binding = channel_state->compute_texture_buffers[index];
        Buffer& buffer = slot_buffers[binding.buffer_id];
        const u32 size = binding.size;
        SynchronizeBuffer(buffer, binding.device_addr, size);

        const bool is_written =
            ((channel_state->written_compute_texture_buffers >> index) & 1) != 0;
        if (is_written) {
            MarkWrittenBuffer(binding.buffer_id, binding.device_addr, size);
        }

        const u32 offset = buffer.Offset(binding.device_addr);
        const PixelFormat format = binding.format;
        V2NoteUsage(buffer, offset, size);
        if constexpr (SEPARATE_IMAGE_BUFFERS_BINDINGS) {
            if (((channel_state->image_compute_texture_buffers >> index) & 1) != 0) {
                runtime.BindImageBuffer(buffer, offset, size, format);
            } else {
                runtime.BindTextureBuffer(buffer, offset, size, format);
            }
        } else {
            runtime.BindTextureBuffer(buffer, offset, size, format);
        }
    });
}

template <class P>
void BufferCache<P>::DoUpdateGraphicsBuffers(bool is_indexed) {
    graphics_segments.clear();
    const bool defers = V2Defers();
    if (defers) [[unlikely]] {
        CommitBinds(); // a previous scope a failed configure left open
    }
    BufferOperations([&]() {
        if (defers) [[unlikely]] {
            v2_creates.Reset(v2_segment++);
            v2_proxy_finds = true;
        }
        if (is_indexed) {
            UpdateIndexBuffer();
        }
        UpdateVertexBuffers();
        UpdateTransformFeedbackBuffers();
        for (size_t stage = 0; stage < NUM_STAGES; ++stage) {
            UpdateUniformBuffers(stage);
            UpdateStorageBuffers(stage);
            UpdateTextureBuffers(stage);
        }
        if (current_draw_indirect) {
            UpdateDrawIndirect();
        }
        if (defers) [[unlikely]] {
            v2_proxy_finds = false;
            V2ApplyCreates();
        }
        RefreshMultiRangeSegments(graphics_segments);
    });
    if (defers) [[unlikely]] {
        V2BeginBindScope();
    }
}

template <class P>
void BufferCache<P>::DoUpdateComputeBuffers() {
    compute_segments.clear();
    const bool defers = V2Defers();
    if (defers) [[unlikely]] {
        CommitBinds();
    }
    BufferOperations([&]() {
        if (defers) [[unlikely]] {
            v2_creates.Reset(v2_segment++);
            v2_proxy_finds = true;
        }
        UpdateComputeUniformBuffers();
        UpdateComputeStorageBuffers();
        UpdateComputeTextureBuffers();
        if (defers) [[unlikely]] {
            v2_proxy_finds = false;
            V2ApplyCreates();
        }
        RefreshMultiRangeSegments(compute_segments);
    });
    if (defers) [[unlikely]] {
        V2BeginBindScope();
    }
}

template <class P>
void BufferCache<P>::V2BeginBindScope() {
    v2_binds.Reset(v2_segment++);
    v2_written_marks.clear();
    v2_scope_uploads.clear();
    v2_defer = true;
}

template <class P>
void BufferCache<P>::V2PatchBindings(u32 proxy, BufferId id) {
    const auto patch = [&](auto& binding) {
        if (binding.buffer_id.index == proxy) {
            binding.buffer_id = id;
        }
    };
    patch(channel_state->index_buffer);
    for (auto& binding : channel_state->vertex_buffers) {
        patch(binding);
    }
    for (auto& binding : v_buffer) {
        patch(binding);
    }
    for (auto& stage : channel_state->uniform_buffers) {
        for (auto& binding : stage) {
            patch(binding);
        }
    }
    for (auto& stage : channel_state->storage_buffers) {
        for (auto& binding : stage) {
            patch(binding);
        }
    }
    for (auto& stage : channel_state->texture_buffers) {
        for (auto& binding : stage) {
            patch(binding);
        }
    }
    for (auto& binding : channel_state->transform_feedback_buffers) {
        patch(binding);
    }
    patch(channel_state->count_buffer_binding);
    patch(channel_state->indirect_buffer_binding);
    for (auto& binding : channel_state->compute_uniform_buffers) {
        patch(binding);
    }
    for (auto& binding : channel_state->compute_storage_buffers) {
        patch(binding);
    }
    for (auto& binding : channel_state->compute_texture_buffers) {
        patch(binding);
    }
}

template <class P>
void BufferCache<P>::V2ApplyCreates() {
    if (v2_creates.Empty()) {
        return;
    }
    const u64 start = commit->stats ? VideoCommon::Nx2Profile::Now() : 0;
    // In request order, each patched before the next applies: a later creation that merges an
    // earlier one's buffer then finds it bound and clears it (DeleteBuffer), and the update
    // pass runs again, as the old in-lookup creations did.
    commit->stage.Apply(v2_creates, [&](const V2::Commit::Request& request) {
        const BufferId id = FindBuffer(request.address, static_cast<u32>(request.size),
                                       request.arg != 0);
        V2PatchBindings(request.proxy, id);
        return id.index;
    });
    if (commit->stats) {
        auto& stats = V2::Commit::Stats::Get();
        const auto counters = commit->stage.TakeCounters();
        for (size_t kind = 0; kind < V2::Commit::NumKinds; ++kind) {
            V2::Commit::Stats::Count(stats.requests[kind], counters.requests[kind]);
        }
        V2::Commit::Stats::Count(stats.proxies_patched, counters.proxies_patched);
        V2::Commit::Stats::Count(stats.commits);
        V2::Commit::Stats::Count(stats.commit_ns, VideoCommon::Nx2Profile::Now() - start);
    }
}

template <class P>
BufferId BufferCache<P>::V2CurrentBuffer(BufferId id, DAddr device_addr, u64 size) {
    if (id && slot_buffers.contains(id) &&
        slot_buffers[id].IsInBounds(device_addr, (std::max<u64>)(size, 1))) {
        return id;
    }
    // Merged or recreated since the request (a replay's fallback lookup): the buffer that serves
    // the address now.
    const u64 page = device_addr >> CACHING_PAGEBITS;
    return page < page_table.size() ? page_table[page] : NULL_BUFFER_ID;
}

template <class P>
bool BufferCache<P>::V2DeferBind(V2::Commit::Kind kind, Buffer& buffer, DAddr device_addr,
                                 u64 size, u32 arg) {
    const u64 page = buffer.CpuAddr() >> CACHING_PAGEBITS;
    if (page >= page_table.size()) {
        return false;
    }
    const BufferId id = page_table[page];
    if (!id || &slot_buffers[id] != &buffer) {
        return false; // not a registered buffer (the null buffer): apply now
    }
    V2::Commit::Request request{};
    request.kind = kind;
    request.resource = V2::Commit::Resource::Buffer;
    request.slot = id.index;
    request.address = device_addr;
    request.size = size;
    request.arg = arg;
    v2_binds.Push(request);
    if (kind == V2::Commit::Kind::BufferUpload || kind == V2::Commit::Kind::BufferGeometrySync) {
        v2_scope_uploads.push_back(id.index);
    }
    return true;
}

template <class P>
bool BufferCache<P>::V2PendingGpuWrite(DAddr device_addr, u64 size) const {
    for (const auto& [address, mark_size] : v2_written_marks) {
        if (address < device_addr + size && device_addr < address + mark_size) {
            return true;
        }
    }
    return false;
}

template <class P>
void BufferCache<P>::V2NoteUsage(Buffer& buffer, u32 offset, u64 size) {
    // F2-19: a usage mark orders only against an upload of the same buffer queued in this scope;
    // without one it applies in place (as at the commit: no request in between reads it).
    // A creation still queued may merge the buffer: then the mark follows it, as before.
    if (v2_defer && (!v2_scope_uploads.empty() || !v2_creates.Empty())) [[unlikely]] {
        const u64 page = buffer.CpuAddr() >> CACHING_PAGEBITS;
        const BufferId id = page < page_table.size() ? page_table[page] : NULL_BUFFER_ID;
        const bool ordered =
            !v2_creates.Empty() ||
            std::ranges::find(v2_scope_uploads, id.index) != v2_scope_uploads.end();
        if (ordered &&
            V2DeferBind(V2::Commit::Kind::BufferUsage, buffer, buffer.CpuAddr() + offset, size,
                        0)) {
            return;
        }
    }
    buffer.MarkUsage(offset, size);
}

template <class P>
void BufferCache<P>::CommitBinds() {
    if (!v2_defer) {
        return;
    }
    v2_defer = false;
    if (v2_binds.Empty()) {
        return;
    }
    namespace C = V2::Commit;
    const u64 start = commit->stats ? VideoCommon::Nx2Profile::Now() : 0;
    commit->stage.Apply(v2_binds, [&](const C::Request& request) {
        const BufferId id =
            V2CurrentBuffer(BufferId{request.slot}, request.address, request.size);
        if (!id) {
            return C::InvalidSlot;
        }
        Buffer& buffer = slot_buffers[id];
        switch (request.kind) {
        case C::Kind::BufferUpload:
            if ((request.arg & C::ArgDirect) != 0) {
                // The native bank's direct upload (no tracker check), as recorded.
                std::array<BufferCopy, 1> copies{BufferCopy{
                    .src_offset = 0,
                    .dst_offset = buffer.Offset(request.address),
                    .size = request.size,
                }};
                UploadMemory(buffer, request.size, request.size, copies);
            } else {
                SynchronizeBuffer(buffer, request.address, static_cast<u32>(request.size));
            }
            break;
        case C::Kind::BufferGeometrySync:
            SynchronizeGeometryBuffer(request.arg, id, request.address,
                                      static_cast<u32>(request.size));
            break;
        case C::Kind::BufferMarkWritten:
            MarkWrittenBuffer(id, request.address, static_cast<u32>(request.size),
                              request.arg != 0);
            break;
        case C::Kind::BufferTouch:
            TouchBuffer(buffer, id);
            break;
        case C::Kind::BufferUsage:
            buffer.MarkUsage(buffer.Offset(request.address), request.size);
            break;
        default:
            break;
        }
        return id.index;
    });
    if (commit->stats) {
        auto& stats = C::Stats::Get();
        const auto counters = commit->stage.TakeCounters();
        for (size_t kind = 0; kind < C::NumKinds; ++kind) {
            C::Stats::Count(stats.requests[kind], counters.requests[kind]);
        }
        C::Stats::Count(stats.commits);
        C::Stats::Count(stats.commit_ns, VideoCommon::Nx2Profile::Now() - start);
    }
}

template <class P>
void BufferCache<P>::UpdateIndexBuffer() {
    // We have to check for the dirty flags and index count
    // The index count is currently changed without updating the dirty flags
    const auto& draw_state = maxwell3d->draw_manager.draw_state;
    const auto& index_buffer_ref = draw_state.index_buffer;
    auto& flags = maxwell3d->dirty.flags;
    if (!flags[Dirty::IndexBuffer]) {
        return;
    }
    flags[Dirty::IndexBuffer] = false;
    if (!draw_state.inline_index_draw_indexes.empty()) [[unlikely]] {
        auto inline_index_size = static_cast<u32>(draw_state.inline_index_draw_indexes.size());
        u32 buffer_size = Common::AlignUp(inline_index_size, CACHING_PAGESIZE);
        if (inline_buffer_id == NULL_BUFFER_ID) [[unlikely]] {
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        if (slot_buffers[inline_buffer_id].SizeBytes() < buffer_size) [[unlikely]] {
            ++buffer_deletions;
            slot_buffers.erase(inline_buffer_id);
            inline_buffer_id = CreateBuffer(0, buffer_size, false);
        }
        channel_state->index_buffer = Binding{
            .device_addr = 0,
            .size = inline_index_size,
            .buffer_id = inline_buffer_id,
        };
        return;
    }

    const GeometryRange range =
        IndexRange(*gpu_memory, index_buffer_ref.StartAddress(), index_buffer_ref.EndAddress(),
                   index_buffer_ref.first, index_buffer_ref.count,
                   u32(index_buffer_ref.FormatSizeInBytes()));
    if (!range.valid) {
        channel_state->index_buffer = NULL_BINDING;
        return;
    }
    channel_state->index_buffer = Binding{
        .device_addr = range.device_addr,
        .size = range.size,
        .buffer_id = FindBuffer(range.device_addr, range.size, false),
    };
}

template <class P>
void BufferCache<P>::UpdateVertexBuffers() {
    auto& flags = maxwell3d->dirty.flags;
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffers]) {
        return;
    }
    flags[Dirty::VertexBuffers] = false;

    for (u32 index = 0; index < NUM_VERTEX_BUFFERS; ++index) {
        UpdateVertexBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateVertexBuffer(u32 index) {
    if (!maxwell3d->dirty.flags[Dirty::VertexBuffer0 + index]) {
        return;
    }
    const auto& array = maxwell3d->regs.vertex_streams[index];
    const auto& limit = maxwell3d->regs.vertex_stream_limits[index];
    const bool native = maxwell3d->GetNativeMacroEngine().native_turing;
    const u64 native_size = native ? maxwell3d->regs.NativeVertexStreamSize(index) : 0;
    // TODO: Analyze stride and number of vertices
    const GeometryRange range = VertexRange(*gpu_memory, array.Address(), array.enable != 0,
                                            limit.Address(), native, native_size);
    if (!range.valid) {
        channel_state->vertex_buffers[index] = NULL_BINDING;
        UpdateVertexBufferSlot(index, NULL_BINDING);
        return;
    }
    const BufferId buffer_id = FindBuffer(range.device_addr, range.size, false);
    const Binding binding{
        .device_addr = range.device_addr,
        .size = range.size,
        .buffer_id = buffer_id,
    };
    channel_state->vertex_buffers[index] = binding;
    UpdateVertexBufferSlot(index, binding);
}

template <class P>
void BufferCache<P>::UpdateDrawIndirect() {
    const auto update = [this](GPUVAddr gpu_addr, size_t size, Binding& binding) {
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        if (!device_addr) {
            binding = NULL_BINDING;
            return;
        }
        binding = Binding{
            .device_addr = *device_addr,
            .size = static_cast<u32>(size),
            .buffer_id = FindBuffer(*device_addr, static_cast<u32>(size), false),
        };
    };
    if (current_draw_indirect->include_count) {
        update(current_draw_indirect->count_start_address, sizeof(u32),
               channel_state->count_buffer_binding);
    }
    update(current_draw_indirect->indirect_start_address, current_draw_indirect->buffer_size,
           channel_state->indirect_buffer_binding);
}

template <class P>
void BufferCache<P>::UpdateUniformBuffers(size_t stage) {
    if (channel_state->enabled_uniform_buffer_masks[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
        Binding& binding = channel_state->uniform_buffers[stage][index];
        if (binding.buffer_id) {
            // Already updated
            return;
        }
        // Mark as dirty
        if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
            channel_state->dirty_uniform_buffers[stage] |= 1U << index;
        }
        // Resolve buffer
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateStorageBuffers(size_t stage) {
    if (channel_state->enabled_storage_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->storage_buffers[stage][index];
        const BufferId buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        binding.buffer_id = buffer_id;
        const bool is_written = ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, graphics_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateTextureBuffers(size_t stage) {
    if (channel_state->enabled_texture_buffers[stage] == 0) [[likely]] {
        return;
    }
    ForEachEnabledBit(channel_state->enabled_texture_buffers[stage], [&](u32 index) {
        Binding& binding = channel_state->texture_buffers[stage][index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffers() {
    if (maxwell3d->regs.transform_feedback_enabled == 0) {
        return;
    }
    for (u32 index = 0; index < NUM_TRANSFORM_FEEDBACK_BUFFERS; ++index) {
        UpdateTransformFeedbackBuffer(index);
    }
}

template <class P>
void BufferCache<P>::UpdateTransformFeedbackBuffer(u32 index) {
    const auto& binding = maxwell3d->regs.transform_feedback.buffers[index];
    const GPUVAddr gpu_addr = binding.Address() + binding.start_offset;
    const u32 size = binding.size;
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    if (binding.enable == 0 || size == 0 || !device_addr) {
        channel_state->transform_feedback_buffers[index] = NULL_BINDING;
        return;
    }
    const BufferId buffer_id = FindBuffer(*device_addr, size, false);
    channel_state->transform_feedback_buffers[index] = Binding{
        .device_addr = *device_addr,
        .size = size,
        .buffer_id = buffer_id,
    };
}

template <class P>
void BufferCache<P>::UpdateComputeUniformBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
        Binding& binding = channel_state->compute_uniform_buffers[index];
        binding = NULL_BINDING;
        const auto& launch_desc = kepler_compute->launch_description;
        if (((launch_desc.const_buffer_enable_mask >> index) & 1) != 0) {
            const auto& cbuf = launch_desc.const_buffer_config[index];
            const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(cbuf.Address());
            if (device_addr) {
                binding.device_addr = *device_addr;
                binding.size = cbuf.size;
            }
        }
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeStorageBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
        // Resolve buffer
        Binding& binding = channel_state->compute_storage_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
        const bool is_written =
            ((channel_state->written_compute_storage_buffers >> index) & 1) != 0;
        ResolveMultiRangeStorage(binding, is_written, compute_segments);
    });
}

template <class P>
void BufferCache<P>::UpdateComputeTextureBuffers() {
    ForEachEnabledBit(channel_state->enabled_compute_texture_buffers, [&](u32 index) {
        Binding& binding = channel_state->compute_texture_buffers[index];
        binding.buffer_id = FindBuffer(binding.device_addr, binding.size, false);
    });
}

template <class P>
void BufferCache<P>::MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync) {
    if (v2_defer && buffer_id) [[unlikely]] {
        // Ordered after the binding's upload request, as the old order was (a sync skips the
        // GPU-written ranges); bind-time reads see it through V2PendingGpuWrite.
        V2::Commit::Request request{};
        request.kind = V2::Commit::Kind::BufferMarkWritten;
        request.resource = V2::Commit::Resource::Buffer;
        request.slot = buffer_id.index;
        request.address = device_addr;
        request.size = size;
        request.arg = needs_sync ? 1U : 0U;
        v2_binds.Push(request);
        v2_written_marks.emplace_back(device_addr, size);
        return;
    }
    if constexpr (!IS_OPENGL) {
        if (needs_sync) {
            Buffer& buffer = slot_buffers[buffer_id];
            buffer.setWriteTick(runtime.CurrentTick());
        }
    }
    // The downloads already taken over this range hold older bytes: when their fences pop
    // they must not write them over this write's result in guest memory, nor drop the range
    // from gpu_modified_ranges (the next fence downloads it instead).
    for (auto& ranges : pending_download_ranges) {
        ranges.Subtract(device_addr, size);
    }
    memory_tracker.MarkRegionAsGpuModified(device_addr, size);
    MarkVertexBuffersStale(device_addr, size);
    gpu_modified_ranges.Add(device_addr, size);
    V2::Versions::HookWrite(device_addr, size, V2::Versions::Source::Gpu);
    if (learned_marked != 0) [[unlikely]] {
        NoteLearnedWrite(device_addr, size);
    }
    if (v2_writer_downloads) [[unlikely]] {
        // A newer write: the prepared downloads hold older bytes for this range.
        for (auto& prepared : writer_downloads_open) {
            prepared.ranges.Subtract(device_addr, size);
        }
        for (auto& fence : writer_downloads_fenced) {
            for (auto& prepared : fence) {
                prepared.ranges.Subtract(device_addr, size);
            }
        }
    }
    QueueFenceDownload(device_addr, size);
    if (Tegra::g_nx2_ampere.load(std::memory_order_relaxed)) {
        Tegra::Nx2MarkGpuWritten(device_addr, size);
        Nx2NoteListWrite(buffer_id, device_addr, size);
    } else if (commit != nullptr) [[unlikely]] {
        // P5a: guest CPU write traps decide lock-free whether a write may meet GPU data.
        Tegra::Nx2MarkGpuWritten(device_addr, size);
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last;) {
            // Storage buffers can be large: skip 64 pages at a time where none is a list page.
            const u64 word =
                Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                (page % 64);
            if (word == 0) {
                page = (page / 64 + 1) * 64;
                continue;
            }
            if (word & 1) {
                const DAddr page_addr = page << page_bits;
                {
                    std::scoped_lock lock{nx2_list_mutex};
                    Nx2ListPage& entry = nx2_list_pages[page_addr];
                    ++entry.writes;
                    if (!nx2_collect_list_copies && !entry.draw_written) {
                        entry.draw_written = true;
                        LOG_INFO(HW_GPU, "Command list page {:#x} is written by draws too: its "
                                         "lists wait for the work before the last host sync",
                                 page_addr);
                    }
                }
                if (nx2_collect_list_copies) {
                    nx2_list_copy_requests.emplace_back(buffer_id, page_addr);
                }
                if (Tegra::Nx2CmdPageDiag::g_enabled) {
                    Tegra::Nx2CmdPageDiag::NoteWrite(page_addr, device_addr, size,
                                                     !nx2_collect_list_copies);
                }
            }
            ++page;
        }
    }
}

template <class P>
bool BufferCache<P>::Nx2CopyListPages(u64 tick) {
    if constexpr (IS_OPENGL) {
        return false;
    } else {
        if (nx2_list_copy_requests.empty()) {
            return false;
        }
        std::ranges::sort(nx2_list_copy_requests, {}, &std::pair<BufferId, DAddr>::second);
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        bool recorded = false;
        for (size_t i = 0; i < nx2_list_copy_requests.size(); ++i) {
            const auto [buffer_id, page] = nx2_list_copy_requests[i];
            if (i + 1 < nx2_list_copy_requests.size() &&
                nx2_list_copy_requests[i + 1].second == page) {
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            const DAddr begin = (std::max<DAddr>)(page, buffer.CpuAddr());
            const DAddr end =
                (std::min<DAddr>)(page + page_size, buffer.CpuAddr() + buffer.SizeBytes());
            if (begin >= end) {
                continue;
            }
            Nx2Staging staging = runtime.DownloadStagingBuffer(page_size, true);
            const BufferCopy copy{
                .src_offset = buffer.Offset(begin),
                .dst_offset = staging.offset + (begin - page),
                .size = end - begin,
            };
            buffer.MarkUsage(copy.src_offset, copy.size);
            runtime.CopyBuffer(staging.buffer, buffer, std::span{&copy, 1}, true);
            recorded = true;
            std::scoped_lock lock{nx2_list_mutex};
            Nx2ListPage& entry = nx2_list_pages[page];
            if (entry.copy) {
                nx2_retired_list_copies.emplace_back(entry.copy_tick, std::move(*entry.copy));
            }
            entry.copy = staging;
            entry.copy_writes = entry.writes;
            entry.copy_tick = tick;
            entry.copy_begin = static_cast<u32>(begin - page);
            entry.copy_end = static_cast<u32>(end - page);
        }
        nx2_list_copy_requests.clear();
        std::erase_if(nx2_retired_list_copies, [this](auto& retired) {
            if (!runtime.IsFree(retired.first)) {
                return false;
            }
            runtime.FreeDeferredStagingBuffer(retired.second);
            return true;
        });
        return recorded;
    }
}

template <class P>
auto BufferCache<P>::Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync,
                                     u64* pending_tick) -> Nx2ListRead {
    if constexpr (IS_OPENGL) {
        return Nx2ListRead::Unknown;
    } else {
        constexpr DAddr page_size = DAddr{1} << Tegra::Nx2WrittenPageBits;
        const DAddr page = device_addr & ~(page_size - 1);
        const u64 offset = device_addr - page;
        if (offset + out.size() > page_size) {
            return Nx2ListRead::Unknown;
        }
        const auto current = [&](const Nx2ListPage& entry) {
            return entry.copy && !entry.draw_written && entry.copy_writes == entry.writes &&
                   offset >= entry.copy_begin && offset + out.size() <= entry.copy_end;
        };
        u64 tick{};
        u64 writes{};
        {
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page);
            if (it == nx2_list_pages.end()) {
                return Nx2ListRead::Unknown;
            }
            const Nx2ListPage& entry = it->second;
            if (entry.draw_written) {
                // EDEN_NX2_CMDPAGE_DRAIN=1: drain and flush for every list on such pages.
                static const bool always_drain = std::getenv("EDEN_NX2_CMDPAGE_DRAIN") != nullptr;
                if (always_drain) {
                    return Nx2ListRead::Unknown;
                }
                if (!past_sync) {
                    return Nx2ListRead::NeedsSync;
                }
                return entry.synced_writes == entry.writes ? Nx2ListRead::Synced
                                                           : Nx2ListRead::Unknown;
            }
            // Pages the CPU writes the lists of (GPU-written once, long before) stay synced.
            if (entry.synced_writes == entry.writes) {
                return Nx2ListRead::Synced;
            }
            if (!current(entry)) {
                return Nx2ListRead::Unknown;
            }
            // The lists after the first of a batch find the copy done: read it at once.
            if (runtime.IsFree(entry.copy_tick)) {
                std::memcpy(out.data(), entry.copy->mapped_span.data() + offset, out.size());
                return Nx2ListRead::Copied;
            }
            tick = entry.copy_tick;
            writes = entry.copy_writes;
            if (pending_tick != nullptr) {
                *pending_tick = tick;
                return Nx2ListRead::Pending;
            }
        }
        // Submitted with the work that wrote the page (Nx2CopyListPages's caller flushes).
        runtime.WaitSubmittedTick(tick);
        std::scoped_lock lock{nx2_list_mutex};
        const auto it = nx2_list_pages.find(page);
        if (it == nx2_list_pages.end() || !current(it->second) || it->second.copy_tick != tick ||
            it->second.copy_writes != writes) {
            return Nx2ListRead::Unknown;
        }
        std::memcpy(out.data(), it->second.copy->mapped_span.data() + offset, out.size());
        return Nx2ListRead::Copied;
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListSynced(DAddr page) {
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        Nx2ListPage& entry = nx2_list_pages[page];
        entry.synced_writes = entry.writes;
    }
}

template <class P>
void BufferCache<P>::Nx2FlushListPages(std::span<const DAddr> pages) {
    constexpr u64 page_size = u64{1} << Tegra::Nx2WrittenPageBits;
    boost::container::small_vector<std::pair<DAddr, u64>, 16> ranges;
    for (const DAddr page : pages) {
        ranges.emplace_back(page, page_size);
    }
    boost::container::small_vector<std::pair<DAddr, u64>, 16> drawn;
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, entry] : nx2_list_pages) {
            if (entry.draw_written && entry.synced_writes != entry.writes) {
                ranges.emplace_back(page, page_size);
                drawn.emplace_back(page, entry.writes);
            }
        }
    }
    DownloadRangesMemory(ranges);
    if constexpr (!IS_OPENGL) {
        std::scoped_lock lock{nx2_list_mutex};
        for (const auto& [page, writes] : drawn) {
            const auto it = nx2_list_pages.find(page);
            if (it != nx2_list_pages.end() && it->second.writes == writes) {
                it->second.synced_writes = writes;
            }
        }
    }
}

template <class P>
void BufferCache<P>::Nx2NoteListHostWrite(DAddr device_addr, u64 size) {
    if constexpr (!IS_OPENGL) {
        if (size == 0 || !Tegra::g_nx2_has_command_list_pages.load(std::memory_order_relaxed)) {
            return;
        }
        constexpr u64 page_bits = Tegra::Nx2WrittenPageBits;
        const u64 last = (std::min<u64>)((device_addr + size - 1) >> page_bits,
                                         Tegra::g_nx2_command_list_pages.size() * 64 - 1);
        for (u64 page = device_addr >> page_bits; page <= last; ++page) {
            if (!((Tegra::g_nx2_command_list_pages[page / 64].load(std::memory_order_relaxed) >>
                   (page % 64)) &
                  1)) {
                continue;
            }
            std::scoped_lock lock{nx2_list_mutex};
            const auto it = nx2_list_pages.find(page << page_bits);
            if (it == nx2_list_pages.end()) {
                continue;
            }
            Nx2ListPage& entry = it->second;
            const bool synced = entry.synced_writes == entry.writes;
            ++entry.writes;
            if (synced) {
                entry.synced_writes = entry.writes;
            }
        }
    }
}

template <class P>
BufferId BufferCache<P>::FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible) {
    const u64 page = device_addr >> CACHING_PAGEBITS;
    if (device_addr == 0 || page >= page_table.size()) {
        return NULL_BUFFER_ID;
    }
    if (commit != nullptr) [[unlikely]] {
        // EDEN_NX2_V2_COMMIT: the pure lookup. A miss is a BufferCreate request, committed on
        // demand: the caller binds the buffer at once. F2-20: FindBuffer runs where the caches
        // are owned (the owner, or under their locks), so it reads the owner's page table; the
        // published map (LookupBuffer) is the shard readers' copy of it.
        const BufferId found = page_table[page];
        const bool usable = static_cast<bool>(PeekBuffer(device_addr, size, sparse_compatible));
        auto& stats = V2::Commit::Stats::Get();
        if (commit->stats) {
            V2::Commit::Stats::Count(usable ? stats.lookup_hits : stats.lookup_misses);
        }
        if (!usable && v2_proxy_finds) {
            // A BufferCreate request with a proxy: the binding holds the proxy and its device
            // address; the end of the update pass creates (joining whatever overlaps: a merge
            // resolves the proxy to the merged buffer, the offset comes from the address) and
            // patches it.
            return BufferId{
                v2_creates.Push(MakeCreateRequest(device_addr, size, sparse_compatible), true)};
        }
        if (!usable && commit->stats) {
            V2::Commit::Stats::Count(
                stats.requests[static_cast<size_t>(V2::Commit::Kind::BufferCreate)]);
        }
        if (commit->verify && !usable) {
            // A miss predicts a creation, and a merge when any page of the range has a buffer.
            const u64 begin = Common::AlignDown(device_addr, CACHING_PAGESIZE);
            const u64 end = Common::AlignUp(device_addr + (std::max)(size, 1U), CACHING_PAGESIZE);
            bool merges = false;
            for (u64 at = begin; at < end && !merges; at += CACHING_PAGESIZE) {
                merges = v2_pages->Load(at) != 0;
            }
            const u64 deletions_before = buffer_deletions;
            const BufferId created = CreateBuffer(device_addr, size, sparse_compatible);
            const bool merged = buffer_deletions != deletions_before;
            V2::Commit::Stats::Count(stats.verify_draws);
            if (merged != merges || !created) {
                V2::Commit::Stats::Count(stats.verify_mismatches);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                    LOG_WARNING(HW_GPU,
                                "V2 commit verify: buffer create at {:#x}+{:#x}: predicted {}, "
                                "the old path {}",
                                device_addr, size, merges ? "a merge" : "a new buffer",
                                merged ? "merged" : "made a new buffer");
                }
            }
            return created;
        }
        if (commit->verify) {
            // The published map must agree with the page table, and predict the old path.
            const BufferId table_id = page_table[page];
            V2::Commit::Stats::Count(stats.verify_draws);
            if (found && found != table_id) {
                V2::Commit::Stats::Count(stats.verify_mismatches);
                static std::atomic<u32> logged{};
                if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                    LOG_WARNING(HW_GPU,
                                "V2 commit verify: buffer page map says {} at {:#x}, the page "
                                "table {}",
                                found.index, device_addr, table_id.index);
                }
            }
        } else if (usable) {
            if (Settings::IsGPUFenceBehaviorAccurate()) {
                SynchronizeBufferWrites(slot_buffers[found]);
            }
            return found;
        }
    }
    const BufferId buffer_id = page_table[page];
    if (buffer_id) {
        Buffer& buffer = slot_buffers[buffer_id];
        if (buffer.IsInBounds(device_addr, size)) {
            const bool should_sync = Settings::IsGPUFenceBehaviorAccurate();
            if (should_sync) {
                SynchronizeBufferWrites(buffer);
            }
            bool usable = true;
            if constexpr (requires { buffer.IsSparseCompatible(); }) {
                if (sparse_compatible && !buffer.IsSparseCompatible()) {
                    usable = false;
                }
            }
            if (usable) {
                return buffer_id;
            }
        }
    }
    return CreateBuffer(device_addr, size, sparse_compatible);
}

template <class P>
void BufferCache<P>::SynchronizeBufferWrites(Buffer& buffer) {
    if constexpr (!IS_OPENGL) {
        const u64 buffer_tick = buffer.getWriteTick();
        if (!runtime.IsFree(buffer_tick)) {
            runtime.Wait(buffer_tick);
        }
    }
}

template <class P>
typename BufferCache<P>::OverlapResult BufferCache<P>::ResolveOverlaps(DAddr device_addr,
                                                                       u32 wanted_size) {
    static constexpr int STREAM_LEAP_THRESHOLD = 16;
    boost::container::small_vector<BufferId, 16> overlap_ids;
    DAddr begin = device_addr;
    DAddr end = device_addr + wanted_size;
    int stream_score = 0;
    bool has_stream_leap = false;
    auto expand_begin = [&](DAddr add_value) {
        static constexpr DAddr min_page = CACHING_PAGESIZE + Core::DEVICE_PAGESIZE;
        if (add_value > begin - min_page) {
            begin = min_page;
            device_addr = Core::DEVICE_PAGESIZE;
            return;
        }
        begin -= add_value;
        device_addr = begin - CACHING_PAGESIZE;
    };
    auto expand_end = [&](DAddr add_value) {
        static constexpr DAddr max_page = 1ULL << Tegra::MaxwellDeviceMemoryManager::AS_BITS;
        if (add_value > max_page - end) {
            end = max_page;
            return;
        }
        end += add_value;
    };
    if (begin == 0) {
        return OverlapResult{
            .ids = std::move(overlap_ids),
            .begin = begin,
            .end = end,
            .has_stream_leap = has_stream_leap,
        };
    }
    for (; device_addr >> CACHING_PAGEBITS < Common::DivCeil(end, CACHING_PAGESIZE);
         device_addr += CACHING_PAGESIZE) {
        const BufferId overlap_id = page_table[device_addr >> CACHING_PAGEBITS];
        if (!overlap_id) {
            continue;
        }
        Buffer& overlap = slot_buffers[overlap_id];
        if (overlap.IsPicked()) {
            continue;
        }
        overlap_ids.push_back(overlap_id);
        overlap.Pick();
        const DAddr overlap_device_addr = overlap.CpuAddr();
        const bool expands_left = overlap_device_addr < begin;
        if (expands_left) {
            begin = overlap_device_addr;
        }
        const DAddr overlap_end = overlap_device_addr + overlap.SizeBytes();
        const bool expands_right = overlap_end > end;
        if (overlap_end > end) {
            end = overlap_end;
        }
        stream_score += overlap.StreamScore();
        if (stream_score > STREAM_LEAP_THRESHOLD && !has_stream_leap) {
            // When this memory region has been joined a bunch of times, we assume it's being used
            // as a stream buffer. Increase the size to skip constantly recreating buffers.
            has_stream_leap = true;
            if (expands_right) {
                expand_begin(CACHING_PAGESIZE * 128);
            }
            if (expands_left) {
                expand_end(CACHING_PAGESIZE * 128);
            }
        }
    }
    return OverlapResult{
        .ids = std::move(overlap_ids),
        .begin = begin,
        .end = end,
        .has_stream_leap = has_stream_leap,
    };
}

template <class P>
void BufferCache<P>::JoinOverlap(BufferId new_buffer_id, BufferId overlap_id,
                                 bool accumulate_stream_score) {
    Buffer& new_buffer = slot_buffers[new_buffer_id];
    Buffer& overlap = slot_buffers[overlap_id];
    if (accumulate_stream_score) {
        new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
    }
    boost::container::small_vector<BufferCopy, 10> copies;
    const size_t dst_base_offset = overlap.CpuAddr() - new_buffer.CpuAddr();
    copies.push_back(BufferCopy{
        .src_offset = 0,
        .dst_offset = dst_base_offset,
        .size = overlap.SizeBytes(),
    });
    new_buffer.MarkUsage(copies[0].dst_offset, copies[0].size);
    runtime.CopyBuffer(new_buffer, overlap, copies, true);
#ifdef YUZU_LEGACY
    if (immediately_free)
        runtime.Finish();
#endif
    DeleteBuffer(overlap_id, true);
}

template <class P>
BufferId BufferCache<P>::CreateBuffer(DAddr device_addr, u32 wanted_size,
                                      bool sparse_compatible) {
    DAddr device_addr_end = Common::AlignUp(device_addr + wanted_size, CACHING_PAGESIZE);
    device_addr = Common::AlignDown(device_addr, CACHING_PAGESIZE);
    wanted_size = static_cast<u32>(device_addr_end - device_addr);
    const OverlapResult overlap = ResolveOverlaps(device_addr, wanted_size);
    const u32 size = static_cast<u32>(overlap.end - overlap.begin);
    const BufferId new_buffer_id =
        slot_buffers.insert(runtime, overlap.begin, size, sparse_compatible);
    auto& new_buffer = slot_buffers[new_buffer_id];
    const size_t size_bytes = new_buffer.SizeBytes();
    runtime.ClearBuffer(new_buffer, 0, size_bytes, 0);
    new_buffer.MarkUsage(0, size_bytes);
    for (const BufferId overlap_id : overlap.ids) {
        JoinOverlap(new_buffer_id, overlap_id, !overlap.has_stream_leap);
    }
    Register(new_buffer_id);
    TouchBuffer(new_buffer, new_buffer_id);
    if (commit != nullptr) [[unlikely]] {
        V2::Commit::NoteEffect(commit, V2::Commit::EffectCreate);
    }
    return new_buffer_id;
}

template <class P>
void BufferCache<P>::Register(BufferId buffer_id) {
    ChangeRegister<true>(buffer_id);
}

template <class P>
void BufferCache<P>::Unregister(BufferId buffer_id) {
    ChangeRegister<false>(buffer_id);
}

template <class P>
template <bool insert>
void BufferCache<P>::ChangeRegister(BufferId buffer_id) {
    Buffer& buffer = slot_buffers[buffer_id];
    const auto size = buffer.SizeBytes();
    if (insert) {
        buffer_bytes += Common::AlignUp(size, 1024);
        total_used_memory += Common::AlignUp(size, 1024);
        buffer.setLRUID(lru_cache.Insert(buffer_id, frame_tick));
    } else {
        buffer_bytes -= Common::AlignUp(size, 1024);
        total_used_memory -= Common::AlignUp(size, 1024);
        lru_cache.Free(buffer.getLRUID());
    }
    const DAddr device_addr_begin = buffer.CpuAddr();
    const DAddr device_addr_end = device_addr_begin + size;
    const u64 page_begin = device_addr_begin / CACHING_PAGESIZE;
    const u64 page_end = Common::DivCeil(device_addr_end, CACHING_PAGESIZE);
    for (u64 page = page_begin; page != page_end; ++page) {
        if constexpr (insert) {
            page_table[page] = buffer_id;
        } else {
            page_table[page] = BufferId{};
        }
    }
    if (v2_pages) [[unlikely]] {
        v2_pages->Store(device_addr_begin, device_addr_end, insert ? buffer_id.index : 0U);
    }
}

template <class P>
void BufferCache<P>::SetCommit(V2::Commit::Context* context) {
    commit = context;
    if (commit == nullptr) {
        v2_pages.reset();
        return;
    }
    v2_pages = std::make_unique<V2::VaMap<CACHING_PAGEBITS, 36>>();
    for (u64 page = 0; page < page_table.size(); ++page) {
        if (page_table[page]) {
            v2_pages->Store(page << CACHING_PAGEBITS, (page + 1) << CACHING_PAGEBITS,
                            page_table[page].index);
        }
    }
}

template <class P>
BufferId BufferCache<P>::LookupBuffer(DAddr device_addr, u32 size) const {
    if (!v2_pages || device_addr == 0) {
        return NULL_BUFFER_ID;
    }
    // A buffer registers every page it spans: the first page's buffer covers the range exactly
    // when its bounds hold it (the caller checks, as FindBuffer does).
    const u32 slot = v2_pages->Load(device_addr);
    if (slot == 0 || !slot_buffers[BufferId{slot}].IsInBounds(device_addr, size)) {
        return NULL_BUFFER_ID;
    }
    return BufferId{slot};
}

template <class P>
void BufferCache<P>::TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept {
    if (buffer_id != NULL_BUFFER_ID) {
        // Perf fix C: a touch (LRU and track id) is applied in place even in a bind scope: no
        // lookup reads it and its order against the scope's requests changes nothing.
        lru_cache.Touch(buffer.getLRUID(), frame_tick);
        if (commit != nullptr) [[unlikely]] {
            buffer.v2_track_id = (std::max)(buffer.v2_track_id, commit->submission);
        }
    }
}

template <class P>
bool BufferCache<P>::SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size) {
    // F2-19 (§5.6 commit at the mutation rate): an upload request only when the pages changed.
    // Nothing in a bind scope makes a range CPU-modified (uploads clear it, written marks are
    // GPU writes), so a range clean now would upload nothing when the request applies.
    if (v2_defer && !memory_tracker.IsRegionCpuModified(device_addr, size)) [[unlikely]] {
        return false;
    }
    if (v2_defer &&
        V2DeferBind(V2::Commit::Kind::BufferUpload, buffer, device_addr, size, 0)) [[unlikely]] {
        // EDEN_NX2_V2_COMMIT: an ordered upload request; the bind only needs the buffer and its
        // offset. The result only feeds a stat.
        return false;
    }
    if (commit != nullptr && commit->verify) [[unlikely]] {
        // COMMIT=verify: an upload request is predicted from the CPU-modified state; an upload
        // the prediction missed is a mismatch (one it predicted but GPU-written ranges covered
        // is dependent on them).
        const bool predicted = memory_tracker.IsRegionCpuModified(device_addr, size);
        const u64 before = v2_upload_count;
        const bool result = SynchronizeBufferImpl(buffer, device_addr, size);
        const bool uploaded = v2_upload_count != before;
        auto& stats = V2::Commit::Stats::Get();
        V2::Commit::Stats::Count(stats.verify_draws);
        if (uploaded && !predicted) {
            V2::Commit::Stats::Count(stats.verify_mismatches);
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                LOG_WARNING(HW_GPU, "V2 commit verify: buffer upload at {:#x}+{:#x} not predicted",
                            device_addr, size);
            }
        } else if (predicted && !uploaded) {
            V2::Commit::Stats::Count(stats.verify_dependent);
        }
        return result;
    }
    return SynchronizeBufferImpl(buffer, device_addr, size);
}

template <class P>
bool BufferCache<P>::SynchronizeBufferImpl(Buffer& buffer, DAddr device_addr, u32 size) {
    // An upload over a clean word of the tracker (word_manager.h) finds and changes nothing:
    // walk only the dirty ones. Verify walks every word and checks the clean words.
    const bool skip_clean = !Nx2Perf::Verify();
    if (Nx2Perf::Verify()) {
        static std::atomic<u32> logged{};
        if (!memory_tracker.CleanWordsHold(device_addr, size) &&
            logged.fetch_add(1, std::memory_order_relaxed) < 64) {
            LOG_ERROR(HW_GPU, "Nx2Perf draw-binds: clean tracker words wrong near {:#x}+{:#x}",
                      device_addr, size);
        }
    }
    const auto for_each_upload_range = [&](auto&& func) {
        if (skip_clean) {
            memory_tracker.ForEachDirtyUploadRange(device_addr, size, func);
        } else {
            memory_tracker.ForEachUploadRange(device_addr, size, func);
        }
    };
#ifdef __ANDROID__
    const bool use_optimized_vertex_buffers = Settings::values.use_optimized_vertex_buffers.GetValue();
#else
    constexpr bool use_optimized_vertex_buffers = true;
#endif

    if (use_optimized_vertex_buffers) {
        upload_copies.clear();
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        const DAddr buffer_start = buffer.cpu_addr_cached;
        const auto add_upload = [&](DAddr start, DAddr end) {
            if (start == end) return;
            const u64 range_size = end - start;
            upload_copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = start - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        };
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            DAddr upload_start = device_addr_out;
            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, [&](DAddr gpu_start, DAddr gpu_end) {
                add_upload(upload_start, gpu_start);
                upload_start = gpu_end;
            });
            add_upload(upload_start, device_addr_out + range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(upload_copies.data(), upload_copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        // Mapped uploads carry their own barriers (or run in the upload command buffer, which
        // ends with one); another barrier here only ended the render pass (~50 times a frame in
        // Pokopia).
        if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
            any_buffer_uploaded = true;
        }
        return false;
    } else {
        boost::container::small_vector<BufferCopy, 4> copies;
        u64 total_size_bytes = 0;
        u64 largest_copy = 0;
        DAddr buffer_start = buffer.CpuAddr();
        for_each_upload_range([&](u64 device_addr_out, u64 range_size) {
            copies.push_back(BufferCopy{
                .src_offset = total_size_bytes,
                .dst_offset = device_addr_out - buffer_start,
                .size = range_size,
            });
            total_size_bytes += range_size;
            largest_copy = (std::max)(largest_copy, range_size);
        });
        if (total_size_bytes == 0) {
            return true;
        }
        const std::span<BufferCopy> copies_span(copies.data(), copies.size());
        UploadMemory(buffer, total_size_bytes, largest_copy, copies_span);
        return false;
    }
}

template <class P>
void BufferCache<P>::SynchronizeGeometryBuffer(u32 slot, BufferId buffer_id, DAddr device_addr,
                                               u32 size) {
    if (v2_defer && buffer_id) [[unlikely]] {
        V2::Commit::Request request{};
        request.kind = V2::Commit::Kind::BufferGeometrySync;
        request.resource = V2::Commit::Resource::Buffer;
        request.slot = buffer_id.index;
        request.address = device_addr;
        request.size = size;
        request.arg = slot;
        v2_binds.Push(request);
        return;
    }
    Buffer& buffer = slot_buffers[buffer_id];
    // The deletion count keeps a freed and reused buffer id from passing for the scanned one.
    GeometrySync& sync = geometry_syncs[slot];
    ++geometry_stats.syncs;
    if (memory_tracker.IsSynchronized(slot) && sync.buffer_id == buffer_id &&
        sync.device_addr == device_addr && sync.size == size &&
        sync.deletions == buffer_deletions) {
        ++geometry_stats.syncs_skipped;
        if (Nx2Perf::Verify() && !SynchronizeBuffer(buffer, device_addr, size)) {
            ++geometry_stats.verify_mismatches;
        }
        return;
    }
    SynchronizeBuffer(buffer, device_addr, size);
    sync = GeometrySync{
        .buffer_id = buffer_id,
        .device_addr = device_addr,
        .size = size,
        .deletions = buffer_deletions,
    };
    memory_tracker.WatchSynchronized(slot, device_addr, size);
}

template <class P>
void BufferCache<P>::UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                                  std::span<BufferCopy> copies) {
    VideoCommon::V2::Stats::CountMutation(VideoCommon::V2::Stats::Mutation::BufferUpload,
                                          total_size_bytes);
    if (commit != nullptr) [[unlikely]] {
        // A BufferUpload request applied (deferred to CommitBinds with COMMIT=1).
        ++v2_upload_count;
        V2::Commit::NoteEffect(commit, V2::Commit::EffectUpload);
        if (commit->stats) {
            auto& stats = V2::Commit::Stats::Get();
            V2::Commit::Stats::Count(
                stats.requests[static_cast<size_t>(V2::Commit::Kind::BufferUpload)]);
            if (buffer.v2_track_id < commit->submission) {
                V2::Commit::Stats::Count(stats.hoistable);
            }
        }
    }
    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        MappedUploadMemory(buffer, total_size_bytes, copies);
    } else {
        ImmediateUploadMemory(buffer, largest_copy, copies);
    }
}

template <class P>
void BufferCache<P>::ImmediateUploadMemory([[maybe_unused]] Buffer& buffer,
                                           [[maybe_unused]] u64 largest_copy,
                                           [[maybe_unused]] std::span<const BufferCopy> copies) {
    if constexpr (!USE_MEMORY_MAPS_FOR_UPLOADS) {
        std::span<u8> immediate_buffer;
        for (const BufferCopy& copy : copies) {
            std::span<const u8> upload_span;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            if (IsRangeGranular(device_addr, copy.size)) {
                auto* const ptr = device_memory.GetPointer<u8>(device_addr);
                if (ptr != nullptr) {
                    upload_span = std::span(ptr, copy.size);
                }
            } else {
                if (immediate_buffer.empty()) {
                    immediate_buffer = ImmediateBuffer(largest_copy);
                }
                device_memory.ReadBlockUnsafe(device_addr, immediate_buffer.data(), copy.size);
                upload_span = immediate_buffer.subspan(0, copy.size);
            }
            buffer.ImmediateUpload(copy.dst_offset, upload_span);
        }
    }
}

template <class P>
void BufferCache<P>::MappedUploadMemory([[maybe_unused]] Buffer& buffer,
                                        [[maybe_unused]] u64 total_size_bytes,
                                        [[maybe_unused]] std::span<BufferCopy> copies) {
    if constexpr (USE_MEMORY_MAPS) {
        auto upload_staging = runtime.UploadStagingBuffer(total_size_bytes);
        const std::span<u8> staging_pointer = upload_staging.mapped_span;
        for (BufferCopy& copy : copies) {
            u8* const src_pointer = staging_pointer.data() + copy.src_offset;
            const DAddr device_addr = buffer.CpuAddr() + copy.dst_offset;
            device_memory.ReadBlockUnsafe(device_addr, src_pointer, copy.size);
            // Large uploads are mostly geometry the host GPU now has its own copy of: TickFrame
            // marks their guest memory cold (see TextureCache::RefreshContents).
            if (copy.size >= 256_KiB) {
                uploaded_buffer_ranges.emplace_back(device_addr, copy.size);
            }
            // Apply the staging offset
            copy.src_offset += upload_staging.offset;
        }
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    }
}

template <class P>
bool BufferCache<P>::InlineMemory(DAddr dest_address, size_t copy_size,
                                  std::span<const u8> inlined_buffer) {
    const bool is_dirty = IsRegionRegistered(dest_address, copy_size);
    if (!is_dirty) {
        return false;
    }
    DAddr aligned_start = Common::AlignDown(dest_address, DEVICE_PAGESIZE);
    DAddr aligned_end = Common::AlignUp(dest_address + copy_size, DEVICE_PAGESIZE);
    if (!IsRegionGpuModified(aligned_start, aligned_end - aligned_start)) {
        return false;
    }

    InlineMemoryImplementation(dest_address, copy_size, inlined_buffer);

    return true;
}

template <class P>
void BufferCache<P>::InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                                std::span<const u8> inlined_buffer) {
    V2::Versions::HookWrite(dest_address, copy_size, V2::Versions::Source::Inline);
    ClearDownload(dest_address, copy_size);
    gpu_modified_ranges.Subtract(dest_address, copy_size);
    MarkVertexBuffersStale(dest_address, copy_size);

    BufferId buffer_id = FindBuffer(dest_address, static_cast<u32>(copy_size), false);
    auto& buffer = slot_buffers[buffer_id];
    SynchronizeBuffer(buffer, dest_address, static_cast<u32>(copy_size));

    if constexpr (USE_MEMORY_MAPS_FOR_UPLOADS) {
        auto upload_staging = runtime.UploadStagingBuffer(copy_size);
        std::array copies{BufferCopy{
            .src_offset = upload_staging.offset,
            .dst_offset = buffer.Offset(dest_address),
            .size = copy_size,
        }};
        u8* const src_pointer = upload_staging.mapped_span.data();
        std::memcpy(src_pointer, inlined_buffer.data(), copy_size);
        const bool can_reorder = runtime.CanReorderUpload(buffer, copies);
        runtime.CopyBuffer(buffer, upload_staging.buffer, copies, true, can_reorder);
    } else {
        buffer.ImmediateUpload(buffer.Offset(dest_address), inlined_buffer.first(copy_size));
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer) {
    DownloadBufferMemory(buffer, buffer.CpuAddr(), buffer.SizeBytes());
}

template <class P>
void BufferCache<P>::DownloadBuffersMemory(std::span<const BufferId> buffer_ids) {
    struct Download {
        BufferId buffer_id;
        BufferCopy copy;
    };
    std::vector<Download> downloads;
    u64 total_size_bytes = 0;
    for (const BufferId buffer_id : buffer_ids) {
        Buffer& buffer = slot_buffers[buffer_id];
        const DAddr buffer_addr = buffer.CpuAddr();
        memory_tracker.ForEachDownloadRangeAndClear(
            buffer_addr, buffer.SizeBytes(), [&](u64 device_addr_out, u64 range_size) {
                gpu_modified_ranges.ForEachInRange(device_addr_out, range_size,
                                                   [&](DAddr start, DAddr end) {
                    downloads.push_back({buffer_id, BufferCopy{
                                                        .src_offset = start - buffer_addr,
                                                        .dst_offset = total_size_bytes,
                                                        .size = end - start,
                                                    }});
                    total_size_bytes += Common::AlignUp(end - start, 64);
                });
                ClearDownload(device_addr_out, range_size);
                gpu_modified_ranges.Subtract(device_addr_out, range_size);
            });
    }
    if (total_size_bytes == 0) {
        return;
    }
    auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
    for (Download& download : downloads) {
        download.copy.dst_offset += download_staging.offset;
        Buffer& buffer = slot_buffers[download.buffer_id];
        buffer.MarkUsage(download.copy.src_offset, download.copy.size);
        runtime.CopyBuffer(download_staging.buffer, buffer, std::span{&download.copy, 1}, true);
    }
    runtime.Finish();
    const u8* const mapped_memory = download_staging.mapped_span.data();
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();
    for (const Download& download : downloads) {
        const DAddr device_addr = slot_buffers[download.buffer_id].CpuAddr() + download.copy.src_offset;
        WriteDownloadedMemory(
            cpu_written, device_addr,
            mapped_memory + (download.copy.dst_offset - download_staging.offset),
            download.copy.size);
    }
}

template <class P>
void BufferCache<P>::DownloadBufferMemory(Buffer& buffer, DAddr device_addr, u64 size) {
    boost::container::small_vector<BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    u64 largest_copy = 0;
    memory_tracker.ForEachDownloadRangeAndClear(
        device_addr, size, [&](u64 device_addr_out, u64 range_size) {
            const DAddr buffer_addr = buffer.CpuAddr();
            const auto add_download = [&](DAddr start, DAddr end) {
                const u64 new_offset = start - buffer_addr;
                const u64 new_size = end - start;
                copies.push_back(BufferCopy{
                    .src_offset = new_offset,
                    .dst_offset = total_size_bytes,
                    .size = new_size,
                });
                // Align up to avoid cache conflicts
                constexpr u64 align = 64ULL;
                constexpr u64 mask = ~(align - 1ULL);
                total_size_bytes += (new_size + align - 1) & mask;
                largest_copy = (std::max)(largest_copy, new_size);
            };

            gpu_modified_ranges.ForEachInRange(device_addr_out, range_size, add_download);
            ClearDownload(device_addr_out, range_size);
            gpu_modified_ranges.Subtract(device_addr_out, range_size);
        });
    if (total_size_bytes == 0) {
        return;
    }
    const Common::RangeSet<DAddr> cpu_written = PendingCpuWriteRanges();

    if constexpr (requires { buffer.Mapped(); }) {
        if (const auto mapped = buffer.Mapped(); !mapped.empty()) {
            runtime.FinishForHostRead();
            for (const BufferCopy& copy : copies) {
                WriteDownloadedMemory(cpu_written, buffer.CpuAddr() + copy.src_offset,
                                      mapped.data() + copy.src_offset, copy.size);
            }
            return;
        }
    }
    if constexpr (USE_MEMORY_MAPS) {
        auto download_staging = runtime.DownloadStagingBuffer(total_size_bytes);
        const u8* const mapped_memory = download_staging.mapped_span.data();
        const std::span<BufferCopy> copies_span(copies.data(), copies.data() + copies.size());
        for (BufferCopy& copy : copies) {
            // Modify copies to have the staging offset in mind
            copy.dst_offset += download_staging.offset;
            buffer.MarkUsage(copy.src_offset, copy.size);
        }
        runtime.CopyBuffer(download_staging.buffer, buffer, copies_span, true);
        runtime.Finish();
        for (const BufferCopy& copy : copies) {
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            // Undo the modified offset
            const u64 dst_offset = copy.dst_offset - download_staging.offset;
            const u8* copy_mapped_memory = mapped_memory + dst_offset;
            WriteDownloadedMemory(cpu_written, copy_device_addr, copy_mapped_memory, copy.size);
        }
    } else {
        const std::span<u8> immediate_buffer = ImmediateBuffer(largest_copy);
        for (const BufferCopy& copy : copies) {
            buffer.ImmediateDownload(copy.src_offset, immediate_buffer.subspan(0, copy.size));
            const DAddr copy_device_addr = buffer.CpuAddr() + copy.src_offset;
            WriteDownloadedMemory(cpu_written, copy_device_addr, immediate_buffer.data(), copy.size);
        }
    }
}

template <class P>
void BufferCache<P>::DeleteBuffer(BufferId buffer_id, bool do_not_mark) {
    ++buffer_deletions;
    VideoCommon::V2::Stats::CountMutation(VideoCommon::V2::Stats::Mutation::BufferDelete);
    if (commit != nullptr) [[unlikely]] {
        V2::Commit::NoteEffect(commit, V2::Commit::EffectDelete);
    }
    if constexpr (requires { runtime.OnBufferDeleted(slot_buffers[buffer_id]); }) {
        runtime.OnBufferDeleted(slot_buffers[buffer_id]);
    }
    read_trapped_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(),
                                 slot_buffers[buffer_id].SizeBytes());
    cpu_read_ranges.Subtract(slot_buffers[buffer_id].CpuAddr(), slot_buffers[buffer_id].SizeBytes());
    bool dirty_index{false};
    boost::container::small_vector<u64, NUM_VERTEX_BUFFERS> dirty_vertex_buffers;
    const auto scalar_replace = [buffer_id](Binding& binding) {
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
        }
    };
    const auto replace = [scalar_replace](std::span<Binding> bindings) {
        std::ranges::for_each(bindings, scalar_replace);
    };

    if (channel_state->index_buffer.buffer_id == buffer_id) {
        channel_state->index_buffer.buffer_id = BufferId{};
        dirty_index = true;
    }

    for (u32 index = 0; index < channel_state->vertex_buffers.size(); index++) {
        auto& binding = channel_state->vertex_buffers[index];
        if (binding.buffer_id == buffer_id) {
            binding.buffer_id = BufferId{};
            UpdateVertexBufferSlot(index, binding);
            dirty_vertex_buffers.push_back(index);
        }
    }
    std::ranges::for_each(channel_state->uniform_buffers, replace);
    std::ranges::for_each(channel_state->storage_buffers, replace);
    replace(channel_state->transform_feedback_buffers);
    replace(channel_state->compute_uniform_buffers);
    replace(channel_state->compute_storage_buffers);

    // Adapted from the pre-refresh branch: merged buffers invalidate segment references too.
    InvalidateMultiRangeSegments(std::span{graphics_segments}, buffer_id);
    InvalidateMultiRangeSegments(std::span{compute_segments}, buffer_id);

    // Mark the whole buffer as CPU written to stop tracking CPU writes
    if (!do_not_mark) {
        Buffer& buffer = slot_buffers[buffer_id];
        // Every CPU-modified mark moves the versions (EDEN_NX2_V2_DRAWCACHE buffer replay).
        V2::Versions::HookWrite(buffer.CpuAddr(), buffer.SizeBytes(), V2::Versions::Source::Host);
        memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddr(), buffer.SizeBytes());
    }

    Unregister(buffer_id);

#ifdef YUZU_LEGACY
    if (!do_not_mark || !immediately_free)
#endif
        delayed_destruction_ring.Push(std::move(slot_buffers[buffer_id]));

    slot_buffers.erase(buffer_id);
    V2::Versions::HookBump(V2::Versions::Kind::Buffer, buffer_id.Value());

    if constexpr (HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS) {
        channel_state->dirty_uniform_buffers.fill(~u32{0});
        channel_state->uniform_buffer_binding_sizes.fill({});
    }

    auto& flags = maxwell3d->dirty.flags;
    if (dirty_index) {
        flags[Dirty::IndexBuffer] = true;
    }

    if (dirty_vertex_buffers.size() > 0) {
        flags[Dirty::VertexBuffers] = true;
        for (auto index : dirty_vertex_buffers) {
            flags[Dirty::VertexBuffer0 + index] = true;
        }
    }
    channel_state->has_deleted_buffers = true;
}

template <class P>
void BufferCache<P>::TraceTablePointer(bool compute, size_t stage, size_t ssbo_index,
                                       u32 cbuf_index, u32 cbuf_offset, u32 indirect_offset,
                                       GPUVAddr holder, const Binding& binding) {
    // A table entry is read here from guest memory, and again by the shader from the holder's
    // host buffer. Logs each change of a slot's entry or size, and whether the two copies of the
    // entry can differ: GPU-written (host newer) or CPU-written and not yet uploaded (guest newer).
    static const bool enabled = std::getenv("EDEN_NX2_TABLE_POINTER_TRACE") != nullptr;
    if (!enabled) {
        return;
    }
    const GPUVAddr entry_addr = holder + indirect_offset;
    const GPUVAddr entry = gpu_memory->Read<u64>(entry_addr);
    const u64 key = (u64{compute} << 63) | (u64{stage & 7} << 60) | (u64{ssbo_index & 63} << 54) |
                    (u64{cbuf_index & 31} << 49) | (u64{cbuf_offset & 0x7ffff} << 30) |
                    (indirect_offset & 0x3fffffff);
    const std::pair<GPUVAddr, u32> state{entry, binding.size};
    const auto [it, inserted] = traced_table_pointers.try_emplace(key, state);
    if (!inserted && it->second == state) {
        return;
    }
    it->second = state;
    static constexpr u64 max_lines = 100000;
    static u64 lines{};
    if (++lines > max_lines) {
        if (lines == max_lines + 1) {
            LOG_INFO(HW_GPU, "Table pointer trace stopped after {} lines", max_lines);
        }
        return;
    }
    const std::optional<DAddr> entry_device = gpu_memory->GpuToCpuAddress(entry_addr);
    const bool gpu_modified =
        entry_device && memory_tracker.IsRegionGpuModified(*entry_device, sizeof(u64));
    const bool cpu_modified =
        entry_device && memory_tracker.IsRegionCpuModified(*entry_device, sizeof(u64));
    LOG_INFO(HW_GPU,
             "Table pointer frame={} {} stage={} ssbo={} c{}[{:#x}]+{:#x}: holder={:#x} "
             "entry={:#x} bound={:#x} size={:#x} entry_gpu_modified={} entry_cpu_modified={}",
             frame_tick, compute ? "compute" : "graphics", stage, ssbo_index, cbuf_index,
             cbuf_offset, indirect_offset, holder, entry, binding.device_addr, binding.size,
             gpu_modified, cpu_modified);
}

template <class P>
Binding BufferCache<P>::StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                             bool native_pointer, bool table_pointer) const {
    const GPUVAddr gpu_addr = gpu_memory->Read<u64>(ssbo_addr);

    if (gpu_addr == 0) {
        return NULL_BINDING;
    }

    const bool reads_size_word = !(table_pointer && !is_written) && !native_pointer;
    const u64 next_qword = reads_size_word ? gpu_memory->Read<u64>(ssbo_addr + 8) : 0;
    const u32 alignment = runtime.GetStorageBufferAlignment();
    struct CachedBinding {
        size_t owner;
        GPUVAddr gpu_addr;
        u64 next_qword;
        u64 serial;
        u32 alignment;
        u32 flags;
        Binding binding;
    };
    constexpr size_t ways = 4;
    thread_local std::array<std::array<CachedBinding, ways>, 512> binding_cache{};
    const u32 flags =
        (is_written ? 1U : 0U) | (native_pointer ? 2U : 0U) | (table_pointer ? 4U : 0U);
    const size_t owner = gpu_memory->GetID() + 1;
    const u64 serial = gpu_memory->MapSerial();
    auto& set = binding_cache[((gpu_addr ^ next_qword ^ flags) * 0x9E3779B97F4A7C15ULL) >> 55];
    for (size_t way = 0; way < ways; ++way) {
        const CachedBinding& cached = set[way];
        if (cached.owner == owner && cached.gpu_addr == gpu_addr &&
            cached.next_qword == next_qword && cached.serial == serial &&
            cached.alignment == alignment && cached.flags == flags && !Nx2Perf::Verify()) {
            const Binding hit = cached.binding;
            std::rotate(set.begin(), set.begin() + way, set.begin() + way + 1);
            return hit;
        }
    }
    const Binding binding = [&]() -> Binding {
        u32 size = [&]() {
            const u32 memory_layout_size = static_cast<u32>((std::min)(
                u64{gpu_memory->GetMemoryLayoutSize(gpu_addr)}, u64{(std::numeric_limits<u32>::max)()}));
            // A pointer read from a table in memory is the base of an object pool, read anywhere in
            // it: bound to the end of its mapping, as far as a storage buffer may reach (at most
            // 1 GiB). Ocarina of Time's grass instances lie 12-18 MiB past theirs (an 8 MiB cap
            // culled every blade), and its light probe records reach about 130 MiB past theirs: a
            // 64 MiB cap left the far probes outside the binding, and the fog took one overbright
            // constant from them. Written pools keep a cap: a write marks the whole binding
            // GPU-modified.
            if (table_pointer && !is_written) {
                const u64 reach = (std::min)(runtime.GetMaxStorageBufferRange(), u64{1_GiB});
                const u32 limit = static_cast<u32>(reach - (std::min)(reach, u64{64_KiB}));
                return (std::min)(memory_layout_size, limit);
            }
            // A native (Switch 2) compute program's pointer is bare: the word after it is another
            // parameter (a dimension, stride or texture handle), not an NVN buffer size. Taking it
            // as one bound Fortune's Weave's DLSS input features to 266 bytes and dropped the rest.
            if (native_pointer) {
                return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
            }
            const u32 packed_size = static_cast<u32>(next_qword);
            const bool next_qword_is_size = static_cast<u32>(next_qword >> 32) == 0 &&
                                            packed_size != 0 && packed_size <= memory_layout_size;
            if (next_qword_is_size) {
                return packed_size;
            }
            return (std::min)(memory_layout_size, static_cast<u32>(8_MiB));
        }();
        if ((native_pointer || table_pointer) && size != 0) {
            // Stop where the mapping or its device range does: the binding is one device range, and
            // a written one isn't split into separate ranges without sparse sources.
            size = static_cast<u32>(gpu_memory->MaxContinuousRange(gpu_addr, size));
        }

        // Alignment only applies to the offset of the buffer
        const GPUVAddr aligned_gpu_addr = Common::AlignDown(gpu_addr, alignment);
        const u32 aligned_size = static_cast<u32>(gpu_addr - aligned_gpu_addr) + size;

        const std::optional<DAddr> aligned_device_addr =
            gpu_memory->GpuToCpuAddress(aligned_gpu_addr);
        if (!aligned_device_addr || size == 0) {
            LOG_DEBUG(HW_GPU, "Failed to find storage buffer for cbuf index {}", cbuf_index);
            return NULL_BINDING;
        }
        const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
        ASSERT_MSG(device_addr, "Unaligned storage buffer address not found for cbuf index {}",
                   cbuf_index);
        // The end address used for size calculation does not need to be aligned
        const DAddr cpu_end = Common::AlignUp(*device_addr + size, Core::DEVICE_PAGESIZE);

        u32 binding_size = static_cast<u32>(cpu_end - *aligned_device_addr);
        if (is_written) {
            binding_size = aligned_size;
        }
        return Binding{
            .device_addr = *aligned_device_addr,
            .gpu_addr = aligned_gpu_addr,
            .size = binding_size,
            .buffer_id = BufferId{},
        };
    }();
    std::rotate(set.begin(), set.end() - 1, set.end());
    set[0] = {owner, gpu_addr, next_qword, serial, alignment, flags, binding};
    return binding;
}

template <class P>
TextureBufferBinding BufferCache<P>::GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                             PixelFormat format) {
    const std::optional<DAddr> device_addr = gpu_memory->GpuToCpuAddress(gpu_addr);
    TextureBufferBinding binding;
    if (!device_addr || size == 0) {
        binding.device_addr = 0;
        binding.size = 0;
        binding.buffer_id = NULL_BUFFER_ID;
        binding.format = PixelFormat::Invalid;
    } else {
        binding.device_addr = *device_addr;
        binding.size = size;
        binding.buffer_id = BufferId{};
        binding.format = format;
    }
    return binding;
}

template <class P>
std::span<const u8> BufferCache<P>::ImmediateBufferWithData(DAddr device_addr, size_t size) {
    u8* const base_pointer = device_memory.GetPointer<u8>(device_addr);
    if (IsRangeGranular(device_addr, size) ||
        base_pointer + size == device_memory.GetPointer<u8>(device_addr + size)) {
        return std::span(base_pointer, size);
    } else {
        const std::span<u8> span = ImmediateBuffer(size);
        device_memory.ReadBlockUnsafe(device_addr, span.data(), size);
        return span;
    }
}

template <class P>
std::span<u8> BufferCache<P>::ImmediateBuffer(size_t wanted_capacity) {
    immediate_buffer_alloc.resize_destructive(wanted_capacity);
    return std::span<u8>(immediate_buffer_alloc.data(), wanted_capacity);
}

template <class P>
bool BufferCache<P>::HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept {
    return ((channel_state->fast_bound_uniform_buffers[stage] >> binding_index) & 1u) != 0;
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectCount() {
    auto& buffer = slot_buffers[channel_state->count_buffer_binding.buffer_id];
    return std::make_pair(&buffer, buffer.Offset(channel_state->count_buffer_binding.device_addr));
}

template <class P>
std::pair<typename BufferCache<P>::Buffer*, u32> BufferCache<P>::GetDrawIndirectBuffer() {
    auto& buffer = slot_buffers[channel_state->indirect_buffer_binding.buffer_id];
    return std::make_pair(&buffer,
                          buffer.Offset(channel_state->indirect_buffer_binding.device_addr));
}

} // namespace VideoCommon
