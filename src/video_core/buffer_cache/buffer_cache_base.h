// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstdlib>
#include <array>
#include <bit>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/container/unordered_map.h"
#include <boost/container/static_vector.hpp>
#include <boost/container/small_vector.hpp>

#include "common/common_types.h"
#include "common/spin_recursive_mutex.h"
#include "common/div_ceil.h"
#include "common/literals.h"
#include "common/lru_cache.h"
#include "common/range_sets.h"
#include "common/scope_exit.h"
#include "common/settings.h"
#include "common/slot_vector.h"
#include "shader_recompiler/shader_info.h"
#include "video_core/buffer_cache/buffer_base.h"
#include "video_core/buffer_cache/read_trap_tracker.h"
#include "video_core/buffer_cache/virtual_range_cache.h"
#include "video_core/control/channel_state_cache.h"
#include "video_core/delayed_destruction_ring.h"
#include "video_core/dirty_flags.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2_download_trace.h"
#include "video_core/surface.h"
#include "video_core/texture_cache/types.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/draw_cache.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/va_map.h"

namespace VideoCommon {

using BufferId = Common::SlotId;

using VideoCore::Surface::PixelFormat;
using namespace Common::Literals;

#ifdef __APPLE__
constexpr u32 NUM_VERTEX_BUFFERS = 16;
#else
constexpr u32 NUM_VERTEX_BUFFERS = 32;
#endif
constexpr u32 NUM_TRANSFORM_FEEDBACK_BUFFERS = 4;
constexpr u32 NUM_GRAPHICS_UNIFORM_BUFFERS = 18;
constexpr u32 NUM_COMPUTE_UNIFORM_BUFFERS = 8;
// Native shaders can reference more than 16 storage buffers. Dropping a binding
// shifts all following descriptor payloads into the wrong descriptor types.
constexpr u32 NUM_STORAGE_BUFFERS = Shader::Info::MAX_SSBOS;
static_assert(NUM_STORAGE_BUFFERS <= 64, "storage binding masks must fit u64");
constexpr u32 NUM_TEXTURE_BUFFERS = 32;
constexpr u32 NUM_STAGES = 5;

static_assert(NUM_GRAPHICS_UNIFORM_BUFFERS <= 32, "fast bitmask must fit u32");

using UniformBufferSizes = std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>;
using ComputeUniformBufferSizes = std::array<u32, NUM_COMPUTE_UNIFORM_BUFFERS>;

enum class ObtainBufferSynchronize : u32 {
    NoSynchronize = 0,
    FullSynchronize = 1,
    SynchronizeNoDirty = 2,
};

enum class ObtainBufferOperation : u32 {
    DoNothing = 0,
    MarkAsWritten = 1,
    DiscardWrite = 2,
    MarkQuery = 3,
};

static constexpr BufferId NULL_BUFFER_ID{0};
static constexpr u32 DEFAULT_SKIP_CACHE_SIZE = static_cast<u32>(4_KiB);

struct Binding {
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    BufferId buffer_id;
    u32 segment_first{};
    u32 segment_count{};
};

struct MultiRangeSegment {
    BufferId buffer_id;
    DAddr device_addr{};
    u32 size{};
};

struct TextureBufferBinding : Binding {
    PixelFormat format;
};

static constexpr Binding NULL_BINDING{
    .device_addr = 0,
    .size = 0,
    .buffer_id = NULL_BUFFER_ID,
};

template <typename Buffer>
struct HostBindings {
    boost::container::static_vector<Buffer*, NUM_VERTEX_BUFFERS> buffers;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> offsets;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> sizes;
    boost::container::static_vector<u64, NUM_VERTEX_BUFFERS> strides;
    u32 min_index{NUM_VERTEX_BUFFERS};
    u32 max_index{0};
};

class BufferCacheChannelInfo : public ChannelInfo {
public:
    BufferCacheChannelInfo() = delete;
    BufferCacheChannelInfo(Tegra::Control::ChannelState& state) noexcept : ChannelInfo(state) {}
    BufferCacheChannelInfo(const BufferCacheChannelInfo& state) = delete;
    BufferCacheChannelInfo& operator=(const BufferCacheChannelInfo&) = delete;

    Binding index_buffer;
    std::array<Binding, NUM_VERTEX_BUFFERS> vertex_buffers;
    std::array<std::array<Binding, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES> uniform_buffers;
    std::array<std::array<Binding, NUM_STORAGE_BUFFERS>, NUM_STAGES> storage_buffers;
    std::array<std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS>, NUM_STAGES> texture_buffers;
    std::array<Binding, NUM_TRANSFORM_FEEDBACK_BUFFERS> transform_feedback_buffers;
    Binding count_buffer_binding;
    Binding indirect_buffer_binding;

    std::array<Binding, NUM_COMPUTE_UNIFORM_BUFFERS> compute_uniform_buffers;
    std::array<Binding, NUM_STORAGE_BUFFERS> compute_storage_buffers;
    std::array<TextureBufferBinding, NUM_TEXTURE_BUFFERS> compute_texture_buffers;

    std::array<u32, NUM_STAGES> enabled_uniform_buffer_masks{};
    u32 enabled_compute_uniform_buffer_mask = 0;

    const UniformBufferSizes* uniform_buffer_sizes{};
    const ComputeUniformBufferSizes* compute_uniform_buffer_sizes{};

    std::array<u64, NUM_STAGES> enabled_storage_buffers{};
    std::array<u64, NUM_STAGES> written_storage_buffers{};
    u64 enabled_compute_storage_buffers = 0;
    u64 written_compute_storage_buffers = 0;
    // Bound by a native (Switch 2) compute program, whose pointers carry no size.
    u64 native_compute_storage_buffers = 0;
    // Bytes from each compute storage binding's start that a dispatch marks GPU-written.
    std::array<u32, NUM_STORAGE_BUFFERS> compute_storage_write_marks{};

    u32 total_graphics_storage_buffers = 0;
    u32 total_compute_storage_buffers = 0;

    std::array<u32, NUM_STAGES> enabled_texture_buffers{};
    std::array<u32, NUM_STAGES> written_texture_buffers{};
    std::array<u32, NUM_STAGES> image_texture_buffers{};
    u32 enabled_compute_texture_buffers = 0;
    u32 written_compute_texture_buffers = 0;
    u32 image_compute_texture_buffers = 0;

    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_hits{};
    std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS> uniform_cache_shots{};

    u32 uniform_buffer_skip_cache_size = DEFAULT_SKIP_CACHE_SIZE;

    bool has_deleted_buffers = false;

    std::array<u32, NUM_STAGES> dirty_uniform_buffers{};
    std::array<u32, NUM_STAGES> fast_bound_uniform_buffers{};
    std::array<std::array<u32, NUM_GRAPHICS_UNIFORM_BUFFERS>, NUM_STAGES>
        uniform_buffer_binding_sizes{};
};

/// The staging buffer type a runtime's deferred downloads use (EDEN_NX2_V2_READBACK).
template <typename R>
struct ReadbackStaging {
    using type = int;
};
template <typename R>
    requires requires(R& runtime) { runtime.DownloadStagingBuffer(size_t{}, true); }
struct ReadbackStaging<R> {
    using type = decltype(std::declval<R&>().DownloadStagingBuffer(size_t{}, true));
};

template <class P>
class BufferCache : public VideoCommon::ChannelSetupCaches<BufferCacheChannelInfo> {
    friend struct BufferCacheTestAccess;
    // Page size for caching purposes.
    // This is unrelated to the CPU page size and it can be changed as it seems optimal.
#ifdef YUZU_LEGACY
    static constexpr u32 CACHING_PAGEBITS = 12;
#else
    static constexpr u32 CACHING_PAGEBITS = 16;
#endif
    static constexpr u64 CACHING_PAGESIZE = u64{1} << CACHING_PAGEBITS;

    static constexpr bool IS_OPENGL = P::IS_OPENGL;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS =
        P::HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT =
        P::HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = P::NEEDS_BIND_UNIFORM_INDEX;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = P::NEEDS_BIND_STORAGE_INDEX;
    static constexpr bool USE_MEMORY_MAPS = P::USE_MEMORY_MAPS;
    static constexpr bool SEPARATE_IMAGE_BUFFERS_BINDINGS = P::SEPARATE_IMAGE_BUFFER_BINDINGS;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = P::USE_MEMORY_MAPS_FOR_UPLOADS;

#ifdef YUZU_LEGACY
    static constexpr s64 TARGET_THRESHOLD = 3_GiB;
#else
    static constexpr s64 TARGET_THRESHOLD = 4_GiB;
#endif

    static constexpr s64 DEFAULT_EXPECTED_MEMORY = 512_MiB;
    static constexpr s64 DEFAULT_CRITICAL_MEMORY = 1_GiB;

    // Debug Flags.

    static constexpr bool DISABLE_DOWNLOADS = true;

    using Maxwell = Tegra::Engines::Maxwell3D::Regs;

    using Runtime = typename P::Runtime;
    using Buffer = typename P::Buffer;
    using Async_Buffer = typename P::Async_Buffer;
    using MemoryTracker = typename P::MemoryTracker;

    struct OverlapResult {
        boost::container::small_vector<BufferId, 16> ids;
        DAddr begin;
        DAddr end;
        bool has_stream_leap = false;
    };

public:
    explicit BufferCache(Tegra::MaxwellDeviceMemoryManager& device_memory_, Runtime& runtime_);

    ~BufferCache();

    void TickFrame();

    bool BindMultiRangeStorage(const Binding& binding, bool is_written,
                               std::span<const MultiRangeSegment> pool,
                               u64 write_mark_limit = ~u64{0});

    void ResolveMultiRangeStorage(Binding& binding, bool is_written,
                                  std::vector<MultiRangeSegment>& pool);

    void RefreshMultiRangeSegments(std::vector<MultiRangeSegment>& pool);

    void UnmapGPUMemory(size_t as_id, GPUVAddr gpu_addr, size_t size);

    /// `keep_page_gpu_data`: download GPU-written bytes sharing the written pages first (false
    /// when the memory is going away).
    void WriteMemory(DAddr device_addr, u64 size, bool keep_page_gpu_data = true);

    void CachedWriteMemory(DAddr device_addr, u64 size);

    /// Cancels pending downloads over the range, so a fence can't write them over newer bytes.
    void ClearDownload(DAddr base_addr, u64 size);

    /// CPU writes collected but not yet applied (SetPendingCpuWrites).
    [[nodiscard]] Common::RangeSet<DAddr> PendingCpuWriteRanges();

    /// Writes downloaded bytes to guest memory, except over `cpu_written`.
    void WriteDownloadedMemory(const Common::RangeSet<DAddr>& cpu_written, DAddr device_addr,
                               const u8* data, u64 size);

    bool OnCPUWrite(DAddr device_addr, u64 size);

    void DownloadMemory(DAddr device_addr, u64 size);

    /// Pipeline v2 (EDEN_NX2_V2_READBACK): records the download of the GPU-modified parts of
    /// the range into a staging buffer and submits it without waiting for the GPU. `land`
    /// (called on the reading thread) waits for that submission and writes guest memory. Returns
    /// false, recording nothing, where the old path must run (buffers read in place, nothing
    /// GPU-modified, no async runtime). `verify`: also download synchronously as before, write
    /// that, and have `land` compare the asynchronous copy with it.
    bool DownloadMemoryAsync(DAddr device_addr, u64 size, std::function<void()>& land,
                             bool verify);

    std::optional<VideoCore::RasterizerDownloadArea> GetFlushArea(DAddr device_addr, u64 size);

    /// The core consumed the read traps of these pages for a read the texture cache answered:
    /// the next GPU write here must arm them again (GetFlushArea forgets them otherwise).
    void ForgetReadTraps(DAddr device_addr, u64 size) {
        read_trapped_ranges.Subtract(
            Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE),
            Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE) -
                Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE));
    }

    /// Without downloads at every fence, a guest CPU read is the only way GPU-written buffer
    /// data reaches guest memory, so reads must ask the buffer cache (GetFlushArea).
    bool FlushesOnCpuRead() const {
        return !AsyncBufferDownloads();
    }

    bool InlineMemory(DAddr dest_address, size_t copy_size, std::span<const u8> inlined_buffer);

    void BindGraphicsUniformBuffer(size_t stage, u32 index, GPUVAddr gpu_addr, u32 size);

    void DisableGraphicsUniformBuffer(size_t stage, u32 index);

    void UpdateGraphicsBuffers(bool is_indexed);

    /// Pipeline v2 P5a (EDEN_NX2_V2_COMMIT=1): applies, in order, the bind mutations deferred
    /// since the last buffer update (uploads, geometry syncs, written marks, LRU touches, usage
    /// marks), before the draw or dispatch is encoded. No-op otherwise.
    void CommitBinds();
    [[nodiscard]] bool HasDeferredBinds() const noexcept {
        return v2_defer;
    }

    void UpdateComputeBuffers();

    void BindHostGeometryBuffers(bool is_indexed);

    void BindHostStageBuffers(size_t stage);

    void BindHostComputeBuffers();

    void SetUniformBuffersState(const std::array<u32, NUM_STAGES>& mask,
                                const UniformBufferSizes* sizes);

    void SetComputeUniformBufferState(u32 mask, const ComputeUniformBufferSizes* sizes);

    void UnbindGraphicsStorageBuffers(size_t stage);

    /// EDEN_NX2_V2_DRAWCACHE: binds a storage buffer as BindGraphicsStorageBuffer bound it for a
    /// draw recipe, without reading its descriptor (the recipe proved it unchanged).
    void ReplayGraphicsStorageBuffer(size_t stage, size_t ssbo_index, DAddr device_addr,
                                     GPUVAddr gpu_addr, u32 size, u32 segment_first,
                                     u32 segment_count, bool is_written);

    /// EDEN_NX2_V2_DRAWCACHE buffer replay. Records the graphics buffer bindings the last
    /// UpdateGraphicsBuffers/BindHost* resolved and bound (with `stamp`, taken before them).
    void RecordGraphicsBuffers(VideoCommon::V2::DrawCache::BufferRecipe& recipe, bool is_indexed,
                               u64 stamp);
    /// Replaces UpdateGraphicsBuffers + BindHostGeometryBuffers for a recipe replay. A binding
    /// whose buffer generation matches and whose guest range is unchanged since the recipe's stamp
    /// (and lies within the recorded one) is bound from the recipe: no FindBuffer, no
    /// SynchronizeBuffer. Any other binding takes the existing lookup and sync. False when the
    /// replay can't run (the caller takes the existing calls; nothing was bound).
    [[nodiscard]] bool ReplayGraphicsBuffers(const VideoCommon::V2::DrawCache::BufferRecipe& recipe,
                                             bool is_indexed,
                                             VideoCommon::V2::Versions::EpochMemo& memo);
    /// Replaces BindHostStageBuffers(stage) after ReplayGraphicsBuffers returned true: uniform
    /// versions of this tick are referenced and cached uniform, storage and texture buffers bound
    /// without SynchronizeBuffer when unchanged; anything else is bound as always.
    void ReplayHostStageBuffers(size_t stage,
                                const VideoCommon::V2::DrawCache::BufferRecipe& recipe,
                                VideoCommon::V2::Versions::EpochMemo& memo);
    /// Bindings ReplayGraphicsBuffers/ReplayHostStageBuffers bound from the recipe and through
    /// the sync path since the last call (stats; any synced one means the recipe is recorded again).
    [[nodiscard]] std::pair<u32, u32> TakeReplayCounts() noexcept {
        return {std::exchange(replay_binds_replayed, 0), std::exchange(replay_binds_synced, 0)};
    }
    /// Verify: recipe bindings the versions vouch for whose ranges the tracker still holds
    /// CPU-modified (a skipped upload would have been needed).
    [[nodiscard]] u32 CountStaleRecipeRanges(const VideoCommon::V2::DrawCache::BufferRecipe& recipe,
                                             VideoCommon::V2::Versions::EpochMemo& memo);
    /// F2-X1: the path BindHostGraphicsUniformBuffer takes for this uniform binding now, as a
    /// recipe kind: Normal (root-table bank, native bank, no host buffer), Version (the stream
    /// path; a recipe holds Version only when a version was found) or Cached (the host buffer).
    /// A replay takes a recorded uniform only on the path the binding takes now.
    [[nodiscard]] VideoCommon::V2::DrawCache::BufferBind::Kind UniformBindKind(
        size_t stage, u32 index, const Binding& binding);

    /// EDEN_NX2_V2_DRAWCACHE: visits the graphics storage buffer bindings of `stage` as
    /// func(index, is_written, binding).
    template <typename Func>
    void ForEachGraphicsStorageBinding(size_t stage, Func&& func) const {
        ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
            const bool is_written =
                ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0;
            func(index, is_written, channel_state->storage_buffers[stage][index]);
        });
    }

    bool BindGraphicsStorageBuffer(size_t stage, size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                   bool is_written, u32 indirect_offset = ~0U);

    void UnbindGraphicsTextureBuffers(size_t stage);

    void BindGraphicsTextureBuffer(size_t stage, size_t tbo_index, GPUVAddr gpu_addr, u32 size,
                                   PixelFormat format, bool is_written, bool is_image);

    void UnbindComputeStorageBuffers();

    /// write_extent: how far the program's stores reach past the pointer (native programs).
    void BindComputeStorageBuffer(size_t ssbo_index, u32 cbuf_index, u32 cbuf_offset,
                                  bool is_written, u32 indirect_offset = ~0U,
                                  const Shader::StorageWriteExtent& write_extent = {});

    void UnbindComputeTextureBuffers();

    void BindComputeTextureBuffer(size_t tbo_index, GPUVAddr gpu_addr, u32 size, PixelFormat format,
                                  bool is_written, bool is_image);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainBuffer(GPUVAddr gpu_addr, u32 size,
                                                       ObtainBufferSynchronize sync_info,
                                                       ObtainBufferOperation post_op);

    [[nodiscard]] std::pair<Buffer*, u32> ObtainCPUBuffer(DAddr gpu_addr, u32 size,
                                                          ObtainBufferSynchronize sync_info,
                                                          ObtainBufferOperation post_op);
    void FlushCachedWrites();

    /// Where the CPU writes into GPU-written memory that were collected but not applied yet
    /// (WriteMemory) come from: downloads must not write the GPU's older copy over them.
    using PendingCpuWrites = std::function<void(const std::function<void(DAddr, size_t)>&)>;
    void SetPendingCpuWrites(PendingCpuWrites source) {
        pending_cpu_writes = std::move(source);
    }

    /// Takes CPU writes about to be applied one by one out of the GPU-written ranges first, so
    /// applying one doesn't download the GPU's copy over another (WriteMemory keeps the GPU data
    /// on the pages it writes).
    void ClaimCpuWrites(std::span<const std::pair<DAddr, u64>> ranges);

    /// Return true when there are uncommitted buffers to be downloaded
    [[nodiscard]] bool HasUncommittedFlushes() const noexcept;

    /// Whether the fence downloads queued are all learned early downloads (EDEN_NX2_V2_READBACK):
    /// Switch 2 titles without async buffer downloads mark pages Preflushable only by learning.
    [[nodiscard]] bool FlushesAreEarlyDownloads() const;

    void AccumulateFlushes();

    /// Return true when the caller should wait for async downloads
    [[nodiscard]] bool ShouldWaitAsyncFlushes() const noexcept;

    /// Commit asynchronous downloads
    void CommitAsyncFlushes();
    void CommitAsyncFlushesHigh();

    /// Pop asynchronous downloads
    void PopAsyncFlushes();
    void PopAsyncBuffers();

    bool DMACopy(GPUVAddr src_address, GPUVAddr dest_address, u64 amount);

    bool DMAClear(GPUVAddr src_address, u64 amount, u32 value);

    /// Switch 2 GPU-written command lists (nx2.h, g_nx2_command_list_pages). While collecting (a
    /// compute dispatch or a DMA copy being recorded), writes to command list pages are noted;
    /// Nx2CopyListPages then records copies of those pages into host memory after the work.
    void Nx2CollectListCopies(bool collect) {
        nx2_collect_list_copies = collect;
        if (!collect) {
            nx2_list_copy_requests.clear();
        }
    }
    /// True when it recorded copies, which the GPU completes at `tick` (the caller submits).
    bool Nx2CopyListPages(u64 tick);
    enum class Nx2ListRead { Unknown, Copied, Synced, NeedsSync, Pending };
    /// Front thread: Synced when guest memory holds the page `device_addr` lies in (flushed and
    /// written by no GPU work since), else Copied with `out` filled from the copy of its last
    /// write once the GPU is done with it. Unknown otherwise: the caller drains and flushes.
    /// Pages draws write are only Synced or Unknown, and NeedsSync until `past_sync`: the
    /// split's back ran the work before the last host-level wait (GpuSplit::WaitForSync).
    /// With `pending_tick`, a copy the GPU has not finished returns Pending and sets the tick
    /// instead of waiting for it.
    Nx2ListRead Nx2ReadListCopy(DAddr device_addr, std::span<u8> out, bool past_sync,
                                u64* pending_tick = nullptr);
    /// Front thread, the split drained: the command list page was flushed to guest memory.
    void Nx2NoteListSynced(DAddr page);

    void Nx2FlushListPages(std::span<const DAddr> pages);
    /// CPU and inline writes: they reach guest memory, so a synced page stays synced, but its copy
    /// no longer holds the page.
    void Nx2NoteListHostWrite(DAddr device_addr, u64 size);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(DAddr addr, size_t size);

    /// Visits the uniform and storage buffers bound for the last draw or dispatch with the cached
    /// host buffer holding each range, if one does, for diagnostics. Nothing is
    /// created or synchronized. Stage 5 is compute.
    template <typename Func>
    void ForEachBoundBuffer(bool compute, Func&& func) {
        const auto visit = [&](u32 stage, bool storage, u32 index, bool written,
                               const Binding& binding, u32 size) {
            if (binding.device_addr == 0 || size == 0) {
                return;
            }
            const Buffer* host = nullptr;
            u32 offset = 0;
            if (const BufferId id = page_table[binding.device_addr >> CACHING_PAGEBITS]) {
                const Buffer& buffer = slot_buffers[id];
                if (buffer.IsInBounds(binding.device_addr, size)) {
                    host = &buffer;
                    offset = buffer.Offset(binding.device_addr);
                }
            }
            func(stage, storage, index, written, binding, size, host, offset);
        };
        if (compute) {
            const auto* const sizes = channel_state->compute_uniform_buffer_sizes;
            ForEachEnabledBit(channel_state->enabled_compute_uniform_buffer_mask, [&](u32 index) {
                const Binding& binding = channel_state->compute_uniform_buffers[index];
                visit(5, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_compute_storage_buffers, [&](u32 index) {
                const Binding& binding = channel_state->compute_storage_buffers[index];
                visit(5, true, index, ((channel_state->written_compute_storage_buffers >> index) & 1) != 0,
                      binding, binding.size);
            });
            return;
        }
        const auto* const sizes = channel_state->uniform_buffer_sizes;
        for (u32 stage = 0; stage < NUM_STAGES; ++stage) {
            ForEachEnabledBit(channel_state->enabled_uniform_buffer_masks[stage], [&](u32 index) {
                const Binding& binding = channel_state->uniform_buffers[stage][index];
                visit(stage, false, index, false, binding,
                      sizes ? std::min(binding.size, (*sizes)[stage][index]) : binding.size);
            });
            ForEachEnabledBit(channel_state->enabled_storage_buffers[stage], [&](u32 index) {
                const Binding& binding = channel_state->storage_buffers[stage][index];
                visit(stage, true, index,
                      ((channel_state->written_storage_buffers[stage] >> index) & 1) != 0, binding,
                      binding.size);
            });
        }
    }

    /// Return true when a region is registered on the cache
    [[nodiscard]] bool IsRegionRegistered(DAddr addr, size_t size);

    /// Return true when a CPU region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(DAddr addr, size_t size);

    void SetDrawIndirect(
        const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect_) {
        current_draw_indirect = current_draw_indirect_;
    }

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectCount();

    [[nodiscard]] std::pair<Buffer*, u32> GetDrawIndirectBuffer();

    /// Fast draw verify (nx2_fast_draw.h): a hash of the bindings a delta draw keeps.
    [[nodiscard]] u64 KeptGraphicsBindingsHash(bool vertex, bool storage, bool texture) const;

    template <typename Func>
    void BufferOperations(Func&& func) {
        do {
            channel_state->has_deleted_buffers = false;
            func();
        } while (channel_state->has_deleted_buffers);
    }

    Common::SpinRecursiveMutex mutex;

    /// Pipeline v2 P5a (EDEN_NX2_V2_COMMIT): the renderer's commit state (null: old path). Builds
    /// the lock-free page map from the page table.
    void SetCommit(V2::Commit::Context* context);

    /// P5a pure lookup: the registered buffer covering [device_addr, device_addr + size) from
    /// the lock-free page map (NULL_BUFFER_ID when none does; any thread reads it, P6 shards
    /// validate the slot by its generation).
    [[nodiscard]] BufferId LookupBuffer(DAddr device_addr, u32 size) const;

    // Pipeline v2 P6 Step 0 (EDEN_NX2_V2_DRAWREC, notes/STEP0_SPEC.md §2 items 11-12): pure
    // twins of the fused update calls, and the appliers ApplyDraw/EncodeDraw run.
    /// FindBuffer's owner-side lookup without creating: the buffer holding the range,
    /// NULL_BUFFER_ID for no address, an invalid id when a creation is due.
    [[nodiscard]] BufferId PeekBuffer(DAddr device_addr, u32 size, bool sparse_compatible) const;
    /// The BufferCreate request FindBuffer queues for a range (with its predicted effects).
    [[nodiscard]] V2::Commit::Request MakeCreateRequest(DAddr device_addr, u32 size,
                                                        bool sparse_compatible) const;
    /// An index or vertex buffer range as UpdateIndexBuffer / UpdateVertexBuffer compute it.
    struct GeometryRange {
        DAddr device_addr{};
        u32 size{};
        bool valid{}; ///< false: NULL_BINDING
    };
    template <class Memory>
    [[nodiscard]] static GeometryRange IndexRange(Memory& memory, GPUVAddr begin, GPUVAddr end,
                                                  u32 first, u32 count, u32 format_size);
    template <class Memory>
    [[nodiscard]] static GeometryRange VertexRange(Memory& memory, GPUVAddr begin, bool enabled,
                                                   GPUVAddr limit_address, bool native,
                                                   u64 native_size);
    template <class View>
    [[nodiscard]] V2::Resolve::ResolvedIndex ResolveIndexBinding(const View& view) const;
    template <class View>
    [[nodiscard]] std::optional<V2::Resolve::ResolvedVertex> ResolveVertexBinding(
        const View& view, u32 slot) const;
    template <class View>
    [[nodiscard]] V2::Resolve::ResolvedUbo ResolveUniformBinding(const View& view, size_t stage,
                                                                 u32 slot) const;
    /// BindGraphicsStorageBuffer + UpdateStorageBuffers without writing anything.
    /// `multi_range`: the binding may need split segments (the record path can't take it).
    template <class View>
    [[nodiscard]] V2::Resolve::ResolvedSsbo ResolveStorageBinding(
        const View& view, size_t stage, u32 ssbo_index, u32 cbuf_index, u32 cbuf_offset,
        bool is_written, u32 indirect_offset, bool& multi_range) const;
    /// A texture buffer binding of a ready view (BindGraphicsTextureBuffer + UpdateTextureBuffers).
    [[nodiscard]] V2::Resolve::ResolvedTbo ResolveTextureBufferBinding(size_t stage, u32 index,
                                                                       GPUVAddr gpu_addr, u32 size,
                                                                       PixelFormat format,
                                                                       bool is_written,
                                                                       bool is_image) const;
    /// Applies one request of a draw record's buffer list (BufferCreate as V2ApplyCreates does).
    u32 ApplyDrawRequest(const V2::Commit::Request& request);
    /// Whether a buffer was deleted since the last call (and clears it).
    [[nodiscard]] bool TakeDeletedBuffers() noexcept {
        return std::exchange(channel_state->has_deleted_buffers, false);
    }
    /// Opens a deferred bind scope as DoUpdateGraphicsBuffers leaves one (with COMMIT=1).
    void BeginBindScope() {
        if (V2Defers()) {
            V2BeginBindScope();
        }
    }
    /// Writes the graphics bindings of a record into the channel state as the Update* calls
    /// leave them (in BufferOperations, as DoUpdateGraphicsBuffers runs).
    void EncodeGraphicsBindings(const V2::Resolve::ResolvedDraw& record);
    /// DRAWREC=verify (step 11): the graphics bindings the old configure left in the channel
    /// state, in a record's shape (index, vertex, uniform, storage and texture buffers of
    /// `stages`), into `observed`.
    void ObserveGraphicsBindings(V2::Resolve::ResolvedDraw& observed, bool is_indexed,
                                 const std::array<bool, NUM_STAGES>& stages) const;
    [[nodiscard]] u32 BufferGeneration(BufferId id) const noexcept {
        return V2::Versions::Gens().Get(V2::Versions::Kind::Buffer, id.index);
    }
    [[nodiscard]] u64 BufferDeletions() const noexcept {
        return buffer_deletions;
    }
    /// DRAWREC=verify: buffers created so far (a join during a draw creates one and deletes
    /// others, so the live count need not grow).
    [[nodiscard]] size_t BufferSlots() const noexcept {
        return buffers_created;
    }
    /// DRAWREC=verify (step 10): a hash of what a cache mutation moves; a pure resolve leaves
    /// it unchanged.
    [[nodiscard]] u64 DrawRecPurity() const noexcept {
        u64 hash = 0xcbf29ce484222325ULL;
        const auto mix = [&](u64 value) { hash = (hash ^ value) * 0x100000001b3ULL; };
        mix(slot_buffers.size());
        mix(vertex_buffers_serial);
        mix(buffer_deletions);
        mix(enabled_vertex_buffers_mask);
        mix(v2_binds.Requests().size());
        mix(v2_creates.Requests().size());
        mix(v2_defer ? 1 : 0);
        mix(channel_state->has_deleted_buffers ? 1 : 0);
        return hash;
    }

    Runtime& runtime;
    // EDEN_NX2_V2_DRAWCACHE buffer replay (ReplayGraphicsBuffers): bindings bound from the recipe
    // or synced, and the vertex buffers that skip their synchronization in this replay.
    u32 replay_binds_replayed{};
    u32 replay_binds_synced{};
    u32 replay_skip_vertex_mask{};
    bool replay_skip_index{};
    bool replay_all_unchanged{}; ///< F2-07: this replay's recipe ranges are all unchanged
    bool replay_all_alive{};     ///< F2-07: and its recorded buffers all alive
    bool any_buffer_uploaded = false;
    /// Indirectly read banks of native programs already uploaded this frame.
    std::unordered_set<DAddr> native_indirect_uniforms_this_frame;

private:
    template <typename Mask, typename Func>
    static void ForEachEnabledBit(Mask enabled_mask, Func&& func) {
        for (u32 index = 0; enabled_mask != 0; ++index, enabled_mask >>= 1) {
            const int disabled_bits = std::countr_zero(enabled_mask);
            index += disabled_bits;
            enabled_mask >>= disabled_bits;
            func(index);
        }
    }

    template <typename Func>
    void ForEachBufferInRange(DAddr device_addr, u64 size, Func&& func) {
        const u64 page_end = Common::DivCeil(device_addr + size, CACHING_PAGESIZE);
        for (u64 page = device_addr >> CACHING_PAGEBITS; page < page_end;) {
            const BufferId buffer_id = page_table[page];
            if (!buffer_id) {
                ++page;
                continue;
            }
            Buffer& buffer = slot_buffers[buffer_id];
            func(buffer_id, buffer);

            const DAddr end_addr = buffer.CpuAddr() + buffer.SizeBytes();
            page = Common::DivCeil(end_addr, CACHING_PAGESIZE);
        }
    }

    static bool IsRangeGranular(DAddr device_addr, size_t size) {
        return (device_addr & ~Core::DEVICE_PAGEMASK) ==
               ((device_addr + size) & ~Core::DEVICE_PAGEMASK);
    }

    void RunGarbageCollector(bool aggressive_gc);

    void BindHostIndexBuffer();

    void BindHostVertexBuffers();

    void BindHostDrawIndirectBuffers();

    void BindHostGraphicsUniformBuffers(size_t stage);

    void BindHostGraphicsUniformBuffer(size_t stage, u32 index, u32 binding_index, bool needs_bind);

    /// EDEN_NX2_V2_CBUFVER: binds a small constant buffer through its version (v2/cbuf_version.h).
    /// True when an unchanged version was referenced; false when it bound a new copy.
    bool BindUniformVersion(size_t stage, u32 index, u32 binding_index, DAddr device_addr,
                            u32 size);
    /// EDEN_NX2_V2_CBUFRING: binds the decode-time version of the range (v2/cbuf_ring.h) while
    /// its bytes equal guest memory. False when there is none or it differs.
    bool BindRingVersion(size_t stage, u32 index, DAddr device_addr, u32 size);
    /// EDEN_NX2_V2_CBUFVER: what guest memory holds at [device_addr, device_addr + size) now
    /// (a direct span, or a copy in version_scratch).
    const u8* V2GuestBytes(DAddr device_addr, u32 size);

    void BindHostGraphicsStorageBuffers(size_t stage);

    void BindHostGraphicsTextureBuffers(size_t stage);

    void BindHostTransformFeedbackBuffers();

    void BindHostVertexBuffer(u32 index, Buffer& buffer, u32 offset, u32 size, u32 stride);

    void BindHostComputeUniformBuffers();

    void BindHostComputeStorageBuffers();

    void BindHostComputeTextureBuffers();

    void DoUpdateGraphicsBuffers(bool is_indexed);

    void DoUpdateComputeBuffers();

    void UpdateIndexBuffer();

    void UpdateVertexBuffers();

    void UpdateVertexBuffer(u32 index);

    void UpdateDrawIndirect();

    void UpdateUniformBuffers(size_t stage);

    void UpdateStorageBuffers(size_t stage);

    void UpdateTextureBuffers(size_t stage);

    void UpdateTransformFeedbackBuffers();

    void UpdateTransformFeedbackBuffer(u32 index);

    void UpdateComputeUniformBuffers();

    void UpdateComputeStorageBuffers();

    void UpdateComputeTextureBuffers();

    void MarkWrittenBuffer(BufferId buffer_id, DAddr device_addr, u32 size, bool needs_sync = false);

    /// EDEN_NX2_V2_READBACK: frees the staging of readbacks their readers landed.
    void FreeLandedReadbacks();
    /// EDEN_NX2_V2_READBACK early downloads (Switch 2): a guest read of [begin, end); pages read
    /// back twice become Preflushable, so the fences download their later GPU writes.
    void LearnReadback(DAddr begin, DAddr end);

public:
    /// EDEN_NX2_V2_XREADBACK: records the queued fence downloads right behind their writers.
    void RecordWriterDownloads();
    bool v2_writer_downloads = false;

private:
    struct PreparedDownload {
        Async_Buffer staging{};
        boost::container::small_vector<BufferCopy, 4> copies;
        Common::RangeSet<DAddr> ranges;
    };
    std::vector<PreparedDownload> writer_downloads_open;
    std::deque<std::vector<PreparedDownload>> writer_downloads_fenced;
    /// ...and GPU writes to them: four writes nobody read in between drop a page again.
    void NoteLearnedWrite(DAddr device_addr, u64 size);

    struct ReadbackState;
    std::vector<std::shared_ptr<ReadbackState>> readbacks_in_flight;
    struct LearnedPage {
        u8 reads{};
        u8 unread_writes{}; ///< Frames with GPU writes and no read since
        bool marked{};
        u64 write_frame{};  ///< learned_frame of the last write counted
    };
    u64 learned_frame{1}; ///< +1 per TickFrame
    std::unordered_map<u64, LearnedPage> learned_pages;
    size_t learned_marked{};
    DAddr learned_lowest{~DAddr{0}};
    DAddr learned_highest{};

    /// Counts a GPU write to the command list pages in the range (invalidating their copies) and,
    /// while collecting, asks for new copies.
    void Nx2NoteListWrite(BufferId buffer_id, DAddr device_addr, u64 size);

    [[nodiscard]] BufferId FindBuffer(DAddr device_addr, u32 size, bool sparse_compatible);

    void SynchronizeBufferWrites(Buffer& buffer);

    [[nodiscard]] OverlapResult ResolveOverlaps(DAddr device_addr, u32 wanted_size);

    void JoinOverlap(BufferId new_buffer_id, BufferId overlap_id, bool accumulate_stream_score);

    [[nodiscard]] BufferId CreateBuffer(DAddr device_addr, u32 wanted_size,
                                        bool sparse_compatible);

    void Register(BufferId buffer_id);

    void Unregister(BufferId buffer_id);

    template <bool insert>
    void ChangeRegister(BufferId buffer_id);

    void TouchBuffer(Buffer& buffer, BufferId buffer_id) noexcept;

    bool SynchronizeBuffer(Buffer& buffer, DAddr device_addr, u32 size);
    bool SynchronizeBufferImpl(Buffer& buffer, DAddr device_addr, u32 size);

    /// SynchronizeBuffer for vertex buffer slots 0-31 and the index buffer (slot 32).
    void SynchronizeGeometryBuffer(u32 slot, BufferId buffer_id, DAddr device_addr, u32 size);

    void UploadMemory(Buffer& buffer, u64 total_size_bytes, u64 largest_copy,
                      std::span<BufferCopy> copies);

    void ImmediateUploadMemory(Buffer& buffer, u64 largest_copy,
                               std::span<const BufferCopy> copies);

    void MappedUploadMemory(Buffer& buffer, u64 total_size_bytes, std::span<BufferCopy> copies);

    void DownloadBufferMemory(Buffer& buffer_id);

    void DownloadBufferMemory(Buffer& buffer_id, DAddr device_addr, u64 size);

    /// Downloads the GPU-written data of several buffers with one wait for the GPU.
    void DownloadBuffersMemory(std::span<const BufferId> buffer_ids);

    void DownloadRangesMemory(std::span<const std::pair<DAddr, u64>> ranges);

    void DeleteBuffer(BufferId buffer_id, bool do_not_mark = false);

    [[nodiscard]] Binding StorageBufferBinding(GPUVAddr ssbo_addr, u32 cbuf_index, bool is_written,
                                               bool native_pointer = false,
                                               bool table_pointer = false) const;

    void TraceTablePointer(bool compute, size_t stage, size_t ssbo_index, u32 cbuf_index,
                           u32 cbuf_offset, u32 indirect_offset, GPUVAddr holder,
                           const Binding& binding);

    [[nodiscard]] TextureBufferBinding GetTextureBufferBinding(GPUVAddr gpu_addr, u32 size,
                                                               PixelFormat format);

    [[nodiscard]] std::span<const u8> ImmediateBufferWithData(DAddr device_addr, size_t size);

    [[nodiscard]] std::span<u8> ImmediateBuffer(size_t wanted_capacity);

    [[nodiscard]] bool HasFastUniformBufferBound(size_t stage, u32 binding_index) const noexcept;

    [[nodiscard]] Binding& VertexBufferSlot(u32 index);

    [[nodiscard]] const Binding& VertexBufferSlot(u32 index) const;

    void UpdateVertexBufferSlot(u32 index, const Binding& binding);

    /// Flags the vertex buffers overlapping a written range so host copies derived from them are
    /// rebuilt.
    void MarkVertexBuffersStale(DAddr device_addr, u64 size);

    /// Re-dirties repacked vertex buffers whose source data was written since they were bound.
    void RefreshRepackedVertexBuffers();

    void InlineMemoryImplementation(DAddr dest_address, size_t copy_size,
                                    std::span<const u8> inlined_buffer);

    Tegra::MaxwellDeviceMemoryManager& device_memory;

    Common::SlotVector<Buffer> slot_buffers;
#ifdef YUZU_LEGACY
    static constexpr size_t TICKS_TO_DESTROY = 6;
#else
    static constexpr size_t TICKS_TO_DESTROY = 8;
#endif
    DelayedDestructionRing<Buffer, TICKS_TO_DESTROY> delayed_destruction_ring;

    const Tegra::Engines::Maxwell3D::DrawManager::IndirectParams* current_draw_indirect{};

    u32 last_index_count = 0;

    u32 enabled_vertex_buffers_mask = 0;
    u32 stale_vertex_buffers_mask = 0;
    u64 vertex_buffers_serial = 0;
    std::array<Binding, 32> v_buffer{};

    /// The range each geometry slot last scanned for uploads (SynchronizeGeometryBuffer).
    struct GeometrySync {
        BufferId buffer_id{};
        DAddr device_addr{};
        u32 size{};
        u64 deletions{};
    };
    std::array<GeometrySync, NUM_VERTEX_BUFFERS + 1> geometry_syncs{};
    /// Counts DeleteBuffer calls: a buffer id older than this count may have been freed or reused.
    u64 buffer_deletions = 0;
    u64 buffers_created = 0; ///< BufferSlots
    struct GeometryStats {
        u64 syncs{};
        u64 syncs_skipped{};
        u64 verify_mismatches{};
    } geometry_stats;

    boost::container::small_vector<BufferCopy, 4> upload_copies;

    MemoryTracker memory_tracker;
    Common::RangeSet<DAddr> uncommitted_gpu_modified_ranges;

    /// Whether GPU-written buffer ranges are copied back to guest memory at the next fence;
    /// otherwise only when something flushes them (as upstream yuzu below High GPU accuracy).
    /// Not for Switch 2 titles: title 1 had 250-330 MB/s copied back (a fence per compute
    /// dispatch, the fencing thread holding the buffer cache while it wrote them), while its CPU
    /// read GPU-written memory at most 64 times in 200 s (counted with reactive flushing) and
    /// the native MME's DMA reads never touched those pages. Without them a fence costs ~1 us
    /// instead of ~5 and draw setup ~7% less. EDEN_NX2_BUFFER_DOWNLOADS=0/1 forces it.
    bool AsyncBufferDownloads() const {
        static const int forced = [] {
            const char* value = std::getenv("EDEN_NX2_BUFFER_DOWNLOADS");
            return value == nullptr ? -1 : (value[0] == '0' ? 0 : 1);
        }();
        if (forced >= 0) {
            return forced == 1;
        }
        return gpu_memory == nullptr || !gpu_memory->UsesNativeTiling();
    }

    /// Whether a GPU write to this range is copied back at the next fence. Ranges the guest CPU
    /// has read (Preflushable, set by the first reactive flush) always are: later reads skip the
    /// flush and rely on these downloads, e.g. collision meshes skinned by compute shaders.
    bool DownloadsAtFence(DAddr device_addr, u64 size) noexcept {
        return AsyncBufferDownloads() || memory_tracker.IsRegionPreflushable(device_addr, size);
    }

    /// Queues a GPU write for download at the next fence where DownloadsAtFence holds. With only
    /// Preflushable marks, just the marked pages are queued: a write that covers one marked page
    /// of a large buffer would otherwise copy the whole buffer back at every fence.
    void QueueFenceDownload(DAddr device_addr, u64 size) {
        static const bool exact = [] {
            const char* value = std::getenv("EDEN_NX2_EXACT_PREFLUSH");
            return value == nullptr || value[0] != '0';
        }();
        if (AsyncBufferDownloads()) {
            uncommitted_gpu_modified_ranges.Add(device_addr, size);
            return;
        }
        if (!exact) {
            if (memory_tracker.IsRegionPreflushable(device_addr, size)) {
                uncommitted_gpu_modified_ranges.Add(device_addr, size);
            } else {
                ArmReadTraps(device_addr, size);
            }
            return;
        }
        const DAddr end = device_addr + size;
        u64 queued = 0;
        memory_tracker.ForEachPreflushableRange(device_addr, size, [&](DAddr begin, u64 length) {
            const DAddr lower = (std::max)(begin, device_addr);
            const DAddr upper = (std::min)(begin + length, end);
            if (lower < upper) {
                uncommitted_gpu_modified_ranges.Add(lower, upper - lower);
                queued += upper - lower;
            }
        });
        if (queued < size) {
            ArmReadTraps(device_addr, size);
        }
        if (Nx2DownloadTrace::g_enabled && queued != 0) [[unlikely]] {
            Nx2DownloadTrace::Queued(device_addr, size, queued);
        }
    }

    /// GPU writes that are not downloaded at fences: trap the guest CPU's next read of their
    /// pages, which flushes them on demand (or marks them Preflushable with preflushing enabled).
    void ArmReadTraps(DAddr device_addr, u64 size) {
        const DAddr begin = Common::AlignDown(device_addr, Core::DEVICE_PAGESIZE);
        const DAddr end = Common::AlignUp(device_addr + size, Core::DEVICE_PAGESIZE);
        // Only the parts not armed yet: the core visits each page it is given, and a buffer the
        // GPU rewrites every frame is mostly still armed (the guest read a few of its pages).
        read_trapped_ranges.Arm(begin, end - begin, [this](DAddr part, u64 part_size) {
            device_memory.ArmReadTraps(part, part_size);
        });
    }

    ReadTrapTracker read_trapped_ranges;
    /// Ranges the guest CPU read after GPU writes. A read that must flush downloads the GPU
    /// writes in all of them, so the reads that usually follow it find nothing left to flush.
    Common::RangeSet<DAddr> cpu_read_ranges;
    bool download_cpu_read_ranges{};
    Common::RangeSet<DAddr> gpu_modified_ranges;
    std::deque<Common::RangeSet<DAddr>> committed_gpu_modified_ranges;

    // Async Buffers
    std::deque<std::optional<Async_Buffer>> async_buffers;
    std::deque<boost::container::small_vector<BufferCopy, 4>> pending_downloads;
    /// For each entry of pending_downloads: the ranges it still writes to guest memory when its
    /// fence pops. A write to a range after a download was taken removes it from that download
    /// (and every other pending one): its bytes are older than the write.
    std::deque<Common::RangeSet<DAddr>> pending_download_ranges;
    PendingCpuWrites pending_cpu_writes;
    std::optional<Async_Buffer> current_buffer;

    std::deque<Async_Buffer> async_buffers_death_ring;

    size_t immediate_buffer_capacity = 0;
    Common::ScratchBuffer<u8> immediate_buffer_alloc;
    std::vector<u8> guest_uint8_indices;
    /// Guest memory large uploads came from since the last cold hint (device address, size).
    std::vector<std::pair<DAddr, u64>> uploaded_buffer_ranges;

    struct LRUItemParams {
        using ObjectType = BufferId;
        using TickType = u64;
    };
    Common::LeastRecentlyUsedCache<LRUItemParams> lru_cache;
    VirtualRangeCache virtual_ranges;
    std::vector<MultiRangeSegment> graphics_segments;
    std::vector<MultiRangeSegment> compute_segments;
    u64 frame_tick = 0;
    /// EDEN_NX2_TABLE_POINTER_TRACE: last table entry and size bound per table pointer slot.
    std::unordered_map<u64, std::pair<GPUVAddr, u32>> traced_table_pointers;
    u64 total_used_memory = 0;
    u64 buffer_bytes = 0; ///< The buffers' own size (total_used_memory may be the device's)
    u64 minimum_memory = 0;
    u64 critical_memory = 0;
    BufferId inline_buffer_id;
#ifdef YUZU_LEGACY
    bool immediately_free = false;
#endif

    // Covers the device address space (MaxwellDeviceTraits::device_virtual_bits).
    std::array<BufferId, ((1ULL << 36) >> CACHING_PAGEBITS)> page_table;
    /// P5a: page_table published for lock-free readers (written with it by the commit owner).
    V2::Commit::Context* commit = nullptr;
    std::unique_ptr<V2::VaMap<CACHING_PAGEBITS, 36>> v2_pages;
    /// P5a buffer requests. Lookups of the update pass return proxy ids for missing buffers
    /// (BufferCreate requests in v2_creates), created and patched into the bindings at its end.
    /// Bind mutations after it are requests in v2_binds, applied by CommitBinds.
    V2::Commit::RequestList v2_creates;
    V2::Commit::RequestList v2_binds;
    std::vector<u32> v2_scope_uploads; ///< F2-19: buffers with an upload queued in the scope
    /// Perf fix C: the scope's pending written marks, so a GPU-modified test scans only them.
    boost::container::small_vector<std::pair<DAddr, u64>, 8> v2_written_marks;
    u64 v2_segment = 1;
    bool v2_proxy_finds = false;
    u64 v2_upload_count = 0; ///< uploads applied (verify)
    bool v2_defer = false;

    /// Whether COMMIT=1 (not verify) defers buffer mutations.
    [[nodiscard]] bool V2Defers() const noexcept {
        return commit != nullptr && !commit->verify;
    }
    void V2BeginBindScope();
    void V2ApplyCreates();
    void V2PatchBindings(u32 proxy, BufferId id);
    /// Defers a bind mutation on `buffer` (by device address); false when it can't be (apply now).
    bool V2DeferBind(V2::Commit::Kind kind, Buffer& buffer, DAddr device_addr, u64 size, u32 arg);
    /// The buffer serving `device_addr` now (the request's own while it lives and covers it).
    BufferId V2CurrentBuffer(BufferId id, DAddr device_addr, u64 size);
    /// A GPU-modified test that also sees the written marks deferred in this bind scope.
    [[nodiscard]] bool V2PendingGpuWrite(DAddr device_addr, u64 size) const;
    void V2NoteUsage(Buffer& buffer, u32 offset, u64 size);
    Common::ScratchBuffer<u8> tmp_buffer;
    std::vector<u8> verify_scratch; ///< EDEN_NX2_V2_CBUFVER=verify
    std::vector<u8> version_scratch; ///< EDEN_NX2_V2_CBUFVER: guest bytes of a bind

    /// A command list page (Nx2CopyListPages): the GPU writes noted to it and the last copy.
    using Nx2Staging = decltype(std::declval<Runtime&>().DownloadStagingBuffer(size_t{}, true));
    struct Nx2ListPage {
        u64 writes{};      ///< GPU writes noted to the page
        u64 copy_writes{}; ///< `writes` when the copy was made: the copy is current while equal
        u64 copy_tick{};   ///< GPU tick that completes the copy
        /// `writes` when the front last flushed the page: guest memory holds it while equal
        u64 synced_writes{~u64{0}};
        u32 copy_begin{};  ///< Bytes of the page the copy holds (a buffer can end inside it)
        u32 copy_end{};
        /// Written outside a compute dispatch or a DMA copy (a draw's storage writes) at least
        /// once: the front can't tell from what it sent which draws write the page, so it reads
        /// no copy. A draw's writes reach a pushbuffer fetch only after a host-level wait, so
        /// the page's lists read guest memory once the back ran the work before the last such
        /// wait and no write came since the front flushed the page (Nx2ReadListCopy).
        bool draw_written{};
        std::optional<Nx2Staging> copy;
    };
    std::mutex nx2_list_mutex; ///< nx2_list_pages: the split's front reads the copies
    std::unordered_map<DAddr, Nx2ListPage> nx2_list_pages;
    /// Copies replaced while the GPU may still write them: freed once it is past their tick.
    std::vector<std::pair<u64, Nx2Staging>> nx2_retired_list_copies;
    boost::container::small_vector<std::pair<BufferId, DAddr>, 8> nx2_list_copy_requests;
    bool nx2_collect_list_copies{};
};

} // namespace VideoCommon
