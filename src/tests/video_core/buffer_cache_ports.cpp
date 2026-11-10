// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <barrier>
#include <cstring>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "common/page_table.h"
#include "common/scope_exit.h"
#include "core/core.h"
#include "core/device_memory.h"
#include "core/hle/kernel/k_process.h"
#include "core/hle/kernel/kernel.h"
#include "core/memory.h"
#include "tests/core/system_test_context.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/buffer_cache/memory_tracker_base.h"
#include "video_core/control/channel_state.h"
#include "video_core/gpu.h"
#include "video_core/renderer_null/null_rasterizer.h"
#include "video_core/v2/draw_cache.h"
#include "video_core/v2/v2_switches.h"
#include "video_core/v2/versions.h"

namespace {

struct Runtime;

struct Buffer : VideoCommon::BufferBase {
    Buffer(Runtime&, VideoCommon::NullBufferParams params) : BufferBase{params} {}
    Buffer(Runtime&, DAddr address, u32 size, bool mapped)
        : BufferBase{address, size}, bytes(size), host_mapped{mapped} {}

    void MarkUsage(u64 offset, u64 size) {
        REQUIRE(offset + size <= bytes.size());
    }
    std::span<u8> Mapped() {
        return host_mapped ? std::span{bytes} : std::span<u8>{};
    }
    u64 LastUsageTick() const {
        return tick;
    }

    std::vector<u8> bytes;
    bool host_mapped{};
    u64 tick{};
};

struct Staging {
    Buffer* buffer;
    u64 offset;
    std::span<u8> mapped_span;
};

struct Runtime {
    bool CanReportMemoryUsage() const {
        return false;
    }
    u64 GetDeviceLocalMemory() const {
        return 0;
    }
    u64 GetMaxStorageBufferRange() const {
        return 1ULL << 30;
    }
    u32 GetStorageBufferAlignment() const {
        return alignment;
    }
    u32 GetUniformBufferAlignment() const {
        return 1;
    }
    bool IsFree(u64) const {
        return true;
    }
    void WaitSubmittedTick(u64) {}
    bool CanReorderUpload(const Buffer&, std::span<VideoCommon::BufferCopy>) const {
        return false;
    }
    Staging DownloadStagingBuffer(u64 size, bool = false) {
        staging = std::make_unique<Buffer>(*this, 0, size + 128, false);
        ++allocations;
        return {staging.get(), 128, std::span{staging->bytes}.subspan(128)};
    }
    Staging UploadStagingBuffer(u64 size) {
        return DownloadStagingBuffer(size);
    }
    void CopyBuffer(Buffer* dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src.bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst->bytes.size());
            pending.push_back([dst, &src, copy] {
                std::memcpy(dst->bytes.data() + copy.dst_offset, src.bytes.data() + copy.src_offset,
                            copy.size);
            });
            ++downloads;
        }
    }
    void CopyBuffer(Buffer& dst, Buffer* src, std::span<const VideoCommon::BufferCopy> copies, bool,
                    bool) {
        for (const auto& copy : copies) {
            REQUIRE(copy.src_offset + copy.size <= src->bytes.size());
            REQUIRE(copy.dst_offset + copy.size <= dst.bytes.size());
            std::memcpy(dst.bytes.data() + copy.dst_offset, src->bytes.data() + copy.src_offset,
                        copy.size);
            ++uploads;
            upload_bytes += copy.size;
        }
    }
    void Finish() {
        if (before_completion) {
            before_completion();
        }
        for (auto& copy : pending) {
            copy();
        }
        pending.clear();
        ++finishes;
    }
    void Wait(u64 tick) {
        if (before_completion) {
            before_completion();
        }
        waited_tick = tick;
        ++waits;
    }
    void FinishForHostRead() {
        ++host_read_barriers;
        Finish();
    }
    void BindUniformBuffer(Buffer&, u32, u32) {
        ++uniform_binds;
    }
    std::span<u8> BindMappedUniformBuffer(size_t, u32, u32 size) {
        stream.resize(size);
        ++stream_copies;
        last_ref = StreamUniformRef{stream_copies};
        return stream;
    }
    // Stream uniform references (P3 versions, P4 buffer replay).
    struct StreamUniformRef {
        u32 serial;
    };
    StreamUniformRef LastStreamUniformRef() const {
        return last_ref;
    }
    void RebindStreamUniform(const StreamUniformRef&, u32) {
        ++stream_rebinds;
    }
    u64 CurrentTick() const {
        return current_tick;
    }
    void ClearBuffer(Buffer&, u64, u64, u32) {}
    void CopyBuffer(Buffer& dst, Buffer& src, std::span<const VideoCommon::BufferCopy> copies, bool) {
        for (const auto& copy : copies) {
            std::memcpy(dst.bytes.data() + copy.dst_offset, src.bytes.data() + copy.src_offset,
                        copy.size);
        }
    }
    bool AnyVertexRepacked() const {
        return false;
    }
    bool WidensUint8IndicesOnHost() const {
        return false;
    }
    void BindStorageBuffer(Buffer&, u32, u32, bool) {
        ++storage_binds;
    }
    void BindTextureBuffer(Buffer&, u32, u32, VideoCore::Surface::PixelFormat) {}

    u32 alignment{256};
    u32 uniform_binds{}, stream_copies{}, stream_rebinds{}, storage_binds{};
    u64 current_tick{1};
    StreamUniformRef last_ref{};
    u32 allocations{}, downloads{}, uploads{}, finishes{}, waits{};
    u32 host_read_barriers{};
    u64 upload_bytes{}, waited_tick{};
    std::unique_ptr<Buffer> staging;
    std::vector<u8> stream;
    std::vector<std::function<void()>> pending;
    std::function<void()> before_completion;
};

struct Tracker : VideoCommon::MemoryTrackerBase<Tegra::MaxwellDeviceMemoryManager> {
    using MemoryTrackerBase::MemoryTrackerBase;
    void MarkRegionAsCpuModified(DAddr address, u64 size) {
        ++cpu_marks;
        MemoryTrackerBase::MarkRegionAsCpuModified(address, size);
    }
    u32 cpu_marks{};
};

struct Policy {
    using Runtime = ::Runtime;
    using Buffer = ::Buffer;
    using Async_Buffer = Staging;
    using MemoryTracker = Tracker;
    static constexpr bool IS_OPENGL = false;
    static constexpr bool HAS_PERSISTENT_UNIFORM_BUFFER_BINDINGS = false;
    static constexpr bool HAS_FULL_INDEX_AND_PRIMITIVE_SUPPORT = false;
    static constexpr bool NEEDS_BIND_UNIFORM_INDEX = false;
    static constexpr bool NEEDS_BIND_STORAGE_INDEX = false;
    static constexpr bool USE_MEMORY_MAPS = true;
    static constexpr bool SEPARATE_IMAGE_BUFFER_BINDINGS = false;
    static constexpr bool USE_MEMORY_MAPS_FOR_UPLOADS = true;
};

using Cache = VideoCommon::BufferCache<Policy>;

} // namespace

namespace VideoCommon {

struct BufferCacheTestAccess {
    static Binding Bind(Cache& cache, Tegra::MemoryManager& memory, GPUVAddr descriptor,
                        bool written = false, bool native = false, bool table = false) {
        cache.gpu_memory = &memory;
        return cache.StorageBufferBinding(descriptor, 0, written, native, table);
    }
    static BufferId Add(Cache& cache, DAddr address, u32 size, bool mapped = false) {
        const auto id = cache.slot_buffers.insert(cache.runtime, address, size, mapped);
        cache.Register(id);
        cache.memory_tracker.ForEachUploadRange(address, size, [](auto, auto) {});
        return id;
    }
    static Buffer& Get(Cache& cache, BufferId id) {
        return cache.slot_buffers[id];
    }
    static void Dirty(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsGpuModified(address, size);
        cache.gpu_modified_ranges.Add(address, size);
        cache.uncommitted_gpu_modified_ranges.Add(address, size);
        cache.committed_gpu_modified_ranges.emplace_back().Add(address, size);
        cache.pending_download_ranges.emplace_back().Add(address, size);
    }
    static bool Pending(Cache& cache, DAddr address, u64 size) {
        if (cache.gpu_modified_ranges.Intersects(address, size) ||
            cache.uncommitted_gpu_modified_ranges.Intersects(address, size)) {
            return true;
        }
        for (const auto& ranges : cache.committed_gpu_modified_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        for (const auto& ranges : cache.pending_download_ranges) {
            if (ranges.Intersects(address, size))
                return true;
        }
        return false;
    }
    static void DrawWrite(Cache& cache, DAddr page) {
        std::scoped_lock lock{cache.nx2_list_mutex};
        auto& entry = cache.nx2_list_pages[page];
        entry.draw_written = true;
        ++entry.writes;
    }
    static void Download(Cache& cache, std::span<const std::pair<DAddr, u64>> ranges) {
        cache.DownloadRangesMemory(ranges);
    }
    static void Uniform(Cache& cache, BufferCacheChannelInfo& channel, BufferId id, DAddr address,
                        u32 size) {
        cache.channel_state = &channel;
        cache.maxwell3d = &channel.maxwell3d;
        cache.gpu_memory = &channel.gpu_memory;
        channel.uniform_buffers[0][0] = {.device_addr = address, .size = size, .buffer_id = id};
        cache.BindHostGraphicsUniformBuffer(0, 0, 0, true);
    }
    static void UseChannel(Cache& cache, BufferCacheChannelInfo& channel) {
        cache.channel_state = &channel;
        cache.maxwell3d = &channel.maxwell3d;
        cache.gpu_memory = &channel.gpu_memory;
    }
    static void BindStage(Cache& cache) {
        cache.BindHostStageBuffers(0);
    }
    static void MarkCpuModifiedSilently(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsCpuModified(address, size);
    }
    static void NextFrame(Cache& cache) {
        cache.native_indirect_uniforms_this_frame.clear();
    }
    static u32 CpuMarks(Cache& cache) {
        return cache.memory_tracker.cpu_marks;
    }
    /// U-01: a page learned for early downloads (Preflushable by learning).
    static void Learn(Cache& cache, DAddr address, u64 size) {
        cache.memory_tracker.MarkRegionAsPreflushable(address, size);
        cache.learned_marked = 1;
    }
    static void GpuWrite(Cache& cache, DAddr address, u64 size) {
        cache.gpu_modified_ranges.Add(address, size);
    }
    static void Landed(Cache& cache, DAddr address, u64 size) {
        cache.gpu_modified_ranges.Subtract(address, size);
    }
    static bool Preemptive(Cache& cache, DAddr address, u64 size) {
        return cache.GetFlushArea(address, size)->preemtive;
    }
};

} // namespace VideoCommon

namespace {

using Access = VideoCommon::BufferCacheTestAccess;

struct Fixture {
    ScopedSystemTestContext context;
    Core::System system;
    Core::DeviceMemory physical;
    Tegra::MaxwellDeviceMemoryManager device{physical};
    Tegra::GPU gpu{system, false, false};
    Null::RasterizerNull rasterizer{gpu};
    std::optional<Tegra::MemoryManager> memory;
    Runtime runtime;
    std::unique_ptr<Cache> cache;
    std::unique_ptr<Kernel::KProcess> process;
    Core::Asid asid{};
    u8* bytes{};

    Fixture() {
        system.Initialize();
        process = std::make_unique<Kernel::KProcess>(system.Kernel());
        auto& table = process->GetPageTable().GetImpl();
        table.Resize(24, 12);
        bytes = physical.GetPointerFromRaw<u8>(0x10000);
        std::memset(bytes, 0, 0x40000);
        for (u64 page = 0; page < 64; ++page) {
            table.entries.GetAndFault(0x10 + page)
                .Store(false, Common::PageType::Memory, 1,
                       reinterpret_cast<uintptr_t>(bytes) - 0x10000);
        }
        process->GetMemory().SetCurrentPageTable(*process);
        asid = device.RegisterProcess(&process->GetMemory());
        device.BindInterface(&rasterizer);
        device.Map(0x10000, 0x10000, 0x40000, asid);
        ResetMemory();
        cache = std::make_unique<Cache>(device, runtime);
    }
    ~Fixture() {
        cache.reset();
        memory.reset();
        device.Unmap(0x10000, 0x40000);
        device.UnregisterProcess(asid);
        system.Kernel().UnregisterKernelObject(process.get());
    }
    void ResetMemory() {
        memory.reset();
        memory.emplace(system, device, 32, 1ULL << 31, 16);
        memory->BindRasterizer(&rasterizer);
        memory->Map(0x100000, 0x10000, 0x10000, Tegra::PTEKind::PITCH, true);
    }
    void Descriptor(GPUVAddr address, u64 size) {
        std::memcpy(bytes, &address, sizeof(address));
        std::memcpy(bytes + 8, &size, sizeof(size));
    }
    VideoCommon::Binding Bind(bool written = false, bool native = false, bool table = false) {
        return Access::Bind(*cache, *memory, 0x100000, written, native, table);
    }
};

} // namespace

TEST_CASE("Storage binding cache tracks map changes and descriptor dependencies",
          "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x200000;
    f.memory->Map(base, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base + 0x123, 0x456);
    for (int i = 0; i < 3; ++i) {
        const auto binding = f.Bind(true);
        REQUIRE(binding.device_addr == 0x20100);
        REQUIRE(binding.gpu_addr == base + 0x100);
        REQUIRE(binding.size == 0x479);
        REQUIRE(binding.buffer_id == VideoCommon::BufferId{});
        REQUIRE(binding.segment_count == 0);
    }
    REQUIRE(f.Bind().size == 0xf00);
    f.Descriptor(base + 0x123, 0x1456);
    REQUIRE(f.Bind(true).size == 0x1479);
    f.runtime.alignment = 4096;
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    REQUIRE(f.Bind(true).size == 0x1579);
    REQUIRE(f.Bind(true, true).size == 0x10000);
    f.memory->Map(base, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
    f.memory->MapSparse(base, 0x10000, true);
    REQUIRE(f.Bind().size == 0);
    f.memory->Unmap(base, 0x10000);
    REQUIRE(f.Bind().size == 0);
    f.memory->Map(base, 0x20000, 0x2000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x2000);
    f.memory->Map(base + 0x1000, 0x40000, 0x1000, Tegra::PTEKind::PITCH, false);
    REQUIRE(f.Bind(true, true).size == 0x1000);
    const auto old_owner = f.memory->GetID();
    f.ResetMemory();
    f.memory->Map(base, 0x40000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(f.memory->GetID() != old_owner);
    REQUIRE(f.Bind(true).device_addr == 0x40000);
    f.Descriptor(0, 0x123);
    REQUIRE(f.Bind().size == 0);
}

TEST_CASE("Storage binding cache preserves native and table size policies", "[buffer_ports]") {
    Fixture f;
    constexpr GPUVAddr base = 0x4000000;
    f.memory->Map(base, 0x10000000, 80 << 20, Tegra::PTEKind::PITCH, true);
    f.Descriptor(base, 256);
    REQUIRE(f.Bind().size == 4096);
    REQUIRE(f.Bind(false, true).size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 80 << 20);
    REQUIRE(f.Bind(true, false, true).size == 256);
    f.Descriptor(base, (1ULL << 32) | 256);
    REQUIRE(f.Bind().size == 8 << 20);
    REQUIRE(f.Bind(false, false, true).size == 80 << 20);
    f.Descriptor(base + (79 << 20), 0);
    REQUIRE(f.Bind(false, false, true).size == 1 << 20);
}

TEST_CASE("Storage binding cache does not reuse a destroyed address space", "[buffer_ports]") {
    Fixture f;
    f.memory->Map(0x200000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    f.Descriptor(0x200000, 0x1000);
    REQUIRE(f.Bind(true).device_addr == 0x20000);
    const auto serial = f.memory->MapSerial();
    const auto owner = f.memory->GetID();
    const auto* address = &*f.memory;
    f.ResetMemory();
    f.memory->Map(0x200000, 0x30000, 0x10000, Tegra::PTEKind::PITCH, true);
    REQUIRE(&*f.memory == address);
    REQUIRE(f.memory->MapSerial() == serial);
    REQUIRE(f.memory->GetID() != owner);
    REQUIRE(f.Bind(true).device_addr == 0x30000);
}

TEST_CASE("Batch downloads clamp buffers, deduplicate pages and finish before publication",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000);
    std::ranges::fill(Access::Get(*f.cache, first).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, second).bytes, 0x34);
    Access::Dirty(*f.cache, 0x2f000, 0x2000);
    Access::Dirty(*f.cache, 0x32000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.bytes[0x1f000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
    };
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x2f008, 0x1000},
        {0x2f000, 0x2000},
        {0x20000, 0},
    }};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 1);
    REQUIRE(f.runtime.downloads == 2);
    REQUIRE(f.bytes[0x1f000] == 0x12);
    REQUIRE(f.bytes[0x1ffff] == 0x12);
    REQUIRE(f.bytes[0x20000] == 0x34);
    REQUIRE(f.bytes[0x20fff] == 0x34);
    REQUIRE(f.bytes[0x22000] == 0);
    REQUIRE_FALSE(Access::Pending(*f.cache, 0x2f000, 0x2000));
    REQUIRE(Access::Pending(*f.cache, 0x32000, 0x1000));
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Mapped and mixed batches make device writes host visible in one completion",
          "[buffer_ports]") {
    Fixture f;
    const auto first = Access::Add(*f.cache, 0x20000, 0x10000, true);
    const auto second = Access::Add(*f.cache, 0x30000, 0x10000, true);
    const auto clean = Access::Add(*f.cache, 0x40000, 0x10000, true);
    auto& a = Access::Get(*f.cache, first);
    auto& b = Access::Get(*f.cache, second);
    a.tick = 8;
    b.tick = 19;
    Access::Get(*f.cache, clean).tick = 99;
    const std::array<std::pair<DAddr, u64>, 3> ranges{{
        {0x20000, 0x1000},
        {0x30000, 0x1000},
        {0x40000, 0x1000},
    }};
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.bytes[0x20000] == 0);
        std::ranges::fill(a.bytes, 0x56);
        std::ranges::fill(b.bytes, 0x78);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.host_read_barriers == 1);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0x56);
    REQUIRE(f.bytes[0x20000] == 0x78);
    b.host_mapped = false;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::Dirty(*f.cache, 0x30000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 2);
        REQUIRE(f.bytes[0x10000] == 0x56);
        REQUIRE(f.bytes[0x20000] == 0x78);
        std::ranges::fill(a.bytes, 0x9a);
        std::ranges::fill(b.bytes, 0xbc);
    };
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.finishes == 2);
    REQUIRE(f.runtime.downloads == 1);
    REQUIRE(f.bytes[0x10000] == 0x9a);
    REQUIRE(f.bytes[0x20000] == 0xbc);
    Access::Download(*f.cache, ranges);
    REQUIRE(f.runtime.host_read_barriers == 2);
    REQUIRE(f.runtime.finishes == 2);
}

TEST_CASE("Single mapped download completes host visibility before guest publication",
          "[buffer_ports]") {
    Fixture f;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, true);
    auto& buffer = Access::Get(*f.cache, id);
    buffer.tick = 7;
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        std::ranges::fill(buffer.bytes, 0xcd);
    };
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.waits == 0);
    REQUIRE(f.runtime.allocations == 0);
    REQUIRE(f.bytes[0x10000] == 0xcd);
    f.cache->DownloadMemory(0x20000, 0x1000);
    REQUIRE(f.runtime.finishes == 1);
}

TEST_CASE("Command page batch publishes only completed matching write generations",
          "[buffer_ports]") {
    Fixture f;
    const bool mapped = GENERATE(false, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000, mapped);
    std::ranges::fill(Access::Get(*f.cache, id).bytes, 0x42);
    Access::Dirty(*f.cache, 0x20000, 0x3000);
    Access::DrawWrite(*f.cache, 0x20000);
    Access::DrawWrite(*f.cache, 0x21000);
    Access::DrawWrite(*f.cache, 0x22000);
    std::array<u8, 4> out{};
    using Read = Cache::Nx2ListRead;
    f.runtime.before_completion = [&] {
        REQUIRE(f.runtime.host_read_barriers == 1);
        REQUIRE(f.bytes[0x10000] == 0);
        REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Unknown);
        Access::DrawWrite(*f.cache, 0x22000);
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == (mapped ? 0U : 3U));
    REQUIRE(f.cache->Nx2ReadListCopy(0x20000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Synced);
    REQUIRE(f.cache->Nx2ReadListCopy(0x22000, out, true) == Read::Unknown);
    REQUIRE(f.bytes[0x12000] == 0x42);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, false) == Read::NeedsSync);
    Access::DrawWrite(*f.cache, 0x21000);
    REQUIRE(f.cache->Nx2ReadListCopy(0x21000, out, true) == Read::Unknown);
}

TEST_CASE("Command page readers remain ordered during batch completion", "[buffer_ports]") {
    struct {
        Core::DeviceMemory physical;
        Tegra::MaxwellDeviceMemoryManager device{physical};
        Runtime runtime;
        std::unique_ptr<Cache> cache = std::make_unique<Cache>(device, runtime);
    } f;
    Access::Add(*f.cache, 0x20000, 0x10000);
    Access::Dirty(*f.cache, 0x20000, 0x1000);
    Access::DrawWrite(*f.cache, 0x20000);
    std::barrier rendezvous{2};
    Cache::Nx2ListRead before{}, after{};
    std::jthread reader{[&] {
        std::array<u8, 4> out{};
        rendezvous.arrive_and_wait();
        before = f.cache->Nx2ReadListCopy(0x20000, out, true);
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
        after = f.cache->Nx2ReadListCopy(0x20000, out, true);
    }};
    f.runtime.before_completion = [&] {
        rendezvous.arrive_and_wait();
        rendezvous.arrive_and_wait();
    };
    const std::array<DAddr, 1> requested{0x20000};
    f.cache->Nx2FlushListPages(requested);
    rendezvous.arrive_and_wait();
    reader.join();
    REQUIRE(before == Cache::Nx2ListRead::Unknown);
    REQUIRE(after == Cache::Nx2ListRead::Synced);
    REQUIRE(f.runtime.finishes == 1);
    REQUIRE(f.runtime.downloads == 1);
}

TEST_CASE("Native uniform banks upload directly without overwriting GPU-owned bytes",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    std::memset(f.bytes + 0x10000, 0x5a, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x10000);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    REQUIRE(Access::Get(*f.cache, id).bytes[0xffff] == 0x5a);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    Access::NextFrame(*f.cache);
    Access::Dirty(*f.cache, 0x21000, 0x1000);
    std::memset(f.bytes + 0x10000, 0xa5, 0x10000);
    Access::Uniform(*f.cache, channel, id, 0x20000, 0x10000);
    REQUIRE(f.runtime.uploads == 1);
    REQUIRE(Access::Get(*f.cache, id).bytes[0x1000] == 0x5a);
}

TEST_CASE("Native uniform uploads stop at holes and keep unaligned bank boundaries",
          "[buffer_ports]") {
    Fixture f;
    f.memory->EnableNativeTiling();
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x10000;
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x20000);
    std::memset(f.bytes + 0x10000, 0x6b, 0x20000);
    f.device.Unmap(0x22000, 0x1000);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x1edd);
    const auto& data = Access::Get(*f.cache, id).bytes;
    REQUIRE(data[0x122] == 0);
    REQUIRE(data[0x123] == 0x6b);
    REQUIRE(data[0x1fff] == 0x6b);
    REQUIRE(data[0x2000] == 0);
    REQUIRE(data[0x3000] == 0);
    REQUIRE(Access::CpuMarks(*f.cache) == 0);
    f.device.Map(0x22000, 0x22000, 0x1000, f.asid);
    Access::NextFrame(*f.cache);
    Access::Uniform(*f.cache, channel, id, 0x20123, 0x10000);
    REQUIRE(f.runtime.upload_bytes == 0x11edd);
    REQUIRE(data[0x10122] == 0x6b);
    REQUIRE(data[0x10123] == 0);
}

TEST_CASE("Downloads keep CPU writes collected and not yet applied", "[buffer_ports]") {
    Fixture f;
    const auto staged = Access::Add(*f.cache, 0x20000, 0x10000);
    const auto mapped = Access::Add(*f.cache, 0x30000, 0x10000, true);
    std::ranges::fill(Access::Get(*f.cache, staged).bytes, 0x12);
    std::ranges::fill(Access::Get(*f.cache, mapped).bytes, 0x34);
    f.cache->SetPendingCpuWrites([](const std::function<void(DAddr, size_t)>& callback) {
        callback(0x20100, 0x10);
        callback(0x30ff8, 0x10);
    });
    std::memset(f.bytes + 0x10100, 0xee, 0x10);
    std::memset(f.bytes + 0x20ff8, 0xee, 0x10);
    Access::Dirty(*f.cache, 0x20000, 0x2000);
    Access::Dirty(*f.cache, 0x30000, 0x2000);
    const std::array<std::pair<DAddr, u64>, 1> ranges{{{0x20000, 0x2000}}};
    Access::Download(*f.cache, ranges);
    REQUIRE(f.bytes[0x100ff] == 0x12);
    REQUIRE(std::all_of(f.bytes + 0x10100, f.bytes + 0x10110, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x10110] == 0x12);
    REQUIRE(f.bytes[0x11fff] == 0x12);
    f.cache->DownloadMemory(0x30000, 0x2000);
    REQUIRE(f.bytes[0x20ff7] == 0x34);
    REQUIRE(std::all_of(f.bytes + 0x20ff8, f.bytes + 0x21008, [](u8 b) { return b == 0xee; }));
    REQUIRE(f.bytes[0x21008] == 0x34);
}

TEST_CASE("Draw cache buffer replay binds unchanged cached uniforms without syncing",
          "[buffer_ports][v2_p4]") {
    namespace V2 = VideoCommon::V2;
    const auto saved = V2::g_modes;
    V2::g_modes[static_cast<size_t>(V2::Switch::DrawCache)] = V2::Mode::On;
    SCOPE_EXIT {
        V2::g_modes = saved;
    };
    Fixture f;
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x2000; // over the skip-cache size: the cached path
    channel.uniform_buffer_sizes = &sizes;
    channel.enabled_uniform_buffer_masks[0] = 1;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    Access::UseChannel(*f.cache, channel);
    channel.uniform_buffers[0][0] = {.device_addr = 0x20000, .size = 0x2000, .buffer_id = id};
    V2::Versions::EpochMemo memo{V2::Versions::Pages()};

    // A resolve: the guest wrote the range, the bind uploads it; the recipe records it.
    std::memset(f.bytes + 0x10000, 0x11, 0x2000);
    f.cache->WriteMemory(0x20000, 0x2000);
    const u64 stamp = V2::Versions::Pages().Stamp();
    Access::BindStage(*f.cache);
    REQUIRE(f.runtime.uploads == 1);
    REQUIRE(f.runtime.uniform_binds == 1);
    V2::DrawCache::BufferRecipe recipe;
    f.cache->RecordGraphicsBuffers(recipe, false, stamp);
    REQUIRE(recipe.valid);
    REQUIRE(recipe.uniform.size() == 1);
    REQUIRE(recipe.uniform[0].kind == V2::DrawCache::BufferBind::Kind::Cached);
    REQUIRE(f.cache->CountStaleRecipeRanges(recipe, memo) == 0);

    // A replay with nothing written: bound from the recipe, no upload.
    f.cache->ReplayHostStageBuffers(0, recipe, memo);
    REQUIRE(f.runtime.uploads == 1);
    REQUIRE(f.runtime.uniform_binds == 2);
    REQUIRE(f.cache->TakeReplayCounts() == std::pair<u32, u32>{1, 0});

    // A guest write (any path that marks it CPU-modified moves its version): synced again.
    f.bytes[0x10010] = 0x22;
    f.cache->WriteMemory(0x20010, 4);
    f.cache->ReplayHostStageBuffers(0, recipe, memo);
    REQUIRE(f.runtime.uploads == 2);
    REQUIRE(Access::Get(*f.cache, id).bytes[0x10] == 0x22);
    REQUIRE(f.cache->TakeReplayCounts() == std::pair<u32, u32>{0, 1});

    // Recorded again after the sync (new stamp): replayed again.
    const u64 restamp = V2::Versions::Pages().Stamp();
    f.cache->RecordGraphicsBuffers(recipe, false, restamp);
    f.cache->ReplayHostStageBuffers(0, recipe, memo);
    REQUIRE(f.runtime.uploads == 2);
    REQUIRE(f.cache->TakeReplayCounts() == std::pair<u32, u32>{1, 0});

    // A rebound buffer (other range) takes the existing path.
    channel.uniform_buffers[0][0] = {.device_addr = 0x24000, .size = 0x2000, .buffer_id = id};
    f.cache->ReplayHostStageBuffers(0, recipe, memo);
    REQUIRE(f.cache->TakeReplayCounts() == std::pair<u32, u32>{0, 1});
    channel.uniform_buffers[0][0] = {.device_addr = 0x20000, .size = 0x2000, .buffer_id = id};

    // Verify's byte check: CPU-modified behind unchanged versions is counted as stale.
    Access::MarkCpuModifiedSilently(*f.cache, 0x20000, 0x1000);
    REQUIRE(f.cache->CountStaleRecipeRanges(recipe, memo) == 1);

    // Fix pass 1 (Ocarina of Time, ~79% of sampled hits "CPU-modified behind unchanged
    // versions"): bindings the replay doesn't take from the host buffer are no stale ranges.
    // A constant buffer version (or a stream copy) reads guest memory and never syncs the
    // buffer, so its CPU-modified bits stay set by design.
    V2::DrawCache::BufferRecipe versioned = recipe;
    versioned.uniform[0].kind = V2::DrawCache::BufferBind::Kind::Version;
    REQUIRE(f.cache->CountStaleRecipeRanges(versioned, memo) == 0);
    versioned.uniform[0].kind = V2::DrawCache::BufferBind::Kind::Normal;
    REQUIRE(f.cache->CountStaleRecipeRanges(versioned, memo) == 0);
    // A recipe whose buffer slot moved on (generation) is never replayed.
    V2::DrawCache::BufferRecipe moved = recipe;
    ++moved.uniform[0].generation;
    REQUIRE(f.cache->CountStaleRecipeRanges(moved, memo) == 0);
}

TEST_CASE("Index and vertex ranges are computed as the update passes compute them",
          "[buffer_ports][v2_drawrec]") {
    // P6 Step 0: UpdateIndexBuffer/UpdateVertexBuffer and the record resolve share these.
    Fixture f;
    // 16-bit indices 4..11 of a 0x400-byte buffer: what the draw reads, within the buffer.
    auto range = Cache::IndexRange(*f.memory, 0x100000, 0x100400, 4, 8, 2);
    REQUIRE(range.valid);
    REQUIRE(range.device_addr == 0x10000);
    REQUIRE(range.size == 24);
    // A buffer shorter than the draw: the buffer's size.
    range = Cache::IndexRange(*f.memory, 0x100000, 0x100010, 4, 8, 2);
    REQUIRE(range.size == 0x10);
    REQUIRE_FALSE(Cache::IndexRange(*f.memory, 0x100000, 0x100400, 0, 0, 2).valid);
    REQUIRE_FALSE(Cache::IndexRange(*f.memory, 0x900000, 0x900400, 0, 8, 2).valid);

    // A vertex stream ends at its limit (inclusive), a native one after its size.
    range = Cache::VertexRange(*f.memory, 0x100000, true, 0x1000ff, false, 0);
    REQUIRE(range.valid);
    REQUIRE(range.device_addr == 0x10000);
    REQUIRE(range.size == 0x100);
    range = Cache::VertexRange(*f.memory, 0x100100, true, 0, true, 0x80);
    REQUIRE(range.device_addr == 0x10100);
    REQUIRE(range.size == 0x80);
    REQUIRE_FALSE(Cache::VertexRange(*f.memory, 0x100000, false, 0x1000ff, false, 0).valid);
    REQUIRE_FALSE(Cache::VertexRange(*f.memory, 0x900000, true, 0x9000ff, false, 0).valid);
    // A stream of 64 MiB or more stops where the mapping does.
    range = Cache::VertexRange(*f.memory, 0x10f000, true, 0x10f000 + 0x4000000ULL - 1, false, 0);
    REQUIRE(range.size == 0x1000);
    REQUIRE_THROWS(Cache::VertexRange(*f.memory, 0x100000, true, 0, true, 0x100000000ULL));
}

TEST_CASE("Record buffer twins peek, resolve and encode without the update passes",
          "[buffer_ports][v2_drawrec]") {
    namespace R = VideoCommon::V2::Resolve;
    Fixture f;
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    channel.uniform_buffer_sizes = &sizes;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    Access::UseChannel(*f.cache, channel);

    // PeekBuffer: the holding buffer, a creation due, or no address.
    REQUIRE(f.cache->PeekBuffer(0x20100, 0x100, false) == id);
    REQUIRE_FALSE(static_cast<bool>(f.cache->PeekBuffer(0x2ff00, 0x200, false)));
    REQUIRE(f.cache->PeekBuffer(0, 0x100, false) == VideoCommon::NULL_BUFFER_ID);

    // A storage descriptor read through the view gives StorageBufferBinding's binding.
    f.Descriptor(0x100040, 0x80);
    const R::LiveView view{channel.maxwell3d};
    bool multi_range = true;
    const R::ResolvedSsbo ssbo =
        f.cache->ResolveStorageBinding(view, 0, 0, 0, 0x100000, false, ~0U, multi_range);
    const VideoCommon::Binding expected = f.Bind();
    REQUIRE(ssbo.daddr == expected.device_addr);
    REQUIRE(ssbo.gpu_addr == expected.gpu_addr);
    REQUIRE(ssbo.size == expected.size);
    // Disabled bindings resolve to nothing.
    REQUIRE(f.cache->ResolveUniformBinding(view, 0, 3).size == 0);
    REQUIRE_FALSE(f.cache->ResolveVertexBinding(view, 0).has_value());
    REQUIRE_FALSE(f.cache->ResolveIndexBinding(view).valid);

    // The record's bindings land in the channel state as the update passes leave them.
    R::ResolvedDraw record;
    record.h.stages[0] = true;
    const std::array vertex{R::ResolvedVertex{
        .slot = 2, .buffer = {id.index, 0}, .daddr = 0x20000, .size = 0x400, .stride = 16}};
    record.h.vertex = record.Push(std::span<const R::ResolvedVertex>(vertex));
    const std::array ubo{R::ResolvedUbo{
        .stage = 0, .slot = 1, .daddr = 0x20400, .size = 0x100, .buffer = {id.index, 0}}};
    record.h.ubo = record.Push(std::span<const R::ResolvedUbo>(ubo));
    const std::array storage{R::ResolvedSsbo{.stage = 0, .index = 3, .written = 1,
                                             .daddr = 0x20800, .gpu_addr = 0x100800,
                                             .size = 0x40, .buffer = {id.index, 0}}};
    record.h.ssbo = record.Push(std::span<const R::ResolvedSsbo>(storage));
    channel.vertex_buffers[5] = {.device_addr = 0x20000, .size = 4, .buffer_id = id};
    f.cache->EncodeGraphicsBindings(record);
    REQUIRE(channel.vertex_buffers[2].device_addr == 0x20000);
    REQUIRE(channel.vertex_buffers[2].buffer_id == id);
    REQUIRE(channel.vertex_buffers[5].buffer_id == VideoCommon::NULL_BUFFER_ID);
    REQUIRE(channel.uniform_buffers[0][1].device_addr == 0x20400);
    REQUIRE(channel.enabled_storage_buffers[0] == (u64{1} << 3));
    REQUIRE(channel.written_storage_buffers[0] == (u64{1} << 3));
    REQUIRE(channel.storage_buffers[0][3].gpu_addr == 0x100800);

    // A creation request applies as V2ApplyCreates does: the lookup creates.
    VideoCommon::V2::Commit::Request create{};
    create.kind = VideoCommon::V2::Commit::Kind::BufferCreate;
    create.address = 0x40000;
    create.size = 0x100;
    const VideoCommon::BufferId created{f.cache->ApplyDrawRequest(create)};
    REQUIRE(static_cast<bool>(created));
    REQUIRE(f.cache->PeekBuffer(0x40000, 0x100, false) == created);
}

TEST_CASE("The record's index binding takes the draw's index range, not the vertex range",
          "[buffer_ports][v2_drawrec]") {
    // F2e (c236 DRAWREC=verify: index mismatches on 96% of draws): DrawPredicted, DrawIndex and
    // DrawIndexSmall never set draw_state.draw_indexed, so a view reading that flag sized the
    // index range from vertex_buffer.first/count. The view takes the rasterizer's is_indexed.
    namespace R = VideoCommon::V2::Resolve;
    Fixture f;
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    channel.uniform_buffer_sizes = &sizes;
    f.memory->Map(0x110000, 0x20000, 0x10000, Tegra::PTEKind::PITCH, true);
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    Access::UseChannel(*f.cache, channel);
    auto& engine = channel.maxwell3d;
    auto& regs = engine.regs;
    regs.index_buffer.start_addr_high = 0;
    regs.index_buffer.start_addr_low = 0x110000;
    regs.index_buffer.limit_addr_high = 0;
    regs.index_buffer.limit_addr_low = 0x110400;
    regs.index_buffer.format = R::MaxwellRegs::IndexFormat::UnsignedShort;
    auto& draw = engine.draw_manager.draw_state;
    draw.index_buffer = regs.index_buffer;
    draw.index_buffer.first = 4;
    draw.index_buffer.count = 8;
    draw.vertex_buffer.first = 0;
    draw.vertex_buffer.count = 300;
    draw.draw_indexed = false; // as DrawPredicted leaves it

    const R::ResolvedIndex index = f.cache->ResolveIndexBinding(R::LiveView{engine, true});
    REQUIRE(index.valid);
    REQUIRE(index.first == 4);
    REQUIRE(index.count == 8);
    REQUIRE(index.daddr == 0x20000);
    // UpdateIndexBuffer's range: (first + count) * 2 bytes.
    const auto range = Cache::IndexRange(*f.memory, 0x110000, 0x110400, 4, 8, 2);
    REQUIRE(index.size == range.size);
    REQUIRE(index.size == 24);
    REQUIRE(index.buffer.id == id.index);
    REQUIRE(R::LiveView{engine, true}.Draw().indexed);
    REQUIRE_FALSE(R::LiveView{engine}.Draw().indexed);
}

TEST_CASE("Draw cache buffer replay binds a uniform on the path its bind takes now",
          "[buffer_ports][v2_p4]") {
    // F2-X1: a recorded uniform is replayed only on the path BindHostGraphicsUniformBuffer
    // takes for it now (the stream for a small range nobody on the GPU wrote, the host buffer
    // once the GPU wrote it).
    namespace V2 = VideoCommon::V2;
    using Kind = V2::DrawCache::BufferBind::Kind;
    const auto saved = V2::g_modes;
    V2::g_modes[static_cast<size_t>(V2::Switch::DrawCache)] = V2::Mode::On;
    SCOPE_EXIT {
        V2::g_modes = saved;
    };
    Fixture f;
    Tegra::Control::ChannelState state{0};
    state.memory_manager = &*f.memory;
    state.payload.emplace(f.system, *f.memory, state);
    VideoCommon::BufferCacheChannelInfo channel{state};
    VideoCommon::UniformBufferSizes sizes{};
    sizes[0][0] = 0x100; // under the skip-cache size: the stream path while not GPU-modified
    channel.uniform_buffer_sizes = &sizes;
    channel.enabled_uniform_buffer_masks[0] = 1;
    const auto id = Access::Add(*f.cache, 0x20000, 0x10000);
    Access::UseChannel(*f.cache, channel);
    const VideoCommon::Binding binding{.device_addr = 0x20000, .size = 0x100, .buffer_id = id};
    channel.uniform_buffers[0][0] = binding;
    REQUIRE(f.cache->UniformBindKind(0, 0, binding) == Kind::Version);
    REQUIRE(f.cache->UniformBindKind(
                0, 0,
                VideoCommon::Binding{.device_addr = 0x20000,
                                     .size = 0x100,
                                     .buffer_id = VideoCommon::NULL_BUFFER_ID}) == Kind::Normal);

    // A recipe that recorded the host buffer (the range was GPU-modified then).
    V2::Versions::EpochMemo memo{V2::Versions::Pages()};
    V2::DrawCache::BufferRecipe recipe;
    recipe.valid = true;
    recipe.stamp = V2::Versions::Pages().Stamp();
    recipe.uniform.push_back({
        .stage = 0,
        .index = 0,
        .kind = Kind::Cached,
        .buffer_id = id.index,
        .generation = V2::Versions::Gens().Get(V2::Versions::Kind::Buffer, id.index),
        .device_addr = 0x20000,
        .size = 0x100,
    });
    recipe.MergeRanges();
    // Not GPU-modified any more: the resolve streams it, so the replay does too.
    f.cache->ReplayHostStageBuffers(0, recipe, memo);
    REQUIRE(f.runtime.uniform_binds == 0);
    REQUIRE(f.cache->TakeReplayCounts() == std::pair<u32, u32>{0, 1});

    // GPU-modified: the host buffer, from the recipe.
    Access::Dirty(*f.cache, 0x20000, 0x100);
    REQUIRE(f.cache->UniformBindKind(0, 0, binding) == Kind::Cached);
    f.cache->ReplayHostStageBuffers(0, recipe, memo);
    REQUIRE(f.runtime.uniform_binds == 1);
    REQUIRE(f.cache->TakeReplayCounts() == std::pair<u32, u32>{1, 0});
}

namespace {
namespace Commit = VideoCommon::V2::Commit;

struct CommitChannel {
    Tegra::Control::ChannelState state{0};
    std::optional<VideoCommon::BufferCacheChannelInfo> channel;
    VideoCommon::UniformBufferSizes sizes{};

    explicit CommitChannel(Fixture& f) {
        state.memory_manager = &*f.memory;
        state.payload.emplace(f.system, *f.memory, state);
        channel.emplace(state);
        channel->uniform_buffer_sizes = &sizes;
        Access::UseChannel(*f.cache, *channel);
    }
};
} // namespace

TEST_CASE("Commit: a missing buffer is a proxy patched at the end of the update pass",
          "[buffer_ports][v2_p5]") {
    Fixture f;
    CommitChannel c{f};
    Commit::Context context;
    f.cache->SetCommit(&context);
    c.sizes[0][0] = 0x2000;
    c.channel->enabled_uniform_buffer_masks[0] = 1;
    c.channel->uniform_buffers[0][0] = {.device_addr = 0x20000, .size = 0x2000, .buffer_id = {}};
    f.cache->UpdateGraphicsBuffers(false);
    const auto counters = context.stage.TakeCounters();
    REQUIRE(counters.requests[static_cast<size_t>(Commit::Kind::BufferCreate)] == 1);
    REQUIRE(counters.proxies_patched == 1);
    const VideoCommon::BufferId id = c.channel->uniform_buffers[0][0].buffer_id;
    REQUIRE(id);
    REQUIRE_FALSE(Commit::IsProxy(id.index));
    REQUIRE(Access::Get(*f.cache, id).IsInBounds(0x20000, 0x2000));
    REQUIRE(f.cache->LookupBuffer(0x20000, 0x2000) == id);
    f.cache->CommitBinds();
    f.cache->SetCommit(nullptr);
}

TEST_CASE("Commit: a proxy whose creation merges a bound buffer resolves both to the merge",
          "[buffer_ports][v2_p5]") {
    Fixture f;
    CommitChannel c{f};
    Commit::Context context;
    const VideoCommon::BufferId old_id = Access::Add(*f.cache, 0x20000, 0x10000);
    f.cache->SetCommit(&context);
    c.sizes[0][0] = 0x2000;
    c.sizes[0][1] = 0x1000;
    c.channel->enabled_uniform_buffer_masks[0] = 0b11;
    // Binding 0 crosses the existing buffer's end: a miss whose creation joins it. Binding 1
    // hits the existing buffer, which the merge deletes.
    c.channel->uniform_buffers[0][0] = {.device_addr = 0x2f000, .size = 0x2000, .buffer_id = {}};
    c.channel->uniform_buffers[0][1] = {.device_addr = 0x20000, .size = 0x1000, .buffer_id = {}};
    f.cache->UpdateGraphicsBuffers(false);
    const VideoCommon::BufferId merged = c.channel->uniform_buffers[0][0].buffer_id;
    REQUIRE(merged);
    REQUIRE_FALSE(Commit::IsProxy(merged.index));
    REQUIRE(c.channel->uniform_buffers[0][1].buffer_id == merged);
    REQUIRE(Access::Get(*f.cache, merged).IsInBounds(0x20000, 0x11000));
    REQUIRE(f.cache->LookupBuffer(0x20000, 0x1000) == merged);
    REQUIRE(f.cache->LookupBuffer(0x2f000, 0x2000) == merged);
    // The binding's offset comes from its address in the merged buffer.
    REQUIRE(Access::Get(*f.cache, merged).Offset(0x2f000) == 0xf000);
    static_cast<void>(old_id);
    f.cache->CommitBinds();
    f.cache->SetCommit(nullptr);
}

TEST_CASE("Commit: bind uploads are ordered requests applied before the draw",
          "[buffer_ports][v2_p5]") {
    Fixture f;
    CommitChannel c{f};
    Commit::Context context;
    const VideoCommon::BufferId id = Access::Add(*f.cache, 0x20000, 0x10000);
    f.cache->SetCommit(&context);
    c.sizes[0][0] = 0x2000; // over the skip-cache size: the cached path syncs
    c.channel->enabled_uniform_buffer_masks[0] = 1;
    c.channel->uniform_buffers[0][0] = {.device_addr = 0x20000, .size = 0x2000, .buffer_id = id};
    std::memset(f.bytes + 0x10000, 0x33, 0x2000);
    f.cache->WriteMemory(0x20000, 0x2000);
    f.cache->UpdateGraphicsBuffers(false);
    REQUIRE(f.cache->HasDeferredBinds());
    Access::BindStage(*f.cache);
    // Bound (buffer + offset) at once; the upload waits for the commit.
    REQUIRE(f.runtime.uniform_binds == 1);
    REQUIRE(f.runtime.uploads == 0);
    f.cache->CommitBinds();
    REQUIRE_FALSE(f.cache->HasDeferredBinds());
    REQUIRE(f.runtime.uploads == 1);
    REQUIRE(Access::Get(*f.cache, id).bytes[0] == 0x33);
    // Off the scope, a bind syncs at once again.
    f.bytes[0x10000] = 0x44;
    f.cache->WriteMemory(0x20000, 4);
    Access::BindStage(*f.cache);
    REQUIRE(f.runtime.uploads == 2);
    f.cache->SetCommit(nullptr);
}

TEST_CASE("U-01: a learned page read before its early download landed is read back",
          "[buffer_ports][v2_p5b]") {
    namespace V2 = VideoCommon::V2;
    const auto saved = V2::g_modes;
    const bool saved_ampere = Tegra::g_nx2_ampere.load();
    V2::g_modes[static_cast<size_t>(V2::Switch::Readback)] = V2::Mode::On;
    Tegra::g_nx2_ampere.store(true);
    SCOPE_EXIT {
        V2::g_modes = saved;
        Tegra::g_nx2_ampere.store(saved_ampere);
    };
    Fixture f;
    Access::Add(*f.cache, 0x20000, 0x10000);
    Access::Learn(*f.cache, 0x21000, 0x1000);
    // Learned and nothing owed: memory is current, no readback.
    REQUIRE(Access::Preemptive(*f.cache, 0x21000, 0x100));
    // A GPU write whose download has not landed: the old answer ("preemptive") was a stale read.
    Access::GpuWrite(*f.cache, 0x21000, 0x1000);
    REQUIRE_FALSE(Access::Preemptive(*f.cache, 0x21000, 0x100));
    // Landed: preemptive again.
    Access::Landed(*f.cache, 0x21000, 0x1000);
    REQUIRE(Access::Preemptive(*f.cache, 0x21000, 0x100));
}
