// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <iterator>
#include <type_traits>

#include "common/common_types.h"

// Pipeline v2 typed IR (MASTER_PLAN.md §5.2): the records the split front sends its back with
// EDEN_NX2_V2_IR=1, in the same SPSC ring as the word stream they replace (gpu_split.h).
//
// Every record in the ring has a fixed size for its kind: a 2-word header, a payload reference
// when the kind carries variable data, and a fixed-size POD body. Variable data (StateRun
// entries, method data words, command list pages, debug callback captures) lives in a separate
// payload arena (ir_arena.h), retired by executed sequence number. Objects (engines, the
// rasterizer, memory managers, ...) are referred to by index into a side table
// (ObjectTable, ir_arena.h), never by pointer.
//
//   word 0: bits 0-3   GpuSplit ring kind (always RingKind)
//           bits 4-7   GpuSplit flags (bit 7: FastDrawNeutral)
//           bits 8-15  Kind
//           bits 16-23 body size in words
//           bits 24-31 Flags
//           bits 32-63 record size in words, header included
//   word 1: sequence number: +1 per record, from 1 after each split start
//   [word 2: PayloadRef, when HasPayload]
//   body
//
// To add a record kind: add it to Kind (before Count) and to kind_names, define its body as a
// trivially copyable struct of whole u64 words below, add an emitter to GpuSplit and a case in
// GpuSplit::ExecuteIr / V2::IrExec. Record layouts are documented in IMPLEMENTATION_LOG.md (P1b).
namespace VideoCommon::V2::Ir {

/// GpuSplit::Kind::Ir: the ring entry kind of every IR record.
inline constexpr u64 RingKind = 5;

enum class Kind : u8 {
    Invalid,
    Engine,     ///< EngineBody: the engine later methods go to
    StateRun,   ///< Payload: method entries (StateRun entry format) to the current engine
    MethodData, ///< MethodDataBody; payload: the data words packed 2 per u64
    Callback,   ///< Debug hooks only: payload = thunk + trivially copyable capture
    Draw,       ///< DrawBody: a draw the front predicts, or one the back executes (DrawOp)
    Dispatch,   ///< DispatchBody: a compute launch trigger the front saw
    Copy,       ///< CopyBody: a DMA or 2D copy launch (informational)
    Release,    ///< ReleaseBody: a semaphore release (executed: rasterizer Query)
    Acquire,    ///< AcquireBody: an acquire the front decided (informational)
    Barrier,    ///< BarrierBody: WFI, cache invalidations, fence releases, TickWork, ...
    Readback,   ///< ReadbackBody: flushes of GPU data for the front or the CPU
    Map,        ///< MapBody: a GPU/SMMU memory map or unmap
    Present,    ///< PresentBody: a composite request (executed by TickWork, see the log)
    SyncPoint,  ///< SyncPointBody: syncpoint increments, CPU queue fences, event signals
    Channel,    ///< ChannelBody: the rasterizer binds a channel
    Segment,    ///< SegmentBody: a P6 parallel resolve segment starts (EDEN_NX2_V2_PARALLEL)
    Compact,    ///< CompactBody: a Draw or Dispatch record in the compact form (no consumer)
    Count,
};

inline constexpr const char* kind_names[] = {
    "invalid",  "engine",  "state run", "method data", "callback", "draw",
    "dispatch", "copy",    "release",   "acquire",     "barrier",  "readback",
    "map",      "present", "sync point", "channel",   "segment", "compact",
};
static_assert(std::size(kind_names) == static_cast<size_t>(Kind::Count));

enum Flags : u64 {
    HasThunk = 1ULL << 24,   ///< Callback: the payload starts with the thunk the back calls
    HasSegment = 1ULL << 25, ///< MethodData: MethodDataBody::segment is the DMA segment
    HasPayload = 1ULL << 26, ///< Word 2 is a PayloadRef
    /// StateRun: S1 classified entries for a Maxwell3D (v2/s1_deltas.h), not method words
    Classified = 1ULL << 27,
};

inline constexpr size_t HeaderWords = 2;

[[nodiscard]] constexpr u64 MakeHeader(Kind kind, size_t body_words, size_t words, u64 flags) {
    return RingKind | flags | (u64{static_cast<u8>(kind)} << 8) | (u64{body_words & 0xff} << 16) |
           (u64{static_cast<u32>(words)} << 32);
}
[[nodiscard]] constexpr Kind KindOf(u64 header) {
    return static_cast<Kind>((header >> 8) & 0xff);
}
[[nodiscard]] constexpr size_t BodyWords(u64 header) {
    return static_cast<size_t>((header >> 16) & 0xff);
}
[[nodiscard]] constexpr size_t RecordWords(u64 header) {
    return static_cast<size_t>(header >> 32);
}
/// Words between the header and the body (the payload reference).
[[nodiscard]] constexpr size_t RefWords(u64 header) {
    return (header & HasPayload) != 0 ? 1 : 0;
}

/// Where a record's variable data is in the payload arena (word index and size).
struct PayloadRef {
    u32 index;
    u32 words;
};
[[nodiscard]] constexpr u64 PackRef(u32 index, u32 words) {
    return u64{index} | (u64{words} << 32);
}
[[nodiscard]] constexpr PayloadRef UnpackRef(u64 word) {
    return {static_cast<u32>(word), static_cast<u32>(word >> 32)};
}

/// An index into the ObjectTable (ir_arena.h).
using ObjectRef = u32;
inline constexpr ObjectRef NoObject = ~0U;

// StateRun entries: one word per method write, two when it carries a DMA segment, in the
// format of the old Method ring entry (gpu_split.h) so the back runs them on the same path:
// bits 0-3 = 2 (method), 4 IsLast, 5 HasSegment, 6 Direct (a macro write), 8-23 method,
// 32-63 argument; a HasSegment entry is followed by the segment address. A Classified run
// holds S1 entries instead (v2/s1_deltas.h); with the S1 check on, method runs carry them too.

/// MethodData: a method with several data words (CallMultiMethod).
struct MethodDataBody {
    u32 method;
    u32 amount; ///< Data words in the payload (packed 2 per u64)
    u32 methods_pending;
    u32 reserved;
    u64 segment; ///< With HasSegment: the engine's current_dma_segment
};

struct EngineBody {
    ObjectRef engine; ///< Tegra::Engines::EngineInterface
    u32 reserved;
};

/// Sentinel for a size or value the front does not know (a GPU-valued result).
inline constexpr u64 Unknown = ~0ULL;

/// What a Draw record does on the back.
enum class DrawOp : u8 {
    Predicted,    ///< Informational: draws the back performs for the method after it
    MacroEnd,     ///< Executed: Maxwell3D::FinishFrontMacro (ConsumeSink + DrawDeferred)
    HostIndirect, ///< Executed: Maxwell3D::RunMacroHleDraw with `hle`
};

/// What kind of draw-like operation the back performs (DrawCheck compares it).
enum class DrawKind : u8 {
    None,
    Array,
    Indexed,
    Indirect,
    IndexedIndirect,
    Clear,
    Texture,
};

inline constexpr const char* draw_kind_names[] = {
    "none", "array", "indexed", "indirect", "indexed indirect", "clear", "texture",
};

/// One draw the front predicts, with the parameters the back's draw manager uses.
struct PredictedDraw {
    DrawKind kind;
    u8 topology; ///< After the topology override (DrawManager::UpdateTopology)
    u16 reserved;
    u32 first;     ///< Vertex or index first (clears: layer count)
    u32 count;     ///< Vertex or index count
    u32 instances; ///< Instance count passed to the rasterizer
    u32 base_vertex;
    u32 base_instance;
};

/// MacroHle::Draw in record form (DrawOp::HostIndirect).
struct HostIndirectDraw {
    u32 kind; ///< MacroHle::Kind
    u32 topology;
    u64 address;
    u64 count_address;
    u32 max_draws;
    u32 stride;
    u32 base_vertex_attribute;
    u32 reserved;
};

inline constexpr size_t MaxPredictedDraws = 2;

/// Draw: emitted by the front before the method (or macro end) that makes the back draw.
/// `dirty_groups` are the V2::RegGroup bits changed since the previous Draw record and
/// `group_hashes` the per-group incremental hashes (GroupTracker) at the draw, `fingerprint`
/// their fold over HashedGroups; all zero (`flags` bit 0 clear) when the front tracks no groups.
struct DrawBody {
    DrawOp op;
    u8 num_draws; ///< Predicted draws (0..MaxPredictedDraws)
    u16 flags;    ///< DrawFlags
    ObjectRef engine;
    u32 method;   ///< The trigger method (Predicted), 0 for macro ends
    u32 argument; ///< Its effective value (after the shadow RAM)
    u32 dirty_groups;
    u32 reserved;
    u64 fingerprint;
    std::array<u64, 9> group_hashes; ///< Hashed groups, in RegGroup order (Pipeline..DrawParams)
    std::array<PredictedDraw, MaxPredictedDraws> draws;
    HostIndirectDraw hle; ///< HostIndirect only
};

/// MASTER_PLAN §5.2's compact Draw/Dispatch record (3 body words, 5 with the header), used
/// when nothing reads the full one (no EDEN_NX2_V2_PARALLEL, IR_DRAWS not verify): the kind it
/// stands for, the DrawOp (MacroEnd is executed), the predicted draw or dispatch count, the
/// dirty register groups and the fingerprint of the group hashes (0 without groups).
struct CompactBody {
    Kind full_kind; ///< Draw or Dispatch
    u8 op;          ///< DrawOp for draws
    u16 count;      ///< Predicted draws, or the maximum dispatches (saturated)
    ObjectRef engine;
    u32 dirty_groups;
    u32 reserved;
    u64 fingerprint;
};

enum DrawFlags : u16 {
    DrawHasGroups = 1 << 0, ///< dirty_groups, group_hashes and fingerprint are valid
};

/// Dispatch: emitted by the front before the method that launches compute work.
struct DispatchBody {
    ObjectRef engine;
    u32 method;   ///< 0xaf legacy launch, 0x107 inline QMD, 0xae-0xb0 QMD signals
    u32 argument;
    u32 dirty_groups; ///< V2::ComputeGroup bits changed since the previous Dispatch record
    u32 min_dispatches;
    u32 max_dispatches; ///< ~0: unknown (QMD dependencies, scheduler launches from memory)
    u64 qmd_address;    ///< GPU address of the QMD, 0 for the legacy launch
    u64 program;        ///< Program address from the inline QMD, 0 when not known
    u64 fingerprint;    ///< Fold of the compute group hashes
    std::array<u64, 4> group_hashes; ///< Hashed compute groups (Pipeline, Launch, Qmd, Pools)
    u64 write_begin; ///< GPU range the dispatch may write, or Unknown
    u64 write_end;
};

enum class CopyOp : u32 {
    Dma,    ///< MaxwellDMA launch_dma
    Blit2D, ///< Fermi2D blit trigger
};

struct CopyBody {
    CopyOp op;
    u32 flags; ///< bit 0: destination is block linear (size unknown)
    u64 source;
    u64 destination;
    u64 bytes; ///< or Unknown
};

enum class ReleaseOp : u32 {
    Payload,  ///< Writes `payload`
    Report,   ///< A GPU counter or timestamp: the value is Unknown
    Reduction,
};

/// Release (executed): rasterizer->Query(address, Payload, bytes == 4 ? IsAFence : HasTimeout,
/// payload, 0) on the bound channel.
struct ReleaseBody {
    u64 address; ///< GPU address
    u64 payload; ///< The value written, or Unknown
    u32 bytes;   ///< 4, 8 or 16; 0 when unknown
    ReleaseOp op;
    u32 address_space; ///< MemoryManager::GetID()
    ObjectRef rasterizer;
};

/// Acquire (informational): the front's decision on a semaphore acquire.
struct AcquireBody {
    u64 address;
    u32 address_space;
    u32 value;
    u32 op;      ///< GpuSemaphoreOperation
    u32 verdict; ///< V2::AcquireVerdict, or ~0 when the old (drained) path decided
};

enum class BarrierOp : u32 {
    WaitForIdle,      ///< rasterizer->WaitForIdle()
    InvalidateCaches, ///< rasterizer->InvalidateGPUCache()
    ReleaseFences,    ///< rasterizer->ReleaseFences()
    SignalReference,  ///< rasterizer->SignalReference()
    TickWork,         ///< gpu->TickWork()
    Invalidate,       ///< rasterizer->OnCacheInvalidation(a, b)
    CommandListEnd,   ///< gpu->FlushCommands(), gpu->OnCommandListEnd()
    SyncFence,        ///< dma_pusher->SignalSyncFence() (a GP entry's sync bit)
    Count,
};

struct BarrierBody {
    BarrierOp op;
    ObjectRef object; ///< The rasterizer, GPU or DMA pusher the op names
    u64 a;
    u64 b;
};
/// BarrierBody::a of a ReleaseFences barrier: release only the fences already signalled (an
/// unmet acquire's suspension), not a forced fence + submit + GPU wait.
inline constexpr u64 ReleaseFencesUnforced = 1;

enum class ReadbackOp : u32 {
    Flush,            ///< rasterizer->FlushRegion(address, bytes) [CPU queue flush]
    FlushCache,       ///< rasterizer->FlushRegion(address, bytes, cache)
    MemoryFlush,      ///< memory->FlushRegion(gpu address, bytes, BufferCache)
    MacroDmaFlush,    ///< memory: flush if dirty, then QueryDma::FlushPending (rasterizer2)
    QueryFlush,       ///< QueryDma::FlushPending(memory, rasterizer2, gpu address, bytes)
    SegmentDirty,     ///< engine->current_dirty = memory2->IsMemoryDirty(gpu address, bytes)
    ListPages,        ///< rasterizer->FlushGpuWrittenCommandListPages(payload device pages)
    WordReadback,     ///< object2 (V2::WordReadback)->Run(rasterizer, device address, bytes)
    /// Engine (Maxwell3D)->RunMacroHleRenderEnable(gpu address, bytes, cache bit 0 invert,
    /// bit 1 S1 entry, bit 2 shadow RAM tracking): macro HLE of a conditional render macro
    /// (Fix F5-03)
    MacroRenderEnable,
    Count,
};

struct ReadbackBody {
    ReadbackOp op;
    ObjectRef object;  ///< Rasterizer, memory manager or engine
    ObjectRef object2; ///< Second object (MacroDmaFlush/QueryFlush: rasterizer, SegmentDirty: memory)
    u32 cache;         ///< VideoCommon::CacheType (FlushCache)
    u64 address;       ///< Device address (Flush, FlushCache) or GPU address
    u64 bytes;
};

enum class MapOp : u32 {
    GmmuMap,       ///< gmmu->Map(gpu, device, size, PITCH)
    GmmuUnmap,     ///< gmmu->Unmap(gpu, size)
    SmmuRelease,   ///< smmu.Unmap(device, size), smmu.Free(device, extra)
    HeapRelease,   ///< mapper->Unmap(cpu, size)
    SparseRelease, ///< gmmu->Unmap(gpu, size), smmu.Unmap(device, size), smmu.Free(device, extra)
    SparseTile,    ///< smmu.Unmap(device, size); cpu != 0: smmu.Map(device, cpu, size, extra)
    Count,
};

/// Map: in the CPU queue `object`/`object2` are pointers (gmmu: Tegra::MemoryManager, smmu:
/// MaxwellDeviceMemoryManager, mapper: NvCore::HeapMapper); in the ring, ObjectTable indices.
struct MapBody {
    MapOp op;
    u32 flags; ///< MapFlags
    u64 gpu_address;
    u64 device_address;
    u64 cpu_address;
    u64 size;
    u64 extra;      ///< Free size (releases) or ASID (SparseTile)
    u64 pending_id; ///< PendingMaps id to remove once done, 0 for none
    u64 object;
    u64 object2;
};

enum MapFlags : u32 {
    MapBeforeLaterReads = 1 << 0, ///< The front waits for it before reading command lists
};

/// Present: a composite request (GPU::RequestComposite). Its layers and fences stay in the
/// GPU's present slots; `slot` names them.
struct PresentBody {
    u64 frame; ///< Composite request number
    u32 slot;
    u32 num_layers;
    u32 num_fences;
    u32 reserved;
};

enum class SyncPointOp : u32 {
    Increment,     ///< rasterizer->SignalSyncPoint(id)
    CommandSignal, ///< The CPU queue's command fence `a` ran (block `b`)
    SignalEvent,   ///< The kernel event `object` (owned by its service) is signalled
    Count,
};

struct SyncPointBody {
    u32 id;    ///< Syncpoint id
    u32 value; ///< Fence value, 0 when the increment is relative
    SyncPointOp op;
    ObjectRef object;
    u64 a;
    u64 b;
};

struct ChannelBody {
    ObjectRef scheduler; ///< Tegra::Control::Scheduler
    ObjectRef gpu;       ///< Tegra::GPU
    s32 bind_id;
    u32 reserved;
};

/// P6a (EDEN_NX2_V2_PARALLEL): the start of resolve segment `id`; its records run up to the next
/// Segment record. The back takes the shards' result for it (V2::Resolve::SplitResolve).
struct SegmentBody {
    u64 id;
    ObjectRef engine;     ///< the Maxwell3D whose draws the segment resolves
    u32 serial;           ///< Resolve::SerialReason known at the cut (informational)
    u32 cut;              ///< Resolve::CutReason
    u32 pending_method;   ///< a write the snapshot holds that follows this record (~0: none)
};

template <typename Body>
inline constexpr size_t BodyWordsOf = (sizeof(Body) + sizeof(u64) - 1) / sizeof(u64);

template <typename Body>
inline constexpr bool IsBody = std::is_trivially_copyable_v<Body> && sizeof(Body) % 8 == 0 &&
                               alignof(Body) <= alignof(u64) && BodyWordsOf<Body> < 256;

static_assert(IsBody<MethodDataBody> && IsBody<EngineBody> && IsBody<DrawBody> &&
              IsBody<DispatchBody> && IsBody<CopyBody> && IsBody<ReleaseBody> &&
              IsBody<AcquireBody> && IsBody<BarrierBody> && IsBody<ReadbackBody> && IsBody<SegmentBody> &&
              IsBody<MapBody> && IsBody<PresentBody> && IsBody<SyncPointBody> &&
              IsBody<ChannelBody> && IsBody<PredictedDraw> && IsBody<HostIndirectDraw> &&
              IsBody<CompactBody>);
static_assert(BodyWordsOf<CompactBody> == 3);

/// A decoded record (for inspection, replay and verification).
struct RecordView {
    Kind kind;
    u64 header;
    u64 sequence;
    const u64* body;
    size_t body_words;
    size_t words;
    bool has_payload;
    PayloadRef payload;

    template <typename Body>
    [[nodiscard]] const Body& As() const {
        static_assert(IsBody<Body>);
        return *reinterpret_cast<const Body*>(body);
    }
};

[[nodiscard]] inline RecordView Decode(const u64* record) {
    const u64 header = record[0];
    const size_t ref = RefWords(header);
    return RecordView{
        .kind = KindOf(header),
        .header = header,
        .sequence = record[1],
        .body = record + HeaderWords + ref,
        .body_words = BodyWords(header),
        .words = RecordWords(header),
        .has_payload = ref != 0,
        .payload = ref != 0 ? UnpackRef(record[HeaderWords]) : PayloadRef{},
    };
}

/// Writes a record of `kind` with `body` (and a payload reference) into `out`, which has room
/// for RecordWordsOf<Body>(with_payload) words. For tests and the emitters.
template <typename Body>
[[nodiscard]] constexpr size_t RecordWordsOf(bool with_payload) {
    return HeaderWords + (with_payload ? 1 : 0) + BodyWordsOf<Body>;
}

/// A point in the IR stream: the record's sequence number and the ring position after it.
/// GpuSplit::WaitFor(Ticket) waits until the back executed it.
struct Ticket {
    u64 sequence{};
    u64 position{};
};

} // namespace VideoCommon::V2::Ir
