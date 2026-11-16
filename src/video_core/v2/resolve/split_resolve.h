// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/ir.h"
#include "video_core/v2/ir_arena.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/resolve/cbuf_table.h"
#include "video_core/v2/resolve/cow_regs.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/resolve/parallel_resolve.h"
#include "video_core/v2/resolve/segment.h"

namespace VideoCommon {
class GpuSplit;
}
namespace VideoCommon::V2 {
struct DrawOutput;
}
namespace VideoCommon::V2::RegSem {
struct Mirror3D;
}
namespace VideoCommon::V2::Resolve {
class SnapshotView;
}

// Pipeline v2 P6a in the split (EDEN_NX2_V2_PARALLEL): the segmenter on the front (S1), shards
// that resolve segments from the typed IR ring, and the back as the commit thread.
//
//  - Front: at a draw the front's register mirror (REGSEM=1) decides a cut. A cut snapshots the
//    mirror (copy-on-write blocks, group hashes) and sends a Segment record. A closed segment is
//    handed to the shards once the ring holds all of it (at the next publish).
//  - Shards: start from the snapshot, apply the segment's StateRun/MethodData records with the
//    same table-driven register update the front uses (RegSem::Write3D), check their group hashes
//    against every Draw record's fingerprint, and resolve each draw's pure part: the P4 draw-cache
//    shape and its key without the constant buffer bindings (the vertex layout and the relevant
//    group fingerprints). That needs only registers, the published shape table and no cache.
//  - Back (commit): at a Segment record it takes the segment's result in order (waiting only for
//    a shard already running it, else stealing it), and each draw's DecideDraw takes the matching
//    record instead of hashing the state itself. Verify recomputes and compares.
//
// What the shards can resolve is limited by what is pure today: see IMPLEMENTATION_LOG "P6" for
// why the rest of the resolve (the P5a pass 2 encode-side list and the live engine reads) is
// still on the commit thread.
namespace VideoCommon::V2::Resolve {

/// What the draw cache needs to compute a draw's shape and key from a register file.
struct ShardKeyInputs {
    const u32* regs = nullptr; ///< Maxwell3D register layout
    const GroupTracker* groups = nullptr;
    const CbufTable* cbufs = nullptr; ///< P6c: the bindings, for the full key
    u8 topology = 0;
    bool indexed = false;
    bool native_turing = false;
    uintptr_t engine = 0;
    uintptr_t memory = 0;
    /// The draw's pipeline key hash (the shape's pipeline part); without it no shape.
    u64 state_hash = 0;
    bool has_state_hash = false;
};
struct ShardKey {
    u64 shape = 0;
    u64 partial = 0;    ///< the key before the constant buffer bindings are mixed in
    u64 full = 0;       ///< P6c: the whole draw-cache key (the bindings mixed in)
    u64 cbuf_masks = 0; ///< P6c: hash of the shape's key constant buffer slots the full key used
    u32 relevant = 0;   ///< the shape's relevant groups the partial key used
    bool has_shape = false;
    bool has_key = false;
    bool has_full = false;
};
/// Set by the Vulkan draw cache (the key code lives with it, shared with the serial path).
using ShardKeyFn = ShardKey (*)(const ShardKeyInputs&);
inline ShardKeyFn g_shard_key = nullptr;

/// The draw cache's shapes, published for the shards (single writer: the resolving thread;
/// lock-free readers through a per-entry sequence lock).
void PublishShape(u64 shape, u32 relevant_groups);
[[nodiscard]] bool FindShape(u64 shape, u32* relevant_groups);
/// P6c: with the shape's key constant buffer slots per stage (DrawCache::ShapeInfo::key_cbufs).
void PublishShape(u64 shape, u32 relevant_groups, const std::array<u32, CbufStages>& key_cbufs);
[[nodiscard]] bool FindShape(u64 shape, u32* relevant_groups,
                             std::array<u32, CbufStages>* key_cbufs);
void ClearShapes();
/// The shape table's publish generation (a sequence lock: odd during PublishShape and
/// ClearShapes, moves twice at each).
[[nodiscard]] u64 ShapeGeneration();
/// After reading the table with `before` from ShapeGeneration(): no publish overlapped it.
[[nodiscard]] bool ShapeGenerationHolds(u64 before);
/// A table generation read around a lookup that a publish overlapped: never equal to another.
inline constexpr u64 UnstableGeneration = 0;
/// The hash of a shape's key constant buffer slots (the commit checks the full key used these).
[[nodiscard]] u64 CbufMasksHash(const std::array<u32, CbufStages>& key_cbufs);

/// P6c: a draw's pipeline key computed by a shard (the Vulkan FixedPipelineState, opaque here).
/// Plain data (F6-01, §5.2): it lives in its segment's arena, records point at it.
inline constexpr size_t MaxPipelineKeyBytes = 1024;
struct PipelineKey {
    u64 hash = 0;   ///< FixedPipelineState::Hash
    u64 inputs = 0; ///< what it was computed from (shards reuse it while that holds; 0: never)
    u32 size = 0;   ///< bytes used (sizeof(FixedPipelineState); 0: no key)
    alignas(16) std::array<u8, MaxPipelineKeyBytes> bytes{};
    [[nodiscard]] bool SameKey(const PipelineKey& other) const {
        return hash == other.hash && size == other.size &&
               std::memcmp(bytes.data(), other.bytes.data(), size) == 0;
    }
};
/// Set by the Vulkan pipeline cache (the key code lives with it, shared with the serial path):
/// what a draw's key is computed from (0: not reusable), and the key itself.
using PipelineKeyInputsFn = u64 (*)(const SnapshotView& view);
using PipelineKeyFn = void (*)(const SnapshotView& view, u64 inputs, PipelineKey& out);
inline PipelineKeyInputsFn g_pipeline_key_inputs = nullptr;
inline PipelineKeyFn g_pipeline_key = nullptr;

/// R3 (S0-02): the graphics programs' shader hashes the commit thread found, published for the
/// shards by (memory manager, program GPU address). Single writer (the commit thread), lock-free
/// readers inside an epoch section of Commit::Epochs() (or the writer). Insert-only (F6-X1): an
/// address's hash is replaced only by a newer hash for that address. The commit compares the
/// hashes a shard used with its own.
void PublishProgram(uintptr_t memory, GPUVAddr address, u64 unique_hash);
[[nodiscard]] bool FindProgram(uintptr_t memory, GPUVAddr address, u64* unique_hash);
void ClearPrograms();
/// A sequence lock as ShapeGeneration's, over PublishProgram and ClearPrograms.
[[nodiscard]] u64 ProgramGeneration();
[[nodiscard]] bool ProgramGenerationHolds(u64 before);

/// §5.6.3: the commit's applier for a shard's request (the slot for the request's proxy), set
/// by the rasterizer that owns the caches the requests mutate.
using ApplyRequestFn = u32 (*)(const Commit::Request& request);
inline ApplyRequestFn g_apply_request = nullptr;

struct PreDraw;
/// R3: the draw's pipeline looked up on the shard from its key (set by the Vulkan pipeline
/// cache, which owns the lookup tables; null: no shard lookup).
using ShardPipelineFn = void (*)(const SnapshotView& view, uintptr_t memory, PreDraw& pre);
inline ShardPipelineFn g_shard_pipeline = nullptr;

struct DynamicState;
struct DynamicFeatures;
/// R5: the device features the record path computes dynamic state with, published by the
/// commit thread before its draws use them (a copy is kept per change; shards compute with the
/// current one). Returns the generation `features` have.
u64 PublishDynamicFeatures(const DynamicFeatures& features);
/// R5: what a draw's dynamic state is computed from on a shard's view with the published
/// features (0: none published yet, or EDEN_NX2_V2_DRAWREC off; a result is reused only while
/// the view's groups are valid). `generation`: the features' generation; `features`: them
/// (valid for the process).
[[nodiscard]] u64 ShardDynamicInputs(const SnapshotView& view, u64& generation,
                                     const DynamicFeatures** features);

/// A draw resolved by a shard.
struct PreDraw {
    u64 fingerprint = 0; ///< the Draw record's (equals the back's group fingerprint at the draw)
    u64 shape = 0;
    u64 partial = 0;
    u64 full = 0;       ///< P6c: the whole draw-cache key
    u64 cbuf_masks = 0; ///< P6c: CbufMasksHash of the slots `full` used
    u64 shape_generation = 0; ///< ShapeGeneration() before the key read the shape table
    /// P6c (null: none computed). F6-01: in the segment's arena (records are plain data, no
    /// reference counts); valid until the commit releases the segment.
    const PipelineKey* pipeline_key = nullptr;
    /// R3: the pipeline the shard's lookup found for (hashes, pipeline_key) (an opaque
    /// GraphicsPipeline*, never dereferenced off the commit thread), valid while the lookup's
    /// generation is `lookup_generation`.
    const void* pipeline = nullptr;
    u64 lookup_generation = 0;
    /// The lookup's Serial() around FindShared, UnstableGeneration when an insert overlapped
    /// (verify: dependent differences).
    u64 lookup_serial = 0;
    u64 program_generation = 0; ///< ProgramGeneration() around the hash reads (or unstable)
    std::array<u64, 6> hashes{};
    bool has_pipeline = false;
    /// R5: the draw's dynamic state (ComputeDynamic on the shard's view; null: none), computed
    /// with the published features of `dynamic_generation`; in the segment's arena as the key.
    const DynamicState* dynamic = nullptr;
    u64 dynamic_generation = 0;
    uintptr_t engine = 0;
    uintptr_t memory = 0;
    u32 relevant = 0;
    u32 ordinal = 0; ///< draw index in the segment
    bool indexed = false;
    bool has_shape = false;
    bool has_key = false;
    bool has_full = false;
    bool drift = false; ///< the shard's registers disagreed with the record's fingerprint
};

/// The front's register state at a cut.
struct FrontSnapshot {
    RegisterSnapshot<0xe00> regs;
    RegisterSnapshot<0xe00> shadow;
    GroupTracker groups;
    const CbufTable* cbufs = nullptr; ///< P6c: shared while the bindings don't change
    u32 control = 0;
    /// F6-X1 (R2-F1-02): render_enable.mode (0x556) is the back's here, not the front's: a
    /// MACROHLE conditional render was sent as a Readback the back evaluates (S3, in ring order,
    /// before every draw after it), and the front's copy is stale until it takes the back's.
    /// No shard resolve reads the register (Misc, unhashed); the snapshot compare skips it.
    bool render_enable_unknown = false;
    bool instance_loop = false;
    bool sink_pending = false;
    bool draw_pending = false;
    bool native_turing = false;
    uintptr_t engine = 0;
    uintptr_t memory = 0;
    u32 engine_ref = 0;
    u32 pending_method = ~0U;
};

/// A segment submitted once its start was published (AP6 §1.3(a)): the shard reads its records
/// as the front publishes them, up to `end`, which the front sets when it cuts.
struct SegmentProgress {
    std::atomic<u64> end{~u64{0}};
    std::atomic<u8> serial{0};        ///< SerialReason found after submission (set before end)
    std::atomic<bool> waiting{false}; ///< the shard waits for the front to publish more
    std::atomic<bool> abandon{false}; ///< the commit is done with the segment, or the split stops
    /// F1-36 (R0-d): the shard's records as it resolves them: `records` (its reserved storage,
    /// which never moves while streaming) holds `ready` finished ones (release per record).
    std::atomic<const PreDraw*> records{nullptr};
    std::atomic<u32> ready{0};

    void Reset() {
        end.store(~u64{0}, std::memory_order_relaxed);
        serial.store(0, std::memory_order_relaxed);
        waiting.store(false, std::memory_order_relaxed);
        abandon.store(false, std::memory_order_relaxed);
        records.store(nullptr, std::memory_order_relaxed);
        ready.store(0, std::memory_order_relaxed);
    }
};
/// Records an early segment's shard streams at most (the storage is reserved up front; draws
/// past it resolve on the commit).
inline constexpr size_t StreamRecords = 2 * SegmentPolicy::MaxK;

struct SegmentInput {
    u64 id = 0;
    u64 begin = 0; ///< ring position after the segment's Segment record
    u64 end = 0;   ///< ring position after the next Segment record (closed segments)
    SegmentProgress* progress = nullptr; ///< submitted before its end was published
    const std::atomic<u64>* published = nullptr; ///< with `progress`: the split's publish position
    /// From the front's epoch pools (F1-30): valid until the commit released this segment.
    const FrontSnapshot* snapshot = nullptr;
    SerialReason serial = SerialReason::None; ///< known up front: no shard resolves it
    const u64* ring = nullptr;
    const Ir::PayloadArena* arena = nullptr;
    u32 draws = 0; ///< draws the front counted in it (the in-flight draw budget)
};

/// Draws the shards may have in flight (AP6 T0: depth in draws, not segments): a segment waits
/// in the producer's queue while the submitted ones hold this many (EDEN_NX2_V2_PARALLEL_DEPTH
/// bounds the slots).
inline constexpr u32 InFlightDraws = 128;
/// Draws an open segment holds before it is submitted with only its start published (F1-36:
/// the commit reached every serial segment before a shard started it; a shard that waits for
/// the front costs an idle core, not the commit).
inline constexpr u32 EarlySubmitDraws = 1;

struct SplitTraits {
    using Input = SegmentInput;
    using Record = PreDraw;
    struct Shard {
        std::unique_ptr<RegSem::Mirror3D> mirror;
        u64 wait_ns = 0; ///< time spent waiting for the front (not busy, F6-01)
        /// The shard's last key and dynamic state, copied into the next segment's arena while
        /// their inputs hold (a segment's records point only into its own arena).
        std::unique_ptr<PipelineKey> last_key;
        std::unique_ptr<DynamicState> last_dynamic;
        u64 last_dynamic_inputs = 0;
    };
    /// What a segment's records point into.
    struct Arena {
        StableArena<PipelineKey> keys;
        StableArena<DynamicState> dynamics;
        void Clear() {
            keys.Clear();
            dynamics.Clear();
        }
    };
    static SerialReason Resolve(Shard& shard, const Input& input, std::vector<Record>& records,
                                Commit::RequestList& requests, Arena& arena);
    /// Verify's serial resolve of a segment: an early segment whose end the front set reads as a
    /// closed one (its progress stays the shard's).
    static Input VerifyInput(const Input& input) {
        Input copy = input;
        if (input.progress != nullptr) {
            const u64 end = input.progress->end.load(std::memory_order_acquire);
            if (end != ~u64{0}) {
                copy.end = end;
                copy.progress = nullptr;
                copy.published = nullptr;
            }
        }
        return copy;
    }
    static void Patch(Record&, const Commit::RequestList&) {}
    static const char* Compare(const Record& a, const Record& b);
};

class SplitResolve {
public:
    /// The front mirror as the front holds it (Maxwell3D::FrontState with REGSEM=1).
    struct FrontView {
        const u32* regs;
        const u32* shadow;
        const GroupTracker* groups;
        const CbufTable* cbufs;
        u32 control;
        bool instance_loop;
        bool sink_pending;
        bool draw_pending;
        /// F6-X1: FrontState::v2_render_enable_unknown (FrontSnapshot::render_enable_unknown).
        bool render_enable_unknown = false;
        /// The writer's dirty-block mask (null: compare every block); cleared by the snapshot.
        u64* dirty_blocks = nullptr;
        /// S1: the shards apply plain deltas and read no shadow RAM (none is snapshotted).
        bool deltas = false;
    };
    /// The engine a draw belongs to.
    struct EngineInfo {
        u32 ref;            ///< object table index (Draw/Engine records)
        uintptr_t engine;   ///< Maxwell3D*
        uintptr_t memory;   ///< its MemoryManager*
        bool native_turing;
    };

    SplitResolve(GpuSplit& split, const u64* ring, const Ir::PayloadArena* arena);
    ~SplitResolve();

    // Front thread.
    /// Before the Draw record(s) for `out`: a cut when the segmenter says so (only where the
    /// mirror is consistent with the ring: `may_cut`). `pending_method`: a write the mirror holds
    /// that the ring gets after the record (~0 if none).
    void FrontDraw(const FrontView& view, const EngineInfo& engine, const DrawOutput& out,
                   u32 pending_method, bool may_cut);
    /// Every record the front sends (serial classification).
    void OnRecord(Ir::Kind kind) {
        if (kind == Ir::Kind::Copy) {
            segmenter.OnRecord(RecordClass::Transfer);
        } else if (kind == Ir::Kind::Readback) {
            segmenter.OnRecord(RecordClass::Readback);
        } else if (kind == Ir::Kind::Map) {
            segmenter.OnRecord(RecordClass::Map);
        }
    }
    void OnBarrier(u32 op) {
        if (op == static_cast<u32>(Ir::BarrierOp::InvalidateCaches) ||
            op == static_cast<u32>(Ir::BarrierOp::Invalidate)) {
            segmenter.OnRecord(RecordClass::CacheInvalidate);
        }
    }
    /// After a publish up to `position`: hands closed segments the ring now holds to the shards.
    void OnPublish(u64 position);

    // Back thread (the commit thread).
    /// `back_regs` (method sink applied), `back_shadow`, `back_cbufs`: the back engine's state
    /// (verify only, else null).
    void OnSegment(const Ir::SegmentBody& body, const u32* back_regs, const u32* back_shadow,
                   const CbufTable* back_cbufs);
    /// F6-01: the back reached `count` predicted draws of `engine` (an S1 Draw entry or a
    /// Predicted/HostIndirect Draw record), in ring order, whether or not they will be drawn
    /// (MACROHLE conditional render predicates some off). Their ordinals follow the previous
    /// ones; `usable`: shard records may stand for them (not for a host indirect draw).
    void NoteDraws(uintptr_t engine, u32 count, bool usable);
    /// The shard's record for the draw the back is about to decide (null: none matches): the
    /// one with the next noted ordinal (§5.6.3: records pair with draws in sequence order), if
    /// its fingerprint and kind are the draw's.
    [[nodiscard]] const PreDraw* TakeDraw(u64 fingerprint, bool indexed);
    /// P6c: the record DecideDraw accepted for the draw being resolved (engine, memory and
    /// engine hint checked), for the later resolve steps of the same draw. Cleared when the
    /// draw's pipeline is chosen (EndDraw), so a step of a later draw never sees it.
    void SetTaken(const PreDraw* pre) {
        taken = pre;
        record_taken = pre;
    }
    /// R5: the accepted record for the draw's record resolve (after its pipeline is chosen),
    /// once. Valid during the draw it was taken for (the next draw's SetTaken replaces it).
    [[nodiscard]] const PreDraw* TakeRecordInputs() {
        return std::exchange(record_taken, nullptr);
    }
    void EndDraw() {
        taken = nullptr;
    }
    /// The shard's pipeline key for the draw being resolved, once (null: none).
    [[nodiscard]] const PipelineKey* TakePipelineKey() {
        const PreDraw* const pre = TakePipelineRecord();
        return pre != nullptr ? pre->pipeline_key : nullptr;
    }
    /// R3: the shard's record for the pipeline step (key and lookup), once (null: none).
    [[nodiscard]] const PreDraw* TakePipelineRecord() {
        return std::exchange(taken, nullptr);
    }
    [[nodiscard]] bool Verify() const {
        return verify;
    }

    /// The commit thread's resolver while the split's back runs (null elsewhere).
    static inline thread_local SplitResolve* t_commit = nullptr;

private:
    using Resolver = ParallelResolver<SplitTraits>;

    GpuSplit& split;
    const u64* ring;
    const Ir::PayloadArena* arena;
    bool verify = false;
    bool stats = false;
    std::unique_ptr<Resolver> resolver;

    // Front.
    Segmenter segmenter;
    SnapshotBuilder<0xe00> regs_builder;
    SnapshotBuilder<0xe00> shadow_builder;
    /// F1-30: what segments point to, reused once the commit released them (`released`).
    EpochPool<FrontSnapshot> snapshot_pool;
    EpochPool<CbufTable> cbuf_pool;
    EpochPool<SegmentProgress> progress_pool;
    CbufTable* last_cbufs = nullptr;
    u64 last_cbuf_changes = ~u64{0};
    bool open = false;
    SegmentInput open_input;
    /// The open segment's progress once it was submitted early (null: not submitted yet).
    SegmentProgress* open_progress = nullptr;
    /// The last early submission, released at shutdown if its end was never published.
    SegmentProgress* last_early = nullptr;
    void ReclaimReleased();
    /// F1-36: closed segments waiting for a slot, submitted by whichever thread gets there
    /// first: the front at a publish, or the commit when it releases a slot (a front stalled in
    /// a wait used to leave them unsubmitted until the commit stole them). The front pushes at
    /// `closed_tail`; the submitter (one at a time, `submitting`) pops at `closed_head`.
    static constexpr u64 ClosedCapacity = 1024;
    std::unique_ptr<std::array<SegmentInput, ClosedCapacity>> closed;
    alignas(64) std::atomic<u64> closed_head{0};
    alignas(64) std::atomic<u64> closed_tail{0};
    std::atomic<bool> submitting{false};
    std::atomic<bool> drain_again{false};
    /// Either thread: submits closed segments whose end is at or before `published`.
    void DrainClosed(u64 published);
    [[nodiscard]] bool ClosedEmpty() const {
        return closed_head.load(std::memory_order_acquire) ==
               closed_tail.load(std::memory_order_acquire);
    }
    u32 applied_size = 0;
    /// Draws of submitted segments the commit has not taken yet (front adds, back subtracts).
    std::atomic<u32> in_flight_draws{0};

    // Back.
    bool holding = false;
    u64 held_id = 0;
    u64 held_start_ns = 0; ///< when the commit took the held segment (the adaptive size)
    Resolver::Result* current = nullptr;
    /// F1-36: the held segment's progress while its shard streams records (null otherwise).
    SegmentProgress* stream = nullptr;
    size_t cursor = 0;
    /// F6-01: ordinals of the current segment's draws as the back reaches them (NoteDraws).
    uintptr_t segment_engine = 0;
    u32 next_ordinal = 0;
    u32 record_base = 0;
    u32 record_count = 0;
    u32 record_used = 0;
    /// The draws of the last noted entry the back never drew (predicated off): counted.
    void NotePredicatedOff();
    const PreDraw* taken = nullptr;
    const PreDraw* record_taken = nullptr;
    AdaptiveSize adaptive;
    u64 last_report_ns = 0;

    // Front <- back.
    std::atomic<u32> adaptive_size{SegmentPolicy::DefaultK};
    /// The newest segment id the commit released: nothing reads its inputs (or older ones).
    std::atomic<u64> released{0};
};

} // namespace VideoCommon::V2::Resolve
