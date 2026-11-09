// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/ir.h"
#include "video_core/v2/ir_arena.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/resolve/cow_regs.h"
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

/// What the draw cache needs to compute a draw's shape and partial key from a register file.
struct ShardKeyInputs {
    const u32* regs = nullptr; ///< Maxwell3D register layout
    const GroupTracker* groups = nullptr;
    u8 topology = 0;
    bool indexed = false;
    bool native_turing = false;
    uintptr_t engine = 0;
    uintptr_t memory = 0;
};
struct ShardKey {
    u64 shape = 0;
    u64 partial = 0;    ///< the key before the constant buffer bindings are mixed in
    u32 relevant = 0;   ///< the shape's relevant groups the partial key used
    bool has_shape = false;
    bool has_key = false;
};
/// Set by the Vulkan draw cache (the key code lives with it, shared with the serial path).
using ShardKeyFn = ShardKey (*)(const ShardKeyInputs&);
inline ShardKeyFn g_shard_key = nullptr;

/// The draw cache's shapes, published for the shards (single writer: the resolving thread;
/// lock-free readers through a per-entry sequence lock).
void PublishShape(u64 shape, u32 relevant_groups);
[[nodiscard]] bool FindShape(u64 shape, u32* relevant_groups);
void ClearShapes();

/// A draw resolved by a shard.
struct PreDraw {
    u64 fingerprint = 0; ///< the Draw record's (equals the back's group fingerprint at the draw)
    u64 shape = 0;
    u64 partial = 0;
    uintptr_t engine = 0;
    uintptr_t memory = 0;
    u32 relevant = 0;
    u32 ordinal = 0; ///< draw index in the segment
    bool indexed = false;
    bool has_shape = false;
    bool has_key = false;
    bool drift = false; ///< the shard's registers disagreed with the record's fingerprint

    bool operator==(const PreDraw&) const = default;
};

/// The front's register state at a cut.
struct FrontSnapshot {
    RegisterSnapshot<0xe00> regs;
    RegisterSnapshot<0xe00> shadow;
    GroupTracker groups;
    u32 control = 0;
    bool instance_loop = false;
    bool sink_pending = false;
    bool draw_pending = false;
    bool native_turing = false;
    uintptr_t engine = 0;
    uintptr_t memory = 0;
    u32 engine_ref = 0;
    u32 pending_method = ~0U;
};

struct SegmentInput {
    u64 id = 0;
    u64 begin = 0; ///< ring position after the segment's Segment record
    u64 end = 0;   ///< ring position after the next Segment record
    std::shared_ptr<const FrontSnapshot> snapshot;
    SerialReason serial = SerialReason::None; ///< known up front: no shard resolves it
    const u64* ring = nullptr;
    const Ir::PayloadArena* arena = nullptr;
};

struct SplitTraits {
    using Input = SegmentInput;
    using Record = PreDraw;
    struct Shard {
        std::unique_ptr<RegSem::Mirror3D> mirror;
    };
    static SerialReason Resolve(Shard& shard, const Input& input, std::vector<Record>& records,
                                Commit::RequestList& requests);
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
        u32 control;
        bool instance_loop;
        bool sink_pending;
        bool draw_pending;
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
    void OnSegment(const Ir::SegmentBody& body, const u32* back_regs);
    /// The shard's record for the draw the back is about to decide (null: none matches).
    [[nodiscard]] const PreDraw* TakeDraw(u64 fingerprint, bool indexed);
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
    bool open = false;
    SegmentInput open_input;
    std::deque<SegmentInput> closed;
    u32 applied_size = 0;

    // Back.
    bool holding = false;
    u64 held_id = 0;
    Resolver::Result* current = nullptr;
    size_t cursor = 0;
    AdaptiveSize adaptive;
    u64 last_report_ns = 0;

    // Front <- back.
    std::atomic<u32> adaptive_size{SegmentPolicy::DefaultK};
};

} // namespace VideoCommon::V2::Resolve
