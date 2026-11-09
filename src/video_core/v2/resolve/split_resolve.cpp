// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstring>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/v2/draw_model.h"
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/resolve/split_resolve.h"
#include "video_core/v2/v2_stats.h"

namespace VideoCommon::V2::Resolve {

// ---- Published shapes -----------------------------------------------------------------------------

namespace {

struct ShapeEntry {
    std::atomic<u64> sequence{};
    std::atomic<u64> shape{};
    std::atomic<u32> relevant{};
};
constexpr size_t ShapeEntries = 4096;

std::array<ShapeEntry, ShapeEntries>& Shapes() {
    static std::array<ShapeEntry, ShapeEntries> entries;
    return entries;
}

size_t ShapeSlot(u64 shape) {
    return static_cast<size_t>((shape * 0x9e3779b97f4a7c15ULL) >> 52) % ShapeEntries;
}

} // Anonymous namespace

void PublishShape(u64 shape, u32 relevant_groups) {
    ShapeEntry& entry = Shapes()[ShapeSlot(shape)];
    const u64 sequence = entry.sequence.load(std::memory_order_relaxed);
    entry.sequence.store(sequence + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    entry.shape.store(shape, std::memory_order_relaxed);
    entry.relevant.store(relevant_groups, std::memory_order_relaxed);
    entry.sequence.store(sequence + 2, std::memory_order_release);
}

bool FindShape(u64 shape, u32* relevant_groups) {
    const ShapeEntry& entry = Shapes()[ShapeSlot(shape)];
    for (u32 attempt = 0; attempt < 16; ++attempt) {
        const u64 before = entry.sequence.load(std::memory_order_acquire);
        if ((before & 1) != 0) {
            CpuRelax();
            continue;
        }
        const u64 stored = entry.shape.load(std::memory_order_relaxed);
        const u32 relevant = entry.relevant.load(std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_acquire);
        if (entry.sequence.load(std::memory_order_relaxed) != before) {
            continue;
        }
        if (stored != shape || stored == 0) {
            return false;
        }
        *relevant_groups = relevant;
        return true;
    }
    return false;
}

void ClearShapes() {
    for (ShapeEntry& entry : Shapes()) {
        const u64 sequence = entry.sequence.load(std::memory_order_relaxed);
        entry.sequence.store(sequence + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        entry.shape.store(0, std::memory_order_relaxed);
        entry.relevant.store(0, std::memory_order_relaxed);
        entry.sequence.store(sequence + 2, std::memory_order_release);
    }
}

// ---- Shards ---------------------------------------------------------------------------------------

namespace {

constexpr u64 RingKindMask = 0xf;
constexpr u64 RingWrap = 0;
constexpr u64 RingMethod = 2;
constexpr u64 MethodHasSegment = 1 << 5;

void LoadMirror(RegSem::Mirror3D& mirror, const FrontSnapshot& snapshot) {
    snapshot.regs.CopyTo(mirror.regs.data());
    snapshot.shadow.CopyTo(mirror.shadow.data());
    mirror.control = static_cast<RegSem::ShadowRamControl>(snapshot.control);
    mirror.instance_loop = snapshot.instance_loop;
    mirror.sink_pending = snapshot.sink_pending;
    mirror.draw_pending = snapshot.draw_pending;
    mirror.skip_same = false; // the ring holds what the back applies: apply all of it
    mirror.v2_groups = snapshot.groups;
}

} // Anonymous namespace

SerialReason SplitTraits::Resolve(Shard& shard, const Input& input, std::vector<Record>& records,
                                  Commit::RequestList& requests) {
    static_cast<void>(requests); // the shard's lookups are pure and need no mutation
    if (input.snapshot == nullptr || input.ring == nullptr || g_shard_key == nullptr) {
        return SerialReason::NoSnapshot;
    }
    if (!shard.mirror) {
        shard.mirror = std::make_unique<RegSem::Mirror3D>();
    }
    RegSem::Mirror3D& mirror = *shard.mirror;
    const FrontSnapshot& snapshot = *input.snapshot;
    LoadMirror(mirror, snapshot);
    const RegSem::Table3D& table = RegSem::GetTable3D();
    const auto no_effects = [](RegSem::Effect, u32, u32) {};
    constexpr u64 RingWords = GpuSplit::RingWords;
    bool active = true;
    u32 ordinal = 0;
    for (u64 position = input.begin; position < input.end;) {
        const size_t index = static_cast<size_t>(position & (RingWords - 1));
        const u64 head = input.ring[index];
        const u64 ring_kind = head & RingKindMask;
        if (ring_kind == RingWrap) {
            position += RingWords - index;
            continue;
        }
        if (ring_kind != Ir::RingKind) {
            return SerialReason::NoSnapshot; // the word stream: nothing typed to read
        }
        const Ir::RecordView record = Ir::Decode(&input.ring[index]);
        position += record.words;
        switch (record.kind) {
        case Ir::Kind::Engine:
            active = record.As<Ir::EngineBody>().engine == snapshot.engine_ref;
            break;
        case Ir::Kind::StateRun: {
            if (!active || !record.has_payload) {
                break;
            }
            const u64* it = input.arena->At(record.payload.index);
            const u64* const end = it + record.payload.words;
            while (it < end) {
                const u64 word = *it;
                const size_t words = (word & MethodHasSegment) != 0 ? 2 : 1;
                if ((word & RingKindMask) == RingMethod) {
                    const u32 method = static_cast<u32>((word >> 8) & 0xffff);
                    const u32 argument = static_cast<u32>(word >> 32);
                    if (method < RegSem::Table3D::NumRegs) {
                        RegSem::Write3D(mirror, table, method, argument, false, no_effects);
                    }
                }
                it += words;
            }
            break;
        }
        case Ir::Kind::MethodData: {
            if (!active || !record.has_payload) {
                break;
            }
            const auto& body = record.As<Ir::MethodDataBody>();
            if (body.method < RegSem::Table3D::NumRegs) {
                RegSem::WriteMulti3D(
                    mirror, table, body.method,
                    reinterpret_cast<const u32*>(input.arena->At(record.payload.index)),
                    body.amount, no_effects);
            }
            break;
        }
        case Ir::Kind::Draw: {
            const auto& body = record.As<Ir::DrawBody>();
            if (body.engine != snapshot.engine_ref || body.op == Ir::DrawOp::HostIndirect) {
                break;
            }
            // The record's hashes are the back's at the draw: either these registers, or these
            // plus the trigger write the record precedes (a draw method the mirror took first).
            GroupTracker groups = mirror.v2_groups;
            bool drift = (body.flags & Ir::DrawHasGroups) == 0;
            if (!drift && groups.Fingerprint(HashedGroups) != body.fingerprint) {
                if (body.method < RegSem::Table3D::NumRegs &&
                    mirror.regs[body.method] != body.argument) {
                    groups.Change(table.groups[body.method], body.method,
                                  mirror.regs[body.method], body.argument);
                }
                drift = groups.Fingerprint(HashedGroups) != body.fingerprint;
            }
            for (u32 i = 0; i < body.num_draws; ++i) {
                const Ir::PredictedDraw& draw = body.draws[i];
                if (draw.kind != Ir::DrawKind::Array && draw.kind != Ir::DrawKind::Indexed) {
                    continue;
                }
                PreDraw pre;
                pre.fingerprint = body.fingerprint;
                pre.indexed = draw.kind == Ir::DrawKind::Indexed;
                pre.ordinal = ordinal++;
                pre.engine = snapshot.engine;
                pre.memory = snapshot.memory;
                pre.drift = drift;
                if (!drift) {
                    const ShardKey key = g_shard_key(ShardKeyInputs{
                        .regs = mirror.regs.data(),
                        .groups = &groups,
                        .topology = draw.topology,
                        .indexed = pre.indexed,
                        .native_turing = snapshot.native_turing,
                        .engine = snapshot.engine,
                        .memory = snapshot.memory,
                    });
                    pre.shape = key.shape;
                    pre.partial = key.partial;
                    pre.relevant = key.relevant;
                    pre.has_shape = key.has_shape;
                    pre.has_key = key.has_key;
                }
                records.push_back(pre);
            }
            break;
        }
        default:
            break;
        }
    }
    return SerialReason::None;
}

const char* SplitTraits::Compare(const Record& a, const Record& b) {
    if (a.fingerprint != b.fingerprint || a.indexed != b.indexed || a.ordinal != b.ordinal) {
        return "draw";
    }
    if (a.drift != b.drift) {
        return "drift";
    }
    if (a.has_shape != b.has_shape || a.shape != b.shape) {
        return "shape";
    }
    if (a.has_key != b.has_key || a.partial != b.partial || a.relevant != b.relevant) {
        return "key";
    }
    return nullptr;
}

// ---- SplitResolve ---------------------------------------------------------------------------------

SplitResolve::SplitResolve(GpuSplit& split_, const u64* ring_, const Ir::PayloadArena* arena_)
    : split{split_}, ring{ring_}, arena{arena_}, segmenter{PolicyFromEnv()} {
    verify = V2::Verify(Switch::Parallel);
    stats = V2::Stats::Enabled();
    Resolver::Config config;
    config.shards = ShardsFromEnv();
    config.depth = DepthFromEnv();
    config.verify = verify;
    config.resolve_stolen = false; // a stolen segment's draws decide on the commit thread live
    config.stats = true;           // the commit stats are cheap and feed the adaptive size
    resolver = std::make_unique<Resolver>(config, [](u32) { return SplitTraits::Shard{}; });
    segmenter.Reset(1);
    applied_size = segmenter.Size();
    adaptive_size.store(applied_size, std::memory_order_relaxed);
    GetStats().segment_size.store(applied_size, std::memory_order_relaxed);
    LOG_INFO(HW_GPU,
             "EDEN_NX2_V2_PARALLEL: {} shards, {} segments in flight, segment policy {} (K {}){}",
             config.shards, config.depth,
             segmenter.Policy().kind == SegmentPolicy::Kind::Pass    ? "pass"
             : segmenter.Policy().kind == SegmentPolicy::Kind::Fixed ? "fixed"
                                                                     : "adaptive",
             segmenter.Size(), verify ? ", verify" : "");
}

SplitResolve::~SplitResolve() {
    resolver.reset(); // joins the shards before the ring goes away
}

void SplitResolve::FrontDraw(const FrontView& view, const EngineInfo& engine,
                             const DrawOutput& out, u32 pending_method, bool may_cut) {
    const u32 size = adaptive_size.load(std::memory_order_relaxed);
    if (size != applied_size) {
        segmenter.SetSize(size);
        applied_size = size;
    }
    for (u32 i = 0; i < out.count; ++i) {
        const Ir::PredictedDraw& draw = out.draws[i];
        DrawEvent event;
        event.dirty_groups = i == 0 ? view.groups->dirty : 0;
        event.engine = engine.ref;
        event.indirect =
            draw.kind == Ir::DrawKind::Indirect || draw.kind == Ir::DrawKind::IndexedIndirect;
        const SerialReason closing_reason = segmenter.OpenReason();
        const CutReason cut = may_cut ? segmenter.OnDraw(event) : CutReason::None;
        if (!may_cut) {
            // Inside a multi-method run the mirror is mid-update: the draw joins the open
            // segment (the segmenter still classifies it).
            if (!segmenter.Started()) {
                continue;
            }
            if (event.indirect) {
                segmenter.MarkSerial(SerialReason::IndirectCount);
            }
            continue;
        }
        if (cut == CutReason::None) {
            continue;
        }
        if (stats) {
            Count(GetStats().cuts[static_cast<size_t>(cut)]);
        }
        auto snapshot = std::make_shared<FrontSnapshot>();
        const u64 copied_before = regs_builder.copied + shadow_builder.copied;
        const u64 shared_before = regs_builder.shared + shadow_builder.shared;
        snapshot->regs = regs_builder.Take(view.regs);
        snapshot->shadow = shadow_builder.Take(view.shadow);
        if (stats) {
            Count(GetStats().snapshot_blocks_copied,
                  regs_builder.copied + shadow_builder.copied - copied_before);
            Count(GetStats().snapshot_blocks_shared,
                  regs_builder.shared + shadow_builder.shared - shared_before);
        }
        snapshot->groups = *view.groups;
        snapshot->control = view.control;
        snapshot->instance_loop = view.instance_loop;
        snapshot->sink_pending = view.sink_pending;
        snapshot->draw_pending = view.draw_pending;
        snapshot->native_turing = engine.native_turing;
        snapshot->engine = engine.engine;
        snapshot->memory = engine.memory;
        snapshot->engine_ref = engine.ref;
        snapshot->pending_method = pending_method;
        const Ir::SegmentBody body{
            .id = segmenter.OpenId(),
            .engine = engine.ref,
            .serial = static_cast<u32>(segmenter.OpenReason()),
            .cut = static_cast<u32>(cut),
            .pending_method = pending_method,
        };
        split.Note(Ir::Kind::Segment, body);
        const u64 position = split.WritePosition();
        if (open) {
            open_input.end = position;
            open_input.serial = closing_reason;
            closed.push_back(std::move(open_input));
        }
        open = true;
        open_input = SegmentInput{};
        open_input.id = body.id;
        open_input.begin = position;
        open_input.snapshot = std::move(snapshot);
        open_input.ring = ring;
        open_input.arena = arena;
    }
}

void SplitResolve::OnPublish(u64 position) {
    while (!closed.empty() && closed.front().end <= position) {
        SegmentInput input = std::move(closed.front());
        closed.pop_front();
        const u64 id = input.id;
        const SerialReason reason = input.serial;
        resolver->Submit(id, std::move(input), reason);
    }
}

void SplitResolve::OnSegment(const Ir::SegmentBody& body, const u32* back_regs) {
    t_commit = this;
    const u64 start = NowNs();
    if (holding) {
        resolver->Release(held_id);
        holding = false;
        current = nullptr;
    }
    Resolver::Result& result = resolver->Take(body.id);
    resolver->NoteOrder(body.id);
    holding = true;
    held_id = body.id;
    current = &result;
    cursor = 0;
    resolver->NoteCommitted(result, start);
    const u32 size = adaptive.Note(result.waited, result.stolen,
                                   adaptive_size.load(std::memory_order_relaxed));
    if (size != 0) {
        adaptive_size.store(size, std::memory_order_relaxed);
        GetStats().segment_size.store(size, std::memory_order_relaxed);
    }
    if (verify && back_regs != nullptr && result.has_input) {
        // The snapshot at the segment start must equal the back's registers here (but the write
        // the snapshot already holds that follows the record).
        const SegmentInput& input = resolver->InputOf(body.id);
        if (input.snapshot != nullptr) {
            Count(GetStats().snapshot_checks);
            for (u32 index = 0; index < 0xe00; ++index) {
                if (index == body.pending_method) {
                    continue;
                }
                const u32 snapshot_value = input.snapshot->regs.Read(index);
                if (snapshot_value != back_regs[index]) {
                    Count(GetStats().snapshot_mismatches);
                    if (ShouldLogMismatch()) {
                        LOG_WARNING(HW_GPU,
                                    "V2 parallel verify: segment {} snapshot register {:#x} is "
                                    "{:#x}, the back has {:#x}",
                                    body.id, index, snapshot_value, back_regs[index]);
                    }
                    break;
                }
            }
        }
    }
    if (stats && start - last_report_ns > 10'000'000'000ULL) {
        if (last_report_ns != 0) {
            ReportStats("GPU back thread", start - last_report_ns);
        }
        last_report_ns = start;
    }
}

const PreDraw* SplitResolve::TakeDraw(u64 fingerprint, bool indexed) {
    if (current == nullptr || !current->by_shard) {
        return nullptr;
    }
    const auto& records = current->records;
    const size_t limit = (std::min)(records.size(), cursor + 8);
    for (size_t index = cursor; index < limit; ++index) {
        const PreDraw& pre = records[index];
        if (pre.fingerprint == fingerprint && pre.indexed == indexed) {
            cursor = index + 1;
            if (pre.drift) {
                Count(GetStats().draws_drift);
                return nullptr;
            }
            return &pre;
        }
    }
    Count(GetStats().draws_unmatched);
    return nullptr;
}

} // namespace VideoCommon::V2::Resolve
