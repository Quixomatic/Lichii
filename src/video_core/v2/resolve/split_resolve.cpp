// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <bit>
#include <cstddef>
#include <cstring>

#include "common/logging.h"
#include "video_core/gpu_split.h"
#include "video_core/v2/draw_model.h"
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/resolve/split_resolve.h"
#include "video_core/v2/resolve/state_view.h"
#include "video_core/v2/s1_deltas.h"
#include "video_core/v2/v2_stats.h"
#include "video_core/v2/v2_switches.h"

namespace VideoCommon::V2::Resolve {

// ---- Published shapes -----------------------------------------------------------------------------

namespace {

struct ShapeEntry {
    std::atomic<u64> sequence{};
    std::atomic<u64> shape{};
    std::atomic<u32> relevant{};
    std::array<std::atomic<u32>, CbufStages> key_cbufs{};
};
constexpr size_t ShapeEntries = 4096;

std::array<ShapeEntry, ShapeEntries>& Shapes() {
    static std::array<ShapeEntry, ShapeEntries> entries;
    return entries;
}

size_t ShapeSlot(u64 shape) {
    return static_cast<size_t>((shape * 0x9e3779b97f4a7c15ULL) >> 52) % ShapeEntries;
}

/// Moves at every publish and clear: two resolves of one draw that read the table at different
/// generations may differ in what the table answered (AP6 §3.2).
std::atomic<u64> g_shape_generation{0};

} // Anonymous namespace

void PublishShape(u64 shape, u32 relevant_groups) {
    PublishShape(shape, relevant_groups, std::array<u32, CbufStages>{});
}

u64 ShapeGeneration() {
    return g_shape_generation.load(std::memory_order_acquire);
}

void PublishShape(u64 shape, u32 relevant_groups, const std::array<u32, CbufStages>& key_cbufs) {
    g_shape_generation.fetch_add(1, std::memory_order_acq_rel);
    ShapeEntry& entry = Shapes()[ShapeSlot(shape)];
    const u64 sequence = entry.sequence.load(std::memory_order_relaxed);
    entry.sequence.store(sequence + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    entry.shape.store(shape, std::memory_order_relaxed);
    entry.relevant.store(relevant_groups, std::memory_order_relaxed);
    for (size_t stage = 0; stage < CbufStages; ++stage) {
        entry.key_cbufs[stage].store(key_cbufs[stage], std::memory_order_relaxed);
    }
    entry.sequence.store(sequence + 2, std::memory_order_release);
}

bool FindShape(u64 shape, u32* relevant_groups) {
    return FindShape(shape, relevant_groups, nullptr);
}

u64 CbufMasksHash(const std::array<u32, CbufStages>& key_cbufs) {
    u64 hash = 0x5bd1e9955bd1e995ULL;
    for (const u32 mask : key_cbufs) {
        hash = Mix64(hash ^ mask);
    }
    return hash;
}

bool FindShape(u64 shape, u32* relevant_groups, std::array<u32, CbufStages>* key_cbufs) {
    const ShapeEntry& entry = Shapes()[ShapeSlot(shape)];
    for (u32 attempt = 0; attempt < 16; ++attempt) {
        const u64 before = entry.sequence.load(std::memory_order_acquire);
        if ((before & 1) != 0) {
            CpuRelax();
            continue;
        }
        const u64 stored = entry.shape.load(std::memory_order_relaxed);
        const u32 relevant = entry.relevant.load(std::memory_order_relaxed);
        std::array<u32, CbufStages> masks{};
        for (size_t stage = 0; stage < CbufStages; ++stage) {
            masks[stage] = entry.key_cbufs[stage].load(std::memory_order_relaxed);
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        if (entry.sequence.load(std::memory_order_relaxed) != before) {
            continue;
        }
        if (stored != shape || stored == 0) {
            return false;
        }
        *relevant_groups = relevant;
        if (key_cbufs != nullptr) {
            *key_cbufs = masks;
        }
        return true;
    }
    return false;
}

void ClearShapes() {
    g_shape_generation.fetch_add(1, std::memory_order_acq_rel);
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
    mirror.v2_cbufs = snapshot.cbufs != nullptr ? *snapshot.cbufs : CbufTable{};
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
    // One predicted draw: its draw-cache shape and keys from the mirror at its position.
    const auto add_draw = [&](const Ir::PredictedDraw& draw, u64 fingerprint, bool drift,
                              const GroupTracker& groups) {
        if (draw.kind != Ir::DrawKind::Array && draw.kind != Ir::DrawKind::Indexed) {
            return;
        }
        PreDraw pre;
        pre.fingerprint = fingerprint;
        pre.indexed = draw.kind == Ir::DrawKind::Indexed;
        pre.ordinal = ordinal++;
        pre.engine = snapshot.engine;
        pre.memory = snapshot.memory;
        pre.drift = drift;
        if (!drift) {
            if (g_pipeline_key != nullptr) {
                // P6c step (a): the draw's pipeline key from the same state view.
                const SnapshotView view{
                    mirror.regs.data(),
                    &mirror.v2_cbufs,
                    &groups,
                    static_cast<MaxwellRegs::PrimitiveTopology>(draw.topology),
                    ViewDraw{
                        .first = draw.first,
                        .count = draw.count,
                        .instances = draw.instances,
                        .base_vertex = draw.base_vertex,
                        .base_instance = draw.base_instance,
                        .indexed = pre.indexed,
                    },
                    snapshot.native_turing};
                pre.pipeline_key = g_pipeline_key(view);
            }
            // The draw-cache shape holds the pipeline key's hash: only the canonical key the
            // serial path uses with EDEN_NX2_V2_PIPEKEY equals the shard's.
            const bool state_hash = pre.pipeline_key != nullptr && On(Switch::PipeKey);
            pre.shape_generation = ShapeGeneration();
            const ShardKey key = g_shard_key(ShardKeyInputs{
                .regs = mirror.regs.data(),
                .groups = &groups,
                .cbufs = &mirror.v2_cbufs,
                .topology = draw.topology,
                .indexed = pre.indexed,
                .native_turing = snapshot.native_turing,
                .engine = snapshot.engine,
                .memory = snapshot.memory,
                .state_hash = state_hash ? pre.pipeline_key->hash : 0,
                .has_state_hash = state_hash,
            });
            pre.shape = key.shape;
            pre.partial = key.partial;
            pre.full = key.full;
            pre.cbuf_masks = key.cbuf_masks;
            pre.relevant = key.relevant;
            pre.has_shape = key.has_shape;
            pre.has_key = key.has_key;
            pre.has_full = key.has_full;
        }
        records.push_back(pre);
    };
    // S1 classified runs (§5.6 step 2): plain stores of the front's deltas, its groups, its
    // draws; no second run of the register semantics.
    constexpr u32 cb_data = static_cast<u32>(offsetof(MaxwellRegs, const_buffer) / 4) + 4;
    const auto apply_s1 = [&](const u64* entry) {
        switch (S1::TagOf(entry[0])) {
        case S1::Tag::Groups: {
            const u32 changed = S1::GroupsDirty(entry[0]);
            const u64* hash = entry + 1;
            for (u32 bits = changed & HashedGroups; bits != 0; bits &= bits - 1) {
                mirror.v2_groups.hashes[std::countr_zero(bits)] = *hash++;
            }
            mirror.v2_groups.dirty |= changed;
            break;
        }
        case S1::Tag::Draw: {
            mirror.v2_groups.BeginDraw(); // as the back's NoteDraw3D before it draws
            add_draw(S1::DrawOf(entry), mirror.v2_groups.Fingerprint(HashedGroups), false,
                     mirror.v2_groups);
            break;
        }
        default:
            S1::ApplyRegisters(entry, cb_data, table.const_buffer_offset,
                               [&](u32 index, u32 value) { mirror.regs[index] = value; });
            if (S1::TagOf(entry[0]) == S1::Tag::Exec &&
                (table.flags[S1::IndexOf(entry[0])] & RegSem::CbufBindFlag) != 0) {
                RegSem::BindCbuf3D(mirror.v2_cbufs, mirror.regs.data(), S1::IndexOf(entry[0]));
            }
            break;
        }
        return S1::EntryWords(entry);
    };
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
            if ((record.header & Ir::Classified) != 0) {
                while (it < end) {
                    it += apply_s1(it);
                }
                break;
            }
            while (it < end) {
                const u64 word = *it;
                if (S1::IsClassified(word)) {
                    it += S1::EntryWords(it); // GpuSplit::S1Check entries: the words decide
                    continue;
                }
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
                add_draw(body.draws[i], body.fingerprint, drift, groups);
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
    // What the shape table answered: the resolves read it at different publish generations, so
    // a difference in the table-derived parts is explained by a publish in between (dependent).
    const bool key_differs = a.has_key != b.has_key || a.relevant != b.relevant ||
                             a.has_full != b.has_full || a.full != b.full ||
                             a.cbuf_masks != b.cbuf_masks;
    const bool pipeline_key_differs =
        (a.pipeline_key == nullptr) != (b.pipeline_key == nullptr) ||
        (a.pipeline_key != nullptr && (a.pipeline_key->hash != b.pipeline_key->hash ||
                                       a.pipeline_key->bytes != b.pipeline_key->bytes));
    if (key_differs && a.shape_generation != b.shape_generation) {
        Count(GetStats().verify_dependent);
        return pipeline_key_differs ? "pipeline key" : nullptr;
    }
    // Which part of the key (Fix pass 1: "key" alone didn't tell the register-derived part from
    // the shape table's relevance mask the shard read).
    if (a.has_key != b.has_key) {
        return "key presence";
    }
    if (a.relevant != b.relevant) {
        return "key relevant groups (the published shape info)";
    }
    if (a.partial != b.partial) {
        return "key (texture pools, vertex layout or handle constant buffers)";
    }
    if (a.has_full != b.has_full || a.full != b.full || a.cbuf_masks != b.cbuf_masks) {
        return "full key (constant buffer bindings)";
    }
    if ((a.pipeline_key == nullptr) != (b.pipeline_key == nullptr)) {
        return "pipeline key presence";
    }
    if (a.pipeline_key != nullptr &&
        (a.pipeline_key->hash != b.pipeline_key->hash ||
         a.pipeline_key->bytes != b.pipeline_key->bytes)) {
        return "pipeline key";
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
            open_input.draws += open ? 1 : 0;
            continue;
        }
        if (cut == CutReason::None) {
            open_input.draws += open ? 1 : 0;
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
        if (view.cbufs != nullptr) {
            // Bindings change far less often than segments are cut: share the last copy.
            if (last_cbufs == nullptr || !(*last_cbufs == *view.cbufs)) {
                last_cbufs = std::make_shared<const CbufTable>(*view.cbufs);
            }
            snapshot->cbufs = last_cbufs;
        }
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
        open_input.draws = 1;
    }
}

void SplitResolve::OnPublish(u64 position) {
    while (!closed.empty() && closed.front().end <= position) {
        SegmentInput& input = closed.front();
        const u64 id = input.id;
        const SerialReason reason = input.serial;
        const u32 draws = input.draws;
        const u32 in_flight = in_flight_draws.load(std::memory_order_acquire);
        if (in_flight != 0 && in_flight + draws > InFlightDraws) {
            break; // the shards hold enough draws: submitted at a later publish
        }
        bool retry = false;
        in_flight_draws.fetch_add(draws, std::memory_order_acq_rel); // before the commit sees it
        if (!resolver->Submit(id, std::move(input), reason, &retry)) {
            in_flight_draws.fetch_sub(draws, std::memory_order_acq_rel);
            if (retry) {
                break; // every slot busy: submitted again at the next publish
            }
        }
        closed.pop_front();
    }
}

void SplitResolve::OnSegment(const Ir::SegmentBody& body, const u32* back_regs,
                             const u32* back_shadow, const CbufTable* back_cbufs) {
    t_commit = this;
    const u64 start = NowNs();
    if (holding) {
        resolver->Release(held_id);
        holding = false;
        current = nullptr;
    }
    taken = nullptr;
    Resolver::Result& result = resolver->Take(body.id);
    if (result.has_input) {
        in_flight_draws.fetch_sub(resolver->InputOf(body.id).draws, std::memory_order_acq_rel);
    }
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
                    if (ShouldLogMismatch(MismatchLog::Snapshot)) {
                        LOG_WARNING(HW_GPU,
                                    "V2 parallel verify: segment {} snapshot register {:#x} is "
                                    "{:#x}, the back has {:#x}",
                                    body.id, index, snapshot_value, back_regs[index]);
                    }
                    break;
                }
            }
            if (back_shadow != nullptr) {
                Count(GetStats().snapshot_checks);
                for (u32 index = 0; index < 0xe00; ++index) {
                    if (index == body.pending_method) {
                        continue;
                    }
                    const u32 snapshot_value = input.snapshot->shadow.Read(index);
                    if (snapshot_value != back_shadow[index]) {
                        Count(GetStats().snapshot_mismatches);
                        if (ShouldLogMismatch(MismatchLog::Snapshot)) {
                            LOG_WARNING(HW_GPU,
                                        "V2 parallel verify: segment {} snapshot shadow {:#x} is "
                                        "{:#x}, the back has {:#x}",
                                        body.id, index, snapshot_value, back_shadow[index]);
                        }
                        break;
                    }
                }
            }
            // P6c: the constant buffer bindings the snapshot carries are the back engine's.
            if (back_cbufs != nullptr) {
                Count(GetStats().snapshot_checks);
                const CbufTable& snapshot_cbufs =
                    input.snapshot->cbufs != nullptr ? *input.snapshot->cbufs : CbufTable{};
                for (size_t stage = 0; stage < CbufStages; ++stage) {
                    for (size_t slot = 0; slot < CbufSlots; ++slot) {
                        const CbufBinding& a = snapshot_cbufs.stages[stage][slot];
                        const CbufBinding& b = back_cbufs->stages[stage][slot];
                        if (a == b) {
                            continue;
                        }
                        Count(GetStats().snapshot_mismatches);
                        if (ShouldLogMismatch(MismatchLog::Snapshot)) {
                            LOG_WARNING(HW_GPU,
                                        "V2 parallel verify: segment {} snapshot constant buffer "
                                        "stage {} slot {} is {:#x}+{:#x} enabled {}, the back has "
                                        "{:#x}+{:#x} enabled {}",
                                        body.id, stage, slot, a.address, a.size, a.enabled,
                                        b.address, b.size, b.enabled);
                        }
                        stage = CbufStages;
                        break;
                    }
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
