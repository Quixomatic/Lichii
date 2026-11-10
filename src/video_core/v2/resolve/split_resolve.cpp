// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <optional>
#include <thread>

#include "common/logging.h"
#include "common/scope_exit.h"
#include "video_core/gpu_split.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/draw_model.h"
#include "video_core/v2/epoch_publish.h"
#include "video_core/v2/regsem_3d.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/resolve/dynamic_state.h"
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

/// A sequence lock over the table (F6-X1): odd while a publish or clear writes it. Two resolves
/// of one draw that read the same even generation before and after (GenerationHolds) got the
/// same answers; otherwise a difference is explained by a publish in between (AP6 §3.2).
std::atomic<u64> g_shape_generation{2};

void BeginGeneration(std::atomic<u64>& generation) {
    generation.store(generation.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
}

void EndGeneration(std::atomic<u64>& generation) {
    generation.store(generation.load(std::memory_order_relaxed) + 1, std::memory_order_release);
}

bool GenerationHolds(const std::atomic<u64>& generation, u64 before) {
    std::atomic_thread_fence(std::memory_order_acquire);
    return (before & 1) == 0 && generation.load(std::memory_order_relaxed) == before;
}

} // Anonymous namespace

void PublishShape(u64 shape, u32 relevant_groups) {
    PublishShape(shape, relevant_groups, std::array<u32, CbufStages>{});
}

u64 ShapeGeneration() {
    return g_shape_generation.load(std::memory_order_acquire);
}

bool ShapeGenerationHolds(u64 before) {
    return GenerationHolds(g_shape_generation, before);
}

void PublishShape(u64 shape, u32 relevant_groups, const std::array<u32, CbufStages>& key_cbufs) {
    // Commit thread only (single writer).
    BeginGeneration(g_shape_generation);
    SCOPE_EXIT {
        EndGeneration(g_shape_generation);
    };
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
    BeginGeneration(g_shape_generation);
    SCOPE_EXIT {
        EndGeneration(g_shape_generation);
    };
    for (ShapeEntry& entry : Shapes()) {
        const u64 sequence = entry.sequence.load(std::memory_order_relaxed);
        entry.sequence.store(sequence + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
        entry.shape.store(0, std::memory_order_relaxed);
        entry.relevant.store(0, std::memory_order_relaxed);
        entry.sequence.store(sequence + 2, std::memory_order_release);
    }
}

// ---- Published programs (R3) ----------------------------------------------------------------------

namespace {

struct ProgramKey {
    u64 memory;
    u64 address;
};

u64 ProgramHash(const ProgramKey& key) {
    return Mix64(key.address ^ Mix64(key.memory));
}

/// F6-X1: insert-only (MASTER_PLAN §5.3 pure lookups): a program never evicts another, so what
/// a shard finds depends only on the publishes before it (the direct-mapped table it replaces
/// overwrote colliding programs, and the commit republished them on every shard miss).
PublishedMap<ProgramKey, u64>& Programs() {
    static PublishedMap<ProgramKey, u64> programs{Commit::Epochs(), 256};
    return programs;
}

std::atomic<u64> g_program_generation{2};

} // Anonymous namespace

void PublishProgram(uintptr_t memory, GPUVAddr address, u64 unique_hash) {
    const ProgramKey key{.memory = memory, .address = address};
    const u64 hash = ProgramHash(key);
    auto& programs = Programs();
    if (programs.Find(key, hash) == unique_hash) {
        return;
    }
    BeginGeneration(g_program_generation);
    programs.Insert(key, hash, unique_hash);
    EndGeneration(g_program_generation);
}

bool FindProgram(uintptr_t memory, GPUVAddr address, u64* unique_hash) {
    const ProgramKey key{.memory = memory, .address = address};
    const u64 found = Programs().Find(key, ProgramHash(key));
    if (found == PublishedMap<ProgramKey, u64>::Missing || found == 0) {
        return false;
    }
    *unique_hash = found;
    return true;
}

void ClearPrograms() {
    BeginGeneration(g_program_generation);
    Programs().Clear();
    EndGeneration(g_program_generation);
}

u64 ProgramGeneration() {
    return g_program_generation.load(std::memory_order_acquire);
}

bool ProgramGenerationHolds(u64 before) {
    return GenerationHolds(g_program_generation, before);
}

// ---- Published dynamic-state features (R5) --------------------------------------------------------

namespace {

struct PublishedFeatures {
    DynamicFeatures features;
    u64 generation;
};
/// Every set of features ever published (commit thread only): they change with the program
/// and a setting, a handful of times a session, and a shard may still compute with an old one.
std::vector<std::unique_ptr<PublishedFeatures>>& FeatureCopies() {
    static std::vector<std::unique_ptr<PublishedFeatures>> copies;
    return copies;
}
std::atomic<const PublishedFeatures*> g_features{nullptr};

} // Anonymous namespace

u64 PublishDynamicFeatures(const DynamicFeatures& features) {
    const PublishedFeatures* const current = g_features.load(std::memory_order_relaxed);
    if (current != nullptr &&
        std::memcmp(&current->features, &features, sizeof(DynamicFeatures)) == 0) {
        return current->generation;
    }
    auto copy = std::make_unique<PublishedFeatures>();
    std::memcpy(&copy->features, &features, sizeof(DynamicFeatures));
    copy->generation = current != nullptr ? current->generation + 1 : 1;
    const PublishedFeatures* const published = copy.get();
    FeatureCopies().push_back(std::move(copy));
    g_features.store(published, std::memory_order_release);
    return published->generation;
}

std::shared_ptr<const DynamicState> ShardDynamic(const SnapshotView& view, u64& generation) {
    const PublishedFeatures* const published = g_features.load(std::memory_order_acquire);
    if (published == nullptr || !On(Switch::DrawRec)) {
        return nullptr;
    }
    generation = published->generation;
    // ComputeDynamic reads these groups and the topology (the record path's same_dynamic
    // rule): a shard reuses its last result while their hashes and the features hold.
    constexpr u32 groups = GroupBit(RegGroup::Pipeline) | GroupBit(RegGroup::Viewport) |
                           GroupBit(RegGroup::VertexStreams) | GroupBit(RegGroup::IndexBuffer) |
                           GroupBit(RegGroup::RenderTargets);
    static_assert((groups & ~HashedGroups) == 0);
    u64 inputs = view.Groups().Fingerprint(groups);
    inputs = Mix64(inputs ^ static_cast<u64>(view.Topology()));
    inputs = Mix64(inputs ^ published->generation);
    thread_local std::shared_ptr<const DynamicState> last;
    thread_local u64 last_inputs = 0;
    if (last != nullptr && last_inputs == inputs && view.Groups().valid) {
        return last;
    }
    last = std::make_shared<const DynamicState>(ComputeDynamic(view, published->features));
    last_inputs = inputs;
    return last;
}

// ---- Shards ---------------------------------------------------------------------------------------

namespace {

constexpr u64 RingKindMask = 0xf;
constexpr u64 RingWrap = 0;
constexpr u64 RingMethod = 2;
constexpr u64 MethodHasSegment = 1 << 5;

void LoadMirror(RegSem::Mirror3D& mirror, const FrontSnapshot& snapshot) {
    snapshot.regs.CopyTo(mirror.regs.data());
    if (snapshot.shadow.Valid()) {
        snapshot.shadow.CopyTo(mirror.shadow.data()); // none with S1 (plain deltas)
    }
    mirror.control = static_cast<RegSem::ShadowRamControl>(snapshot.control);
    mirror.instance_loop = snapshot.instance_loop;
    mirror.sink_pending = snapshot.sink_pending;
    mirror.draw_pending = snapshot.draw_pending;
    mirror.skip_same = false; // the ring holds what the back applies: apply all of it
    mirror.v2_groups = snapshot.groups;
    mirror.v2_cbufs = snapshot.cbufs != nullptr ? *snapshot.cbufs : CbufTable{};
}

/// An early-submitted segment: the position the shard may read up to, waiting for the front to
/// publish more; `position` itself once it is the segment's end; nothing when abandoned.
std::optional<u64> WaitPublished(const SegmentInput& input, u64 position) {
    SegmentProgress& progress = *input.progress;
    for (u32 spins = 0;; ++spins) {
        // F1-X2: the publish position first. The front stores `end` before it publishes anything
        // past the cut, so an end read after this acquire is never older than the position read;
        // in the other order a stale "no end yet" with a newer position read on into the next
        // segment's draws (PARALLEL=verify "record count" at the size cut).
        const u64 published = input.published->load(std::memory_order_acquire);
        const u64 end = progress.end.load(std::memory_order_acquire);
        const u64 limit = (std::min)(end, published);
        if (limit > position || position >= end) {
            progress.waiting.store(false, std::memory_order_relaxed);
            return limit;
        }
        if (progress.abandon.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        if (spins == 64) {
            progress.waiting.store(true, std::memory_order_release);
        }
        if (spins < 1024) {
            CpuRelax();
        } else {
            std::this_thread::yield();
        }
    }
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
    // Counts every draw of the segment, skipped ones too (other kinds, other engines, host
    // indirect), so a draw one side skips does not shift the ordinals Compare checks.
    u32 ordinal = 0;
    // One predicted draw: its draw-cache shape and keys from the mirror at its position.
    const auto add_draw = [&](const Ir::PredictedDraw& draw, u64 fingerprint, bool drift,
                              const GroupTracker& groups) {
        const u32 draw_ordinal = ordinal++;
        if (draw.kind != Ir::DrawKind::Array && draw.kind != Ir::DrawKind::Indexed) {
            return;
        }
        PreDraw pre;
        pre.fingerprint = fingerprint;
        pre.indexed = draw.kind == Ir::DrawKind::Indexed;
        pre.ordinal = draw_ordinal;
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
                // R3: and the pipeline it names, from the published programs and lookup.
                if (pre.pipeline_key != nullptr && g_shard_pipeline != nullptr) {
                    g_shard_pipeline(view, snapshot.memory, pre);
                }
                // R5: the draw's dynamic state (scale-independent: the view alone).
                pre.dynamic = ShardDynamic(view, pre.dynamic_generation);
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
            if (!ShapeGenerationHolds(pre.shape_generation)) {
                pre.shape_generation = UnstableGeneration;
            }
            pre.shape = key.shape;
            pre.partial = key.partial;
            pre.full = key.full;
            pre.cbuf_masks = key.cbuf_masks;
            pre.relevant = key.relevant;
            pre.has_shape = key.has_shape;
            pre.has_key = key.has_key;
            pre.has_full = key.has_full;
        }
        records.push_back(std::move(pre));
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
    u64 limit = input.progress != nullptr ? input.begin : input.end;
    for (u64 position = input.begin;;) {
        if (position >= limit) {
            if (input.progress == nullptr) {
                break;
            }
            const std::optional<u64> available = WaitPublished(input, position);
            if (!available) {
                // F1-36: the commit reached the segment while the front still writes it. The
                // draws resolved so far are published ones: the commit takes them (TakeDraw
                // matches the prefix, the rest resolve live) instead of none.
                return records.empty() ? SerialReason::Stolen : SerialReason::None;
            }
            limit = *available;
            if (position >= limit) {
                // The segment's end: a reason the front found after submitting it decides.
                return static_cast<SerialReason>(
                    input.progress->serial.load(std::memory_order_acquire));
            }
        }
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
                ordinal += body.op == Ir::DrawOp::HostIndirect ? 1 : body.num_draws;
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

namespace {

/// F6-X1: two table reads answered alike unless a publish happened between or during them.
bool GenerationsDiffer(u64 a, u64 b) {
    return a != b || a == UnstableGeneration;
}

} // Anonymous namespace

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
    if (key_differs && GenerationsDiffer(a.shape_generation, b.shape_generation)) {
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
    // R3: the pipeline object. A lookup insert or clear, or a program published in between,
    // changes what the tables answer (dependent).
    if (a.has_pipeline != b.has_pipeline || a.pipeline != b.pipeline || a.hashes != b.hashes) {
        if (GenerationsDiffer(a.lookup_serial, b.lookup_serial) ||
            GenerationsDiffer(a.program_generation, b.program_generation)) {
            Count(GetStats().verify_dependent);
            return nullptr;
        }
        return a.hashes != b.hashes ? "pipeline shader hashes" : "pipeline object";
    }
    // R5: the dynamic state (features published in between: dependent).
    if ((a.dynamic == nullptr) != (b.dynamic == nullptr) ||
        (a.dynamic != nullptr && a.dynamic != b.dynamic &&
         std::memcmp(a.dynamic.get(), b.dynamic.get(), sizeof(DynamicState)) != 0)) {
        if (a.dynamic_generation != b.dynamic_generation) {
            Count(GetStats().verify_dependent);
            return nullptr;
        }
        return "dynamic state";
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
    closed = std::make_unique<std::array<SegmentInput, ClosedCapacity>>();
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
    if (last_early != nullptr) {
        last_early->abandon.store(true, std::memory_order_release); // its end may never publish
    }
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
        const u64 snapshot_start = stats ? NowNs() : 0;
        ReclaimReleased();
        const u64 epoch = segmenter.OpenId(); // the id of the segment this snapshot starts
        FrontSnapshot* const snapshot = snapshot_pool.Allocate();
        snapshot_pool.Retire(snapshot, epoch);
        *snapshot = FrontSnapshot{};
        const u64 copied_before = regs_builder.copied + shadow_builder.copied;
        const u64 shared_before = regs_builder.shared + shadow_builder.shared;
        if (view.dirty_blocks != nullptr) {
            snapshot->regs = regs_builder.Take(view.regs, *view.dirty_blocks, epoch);
            *view.dirty_blocks = 0;
        } else {
            snapshot->regs = regs_builder.Take(view.regs, ~u64{0}, epoch);
        }
        if (!view.deltas) {
            snapshot->shadow = shadow_builder.Take(view.shadow, ~u64{0}, epoch);
        }
        if (stats) {
            Count(GetStats().snapshot_blocks_copied,
                  regs_builder.copied + shadow_builder.copied - copied_before);
            Count(GetStats().snapshot_blocks_shared,
                  regs_builder.shared + shadow_builder.shared - shared_before);
        }
        snapshot->groups = *view.groups;
        if (view.cbufs != nullptr) {
            // Bindings change far less often than segments are cut: share the last copy while
            // no bind happened (a counter compare, not a table compare).
            if (last_cbufs == nullptr || last_cbuf_changes != view.cbufs->changes) {
                if (last_cbufs != nullptr) {
                    cbuf_pool.Retire(last_cbufs, epoch - 1);
                }
                last_cbufs = cbuf_pool.Allocate();
                *last_cbufs = *view.cbufs;
                last_cbuf_changes = view.cbufs->changes;
            }
            snapshot->cbufs = last_cbufs;
        }
        if (stats) {
            Count(GetStats().front_snapshot_ns, NowNs() - snapshot_start);
        }
        snapshot->control = view.control;
        snapshot->render_enable_unknown = view.render_enable_unknown;
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
            if (open_progress != nullptr) {
                // Submitted early: its shard reads on up to here once it is published.
                open_progress->serial.store(static_cast<u8>(closing_reason),
                                            std::memory_order_relaxed);
                open_progress->end.store(position, std::memory_order_release);
                open_progress = nullptr;
            } else {
                open_input.end = position;
                open_input.serial = closing_reason;
                const u64 tail = closed_tail.load(std::memory_order_relaxed);
                if (tail - closed_head.load(std::memory_order_acquire) < ClosedCapacity) {
                    (*closed)[tail % ClosedCapacity] = open_input;
                    closed_tail.store(tail + 1, std::memory_order_release);
                } else {
                    // Every queued slot taken: the id is given up (the commit resolves it
                    // serially); a refused submission tells the shards to move past it.
                    resolver->Submit(open_input.id, std::move(open_input), SerialReason::BackPressure);
                }
            }
        }
        open = true;
        open_input = SegmentInput{};
        open_input.id = body.id;
        open_input.begin = position;
        open_input.snapshot = snapshot;
        open_input.ring = ring;
        open_input.arena = arena;
        open_input.draws = 1;
    }
}

void SplitResolve::ReclaimReleased() {
    // A submitted input is never read after the commit released its id (Submit refuses it
    // then), but the front itself still writes the open segment's progress.
    u64 through = released.load(std::memory_order_acquire);
    if (open) {
        through = (std::min)(through, open_input.id - 1);
    }
    regs_builder.Reclaim(through);
    shadow_builder.Reclaim(through);
    snapshot_pool.Reclaim(through);
    cbuf_pool.Reclaim(through);
    progress_pool.Reclaim(through);
}

void SplitResolve::DrainClosed(u64 published) {
    for (;;) {
        if (submitting.exchange(true, std::memory_order_acquire)) {
            // The other thread submits: it looks again when it is done.
            drain_again.store(true, std::memory_order_release);
            return;
        }
        u64 head = closed_head.load(std::memory_order_relaxed);
        const u64 tail = closed_tail.load(std::memory_order_acquire);
        while (head != tail) {
            SegmentInput& input = (*closed)[head % ClosedCapacity];
            if (input.end > published) {
                break;
            }
            const u32 draws = input.draws;
            const u32 in_flight = in_flight_draws.load(std::memory_order_acquire);
            if (in_flight != 0 && in_flight + draws > InFlightDraws) {
                break; // the shards hold enough draws: submitted when the commit takes some
            }
            bool retry = false;
            in_flight_draws.fetch_add(draws, std::memory_order_acq_rel); // before the commit sees it
            if (!resolver->Submit(input.id, SegmentInput{input}, input.serial, &retry)) {
                in_flight_draws.fetch_sub(draws, std::memory_order_acq_rel);
                if (retry) {
                    break; // every slot busy: submitted when the commit releases one
                }
            }
            ++head;
        }
        closed_head.store(head, std::memory_order_release);
        submitting.store(false, std::memory_order_release);
        if (!drain_again.exchange(false, std::memory_order_acq_rel)) {
            return;
        }
        published = (std::max)(published, split.published.load(std::memory_order_acquire));
    }
}

void SplitResolve::OnPublish(u64 position) {
    DrainClosed(position);
    // The open segment goes once its start is published and it holds a few draws, so the back
    // does not reach it before a shard could start (AP6 §1.3(a)). Ids stay in order: only when
    // every closed segment went.
    if (!open || open_progress != nullptr || !ClosedEmpty() || open_input.begin > position ||
        open_input.draws < EarlySubmitDraws || segmenter.OpenReason() != SerialReason::None ||
        open_input.snapshot == nullptr) {
        return;
    }
    const u32 draws = open_input.draws;
    const u32 in_flight = in_flight_draws.load(std::memory_order_acquire);
    if (in_flight != 0 && in_flight + draws > InFlightDraws) {
        return;
    }
    SegmentProgress* const progress = progress_pool.Allocate();
    progress_pool.Retire(progress, open_input.id);
    progress->Reset();
    open_input.progress = progress;
    open_input.published = &split.published;
    bool retry = false;
    in_flight_draws.fetch_add(draws, std::memory_order_acq_rel);
    if (resolver->Submit(open_input.id, std::move(open_input), SerialReason::None, &retry)) {
        open_progress = progress;
        last_early = progress;
        if (stats) {
            Count(GetStats().early_submits);
        }
        return;
    }
    in_flight_draws.fetch_sub(draws, std::memory_order_acq_rel);
    open_input.progress = nullptr;
    open_input.published = nullptr;
}

void SplitResolve::OnSegment(const Ir::SegmentBody& body, const u32* back_regs,
                             const u32* back_shadow, const CbufTable* back_cbufs) {
    t_commit = this;
    const u64 start = NowNs();
    if (holding) {
        resolver->Release(held_id);
        holding = false;
        current = nullptr;
        // F1-36: the freed slot takes the next closed segment now, not at the front's next
        // publish (a front waiting on this thread publishes nothing meanwhile).
        DrainClosed(split.published.load(std::memory_order_acquire));
    }
    taken = nullptr;
    record_taken = nullptr; // it pointed into the released segment's records
    Resolver::Result& result = resolver->Take(body.id);
    // Segments are taken in id order and Take returns once no shard reads `body.id`: every older
    // one's input is done with (the front reuses their snapshots, F1-30).
    if (body.id > 1) {
        released.store(body.id - 1, std::memory_order_release);
    }
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
            // F6-X1: the back owns render_enable.mode while the front's copy is stale.
            constexpr u32 render_enable_mode =
                static_cast<u32>(offsetof(MaxwellRegs, render_enable.mode) / sizeof(u32));
            const u32 back_owned =
                input.snapshot->render_enable_unknown ? render_enable_mode : ~0U;
            for (u32 index = 0; index < 0xe00; ++index) {
                if (index == body.pending_method || index == back_owned) {
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
                    if (index == body.pending_method || index == back_owned) {
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
