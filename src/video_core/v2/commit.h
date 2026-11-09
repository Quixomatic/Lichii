// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/epoch_publish.h"
#include "video_core/v2/v2_switches.h"

// Pipeline v2 P5a: the S3 commit stage (MASTER_PLAN §5.3 "No mutation in S2", §5.4).
//
// Cache lookups are pure reads. A lookup returns either a ready resource or a mutation request
// plus a proxy handle. The commit owner (one thread) applies requests strictly in sequence order
// and patches the proxies with the real resources. Requests are POD records; the applier is the
// cache code that used to run inside the lookup.
//
// P5a runs lookups and commit on the same thread, right before the draw is encoded. P6 shards run
// lookups on N threads: each shard fills one RequestList per IR segment (its own proxies), submits
// it with Stage::Submit from any thread, and the commit thread applies whole segments in segment
// order (Stage::CommitReady). A lookup that needs the real resource before it can go on (a buffer
// it must read, the id of a view a later lookup indexes) cannot proceed on a shard: it marks the
// segment Serial, and the commit thread resolves that segment itself.
//
// External threads (guest CPU write traps, the fencing thread, readers of GPU-written memory)
// never touch the caches while commit owns them: they post ExternalOps (asynchronous, never
// blocking) or owner calls (synchronous) that the owner runs at its next commit point. See
// IMPLEMENTATION_LOG.md, "P5a", for the full protocol and its invariants.
namespace VideoCommon::V2::Commit {

/// What a lookup asks the commit stage to do.
enum class Kind : u8 {
    BufferCreate,        ///< no buffer covers the range: CreateBuffer (joins overlaps)
    BufferUpload,        ///< CPU-modified bytes in a bound range: SynchronizeBuffer
    ViewCreate,          ///< a descriptor with no view: CreateImageView (may create/join images)
    ViewBind,            ///< a changed descriptor table entry naming a known view
    ImagePrepare,        ///< a bound view: PrepareImage (refresh, alias sync, LRU)
    SamplerCreate,       ///< a sampler descriptor with no sampler: FindSampler
    SamplerBind,         ///< a changed sampler table entry naming a known sampler
    RenderTargets,       ///< render target registers changed: UpdateRenderTargets
    RenderTargetPrepare, ///< clean render target registers: PrepareImageView(rt, modification)
    Rescale,             ///< blacklisted view: ScaleDown
    External,            ///< a write, invalidation or unmap another thread posted
    Writeback,           ///< deferred GPU data owed to guest memory
    Serial,              ///< the lookup needed the result at once (P6: a serial segment)
    Count,
};
inline constexpr size_t NumKinds = static_cast<size_t>(Kind::Count);
inline constexpr std::array<const char*, NumKinds> kind_names{
    "buffer create", "buffer upload", "view create", "view bind",  "image prepare",
    "sampler create", "sampler bind", "rt update",  "rt prepare", "rescale",
    "external",      "writeback",    "serial",
};

/// What applying a request did (hooks inside the caches). The old in-lookup path notes the same
/// effects, which is what verify compares.
enum Effect : u32 {
    EffectNone = 0,
    EffectCreate = 1U << 0,  ///< an image, view, buffer or sampler was made
    EffectUpload = 1U << 1,  ///< guest bytes went to the upload stream
    EffectAlias = 1U << 2,   ///< alias copies
    EffectRescale = 1U << 3, ///< ScaleUp/ScaleDown changed an image
    EffectDelete = 1U << 4,  ///< an image or buffer was deleted
    EffectModify = 1U << 5,  ///< a GPU modification mark
    NumEffects = 6,
};
inline constexpr std::array<const char*, NumEffects> effect_names{
    "create", "upload", "alias", "rescale", "delete", "modify",
};

enum class Resource : u8 {
    None,
    Buffer,
    Image,
    ImageView,
    Sampler,
    RenderTargets,
    Range,
};

/// Slots are u32 indices (SlotId). Proxies use the top bit; slot vectors never get that large.
inline constexpr u32 InvalidSlot = ~0U;
inline constexpr u32 ProxyBit = 0x8000'0000U;
[[nodiscard]] constexpr bool IsProxy(u32 slot) {
    return slot != InvalidSlot && (slot & ProxyBit) != 0;
}
[[nodiscard]] constexpr u32 MakeProxy(u32 index) {
    return ProxyBit | index;
}
[[nodiscard]] constexpr u32 ProxyIndex(u32 slot) {
    return slot & ~ProxyBit;
}

/// Sequence = segment (P5a: one per draw/dispatch; P6: an IR segment) and the index within it.
inline constexpr u32 SegmentIndexBits = 24;
[[nodiscard]] constexpr u64 MakeSequence(u64 segment, u32 index) {
    return (segment << SegmentIndexBits) | index;
}
[[nodiscard]] constexpr u64 SegmentOf(u64 sequence) {
    return sequence >> SegmentIndexBits;
}

struct Request {
    u64 sequence = 0;
    u64 address = 0; ///< device/GPU address, or a descriptor index
    u64 size = 0;
    u32 slot = InvalidSlot;  ///< resource the request is on (may itself be a proxy)
    u32 proxy = InvalidSlot; ///< proxy the commit patches with the applier's result
    u32 arg = 0;             ///< kind-specific (flags)
    u32 effects = 0;         ///< effects the lookup predicted (verify)
    Kind kind = Kind::Serial;
    Resource resource = Resource::None;
    u8 reserved[6]{};
};
static_assert(std::is_trivially_copyable_v<Request>);
static_assert(sizeof(Request) == 48);

/// Request::arg flags.
inline constexpr u32 ArgCompute = 1U << 0;
inline constexpr u32 ArgModification = 1U << 1;
inline constexpr u32 ArgInvalidate = 1U << 2;
inline constexpr u32 ArgBlacklist = 1U << 3;
inline constexpr u32 ArgDirect = 1U << 4; ///< prepare the slot as is (no descriptor lookup)

/// One producer's requests for one segment, in program order, with the segment's proxies.
class RequestList {
public:
    void Reset(u64 segment_) {
        segment = segment_;
        requests.clear();
        proxies.clear();
        serial = false;
    }

    /// Appends a request; with `wants_proxy` reserves a proxy for its result and returns it.
    u32 Push(Request request, bool wants_proxy = false) {
        request.sequence = MakeSequence(segment, static_cast<u32>(requests.size()));
        if (wants_proxy) {
            request.proxy = MakeProxy(static_cast<u32>(proxies.size()));
            proxies.push_back(InvalidSlot);
        }
        requests.push_back(request);
        return request.proxy;
    }

    /// The commit stage's patch of a proxy.
    void Patch(u32 proxy, u32 slot) {
        if (IsProxy(proxy) && ProxyIndex(proxy) < proxies.size()) {
            proxies[ProxyIndex(proxy)] = slot;
        }
    }

    /// A proxy's patched slot (InvalidSlot until patched); other ids are returned unchanged.
    [[nodiscard]] u32 Resolve(u32 id) const {
        if (!IsProxy(id)) {
            return id;
        }
        const u32 index = ProxyIndex(id);
        return index < proxies.size() ? proxies[index] : InvalidSlot;
    }

    [[nodiscard]] bool AllPatched() const {
        return std::ranges::none_of(proxies, [](u32 slot) { return slot == InvalidSlot; });
    }

    void MarkSerial() {
        serial = true;
    }
    [[nodiscard]] bool Serial() const {
        return serial;
    }
    [[nodiscard]] u64 Segment() const {
        return segment;
    }
    [[nodiscard]] std::span<const Request> Requests() const {
        return requests;
    }
    [[nodiscard]] std::span<Request> Requests() {
        return requests;
    }
    [[nodiscard]] bool Empty() const {
        return requests.empty();
    }
    [[nodiscard]] size_t ProxyCount() const {
        return proxies.size();
    }

private:
    u64 segment = 0;
    std::vector<Request> requests;
    std::vector<u32> proxies;
    bool serial = false;
};

/// The commit owner's ordering. Lists are applied whole, strictly in segment order, each in
/// request order; `apply(request)` returns the real slot for the request's proxy.
class Stage {
public:
    struct Counters {
        std::array<u64, NumKinds> requests{};
        u64 proxies_patched = 0;
        u64 segments = 0;
        u64 out_of_order = 0; ///< segments that arrived before an older one (P6)
    };

    /// Applies one list now (P5a: the producer is the owner, segments arrive in order).
    template <typename Fn>
    void Apply(RequestList& list, Fn&& apply) {
        for (Request& request : list.Requests()) {
            ++counters.requests[static_cast<size_t>(request.kind)];
            if (IsProxy(request.slot)) {
                // A request on an earlier request's result (a prepare of a view created first).
                request.slot = list.Resolve(request.slot);
            }
            const u32 slot = apply(static_cast<const Request&>(request));
            if (IsProxy(request.proxy)) {
                list.Patch(request.proxy, slot);
                ++counters.proxies_patched;
            }
        }
        ++counters.segments;
        next_segment = (std::max)(next_segment, list.Segment() + 1);
    }

    /// P6: any thread hands a finished segment to the commit thread.
    void Submit(RequestList* list) {
        Node* const node = new Node{list, inbox.load(std::memory_order_relaxed)};
        while (!inbox.compare_exchange_weak(node->next, node, std::memory_order_release,
                                            std::memory_order_relaxed)) {
        }
    }

    /// Commit thread: applies every submitted segment that is next in order; returns the lists
    /// applied (in order) so their owners can read the patched proxies.
    template <typename Fn>
    std::vector<RequestList*> CommitReady(Fn&& apply) {
        for (Node* node = inbox.exchange(nullptr, std::memory_order_acquire); node != nullptr;) {
            Node* const next = node->next;
            if (node->list->Segment() != next_segment) {
                ++counters.out_of_order;
            }
            pending.emplace(node->list->Segment(), node->list);
            delete node;
            node = next;
        }
        std::vector<RequestList*> applied;
        while (!pending.empty() && pending.begin()->first <= next_segment) {
            RequestList* const list = pending.begin()->second;
            pending.erase(pending.begin());
            Apply(*list, apply);
            next_segment = list->Segment() + 1;
            applied.push_back(list);
        }
        return applied;
    }

    /// The next segment the stage will apply (P6 segmenters number from here).
    [[nodiscard]] u64 NextSegment() const {
        return next_segment;
    }
    void SetNextSegment(u64 segment) {
        next_segment = segment;
    }

    [[nodiscard]] Counters TakeCounters() {
        return std::exchange(counters, Counters{});
    }

    ~Stage() {
        for (Node* node = inbox.exchange(nullptr); node != nullptr;) {
            Node* const next = node->next;
            delete node;
            node = next;
        }
    }

private:
    struct Node {
        RequestList* list;
        Node* next;
    };

    std::atomic<Node*> inbox{nullptr};
    std::map<u64, RequestList*> pending; ///< Commit thread only
    u64 next_segment = 0;
    Counters counters;
};

/// What another thread asks the owner to do with guest memory state.
enum class ExternalKind : u32 {
    CpuWrite,        ///< a trapped guest CPU write (page aligned)
    Invalidate,      ///< InvalidateRegion(address, size, arg = CacheType)
    CacheInvalidate, ///< OnCacheInvalidation(address, size)
    ClaimInvalidate, ///< OnCacheInvalidations: claimed from GPU-written ranges first
    Unmap,           ///< UnmapMemory(address, size)
    UnmapGpu,        ///< ModifyGPUMemory(arg = address space, GPU address, size)
    ClearDownload,   ///< PrepareRegionWrite(address, size)
    Call,            ///< a synchronous owner call (address = OwnerCall*)
    Count,
};
inline constexpr size_t NumExternalKinds = static_cast<size_t>(ExternalKind::Count);
inline constexpr std::array<const char*, NumExternalKinds> external_names{
    "cpu write", "invalidate", "cache invalidate", "claim", "unmap", "unmap gpu", "clear download",
    "call",
};

struct ExternalOp {
    u64 address = 0;
    u64 size = 0;
    ExternalKind kind = ExternalKind::CpuWrite;
    u32 arg = 0;
};
static_assert(std::is_trivially_copyable_v<ExternalOp>);

/// A synchronous call another thread waits on (lives on the caller's stack).
struct OwnerCall {
    void (*function)(void* context) = nullptr;
    void* context = nullptr;
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;

    void Run() {
        function(context);
        {
            std::scoped_lock lock{mutex};
            done = true;
        }
        cv.notify_all();
    }
};

/// Bounded multi-producer single-consumer queue (Vyukov). Push never blocks: when full, write
/// operations widen an overflow range ("flood") the owner applies as one invalidation, which is
/// always safe for writes (it only uploads more). Calls retry until there is room.
class ExternalQueue {
public:
    static constexpr size_t Capacity = 4096;

    ExternalQueue() : cells{std::make_unique<Cell[]>(Capacity)} {
        for (size_t index = 0; index < Capacity; ++index) {
            cells[index].sequence.store(index, std::memory_order_relaxed);
        }
    }

    /// Any thread. False when full.
    bool TryPush(const ExternalOp& op) {
        u64 position = tail.load(std::memory_order_relaxed);
        for (;;) {
            Cell& cell = cells[position & (Capacity - 1)];
            const u64 sequence = cell.sequence.load(std::memory_order_acquire);
            const s64 difference = static_cast<s64>(sequence) - static_cast<s64>(position);
            if (difference == 0) {
                if (tail.compare_exchange_weak(position, position + 1,
                                               std::memory_order_relaxed)) {
                    cell.op = op;
                    cell.sequence.store(position + 1, std::memory_order_release);
                    return true;
                }
            } else if (difference < 0) {
                return false;
            } else {
                position = tail.load(std::memory_order_relaxed);
            }
        }
    }

    /// Any thread: a write that didn't fit. The owner treats the union as written.
    void Flood(u64 address, u64 size) {
        u64 begin = flood_begin.load(std::memory_order_relaxed);
        while (address < begin && !flood_begin.compare_exchange_weak(
                                      begin, address, std::memory_order_relaxed)) {
        }
        u64 end = flood_end.load(std::memory_order_relaxed);
        while (address + size > end &&
               !flood_end.compare_exchange_weak(end, address + size, std::memory_order_relaxed)) {
        }
        flooded.store(true, std::memory_order_release);
    }

    /// Owner: true when nothing was pushed or flooded since the last drain (one acquire load
    /// per cell checked, the common case).
    [[nodiscard]] bool Empty() const {
        const Cell& cell = cells[head & (Capacity - 1)];
        return cell.sequence.load(std::memory_order_acquire) != head + 1 &&
               !flooded.load(std::memory_order_acquire);
    }

    /// Owner: pops in push order. Returns how many were popped.
    template <typename Func>
    size_t Drain(Func&& func) {
        size_t popped = 0;
        for (;;) {
            Cell& cell = cells[head & (Capacity - 1)];
            const u64 sequence = cell.sequence.load(std::memory_order_acquire);
            if (sequence != head + 1) {
                break;
            }
            const ExternalOp op = cell.op;
            cell.sequence.store(head + Capacity, std::memory_order_release);
            ++head;
            ++popped;
            func(op);
        }
        return popped;
    }

    /// Owner: the flood range, if any, cleared.
    bool TakeFlood(u64& address, u64& size) {
        if (!flooded.exchange(false, std::memory_order_acq_rel)) {
            return false;
        }
        const u64 begin = flood_begin.exchange(~u64{0}, std::memory_order_relaxed);
        const u64 end = flood_end.exchange(0, std::memory_order_relaxed);
        address = begin;
        size = end > begin ? end - begin : 0;
        return size != 0;
    }

private:
    struct alignas(64) Cell {
        std::atomic<u64> sequence;
        ExternalOp op;
    };

    std::unique_ptr<Cell[]> cells;
    alignas(64) std::atomic<u64> tail{0};
    alignas(64) u64 head{0}; ///< Owner only
    std::atomic<bool> flooded{false};
    std::atomic<u64> flood_begin{~u64{0}};
    std::atomic<u64> flood_end{0};
};

/// The GPU thread sets this at start: with the split stopped it executes GPU work and owns the
/// caches; while the split runs, its back does (GpuSplit::OnBackThread). The rasterizer combines
/// the two into "is the calling thread the owner".
inline thread_local bool t_gpu_thread = false;

/// Whether an owner exists yet. Before the first claim (boot, before the GPU thread executes
/// anything), other threads use the caches directly; the claim waits for those in flight, and
/// from then on they hand their work to the owner. Ownership itself moves only at quiescent
/// points (the split starting or stopping), which is why "who" is not stored here.
class Owner {
public:
    [[nodiscard]] bool HasOwner() const {
        return owned.load(std::memory_order_acquire);
    }

    /// The owner's first cache access: from now on other threads route through it.
    void Claim() {
        if (owned.load(std::memory_order_relaxed)) {
            return;
        }
        owned.store(true, std::memory_order_seq_cst);
        // A thread that started direct access before this claim finishes first.
        while (direct_users.load(std::memory_order_seq_cst) != 0) {
            std::this_thread::yield();
        }
    }

    /// External thread: true when there is no owner and it may use the caches directly until
    /// EndDirect. False: route through the owner.
    [[nodiscard]] bool BeginDirect() {
        direct_users.fetch_add(1, std::memory_order_seq_cst);
        if (!owned.load(std::memory_order_seq_cst)) {
            return true;
        }
        direct_users.fetch_sub(1, std::memory_order_release);
        return false;
    }
    void EndDirect() {
        direct_users.fetch_sub(1, std::memory_order_release);
    }

private:
    std::atomic<bool> owned{false};
    std::atomic<u32> direct_users{0};
};

/// DXVK's per-resource track id: the last submission (scheduler tick) that used the resource.
/// Held by value in the resource (slot vectors move their elements), written by the commit owner
/// only. A request on a resource the current submission hasn't used yet may be hoisted ahead of
/// that submission's work (P5b's transfer queue / init stream).
struct TrackId {
    u64 last = 0;

    /// Marks a use in `submission`; true the first time in that submission.
    bool Track(u64 submission) {
        if (last >= submission) {
            return false;
        }
        last = submission;
        return true;
    }
    [[nodiscard]] bool UsedIn(u64 submission) const {
        return last >= submission;
    }
};

/// Mutations noted while a scope runs (verify): the old path's, or the commit's.
struct LogEntry {
    Kind kind = Kind::Serial;
    Resource resource = Resource::None;
    u32 slot = InvalidSlot;
    u32 effects = 0;
};

class MutationLog {
public:
    void Begin() {
        entries.clear();
        loose_effects = 0;
    }
    /// A request-equivalent step (a lookup of the old path, or a request being applied).
    void Step(Kind kind, Resource resource, u32 slot) {
        entries.push_back(LogEntry{kind, resource, slot, 0});
    }
    /// An effect of the current step (or before any step: loose).
    void NoteEffect(u32 effect) {
        if (entries.empty()) {
            loose_effects |= effect;
        } else {
            entries.back().effects |= effect;
        }
    }
    /// The slot the current step resolved to (a created view's id).
    void NoteSlot(u32 slot) {
        if (!entries.empty()) {
            entries.back().slot = slot;
        }
    }
    [[nodiscard]] std::span<const LogEntry> Entries() const {
        return entries;
    }
    [[nodiscard]] u32 LooseEffects() const {
        return loose_effects;
    }

private:
    std::vector<LogEntry> entries;
    u32 loose_effects = 0;
};

/// The verify log of the owner thread, null unless verify records (one load per hook).
inline MutationLog* g_log = nullptr;

inline void LogStep(Kind kind, Resource resource, u32 slot) {
    if (g_log != nullptr) [[unlikely]] {
        g_log->Step(kind, resource, slot);
    }
}
inline void LogEffect(u32 effect) {
    if (g_log != nullptr) [[unlikely]] {
        g_log->NoteEffect(effect);
    }
}
inline void LogSlot(u32 slot) {
    if (g_log != nullptr) [[unlikely]] {
        g_log->NoteSlot(slot);
    }
}

struct CompareResult {
    size_t compared = 0;
    size_t mismatches = 0; ///< differences no earlier step explains
    size_t dependent = 0;  ///< differences after an earlier create/delete/alias/rescale
    size_t first_mismatch = ~size_t{0};
    enum class Field : u8 { None, Count, Kind, Slot, Effects } field = Field::None;
};

/// Compares the requests the pure lookups predicted with what the old path did, in order: same
/// count, kinds and resources (a proxy or InvalidSlot matches any slot), and the predicted effects
/// (none predicted for those). Once a step created, deleted, alias-copied or rescaled anything,
/// later differences count as dependent (a modification too: it changes later alias decisions):
/// the old path re-read state an earlier mutation
/// changed, which commit orders through proxies and re-checks when it applies.
[[nodiscard]] CompareResult Compare(std::span<const Request> predicted,
                                    std::span<const LogEntry> actual);

/// What PrepareImage will do to an image, from its state (pure; the texture cache fills it).
struct PrepareInputs {
    bool prepares = false;    ///< the view prepares an image (not null, not a buffer view)
    bool cpu_modified = false;
    bool newer_alias = false; ///< an aliased image holds a newer modification tick
    bool modification = false;
    bool invalidate = false;
};
[[nodiscard]] constexpr u32 PredictPrepare(const PrepareInputs& in) {
    if (!in.prepares) {
        return EffectNone;
    }
    u32 effects = EffectNone;
    if (!in.invalidate) {
        if (in.cpu_modified) {
            effects |= EffectUpload;
        }
        if (in.newer_alias) {
            effects |= EffectAlias;
        }
    }
    if (in.modification) {
        effects |= EffectModify;
    }
    return effects;
}

/// A render target image on a recipe replay with clean render target registers (P5a item 4).
struct RenderTargetState {
    bool generation_matches = false; ///< the recipe's view is still the bound, live view
    bool cpu_modified = false;
    bool newer_alias = false;
    bool gpu_modified = false;
    bool newest_modification = false; ///< no image was marked modified after it
    bool special = false;             ///< sparse/unwritten-sparse handling on modification
    bool touched_this_frame = false;  ///< the LRU already holds this frame for it
};
/// Whether drawing into the target needs a prepare request at all. Re-marking an image that is
/// already GPU-modified and holds the newest modification tick changes no ordering decision (alias
/// resolution compares ticks, and it stays the newest), so it is skipped; anything else (an upload,
/// a newer alias, a first GPU write, sparse handling, a changed view, the frame's first LRU touch)
/// prepares as before.
[[nodiscard]] constexpr bool RenderTargetNeedsPrepare(const RenderTargetState& in) {
    return !in.generation_matches || in.cpu_modified || in.newer_alias || !in.gpu_modified ||
           !in.newest_modification || in.special || !in.touched_this_frame;
}

/// Effects whose prediction is exact from flags (refresh, alias sync, modification).
inline constexpr u32 PredictedEffects = EffectUpload | EffectAlias | EffectModify;

} // namespace VideoCommon::V2::Commit

namespace VideoCommon::V2::Commit {

/// Renderer-wide commit state, owned by the rasterizer and shared by both caches (each holds a
/// pointer that is null with EDEN_NX2_V2_COMMIT off, so the old code runs behind one branch).
struct Context {
    bool verify = false;    ///< COMMIT=verify: the old path runs; predictions are compared
    bool lock_free = false; ///< the draw path takes no cache lock (COMMIT=1, not COMMIT_LOCKED)
    Stage stage;
    RequestList list;         ///< the open segment's requests (P5a: this draw's)
    u64 next_segment = 1;
    MutationLog log;          ///< verify: what the old path did
    Owner owner;
    ExternalQueue external;
    u64 submission = 0;       ///< the owner's current scheduler tick (track ids)
    bool stats = false;
    bool writeback = false;   ///< EDEN_NX2_V2_WRITEBACK: owed downloads wait for a guest sync

    /// Opens a segment for the lookups that follow.
    void BeginSegment() {
        list.Reset(next_segment++);
    }
};

/// The domain that retires the caches' published lookup tables (process-wide).
EpochDomain& Epochs();

/// Counts an effect in the stats (effects are single bits).
void CountEffect(u32 effect);

/// An effect of the request being applied (or of the old path, in verify).
inline void NoteEffect(Context* context, u32 effect) {
    if (context->stats) {
        CountEffect(effect);
    }
    LogEffect(effect);
}

} // namespace VideoCommon::V2::Commit

namespace VideoCommon::V2::Commit::Stats {

/// EDEN_NX2_V2_STATS counters of the commit stage (any thread; relaxed atomics).
struct Data {
    std::array<std::atomic<u64>, NumKinds> requests{};
    std::array<std::atomic<u64>, NumEffects> effects{};
    std::atomic<u64> commits{};
    std::atomic<u64> commit_ns{};
    std::atomic<u64> lookup_hits{};   ///< lock-free lookups that returned a ready resource
    std::atomic<u64> lookup_misses{}; ///< lock-free lookups that returned a request
    std::atomic<u64> proxies_patched{};
    std::array<std::atomic<u64>, NumExternalKinds> external{};
    std::atomic<u64> external_direct{}; ///< external calls made before any owner existed
    std::atomic<u64> floods{};
    std::atomic<u64> owner_calls{};
    std::atomic<u64> fence_backlog{};  ///< fence batches handed to the owner (async flush pops)
    std::atomic<u64> hoistable{};      ///< uploads to resources unused in this submission
    std::atomic<u64> writeback_queued{};
    std::atomic<u64> writeback_flushed{};
    std::atomic<u64> writeback_dropped{};
    std::atomic<u64> rt_skipped{};     ///< replays whose render targets needed no request
    std::atomic<u64> rt_requests{};
    std::atomic<u64> verify_draws{};
    std::atomic<u64> verify_mismatches{};
    std::atomic<u64> verify_dependent{};
};

Data& Get();

inline void Count(std::atomic<u64>& counter, u64 amount = 1) {
    counter.fetch_add(amount, std::memory_order_relaxed);
}

/// Logs and resets (Stats::ReportSlow, the drawing thread).
void Report(const char* thread_name, u64 window_ns);

} // namespace VideoCommon::V2::Commit::Stats
