// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/residency.h"
#include "video_core/v2/versions.h"

// Pipeline v2 P4 (EDEN_NX2_V2_DRAWCACHE): fingerprint -> recipe draw cache (MASTER_PLAN §5.3
// step 2, DIAGRAMS 2.4; AnyPS5 DrawCache/Recipe, notes/05 §3.2).
//
// Key. Two levels, as AnyPS5's shape key and exact key:
//   - shape = the P1 fingerprint of the pipeline-relevant register groups (pipeline, viewport,
//     shaders, render targets) + topology, indexed-ness and the key's non-register inputs. The
//     shape table remembers, from the last resolve of a shape, what the bound shaders use: whether
//     they read vertex attributes and the texture pools (P3's relevance masking), and which constant
//     buffer slots hold texture handles or storage buffer descriptors.
//   - key = shape + the fingerprint of the relevant binding groups (vertex streams if read,
//     texture pools if read) + the identities {address, size, enabled} of the constant buffers that
//     hold handles or storage descriptors. Uniform-only constant buffers are not in the key: the
//     buffer cache binds them (P3 versions) per draw anyway.
// Compute: the same over the launch description (grid dimensions and constant buffer addresses
// out of the shape), the program address and the texture pools.
//
// Recipe. Built after a full resolve of a cacheable draw: the pipeline (weak: pointer + the
// pipeline epoch), the image views and samplers (weak: slot id + generation), the storage buffer
// bindings, the render target views, the push data, and the captured guest inputs with the stamp V
// taken before the resolve read them:
//   - words: constant buffer words holding handles, the TIC/TSC entries they select, storage buffer
//     descriptors; each with its device address (versions), host pointer and bytes (compare);
//   - content ranges: the guest pages of the sampled images and of the bound buffers.
// A hit validates the words through the epoch-memoised versions (pages unchanged since V vouch for
// them; otherwise the bytes are compared, AnyPS5's model), the content ranges by version, and every
// weak handle by generation. Any change is a miss and the dirty-group resolve rebuilds the recipe.
namespace VideoCommon::V2::DrawCache {

[[nodiscard]] constexpr u64 Mix(u64 hash, u64 value) {
    u64 x = hash ^ (value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/// Hash of a word span (compute launch descriptions).
[[nodiscard]] inline u64 HashWords(std::span<const u32> words, u64 seed) {
    u64 hash = seed;
    for (const u32 word : words) {
        hash = Mix(hash, word);
    }
    return hash;
}

/// What the shape table remembers from the last resolve of a shape.
struct ShapeInfo {
    u64 shape{};
    u32 relevant_groups{}; ///< binding groups (RegGroup bits) the bound shaders read
    std::array<u32, 5> key_cbufs{}; ///< per stage: constant buffer slots in the key
    bool valid{};
};

/// Direct-mapped shape table (owned by the resolving thread).
class ShapeTable {
public:
    static constexpr size_t Entries = 4096;

    [[nodiscard]] const ShapeInfo* Find(u64 shape) const {
        const ShapeInfo& info = entries[Slot(shape)];
        return info.valid && info.shape == shape ? &info : nullptr;
    }
    void Insert(const ShapeInfo& info) {
        entries[Slot(info.shape)] = info;
        entries[Slot(info.shape)].valid = true;
    }
    void Clear() {
        entries.fill(ShapeInfo{});
    }

private:
    static size_t Slot(u64 shape) {
        return static_cast<size_t>((shape * 0x9e3779b97f4a7c15ULL) >> 52) % Entries;
    }
    std::array<ShapeInfo, Entries> entries{};
};

/// Admission control (AnyPS5's probation): a key gets a recipe on its second sighting, so keys
/// seen once (most of a scene's unique draws) never capture, insert or evict. Direct mapped: a
/// key whose slot another key took since counts as unseen again.
class SeenFilter {
public:
    static constexpr size_t Entries = 8192;

    /// Whether `key` was seen before (and still is remembered); remembers it either way.
    [[nodiscard]] bool Admit(u64 key) {
        const u64 tag = key | 1; // 0 is the empty slot
        u64& slot = tags[static_cast<size_t>((key * 0x9e3779b97f4a7c15ULL) >> 51) % Entries];
        if (slot == tag) {
            return true;
        }
        slot = tag;
        return false;
    }
    void Clear() {
        tags.fill(0);
    }

private:
    std::array<u64, Entries> tags{};
};

/// A captured guest input word run: where it is (device address for versions, host pointer for the
/// compare) and the bytes the resolve read.
struct Capture {
    enum class What : u8 {
        Handle,     ///< constant buffer word holding a texture handle
        Tic,        ///< texture header
        Tsc,        ///< sampler descriptor
        StorageDesc, ///< storage buffer descriptor (address + size words)
        Count,
    };
    const u8* host{};
    DAddr device{};
    u32 size{}; ///< bytes, <= 32
    What what{};
    std::array<u8, 32> bytes{};
};
inline constexpr size_t NumCaptureKinds = static_cast<size_t>(Capture::What::Count);

/// A guest range whose content the recipe depends on (sampled image pages, bound buffers).
struct ContentRange {
    DAddr device{};
    u64 size{};
    bool image{};
};

/// The inputs of one recipe and the stamp they were read after.
struct CapturedInputs {
    u64 stamp{};
    std::vector<Capture> words;
    std::vector<ContentRange> content;

    void Clear() {
        stamp = 0;
        words.clear();
        content.clear();
    }
};

/// Why a lookup did not replay.
enum class Miss : u8 {
    Ineligible,    ///< the draw or its pipeline can't be cached (or the cache is off for it)
    NoShape,       ///< the shape was never resolved
    NoEntry,       ///< no recipe for the key
    Pipeline,      ///< pipeline evicted, rebuilt or its shader invalidated
    Generation,    ///< an image view, sampler or render target was freed or recreated
    Epoch,         ///< rescale, map or host pointer epoch moved
    Tables,        ///< the texture/sampler descriptor tables changed
    Handle,        ///< a constant buffer word holding a handle changed
    Tic,           ///< a texture header changed
    Tsc,           ///< a sampler descriptor changed
    StorageDesc,   ///< a storage buffer descriptor changed
    ImagePages,    ///< a sampled image's guest pages were written
    BufferPages,   ///< (reported only: buffer pages are synchronized on the replay)
    RenderTargets, ///< the render targets resolved to other views
    Prepare,       ///< preparing the sampled images deleted images
    Count,
};
inline constexpr size_t NumMisses = static_cast<size_t>(Miss::Count);
inline constexpr std::array<const char*, NumMisses> miss_names{
    "ineligible", "no shape",   "no entry",  "pipeline",     "generation",
    "epoch",      "tables",     "handle",    "tic",          "tsc",
    "storage descriptor", "image pages", "buffer pages", "render targets", "prepare"};

[[nodiscard]] constexpr Miss MissFor(Capture::What what) {
    switch (what) {
    case Capture::What::Handle:
        return Miss::Handle;
    case Capture::What::Tic:
        return Miss::Tic;
    case Capture::What::Tsc:
        return Miss::Tsc;
    default:
        return Miss::StorageDesc;
    }
}

/// Validates captured inputs: content ranges by version since the stamp (through the epoch memo;
/// without `use_memo` every range is walked, verify), the words by comparing them with memory.
/// Returns Miss::Count when the inputs hold. Buffer content ranges never fail it: with
/// `buffers_changed` they are checked and reported (the buffer cache synchronizes them on the
/// replay). `restamped` is kept for callers and never set (words need no stamp).
[[nodiscard]] Miss ValidateInputs(CapturedInputs& inputs, Versions::EpochMemo& memo,
                                  Versions::PageVersions& versions, bool use_memo = true,
                                  bool* restamped = nullptr, bool* buffers_changed = nullptr);

/// A view of a recipe: TIC index (as the resolve fills ImageViewInOut), slot id and generation.
struct RecipeView {
    u32 index{};
    u32 id{};
    u32 generation{};
    bool blacklist{};
};

/// A storage buffer binding as the buffer cache resolved it (its Binding without the buffer id).
struct RecipeStorage {
    u8 stage{};
    u8 index{};
    bool written{};
    DAddr device_addr{};
    GPUVAddr gpu_addr{};
    u32 size{};
    u32 segment_first{};
    u32 segment_count{};

    bool operator==(const RecipeStorage&) const = default;
};

/// A guest word run the resolve read, with the bytes it read (recorded by the pipeline's
/// Configure while a capture is set).
struct InputRead {
    GPUVAddr address{};
    u32 size{}; ///< 4 (handle word) or 16 (storage buffer descriptor)
    Capture::What what{};
    std::array<u8, 16> bytes{};
};

/// What a capturing Configure records for the recipe builder.
struct RecipeCapture {
    bool ineligible{}; ///< something data-dependent was read (computed handles, table pointers)
    std::vector<InputRead> reads;
    std::vector<RecipeView> views;   ///< generations filled in by the builder
    std::vector<u32> samplers;       ///< SamplerId values
    std::vector<u32> tsc_indices;    ///< per sampled texture: the sampler index of its handle
    std::array<u32, 32> push{};
    u32 push_words{};
    std::vector<u32> heap_words;

    void Clear() {
        ineligible = false;
        reads.clear();
        views.clear();
        samplers.clear();
        tsc_indices.clear();
        push = {};
        push_words = 0;
        heap_words.clear();
    }
    void AddWord(GPUVAddr address, u32 value) {
        InputRead read{.address = address, .size = 4, .what = Capture::What::Handle};
        std::memcpy(read.bytes.data(), &value, sizeof(value));
        reads.push_back(read);
    }
};

inline constexpr size_t NumRecipeTargets = 9; ///< 8 colour + depth

/// One buffer binding of a draw as the buffer cache bound it: the host buffer (weak: slot id +
/// generation), the guest range it covers, and for uniform buffers the path it took.
struct BufferBind {
    enum class Kind : u8 {
        Cached,  ///< bound from the host buffer (the buffer cache's cached path)
        Version, ///< a P3 constant buffer version in the stream buffer (`ref`, made in `tick`)
        Normal,  ///< a path the replay doesn't take (stream copy without versions, root table,
                 ///< native 64 KiB bank): bound by the buffer cache as always
    };
    u8 stage{};
    u8 index{};
    Kind kind{Kind::Cached};
    u8 flags{}; ///< bit 0: image texture buffer
    u32 buffer_id{};
    u32 generation{};
    DAddr device_addr{};
    u32 size{};
    u32 format{};
    std::array<u64, 3> ref{}; ///< the runtime's stream uniform reference (Version)
    u64 tick{};

    /// Whether [address, address + size) lies within this binding's range.
    [[nodiscard]] bool Contains(DAddr address, u64 bytes) const {
        return address >= device_addr && address + bytes <= device_addr + size;
    }
    [[nodiscard]] bool SameBinding(const BufferBind& other) const {
        return stage == other.stage && index == other.index && buffer_id == other.buffer_id &&
               device_addr == other.device_addr && size == other.size && kind == other.kind;
    }
};

/// The buffer bindings of a draw recipe (EDEN_NX2_V2_DRAWCACHE buffer replay). `stamp` is taken
/// before the resolve that synchronized them and never restamped with the words: a binding is
/// replayed (no FindBuffer, no SynchronizeBuffer, no uniform copy) only while its guest range is
/// unchanged since then and its buffer's generation is the one recorded.
struct BufferRecipe {
    bool valid{};
    u64 stamp{};
    bool has_index{};
    BufferBind index;
    std::vector<BufferBind> vertex;  ///< index = vertex buffer slot
    std::vector<BufferBind> uniform; ///< stage, index = constant buffer slot
    std::vector<BufferBind> storage; ///< stage, index = storage buffer slot
    std::vector<BufferBind> texture; ///< stage, index = texture buffer slot
    /// F2-07 (§5.3.2, one validation per recipe): the recorded ranges merged, and the write
    /// sequence before the last check that found all of them unchanged since `stamp` (while no
    /// write reached their buckets since, they still are: no walk, no per-binding query).
    std::vector<std::pair<DAddr, DAddr>> merged;
    mutable u64 checked_sequence{};
    mutable bool checked{};
    /// The buffer cache's deletion count when every recorded buffer was last found alive.
    mutable u64 alive_deletions{~0ULL};

    void Clear() {
        valid = false;
        stamp = 0;
        has_index = false;
        index = {};
        vertex.clear();
        uniform.clear();
        storage.clear();
        texture.clear();
        merged.clear();
        checked = false;
        checked_sequence = 0;
        alive_deletions = ~0ULL;
    }
    /// Builds `merged` from the bindings (after recording them).
    void MergeRanges() {
        merged.clear();
        const auto add = [&](const BufferBind& bind) {
            if (bind.size != 0) {
                merged.emplace_back(bind.device_addr, bind.device_addr + bind.size);
            }
        };
        if (has_index) {
            add(index);
        }
        for (const auto* list : {&vertex, &uniform, &storage, &texture}) {
            for (const BufferBind& bind : *list) {
                add(bind);
            }
        }
        std::sort(merged.begin(), merged.end());
        size_t out = 0;
        for (size_t i = 0; i < merged.size(); ++i) {
            if (out != 0 && merged[i].first <= merged[out - 1].second) {
                merged[out - 1].second = (std::max)(merged[out - 1].second, merged[i].second);
            } else {
                merged[out++] = merged[i];
            }
        }
        merged.resize(out);
    }
    /// Whether every recorded range is unchanged since `stamp` (one check per recipe while
    /// nothing wrote their buckets; else one memo query per merged range).
    [[nodiscard]] bool AllUnchanged(Versions::EpochMemo& memo,
                                    const Versions::PageVersions& versions) const {
        if (checked) {
            bool written = false;
            for (const auto& [begin, end] : merged) {
                if (versions.WrittenSince(checked_sequence, begin, end)) {
                    written = true;
                    break;
                }
            }
            if (!written) {
                return true;
            }
        }
        const u64 sequence = versions.Sequence();
        for (const auto& [begin, end] : merged) {
            if (!memo.Unchanged(begin, end - begin, stamp)) {
                checked = false;
                return false;
            }
        }
        checked = true;
        checked_sequence = sequence;
        return true;
    }
    [[nodiscard]] static const BufferBind* Find(const std::vector<BufferBind>& binds, u32 stage,
                                                u32 index) {
        for (const BufferBind& bind : binds) {
            if (bind.stage == stage && bind.index == index) {
                return &bind;
            }
        }
        return nullptr;
    }
};

/// Compares the buffer bindings a replay of `hit` would bind with the ones a full resolve bound
/// (`fresh`): a binding the replay takes from the recipe (its range holds the fresh one, same
/// path, its buffer's generation unchanged) must have the same buffer. Appends "buffers (...)"
/// to `reasons` on a difference.
[[nodiscard]] bool CompareBufferRecipes(const BufferRecipe& hit, const BufferRecipe& fresh,
                                        const Versions::Generations& gens, std::string& reasons);

} // namespace VideoCommon::V2::DrawCache

namespace VideoCommon::V2::Resolve {
struct ResolvedDraw;
}

namespace VideoCommon::V2::DrawCache {

/// P6 Step 0 (step 15, EDEN_NX2_V2_DRAWREC): a recipe's ResolvedDraw template, proxies patched.
/// Held through a pointer because v2/resolve/draw_record.h includes this header; copies are deep
/// (draw_record.cpp). Invalidate keeps the storage for the next build.
class RecordTemplate {
public:
    RecordTemplate() = default;
    RecordTemplate(const RecordTemplate& other);
    RecordTemplate& operator=(const RecordTemplate& other);
    RecordTemplate(RecordTemplate&&) noexcept = default;
    RecordTemplate& operator=(RecordTemplate&&) noexcept = default;
    ~RecordTemplate() = default;

    [[nodiscard]] bool Valid() const noexcept {
        return valid;
    }
    /// The template; only while Valid.
    [[nodiscard]] const Resolve::ResolvedDraw& Get() const noexcept {
        return *record;
    }
    void Store(const Resolve::ResolvedDraw& source);
    void Invalidate() noexcept {
        valid = false;
    }

private:
    struct Deleter {
        void operator()(Resolve::ResolvedDraw* pointer) const noexcept;
    };
    std::unique_ptr<Resolve::ResolvedDraw, Deleter> record;
    bool valid{};
};

/// Everything a hit needs to record the draw (or dispatch) without resolving it again.
struct Recipe {
    u64 key{};
    void* pipeline{};            ///< GraphicsPipeline* / ComputePipeline* (weak, see epochs)
    const void* engine{};        ///< the Maxwell3D / KeplerCompute it was resolved for
    const void* memory{};        ///< the address space
    u64 pipeline_epoch{};
    u64 rescale_epoch{};
    u64 map_serial{};
    u64 host_serial{};
    u64 image_changes{};
    u64 sampler_changes{};
    u64 native_key{};            ///< compute: PipelineCache's native program key
    std::vector<RecipeView> views;
    std::vector<Versions::WeakRef> samplers;
    std::vector<RecipeStorage> storage;
    std::array<Versions::WeakRef, NumRecipeTargets> targets{}; ///< id 0: none
    u64 render_pass{};           ///< VkRenderPass of the framebuffer (verify)
    std::array<u32, 32> push{};  ///< rescaling push words (verify)
    u32 push_words{};
    std::vector<u32> heap_words; ///< bindless heap words (verify)
    CapturedInputs inputs;
    BufferRecipe buffers;
    ResidencyStamp residency; ///< F2-09: the views were prepared under this stamp
    /// F2-04: the bindless heap's release count when a hit last found `heap_words` (~0: not
    /// yet), and the submission that last marked their slots used.
    u64 heap_releases{~0ULL};
    u64 heap_tick{};
    u64 uses{};
    /// DRAWREC: the record the recipe was built from (a hit copies it, step 16).
    RecordTemplate record;

    void Clear() {
        const u64 kept_key = key;
        record.Invalidate();
        residency = {};
        heap_releases = ~0ULL;
        heap_tick = 0;
        views.clear();
        samplers.clear();
        storage.clear();
        heap_words.clear();
        inputs.Clear();
        buffers.Clear();
        pipeline = nullptr;
        engine = nullptr;
        memory = nullptr;
        targets = {};
        push = {};
        push_words = 0;
        render_pass = 0;
        uses = 0;
        native_key = 0;
        key = kept_key;
    }
};

/// Whether every weak handle of the recipe is alive.
[[nodiscard]] bool HandlesAlive(const Recipe& recipe, const Versions::Generations& generations);

/// Field-by-field comparison of a hit's recipe with a fresh resolve (verify). Appends the names of
/// the fields that differ to `reasons` and returns whether all match.
[[nodiscard]] bool CompareRecipes(const Recipe& hit, const Recipe& fresh, std::string& reasons);

/// Set-associative recipe table with LRU replacement per set (owned by one resolving thread).
class RecipeTable {
public:
    static constexpr size_t Ways = 4;

    explicit RecipeTable(size_t sets_) : sets{sets_}, entries(sets_ * Ways) {}

    /// The recipe for `key`, or nullptr (touches it).
    [[nodiscard]] Recipe* Find(u64 key);
    /// A cleared recipe for `key` (an existing one is reused; else the LRU way of the set is
    /// evicted). `drop(recipe)` sees a live recipe before it is cleared; `evicted` is set when it
    /// was another key's.
    template <typename Drop>
    Recipe& Insert(u64 key, bool& evicted, Drop&& drop) {
        evicted = false;
        Entry* const set = &entries[SetOf(key) * Ways];
        Entry* victim = nullptr;
        for (size_t way = 0; way < Ways; ++way) {
            if (set[way].live && set[way].key == key) {
                victim = &set[way];
                break;
            }
        }
        if (victim == nullptr) {
            for (size_t way = 0; way < Ways; ++way) {
                if (!set[way].live) {
                    victim = &set[way];
                    break;
                }
                if (victim == nullptr || set[way].touched < victim->touched) {
                    victim = &set[way];
                }
            }
            evicted = victim->live;
            occupied += victim->live ? 0 : 1;
        }
        if (victim->live) {
            drop(victim->recipe);
        }
        victim->key = key;
        victim->live = true;
        victim->touched = ++clock;
        victim->recipe.Clear();
        victim->recipe.key = key;
        return victim->recipe;
    }
    Recipe& Insert(u64 key, bool& evicted) {
        return Insert(key, evicted, [](const Recipe&) {});
    }
    /// Drops the recipe for `key` (`drop` sees it first).
    template <typename Drop>
    void Erase(u64 key, Drop&& drop) {
        Entry* const set = &entries[SetOf(key) * Ways];
        for (size_t way = 0; way < Ways; ++way) {
            if (set[way].live && set[way].key == key) {
                drop(set[way].recipe);
                set[way].live = false;
                set[way].recipe.Clear();
                --occupied;
                return;
            }
        }
    }
    void Erase(u64 key) {
        Erase(key, [](const Recipe&) {});
    }
    /// Drops every recipe (`drop` sees each live one first).
    template <typename Drop>
    void Clear(Drop&& drop) {
        for (Entry& entry : entries) {
            if (entry.live) {
                drop(entry.recipe);
            }
            entry.live = false;
            entry.recipe.Clear();
        }
        occupied = 0;
    }
    void Clear() {
        Clear([](const Recipe&) {});
    }

    [[nodiscard]] size_t Occupied() const {
        return occupied;
    }
    [[nodiscard]] size_t Capacity() const {
        return entries.size();
    }
    /// Visits every live recipe (dropping held pages, stats).
    template <typename Func>
    void ForEach(Func&& func) {
        for (Entry& entry : entries) {
            if (entry.live) {
                func(entry.recipe);
            }
        }
    }

private:
    struct Entry {
        u64 key{};
        u64 touched{};
        bool live{};
        Recipe recipe;
    };
    [[nodiscard]] size_t SetOf(u64 key) const {
        return static_cast<size_t>((key * 0x9e3779b97f4a7c15ULL) >> 40) % sets;
    }

    size_t sets;
    std::vector<Entry> entries;
    u64 clock{};
    size_t occupied{};
};

/// The fast-path ladder (MASTER_PLAN §5.3, DIAGRAMS 2.4): continued draw, then draw-cache hit,
/// then dirty-group resolve. One decision, in this order.
enum class Path : u8 {
    Continued, ///< P3 delta draw: pipeline, descriptors and pass kept
    Replay,    ///< P4 recipe replay
    Resolve,   ///< P3/full resolve (builds a recipe when cacheable)
};

struct Decision {
    Path path{Path::Resolve};
    Miss miss{Miss::Count}; ///< why it is not a replay (Count: replay or continued)
    Recipe* recipe{};
    bool verify{};          ///< replay sampled for verify: resolve in full and compare
    bool invalidated{};     ///< an entry existed but failed validation
};

/// The ladder on its inputs: `continued` (P3 took the delta draw), whether the cache applies,
/// `lookup(miss)` -> Recipe* (or nullptr, setting `miss`), `validate(recipe)` -> Miss (Count when
/// it holds) and `sample()`, asked on hits only, whether this hit is a verify sample.
template <typename Lookup, typename Validate, typename Sample>
[[nodiscard]] Decision Decide(bool continued, bool cache_on, Lookup&& lookup, Validate&& validate,
                              Sample&& sample) {
    Decision decision;
    if (continued) {
        decision.path = Path::Continued;
        return decision;
    }
    if (!cache_on) {
        decision.miss = Miss::Ineligible;
        return decision;
    }
    Miss miss = Miss::NoEntry;
    Recipe* const recipe = lookup(miss);
    if (recipe == nullptr) {
        decision.miss = miss;
        return decision;
    }
    const Miss failed = validate(*recipe);
    if (failed != Miss::Count) {
        decision.miss = failed;
        decision.invalidated = true;
        return decision;
    }
    decision.path = Path::Replay;
    decision.recipe = recipe;
    decision.verify = sample();
    return decision;
}

/// 1-in-N verify sampling (EDEN_NX2_V2_DRAWCACHE_VERIFY_N, default 64; 1 = every hit).
class VerifySampler {
public:
    explicit VerifySampler(u32 n_) : n{n_ == 0 ? 1 : n_} {}
    [[nodiscard]] bool Next() {
        if (++count >= n) {
            count = 0;
            return true;
        }
        return false;
    }

private:
    u32 n;
    u32 count{};
};

/// EDEN_NX2_V2_DRAWCACHE_VERIFY_N, read once.
[[nodiscard]] u32 VerifyN();

/// Per-thread draw cache counters, reported every 10 s with EDEN_NX2_V2_STATS.
struct Counters {
    u64 lookups{};
    u64 continued{};
    u64 hits{};
    u64 replays_failed{}; ///< validated, but the replay found a reason to resolve
    u64 invalidated{};
    std::array<u64, NumMisses> misses{};
    u64 inserts{};
    u64 evictions{};
    u64 probation{}; ///< misses whose key was seen for the first time (no recipe built)
    u64 verify_samples{};
    u64 verify_mismatches{};
    u64 verify_deferred{}; ///< the memo deferred a CPU write the fresh walk saw (not a mismatch)
    u64 replay_ns{};
    u64 replay_draws{};
    u64 resolve_ns{};
    u64 resolve_draws{};
    u64 restamped{};
    u64 buffers_written{}; ///< hits whose storage buffer pages were written (synced on replay)
    u64 buffer_replays{};  ///< replays that bound every buffer from the recipe
    u64 buffer_partial{};  ///< replays where some bindings took the sync path
    u64 buffer_normal{};   ///< replays whose buffers all took the existing path
    u64 buffer_binds_replayed{};
    u64 buffer_binds_synced{};
    u64 buffer_stale{};    ///< verify: a range the versions vouched for was CPU-modified
    u64 watched_pages{};
};

struct ThreadData {
    Counters graphics;
    Counters compute;
    size_t graphics_occupied{};
    size_t graphics_capacity{};
    size_t compute_occupied{};
    size_t compute_capacity{};
    u64 window_start{};
};

/// The calling thread's stats (heap allocated on first use).
[[nodiscard]] ThreadData& ThreadStats();
/// Logs and resets the thread's stats every 10 s (only with EDEN_NX2_V2_STATS).
void Report(const char* thread, Versions::EpochMemo* memo);

/// The first 32 verify mismatches are logged.
void LogMismatch(bool compute, u64 key, const std::string& reasons);

} // namespace VideoCommon::V2::DrawCache
