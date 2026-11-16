// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/epoch_publish.h"
#include "video_core/v2/regsem.h"

// Pipeline v2 P3 (EDEN_NX2_V2_PIPEKEY): DXVK's two-level pipeline lookup (notes/04 §2.3, §10.1).
// Level 1 maps the shader set (the six program hashes) to its ShaderSet through a direct-mapped
// cache in front of a map; level 2 is the shader set's variant table, keyed by the packed
// fixed-function state. Readers take no lock: published entries are immutable, and the tables
// they live in are only freed by Clear (callers clear when pipelines are destroyed). Inserts
// take a mutex, as only misses insert.
namespace VideoCommon::V2 {

/// The variants of one shader set: packed state (hash + bytes) -> value. Below LinearLimit
/// variants a lookup compares linearly without using the hash (DXVK's HashThreshold), above it
/// probes an open-addressed table.
template <typename Value>
class VariantTable {
public:
    static constexpr u32 LinearLimit = 4;

    VariantTable() = default;
    VariantTable(const VariantTable&) = delete;
    VariantTable& operator=(const VariantTable&) = delete;

    /// The value whose key is `size` bytes at `key` with hash `hash`, or nullptr.
    [[nodiscard]] Value* Find(u64 hash, const void* key, size_t size) const {
        const u32 linear = linear_count.load(std::memory_order_acquire);
        for (u32 i = 0; i < linear; ++i) {
            const Variant& variant = *linear_entries[i];
            if (variant.hash == hash && variant.size == size &&
                std::memcmp(variant.key, key, size) == 0) {
                return variant.value;
            }
        }
        const Table* const table = hashed.load(std::memory_order_acquire);
        if (table == nullptr) {
            return nullptr;
        }
        const size_t mask = table->slots.size() - 1;
        for (size_t probe = 0; probe <= mask; ++probe) {
            const Variant* const variant =
                table->slots[(hash + probe) & mask].load(std::memory_order_acquire);
            if (variant == nullptr) {
                return nullptr;
            }
            if (variant->hash == hash && variant->size == size &&
                std::memcmp(variant->key, key, size) == 0) {
                return variant->value;
            }
        }
        return nullptr;
    }

    /// Adds a variant with a copy of its `size` key bytes (R3: a shard may still compare keys
    /// of a retired table after the value that owned them is gone). The caller holds the
    /// owner's insert lock.
    void Insert(u64 hash, const void* key, size_t size, Value* value) {
        auto variant = std::make_unique<Variant>();
        variant->owned = std::make_unique<u8[]>(size);
        std::memcpy(variant->owned.get(), key, size);
        variant->hash = hash;
        variant->key = variant->owned.get();
        variant->size = size;
        variant->value = value;
        const Variant* const published = variant.get();
        storage.push_back(std::move(variant));
        const u32 linear = linear_count.load(std::memory_order_relaxed);
        if (linear < LinearLimit) {
            linear_entries[linear] = published;
            linear_count.store(linear + 1, std::memory_order_release);
            return;
        }
        Table* table = hashed.load(std::memory_order_relaxed);
        if (table == nullptr || (hashed_count + 1) * 2 > table->slots.size()) {
            table = Grow(table);
        }
        Place(*table, published);
        ++hashed_count;
    }

    [[nodiscard]] size_t Size() const {
        return storage.size();
    }

private:
    struct Variant {
        u64 hash{};
        const u8* key{};
        size_t size{};
        Value* value{};
        std::unique_ptr<u8[]> owned;
    };
    struct Table {
        explicit Table(size_t size) : slots(size) {}
        std::vector<std::atomic<const Variant*>> slots;
    };

    static void Place(Table& table, const Variant* variant) {
        const size_t mask = table.slots.size() - 1;
        for (size_t probe = 0;; ++probe) {
            auto& slot = table.slots[(variant->hash + probe) & mask];
            if (slot.load(std::memory_order_relaxed) == nullptr) {
                slot.store(variant, std::memory_order_release);
                return;
            }
        }
    }

    Table* Grow(const Table* old) {
        const size_t size = old == nullptr ? 16 : old->slots.size() * 2;
        auto table = std::make_unique<Table>(size);
        if (old != nullptr) {
            for (const auto& slot : old->slots) {
                if (const Variant* variant = slot.load(std::memory_order_relaxed)) {
                    Place(*table, variant);
                }
            }
        }
        Table* const result = table.get();
        // Readers may still probe the old table: it stays alive with the shader set.
        tables.push_back(std::move(table));
        hashed.store(result, std::memory_order_release);
        return result;
    }

    std::array<const Variant*, LinearLimit> linear_entries{};
    std::atomic<u32> linear_count{};
    std::atomic<Table*> hashed{};
    size_t hashed_count{};
    std::vector<std::unique_ptr<Variant>> storage;
    std::vector<std::unique_ptr<Table>> tables;
};

/// Shader set -> VariantTable, through a direct-mapped cache of 4096 entries.
template <typename Value>
class PipelineLookup {
public:
    using Hashes = std::array<u64, 6>;
    static constexpr size_t CacheEntries = 4096;

    struct ShaderSet {
        Hashes hashes{};
        VariantTable<Value> variants;
    };

    struct Counters {
        u64 l1_hits{};
        u64 l1_misses{};
        u64 variant_hits{};
        u64 variant_misses{};
    };

    PipelineLookup() : cache(std::make_unique<std::array<std::atomic<ShaderSet*>, CacheEntries>>()) {}

    /// The value for (shader set, state), or nullptr.
    [[nodiscard]] Value* Find(const Hashes& hashes, u64 state_hash, const void* key,
                              size_t size) {
        ShaderSet* const set = FindSet(hashes);
        if (set == nullptr) {
            ++counters.variant_misses;
            return nullptr;
        }
        Value* const value = set->variants.Find(state_hash, key, size);
        ++(value != nullptr ? counters.variant_hits : counters.variant_misses);
        return value;
    }

    /// Adds (shader set, state) -> value (the table keeps a copy of the key).
    void Insert(const Hashes& hashes, u64 state_hash, const void* key, size_t size, Value* value) {
        std::scoped_lock lock{mutex};
        auto [it, is_new] = sets.try_emplace(hashes);
        ShaderSet* set = it->second.get();
        const bool new_variant = is_new || set->variants.Find(state_hash, key, size) == nullptr;
        if (new_variant) {
            BeginChange();
        }
        if (is_new) {
            it->second = std::make_unique<ShaderSet>();
            it->second->hashes = hashes;
            set = it->second.get();
            if (published) {
                published->Insert(hashes, Mix(hashes), reinterpret_cast<uintptr_t>(set));
            }
        }
        if (new_variant) {
            set->variants.Insert(state_hash, key, size, value);
        }
        (*cache)[Slot(hashes)].store(set, std::memory_order_release);
        if (new_variant) {
            EndChange();
        }
    }

    /// Forgets every entry (pipelines were destroyed). With a domain (SetDomain) the shader sets
    /// are retired through it, so FindShared readers inside an epoch section keep them; the
    /// values they found are stale from the generation this moves to.
    void Clear() {
        std::scoped_lock lock{mutex};
        BeginChange();
        for (auto& slot : *cache) {
            slot.store(nullptr, std::memory_order_relaxed);
        }
        if (published) {
            published->Clear();
        }
        // After the tables are empty: a reader that sees the new generation sees no old set.
        generation.fetch_add(1, std::memory_order_release);
        EndChange();
        if (domain != nullptr && !sets.empty()) {
            using Retired = std::vector<std::unique_ptr<ShaderSet>>;
            auto* const retired = new Retired;
            retired->reserve(sets.size());
            for (auto& entry : sets) {
                retired->push_back(std::move(entry.second));
            }
            domain->Retire(retired, [](void* pointer) { delete static_cast<Retired*>(pointer); });
        }
        sets.clear();
    }

    /// EDEN_NX2_V2_PARALLEL R3: lookups off the owner's thread. Readers (FindShared) run inside
    /// an epoch section of `domain_`; set before any reader starts.
    void SetDomain(EpochDomain* domain_) {
        std::scoped_lock lock{mutex};
        domain = domain_;
        published.reset();
        if (domain != nullptr) {
            // F6-X1: every set inserted so far, then each new one as Insert makes it.
            published = std::make_unique<PublishedMap<Hashes, u64>>(*domain, 64);
            for (const auto& [set_hashes, set] : sets) {
                published->Insert(set_hashes, Mix(set_hashes), reinterpret_cast<uintptr_t>(set.get()));
            }
        }
    }
    /// Moves at every Clear: a value FindShared returned is still valid (not destroyed) while
    /// this has the value read before the lookup.
    [[nodiscard]] u64 Generation() const {
        return generation.load(std::memory_order_acquire);
    }
    /// A sequence lock over what lookups can answer (F6-X1): odd while an Insert or Clear
    /// changes the tables, even otherwise. Two FindShared calls of one key that both read the
    /// same even serial before (Serial) and after (SerialHolds) answer the same; verify tells
    /// dependent differences from real ones by it.
    [[nodiscard]] u64 Serial() const {
        return serial.load(std::memory_order_acquire);
    }
    /// After a lookup that read `before` from Serial(): no change overlapped it.
    [[nodiscard]] bool SerialHolds(u64 before) const {
        std::atomic_thread_fence(std::memory_order_acquire);
        return (before & 1) == 0 && serial.load(std::memory_order_relaxed) == before;
    }
    /// Any thread, inside an epoch section of the domain (SetDomain first): the published,
    /// insert-only shader set table (MASTER_PLAN §5.3: pure lookups). The answer depends only on
    /// the Inserts and Clears before it, never on which set the owner's direct-mapped cache holds
    /// (F6-X1: a cache slot two sets share made two lookups of one key with equal serials differ).
    [[nodiscard]] Value* FindShared(const Hashes& hashes, u64 state_hash, const void* key,
                                    size_t size) const {
        const PublishedMap<Hashes, u64>* const map = published.get();
        if (map == nullptr) {
            return nullptr;
        }
        const u64 found = map->Find(hashes, Mix(hashes));
        if (found == PublishedMap<Hashes, u64>::Missing) {
            return nullptr;
        }
        const ShaderSet* const set = reinterpret_cast<const ShaderSet*>(static_cast<uintptr_t>(found));
        return set->variants.Find(state_hash, key, size);
    }

    [[nodiscard]] size_t NumSets() const {
        return sets.size();
    }

    [[nodiscard]] Counters TakeCounters() {
        return std::exchange(counters, Counters{});
    }

private:
    struct HashesHash {
        size_t operator()(const Hashes& hashes) const noexcept {
            return static_cast<size_t>(Mix(hashes));
        }
    };

    static u64 Mix(const Hashes& hashes) {
        u64 result = 0x9e3779b97f4a7c15ULL;
        for (const u64 hash : hashes) {
            result = Mix64(result ^ hash);
        }
        return result;
    }

    static size_t Slot(const Hashes& hashes) {
        return static_cast<size_t>(Mix(hashes)) & (CacheEntries - 1);
    }

    /// Writers hold `mutex`.
    void BeginChange() {
        serial.store(serial.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_release);
    }
    void EndChange() {
        serial.store(serial.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    /// The owner's lookup: the direct-mapped cache is its accelerator only (FindShared does not
    /// read it, so restoring an evicted set here changes no shard's answer).
    ShaderSet* FindSet(const Hashes& hashes) {
        ShaderSet* const cached = (*cache)[Slot(hashes)].load(std::memory_order_acquire);
        if (cached != nullptr && cached->hashes == hashes) {
            ++counters.l1_hits;
            return cached;
        }
        ++counters.l1_misses;
        std::scoped_lock lock{mutex};
        const auto it = sets.find(hashes);
        if (it == sets.end()) {
            return nullptr;
        }
        (*cache)[Slot(hashes)].store(it->second.get(), std::memory_order_release);
        return it->second.get();
    }

    std::unique_ptr<std::array<std::atomic<ShaderSet*>, CacheEntries>> cache;
    std::unordered_map<Hashes, std::unique_ptr<ShaderSet>, HashesHash> sets;
    std::mutex mutex;
    Counters counters; ///< Owner thread only (Find); shards count nothing here
    EpochDomain* domain = nullptr;
    /// F6-X1: shader sets for FindShared (null until SetDomain).
    std::unique_ptr<PublishedMap<Hashes, u64>> published;
    std::atomic<u64> generation{1};
    std::atomic<u64> serial{2};
};

} // namespace VideoCommon::V2
