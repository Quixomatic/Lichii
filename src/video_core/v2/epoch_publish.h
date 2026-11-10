// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "common/common_types.h"

// Pipeline v2 P5a: epoch-published lookup structures for the cache split.
//
// EpochDomain is epoch-based reclamation (RCU-like). Readers enter a critical section, read
// published pointers with acquire loads and leave; the single writer replaces a structure by
// publishing a new pointer and retiring the old one, which is freed only once every reader that
// could still hold it has left. Readers never block and never write shared lines other than
// their own slot.
//
// PublishedMap is an open-addressed hash map from a POD key to a u32 value with one writer and
// lock-free readers: a slot's key is written before its hash is published (release), and never
// rewritten while the table is live, so a reader that acquires the hash reads a complete key.
// Erase writes a tombstone value. Growing (or compacting tombstones) builds a new table,
// publishes it and retires the old one through the domain.
namespace VideoCommon::V2 {

class EpochDomain {
public:
    static constexpr size_t MaxReaders = 64;

    struct Reader {
        u32 slot = ~0U;
        [[nodiscard]] bool Valid() const {
            return slot != ~0U;
        }
    };

    EpochDomain() = default;
    EpochDomain(const EpochDomain&) = delete;
    EpochDomain& operator=(const EpochDomain&) = delete;

    ~EpochDomain() {
        for (const Retired& retired : retired_list) {
            retired.deleter(retired.pointer);
        }
    }

    /// Gives a thread a reader slot (once per thread and domain). Invalid when all are taken.
    [[nodiscard]] Reader Register() {
        for (u32 slot = 0; slot < MaxReaders; ++slot) {
            bool expected = false;
            if (readers[slot].used.compare_exchange_strong(expected, true,
                                                           std::memory_order_acq_rel)) {
                return Reader{slot};
            }
        }
        return Reader{};
    }

    void Unregister(Reader reader) {
        if (reader.Valid()) {
            readers[reader.slot].epoch.store(0, std::memory_order_release);
            readers[reader.slot].used.store(false, std::memory_order_release);
        }
    }

    /// Starts a read-side section: published pointers loaded until Exit stay valid.
    void Enter(Reader reader) {
        // The acquire load pairs with Retire's increment: a reader that sees an epoch newer than
        // a retirement also sees the pointer published before it. The seq_cst store and fence
        // pair with Reclaim's scan: either the scan sees this reader, or this reader's later
        // loads see every pointer published before the scan.
        readers[reader.slot].epoch.store(global.load(std::memory_order_acquire),
                                         std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void Exit(Reader reader) {
        readers[reader.slot].epoch.store(0, std::memory_order_release);
    }

    /// Writer: `pointer` is unpublished; free it with `deleter` once no reader can hold it.
    void Retire(void* pointer, void (*deleter)(void*)) {
        const u64 epoch = global.fetch_add(1, std::memory_order_seq_cst);
        retired_list.push_back(Retired{pointer, deleter, epoch});
        Reclaim();
    }

    /// Writer: frees what no reader can still hold. Returns how many objects were freed.
    size_t Reclaim() {
        std::atomic_thread_fence(std::memory_order_seq_cst);
        u64 oldest = ~u64{0};
        for (const Slot& reader : readers) {
            const u64 epoch = reader.epoch.load(std::memory_order_seq_cst);
            if (epoch != 0 && epoch < oldest) {
                oldest = epoch;
            }
        }
        size_t freed = 0;
        std::erase_if(retired_list, [&](const Retired& retired) {
            // A reader in epoch E entered before every retirement stamped >= E happened, so it
            // may hold those; objects retired before it entered (< E) are unreachable to it.
            if (retired.epoch < oldest) {
                retired.deleter(retired.pointer);
                ++freed;
                return true;
            }
            return false;
        });
        return freed;
    }

    [[nodiscard]] size_t PendingRetired() const {
        return retired_list.size();
    }

private:
    struct alignas(64) Slot {
        std::atomic<u64> epoch{0};
        std::atomic<bool> used{false};
    };
    struct Retired {
        void* pointer;
        void (*deleter)(void*);
        u64 epoch;
    };

    std::atomic<u64> global{1};
    std::array<Slot, MaxReaders> readers{};
    std::vector<Retired> retired_list; ///< Writer only
};

/// RAII read-side section.
class EpochGuard {
public:
    EpochGuard(EpochDomain& domain_, EpochDomain::Reader reader_)
        : domain{domain_}, reader{reader_} {
        domain.Enter(reader);
    }
    ~EpochGuard() {
        domain.Exit(reader);
    }
    EpochGuard(const EpochGuard&) = delete;
    EpochGuard& operator=(const EpochGuard&) = delete;

private:
    EpochDomain& domain;
    EpochDomain::Reader reader;
};

/// A read-side section on the calling thread's own reader slot of `domain`, registered on the
/// thread's first section and given back when the thread exits. Nested sections of one domain
/// share the outermost (Exit clears the slot). Invalid (protects nothing) when every slot is
/// taken: the caller must not read published pointers then.
class ThreadEpochGuard {
public:
    explicit ThreadEpochGuard(EpochDomain& domain) {
        Entry* const entry = Find(domain);
        if (entry == nullptr) {
            return;
        }
        if (entry->depth++ == 0) {
            domain.Enter(entry->reader);
        }
        current = entry;
    }
    ~ThreadEpochGuard() {
        if (current != nullptr && --current->depth == 0) {
            current->domain->Exit(current->reader);
        }
    }
    ThreadEpochGuard(const ThreadEpochGuard&) = delete;
    ThreadEpochGuard& operator=(const ThreadEpochGuard&) = delete;

    [[nodiscard]] bool Valid() const {
        return current != nullptr;
    }

private:
    struct Entry {
        EpochDomain* domain = nullptr;
        EpochDomain::Reader reader;
        u32 depth = 0;
    };
    struct Registrations {
        std::array<Entry, 4> entries{};
        ~Registrations() {
            for (Entry& entry : entries) {
                if (entry.domain != nullptr) {
                    entry.domain->Unregister(entry.reader);
                }
            }
        }
    };

    static Entry* Find(EpochDomain& domain) {
        thread_local Registrations registrations;
        for (Entry& entry : registrations.entries) {
            if (entry.domain == &domain) {
                return &entry;
            }
            if (entry.domain == nullptr) {
                const EpochDomain::Reader reader = domain.Register();
                if (!reader.Valid()) {
                    return nullptr;
                }
                entry.domain = &domain;
                entry.reader = reader;
                return &entry;
            }
        }
        return nullptr;
    }

    Entry* current = nullptr;
};

template <typename Key>
class PublishedMap {
    static_assert(std::is_trivially_copyable_v<Key>);

public:
    static constexpr u32 Missing = ~0U;

    explicit PublishedMap(EpochDomain& domain_, size_t initial_capacity = 64) : domain{domain_} {
        size_t capacity = 16;
        while (capacity < initial_capacity) {
            capacity <<= 1;
        }
        table.store(new Table(capacity), std::memory_order_release);
    }

    ~PublishedMap() {
        delete table.load(std::memory_order_relaxed);
    }

    PublishedMap(const PublishedMap&) = delete;
    PublishedMap& operator=(const PublishedMap&) = delete;

    /// Any thread, inside an EpochGuard of the map's domain (or on the writer thread).
    [[nodiscard]] u32 Find(const Key& key, u64 hash) const {
        hash = Normalize(hash);
        const Table* const current = table.load(std::memory_order_acquire);
        const size_t mask = current->capacity - 1;
        for (size_t probe = 0, index = hash & mask; probe < current->capacity;
             ++probe, index = (index + 1) & mask) {
            const Entry& entry = current->entries[index];
            const u64 stored = entry.hash.load(std::memory_order_acquire);
            if (stored == 0) {
                return Missing;
            }
            if (stored == hash && std::memcmp(&entry.key, &key, sizeof(Key)) == 0) {
                return entry.value.load(std::memory_order_acquire);
            }
        }
        return Missing;
    }

    /// Writer only: inserts or overwrites.
    void Insert(const Key& key, u64 hash, u32 value) {
        hash = Normalize(hash);
        Table* current = table.load(std::memory_order_relaxed);
        if ((current->used + 1) * 4 > current->capacity * 3) {
            current = Rebuild(current->live * 2 + 2 > current->capacity / 2 ? current->capacity * 2
                                                                             : current->capacity);
        }
        const size_t mask = current->capacity - 1;
        for (size_t index = hash & mask;; index = (index + 1) & mask) {
            Entry& entry = current->entries[index];
            const u64 stored = entry.hash.load(std::memory_order_relaxed);
            if (stored == 0) {
                std::memcpy(&entry.key, &key, sizeof(Key));
                entry.value.store(value, std::memory_order_relaxed);
                entry.hash.store(hash, std::memory_order_release);
                ++current->used;
                ++current->live;
                return;
            }
            if (stored == hash && std::memcmp(&entry.key, &key, sizeof(Key)) == 0) {
                if (entry.value.load(std::memory_order_relaxed) == Missing) {
                    ++current->live;
                }
                entry.value.store(value, std::memory_order_release);
                return;
            }
        }
    }

    /// Writer only. Returns whether the key was present.
    bool Erase(const Key& key, u64 hash) {
        hash = Normalize(hash);
        Table* const current = table.load(std::memory_order_relaxed);
        const size_t mask = current->capacity - 1;
        for (size_t probe = 0, index = hash & mask; probe < current->capacity;
             ++probe, index = (index + 1) & mask) {
            Entry& entry = current->entries[index];
            const u64 stored = entry.hash.load(std::memory_order_relaxed);
            if (stored == 0) {
                return false;
            }
            if (stored == hash && std::memcmp(&entry.key, &key, sizeof(Key)) == 0) {
                if (entry.value.load(std::memory_order_relaxed) == Missing) {
                    return false;
                }
                entry.value.store(Missing, std::memory_order_release);
                --current->live;
                return true;
            }
        }
        return false;
    }

    /// Writer only: erases every entry whose value satisfies `pred`.
    template <typename Pred>
    size_t EraseIf(Pred&& pred) {
        Table* const current = table.load(std::memory_order_relaxed);
        size_t erased = 0;
        for (size_t index = 0; index < current->capacity; ++index) {
            Entry& entry = current->entries[index];
            if (entry.hash.load(std::memory_order_relaxed) == 0) {
                continue;
            }
            const u32 value = entry.value.load(std::memory_order_relaxed);
            if (value != Missing && pred(value)) {
                entry.value.store(Missing, std::memory_order_release);
                --current->live;
                ++erased;
            }
        }
        return erased;
    }

    /// Writer only: drops everything (publishes an empty table).
    void Clear() {
        Table* const old = table.load(std::memory_order_relaxed);
        table.store(new Table(old->capacity), std::memory_order_release);
        domain.Retire(old, [](void* pointer) { delete static_cast<Table*>(pointer); });
    }

    [[nodiscard]] size_t Size() const {
        return table.load(std::memory_order_acquire)->live;
    }

private:
    struct Entry {
        std::atomic<u64> hash{0}; ///< 0 = empty; published last
        std::atomic<u32> value{Missing};
        Key key{};
    };
    struct Table {
        explicit Table(size_t capacity_)
            : capacity{capacity_}, entries{std::make_unique<Entry[]>(capacity_)} {}
        size_t capacity;
        size_t used = 0; ///< Slots with a hash (live + tombstones)
        size_t live = 0;
        std::unique_ptr<Entry[]> entries;
    };

    static u64 Normalize(u64 hash) {
        return hash == 0 ? 1 : hash;
    }

    Table* Rebuild(size_t capacity) {
        Table* const old = table.load(std::memory_order_relaxed);
        auto* const fresh = new Table(capacity);
        const size_t mask = capacity - 1;
        for (size_t index = 0; index < old->capacity; ++index) {
            const Entry& entry = old->entries[index];
            const u64 hash = entry.hash.load(std::memory_order_relaxed);
            const u32 value = entry.value.load(std::memory_order_relaxed);
            if (hash == 0 || value == Missing) {
                continue;
            }
            for (size_t slot = hash & mask;; slot = (slot + 1) & mask) {
                Entry& target = fresh->entries[slot];
                if (target.hash.load(std::memory_order_relaxed) == 0) {
                    std::memcpy(&target.key, &entry.key, sizeof(Key));
                    target.value.store(value, std::memory_order_relaxed);
                    target.hash.store(hash, std::memory_order_relaxed);
                    ++fresh->used;
                    ++fresh->live;
                    break;
                }
            }
        }
        table.store(fresh, std::memory_order_release);
        domain.Retire(old, [](void* pointer) { delete static_cast<Table*>(pointer); });
        return fresh;
    }

    EpochDomain& domain;
    std::atomic<Table*> table;
};

} // namespace VideoCommon::V2
