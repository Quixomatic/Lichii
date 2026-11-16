// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstring>
#include <span>
#include <vector>

#include "common/common_types.h"

// Pipeline v2 P3 (EDEN_NX2_V2_DESCCACHE): content-keyed descriptor reuse, the stepping stone
// before the bindless heap (notes/06 §4.2, notes/05 §3.3). The key is the descriptor payload a
// draw would write: image view and sampler handles, buffer handles with offsets and ranges. A
// draw whose payload equals a cached one reuses what was made from it (a pushed set that is
// still bound, a written descriptor set or descriptor buffer range that is still alive) instead
// of writing descriptors again.
namespace VideoCommon::V2 {

/// FNV-1a over the payload: cheap for the few hundred bytes of a draw's descriptors.
[[nodiscard]] inline u64 HashDescriptorBytes(std::span<const u8> bytes) {
    u64 hash = 0xcbf29ce484222325ULL;
    size_t i = 0;
    for (; i + 8 <= bytes.size(); i += 8) {
        u64 word;
        std::memcpy(&word, bytes.data() + i, sizeof(word));
        hash = (hash ^ word) * 0x100000001b3ULL;
    }
    for (; i < bytes.size(); ++i) {
        hash = (hash ^ bytes[i]) * 0x100000001b3ULL;
    }
    return hash ^ (hash >> 29);
}

/// A small LRU of payloads and what was made from them, each valid for one epoch (a command
/// buffer tick, a descriptor ring generation, a recording epoch).
template <typename Value, size_t Ways = 8>
class DescriptorContentCache {
public:
    /// The value made from `bytes` in `epoch`, or nullptr. A hit becomes the most recent entry.
    /// With `trust_hash` the bytes are not compared (verify mode compares them separately).
    [[nodiscard]] Value* Find(std::span<const u8> bytes, u64 hash, u64 epoch,
                              bool trust_hash = false) {
        for (Entry& entry : entries) {
            if (!entry.valid || entry.hash != hash || entry.epoch != epoch ||
                entry.bytes.size() != bytes.size()) {
                continue;
            }
            if (!trust_hash && std::memcmp(entry.bytes.data(), bytes.data(), bytes.size()) != 0) {
                continue;
            }
            entry.stamp = ++clock;
            return &entry.value;
        }
        return nullptr;
    }

    /// Whether an entry with `hash` in `epoch` holds other bytes (a hash collision).
    [[nodiscard]] bool Collides(std::span<const u8> bytes, u64 hash, u64 epoch) const {
        for (const Entry& entry : entries) {
            if (entry.valid && entry.hash == hash && entry.epoch == epoch &&
                (entry.bytes.size() != bytes.size() ||
                 std::memcmp(entry.bytes.data(), bytes.data(), bytes.size()) != 0)) {
                return true;
            }
        }
        return false;
    }

    /// Stores `bytes` for `epoch`, replacing an entry of another epoch or the least recent one.
    Value& Insert(std::span<const u8> bytes, u64 hash, u64 epoch) {
        Entry* victim = &entries[0];
        for (Entry& entry : entries) {
            if (!entry.valid || entry.epoch != epoch) {
                victim = &entry;
                break;
            }
            if (entry.stamp < victim->stamp) {
                victim = &entry;
            }
        }
        victim->valid = true;
        victim->hash = hash;
        victim->epoch = epoch;
        victim->stamp = ++clock;
        victim->bytes.assign(bytes.begin(), bytes.end());
        victim->value = Value{};
        return victim->value;
    }

    void Clear() {
        for (Entry& entry : entries) {
            entry.valid = false;
        }
    }

private:
    struct Entry {
        bool valid{};
        u64 hash{};
        u64 epoch{};
        u64 stamp{};
        std::vector<u8> bytes;
        Value value{};
    };
    std::array<Entry, Ways> entries{};
    u64 clock{};
};

} // namespace VideoCommon::V2
