// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/regsem.h"

// Pipeline v2 P6a: copy-on-write register snapshots (MASTER_PLAN §5.6.1).
//
// A register file is split into fixed blocks. A snapshot is an array of pointers to immutable
// blocks, so taking one costs a pointer copy per block plus a copy of the blocks that changed
// since the previous snapshot. Each block carries the fold of MixRegister(index, value) over its
// words, so a snapshot's fingerprint is the XOR of its blocks' and equals FingerprintOf() of the
// same registers computed from scratch. Snapshots are read-only and can be read by any number of
// threads. Blocks come from an epoch pool owned by the thread taking the snapshots: a block a
// later snapshot replaced is reused once every reader of the snapshots before is done (AP6 §4),
// so no reference counts, and no allocation or free on any other thread.
namespace VideoCommon::V2::Resolve {

/// Fixed-size objects owned by one producer thread, with no heap operation per object in steady
/// state. An object retired at `epoch` is reused after Reclaim(released) with released >= epoch;
/// epochs are retired in increasing order (the P6 front: segment ids, released by the commit).
template <typename T, size_t ChunkObjects = 256>
class EpochPool {
public:
    EpochPool() = default;
    EpochPool(const EpochPool&) = delete;
    EpochPool& operator=(const EpochPool&) = delete;

    [[nodiscard]] T* Allocate() {
        if (free_objects.empty()) [[unlikely]] {
            chunks.push_back(std::make_unique<T[]>(ChunkObjects));
            T* const chunk = chunks.back().get();
            for (size_t index = ChunkObjects; index-- > 0;) {
                free_objects.push_back(chunk + index);
            }
        }
        T* const object = free_objects.back();
        free_objects.pop_back();
        return object;
    }
    void Retire(T* object, u64 epoch) {
        retired.push_back(Entry{object, epoch});
    }
    void Reclaim(u64 released) {
        while (!retired.empty() && retired.front().epoch <= released) {
            free_objects.push_back(retired.front().object);
            retired.pop_front();
        }
    }
    [[nodiscard]] size_t RetiredCount() const {
        return retired.size();
    }

private:
    struct Entry {
        T* object;
        u64 epoch;
    };
    std::vector<std::unique_ptr<T[]>> chunks;
    std::vector<T*> free_objects;
    std::deque<Entry> retired;
};

template <size_t NumWords, size_t BlockWords = 64>
class RegisterSnapshot {
public:
    static constexpr size_t Words = NumWords;
    static constexpr size_t NumBlocks = (NumWords + BlockWords - 1) / BlockWords;

    struct Block {
        std::array<u32, BlockWords> words{};
        u64 hash = 0;
    };

    [[nodiscard]] bool Valid() const {
        return blocks[0] != nullptr;
    }
    [[nodiscard]] u32 Read(size_t index) const {
        return blocks[index / BlockWords]->words[index % BlockWords];
    }
    void CopyTo(u32* out) const {
        for (size_t block = 0; block < NumBlocks; ++block) {
            const size_t begin = block * BlockWords;
            const size_t count = (std::min)(BlockWords, NumWords - begin);
            std::memcpy(out + begin, blocks[block]->words.data(), count * sizeof(u32));
        }
    }
    [[nodiscard]] u64 Fingerprint() const {
        u64 result = 0;
        for (const auto& block : blocks) {
            result ^= block->hash;
        }
        return result;
    }
    /// Blocks this snapshot shares with `other` (the CoW saving).
    [[nodiscard]] size_t SharedWith(const RegisterSnapshot& other) const {
        size_t shared = 0;
        for (size_t block = 0; block < NumBlocks; ++block) {
            shared += blocks[block] == other.blocks[block] ? 1 : 0;
        }
        return shared;
    }

    [[nodiscard]] static u64 HashBlock(const u32* words, size_t first_index, size_t count) {
        u64 hash = 0;
        for (size_t i = 0; i < count; ++i) {
            hash ^= MixRegister(static_cast<u32>(first_index + i), words[i]);
        }
        return hash;
    }
    /// The fingerprint of a plain register array, as a snapshot of it would have.
    [[nodiscard]] static u64 FingerprintOf(const u32* words) {
        return HashBlock(words, 0, NumWords);
    }

    std::array<const Block*, NumBlocks> blocks{};
};

/// Takes snapshots of a plain register array (the front's mirror, which is written in place):
/// blocks equal to the previous snapshot's are shared, the others copied ("copy on snapshot").
/// The cost is one compare per block plus a copy per changed block, and no hook in the write path.
/// Snapshots stay valid while the builder lives and Reclaim has not passed their epoch.
template <size_t NumWords, size_t BlockWords = 64>
class SnapshotBuilder {
public:
    using Snapshot = RegisterSnapshot<NumWords, BlockWords>;
    using Block = typename Snapshot::Block;

    /// `dirty`: bit b set when block b may have changed since the previous Take (the writer's
    /// dirty-block mask, AP6 T0); clean blocks are shared without a compare. `epoch`: the
    /// snapshot's (increasing; the P6 segment id): a block it replaces was last read by `epoch-1`.
    Snapshot Take(const u32* words, u64 dirty = ~u64{0}, u64 epoch = 0) {
        static_assert(Snapshot::NumBlocks <= 64);
        Snapshot result;
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            const size_t begin = block * BlockWords;
            const size_t count = (std::min)(BlockWords, NumWords - begin);
            const Block* const previous = last.blocks[block];
            if (previous != nullptr && (dirty & (u64{1} << block)) == 0) {
                result.blocks[block] = previous;
                ++shared;
                continue;
            }
            if (previous != nullptr &&
                std::memcmp(previous->words.data(), words + begin, count * sizeof(u32)) == 0) {
                result.blocks[block] = previous;
                ++shared;
                continue;
            }
            Block* const fresh = pool.Allocate();
            std::memcpy(fresh->words.data(), words + begin, count * sizeof(u32));
            fresh->hash = Snapshot::HashBlock(words + begin, begin, count);
            result.blocks[block] = fresh;
            if (previous != nullptr) {
                pool.Retire(const_cast<Block*>(previous), epoch != 0 ? epoch - 1 : 0);
            }
            ++copied;
        }
        last = result;
        last_epoch = epoch;
        return result;
    }

    /// Every snapshot of epoch <= `released` is no longer read: their replaced blocks are reused.
    void Reclaim(u64 released) {
        pool.Reclaim(released);
    }

    /// Forgets the previous snapshot (the next Take copies every block).
    void Reset() {
        for (const Block* const block : last.blocks) {
            if (block != nullptr) {
                pool.Retire(const_cast<Block*>(block), last_epoch);
            }
        }
        last = {};
    }

    u64 copied = 0; ///< blocks copied (stats)
    u64 shared = 0; ///< blocks shared with the previous snapshot (stats)

private:
    EpochPool<Block> pool;
    Snapshot last;
    u64 last_epoch = 0;
};

/// A register file written through Write: a block a snapshot holds is copied before it is
/// written, so TakeSnapshot() is a pointer copy per block (true copy-on-write). Snapshots are
/// numbered by TakeSnapshot's epoch; Reclaim(released) reuses the blocks only snapshots up to
/// `released` held.
template <size_t NumWords, size_t BlockWords = 64>
class CowRegisterFile {
public:
    using Snapshot = RegisterSnapshot<NumWords, BlockWords>;
    using Block = typename Snapshot::Block;

    CowRegisterFile() {
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            blocks[block] = pool.Allocate();
            *blocks[block] = Block{};
        }
        RecomputeHashes();
    }

    [[nodiscard]] u32 Read(size_t index) const {
        return blocks[index / BlockWords]->words[index % BlockWords];
    }
    void Write(size_t index, u32 value) {
        const size_t block_index = index / BlockWords;
        Block* block = blocks[block_index];
        if (block->words[index % BlockWords] == value) {
            return;
        }
        if ((held & (u64{1} << block_index)) != 0) {
            // A snapshot (or the loaded one) holds this block: write a copy.
            Block* const copy = pool.Allocate();
            *copy = *block;
            if ((owned & (u64{1} << block_index)) != 0) {
                pool.Retire(block, epoch); // the snapshots up to now read it
            }
            blocks[block_index] = block = copy;
            held &= ~(u64{1} << block_index);
            owned |= u64{1} << block_index;
            ++copies;
        }
        u32& target = block->words[index % BlockWords];
        block->hash ^= MixRegister(static_cast<u32>(index), target) ^
                       MixRegister(static_cast<u32>(index), value);
        target = value;
    }
    [[nodiscard]] Snapshot TakeSnapshot() {
        Snapshot result;
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            result.blocks[block] = blocks[block];
        }
        held = ~u64{0};
        ++epoch;
        return result;
    }
    /// Starts from a snapshot (shares every block until written; the snapshot's blocks are not
    /// this file's to reuse).
    void Load(const Snapshot& snapshot) {
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            if ((owned & (u64{1} << block)) != 0) {
                pool.Retire(blocks[block], epoch);
            }
            blocks[block] = const_cast<Block*>(snapshot.blocks[block]);
        }
        held = ~u64{0};
        owned = 0;
    }
    void Reclaim(u64 released) {
        pool.Reclaim(released);
    }

    u64 copies = 0; ///< blocks copied on write (stats)

private:
    static_assert(Snapshot::NumBlocks <= 64);

    void RecomputeHashes() {
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            const size_t begin = block * BlockWords;
            const size_t count = (std::min)(BlockWords, NumWords - begin);
            blocks[block]->hash = Snapshot::HashBlock(blocks[block]->words.data(), begin, count);
        }
    }

    EpochPool<Block> pool;
    std::array<Block*, Snapshot::NumBlocks> blocks{};
    u64 held = 0;          ///< blocks a snapshot or Load shares (copied before a write)
    u64 owned = ~u64{0};   ///< blocks from this file's pool
    u64 epoch = 0;         ///< snapshots taken
};

} // namespace VideoCommon::V2::Resolve
