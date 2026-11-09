// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <memory>

#include "common/common_types.h"
#include "video_core/v2/regsem.h"

// Pipeline v2 P6a: copy-on-write register snapshots (MASTER_PLAN §5.6.1).
//
// A register file is split into fixed blocks. A snapshot is an array of shared pointers to
// immutable blocks, so taking one costs a pointer copy per block plus a copy of the blocks that
// changed since the previous snapshot. Each block carries the fold of MixRegister(index, value)
// over its words, so a snapshot's fingerprint is the XOR of its blocks' and equals
// FingerprintOf() of the same registers computed from scratch. Snapshots are read-only and can be
// shared by any number of threads (shared_ptr reference counts are atomic).
namespace VideoCommon::V2::Resolve {

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

    std::array<std::shared_ptr<const Block>, NumBlocks> blocks{};
};

/// Takes snapshots of a plain register array (the front's mirror, which is written in place):
/// blocks equal to the previous snapshot's are shared, the others copied ("copy on snapshot").
/// The cost is one compare per block plus a copy per changed block, and no hook in the write path.
template <size_t NumWords, size_t BlockWords = 64>
class SnapshotBuilder {
public:
    using Snapshot = RegisterSnapshot<NumWords, BlockWords>;
    using Block = typename Snapshot::Block;

    Snapshot Take(const u32* words) {
        Snapshot result;
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            const size_t begin = block * BlockWords;
            const size_t count = (std::min)(BlockWords, NumWords - begin);
            const auto& previous = last.blocks[block];
            if (previous != nullptr &&
                std::memcmp(previous->words.data(), words + begin, count * sizeof(u32)) == 0) {
                result.blocks[block] = previous;
                ++shared;
                continue;
            }
            auto fresh = std::make_shared<Block>();
            std::memcpy(fresh->words.data(), words + begin, count * sizeof(u32));
            fresh->hash = Snapshot::HashBlock(words + begin, begin, count);
            result.blocks[block] = std::move(fresh);
            ++copied;
        }
        last = result;
        return result;
    }

    /// Forgets the previous snapshot (the next Take copies every block).
    void Reset() {
        last = {};
    }

    u64 copied = 0; ///< blocks copied (stats)
    u64 shared = 0; ///< blocks shared with the previous snapshot (stats)

private:
    Snapshot last;
};

/// A register file written through Write: a block shared with a live snapshot is copied before
/// it is written, so Snapshot() is a pointer copy per block (true copy-on-write).
template <size_t NumWords, size_t BlockWords = 64>
class CowRegisterFile {
public:
    using Snapshot = RegisterSnapshot<NumWords, BlockWords>;
    using Block = typename Snapshot::Block;

    CowRegisterFile() {
        for (auto& block : blocks) {
            block = std::make_shared<Block>();
        }
        RecomputeHashes();
    }

    [[nodiscard]] u32 Read(size_t index) const {
        return blocks[index / BlockWords]->words[index % BlockWords];
    }
    void Write(size_t index, u32 value) {
        auto& block = blocks[index / BlockWords];
        u32& word = block->words[index % BlockWords];
        if (word == value) {
            return;
        }
        if (block.use_count() > 1) {
            // A snapshot holds this block: copy it first (only this thread creates new holders).
            block = std::make_shared<Block>(*block);
            ++copies;
        }
        u32& target = block->words[index % BlockWords];
        block->hash ^= MixRegister(static_cast<u32>(index), target) ^
                       MixRegister(static_cast<u32>(index), value);
        target = value;
    }
    [[nodiscard]] Snapshot TakeSnapshot() const {
        Snapshot result;
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            result.blocks[block] = blocks[block];
        }
        return result;
    }
    /// Starts from a snapshot (shares every block until written).
    void Load(const Snapshot& snapshot) {
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            blocks[block] = std::const_pointer_cast<Block>(snapshot.blocks[block]);
        }
    }

    u64 copies = 0; ///< blocks copied on write (stats)

private:
    void RecomputeHashes() {
        for (size_t block = 0; block < Snapshot::NumBlocks; ++block) {
            const size_t begin = block * BlockWords;
            const size_t count = (std::min)(BlockWords, NumWords - begin);
            blocks[block]->hash = Snapshot::HashBlock(blocks[block]->words.data(), begin, count);
        }
    }

    std::array<std::shared_ptr<Block>, Snapshot::NumBlocks> blocks;
};

} // namespace VideoCommon::V2::Resolve
