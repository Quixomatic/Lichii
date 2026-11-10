// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/v2_switches.h"

// Pipeline v2 F2-03 (EDEN_NX2_V2_CBUFRING, EDEN_NX2_CBUFWATCH=0 turns it off): constant buffer
// versions trusted without a byte compare per bind (notes/F2-03_DESIGN.md).
//
// One-shot write watch. The back arms a device page by adding 1 to its cached count (the page
// is write-protected; a guest CPU store faults and takes the slow path through
// Core::Memory::HandleRasterizerWrite). The first trapped write disarms it and gives the count
// back; the back re-arms it later, after a backoff that doubles for pages trapped soon after
// their arm (up to 64 frames) and halves otherwise.
//
// Every device page has a word: (sequence << 3) | owed | queued | armed. A version snapshots
// the words of its 1-2 pages before its bytes are read; it is trusted while every page was armed
// at the snapshot and the words are still the same: no CPU store trapped (the trap moves the
// sequence before the store lands) and no foreign write was noted since.
//  - Trap (CPU thread): the thread whose CAS clears `armed` gives the count back, inline.
//  - Foreign write (host, DMA, download, unmap; any thread): an armed page is disarmed with the
//    count owed (the back gives it back), an unarmed one only moves its sequence. Disarming
//    closes the window between the note and the bytes landing: a snapshot in it is unarmed.
//  - Queued: the front asked for an arm (one request per page until the back handles it).
// Words live in chunks allocated on first request (front) or use (back), never freed, so any
// thread can read them without a lock. An unallocated page reads 0 (unarmed).
namespace VideoCommon::V2::CbufWatch {

inline constexpr u64 PageBits = 12;
inline constexpr u64 PageSize = u64{1} << PageBits;
inline constexpr u32 ArmedBit = 1;
inline constexpr u32 QueuedBit = 2;
inline constexpr u32 OwedBit = 4;
inline constexpr u32 SeqOne = 8;

/// What a version snapshots of a page word: sequence and armed (queued/owed are bookkeeping).
[[nodiscard]] constexpr u32 SnapshotOf(u32 word) noexcept {
    return word & ~(QueuedBit | OwedBit);
}

/// The words of a window's pages (second page 0 for one-page windows) and whether all were armed.
struct Snap {
    u64 words{};
    bool armed{};
};

class PageWords {
public:
    static constexpr u64 ChunkBits = 12; ///< 4096 pages (16 MiB) per chunk
    static constexpr u64 ChunkSize = u64{1} << ChunkBits;
    static constexpr u64 MaxChunks = u64{1} << (36 - PageBits - ChunkBits); ///< 36-bit device AS

    PageWords() = default;
    PageWords(const PageWords&) = delete;
    PageWords& operator=(const PageWords&) = delete;
    ~PageWords();

    /// The page's word slot; nullptr when its chunk was never allocated (or past the table) and
    /// `allocate` is false.
    [[nodiscard]] std::atomic<u32>* Slot(u64 page, bool allocate = false);
    [[nodiscard]] u32 Load(u64 page) const noexcept {
        const u64 chunk = page >> ChunkBits;
        if (chunk >= MaxChunks) {
            return 0;
        }
        const std::atomic<u32>* const array = chunks[chunk].load(std::memory_order_acquire);
        return array == nullptr ? 0
                                : array[page & (ChunkSize - 1)].load(std::memory_order_acquire);
    }
    /// Every allocated word back to 0 (a new renderer). Chunks stay allocated.
    void Clear();

private:
    std::array<std::atomic<std::atomic<u32>*>, MaxChunks> chunks{};
};

/// Counters (relaxed; the P3 stats line reads them).
struct Stats {
    std::atomic<u64> traps{};           ///< CPU stores that disarmed a page
    std::atomic<u64> foreign_disarms{}; ///< foreign writes that disarmed a page
    std::atomic<u64> foreign_bumps{};   ///< foreign writes that moved an unarmed page
    std::atomic<u64> requests{};        ///< pages the front asked to arm
    std::atomic<u64> arms{};
    std::atomic<u64> releases{};        ///< counts given back by the back (owed, over the cap)
    // Gauges set by the back once per frame.
    std::atomic<u64> armed{};
    std::atomic<u64> waiting{};
    std::atomic<u64> backoff{}; ///< waiting pages still in their backoff
    std::atomic<u64> hot{};     ///< pages whose backoff reached 8 frames or more
};

/// The process-wide watch (one GPU).
class Watch {
public:
    /// The words of the window [addr, addr + size), at most two pages (acquire loads; take it
    /// before reading the bytes).
    [[nodiscard]] Snap Take(u64 addr, u32 size) const noexcept {
        if (size == 0) {
            return {};
        }
        const u64 first = addr >> PageBits;
        const u64 last = (addr + size - 1) >> PageBits;
        if (last > first + 1) {
            return {};
        }
        const u32 w0 = SnapshotOf(words.Load(first));
        const u32 w1 = last != first ? SnapshotOf(words.Load(last)) : 0;
        const bool armed = (w0 & ArmedBit) != 0 && (last == first || (w1 & ArmedBit) != 0);
        return Snap{u64{w0} | (u64{w1} << 32), armed};
    }

    /// A trapped guest CPU write (Core::Memory, before the store lands): the pages it disarms
    /// give their count back through `release(page_address)` on this thread.
    template <typename Release>
    void OnCpuWrite(u64 addr, u64 size, Release&& release) {
        if (size == 0) {
            return;
        }
        const u64 last = (addr + size - 1) >> PageBits;
        for (u64 page = addr >> PageBits; page <= last; ++page) {
            std::atomic<u32>* const slot = words.Slot(page);
            if (slot == nullptr) {
                page |= PageWords::ChunkSize - 1; // the rest of the chunk is unallocated too
                continue;
            }
            u32 word = slot->load(std::memory_order_relaxed);
            while ((word & ArmedBit) != 0) {
                if (slot->compare_exchange_weak(word, (word & ~ArmedBit) + SeqOne,
                                                std::memory_order_acq_rel)) {
                    stats.traps.fetch_add(1, std::memory_order_relaxed);
                    release(page << PageBits);
                    break;
                }
            }
        }
    }

    /// A write the front's versions did not decode (any thread), noted as close to its bytes
    /// as the site allows.
    void NoteForeign(u64 addr, u64 size);

    /// The device pages [addr, addr + size) are about to be unmapped: armed and owed pages give
    /// their count back now through `release(page_address)`, while the CPU backing still exists
    /// (a count left on an unmapped device page would outlive the mapping).
    template <typename Release>
    void ReleaseRange(u64 addr, u64 size, Release&& release) {
        if (size == 0) {
            return;
        }
        const u64 last = (addr + size - 1) >> PageBits;
        for (u64 page = addr >> PageBits; page <= last; ++page) {
            std::atomic<u32>* const slot = words.Slot(page);
            if (slot == nullptr) {
                page |= PageWords::ChunkSize - 1;
                continue;
            }
            u32 word = slot->load(std::memory_order_relaxed);
            while ((word & (ArmedBit | OwedBit)) != 0) {
                if (slot->compare_exchange_weak(word, (word & ~(ArmedBit | OwedBit)) + SeqOne,
                                                std::memory_order_acq_rel)) {
                    stats.releases.fetch_add(1, std::memory_order_relaxed);
                    release(page << PageBits);
                    break;
                }
            }
        }
    }

    /// Front: the window's unarmed pages are wanted armed (one request per page until the back
    /// handled it).
    void RequestArm(u64 addr, u32 size);

    /// Back: the window's words become nonzero, so every foreign write from now on moves them
    /// (the compare memo keys on them).
    void MarkUsed(u64 addr, u32 size);

    /// Back: the requests since the last call.
    void TakeRequests(std::vector<u32>& out) {
        std::scoped_lock lock{queue_mutex};
        out.insert(out.end(), queue.begin(), queue.end());
        queue.clear();
    }

    /// Forget every page state (a new renderer; no page may still be armed).
    void Reset() {
        {
            std::scoped_lock lock{queue_mutex};
            queue.clear();
        }
        words.Clear();
    }

    PageWords words;
    Stats stats;

private:
    std::mutex queue_mutex;
    std::vector<u32> queue; ///< page numbers (36-bit device addresses: 24-bit pages)
};

[[nodiscard]] Watch& Get();

/// EDEN_NX2_CBUFWATCH (default 1): 0 keeps the per-bind byte compare (A/B).
[[nodiscard]] bool Enabled();

/// The foreign write hook: one cached-mode branch when the ring is off.
inline void HookForeign(u64 addr, u64 size) {
    if (On(Switch::CbufRing)) [[unlikely]] {
        Get().NoteForeign(addr, size);
    }
}

/// The back's arming: the front's requests, the armed list, backoff and the count balance.
///
/// Ops (the device memory's cached counts):
///  - `void Hold(std::span<const std::pair<u64, u64>> ranges, s32 delta)`: +1/-1 for each page
///    of the sorted, separated ranges;
///  - `bool TryHold(u64 page_address)`: +1 only when the page's count is already at least 1 (no
///    protection change; reactive flushing Phase 1).
class Arming {
public:
    static constexpr size_t MaxArmed = 4096; ///< 16 MiB of armed pages
    static constexpr size_t ArmBudget = 256; ///< new arms per tick
    static constexpr u32 MaxBackoff = 64;    ///< frames
    static constexpr u32 HotBackoff = 8;
    static constexpr u64 UnarmableRetry = 64; ///< frames before a Phase 1 page is tried again

    explicit Arming(Watch& watch_) : watch{&watch_} {}

    /// Once per frame: disarmed pages leave the armed list (backoff), owed counts are given
    /// back, requests whose backoff ended are armed (`only_held`: only pages others hold), and
    /// the oldest pages over the cap are released.
    template <typename Ops>
    void Tick(u64 frame, bool only_held, Ops& ops) {
        release.clear();
        ScanArmed(frame);
        if (!release.empty()) {
            HoldPages(ops, release, -1);
            release.clear();
        }
        watch->TakeRequests(waiting);
        Candidates(frame);
        if (!candidates.empty()) {
            Arm(frame, only_held, ops);
        }
        if (armed.size() > MaxArmed) {
            const size_t over = armed.size() - MaxArmed;
            for (size_t i = 0; i < over; ++i) {
                meta[armed[i]].listed = false;
                if (DisarmBack(armed[i])) {
                    release.push_back(armed[i]);
                }
            }
            armed.erase(armed.begin(), armed.begin() + static_cast<std::ptrdiff_t>(over));
        }
        if (!release.empty()) {
            HoldPages(ops, release, -1);
            release.clear();
        }
        Gauges(frame);
    }

    /// Every count the watch holds is given back (the renderer goes away).
    template <typename Ops>
    void ReleaseAll(Ops& ops) {
        release.clear();
        for (const u32 page : armed) {
            if (DisarmBack(page)) {
                release.push_back(page);
            }
        }
        armed.clear();
        watch->TakeRequests(waiting);
        for (const u32 page : waiting) {
            ClearBits(page, QueuedBit);
        }
        waiting.clear();
        if (!release.empty()) {
            HoldPages(ops, release, -1);
            release.clear();
        }
        meta.clear();
        Gauges(0);
    }

    [[nodiscard]] size_t Armed() const noexcept {
        return armed.size();
    }
    [[nodiscard]] size_t Waiting() const noexcept {
        return waiting.size();
    }
    /// The page's backoff in frames (tests, stats).
    [[nodiscard]] u32 Backoff(u64 page_address) const {
        const auto it = meta.find(static_cast<u32>(page_address >> PageBits));
        return it == meta.end() ? 1 : it->second.backoff;
    }

private:
    struct Meta {
        u64 last_arm{};
        u64 not_before{};
        u32 backoff{1};
        bool listed{};
    };

    /// Clears `bits` (never `armed`) whatever the sequence does meanwhile; true if this call
    /// cleared them.
    bool ClearBits(u32 page, u32 bits) {
        std::atomic<u32>* const slot = watch->words.Slot(page);
        if (slot == nullptr) {
            return false;
        }
        u32 word = slot->load(std::memory_order_relaxed);
        while ((word & bits) != 0) {
            if (slot->compare_exchange_weak(word, word & ~bits, std::memory_order_acq_rel)) {
                return true;
            }
        }
        return false;
    }

    /// The back takes an armed page's count back itself (cap, shutdown) or collects an owed one;
    /// true if it now owes the -1.
    bool DisarmBack(u32 page) {
        std::atomic<u32>* const slot = watch->words.Slot(page);
        if (slot == nullptr) {
            return false;
        }
        u32 word = slot->load(std::memory_order_relaxed);
        while ((word & (ArmedBit | OwedBit)) != 0) {
            if (slot->compare_exchange_weak(word, (word & ~(ArmedBit | OwedBit)) + SeqOne,
                                            std::memory_order_acq_rel)) {
                watch->stats.releases.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
        return false; // a trap or an unmap gave it back
    }

    void ScanArmed(u64 frame) {
        size_t kept = 0;
        for (const u32 page : armed) {
            const u32 word = watch->words.Load(page);
            if ((word & ArmedBit) != 0) {
                armed[kept++] = page;
                continue;
            }
            if ((word & OwedBit) != 0 && ClearBits(page, OwedBit)) {
                watch->stats.releases.fetch_add(1, std::memory_order_relaxed);
                release.push_back(page);
            }
            Meta& m = meta[page];
            m.listed = false;
            m.backoff = frame - m.last_arm <= m.backoff ? (std::min)(m.backoff * 2, MaxBackoff)
                                                        : (std::max)(m.backoff / 2, 1U);
            m.not_before = frame + m.backoff;
        }
        armed.resize(kept);
    }

    void Candidates(u64 frame) {
        candidates.clear();
        size_t kept = 0;
        for (const u32 page : waiting) {
            const u32 word = watch->words.Load(page);
            Meta& m = meta[page];
            if ((word & ArmedBit) != 0 || m.listed) {
                ClearBits(page, QueuedBit);
                continue;
            }
            if ((word & OwedBit) != 0 || m.not_before > frame ||
                candidates.size() >= ArmBudget) {
                waiting[kept++] = page;
                continue;
            }
            candidates.push_back(page);
        }
        waiting.resize(kept);
        std::ranges::sort(candidates);
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    }

    template <typename Ops>
    void Arm(u64 frame, bool only_held, Ops& ops) {
        // Each word before the count moves: any write after this load fails the arm CAS.
        before.clear();
        for (const u32 page : candidates) {
            before.push_back(watch->words.Load(page));
        }
        held.clear();
        if (only_held) {
            for (size_t i = 0; i < candidates.size(); ++i) {
                if (ops.TryHold(u64{candidates[i]} << PageBits)) {
                    held.push_back(static_cast<u32>(i));
                } else {
                    ClearBits(candidates[i], QueuedBit);
                    meta[candidates[i]].not_before = frame + UnarmableRetry;
                }
            }
        } else {
            HoldPages(ops, candidates, +1);
            for (size_t i = 0; i < candidates.size(); ++i) {
                held.push_back(static_cast<u32>(i));
            }
        }
        undo.clear();
        for (const u32 i : held) {
            const u32 page = candidates[i];
            std::atomic<u32>* const slot = watch->words.Slot(page, true);
            u32 expected = before[i];
            // Arm complete: the page is protected now; a write in between leaves it unarmed.
            if ((expected & (ArmedBit | OwedBit)) == 0 &&
                slot->compare_exchange_strong(expected, (expected | ArmedBit) & ~QueuedBit,
                                              std::memory_order_acq_rel)) {
                Meta& m = meta[page];
                m.listed = true;
                m.last_arm = frame;
                armed.push_back(page);
                watch->stats.arms.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            undo.push_back(page);
            ClearBits(page, QueuedBit);
            meta[page].not_before = frame + 1;
        }
        if (!undo.empty()) {
            HoldPages(ops, undo, -1);
        }
    }

    template <typename Ops>
    static void HoldPages(Ops& ops, std::vector<u32>& pages, s32 delta) {
        std::ranges::sort(pages);
        std::vector<std::pair<u64, u64>> ranges;
        for (const u32 page : pages) {
            const u64 addr = u64{page} << PageBits;
            if (!ranges.empty() && ranges.back().first + ranges.back().second == addr) {
                ranges.back().second += PageSize;
            } else {
                ranges.emplace_back(addr, PageSize);
            }
        }
        // Adjacent runs were merged above; the ranges are sorted and separated.
        ops.Hold(std::span<const std::pair<u64, u64>>{ranges}, delta);
    }

    void Gauges(u64 frame) {
        u64 in_backoff = 0;
        u64 hot = 0;
        for (const u32 page : waiting) {
            const auto it = meta.find(page);
            if (it != meta.end() && it->second.not_before > frame) {
                ++in_backoff;
            }
        }
        for (const auto& [page, m] : meta) {
            if (m.backoff >= HotBackoff) {
                ++hot;
            }
        }
        auto& stats = watch->stats;
        stats.armed.store(armed.size(), std::memory_order_relaxed);
        stats.waiting.store(waiting.size(), std::memory_order_relaxed);
        stats.backoff.store(in_backoff, std::memory_order_relaxed);
        stats.hot.store(hot, std::memory_order_relaxed);
    }

    Watch* watch;
    std::unordered_map<u32, Meta> meta;
    std::vector<u32> armed;   ///< oldest first
    std::vector<u32> waiting; ///< requests not armed yet (queued bit set)
    std::vector<u32> candidates;
    std::vector<u32> before;
    std::vector<u32> held;
    std::vector<u32> undo;
    std::vector<u32> release;
};

} // namespace VideoCommon::V2::CbufWatch
