// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "common/common_types.h"

// Pipeline v2 F2-02 (EDEN_NX2_V2_CBUFRING): decode-time constant buffer versions in a ring
// (MASTER_PLAN §5.3, §2.1). S1 (the front) makes a version when it decodes constant buffer data:
// the first write to a window {GPU address, size} since a draw consumed its version allocates
// ring space, takes the window's bytes and patches the words in; later writes patch in place.
// At the next draw the front publishes the changed windows' versions to the back as S1
// CbufVersion entries (before the Draw entries). The back maps them by device address and binds
// a version's ring range for a uniform buffer instead of copying the bytes into the stream.
//
// Positions are monotonic byte counts (the ring offset is position % capacity). The two threads
// share two counters:
//  - head (front writes, back reads): the end of the newest allocation;
//  - retired (back writes, front reads): every position below it is free again.
// Retirement is by timeline value: the back notes, per command buffer (scheduler tick), the
// lowest position it referenced and retires above it once that tick completed. Versions survive
// submits. A bind floor (head - capacity / 2, raised per tick) bounds how old a version the back
// still binds, so a version that stays bound for a long time can't hold the ring.
namespace VideoCommon::V2::CbufRing {

inline constexpr u64 NoPosition = std::numeric_limits<u64>::max();

class Ring {
public:
    explicit Ring(u64 capacity_) : capacity{capacity_} {}

    [[nodiscard]] u64 Capacity() const noexcept {
        return capacity;
    }

    /// Front: room for `size` bytes aligned to `align` (a power of two) that do not wrap; the
    /// allocation's position, or nothing when the unretired span would exceed the capacity (the
    /// caller makes no version). A tail too short to hold it is skipped.
    [[nodiscard]] std::optional<u64> Allocate(u64 size, u64 align) {
        const u64 current = head.load(std::memory_order_relaxed);
        u64 begin = (current + align - 1) & ~(align - 1);
        const u64 offset = begin % capacity;
        if (offset + size > capacity) {
            begin += capacity - offset; // the next lap's start
        }
        const u64 end = begin + size;
        if (size > capacity || end - retired.load(std::memory_order_acquire) > capacity) {
            ++refusals;
            return std::nullopt;
        }
        head.store(end, std::memory_order_release);
        return begin;
    }

    /// Ring offset of a position.
    [[nodiscard]] u64 Offset(u64 position) const noexcept {
        return position % capacity;
    }

    /// Back: whether a version at `position` may still be bound (not below the bind floor).
    [[nodiscard]] bool Usable(u64 position) const noexcept {
        return position >= floor;
    }

    /// Back: the command buffer of `tick` references `position` (Usable).
    void Use(u64 tick, u64 position) {
        if (tick == current_tick) {
            current_low = (std::min)(current_low, position);
        }
    }

    /// Back, before any Use in `tick`: when the tick moved, the previous command buffer's lowest
    /// reference is noted for retirement, the bind floor rises, and the ticks `is_free` reports
    /// complete give their space back.
    template <typename IsFree>
    void Roll(u64 tick, IsFree&& is_free) {
        if (tick == current_tick) {
            return;
        }
        if (current_low != NoPosition) {
            pending.push_back({current_tick, current_low});
        }
        current_tick = tick;
        current_low = NoPosition;
        const u64 newest = head.load(std::memory_order_acquire);
        if (newest > capacity / 2) {
            floor = (std::max)(floor, newest - capacity / 2);
        }
        while (!pending.empty() && is_free(pending.front().tick)) {
            pending.pop_front();
        }
        u64 free_below = floor;
        for (const Submit& submit : pending) {
            free_below = (std::min)(free_below, submit.low);
        }
        // Monotonic: every reference is at or above the floor of its time, which is at or above
        // the retired position then.
        if (free_below > retired.load(std::memory_order_relaxed)) {
            retired.store(free_below, std::memory_order_release);
        }
    }

    [[nodiscard]] u64 Head() const noexcept {
        return head.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 Retired() const noexcept {
        return retired.load(std::memory_order_acquire);
    }
    [[nodiscard]] u64 Floor() const noexcept {
        return floor;
    }
    [[nodiscard]] u64 Refusals() const noexcept {
        return refusals;
    }

private:
    struct Submit {
        u64 tick;
        u64 low;
    };
    u64 capacity;
    std::atomic<u64> head{};
    std::atomic<u64> retired{};
    u64 refusals{}; ///< front only
    // Back only.
    std::deque<Submit> pending;
    u64 current_tick{NoPosition};
    u64 current_low{NoPosition};
    u64 floor{};
};

/// The back's versions by device address (direct mapped), set by the S1 CbufVersion entries.
class BackTable {
public:
    /// Above the windows a frame binds (OoT: ~2-4k), so a version bound again later in the frame
    /// (another pass) is still found.
    static constexpr size_t NumEntries = 16384;

    struct Entry {
        u64 device_addr{};
        u32 size{};
        u64 position{NoPosition};
    };

    void Set(u64 device_addr, u32 size, u64 position) {
        entries[Slot(device_addr)] = Entry{device_addr, size, position};
    }
    void Drop(u64 device_addr) {
        Entry& entry = entries[Slot(device_addr)];
        if (entry.device_addr == device_addr) {
            entry.position = NoPosition;
        }
    }
    /// The version of a bind of `size` bytes at `device_addr`, if one covers it.
    [[nodiscard]] const Entry* Find(u64 device_addr, u32 size) const {
        const Entry& entry = entries[Slot(device_addr)];
        if (entry.device_addr != device_addr || entry.position == NoPosition ||
            entry.size < size) {
            return nullptr;
        }
        return &entry;
    }
    /// Why Find found nothing (the stats line's no-version split).
    enum class Miss : u8 { Absent, Dropped, Small };
    [[nodiscard]] Miss WhyMissed(u64 device_addr, u32 size) const {
        const Entry& entry = entries[Slot(device_addr)];
        if (entry.device_addr != device_addr) {
            return Miss::Absent;
        }
        return entry.position == NoPosition ? Miss::Dropped : Miss::Small;
    }
    void Clear() {
        entries.fill(Entry{});
    }

private:
    static size_t Slot(u64 address) {
        u64 key = address >> 6;
        key ^= key >> 17;
        key *= 0xbf58476d1ce4e5b9ULL;
        key ^= key >> 31;
        return static_cast<size_t>(key) & (NumEntries - 1);
    }
    std::array<Entry, NumEntries> entries{};
};

/// The ring's memory and the counters both threads report (one GPU).
struct Shared {
    static constexpr u64 Capacity = u64{16} << 20;

    /// Back (the Vulkan buffer cache): the ring buffer's mapping exists.
    void Init(u8* mapped_, u32 align_) {
        shadow = std::make_unique<u8[]>(Capacity);
        mapped = mapped_;
        align = (std::max)(align_, 16U);
        table.Clear();
        ready.store(true, std::memory_order_release);
    }
    void Shutdown() {
        ready.store(false, std::memory_order_release);
    }
    [[nodiscard]] bool Ready() const {
        return ready.load(std::memory_order_acquire);
    }

    Ring ring{Capacity};
    /// The front's cached copy of every version (the mapping may be write-combined): the front
    /// builds and patches versions here and copies them to the mapping when it publishes; the
    /// back compares with it.
    std::unique_ptr<u8[]> shadow;
    u8* mapped{};
    u32 align{256};
    BackTable table; ///< back only
    std::atomic<bool> ready{};
    // Front counters (relaxed; the back's stats line reads them).
    std::atomic<u64> versions{};
    std::atomic<u64> guest_bases{};
    std::atomic<u64> kept_bases{};
    std::atomic<u64> no_version{};
    std::atomic<u64> bind_versions{};   ///< made at a bind (no constant buffer data written)
    std::atomic<u64> in_flight_bases{}; ///< guest bytes taken while writers were in flight
    std::atomic<u64> refused{};         ///< no ring space
};

[[nodiscard]] inline Shared& Get() {
    static Shared shared;
    return shared;
}

/// Front: the constant buffer windows written since the last draw and their versions.
///
/// A new version's bytes come from guest memory when nothing writes the range in flight (the
/// front's hazard table, in-flight constant buffer data included, so guest memory is current);
/// from the window's previous version when only constant buffer data is in flight over it and
/// that version is not retired; from guest memory too when only constant buffer data is in
/// flight and the window has no version to copy (its earlier words may be rewritten before the
/// draw); otherwise there is no version. A bind of a window with no current version (one the
/// guest CPU fills) makes one from guest memory, and takes it again when guest memory no longer
/// holds its bytes. Either way the back binds a version only while its bytes equal guest memory
/// at the bind, so a write neither saw (a guest CPU write) costs a fallback, never a wrong bind.
class Decode {
public:
    static constexpr u32 MaxWindow = 4096; ///< DEFAULT_SKIP_CACHE_SIZE: larger ones bind cached
    /// Above the windows a frame writes (OoT: ~2-4k), so a window keeps its version between
    /// frames and its next write copies it instead of finding its own data in flight.
    static constexpr size_t NumWindows = 4096;
    /// Env::Writers result when the range's in-flight writers are unknown.
    static constexpr u32 UnknownWriters = ~0U;

    explicit Decode(Shared& shared_) : shared{&shared_} {}

    /// A constant buffer data write of `count` words at byte `offset` of the window {va, size}.
    /// Env: `u32 Writers(u64 va, u32 size)` (the producer bits of in-flight writes over the
    /// range, UnknownWriters when untranslatable) and `bool Read(u64 va, u32 size, u8* dst)`
    /// (guest memory).
    template <typename Env>
    void Write(u64 va, u32 size, u32 offset, const u32* words, u32 count, u32 cbuf_bit,
               Env& env) {
        Window& window = Lookup(va, size);
        if (size == 0 || size > MaxWindow || u64{offset} + u64{count} * 4 > size) {
            Invalidate(window);
            return;
        }
        Ring& ring = shared->ring;
        if (!window.valid || window.published || window.position < ring.Retired()) {
            if (!NewVersion(window, cbuf_bit, env)) {
                Invalidate(window);
                return;
            }
        }
        std::memcpy(shared->shadow.get() + ring.Offset(window.position) + offset, words,
                    size_t{count} * 4);
        window.written = true;
        MarkDirty(window);
    }

    /// A bind of the window {va, size} (the const_buffer registers at a bind_groups write).
    /// A window without a current version gets one from guest memory, published before the
    /// next draw; one that constant buffer data never wrote is taken again when guest memory no
    /// longer holds its bytes (a guest CPU write). Env as Write's, plus
    /// `bool Same(u64 va, u32 size, const u8* bytes)` (guest memory holds `bytes`).
    template <typename Env>
    void Bind(u64 va, u32 size, u32 cbuf_bit, Env& env) {
        if (size == 0 || size > MaxWindow) {
            return;
        }
        Window& window = Lookup(va, size);
        if (window.dirty) {
            return; // written since the last draw: its version is published with the draw
        }
        Ring& ring = shared->ring;
        if (window.valid && window.position >= ring.Retired()) {
            if (window.written ||
                env.Same(va, size, shared->shadow.get() + ring.Offset(window.position))) {
                return;
            }
        }
        if (!NewVersion(window, cbuf_bit, env, false)) {
            return; // the back keeps the old one; its compare decides
        }
        shared->bind_versions.fetch_add(1, std::memory_order_relaxed);
        window.written = false;
        MarkDirty(window);
    }

    /// Before a draw: `emit(va, size, position)` for each window changed since the last call
    /// (position NoPosition: the window has no version now), the versions copied to the ring.
    template <typename Emit>
    void Publish(Emit&& emit) {
        for (const Dropped& dropped : drops) {
            emit(dropped.va, dropped.size, NoPosition);
        }
        drops.clear();
        Ring& ring = shared->ring;
        const bool ready = shared->Ready(); // the ring buffer is gone after the renderer
        for (const u32 index : dirty) {
            Window& window = windows[index];
            window.dirty = false;
            if (ready && window.valid && window.position >= ring.Retired()) {
                const u64 offset = ring.Offset(window.position);
                if (shared->mapped != nullptr) {
                    std::memcpy(shared->mapped + offset, shared->shadow.get() + offset,
                                window.size);
                }
                window.published = true;
                emit(window.va, window.size, window.position);
            } else {
                window.valid = false;
                emit(window.va, window.size, NoPosition);
            }
        }
        dirty.clear();
    }

    [[nodiscard]] bool Pending() const {
        return !dirty.empty() || !drops.empty();
    }

    void Reset() {
        windows.fill(Window{});
        dirty.clear();
        drops.clear();
    }

private:
    struct Window {
        u64 va{NoPosition};
        u32 size{};
        u64 position{};
        bool valid{};
        bool published{};
        bool dirty{};
        bool written{}; ///< constant buffer data made or patched the version
    };
    struct Dropped {
        u64 va;
        u32 size;
    };

    Window& Lookup(u64 va, u32 size) {
        u64 key = (va >> 4) ^ (u64{size} << 44);
        key ^= key >> 29;
        key *= 0xbf58476d1ce4e5b9ULL;
        key ^= key >> 32;
        Window& window = windows[static_cast<size_t>(key) & (NumWindows - 1)];
        if (window.va != va || window.size != size) {
            if (window.dirty) {
                // Its writes since the last draw have no version: the back drops the old one.
                dirty.erase(std::find(dirty.begin(), dirty.end(),
                                      static_cast<u32>(&window - windows.data())));
                drops.push_back({window.va, window.size});
            }
            window = Window{va, size};
        }
        return window;
    }

    /// `may_keep`: the previous version may be the base (constant buffer data); a bind takes
    /// guest memory.
    template <typename Env>
    bool NewVersion(Window& window, u32 cbuf_bit, Env& env, bool may_keep = true) {
        Ring& ring = shared->ring;
        // Nothing in flight: guest memory is current (a guest CPU write since included).
        // Only this window's own data in flight: the previous version holds it; without one,
        // guest memory (the words in flight may be written again before the draw; the back's
        // compare drops the version if not).
        const u32 writers = env.Writers(window.va, window.size);
        const bool only_cbuf =
            writers != 0 && writers != UnknownWriters && (writers & ~cbuf_bit) == 0;
        const bool kept = may_keep && only_cbuf && window.valid &&
                          window.position >= ring.Retired();
        if (!kept && writers != 0 && !only_cbuf) {
            shared->no_version.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const std::optional<u64> position = ring.Allocate(window.size, shared->align);
        if (!position) {
            shared->no_version.fetch_add(1, std::memory_order_relaxed);
            shared->refused.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        u8* const bytes = shared->shadow.get() + ring.Offset(*position);
        if (kept) {
            // Allocations alias only retired positions, so the previous version is intact.
            std::memcpy(bytes, shared->shadow.get() + ring.Offset(window.position), window.size);
            shared->kept_bases.fetch_add(1, std::memory_order_relaxed);
        } else if (env.Read(window.va, window.size, bytes)) {
            shared->guest_bases.fetch_add(1, std::memory_order_relaxed);
            if (writers != 0) {
                shared->in_flight_bases.fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            shared->no_version.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        shared->versions.fetch_add(1, std::memory_order_relaxed);
        window.position = *position;
        window.valid = true;
        window.published = false;
        return true;
    }

    void MarkDirty(Window& window) {
        if (!window.dirty) {
            window.dirty = true;
            dirty.push_back(static_cast<u32>(&window - windows.data()));
        }
    }
    void Invalidate(Window& window) {
        window.valid = false;
        MarkDirty(window);
    }

    Shared* shared;
    std::array<Window, NumWindows> windows{};
    std::vector<u32> dirty;
    std::vector<Dropped> drops;
};

} // namespace VideoCommon::V2::CbufRing
