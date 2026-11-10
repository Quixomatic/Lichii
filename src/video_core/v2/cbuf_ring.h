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
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/cbuf_watch.h"

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
        // F2-03 (v2/cbuf_watch.h): the version is what the shader reads while `trusted`, its
        // decoder's epoch is current and the watch's page words still equal `words`.
        u64 words{};
        u32 epoch{};
        bool trusted{};
        /// The fallback memo: the first `checked_size` bytes equalled guest memory at this
        /// collect epoch, with the pages' words then (a foreign write since moves them).
        mutable u64 checked_epoch{};
        mutable u64 checked_words{};
        mutable u32 checked_size{};
    };

    void Set(u64 device_addr, u32 size, u64 position, u64 words = 0, u32 epoch = 0,
             bool trusted = false) {
        entries[Slot(device_addr)] = Entry{
            .device_addr = device_addr,
            .size = size,
            .position = position,
            .words = words,
            .epoch = epoch,
            .trusted = trusted,
        };
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

/// F2-03: how the back validates a ring version at a bind.
enum class BindPath : u8 {
    Trusted,  ///< trusted by its decoder, and the watch's words did not move: no compare
    Memo,     ///< compared in this collect epoch, and the words did not move since
    Compared, ///< compare now
};

/// `size`: the bind's bytes (at most the entry's); `words_now`: the watch's words of the entry's
/// pages now; `collect_epoch`: the epoch the back validates at; `decode_epoch`, `decoders`:
/// Shared's now. The memo vouches only for the bytes its compare covered.
[[nodiscard]] inline BindPath ChooseBindPath(const BackTable::Entry& entry, u32 size,
                                             u64 words_now, u64 collect_epoch, u32 decode_epoch,
                                             u32 decoders) {
    if (entry.trusted && words_now == entry.words && entry.epoch == decode_epoch &&
        decoders == 1) {
        return BindPath::Trusted;
    }
    if (entry.checked_epoch == collect_epoch && entry.checked_words == words_now &&
        words_now != 0 && size <= entry.checked_size) {
        return BindPath::Memo;
    }
    return BindPath::Compared;
}

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
    std::atomic<u64> overlap_patches{}; ///< F2-03: data through one window patched another
    std::atomic<u64> cow_versions{};    ///< ... a published one, copied first
    std::atomic<u64> write_drops{};     ///< windows dropped by another producer's ranged write
    std::atomic<u64> trusted_versions{}; ///< versions published trusted
    std::atomic<u64> trust_upgrades{};   ///< versions trusted after one compare at a bind
    /// F2-03: the live decoders (one per 3D front). Versions are trusted only while there is
    /// exactly one: a decoder patches the windows it knows, not another channel's.
    std::atomic<u32> decoders{};
    /// Moves with every decoder made: back entries of an earlier decoder are not trusted.
    std::atomic<u32> decode_epoch{};
};

[[nodiscard]] inline Shared& Get() {
    static Shared shared;
    return shared;
}

/// A window's version as the front publishes it (position NoPosition: none now).
struct Published {
    u64 va{};
    u32 size{};
    u64 position{NoPosition};
    u64 device{NoPosition}; ///< F2-03: the front's device address (NoPosition: the back translates)
    u64 words{};            ///< ... the watch's page words before the version's bytes
    u32 epoch{};            ///< ... Shared::decode_epoch it was trusted under
    bool trusted{};
};

/// Front: the constant buffer windows written since the last draw and their versions.
///
/// A new version's bytes come from guest memory when nothing writes the range in flight (the
/// front's hazard table, in-flight constant buffer data included, so guest memory is current);
/// from the window's previous version when only constant buffer data is in flight over it and
/// that version is not retired; from guest memory too when only constant buffer data is in
/// flight and the window has no version to copy (its earlier words may be rewritten before the
/// draw); otherwise there is no version. A bind of a window with no current version (one the
/// guest CPU fills) makes one from guest memory, and takes it again when guest memory no longer
/// holds its bytes. Without the write watch the back binds a version only while its bytes equal
/// guest memory at the bind, so a write neither saw (a guest CPU write) costs a fallback, never a
/// wrong bind.
///
/// F2-03 (`watch`): versions are exact for every write the front decodes. Windows are indexed by
/// device page, so constant buffer data through one window patches every other window over the
/// same bytes (an unpublished version in place, a published one copied first), aliases
/// included; another producer's ranged write drops the windows it overlaps (NoteWrite); an
/// evicted window drops its back entry. Writes the front never decodes reach the back's trust
/// check through the watch's page words (v2/cbuf_watch.h).
class Decode {
public:
    static constexpr u32 MaxWindow = 4096; ///< DEFAULT_SKIP_CACHE_SIZE: larger ones bind cached
    /// Above the windows a frame writes (OoT: ~2-4k), so a window keeps its version between
    /// frames and its next write copies it instead of finding its own data in flight.
    static constexpr size_t NumWindows = 4096;
    /// Env::Writers result when the range's in-flight writers are unknown.
    static constexpr u32 UnknownWriters = ~0U;

    explicit Decode(Shared& shared_, bool watch_ = false) : shared{&shared_}, watch{watch_} {
        shared->decoders.fetch_add(1, std::memory_order_acq_rel);
        epoch = shared->decode_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    }
    ~Decode() {
        shared->decoders.fetch_sub(1, std::memory_order_acq_rel);
    }
    Decode(const Decode&) = delete;
    Decode& operator=(const Decode&) = delete;

    /// A constant buffer data write of `count` words at byte `offset` of the window {va, size}.
    /// Env: `u32 Writers(u64 va, u32 size)` (the producer bits of in-flight writes over the
    /// range, UnknownWriters when untranslatable), `bool Read(u64 va, u32 size, u8* dst)`
    /// (guest memory); with the watch also `u64 Device(u64 va, u32 size)` (the device address
    /// of a range that translates contiguously, else NoPosition) and `u64 Maps()` (moves at
    /// every GPU map).
    template <typename Env>
    void Write(u64 va, u32 size, u32 offset, const u32* words, u32 count, u32 cbuf_bit,
               Env& env) {
        Window& window = Lookup(va, size, env);
        const bool fits = size != 0 && size <= MaxWindow && u64{offset} + u64{count} * 4 <= size;
        Ring& ring = shared->ring;
        bool made = fits;
        if (fits && (!window.valid || window.published || window.position < ring.Retired())) {
            made = NewVersion(window, cbuf_bit, env);
        }
        if (made) {
            std::memcpy(shared->shadow.get() + ring.Offset(window.position) + offset, words,
                        size_t{count} * 4);
            window.written = true;
            MarkDirty(window);
        } else {
            Invalidate(window);
        }
        if constexpr (requires { env.Maps(); }) {
            if (watch && count != 0) {
                const u64 written = fits && window.device != NoPosition
                                        ? window.device + offset
                                        : env.Device(va + offset, count * 4);
                PatchOverlaps(window, written, reinterpret_cast<const u8*>(words), count * 4,
                              env);
            }
        }
    }

    /// F2-03: another producer's ranged write over device bytes [device, device + size) (front,
    /// GpuSplit's range write notes): the windows over it lose their versions.
    void NoteWrite(u64 device, u64 size) {
        if (!watch || size == 0) {
            return;
        }
        Collect(device, size, nullptr);
        for (const u32 index : overlaps) {
            Window& other = windows[index];
            if (other.valid) {
                Invalidate(other);
                shared->write_drops.fetch_add(1, std::memory_order_relaxed);
            }
        }
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
        Window& window = Lookup(va, size, env);
        if (window.dirty) {
            return; // written since the last draw: its version is published with the draw
        }
        Ring& ring = shared->ring;
        if (window.valid && window.position >= ring.Retired()) {
            const u8* const bytes = shared->shadow.get() + ring.Offset(window.position);
            bool current = false;
            if constexpr (requires { env.Words(u64{}, u32{}); }) {
                if (watch && window.device != NoPosition) {
                    // F2-03: a trusted version whose pages saw no write since needs no compare.
                    // Otherwise one compare, after the words: equal bytes on armed pages with
                    // nothing in flight make it trusted from now on.
                    const CbufWatch::Snap snap = env.Words(window.device, size);
                    if (Trusted(window) && snap.words == window.words) {
                        return;
                    }
                    if (!snap.armed) {
                        env.Arm(window.device, size);
                        if (window.written) {
                            return; // as without the watch: the back's compare decides
                        }
                        current = env.Same(va, size, bytes);
                    } else if (env.Writers(va, size) != 0) {
                        current = window.written || env.Same(va, size, bytes);
                    } else if (env.Same(va, size, bytes)) {
                        current = true;
                        Trust(window, snap);
                    }
                } else {
                    current = window.written || env.Same(va, size, bytes);
                }
            } else {
                current = window.written || env.Same(va, size, bytes);
            }
            if (current) {
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
        // `emit(const Published&)` (F2-03), or `emit(va, size, position)`.
        const auto send = [&](const Published& version) {
            if constexpr (std::is_invocable_v<Emit&, const Published&>) {
                emit(version);
            } else {
                emit(version.va, version.size, version.position);
            }
        };
        for (const Dropped& dropped : drops) {
            send(Published{.va = dropped.va, .size = dropped.size, .device = dropped.device});
        }
        drops.clear();
        Ring& ring = shared->ring;
        const bool ready = shared->Ready(); // the ring buffer is gone after the renderer
        for (const u32 index : dirty) {
            Window& window = windows[index];
            window.dirty = false;
            const u64 device = window.located ? window.device : NoPosition;
            if (ready && window.valid && window.position >= ring.Retired()) {
                const u64 offset = ring.Offset(window.position);
                // A published version only learnt its trust (Trust): its bytes are there.
                if (shared->mapped != nullptr && !window.published) {
                    std::memcpy(shared->mapped + offset, shared->shadow.get() + offset,
                                window.size);
                }
                window.published = true;
                const bool trusted = Trusted(window);
                if (trusted) {
                    shared->trusted_versions.fetch_add(1, std::memory_order_relaxed);
                }
                send(Published{
                    .va = window.va,
                    .size = window.size,
                    .position = window.position,
                    .device = device,
                    .words = window.words,
                    .epoch = epoch,
                    .trusted = trusted,
                });
            } else {
                window.valid = false;
                send(Published{.va = window.va, .size = window.size, .device = device});
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
        page_windows.clear();
        UntrustAll();
    }

private:
    struct Window {
        u64 va{NoPosition};
        u32 size{};
        u64 position{};
        u64 device{NoPosition}; ///< F2-03: where the window was located (Locate), indexed there
        u64 maps{};             ///< ... the env's Maps() then
        u64 words{};            ///< ... the watch's page words before the version's bytes
        u32 epoch{};            ///< ... the decoder's epoch when trusted
        bool trusted{};
        bool located{};
        bool valid{};
        bool published{};
        bool dirty{};
        bool written{}; ///< constant buffer data made or patched the version
    };
    struct Dropped {
        u64 va;
        u32 size;
        u64 device; ///< F2-03: the device address the back entry has (NoPosition: translate)
    };

    template <typename Env>
    Window& Lookup(u64 va, u32 size, Env& env) {
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
                drops.push_back({window.va, window.size, window.device});
            } else if (watch && window.valid) {
                // F2-03: no overlap patches it once it leaves the index.
                drops.push_back({window.va, window.size, window.device});
            }
            Unindex(window);
            window = Window{va, size};
        }
        if constexpr (requires { env.Maps(); }) {
            if (watch) {
                Locate(window, env);
            }
        }
        return window;
    }

    /// F2-03: the window's device address for the current mappings, indexed by device page. A
    /// window whose bytes moved to another device address loses its version.
    template <typename Env>
    void Locate(Window& window, Env& env) {
        const u64 maps = env.Maps();
        if (window.located && window.maps == maps) {
            return;
        }
        const u64 device = window.size != 0 && window.size <= MaxWindow
                               ? env.Device(window.va, window.size)
                               : NoPosition;
        if (window.located && device != window.device) {
            if (window.valid) {
                drops.push_back({window.va, window.size, window.device});
                Invalidate(window);
            }
            Unindex(window);
        }
        if (!window.located || device != window.device) {
            window.device = device;
            Index(window);
        }
        window.located = true;
        window.maps = maps;
    }

    void Index(const Window& window) {
        if (window.device == NoPosition) {
            return;
        }
        const u32 index = static_cast<u32>(&window - windows.data());
        const u64 last = (window.device + window.size - 1) >> 12;
        for (u64 page = window.device >> 12; page <= last; ++page) {
            page_windows[page].push_back(index);
        }
    }
    void Unindex(const Window& window) {
        if (!window.located || window.device == NoPosition) {
            return;
        }
        const u32 index = static_cast<u32>(&window - windows.data());
        const u64 last = (window.device + window.size - 1) >> 12;
        for (u64 page = window.device >> 12; page <= last; ++page) {
            const auto it = page_windows.find(page);
            if (it == page_windows.end()) {
                continue;
            }
            std::vector<u32>& list = it->second;
            const auto found = std::find(list.begin(), list.end(), index);
            if (found != list.end()) {
                *found = list.back();
                list.pop_back();
            }
            if (list.empty()) {
                page_windows.erase(it);
            }
        }
    }

    /// The located windows (but `self`) over device bytes [device, device + size), in overlaps.
    void Collect(u64 device, u64 size, const Window* self) {
        overlaps.clear();
        const u64 end = device + size;
        const auto take = [&](const std::vector<u32>& list) {
            for (const u32 index : list) {
                const Window& other = windows[index];
                if (&other == self || other.device >= end ||
                    other.device + other.size <= device) {
                    continue;
                }
                if (std::find(overlaps.begin(), overlaps.end(), index) == overlaps.end()) {
                    overlaps.push_back(index);
                }
            }
        };
        const u64 first = device >> 12;
        const u64 last = (end - 1) >> 12;
        if (last - first + 1 > page_windows.size()) {
            // A large write: walk the index instead of the pages.
            for (const auto& [page, list] : page_windows) {
                if (page >= first && page <= last) {
                    take(list);
                }
            }
            return;
        }
        for (u64 page = first; page <= last; ++page) {
            if (const auto it = page_windows.find(page); it != page_windows.end()) {
                take(it->second);
            }
        }
    }

    /// F2-03 (G1): constant buffer data `src` written to device bytes [device, device + size)
    /// through `self` reaches every other window over them: an unpublished version is patched in
    /// place, a published one is copied into a new version first; a window with no ring space
    /// loses its version. Untranslatable data untrusts every version.
    template <typename Env>
    void PatchOverlaps(Window& self, u64 device, const u8* src, u32 size, Env& env) {
        if (device == NoPosition) {
            UntrustAll();
            return;
        }
        Collect(device, size, &self);
        Ring& ring = shared->ring;
        const u64 maps = env.Maps();
        for (const u32 index : overlaps) {
            Window& other = windows[index];
            if (!other.valid || other.position < ring.Retired()) {
                continue;
            }
            if (other.maps != maps) {
                Invalidate(other); // located under older mappings: its next use locates it
                continue;
            }
            if (other.published) {
                const std::optional<u64> position = ring.Allocate(other.size, shared->align);
                if (!position) {
                    shared->refused.fetch_add(1, std::memory_order_relaxed);
                    Invalidate(other);
                    continue;
                }
                std::memcpy(shared->shadow.get() + ring.Offset(*position),
                            shared->shadow.get() + ring.Offset(other.position), other.size);
                other.position = *position;
                other.published = false;
                shared->cow_versions.fetch_add(1, std::memory_order_relaxed);
            }
            const u64 begin = (std::max)(device, other.device);
            const u64 end = (std::min)(device + size, other.device + other.size);
            std::memcpy(shared->shadow.get() + ring.Offset(other.position) +
                            (begin - other.device),
                        src + (begin - device), static_cast<size_t>(end - begin));
            other.written = true;
            MarkDirty(other);
            shared->overlap_patches.fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// Every version made so far stops being trusted (front and back: the epoch moves).
    void UntrustAll() {
        epoch = shared->decode_epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
    }

    /// `may_keep`: the previous version may be the base (constant buffer data); a bind takes
    /// guest memory.
    template <typename Env>
    bool NewVersion(Window& window, u32 cbuf_bit, Env& env, bool may_keep = true) {
        Ring& ring = shared->ring;
        // F2-03: the pages' words before any byte is read (a write after this moves them).
        CbufWatch::Snap snap{};
        bool watched = false;
        if constexpr (requires { env.Words(u64{}, u32{}); }) {
            watched = watch && window.device != NoPosition;
            if (watched) {
                snap = env.Words(window.device, window.size);
            }
        }
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
        // Trusted: every page armed at the snapshot, and the base exact for it: guest memory
        // with nothing in flight, or a trusted previous version whose pages saw no write since
        // (bases read while constant buffer data was in flight never are, G2).
        const bool trusted =
            watched && snap.armed &&
            (kept ? Trusted(window) && window.words == snap.words : writers == 0);
        window.trusted = trusted;
        window.words = snap.words;
        window.epoch = epoch;
        if constexpr (requires { env.Arm(u64{}, u32{}); }) {
            if (watched && !snap.armed) {
                env.Arm(window.device, window.size);
            }
        }
        window.position = *position;
        window.valid = true;
        window.published = false;
        return true;
    }

    [[nodiscard]] bool Trusted(const Window& window) const {
        return window.trusted && window.epoch == epoch;
    }
    /// F2-03: the current version equals guest memory read after `snap` with nothing in flight.
    void Trust(Window& window, const CbufWatch::Snap& snap) {
        window.trusted = true;
        window.words = snap.words;
        window.epoch = epoch;
        MarkDirty(window); // the back's entry learns it (no copy: the bytes did not change)
        shared->trust_upgrades.fetch_add(1, std::memory_order_relaxed);
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
    bool watch; ///< F2-03 (CbufWatch::Enabled() when made)
    u32 epoch{};
    std::array<Window, NumWindows> windows{};
    std::vector<u32> dirty;
    std::vector<Dropped> drops;
    std::unordered_map<u64, std::vector<u32>> page_windows; ///< device page → window indices
    std::vector<u32> overlaps;                              ///< Collect
};

} // namespace VideoCommon::V2::CbufRing
