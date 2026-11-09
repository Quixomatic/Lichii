// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <memory>

#include "common/common_types.h"

// Pipeline v2 typed IR storage next to the ring (MASTER_PLAN.md §5.2): the payload arena for
// variable-size record data and the side table for object references. Both are owned by the
// split's front, which allocates; the back only reads. Neither allocates per record.
namespace VideoCommon::V2::Ir {

/// A FIFO arena of u64 words for record payloads. The front allocates contiguous spans tagged
/// with the record's sequence number and ring position; spans are retired in order once the
/// back's executed sequence number passes them (Retire). A span never crosses the arena's end:
/// the words skipped to wrap are retired with the span that skipped them. Bounded: Words words
/// and MaxMarks spans; when either is full, Begin fails and the caller retires or waits for the
/// back to execute Oldest().
class PayloadArena {
public:
    struct Mark {
        u64 sequence; ///< Record that owns the span
        u64 position; ///< Ring position after that record (what to wait for)
        u64 end;      ///< Arena position after the span
    };

    /// Allocates `words` (a power of two) of storage and `marks` (a power of two) span marks.
    void Allocate(size_t words, size_t marks) {
        storage = std::make_unique<u64[]>(words);
        mark_storage = std::make_unique<Mark[]>(marks);
        capacity = words;
        mark_capacity = marks;
        Clear();
    }
    void Release() {
        storage.reset();
        mark_storage.reset();
        capacity = 0;
        mark_capacity = 0;
        Clear();
    }
    void Clear() {
        head = 0;
        tail = 0;
        mark_head = 0;
        mark_tail = 0;
    }
    [[nodiscard]] bool Allocated() const {
        return capacity != 0;
    }

    /// Front: a new span of `words` for the record `sequence` that ends at ring `position`.
    /// Returns nullptr when there is no room.
    [[nodiscard]] u64* Begin(size_t words, u64 sequence, u64 position) {
        const size_t index = static_cast<size_t>(head & (capacity - 1));
        const size_t skip = index + words > capacity ? capacity - index : 0;
        if (words > capacity || head + skip + words - tail > capacity ||
            mark_head - mark_tail == mark_capacity) [[unlikely]] {
            return nullptr;
        }
        head += skip;
        u64* const span = &storage[head & (capacity - 1)];
        head += words;
        mark_storage[mark_head++ & (mark_capacity - 1)] = Mark{sequence, position, head};
        return span;
    }
    /// Front: grows the newest span by `words` in place. Returns where they go, or nullptr when
    /// they would not be contiguous or there is no room.
    [[nodiscard]] u64* Extend(size_t words) {
        const size_t index = static_cast<size_t>(head & (capacity - 1));
        if (index == 0 || index + words > capacity || head + words - tail > capacity)
            [[unlikely]] {
            return nullptr;
        }
        u64* const out = &storage[index];
        head += words;
        mark_storage[(mark_head - 1) & (mark_capacity - 1)].end = head;
        return out;
    }
    /// Front: frees the spans of records the back executed (sequence <= executed).
    void Retire(u64 executed) {
        while (mark_tail != mark_head) {
            const Mark& mark = mark_storage[mark_tail & (mark_capacity - 1)];
            if (mark.sequence > executed) {
                break;
            }
            tail = mark.end;
            ++mark_tail;
        }
    }
    [[nodiscard]] bool HasPending() const {
        return mark_tail != mark_head;
    }
    /// The oldest span not retired yet (HasPending).
    [[nodiscard]] const Mark& Oldest() const {
        return mark_storage[mark_tail & (mark_capacity - 1)];
    }
    [[nodiscard]] size_t Pending() const {
        return static_cast<size_t>(mark_head - mark_tail);
    }
    [[nodiscard]] u64 Used() const {
        return head - tail;
    }
    /// Word index of a span pointer (for a PayloadRef).
    [[nodiscard]] u32 IndexOf(const u64* span) const {
        return static_cast<u32>(span - storage.get());
    }
    /// Back: the words of a PayloadRef index.
    [[nodiscard]] const u64* At(u32 index) const {
        return &storage[index];
    }
    [[nodiscard]] size_t Capacity() const {
        return capacity;
    }

private:
    std::unique_ptr<u64[]> storage;
    std::unique_ptr<Mark[]> mark_storage;
    size_t capacity{};
    size_t mark_capacity{};
    u64 head{}; ///< Arena position of the next word (monotonic)
    u64 tail{}; ///< Arena position of the oldest live word
    u64 mark_head{};
    u64 mark_tail{};
};

/// Object references for records: the front interns a pointer and sends its index; the back
/// reads the pointer back. Append-only between Clear()s, so an index the back reads was written
/// before the record that names it was published. When full, the front drains the back and
/// clears it (GpuSplit::Ref).
class ObjectTable {
public:
    static constexpr u32 Capacity = 4096;
    static constexpr u32 None = ~0U;

    void Clear() {
        count = 0;
        slots.fill(0);
    }
    [[nodiscard]] u32 Size() const {
        return count;
    }

    /// Front: the index of `object` (nullptr included), or None when the table is full.
    [[nodiscard]] u32 Intern(const void* object) {
        const u64 key = reinterpret_cast<uintptr_t>(object);
        size_t slot = static_cast<size_t>((key >> 4) * 0x9e3779b97f4a7c15ULL >> 51) & SlotMask;
        for (;;) {
            const u32 entry = slots[slot];
            if (entry == 0) {
                if (count == Capacity) {
                    return None;
                }
                objects[count] = object;
                slots[slot] = ++count;
                return count - 1;
            }
            if (objects[entry - 1] == object) {
                return entry - 1;
            }
            slot = (slot + 1) & SlotMask;
        }
    }
    /// Back: the object an index names.
    [[nodiscard]] void* Get(u32 index) const {
        return const_cast<void*>(objects[index]);
    }
    template <typename T>
    [[nodiscard]] T* As(u32 index) const {
        return static_cast<T*>(Get(index));
    }

private:
    static constexpr size_t SlotCount = Capacity * 2;
    static constexpr size_t SlotMask = SlotCount - 1;
    static_assert(std::has_single_bit(SlotCount));

    std::array<const void*, Capacity> objects{};
    std::array<u32, SlotCount> slots{}; ///< 0: empty, else index + 1
    u32 count{};
};

} // namespace VideoCommon::V2::Ir
