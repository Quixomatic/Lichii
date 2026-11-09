// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <iterator>
#include <type_traits>

#include "common/common_types.h"

// Pipeline v2 typed IR (MASTER_PLAN.md §5.2): the records the split front sends its back with
// EDEN_NX2_V2_IR=1, in the same SPSC ring as the word stream they replace (gpu_split.h).
//
// Every record is a fixed 2-word header, an optional fixed-size POD body that depends on the
// kind, and for some kinds a variable payload. Payloads (method data words, StateRun entries,
// callback captures) follow the record in the ring itself, which is the payload arena: it is
// allocated and freed in record order, so it needs no positions of its own.
//
//   word 0: bits 0-3   GpuSplit ring kind (always RingKind)
//           bits 4-7   GpuSplit flags (bit 7: FastDrawNeutral)
//           bits 8-15  Kind
//           bits 16-23 body size in words
//           bits 24-31 Flags
//           bits 32-63 record size in words, header included
//   word 1: sequence number: +1 per record, from 1 after each split start
//   body, then [thunk pointer + capture words] when HasThunk, else the kind's payload
//
// To add a record kind: add it to Kind (before Count) and to kind_names, define its body as a
// trivially copyable struct of whole u64 words below, emit it with GpuSplit::DeferTyped (with a
// thunk the back runs) or a dedicated GpuSplit emitter, and decode it in GpuSplit::ExecuteIr.
namespace VideoCommon::V2::Ir {

/// GpuSplit::Kind::Ir: the ring entry kind of every IR record.
inline constexpr u64 RingKind = 5;

enum class Kind : u8 {
    Invalid,
    Engine,     ///< EngineBody: the engine later methods go to (raw pointer, as before)
    StateRun,   ///< Method entries (StateEntry*) to the current engine, in order
    MethodData, ///< MethodDataBody, [segment], then the data words packed 2 per u64
    Callback,   ///< A deferred function no typed kind describes yet
    Draw,       ///< DrawBody (defined; draws still travel as StateRun method writes)
    Dispatch,   ///< DispatchBody (defined; compute launches are still method writes)
    Copy,       ///< CopyBody (defined; DMA and 2D copies are still method writes)
    Release,    ///< ReleaseBody: a semaphore or report release
    Acquire,    ///< AcquireBody (defined; acquires are evaluated on the front)
    Barrier,    ///< BarrierBody: WFI, cache invalidations, fence releases
    Readback,   ///< ReadbackBody: a flush of GPU data for a CPU read
    Map,        ///< MapBody (defined; maps are still callbacks)
    Present,    ///< PresentBody (defined; presents are still callbacks)
    SyncPoint,  ///< SyncPointBody: a host1x syncpoint increment or a CPU fence signal
    Count,
};

inline constexpr const char* kind_names[] = {
    "invalid", "engine",  "state run", "method data", "callback", "draw",    "dispatch",
    "copy",    "release", "acquire",   "barrier",     "readback", "map",     "present",
    "sync point",
};
static_assert(std::size(kind_names) == static_cast<size_t>(Kind::Count));

enum Flags : u64 {
    HasThunk = 1ULL << 24,   ///< The body is followed by a thunk the back calls with the capture
    HasSegment = 1ULL << 25, ///< MethodData: the body is followed by the DMA segment address
};

inline constexpr size_t HeaderWords = 2;

[[nodiscard]] constexpr u64 MakeHeader(Kind kind, size_t body_words, size_t words, u64 flags) {
    return RingKind | flags | (u64{static_cast<u8>(kind)} << 8) | (u64{body_words & 0xff} << 16) |
           (u64{static_cast<u32>(words)} << 32);
}
[[nodiscard]] constexpr Kind KindOf(u64 header) {
    return static_cast<Kind>((header >> 8) & 0xff);
}
[[nodiscard]] constexpr size_t BodyWords(u64 header) {
    return static_cast<size_t>((header >> 16) & 0xff);
}
[[nodiscard]] constexpr size_t RecordWords(u64 header) {
    return static_cast<size_t>(header >> 32);
}

// StateRun entries: one word per method write, two when it carries a DMA segment, in the
// format of the old Method ring entry (gpu_split.h) so the back runs them on the same path:
// bits 0-3 = 2 (method), 4 IsLast, 5 HasSegment, 6 Direct (a macro write), 8-23 method,
// 32-63 argument; a HasSegment entry is followed by the segment address.

/// MethodData: a method with several data words (CallMultiMethod).
struct MethodDataBody {
    u32 method;
    u32 amount; ///< Data words that follow (packed 2 per u64)
    u32 methods_pending;
    u32 reserved;
};

struct EngineBody {
    u64 engine; ///< Tegra::Engines::EngineInterface*
};

/// Sentinel for a size or value the front does not know (a GPU-valued result).
inline constexpr u64 Unknown = ~0ULL;

struct DrawBody {
    u64 fingerprint;   ///< V2::GroupTracker::Fingerprint of the draw's state groups
    u32 dirty_groups;  ///< V2::RegGroup bits changed since the previous draw
    u32 topology;
    u32 first;
    u32 count;
    u32 instances;
    u32 flags;         ///< bit 0 indexed, bit 1 indirect
    u64 indirect_address;
};

struct DispatchBody {
    u64 qmd_address;
    u64 write_begin; ///< GPU range the dispatch may write, or Unknown
    u64 write_end;
};

struct CopyBody {
    u64 source;
    u64 destination;
    u64 bytes; ///< or Unknown
};

enum class ReleaseOp : u32 {
    Payload,  ///< Writes `payload`
    Report,   ///< A GPU counter or timestamp: the value is Unknown
    Reduction,
};

struct ReleaseBody {
    u64 address;       ///< GPU address
    u64 payload;       ///< The value written, or Unknown
    u32 bytes;         ///< 4, 8 or 16; 0 when unknown
    ReleaseOp op;
    u64 address_space; ///< MemoryManager::GetID()
};

struct AcquireBody {
    u64 address;
    u64 address_space;
    u32 value;
    u32 op;
};

enum class BarrierOp : u32 {
    WaitForIdle,
    InvalidateCaches,
    ReleaseFences,
    SignalReference,
    TickWork,
    Invalidate, ///< A CPU write invalidated a range of the caches
};

struct BarrierBody {
    BarrierOp op;
    u32 reserved;
};

struct ReadbackBody {
    u64 address; ///< Device address
    u64 bytes;
};

struct MapBody {
    u64 gpu_address;
    u64 device_address;
    u64 size;
    u64 address_space;
};

struct PresentBody {
    u64 frame;
};

struct SyncPointBody {
    u32 id;    ///< Syncpoint id, or ~0 for a CPU fence of the command queue
    u32 value; ///< Fence value, 0 when the increment is relative
};

template <typename Body>
inline constexpr size_t BodyWordsOf = (sizeof(Body) + sizeof(u64) - 1) / sizeof(u64);

template <typename Body>
inline constexpr bool IsBody = std::is_trivially_copyable_v<Body> && sizeof(Body) % 8 == 0 &&
                               alignof(Body) <= alignof(u64) && BodyWordsOf<Body> < 256;

static_assert(IsBody<MethodDataBody> && IsBody<EngineBody> && IsBody<DrawBody> &&
              IsBody<DispatchBody> && IsBody<CopyBody> && IsBody<ReleaseBody> &&
              IsBody<AcquireBody> && IsBody<BarrierBody> && IsBody<ReadbackBody> &&
              IsBody<MapBody> && IsBody<PresentBody> && IsBody<SyncPointBody>);

/// A decoded record (for inspection, replay and verification).
struct RecordView {
    Kind kind;
    u64 header;
    u64 sequence;
    const u64* body;
    size_t body_words;
    size_t words;

    template <typename Body>
    [[nodiscard]] const Body& As() const {
        static_assert(IsBody<Body>);
        return *reinterpret_cast<const Body*>(body);
    }
};

[[nodiscard]] inline RecordView Decode(const u64* record) {
    const u64 header = record[0];
    return RecordView{KindOf(header), header,           record[1],
                      record + HeaderWords, BodyWords(header), RecordWords(header)};
}

/// A point in the IR stream: the record's sequence number and the ring position after it.
/// GpuSplit::WaitFor(Ticket) waits until the back executed it.
struct Ticket {
    u64 sequence{};
    u64 position{};
};

} // namespace VideoCommon::V2::Ir
