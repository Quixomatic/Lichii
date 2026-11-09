// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "common/common_types.h"
#include "video_core/v2/regsem.h"

// Pipeline v2 P6a: segmentation of the typed IR draw stream (MASTER_PLAN §5.6.1).
//
// The segmenter is the pure decision part. It is fed the draws and the other records of one IR
// stream in order and says where a segment starts (cut before a draw) and whether the open segment
// must be resolved serially on the commit thread, with the reason. Snapshots are taken by the
// caller at each cut (cow_regs.h).
namespace VideoCommon::V2::Resolve {

/// Why a segment is resolved on the commit thread instead of a shard. The first group is P5a pass
/// 2's list (IMPLEMENTATION_LOG "What still forces a P6 segment serial"), then data-dependent
/// resolves (§5.6.4), then the scheduling reasons. As in `gpu-parallel-record`, every serial
/// segment carries exactly one reason, the first one found.
enum class SerialReason : u8 {
    None,
    IndirectCount,     ///< indirect / macro HLE draw: arguments or counts from GPU memory
    InlineIndex,       ///< inline index draw: the index buffer is (re)created at bind
    MultiRangeStorage, ///< sparse storage buffers bound through the source pool (resolver)
    RecipeFallback,    ///< a replay's relocated binding misses and creates at once (resolver)
    HeapFull,          ///< a full bindless heap binding recycles a published slot (resolver)
    Transfer,          ///< DMA/2D copies, inline-to-memory, blits in the segment
    Readback,          ///< a guest-observable flush/readback record in the segment
    Map,               ///< a GMMU map/unmap in the segment
    CacheInvalidate,   ///< a cache invalidation barrier in the segment
    ComputedHandle,    ///< a handle computed by the shader over storage buffers (resolver)
    GpuWrittenHandles, ///< handle constant buffers on GPU-written pages (resolver)
    ReadAfterWrite,    ///< an in-segment hazard proxies and the barrier tracker can't order
    BackPressure,      ///< the bound on segments in flight was reached
    NoSnapshot,        ///< no register snapshot for the segment (resolver state unavailable)
    Stolen,            ///< the commit reached the segment before any shard started it
    Count,
};
inline constexpr size_t NumSerialReasons = static_cast<size_t>(SerialReason::Count);
inline constexpr std::array<const char*, NumSerialReasons> serial_reason_names{
    "none",          "indirect",      "inline index", "multi-range storage", "recipe fallback",
    "heap full",     "transfer",      "readback",     "map",                 "cache invalidate",
    "computed handle", "gpu-written handles", "read after write", "back pressure", "no snapshot",
    "stolen",
};

/// Why a segment started.
enum class CutReason : u8 {
    First,        ///< the first draw of a stream
    RenderPass,   ///< render target registers changed (a render pass change)
    Size,         ///< the segment reached its size (fixed K or the adaptive size)
    EngineSwitch, ///< a draw of another 3D engine (another channel)
    Isolate,      ///< the previous segment went serial: start a clean one
    None,         ///< no cut
    Count = None,
};
inline constexpr size_t NumCutReasons = static_cast<size_t>(CutReason::Count);
inline constexpr std::array<const char*, NumCutReasons> cut_reason_names{
    "first", "render pass", "size", "engine switch", "isolate",
};

/// EDEN_NX2_V2_PARALLEL_SEGMENT: `pass` (render pass changes only, capped at MaxK draws), a
/// number K (every K draws, and at render pass changes), or `adaptive` (the default: K adapts
/// between MinK and MaxK from the merge statistics, and render pass changes cut too).
struct SegmentPolicy {
    enum class Kind : u8 { Pass, Fixed, Adaptive };
    static constexpr u32 MinK = 16;
    static constexpr u32 MaxK = 512;
    static constexpr u32 DefaultK = 64;

    Kind kind = Kind::Adaptive;
    u32 k = DefaultK;

    [[nodiscard]] static SegmentPolicy Parse(const char* value) {
        SegmentPolicy policy;
        if (value == nullptr || *value == '\0' || std::strcmp(value, "adaptive") == 0) {
            return policy;
        }
        if (std::strcmp(value, "pass") == 0) {
            policy.kind = Kind::Pass;
            policy.k = MaxK;
            return policy;
        }
        const long parsed = std::strtol(value, nullptr, 10);
        if (parsed > 0) {
            policy.kind = Kind::Fixed;
            policy.k = static_cast<u32>(std::clamp<long>(parsed, 1, 1L << 20));
        }
        return policy;
    }
};

/// What the segmenter needs to know of a draw.
struct DrawEvent {
    u32 dirty_groups = 0; ///< RegGroup bits changed since the previous draw
    u64 engine = 0;       ///< identity of the 3D engine (channel) that draws
    bool indirect = false;
    bool inline_index = false;
};

/// What the segmenter needs to know of a non-draw record in the stream.
enum class RecordClass : u8 {
    Neutral,         ///< state, releases, syncpoints, dispatches: no effect on the decision
    Transfer,        ///< DMA or 2D copy, inline-to-memory
    Readback,        ///< guest-observable flush or readback
    Map,             ///< GMMU map/unmap
    CacheInvalidate, ///< cache invalidation barrier
};

class Segmenter {
public:
    /// Draws a segment holds before a render target change may cut it.
    static constexpr u32 MinPassDraws = 16;

    explicit Segmenter(SegmentPolicy policy_ = {}) : policy{policy_}, k{policy_.k} {}

    /// Before a draw: whether a segment starts at it (CutReason::None: the draw joins the open
    /// segment). A cut closes the open segment; the caller snapshots state for the new one.
    CutReason OnDraw(const DrawEvent& draw) {
        CutReason cut = CutReason::None;
        if (!started) {
            cut = CutReason::First;
        } else if (isolate_next || cut_next) {
            cut = CutReason::Isolate;
        } else if (draw.engine != engine) {
            cut = CutReason::EngineSwitch;
        } else if ((draw.dirty_groups & GroupBit(RegGroup::RenderTargets)) != 0 &&
                   draws >= (std::min)(MinPassDraws, k)) {
            // A pass boundary is a hint (the shards replay target changes as any register):
            // small passes coalesce until the segment holds MinPassDraws (F1-36: 90% of cuts
            // at Castle Town were passes of ~7 draws).
            cut = CutReason::RenderPass;
        } else if (draws >= k) {
            cut = CutReason::Size;
        }
        if (cut != CutReason::None) {
            Open(draw.engine);
        }
        ++draws;
        if (draw.indirect) {
            MarkSerial(SerialReason::IndirectCount);
        } else if (draw.inline_index) {
            MarkSerial(SerialReason::InlineIndex);
        }
        return cut;
    }

    /// A record between draws, in stream order (records before the first draw are ignored).
    /// Only the classes the resolvers read (SetRecordCapabilities) matter, and a record only
    /// affects the draws after it: they start a new segment, the open one stays parallel
    /// (AP6 T0: no isolate cascade). Shards that read only registers see none of them.
    void OnRecord(RecordClass record) {
        if (record == RecordClass::Neutral || !started ||
            (capabilities & (1U << static_cast<u32>(record))) == 0) {
            return;
        }
        cut_next = true;
    }
    /// The record classes whose effects the resolvers read (bits of 1 << RecordClass).
    void SetRecordCapabilities(u32 mask) {
        capabilities = mask;
    }

    /// The open segment must be resolved serially (the first reason is kept). The next draw then
    /// starts a clean segment, so one serial event costs at most the segment it falls in.
    void MarkSerial(SerialReason why) {
        if (!started || why == SerialReason::None) {
            return;
        }
        if (reason == SerialReason::None) {
            reason = why;
        }
        isolate_next = true;
    }

    /// The adaptive size (ignored unless the policy is adaptive).
    void SetSize(u32 size) {
        if (policy.kind == SegmentPolicy::Kind::Adaptive) {
            k = std::clamp(size, SegmentPolicy::MinK, SegmentPolicy::MaxK);
        }
    }

    [[nodiscard]] bool Started() const {
        return started;
    }
    [[nodiscard]] u64 OpenId() const {
        return id;
    }
    [[nodiscard]] u32 OpenDraws() const {
        return draws;
    }
    [[nodiscard]] SerialReason OpenReason() const {
        return reason;
    }
    [[nodiscard]] u32 Size() const {
        return k;
    }
    [[nodiscard]] const SegmentPolicy& Policy() const {
        return policy;
    }

    /// Restarts numbering (a new stream, e.g. a split start); the first segment gets `first_id`.
    void Reset(u64 first_id) {
        started = false;
        id = first_id - 1;
        draws = 0;
        reason = SerialReason::None;
        isolate_next = false;
        cut_next = false;
    }

private:
    void Open(u64 engine_) {
        started = true;
        ++id;
        engine = engine_;
        draws = 0;
        reason = SerialReason::None;
        isolate_next = false;
        cut_next = false;
    }

    SegmentPolicy policy;
    u32 k;
    bool started = false;
    bool isolate_next = false; ///< The open segment is serial: the next draw starts a clean one
    bool cut_next = false;     ///< A record the resolvers read: the next draw starts a segment
    u32 capabilities = 0;      ///< SetRecordCapabilities
    u64 id = ~u64{0}; ///< the first Open makes it 0 unless Reset set another start
    u64 engine = 0;
    u32 draws = 0;
    SerialReason reason = SerialReason::None;
};

/// The adaptive segment size: segments must be long enough that the per-segment cost (snapshot,
/// hand-off, merge) is small against the resolve, and short enough that the commit rarely waits
/// for the segment it needs next. Fed the merge statistics of a window of segments.
class AdaptiveSize {
public:
    static constexpr u32 Window = 256;

    /// One committed segment: `waited` if the commit had to wait for a running shard, `stolen` if
    /// no shard had started it, `utilisation` of the shards in percent over the window so far.
    /// Returns the new size at the end of a window, else 0.
    u32 Note(bool waited, bool stolen, u32 current) {
        waits += waited ? 1 : 0;
        steals += stolen ? 1 : 0;
        if (++count < Window) {
            return 0;
        }
        u32 size = current;
        if (waits * 10 > count) {
            // More than 10% of segments made the commit wait: shorter segments finish sooner.
            size = (std::max)(SegmentPolicy::MinK, current * 3 / 4);
        } else if (steals * 2 > count) {
            // Most segments are reached before a shard starts them: the producer lags, and
            // shorter segments would only add per-segment cost (AP6 §1.3(e)). Keep the size.
        } else if (waits == 0 && steals * 10 < count) {
            // Shards are ahead and the commit never waits: longer segments cost less overhead.
            size = (std::min)(SegmentPolicy::MaxK, current * 5 / 4 + 1);
        }
        count = 0;
        waits = 0;
        steals = 0;
        return size;
    }

private:
    u32 count = 0;
    u32 waits = 0;
    u32 steals = 0;
};

} // namespace VideoCommon::V2::Resolve
