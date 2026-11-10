// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/v2/v2_switches.h"

namespace Tegra::Engines {
class Maxwell3D;
}

namespace VideoCommon {
enum class DrainSite : u32;
}

// EDEN_NX2_V2_STATS=1 (pipeline v2 P0): counters that size the later phases, logged every 10 s
// by each thread that has some (HW_GPU, "V2 stats ..." lines):
// - the split front's waits for the back by site and sync class (MASTER_PLAN.md §6), with the
//   semaphore acquires classified against the releases the front itself parsed;
// - per 3D draw: the register groups changed since the previous draw, draws where only light
//   groups changed, runs of draws where nothing but draw parameters changed, and the hit rate
//   of a simulated draw cache keyed by a fingerprint of the state;
// - buffer and texture cache mutations (uploads, alias copies, rescales, deletions), split into
//   those made while a draw resolves and the rest.
// Off, every hook costs one branch on the cached switch.
namespace VideoCommon::V2::Stats {

[[nodiscard]] inline bool Enabled() {
    return On(Switch::Stats);
}

/// MASTER_PLAN.md §6, plus waits no class covers.
enum class SyncClass : u8 {
    A,          ///< The value is one the front produced (a release with a literal payload)
    B,          ///< As A, with the release still queued for the back ahead of the wait
    C,          ///< Data written by a producer (macro DMA reads, command lists)
    D,          ///< A GPU result (report counters, timestamps)
    Unknown,    ///< Released by the CPU or by work the front doesn't parse (compute, DMA)
    Structural, ///< Not a data dependency (maps, macro takeover, debug drains)
    Count,
};

enum class ReleaseKind : u8 {
    Literal,
    GpuValued,
};

enum class AcquireOp : u8 {
    Equal,
    Gequal,
    CircularGequal,
    Mask,     ///< (word & value) != 0
    MaskZero, ///< (word & value) == 0
};

// Split front (the GPU thread): semaphores and waits for the back.

/// A semaphore release the front sent (`address_space`: MemoryManager::GetID()).
void NoteRelease(size_t address_space, GPUVAddr address, u32 payload, ReleaseKind kind);
/// A 3D report semaphore trigger, with the 3D registers it reads (the front's copy).
void NoteReportSemaphore(size_t address_space, const u32* regs);
/// The class of an acquire the front evaluates now.
[[nodiscard]] SyncClass ClassifyAcquire(size_t address_space, GPUVAddr address, u32 value,
                                        AcquireOp op);
/// An acquire (or a poll of a suspended one) and whether its value was met.
void NoteAcquire(SyncClass sync_class, bool poll, bool met);
/// The front waited for the back since `start` (Nx2Profile::Ticks) for `site`.
void NoteDrain(DrainSite site, u64 start);

/// P2 (EDEN_NX2_V2_ACQUIRE / HAZARD): how a wait site was decided.
enum class Decision : u8 {
    Avoided,   ///< No wait: the value was known or nothing queued writes the range
    Forwarded, ///< No wait: decided by a queued write the front knows (class B, store forwarding)
    Narrowed,  ///< A targeted wait for one producer instead of the old wait
    Waited,    ///< The old wait, for the Reason
    Count,
};

/// Why a site still took the old wait (or what verify found).
enum class Reason : u8 {
    None,
    Settings,     ///< Accuracy or fence settings, macros on the back
    Operation,    ///< An acquire the view doesn't decide alone
    Unmapped,     ///< Unmapped or not contiguous in device memory
    GpuWritten,   ///< Pages the buffer cache may hold newer than memory
    UnknownWrite, ///< A queued write of unknown bytes (counter report, reduction)
    QueuedNotMet, ///< The queued write doesn't meet the acquire
    NotMet,       ///< Not met by memory (the acquire suspends as before)
    ReportPage,   ///< Pages reports write (query results may be held back)
    Overflow,     ///< The hazard table overflowed or a lookup was too long
    /// A suspended acquire with no producer known and nothing queued: the old read (Fix F3h:
    /// once per suspension; no timed net)
    Valve,
    Unlanded,     ///< A fence-flagged release to the word isn't in memory yet
    QmdProducer,  ///< A suspended acquire waits on a known producer that lands at a fence
    /// A suspended acquire with the back idle made its pending operations land (F3h: the
    /// waiter's back job: fence, submit, forced release; only while an operation is pending)
    Forced,
    /// An acquire passed (forwarded) on the word its pending QMD reductions leave (F3h, class A)
    ReductionView,
    /// SYNC: a SEM_EXECUTE acquire passed unmet after 2 s (counted residual)
    SemExecutePass,
    Count,
};

void NoteDecision(DrainSite site, Decision decision, Reason reason = Reason::None);
/// A targeted wait at `site` for the producers in `producers` (V2::ProducerBit): which writers
/// the front's narrowed waits wait for.
void NoteNarrowedBy(DrainSite site, u32 producers);
/// Verify: the new decision disagreed with the old path at `site`. `benign`: the new path waits
/// where the old one went on (it costs time, never ordering); counted apart.
void NoteMismatch(DrainSite site, bool benign = false);

/// SYNC_SUBMIT_SPEC Step 0 (R2-B-05): what a suspended acquire waits for (spec §2.2 wake classes).
enum class WakeClass : u8 {
    ReleaseParsed, ///< A release the front has yet to parse (a queued one doesn't meet it)
    Consumed,      ///< Work the back hasn't run yet
    Landing,       ///< A producer that lands at a fence operation
    Map,           ///< A map the back hasn't run yet (unmapped word)
    Readback,      ///< The GPU-written word's readback
    Cpu,           ///< Nothing known: a guest CPU write (memory read)
    Count,
};
/// A suspended acquire resumed after `ns`, last waiting for `wake` (front thread).
void NoteSuspended(WakeClass wake, u64 ns);
/// SYNC Step 1+: an acquire decided by S1's semaphore view, by row (V2::SemRow), and slot
/// evictions.
void NoteSemRow(u8 row);
void NoteSemEvictions(u64 count);
/// SYNC Step 5: a pending-channel sleep of the front ended (V2::FrontWake::SleepEnd).
void NoteFrontSleep(u8 end);

/// Attributes the waits for the back made while it lives (acquire sites) to `sync_class`.
class AcquireContext {
public:
    explicit AcquireContext(SyncClass sync_class);
    ~AcquireContext();
    AcquireContext(const AcquireContext&) = delete;
    AcquireContext& operator=(const AcquireContext&) = delete;

private:
    SyncClass previous{};
    bool enabled{};
};

// 3D draws (the thread running Maxwell3D methods: the back, or the GPU thread unsplit).

/// Per Maxwell3D instance: what changed since its last draw, and incremental hashes of its
/// state registers (XOR of a mix of each register index and value).
struct DrawTracker {
    u32 pending{};     ///< Bits of RegisterGroup written (changed) since the last draw
    u32 since_rehash{};
    u64 state_hash{};  ///< Pipeline, viewport, texture pools, render targets, shaders
    u64 stream_hash{}; ///< Vertex streams and the index buffer
    u64 cbuf_hash{};   ///< Constant buffer bindings at the last draw
    u32 topology{~0U};
    bool valid{};
};

/// A 3D register write (ProcessDirtyRegisters), before the register takes `value`.
void NoteWrite(DrawTracker& tracker, u32 method, u32 old_value, u32 value);
/// A register group hash (V2::GroupTracker, EDEN_NX2_V2_REGSEM) drifted from its registers:
/// counted with the P0 "hash drift".
void NoteHashDrift();
/// Back: an IR record of `kind` (V2::Ir::Kind) ran: its ring words and payload arena words.
void NoteIrRecord(u32 kind, size_t words, size_t payload_words);
/// Constant buffer data written (any value: the shader sees new data).
void NoteConstantData(DrawTracker& tracker);
/// A draw about to be resolved.
void NoteDraw(Tegra::Engines::Maxwell3D& maxwell3d, u32 topology, bool indexed, bool indirect);

/// Marks the cache mutations below as made while a draw resolves.
class DrawScope {
public:
    explicit DrawScope(bool enabled_);
    ~DrawScope();
    DrawScope(const DrawScope&) = delete;
    DrawScope& operator=(const DrawScope&) = delete;

private:
    bool enabled;
};

// Buffer and texture cache mutations (any thread).

enum class Mutation : u32 {
    BufferUpload,   ///< bytes: uploaded
    TextureRefresh, ///< bytes: guest size of the image
    AliasSync,      ///< bytes: alias copies made
    Rescale,
    ImageDelete,
    BufferDelete,
    Count,
};

void CountMutationSlow(Mutation mutation, u64 amount);

inline void CountMutation(Mutation mutation, u64 amount = 0) {
    if (Enabled()) [[unlikely]] {
        CountMutationSlow(mutation, amount);
    }
}

void ReportSlow(const char* thread_name);

/// Called by the GPU thread and the split's back between commands: logs and resets this
/// thread's totals every 10 s.
inline void Report(const char* thread_name) {
    if (Enabled()) [[unlikely]] {
        ReportSlow(thread_name);
    }
}

} // namespace VideoCommon::V2::Stats
