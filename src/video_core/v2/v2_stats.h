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
