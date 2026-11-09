// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <bit>
#include <bitset>
#include <limits>

#include "common/common_types.h"

// Pipeline v2 register semantics (MASTER_PLAN.md §5.1, P1): one table per engine that says,
// for every register, which group it belongs to and which side effects a write to it has.
// The split front's register mirrors use it to update themselves (regsem_3d.h, with
// EDEN_NX2_V2_REGSEM=1, or alongside the hand-written mirror with =verify), and the engine
// that executes the methods uses the groups for its dirty bits and fingerprints.
//
// Groups (the same 11 as the P0 stats, v2_stats.cpp) get a dirty bit when a register in them
// changes value, and an incremental 64-bit hash: XOR of mix(register, value) over the group,
// updated in O(1) per changed write (AnyPS5's QueueState). Same-value writes change neither.
// The table is per engine class, not per title: Switch 1 (legacy Maxwell3D) titles can use the
// same table once their path adopts it.
namespace VideoCommon::V2 {

enum class RegGroup : u8 {
    Pipeline, ///< Everything not below: fixed-function and pipeline-key state
    Viewport, ///< Viewport transforms, viewports, scissors, window offset
    VertexStreams,
    IndexBuffer, ///< Address, limit, format (first and count are draw parameters)
    CbufBind,    ///< Dirty on every constant buffer bind (its registers are in Misc)
    CbufData,    ///< Constant buffer upload registers and data (not hashed)
    TexPools,    ///< TIC/TSC pools, sampler binding, bindless slot
    RenderTargets,
    Shaders,    ///< pipelines[], program_region
    DrawParams, ///< Vertex/index first and count, draw begin/end, base vertex/instance
    Misc,       ///< Triggers and state no draw reads (not hashed)
    Count,
};
inline constexpr size_t NumRegGroups = static_cast<size_t>(RegGroup::Count);

inline constexpr std::array<const char*, NumRegGroups> reg_group_names{
    "pipeline", "viewport", "vertex streams", "index buffer", "cbuf bind", "cbuf data",
    "tex pools", "render targets", "shaders", "draw params", "misc",
};

[[nodiscard]] constexpr u32 GroupBit(RegGroup group) {
    return 1U << static_cast<u32>(group);
}
inline constexpr u32 AllGroups = (1U << NumRegGroups) - 1;
/// Groups with a hash: CbufData and Misc change behind the write path (offset auto-increment,
/// back-side pointers) and no draw state depends on their register values.
inline constexpr u32 HashedGroups = AllGroups & ~GroupBit(RegGroup::CbufData) &
                                    ~GroupBit(RegGroup::Misc);

/// splitmix64's finalizer.
[[nodiscard]] constexpr u64 Mix64(u64 value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}
[[nodiscard]] constexpr u64 MixRegister(u32 index, u32 value) {
    return Mix64((u64{index} << 32) | value);
}

/// Native KeplerCompute register groups (the 0x1000 compute registers, Kepler and native
/// classes share the index space), in bit order.
enum class ComputeGroup : u8 {
    Pipeline, ///< Everything not below
    Upload,   ///< Inline-to-memory registers 0x60-0x6d (not hashed)
    Launch,   ///< Launch control and the QMD address 0xa0-0xbf (signals are Misc)
    Qmd,      ///< The inline QMD: its address 0xc6-0xc7 and words 0xc8-0x107 (native QMD state)
    Pools,    ///< TSC/TIC pools, code location, bindless texture cbuf slot
    Misc,     ///< Macro upload/bind, launches and QMD signals, report semaphore (not hashed)
    Count,
};
inline constexpr size_t NumComputeGroups = static_cast<size_t>(ComputeGroup::Count);
inline constexpr std::array<const char*, NumComputeGroups> compute_group_names{
    "pipeline", "upload", "launch", "qmd", "pools", "misc",
};
[[nodiscard]] constexpr u32 ComputeGroupBit(ComputeGroup group) {
    return 1U << static_cast<u32>(group);
}
inline constexpr u32 AllComputeGroups = (1U << NumComputeGroups) - 1;
inline constexpr u32 HashedComputeGroups = AllComputeGroups &
                                           ~ComputeGroupBit(ComputeGroup::Upload) &
                                           ~ComputeGroupBit(ComputeGroup::Misc);

/// P3 consumers of the dirty groups that don't run on every draw (a skipped or continued draw
/// doesn't look at the pipeline key): each takes what changed since it last looked.
enum class GroupSink : u8 {
    PipelineKey,   ///< The incremental pipeline key (EDEN_NX2_V2_PIPEKEY)
    ContinuedDraw, ///< The continued-draw decision (EDEN_NX2_V2_CONTDRAW)
    Count,
};
inline constexpr size_t NumGroupSinks = static_cast<size_t>(GroupSink::Count);
static_assert(NumGroupSinks == 2, "BasicGroupTracker::sinks initializes every sink");

/// Dirty bits and incremental hashes per group, for one engine's register file.
template <typename Group, size_t NumGroups, u32 Hashed>
struct BasicGroupTracker {
    static constexpr u32 All = (1U << NumGroups) - 1;
    static constexpr u32 HashedMask = Hashed;

    u32 dirty{All};   ///< Groups changed since the last BeginDraw (or TakeDirty)
    u32 draw_dirty{}; ///< `dirty` as BeginDraw took it: what the current draw sees changed
    u32 draws{};
    bool valid{}; ///< The hashes were computed from the registers once
    std::array<u64, NumGroups> hashes{};
    /// `dirty` accumulated per GroupSink since the sink last took it (BeginDraw adds to all).
    std::array<u32, NumGroupSinks> sinks{All, All};

    /// A register of `group` changes from `old_value` to `value` (callers compare first).
    void Change(Group group, u32 index, u32 old_value, u32 value) {
        const u32 bit = 1U << static_cast<u32>(group);
        dirty |= bit;
        if ((bit & Hashed) != 0) {
            hashes[static_cast<size_t>(group)] ^=
                MixRegister(index, old_value) ^ MixRegister(index, value);
        }
    }
    /// The group's data changed without a register value changing (constant buffer data, a bind).
    void Touch(Group group) {
        dirty |= 1U << static_cast<u32>(group);
    }
    /// Returns and clears the groups changed since the last call.
    u32 TakeDirty() {
        const u32 result = dirty;
        dirty = 0;
        return result;
    }
    /// Per draw: moves the dirty bits to draw_dirty for the draw's consumers.
    void BeginDraw() {
        draw_dirty = dirty;
        for (u32& sink : sinks) {
            sink |= dirty;
        }
        dirty = 0;
        ++draws;
    }
    /// The groups changed since `sink` last took them, including changes no draw took yet.
    u32 TakeSink(GroupSink sink) {
        u32& bits = sinks[static_cast<size_t>(sink)];
        const u32 result = bits | dirty;
        bits = 0;
        return result;
    }
    /// What TakeSink would return, without taking it.
    [[nodiscard]] u32 PeekSink(GroupSink sink) const {
        return sinks[static_cast<size_t>(sink)] | dirty;
    }
    /// Everything changed for every sink (the consumer's state is unknown, e.g. a new channel).
    void DirtyAllSinks() {
        sinks.fill(All);
    }
    /// A fingerprint of the hashed groups in `mask`.
    [[nodiscard]] u64 Fingerprint(u32 mask) const {
        mask &= Hashed;
        u64 result = mask;
        for (u32 bits = mask; bits != 0; bits &= bits - 1) {
            const u32 group = static_cast<u32>(std::countr_zero(bits));
            result ^= std::rotl(hashes[group], static_cast<int>(group * 5 + 1));
        }
        return Mix64(result);
    }
};

using GroupTracker = BasicGroupTracker<RegGroup, NumRegGroups, HashedGroups>;
using ComputeGroupTracker =
    BasicGroupTracker<ComputeGroup, NumComputeGroups, HashedComputeGroups>;

namespace RegSem {

/// Effects of a register write outside the register file (regsem_3d.h).
enum class Effect : u8 {
    ClearCode, ///< a: instruction pointer (compute: the value written)
    AddCode,   ///< a: instruction pointer, b: word
    Bind,      ///< a: position index, b: macro start. Returns whether `a` was in range.
};

/// Maxwell3D register flags (Table3D::flags).
enum Flags3D : u16 {
    Exec = 1 << 0,           ///< execution_mask: has side effects, sent even with its value
    CbData = 1 << 1,         ///< const_buffer.buffer[]: offset += 4, constant data written
    ShadowControl = 1 << 2,  ///< shadow_ram_control: its own shadow copy is the mode
    MacroCodePointer = 1 << 3, ///< load_mme.instruction_ptr: clears the macro code there
    MacroCode = 1 << 4,      ///< load_mme.instruction: uploads a word at instruction_ptr
    MacroBind = 1 << 5,      ///< load_mme.start_address: positions[start_address_ptr++]
    Falcon4 = 1 << 6,        ///< falcon[4]: shadow_scratch[0] = 1
    DrawManager = 1 << 7,    ///< Executable and handled by the draw manager
    DrawBegin = 1 << 8,      ///< draw.begin: instance id bits enter or leave an instance loop
    InlineIndex = 1 << 9,    ///< Inline index methods: leave the instance loop
    LoopSkippable = 1 << 10, ///< draw.end, vertex_buffer.first/count: droppable when unchanged
    DrawEnd = 1 << 11,
    InlineData = 1 << 12,    ///< inline_data: upload data only
    CbufBindFlag = 1 << 13,  ///< bind_groups[].raw_config: CbufBind is dirty on every write
    ReportSemaphore = 1 << 14, ///< report_semaphore.query: a release
    FastDrawQuiet = 1 << 15,   ///< Instance loop or fast draw delta method: no method break
};

/// What the executing Maxwell3D does with a method after the register file took it
/// (Maxwell3D::TableWrite, EDEN_NX2_V2_REGSEM=1): the side effects of ProcessMethodCall and the
/// draw manager, one per register, from the same table the front mirror uses.
enum class Action3D : u8 {
    Store,            ///< Not executable: stored when it changes the register (compare-on-write)
    StoreExec,        ///< Executable, stored only (vertex_buffer.first/count, index_buffer.first)
    TopologyOverride, ///< Stored; outside the fast path a method break
    CbData,           ///< Constant buffer data: upload, offset += 4
    CbufBind0,        ///< Constant buffer binds of stages 0-4
    CbufBind1,
    CbufBind2,
    CbufBind3,
    CbufBind4,
    DrawBegin,
    DrawEnd,
    RootSelect,
    RootLoad,
    WaitForIdle,
    ShadowControl,    ///< The shadow RAM mode: the unshadowed argument
    MacroCodePointer, ///< Macro upload pointer (the front's with front macros)
    MacroCode,
    MacroBind,        ///< positions[start_address_ptr++]
    Falcon4,          ///< shadow_scratch[0] = 1
    ReportSemaphore,
    RenderEnable,
    ClearReportValue,
    SyncInfo,
    LaunchDma,
    InlineData,
    FragmentBarrier,
    InvalidateTextureDataCache,
    TiledCacheBarrier,
    DrawManager, ///< The draw manager's ProcessMethodCall
};

struct Table3D {
    static constexpr size_t NumRegs = 0xe00;
    std::array<u16, NumRegs> flags{};
    std::array<RegGroup, NumRegs> groups{};
    std::array<Action3D, NumRegs> actions{};
    // Register indices the side effects touch.
    u32 const_buffer_offset{};
    u32 instruction_ptr{};
    u32 start_address_ptr{};
    u32 shadow_scratch{};
    u32 shadow_control{}; ///< shadow_ram_control: its shadow copy is the shadow RAM mode
};

/// EDEN_NX2_V2_REGSEM=verify on the executing Maxwell3D: the registers as the table-driven
/// update function alone makes them (no side effects), compared with the engine's after each
/// write (regsem_3d.h DryWrite3D / CompareBackWrite3D).
struct BackMirror3D {
    std::array<u32, Table3D::NumRegs> regs{};
    std::array<u32, Table3D::NumRegs> shadow{};
    u64 writes{};
    u64 mismatches{};
};

/// Builds the Maxwell3D table once (the first Maxwell3D constructor), from its execution mask.
void InitTable3D(const std::bitset<(std::numeric_limits<u16>::max)()>& execution_mask);
/// The Maxwell3D table; InitTable3D must have run.
[[nodiscard]] const Table3D& GetTable3D();

/// Recomputes the hashed groups of `tracker` from `regs` (Maxwell3D registers) and returns the
/// groups whose incremental hash had drifted from it.
u32 Rehash3D(GroupTracker& tracker, const u32* regs);

/// The executing Maxwell3D's register write path (ProcessDirtyRegisters), before the register
/// takes `value`.
inline void NoteWrite3D(GroupTracker& tracker, u32 method, u32 old_value, u32 value) {
    const Table3D& table = GetTable3D();
    if (method >= Table3D::NumRegs) {
        return;
    }
    const u16 flags = table.flags[method];
    if ((flags & (CbData | CbufBindFlag)) != 0) [[unlikely]] {
        tracker.Touch((flags & CbData) != 0 ? RegGroup::CbufData : RegGroup::CbufBind);
    }
    if (old_value != value) {
        tracker.Change(table.groups[method], method, old_value, value);
    }
}

/// The executing Maxwell3D is about to draw: dirty bits move to draw_dirty. With
/// EDEN_NX2_V2_REGSEM=verify or EDEN_NX2_V2_STATS, the hashes are recomputed every 4096 draws
/// and drift is counted.
void NoteDraw3D(GroupTracker& tracker, const u32* regs);

/// Native KeplerCompute register flags (the 0x1000 native registers).
enum FlagsCompute : u8 {
    ComputeClearCode = 1 << 0,   ///< 0x45: clears the macro code, compute code uploaded
    ComputeAddCode = 1 << 1,     ///< 0x46: uploads a macro word
    ComputeBindPointer = 1 << 2, ///< 0x47: sets the bind pointer, compute code uploaded
    ComputeBind = 1 << 3,        ///< 0x48: positions[bind_pointer++] when in range
    ComputeUploaded = 1 << 4,    ///< 0x45, 0x47: the compute macro set is the one in use
    /// 0x107 (inline QMD): the back points 0xad at the QMD for one launch and restores it, so
    /// the mirror sees no change.
    ComputeInlineQmd = 1 << 5,
    ComputeLaunch = 1 << 6,      ///< 0xae-0xb0: QMD launches and dependencies
    ComputeRelease = 1 << 7,     ///< 0x6c3: report semaphore release
};

[[nodiscard]] const std::array<u8, 0x1000>& TableCompute();
/// Native compute methods that only store their value (no table flag, not an inline upload,
/// not a macro call): KeplerCompute's plain-store fast path, from the table.
[[nodiscard]] const std::bitset<0xe00>& TableComputePlain();
/// The group of each compute register.
[[nodiscard]] const std::array<ComputeGroup, 0x1000>& TableComputeGroups();

/// Recomputes the hashed groups of a compute tracker from the first `count` (<= 0x1000)
/// registers; returns the groups that had drifted.
u32 RehashCompute(ComputeGroupTracker& tracker, const u32* registers, u32 count = 0x1000);

/// A compute register write (front mirror or the executing engine), before it stores.
inline void NoteWriteCompute(ComputeGroupTracker& tracker, u32 method, u32 old_value, u32 value) {
    if (method < 0x1000 && old_value != value) {
        tracker.Change(TableComputeGroups()[method], method, old_value, value);
    }
}

/// The executing KeplerCompute is about to dispatch: dirty bits move to draw_dirty, and with
/// EDEN_NX2_V2_REGSEM=verify or EDEN_NX2_V2_STATS the hashes are checked every 4096 dispatches.
void NoteDispatchCompute(ComputeGroupTracker& tracker, const u32* registers, u32 count);

} // namespace RegSem

} // namespace VideoCommon::V2
