// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>

#include "common/logging.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/resolve/cbuf_table.h"
#include "video_core/v2/v2_switches.h"

// Table-driven register mirrors for the split front (EDEN_NX2_V2_REGSEM). The functions here
// do what Maxwell3D::FrontWrite / FrontCallMultiMethod and KeplerCompute::FrontApply do by
// hand, from the flags in regsem.h, on any state with the members they use:
//   3D:      regs, shadow, control, instance_loop, sink_pending, draw_pending, skip_same,
//            v2_groups (Maxwell3D::FrontState and Mirror3D)
//   compute: registers, compute_uploaded, bind_pointer (KeplerCompute::FrontState, MirrorCompute)
// Effects outside the register file (macro code and bindings) go to an `effects(Effect, a, b)`
// callback; verify mode passes one that does nothing.
namespace VideoCommon::V2::RegSem {

using ShadowRamControl = Tegra::Engines::Maxwell3D::Regs::ShadowRamControl;

struct Mirror3D {
    std::array<u32, Table3D::NumRegs> regs{};
    std::array<u32, Table3D::NumRegs> shadow{};
    ShadowRamControl control{};
    bool instance_loop{};
    bool sink_pending{};
    bool draw_pending{};
    bool skip_same{};
    GroupTracker v2_groups;
    Resolve::CbufTable v2_cbufs; ///< P6c: the constant buffer bindings these registers made
};

/// Per front Maxwell3D: the mode it started with and, for verify, the table-driven copy.
struct Front3D {
    Mode mode{};
    std::unique_ptr<Mirror3D> mirror; ///< Verify only
    u64 writes{};
    u64 mismatches{};
};

/// The shadow RAM step of every Maxwell3D register write (ProcessShadowRam): Track keeps the
/// argument in the shadow copy, Replay writes the shadow copy instead. Shared by the front
/// mirrors and the executing engine's table path. `kept`: whether the shadow already held it.
[[nodiscard]] inline u32 ApplyShadow3D(ShadowRamControl control, u32* shadow, u32 method,
                                       u32 argument, bool* kept = nullptr) {
    if (control == ShadowRamControl::Track || control == ShadowRamControl::TrackWithFilter) {
        if (kept != nullptr) {
            *kept = shadow[method] == argument;
        }
        shadow[method] = argument;
        return argument;
    }
    if (control == ShadowRamControl::Replay) {
        return shadow[method];
    }
    return argument;
}

/// P6c: a write to bind_groups[stage].raw_config binds the const_buffer registers to a slot
/// (Maxwell3D::ProcessCBBind), on states that keep the bindings (v2_cbufs).
inline void BindCbuf3D(Resolve::CbufTable& cbufs, const u32* regs, u32 method) {
    using Regs = Tegra::Engines::Maxwell3D::Regs;
    constexpr u32 first = static_cast<u32>(offsetof(Regs, bind_groups) / sizeof(u32)) + 4;
    constexpr u32 stride = static_cast<u32>(sizeof(Regs::BindGroup) / sizeof(u32));
    constexpr u32 cbuf = static_cast<u32>(offsetof(Regs, const_buffer) / sizeof(u32));
    static_assert(offsetof(Regs::BindGroup, raw_config) == 4 * sizeof(u32));
    static_assert(offsetof(Regs::ConstantBuffer, address_high) == 4 &&
                  offsetof(Regs::ConstantBuffer, address_low) == 8);
    const u32 stage = (method - first) / stride;
    const u32 config = regs[method];
    const u32 slot = (config >> 4) & 0x1f;
    if (stage >= Resolve::CbufStages || slot >= Resolve::CbufSlots) {
        return; // the slot field is 5 bits; the state array holds 18
    }
    cbufs.stages[stage][slot] = Resolve::CbufBinding{
        .address = (u64{regs[cbuf + 1]} << 32) | u64{regs[cbuf + 2]},
        .size = regs[cbuf],
        .enabled = config & 1,
    };
    ++cbufs.changes;
}

template <typename State>
inline void NoteCbufBind3D(State& s, u32 method) {
    if constexpr (requires { s.v2_cbufs_wanted; }) {
        if (!s.v2_cbufs_wanted) {
            return; // nobody reads them (the front without EDEN_NX2_V2_PARALLEL)
        }
    }
    if constexpr (requires { s.v2_cbufs; }) {
        BindCbuf3D(s.v2_cbufs, s.regs.data(), method);
    }
}

#if defined(_MSC_VER) && !defined(__clang__)
#define REGSEM_FORCE_INLINE __forceinline
#else
#define REGSEM_FORCE_INLINE inline __attribute__((always_inline))
#endif

/// States with a `v2_hash_log` (the front's FrontState) defer the group hashes to the log's
/// Fold (HashLog3D); the others (verify mirrors, tests) update them per write.
template <typename State>
REGSEM_FORCE_INLINE void Store3D(State& s, const Table3D& table, u32 index, u32 value) {
    const u32 old_value = s.regs[index];
    if (old_value != value) {
        s.regs[index] = value;
        if constexpr (requires { s.v2_dirty_blocks; }) {
            s.v2_dirty_blocks |= u64{1} << (index >> 6); // P6 snapshots: blocks of 64
        }
        if constexpr (requires { s.v2_hash_log; }) {
            s.v2_hash_log.Change(s.v2_groups, table, index, old_value, s.regs.data());
        } else {
            s.v2_groups.Change(table.groups[index], index, old_value, value);
        }
    }
}

/// One method write (FrontWrite): applies the shadow RAM, drops writes that change nothing
/// (compare-on-write: no dirty bit, no ring entry), updates the registers, the groups and the
/// back-side effects the mirror must reproduce. Returns whether the write goes to the back.
template <typename State, typename Effects>
REGSEM_FORCE_INLINE bool Write3D(State& s, const Table3D& table, u32 method, u32 argument,
                                 bool from_macro, Effects&& effects) {
    bool shadow_kept = true;
    const u32 value = ApplyShadow3D(s.control, s.shadow.data(), method, argument, &shadow_kept);
    const u16 flags = table.flags[method];
    if ((flags & Exec) == 0) {
        // The back skips same-value state writes and its shadow RAM holds the value already.
        if (s.regs[method] == value && shadow_kept && s.skip_same) {
            return false;
        }
        Store3D(s, table, method, value);
        s.sink_pending = !from_macro;
        return true;
    }
    if ((flags & DrawBegin) != 0) {
        // Instance id, bits 27:26: First leaves an instance loop, Subsequent enters one.
        const u32 instance = (value >> 26) & 3;
        if (instance == 0) {
            s.instance_loop = false;
        } else if (instance == 1) {
            s.instance_loop = true;
        }
    } else if ((flags & InlineIndex) != 0) {
        s.instance_loop = false;
    }
    // Instance loop methods the draw manager ignores when unchanged, unless they would end a
    // deferred draw (state writes still in the back's sink) or draw (draw.end outside a loop).
    if ((flags & LoopSkippable) != 0 && s.skip_same && !s.sink_pending && shadow_kept &&
        s.regs[method] == value && ((flags & DrawEnd) == 0 || s.instance_loop)) {
        return false;
    }
    Store3D(s, table, method, value);
    if ((flags & ShadowControl) != 0) {
        s.control = static_cast<ShadowRamControl>(argument);
        s.shadow[method] = argument;
    } else if ((flags & MacroCodePointer) != 0) {
        effects(Effect::ClearCode, s.regs[table.instruction_ptr], 0);
    } else if ((flags & MacroCode) != 0) {
        effects(Effect::AddCode, s.regs[table.instruction_ptr], value);
    } else if ((flags & MacroBind) != 0) {
        const u32 pointer = s.regs[table.start_address_ptr];
        effects(Effect::Bind, pointer, value);
        Store3D(s, table, table.start_address_ptr, pointer + 1);
    } else if ((flags & Falcon4) != 0) {
        Store3D(s, table, table.shadow_scratch, 1);
    } else if ((flags & CbData) != 0) {
        Store3D(s, table, table.const_buffer_offset,
                s.regs[table.const_buffer_offset] + static_cast<u32>(sizeof(u32)));
        s.v2_groups.Touch(RegGroup::CbufData);
    } else if ((flags & DrawManager) != 0) {
        s.draw_pending = true;
    }
    if ((flags & CbufBindFlag) != 0) {
        s.v2_groups.Touch(RegGroup::CbufBind);
        NoteCbufBind3D(s, method);
    }
    s.sink_pending = false;
    return true;
}

/// A method with several data words (FrontCallMultiMethod), always sent.
template <typename State, typename Effects>
void WriteMulti3D(State& s, const Table3D& table, u32 method, const u32* words, u32 amount,
                  Effects&& effects) {
    const u16 flags = table.flags[method];
    const bool tracked =
        s.control == ShadowRamControl::Track || s.control == ShadowRamControl::TrackWithFilter;
    if ((flags & CbData) != 0) {
        Store3D(s, table, table.const_buffer_offset,
                s.regs[table.const_buffer_offset] + amount * static_cast<u32>(sizeof(u32)));
        s.v2_groups.Touch(RegGroup::CbufData);
    } else if ((flags & InlineData) != 0) {
        // Upload data only.
    } else if ((flags & InlineIndex) != 0 && s.control != ShadowRamControl::Replay) {
        s.instance_loop = false;
        if (amount != 0) {
            const u32 last = words[amount - 1];
            if (tracked) {
                s.shadow[method] = last;
            }
            Store3D(s, table, method, last);
        }
        s.draw_pending = true;
    } else {
        // Each word as a CallMethod; a word may change the shadow RAM mode for the next.
        for (u32 i = 0; i < amount; ++i) {
            const u32 argument = words[i];
            const u32 value = ApplyShadow3D(s.control, s.shadow.data(), method, argument);
            Store3D(s, table, method, value);
            if ((flags & DrawBegin) != 0) {
                const u32 instance = (value >> 26) & 3;
                if (instance == 0) {
                    s.instance_loop = false;
                } else if (instance == 1) {
                    s.instance_loop = true;
                }
            } else if ((flags & InlineIndex) != 0) {
                s.instance_loop = false;
            }
            if ((flags & Exec) == 0) {
                continue;
            }
            if ((flags & ShadowControl) != 0) {
                s.control = static_cast<ShadowRamControl>(argument);
                s.shadow[method] = argument;
            } else if ((flags & MacroCodePointer) != 0) {
                effects(Effect::ClearCode, value, 0);
            } else if ((flags & MacroCode) != 0) {
                effects(Effect::AddCode, s.regs[table.instruction_ptr], value);
            } else if ((flags & MacroBind) != 0) {
                const u32 pointer = s.regs[table.start_address_ptr];
                effects(Effect::Bind, pointer, value);
                Store3D(s, table, table.start_address_ptr, pointer + 1);
            } else if ((flags & Falcon4) != 0) {
                Store3D(s, table, table.shadow_scratch, 1);
            } else if ((flags & DrawManager) != 0) {
                s.draw_pending = true;
            }
            if ((flags & CbufBindFlag) != 0) {
                s.v2_groups.Touch(RegGroup::CbufBind);
                NoteCbufBind3D(s, method);
            }
        }
    }
    s.sink_pending = false; // The back consumes its sink before a multi-word method.
}

template <typename State>
void CopyMirror3D(Mirror3D& mirror, const State& s) {
    std::memcpy(mirror.regs.data(), s.regs.data(), sizeof(mirror.regs));
    std::memcpy(mirror.shadow.data(), s.shadow.data(), sizeof(mirror.shadow));
    mirror.control = s.control;
    mirror.instance_loop = s.instance_loop;
    mirror.sink_pending = s.sink_pending;
    mirror.draw_pending = s.draw_pending;
    mirror.skip_same = s.skip_same;
    mirror.v2_groups = GroupTracker{};
    Rehash3D(mirror.v2_groups, mirror.regs.data());
    if constexpr (requires { s.v2_cbufs; }) {
        mirror.v2_cbufs = s.v2_cbufs;
    }
}

inline bool ReportMismatch() {
    static std::atomic<u32> reports{};
    return reports.fetch_add(1, std::memory_order_relaxed) < 64;
}

/// Verify: after the hand-written mirror `s` and the table-driven `mirror` both took the same
/// write, compares what it can touch. Logs the first mismatches and resynchronizes the copy.
template <typename State>
void CompareWrite3D(Front3D& front, const State& s, const Table3D& table, u32 method,
                    u32 argument, bool sent, bool table_sent) {
    Mirror3D& m = *front.mirror;
    const u32 touched[] = {method, table.const_buffer_offset, table.start_address_ptr,
                           table.shadow_scratch};
    bool same = sent == table_sent && m.control == s.control &&
                m.instance_loop == s.instance_loop && m.draw_pending == s.draw_pending &&
                m.shadow[method] == s.shadow[method] && (!sent || m.sink_pending == s.sink_pending);
    for (const u32 index : touched) {
        same = same && m.regs[index] == s.regs[index];
    }
    if (same) [[likely]] {
        // Every 4096 writes compare everything, and check the group hashes.
        if ((++front.writes & 4095) == 0) {
            for (u32 i = 0; i < Table3D::NumRegs; ++i) {
                if ((m.regs[i] != s.regs[i] || m.shadow[i] != s.shadow[i]) && ReportMismatch()) {
                    LOG_WARNING(HW_GPU,
                                "V2 regsem 3D mirror: register {:#x} table {:#x} hand {:#x}, "
                                "shadow table {:#x} hand {:#x}",
                                i, m.regs[i], s.regs[i], m.shadow[i], s.shadow[i]);
                    same = false;
                }
            }
            GroupTracker check = m.v2_groups;
            if (const u32 drifted = Rehash3D(check, m.regs.data()); drifted != 0 &&
                                                                     ReportMismatch()) {
                LOG_WARNING(HW_GPU, "V2 regsem 3D mirror: group hashes drifted, groups {:#x}",
                            drifted);
            }
            if (same) {
                return;
            }
        } else {
            return;
        }
    } else if (ReportMismatch()) {
        LOG_WARNING(HW_GPU,
                    "V2 regsem 3D write {:#x} = {:#x}: sent table {} hand {}; register table {:#x} "
                    "hand {:#x}; shadow table {:#x} hand {:#x}; control {} / {}; instance loop "
                    "{} / {}; draw pending {} / {}; sink pending {} / {}; offset {:#x} / {:#x}; "
                    "macro bind {:#x} / {:#x}",
                    method, argument, table_sent, sent, m.regs[method], s.regs[method],
                    m.shadow[method], s.shadow[method], static_cast<u32>(m.control),
                    static_cast<u32>(s.control), m.instance_loop, s.instance_loop, m.draw_pending,
                    s.draw_pending, m.sink_pending, s.sink_pending,
                    m.regs[table.const_buffer_offset], s.regs[table.const_buffer_offset],
                    m.regs[table.start_address_ptr], s.regs[table.start_address_ptr]);
    }
    ++front.mismatches;
    CopyMirror3D(m, s);
}

// The executing Maxwell3D (EDEN_NX2_V2_REGSEM=verify): what the table-driven update function
// does to the register file, without the side effects, next to the engine's own path.

/// One method write as Maxwell3D::TableWrite applies it to the registers. Returns the action.
inline Action3D DryWrite3D(BackMirror3D& m, const Table3D& table, u32 method, u32 argument,
                           bool skip_same) {
    const auto control = static_cast<ShadowRamControl>(m.shadow[table.shadow_control]);
    const u32 value = ApplyShadow3D(control, m.shadow.data(), method, argument);
    const Action3D action = table.actions[method];
    if (action == Action3D::Store) {
        if (!skip_same || m.regs[method] != value) {
            m.regs[method] = value;
        }
        return action;
    }
    m.regs[method] = value;
    switch (action) {
    case Action3D::CbData:
        m.regs[table.const_buffer_offset] += static_cast<u32>(sizeof(u32));
        break;
    case Action3D::ShadowControl:
        m.shadow[table.shadow_control] = argument;
        break;
    case Action3D::MacroBind:
        ++m.regs[table.start_address_ptr];
        break;
    case Action3D::Falcon4:
        m.regs[table.shadow_scratch] = 1;
        break;
    default:
        break;
    }
    return action;
}

/// A multi-method run outside macros (Maxwell3D::CallMultiMethod) on the registers.
inline void DryWriteMulti3D(BackMirror3D& m, const Table3D& table, u32 method, const u32* words,
                            u32 amount, bool skip_same) {
    const u16 flags = table.flags[method];
    const auto control = static_cast<ShadowRamControl>(m.shadow[table.shadow_control]);
    if ((flags & CbData) != 0) {
        m.regs[table.const_buffer_offset] += amount * static_cast<u32>(sizeof(u32));
    } else if ((flags & InlineData) != 0) {
        // Upload data only.
    } else if ((flags & InlineIndex) != 0 && control != ShadowRamControl::Replay) {
        if (amount != 0) {
            m.regs[method] = ApplyShadow3D(control, m.shadow.data(), method, words[amount - 1]);
        }
    } else {
        for (u32 i = 0; i < amount; ++i) {
            DryWrite3D(m, table, method, words[i], skip_same);
        }
    }
}

/// Starts the mirror over from the engine's registers.
inline void ResyncBackMirror3D(BackMirror3D& m, const u32* regs, const u32* shadow) {
    std::memcpy(m.regs.data(), regs, sizeof(m.regs));
    std::memcpy(m.shadow.data(), shadow, sizeof(m.shadow));
}

/// After the engine's own path ran a write the mirror also took: compares the registers the
/// write path touches (all of them every 4096 writes), logs the first mismatches ("V2 regsem
/// back write ...") and resynchronizes. Returns whether they matched.
inline bool CompareBackWrite3D(BackMirror3D& m, const Table3D& table, u32 method, u32 argument,
                               const u32* regs, const u32* shadow) {
    const u32 touched[] = {method, table.const_buffer_offset, table.start_address_ptr,
                           table.shadow_scratch};
    bool same = m.shadow[method] == shadow[method] &&
                m.shadow[table.shadow_control] == shadow[table.shadow_control];
    for (const u32 index : touched) {
        same = same && m.regs[index] == regs[index];
    }
    if (same && (++m.writes & 4095) != 0) [[likely]] {
        return true;
    }
    if (same) {
        for (u32 i = 0; i < Table3D::NumRegs; ++i) {
            if (m.regs[i] != regs[i] || m.shadow[i] != shadow[i]) {
                if (ReportMismatch()) {
                    LOG_WARNING(HW_GPU,
                                "V2 regsem back registers: {:#x} table {:#x} engine {:#x}, "
                                "shadow table {:#x} engine {:#x}",
                                i, m.regs[i], regs[i], m.shadow[i], shadow[i]);
                }
                same = false;
            }
        }
        if (same) {
            return true;
        }
    } else if (ReportMismatch()) {
        LOG_WARNING(HW_GPU,
                    "V2 regsem back write {:#x} = {:#x} ({}): register table {:#x} engine {:#x}, "
                    "shadow table {:#x} engine {:#x}, mode table {:#x} engine {:#x}, offset "
                    "{:#x} / {:#x}, macro bind {:#x} / {:#x}",
                    method, argument, static_cast<u32>(table.actions[method]), m.regs[method],
                    regs[method], m.shadow[method], shadow[method],
                    m.shadow[table.shadow_control], shadow[table.shadow_control],
                    m.regs[table.const_buffer_offset], regs[table.const_buffer_offset],
                    m.regs[table.start_address_ptr], regs[table.start_address_ptr]);
    }
    ++m.mismatches;
    ResyncBackMirror3D(m, regs, shadow);
    return false;
}

// Native compute.

struct MirrorCompute {
    std::array<u32, 0x1000> registers{};
    bool compute_uploaded{};
    u32 bind_pointer{};
    ComputeGroupTracker v2_groups;
};

/// FrontApply: a native compute register write (all of them go to the back).
template <typename State, typename Effects>
void ApplyCompute(State& s, u32 method, u32 argument, Effects&& effects) {
    const u8 flags = TableCompute()[method];
    if ((flags & ComputeUploaded) != 0) {
        s.compute_uploaded = true;
    }
    NoteWriteCompute(s.v2_groups, method, s.registers[method], argument);
    s.registers[method] = argument;
    if ((flags & ComputeClearCode) != 0) {
        effects(Effect::ClearCode, argument, 0);
    } else if ((flags & ComputeAddCode) != 0) {
        effects(Effect::AddCode, 0, argument);
    } else if ((flags & ComputeBindPointer) != 0) {
        s.bind_pointer = argument;
    } else if ((flags & ComputeBind) != 0) {
        if (effects(Effect::Bind, s.bind_pointer, argument)) {
            ++s.bind_pointer;
        }
    }
}

} // namespace VideoCommon::V2::RegSem
