// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstring>
#include <memory>

#include "common/logging.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/v2/regsem.h"
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
};

/// Per front Maxwell3D: the mode it started with and, for verify, the table-driven copy.
struct Front3D {
    Mode mode{};
    std::unique_ptr<Mirror3D> mirror; ///< Verify only
    u64 writes{};
    u64 mismatches{};
};

template <typename State>
void Store3D(State& s, const Table3D& table, u32 index, u32 value) {
    const u32 old_value = s.regs[index];
    if (old_value != value) {
        s.regs[index] = value;
        s.v2_groups.Change(table.groups[index], index, old_value, value);
    }
}

/// One method write (FrontWrite): applies the shadow RAM, drops writes that change nothing
/// (compare-on-write: no dirty bit, no ring entry), updates the registers, the groups and the
/// back-side effects the mirror must reproduce. Returns whether the write goes to the back.
template <typename State, typename Effects>
bool Write3D(State& s, const Table3D& table, u32 method, u32 argument, bool from_macro,
             Effects&& effects) {
    u32 value = argument;
    bool shadow_kept = true;
    if (s.control == ShadowRamControl::Track || s.control == ShadowRamControl::TrackWithFilter) {
        shadow_kept = s.shadow[method] == argument;
        s.shadow[method] = argument;
    } else if (s.control == ShadowRamControl::Replay) {
        value = s.shadow[method];
    }
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
            u32 value = argument;
            if (s.control == ShadowRamControl::Track ||
                s.control == ShadowRamControl::TrackWithFilter) {
                s.shadow[method] = argument;
            } else if (s.control == ShadowRamControl::Replay) {
                value = s.shadow[method];
            }
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

// Native compute.

struct MirrorCompute {
    std::array<u32, 0x1000> registers{};
    bool compute_uploaded{};
    u32 bind_pointer{};
};

/// FrontApply: a native compute register write (all of them go to the back).
template <typename State, typename Effects>
void ApplyCompute(State& s, u32 method, u32 argument, Effects&& effects) {
    const u8 flags = TableCompute()[method];
    if ((flags & ComputeUploaded) != 0) {
        s.compute_uploaded = true;
    }
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
