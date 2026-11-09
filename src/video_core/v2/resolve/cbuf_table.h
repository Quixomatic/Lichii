// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cstddef>

#include "common/common_types.h"

// Pipeline v2 P6c: the Maxwell3D constant buffer bindings (state.shader_stages[].const_buffers)
// as plain data, so register mirrors (the split front, the P6 shards) can keep them next to their
// registers. A binding is made by a write to bind_groups[stage].raw_config from the const_buffer
// registers at that moment (Maxwell3D::ProcessCBBind); nothing else changes them.
namespace VideoCommon::V2::Resolve {

inline constexpr size_t CbufStages = 5;
inline constexpr size_t CbufSlots = 18; ///< Maxwell3D::Regs::MaxConstBuffers

struct CbufBinding {
    u64 address = 0;
    u32 size = 0;
    u32 enabled = 0;

    bool operator==(const CbufBinding&) const = default;
};

struct CbufTable {
    std::array<std::array<CbufBinding, CbufSlots>, CbufStages> stages{};

    bool operator==(const CbufTable&) const = default;
};

} // namespace VideoCommon::V2::Resolve
