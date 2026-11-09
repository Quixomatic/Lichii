// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <optional>

#include "common/common_types.h"

namespace Shader {
struct Info;
}

// Pipeline v2 P2: what a shader program writes to memory, for the split front's write ranges
// (v2/hazard.h). The back records it when it binds a pipeline, keyed by the program's device
// address (the shader cache's key); the front looks it up when it sends a draw or a dispatch:
// - a program that stores to no memory adds nothing;
// - one whose only stores go to storage buffers whose descriptors sit in constant buffers can be
//   bounded by reading those descriptors (compute, where the front holds the QMD);
// - anything else (images, global memory, indirect descriptors) is unbounded.
// A program never bound yet is unknown, which the front treats as unbounded. The table is lock
// free: one writer (the back), seqlocked slots; invalidations (the shader cache dropping a
// program) clear the key.
namespace VideoCommon::V2::ShaderWrites {

struct StorageWrite {
    u32 cbuf_index;
    u32 cbuf_offset;
    u32 count;
};

struct Summary {
    static constexpr size_t MaxStorage = 8;
    bool unbounded = false; ///< Stores the front can't bound
    u32 num_storage = 0;
    std::array<StorageWrite, MaxStorage> storage{};

    [[nodiscard]] bool WritesNothing() const {
        return !unbounded && num_storage == 0;
    }
};

/// Whether summaries are recorded (EDEN_NX2_V2_HAZARD or ACQUIRE on).
[[nodiscard]] bool Enabled();

/// The back bound a pipeline with this program at `device_address`.
void Record(u64 device_address, const Shader::Info& info);

/// The front: the program's summary, if the back bound it since it was last invalidated.
[[nodiscard]] std::optional<Summary> Lookup(u64 device_address);

/// The shader cache dropped programs in [begin, end).
void Invalidate(u64 begin, u64 end);

} // namespace VideoCommon::V2::ShaderWrites
