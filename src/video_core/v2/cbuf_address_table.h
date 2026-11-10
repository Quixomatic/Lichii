// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

#include "common/common_types.h"

namespace VideoCommon::V2 {

/// EDEN_NX2_V2_CBUFBDA (shader_recompiler/backend/spirv/cbuf_bda.h): the constant buffer address
/// tables of a pipeline whose addresses don't fit the push constants. Chunks of the stream buffer
/// per tick; a draw's table equal to the last one is not written again (its address is pushed
/// again).
struct CbufAddressTable {
    static constexpr size_t ChunkBytes = 4096;

    u64 base{};
    u8* host{};
    size_t size{};
    size_t used{};
    u64 tick{~u64{}};
    u64 last_address{};
    std::vector<u64> last;

    /// Puts `addresses` in the table (`allocate(bytes)` returns a StagingBufferRef-like
    /// allocation living for the tick: device_address, offset, mapped_span) and returns their
    /// device address; `reused` tells whether the last table was equal.
    template <typename Allocate>
    u64 Append(std::span<const u64> addresses, u64 current_tick, bool& reused,
               Allocate&& allocate) {
        const size_t bytes = addresses.size_bytes();
        reused = tick == current_tick && last.size() == addresses.size() &&
                 std::equal(addresses.begin(), addresses.end(), last.begin());
        if (reused) {
            return last_address;
        }
        if (tick != current_tick || used + bytes > size) {
            const size_t chunk = (std::max)(ChunkBytes, bytes);
            const auto ref = allocate(chunk);
            base = ref.device_address + ref.offset;
            host = ref.mapped_span.data();
            size = chunk;
            used = 0;
            tick = current_tick;
        }
        std::memcpy(host + used, addresses.data(), bytes);
        last_address = base + used;
        used += bytes;
        last.assign(addresses.begin(), addresses.end());
        return last_address;
    }
};

} // namespace VideoCommon::V2
