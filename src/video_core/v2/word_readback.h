// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <functional>

#include "common/common_types.h"
#include "video_core/cache_types.h"
#include "video_core/rasterizer_interface.h"

// Pipeline v2 (MASTER_PLAN §6 class C, Fix F3-06/F3-08): the split front reads a few words of
// GPU-written guest memory without draining the back. The front sends a Readback record
// (Ir::ReadbackOp::WordReadback) naming this slot; the back runs it in order, after the writers
// sent before it, and records the download through the readback path (FlushRegionForRead: a
// submitted copy the reader waits for, not a synchronous download on the back). The front lands
// it once the back consumed the record: it waits for that download's tick only.
namespace VideoCommon::V2 {

struct WordReadback {
    /// Back: records the download of [address, address + size) if the buffer cache holds GPU
    /// data there, then publishes the slot.
    void Run(VideoCore::RasterizerInterface& rasterizer, DAddr address, u64 size) {
        land = {};
        if (rasterizer.MustFlushRegion(address, size, VideoCommon::CacheType::BufferCache)) {
            rasterizer.FlushRegionForRead(address, size, land);
        }
        ready.store(true, std::memory_order_release);
    }

    /// Front: whether the back ran the record.
    [[nodiscard]] bool Ready() const {
        return ready.load(std::memory_order_acquire);
    }

    /// Front, once Ready: waits for the download (if one was submitted) and writes it to guest
    /// memory. The slot can be requested again afterwards.
    void Land() {
        if (land) {
            land();
            land = {};
        }
        ready.store(false, std::memory_order_relaxed);
    }

    std::function<void()> land;
    std::atomic<bool> ready{false};
};

} // namespace VideoCommon::V2
