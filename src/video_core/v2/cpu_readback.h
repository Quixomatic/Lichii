// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <functional>
#include <utility>

// SYNC_SUBMIT_SPEC §12 R4 (MASTER_PLAN §6 "guest CPU reads of GPU-written memory are served on
// the reading thread"): with EDEN_NX2_V2_READBACK the caches' owner records the downloads a
// guest CPU read needs into the open submission and hands the reader a `land` step; the reader
// asks for the submission (rule W, Scheduler::EnsureSubmitted), waits for its tick and writes
// guest memory itself. The owner never waits for the GPU for it (RasterizerVulkan::
// FlushRegionForRead).
namespace VideoCommon::V2::CpuReadback {

/// Whether the images of a read may land on the reader. Guest memory must get FlushRegion's
/// order (images, then buffers): the images go to the reader only when the buffers do too, or
/// when no buffer data is owed (`buffer_data` false). Verify keeps the synchronous image path.
[[nodiscard]] constexpr bool ImagesOnReader(bool verify, bool buffer_data,
                                            bool buffers_on_reader) noexcept {
    return !verify && (buffers_on_reader || !buffer_data);
}

/// The reader's step: the images' land, then the buffers' (either may be empty).
[[nodiscard]] inline std::function<void()> Chain(std::function<void()> images,
                                                 std::function<void()> buffers) {
    if (!images) {
        return buffers;
    }
    return [images = std::move(images), buffers = std::move(buffers)] {
        images();
        if (buffers) {
            buffers();
        }
    };
}

} // namespace VideoCommon::V2::CpuReadback
