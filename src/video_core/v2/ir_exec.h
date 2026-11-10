// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "common/common_types.h"
#include "video_core/v2/ir.h"
#include "video_core/v2/ir_arena.h"

namespace Core {
class System;
}
namespace Tegra {
class MemoryManager;
}
namespace VideoCore {
class RasterizerInterface;
}

// The back's executor for typed IR records (EDEN_NX2_V2_IR): what used to be deferred closures
// is decoded from the record's body and run here, with its objects read from the side table.
// The helpers below are what both the executor and the word stream's deferred calls run.
namespace VideoCommon::V2::IrExec {

struct Context {
    Core::System& system;
    const Ir::ObjectTable& objects;
    const Ir::PayloadArena& arena;
};

/// Runs a typed record that has no thunk. Returns false for kinds it doesn't know.
bool Execute(const Ir::RecordView& record, const Context& context);

/// Whether the back runs anything for a record of this kind and body (informational records
/// only feed checks and don't end fast draw reuse).
[[nodiscard]] bool IsInformational(const Ir::RecordView& record);

/// ReadbackOp::MacroDmaFlush: the buffer cache's newer copy of the words, then the reports
/// held in the query cache, reach guest memory (Maxwell3D's front macro DMA reads).
void MacroDmaFlush(Tegra::MemoryManager& memory, VideoCore::RasterizerInterface& rasterizer,
                   u64 address, u64 bytes);

/// Back (a front job between entries): when operations this thread gave landing tickets are
/// still to land, signals a fence behind them (never skipped as empty) and submits it; returns
/// the newest such ticket for the front to wait for (GpuSplit::CompleteLanding), else 0.
[[nodiscard]] u64 LandCaptured(VideoCore::RasterizerInterface& rasterizer);

} // namespace VideoCommon::V2::IrExec
