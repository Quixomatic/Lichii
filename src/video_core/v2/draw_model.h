// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>

#include "common/common_types.h"
#include "video_core/v2/ir.h"

// Pipeline v2 P1b: the split front's model of the back's draw manager (EDEN_NX2_V2_IR_DRAWS).
// It sees the same method stream the back's Maxwell3D::DrawManager sees, in the same order, and
// predicts the draws it performs (Draw IR records). The rules are the draw manager's, method
// for method (engines/draw_manager.cpp, Maxwell3D::ProcessDirtyRegisters/CallMultiMethod):
//   - a register write that reaches ProcessDirtyRegisters ends a deferred instanced batch
//     unless it is an instance loop method (OnStore, before the register takes the value);
//   - executable methods drive the draw state machine (OnExecute, after it took the value);
//   - inline index runs append indices (OnInlineIndex);
//   - a front macro's end issues the deferred batch (OnMacroEnd), a macro HLE entry one
//     indirect draw (OnIndirect).
// Registers are read through a Maxwell3D register array (the front's mirror).
namespace VideoCommon::V2 {

/// Draws one step produced, in order.
struct DrawOutput {
    std::array<Ir::PredictedDraw, 8> draws{};
    /// S1::DrawVariant bits per draw: how the draw manager reached it (S1 Draw entries).
    std::array<u8, 8> variants{};
    u32 count{};

    void Push(const Ir::PredictedDraw& draw, u8 variant = 0) {
        if (count < draws.size()) {
            variants[count] = variant;
            draws[count++] = draw;
        }
    }
};

/// The draw manager's state the model tracks (DrawManager::State, the parts draws depend on).
struct DrawModelState {
    u8 mode{};      ///< DrawManager::DrawMode: 0 general, 1 instance, 2 inline index
    u8 topology{};  ///< draw_state.topology
    bool draw_indexed{};
    u32 instance_count{};
    u32 drawn_instances{};
    u32 base_index{};
    u32 base_instance{};
    u32 inline_bytes{}; ///< Bytes of inline indices collected
    u32 vertex_first{}; ///< draw_state.vertex_buffer.first/count
    u32 vertex_count{};
    u32 index_first{}; ///< draw_state.index_buffer.first/count
    u32 index_count{};
};

class DrawModel {
public:
    DrawModelState state;

    /// Methods an instance loop writes between its draws (DrawManager::IsInstanceLoopMethod).
    [[nodiscard]] static bool IsInstanceLoopMethod(u32 method);
    /// Methods the model acts on when executable (draw triggers and draw state).
    [[nodiscard]] static bool IsDrawMethod(u32 method);

    [[nodiscard]] bool HasDeferredDraw() const {
        return state.mode == 1 && state.instance_count != 0;
    }

    /// A write that reaches the back's ProcessDirtyRegisters, `regs` still before it.
    void OnStore(u32 method, const u32* regs, DrawOutput& out);
    /// An executable method the draw manager handles, `regs` holding its value.
    void OnExecute(u32 method, u32 argument, const u32* regs, DrawOutput& out);
    /// A multi-method run of inline indices (CallMultiMethod outside shadow RAM replay).
    void OnInlineIndex(u32 method, u32 amount);
    /// CallMultiMethod's own check before a run of a method that is not an instance loop method.
    void OnMultiStart(u32 method, const u32* regs, DrawOutput& out);
    /// FinishFrontMacro: the end of a front macro (DrawDeferred).
    void OnMacroEnd(const u32* regs, DrawOutput& out);
    /// A host indirect draw (Maxwell3D::RunMacroHleDraw).
    void OnIndirect(bool indexed, u32 topology, const u32* regs, DrawOutput& out);

private:
    void DrawDeferred(bool loop_continues, const u32* regs, DrawOutput& out);
    void DrawEnd(u32 instance_count, bool force_draw, const u32* regs, DrawOutput& out);
    void ProcessDraw(bool indexed, u32 instance_count, const u32* regs, DrawOutput& out,
                     u8 variant = 0);
    void UpdateTopology(const u32* regs);
};

} // namespace VideoCommon::V2
