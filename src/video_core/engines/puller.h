// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <vector>
#include "common/bit_field.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "video_core/engines/engine_interface.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
class DmaPusher;

enum class EngineID {
    NV01_TIMER = 0x0004,
    FERMI_TWOD_A = 0x902D, // 2D Engine
    MAXWELL_B = 0xB197,    // 3D Engine
    KEPLER_COMPUTE_B = 0xB1C0,
    KEPLER_INLINE_TO_MEMORY_B = 0xA140,
    MAXWELL_DMA_COPY_A = 0xB0B5,
};

namespace Control {
struct ChannelState;
}
} // namespace Tegra

namespace VideoCore {
class RasterizerInterface;
}

namespace VideoCommon::V2 {
struct WordReadback;
}

namespace Tegra::Engines {

class Puller final {
public:
    struct MethodCall {
        u32 method{};
        u32 argument{};
        u32 subchannel{};
        u32 method_count{};

        explicit MethodCall(u32 method_, u32 argument_, u32 subchannel_ = 0, u32 method_count_ = 0)
            : method(method_), argument(argument_), subchannel(subchannel_),
              method_count(method_count_) {}

        [[nodiscard]] bool IsLastCall() const {
            return method_count <= 1;
        }
    };

    enum class FenceOperation : u32 {
        Acquire = 0,
        Increment = 1,
    };

    union FenceAction {
        u32 raw;
        BitField<0, 1, FenceOperation> op;
        BitField<8, 24, u32> syncpoint_id;
    };

    /// Switch 2: a semaphore acquire this channel is suspended on.
    struct PendingAcquire {
        GPUVAddr address;
        u32 value;
        u32 operation;
        std::chrono::steady_clock::time_point since;
        std::chrono::steady_clock::time_point last_fences; ///< When it last released fences
        /// EDEN_NX2_V2_ACQUIRE: when an unmet poll last asked the back for a submit
        std::chrono::steady_clock::time_point last_request;
        bool reported;
        /// EDEN_NX2_V2_ACQUIRE: the comparison (V2::SemaphoreOp), and whether it is an Ampere
        /// SEM_EXECUTE acquire suspended like a trigger acquire (its operation is not one of
        /// GpuSemaphoreOperation's).
        u8 view_op{};
        bool sem_execute{};
        /// A GPU-written word's readback (V2::WordReadback) was sent and not landed yet.
        bool readback_sent{};
        /// SYNC_SUBMIT_SPEC §2.2 (V2::PlanSuspendedPoll): the fence tick it asked to be
        /// submitted last (one request per tick change), and the back position + 1 at which it
        /// had a fence signalled for an uncarried capture (0: never).
        u64 requested_tick{};
        u64 fence_job_at{};
        /// SYNC Step 0: what its last poll waited for (V2::Stats::WakeClass), for the STATS line.
        u8 wake{};
        /// SYNC_SUBMIT_SPEC §2.2: the wake classes that can change its answer (V2::SemWake; 0:
        /// check every pass) and their epochs at the last check.
        u8 wake_bits{};
        u64 seen_notes{};
        u64 seen_consumed{};
        u64 seen_landings{};
        u64 seen_maps{};

        [[nodiscard]] bool Matches(u32 current) const;
    };

    [[nodiscard]] bool HasPendingAcquire() const {
        return pending_acquire.has_value();
    }
    [[nodiscard]] bool CanResumeAcquire(DmaPusher& dma_pusher);
    /// Clears the pending acquire once it is met; false while the channel has to keep waiting.
    bool ResumeAcquire(DmaPusher& dma_pusher);

    void CallMethod(DmaPusher& dma_pusher, const MethodCall& method_call);
    void CallMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending);
    void BindRasterizer(DmaPusher& dma_pusher, VideoCore::RasterizerInterface* rasterizer);
    void CallPullerMethod(DmaPusher& dma_pusher, const MethodCall& method_call);
    void CallEngineMethod(DmaPusher& dma_pusher, const MethodCall& method_call);
    void CallEngineMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending);
private:
    /// CanResumeAcquire's check (CanResumeAcquire also counts it for EDEN_NX2_V2_STATS).
    [[nodiscard]] bool PollAcquire(DmaPusher& dma_pusher);
    /// EDEN_NX2_V2_ACQUIRE: PollAcquire decided by the front's semaphore view.
    [[nodiscard]] bool PollAcquireView(DmaPusher& dma_pusher);
    /// EDEN_NX2_V2_ACQUIRE=1: suspends the channel on an acquire the view doesn't pass, without
    /// draining (MASTER_PLAN §6). `gpu_written`: its word is read back (V2::WordReadback).
    void SuspendOnView(DmaPusher& dma_pusher, PendingAcquire acquire, bool gpu_written);
    /// Sends the readback of the pending acquire's GPU-written word (no wait).
    void SendWordReadback(DmaPusher& dma_pusher);

    static constexpr std::size_t NUM_REGS = 0x800;
    struct Regs {
        static constexpr size_t NUM_REGS = 0x40;

        union {
            struct {
                INSERT_PADDING_WORDS_NOINIT(0x4);
                struct {
                    u32 address_high;
                    u32 address_low;

                    [[nodiscard]] GPUVAddr SemaphoreAddress() const {
                        return static_cast<GPUVAddr>((static_cast<GPUVAddr>(address_high) << 32) |
                                                     address_low);
                    }
                } semaphore_address;

                u32 semaphore_sequence;
                u32 semaphore_trigger;
                INSERT_PADDING_WORDS_NOINIT(0xC);

                // The pusher and the puller share the reference counter, the pusher only has read
                // access
                u32 reference_count;
                INSERT_PADDING_WORDS_NOINIT(0x5);

                u32 semaphore_acquire;
                u32 semaphore_release;
                u32 fence_value;
                FenceAction fence_action;
                INSERT_PADDING_WORDS_NOINIT(0xE2);

                // Puller state
                u32 acquire_mode;
                u32 acquire_source;
                u32 acquire_active;
                u32 acquire_timeout;
                u32 acquire_value;
            };
            std::array<u32, NUM_REGS> reg_array;
        };
    } regs{};

    void ProcessBindMethod(DmaPusher& dma_pusher, const MethodCall& method_call);
    void ProcessFenceActionMethod(DmaPusher& dma_pusher);
    void ProcessSemaphoreAcquire(DmaPusher& dma_pusher);
    void ProcessSemaphoreRelease(DmaPusher& dma_pusher);
    void ProcessSemaphoreTriggerMethod(DmaPusher& dma_pusher);
    void ProcessSemaphoreExecute(DmaPusher& dma_pusher, u32 argument);
    [[nodiscard]] bool ExecuteMethodOnEngine(DmaPusher& dma_pusher, u32 method);

    std::optional<PendingAcquire> pending_acquire;
    /// EDEN_NX2_V2_ACQUIRE: the slot a GPU-written semaphore word is read back through.
    std::shared_ptr<VideoCommon::V2::WordReadback> word_readback;

    /// Mapping of command subchannels to their bound engine ids
    std::array<EngineID, 8> bound_engines{};

    enum class GpuSemaphoreOperation {
        AcquireEqual = 0x1,
        WriteLong = 0x2,
        AcquireGequal = 0x4,
        AcquireMask = 0x8,
    };

#define ASSERT_REG_POSITION(field_name, position)                                                  \
    static_assert(offsetof(Regs, field_name) == position * 4, "Field " #field_name " has invalid position")

    ASSERT_REG_POSITION(semaphore_address, 0x4);
    ASSERT_REG_POSITION(semaphore_sequence, 0x6);
    ASSERT_REG_POSITION(semaphore_trigger, 0x7);
    ASSERT_REG_POSITION(reference_count, 0x14);
    ASSERT_REG_POSITION(semaphore_acquire, 0x1A);
    ASSERT_REG_POSITION(semaphore_release, 0x1B);
    ASSERT_REG_POSITION(fence_value, 0x1C);
    ASSERT_REG_POSITION(fence_action, 0x1D);

    ASSERT_REG_POSITION(acquire_mode, 0x100);
    ASSERT_REG_POSITION(acquire_source, 0x101);
    ASSERT_REG_POSITION(acquire_active, 0x102);
    ASSERT_REG_POSITION(acquire_timeout, 0x103);
    ASSERT_REG_POSITION(acquire_value, 0x104);

#undef ASSERT_REG_POSITION
};

} // namespace Tegra::Engines
