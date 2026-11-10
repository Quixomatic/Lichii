// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: 2022 yuzu Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <atomic>
#include <cstdlib>
#include <optional>
#include <string>
#include <thread>

#include "common/assert.h"
#include "common/logging.h"
#include "common/settings.h"
#include "core/core.h"
#include "video_core/v2/resolve/multifront.h"
#include "video_core/nx2_profile.h"
#include "video_core/control/channel_state.h"
#include "video_core/dma_pusher.h"
#include "video_core/engines/fermi_2d.h"
#include "video_core/engines/kepler_compute.h"
#include "video_core/engines/kepler_memory.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/engines/maxwell_dma.h"
#include "video_core/engines/puller.h"
#include "video_core/gpu.h"
#include "video_core/gpu_split.h"
#include "video_core/memory_manager.h"
#include "video_core/nx2.h"
#include "video_core/nx2_fast_draw.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/v2/front_writes.h"
#include "video_core/v2/hazard.h"
#include "video_core/v2/ir_exec.h"
#include "video_core/v2/landing.h"
#include "video_core/v2/landing_fence.h"
#include "video_core/v2/v2_stats.h"
#include "video_core/v2/versions.h"
#include "video_core/v2/word_readback.h"

namespace Tegra::Engines {

namespace {

constexpr u32 Gf100BindClassMask = 0xffff;
constexpr u32 Gf100BindValidMask =  0x1f0000 | Gf100BindClassMask;

// Reads a semaphore word, flushing a GPU-written copy out of the buffer cache first.
u32 ReadSemaphoreWord(DmaPusher& dma_pusher, GPUVAddr address) {
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
        device && dma_pusher.rasterizer->MustFlushRegion(*device, sizeof(u32),
                                                          VideoCommon::CacheType::BufferCache)) {
        auto& split = VideoCommon::GpuSplit::Get();
        const VideoCommon::V2::Ir::ReadbackBody body{
            .op = VideoCommon::V2::Ir::ReadbackOp::FlushCache,
            .object = split.Ref(dma_pusher.rasterizer),
            .cache = static_cast<u32>(VideoCommon::CacheType::BufferCache),
            .address = *device,
            .bytes = sizeof(u32),
        };
        split.RunTyped(
            VideoCommon::V2::Ir::Kind::Readback, body,
            [rasterizer = dma_pusher.rasterizer, device = *device] {
                rasterizer->FlushRegion(device, sizeof(u32), VideoCommon::CacheType::BufferCache);
            },
            VideoCommon::DrainSite::SemaphoreFlush);
    }
    return dma_pusher.memory_manager.Read<u32>(address);
}

// With the GPU thread split (gpu_split.h) the rasterizer belongs to the back thread: work that
// only feeds it runs there in order; reads of what it wrote first wait for it (DrainBack). With
// the typed IR it is a Barrier record (EDEN_NX2_V2_IR).
template <typename Func>
void OnBackBarrier(DmaPusher& dma_pusher, VideoCommon::V2::Ir::BarrierOp op, const Func& func,
                   u64 a = 0) {
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        const VideoCommon::V2::Ir::BarrierBody body{
            .op = op, .object = split.Ref(dma_pusher.rasterizer), .a = a, .b = 0};
        split.Emit(VideoCommon::V2::Ir::Kind::Barrier, body, func);
    } else {
        func();
    }
}

// A semaphore release with a literal payload: a typed IR Release record. With
// EDEN_NX2_V2_ACQUIRE/HAZARD the front's semaphore view records what it writes.
template <typename Func>
void OnBackRelease(DmaPusher& dma_pusher, GPUVAddr address, u32 payload, u32 bytes,
                   const Func& func) {
    VideoCommon::V2::Resolve::MultiFront::Note(
        VideoCommon::V2::Resolve::MultiFront::EventKind::Release, address, payload);
    if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
        const VideoCommon::V2::Ir::ReleaseBody body{
            .address = address,
            .payload = payload,
            .bytes = bytes,
            .op = VideoCommon::V2::Ir::ReleaseOp::Payload,
            .address_space = static_cast<u32>(dma_pusher.memory_manager.GetID()),
            .rasterizer = split.Ref(dma_pusher.rasterizer),
        };
        split.Emit(VideoCommon::V2::Ir::Kind::Release, body, func);
        if (split.TracksWrites()) [[unlikely]] {
            // 4-byte releases are fence-flagged (IsAFence); the 16-byte WriteLong is not.
            VideoCommon::V2::NoteReleaseWrite(split, dma_pusher.memory_manager, address, payload,
                                              bytes, true, VideoCommon::V2::Producer::Release,
                                              bytes == 4);
        }
    } else {
        func();
    }
}

// EDEN_NX2_V2_IR_DRAWS: an Acquire record of what the front decided (informational: emitted only
// when the full records are read, IR_DRAWS=verify or PARALLEL).
void NoteAcquireRecord(DmaPusher& dma_pusher, GPUVAddr address, u32 value, u32 operation,
                       u32 verdict) {
    if (auto& split = VideoCommon::GpuSplit::Get(); split.FullDrawRecords()) [[unlikely]] {
        split.Note(VideoCommon::V2::Ir::Kind::Acquire,
                   VideoCommon::V2::Ir::AcquireBody{
                       .address = address,
                       .address_space = static_cast<u32>(dma_pusher.memory_manager.GetID()),
                       .value = value,
                       .op = operation,
                       .verdict = verdict,
                   });
    }
}

void DrainBack() {
    VideoCommon::GpuSplit::Get().Drain(VideoCommon::DrainSite::Acquire);
}

// The fence manager records into the scheduler and downloads the CPU's writes over GPU data:
// the back's state, which a drained back still uses from its idle hook (it submits).
void ReleaseFences(DmaPusher& dma_pusher) {
    auto& split = VideoCommon::GpuSplit::Get();
    const VideoCommon::V2::Ir::BarrierBody body{
        .op = VideoCommon::V2::Ir::BarrierOp::ReleaseFences,
        .object = split.Ref(dma_pusher.rasterizer),
        .a = 0,
        .b = 0,
    };
    split.RunTyped(VideoCommon::V2::Ir::Kind::Barrier, body,
                   [rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); },
                   VideoCommon::DrainSite::FenceRelease);
}

// F3-X1 follow-up (019d livelock): makes everything the back holds land, as a job on the back
// after its current entry (no ring record: the waiter's guard on `consumed` holds): a delayed
// fence that carries the operations queued without one (SyncOperation, as LandCaptured), the
// submit of the open command buffer, then the forced release that waits for every fence and
// runs their operations. Returns false if the back stopped before running it.
bool LandEverythingPending(VideoCore::RasterizerInterface* rasterizer) {
    struct Job {
        VideoCore::RasterizerInterface* rasterizer{};
        std::atomic<bool> done{false};
    };
    // Static: a job a stopping back never ran must not point at a dead frame. Front only.
    static Job job;
    job.rasterizer = rasterizer;
    job.done.store(false, std::memory_order_relaxed);
    auto& split = VideoCommon::GpuSplit::Get();
    split.RequestBackJob(
        [](void* argument) {
            auto& self = *static_cast<Job*>(argument);
            VideoCore::g_signaling_query_report_fence = true;
            self.rasterizer->SignalFence([] {});
            VideoCore::g_signaling_query_report_fence = false;
            self.rasterizer->FlushCommands();
            self.rasterizer->ReleaseFences(true);
            self.done.store(true, std::memory_order_release);
        },
        &job);
    while (!job.done.load(std::memory_order_acquire) && split.BackRunning()) {
        std::this_thread::yield();
    }
    return job.done.load(std::memory_order_acquire);
}

namespace V2Stats = VideoCommon::V2::Stats;

/// A PendingAcquire's operation (GpuSemaphoreOperation: 1 equal, 4 greater or equal, 8 mask).
V2Stats::AcquireOp StatsAcquireOp(u32 operation) {
    switch (operation) {
    case 0x1:
        return V2Stats::AcquireOp::Equal;
    case 0x8:
        return V2Stats::AcquireOp::Mask;
    default:
        return V2Stats::AcquireOp::Gequal;
    }
}

namespace V2 = VideoCommon::V2;
using V2Stats::Decision;
using V2Stats::Reason;

/// GpuSemaphoreOperation (1 equal, 4 greater or equal, 8 mask) as the view's comparison.
V2::SemaphoreOp ViewOp(u32 operation) {
    switch (operation) {
    case 0x1:
        return V2::SemaphoreOp::Equal;
    case 0x8:
        return V2::SemaphoreOp::Mask;
    default:
        return V2::SemaphoreOp::Gequal;
    }
}

// Whether the back writes a payload release to memory when it runs it (or at the fence the
// query cache gives a fence-flagged one): below High GPU accuracy, with fences that run their
// operations at once (FenceManager::SignalFence). As upstream's ReleasesLandAtOnce.
bool ReleasesLandAtOnce() {
    const bool delay_fence =
        Settings::IsGPUFenceBehaviorDefault()
            ? Settings::IsGPULevelHigh()
            : Settings::IsGPUFenceBehaviorBalanced() || Settings::IsGPUFenceBehaviorAccurate();
    return !delay_fence && !Settings::IsGPULevelHigh();
}

struct ViewCheck {
    V2::AcquireVerdict verdict; ///< Met, MetByQueued, NotMet or Fallback
    Reason reason;
    bool narrowed;  ///< It waited for one producer first
    u64 landing{}; ///< MetByQueued on a release still to land: its landing ticket
    u64 release{}; ///< MetByQueued: the ring position of the release that decided
    u64 qmd_target{}; ///< The QMD producer an Equal/Mask acquire waits for (GpuSplit::QmdTargetFor)
    bool by_reductions{}; ///< MetByQueued on the pending QMD reductions (QmdReductionView)
};

/// CheckWithView's `qmd_target` when the acquire is parsed: snapshot the producer now.
constexpr u64 SnapshotQmdTarget = ~u64{0};

// EDEN_NX2_V2_ACQUIRE (MASTER_PLAN.md section 6): decides an acquire from the front's semaphore
// view (the hazard table's releases) and memory, without draining the split. A targeted wait
// for one write (V2::AcquireVerdict::WaitFor) is done here, after which memory decides.
// `wait`: a decision that needs a queued dispatch to run first waits for it here. Without it
// (ACQUIRE=1) such an acquire is not met yet: the channel suspends and its polls decide once the
// back ran the dispatch (MASTER_PLAN §6: suspend, don't block the front).
// `word_landed`: a GPU-written word was just read back (V2::WordReadback): memory holds it.
// `qmd_target` (Fix F3h, F3-X1): the QMD release an Equal/Mask acquire orders behind, recorded
// when it was parsed (SnapshotQmdTarget there); its polls wait for that one producer, not for
// the newest QMD release the front noted since (a moving target only the net used to end).
ViewCheck CheckWithView(DmaPusher& dma_pusher, GPUVAddr address, u32 value, V2::SemaphoreOp op,
                        bool wait = true, bool word_landed = false,
                        u64 qmd_target = SnapshotQmdTarget) {
    auto& split = VideoCommon::GpuSplit::Get();
    // Macros on the back send 3D reports the front doesn't see.
    if (!split.Active() || !split.FrontMacros() || !ReleasesLandAtOnce()) {
        return {V2::AcquireVerdict::Fallback, Reason::Settings, false};
    }
    auto& memory = dma_pusher.memory_manager;
    const auto device = memory.GpuToCpuAddress(address);
    if (!device) {
        // Fix F3h (R2-F3-03): an unmapped word. A map the back hasn't run yet may map it: wait
        // for it (suspended; the back's `consumed` store rings the front). Otherwise nothing can
        // write it: decide on the word as the guest reads it (the old path's read, 84de5dbac4:
        // gequal 0 passes at once), without the old path's drain and forced fences.
        if (split.MapQueued()) {
            return {V2::AcquireVerdict::NotMet, Reason::Unmapped, false};
        }
        const bool met = V2::SemaphoreMet(op, memory.Read<u32>(address), value);
        return {met ? V2::AcquireVerdict::Met : V2::AcquireVerdict::NotMet, Reason::Unmapped,
                false};
    }
    if (qmd_target == SnapshotQmdTarget) {
        qmd_target =
            V2::SemaphoreWaitsToReach(op) ? 0 : split.QmdTargetFor(*device, sizeof(u32));
    }
    // A word the buffer cache may hold newer than memory is read back first (class C).
    if (!word_landed && Nx2MayBeGpuWritten(*device, sizeof(u32))) {
        return {V2::AcquireVerdict::Fallback, Reason::GpuWritten, false};
    }
    V2::HazardFilter filter;
    filter.ranged = V2::ProducerBit(V2::Producer::Release) | V2::ProducerBit(V2::Producer::Report);
    // Releases queued or run but not landed, fence-flagged or not (class B; the landing rule in
    // v2/landing.h keeps later CPU-visible writes behind them).
    u32 memory_word = memory.Read<u32>(address);
    u32 forwarded = memory_word;
    const V2::HazardResult word =
        split.FindWrites(*device, sizeof(u32), filter, reinterpret_cast<u8*>(&forwarded));
    if (word.kind == V2::HazardResult::Kind::Clear) {
        // A release that landed after the read above was retired by the lookup (it loads
        // `landed` after the operation wrote memory): read memory again, which now holds it.
        memory_word = memory.Read<u32>(address);
    }
    if (word.kind == V2::HazardResult::Kind::Clear &&
        !split.UnknownQmdMayWrite(*device, sizeof(u32))) {
        // Fix F3h (chunk C2: U-02 / R2-F3-05, §6 class A, diagram 2.3): only QMD reduction and
        // max releases the front's exact model saw write the word. Since d5e451eebf they land
        // at their fence; the word they leave is known now: memory plus the pending ones in
        // order. Met by it: the in-order back applies them before the acquire's channel goes on
        // (class B ordering, landing rule), so the channel goes on now instead of waiting for
        // the fence (67e9: ~817 such waits/s). Not met by it: decided as before.
        const auto view = split.QmdReductionView(*device, memory, address);
        if (view.known && view.any && V2::SemaphoreMet(op, view.value, value)) {
            return {V2::AcquireVerdict::MetByQueued, Reason::None, false, 0, view.position,
                    qmd_target, true};
        }
    }
    // The producer stops counting once its entry ran and its operations landed.
    const u64 pending_qmd = split.QmdTargetSettled(qmd_target) ? 0 : qmd_target;
    const V2::AcquireDecision decision =
        V2::DecideAcquire(op, value, word, forwarded, memory_word, pending_qmd);
    switch (decision.verdict) {
    case V2::AcquireVerdict::Met:
        return {decision.verdict, Reason::None, false, 0, 0, qmd_target};
    case V2::AcquireVerdict::MetByQueued:
        return {decision.verdict, Reason::None, false, word.landing ? word.position : 0,
                V2::RingPositionOf(word), qmd_target};
    case V2::AcquireVerdict::NotMet:
        return {decision.verdict,
                word.kind == V2::HazardResult::Kind::Forward ? Reason::QueuedNotMet
                                                              : Reason::NotMet,
                false, 0, 0, qmd_target};
    case V2::AcquireVerdict::WaitFor: {
        if (!wait) {
            // Suspended: its polls resume it once that producer settled (the back's `consumed`
            // and the operation's landing both ring the front's sleep).
            return {V2::AcquireVerdict::NotMet, Reason::QmdProducer, false, 0, 0, qmd_target};
        }
        split.WaitForPosition(decision.position, VideoCommon::DrainSite::Acquire);
        // The dispatch waited for may have written the word through the buffer cache.
        if (!word_landed && Nx2MayBeGpuWritten(*device, sizeof(u32))) {
            return {V2::AcquireVerdict::Fallback, Reason::GpuWritten, true, 0, 0, qmd_target};
        }
        const bool met = V2::SemaphoreMet(op, memory.Read<u32>(address), value);
        return {met ? V2::AcquireVerdict::Met : V2::AcquireVerdict::NotMet,
                met ? Reason::None : Reason::NotMet, true, 0, 0, qmd_target};
    }
    case V2::AcquireVerdict::Fallback:
        break;
    }
    const Reason reason = word.landing ? Reason::Unlanded
                          : word.overflow ? Reason::Overflow
                                          : Reason::UnknownWrite;
    return {V2::AcquireVerdict::Fallback, reason, false, 0, 0, qmd_target};
}

bool Passes(const ViewCheck& check) {
    return check.verdict == V2::AcquireVerdict::Met ||
           check.verdict == V2::AcquireVerdict::MetByQueued;
}

// EDEN_NX2_V2_ACQUIRE=1 (MASTER_PLAN §6, Fix F3-05): an acquire the view doesn't pass suspends
// the channel; no drain, no forced fence. Settings the view doesn't model (accurate GPU, delayed
// fences, macros on the back) take the old path; unmapped words suspend too (Fix F3h, R2-F3-03:
// the view reads them as the guest would, CheckWithView).
bool SuspendsOnView(const ViewCheck& check) {
    return V2::ViewSuspends(check.verdict, check.reason == Reason::Settings);
}

void NoteViewDecision(const ViewCheck& check) {
    if (!V2Stats::Enabled()) [[likely]] {
        return;
    }
    const auto site = VideoCommon::DrainSite::Acquire;
    if (check.verdict == V2::AcquireVerdict::MetByQueued) {
        // F3h chunk C2: passed by the pending QMD reductions (class A) counted apart.
        V2Stats::NoteDecision(site, Decision::Forwarded,
                              check.by_reductions ? Reason::ReductionView : Reason::None);
    } else if (Passes(check)) {
        V2Stats::NoteDecision(site, check.narrowed ? Decision::Narrowed : Decision::Avoided);
    } else {
        V2Stats::NoteDecision(site, Decision::Waited, check.reason);
    }
}

// EDEN_NX2_V2_ACQUIRE=verify: the view decided, the old path ran; logs where they disagree.
// The serious case is a pass the old path doesn't make. A poll the view keeps waiting that the
// old path resumes only costs time (the latch or the valve resumes it); it is counted too.
// `landing`: the view passed on a release still to land (class B). The drained path reads memory,
// which gets the release only when its operation runs: the comparison makes every operation the
// back gave a landing ticket land and asks memory again, so only a pass the landed releases
// don't confirm counts (Fix F3-04). F3F5b-2: not only the view's ticket (the front may number a
// write the back made at once, so `landed` can stay below it), and not only for landing
// passes: a release the front noted as made at once is deferred by the back behind an earlier
// one still to land (Landing::MustDefer), and the drain alone doesn't land it.
bool LandForVerify(DmaPusher& dma_pusher) {
    auto& split = VideoCommon::GpuSplit::Get();
    return split.CompleteLanding(
        V2::Landing::Ticket(split.WritePosition(), V2::Landing::SubMask),
        VideoCommon::DrainSite::Acquire,
        [](void* rasterizer) -> u64 {
            return VideoCommon::V2::IrExec::LandCaptured(
                *static_cast<VideoCore::RasterizerInterface*>(rasterizer));
        },
        dma_pusher.rasterizer);
}

void ReportViewMismatch(DmaPusher& dma_pusher, GPUVAddr address, u32 value, u32 operation,
                        bool view_passes, bool old_passed, const char* where, u64 landing = 0,
                        V2::SemaphoreOp op = V2::SemaphoreOp::Gequal) {
    if (view_passes == old_passed) {
        return;
    }
    const bool landed = view_passes && LandForVerify(dma_pusher);
    if (landed && V2::SemaphoreMet(op, dma_pusher.memory_manager.Read<u32>(address), value)) {
        return;
    }
    // The view waiting where the drained path goes on is benign: a release landed between the two
    // checks (the drain ran it, or its fence ran meanwhile). Counted apart, logged less.
    const bool benign = !view_passes;
    V2Stats::NoteMismatch(VideoCommon::DrainSite::Acquire, benign);
    static std::atomic<u32> logged{};
    static std::atomic<u32> logged_benign{};
    if ((benign ? logged_benign : logged).fetch_add(1) < (benign ? 4U : 32U)) {
        LOG_WARNING(HW_GPU,
                    "V2 acquire verify ({}): channel {} semaphore {:#x} op {} value {:#x}: the "
                    "view {}, the drained path {} (word now {:#x}; view ticket {:#x}, landed "
                    "{:#x}, captured operations landed {})",
                    where, dma_pusher.channel_state.bind_id, address, operation, value,
                    view_passes ? "passes" : "waits", old_passed ? "passes" : "waits",
                    dma_pusher.memory_manager.Read<u32>(address), landing,
                    V2::Landing::landed.load(std::memory_order_acquire), landed);
    }
}

// F3-X1: a release noted to the word that meets the acquire, whose note the view no longer
// holds (CheckWithView found nothing), while an operation the back captured is still to land:
// that operation is most likely the release, and its landing latches the waiter once its fence
// is submitted (the poll asks for it).
bool NotedReleaseOnItsWay(DmaPusher& dma_pusher, GPUVAddr address, u32 value,
                          V2::SemaphoreOp op) {
    if (!VideoCommon::GpuSplit::CapturedLandingPending()) {
        return false;
    }
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
    if (!device) {
        return false;
    }
    using Kind = VideoCommon::GpuSplit::ReleaseKind;
    const auto recent = VideoCommon::GpuSplit::Get().FindRecentRelease(*device);
    // A reduction's or max release's value is its operand, not the word it leaves (those are
    // untracked: UntrackedReleasePending covers them).
    return recent && recent->kind != Kind::QmdReduction && recent->kind != Kind::QmdMax &&
           V2::SemaphoreMet(op, recent->value, value);
}

} // Anonymous namespace

bool Puller::PendingAcquire::Matches(u32 current) const {
    if (sem_execute) {
        return V2::SemaphoreMet(static_cast<V2::SemaphoreOp>(view_op), current, value);
    }
    switch (static_cast<GpuSemaphoreOperation>(operation)) {
    case GpuSemaphoreOperation::AcquireEqual:
        return current == value;
    case GpuSemaphoreOperation::AcquireGequal:
        return current >= value;
    case GpuSemaphoreOperation::AcquireMask:
        return (current & value) != 0;
    default:
        return true;
    }
}

bool Puller::CanResumeAcquire(DmaPusher& dma_pusher) {
    if (pending_acquire && VideoCommon::GpuSplit::Get().AcquireMode() != V2::Mode::Off)
        [[unlikely]] {
        return PollAcquireView(dma_pusher);
    }
    if (!pending_acquire || !V2Stats::Enabled()) [[likely]] {
        return PollAcquire(dma_pusher);
    }
    const V2Stats::SyncClass sync_class = V2Stats::ClassifyAcquire(
        dma_pusher.memory_manager.GetID(), pending_acquire->address, pending_acquire->value,
        StatsAcquireOp(pending_acquire->operation));
    const V2Stats::AcquireContext context{sync_class};
    const bool met = PollAcquire(dma_pusher);
    V2Stats::NoteAcquire(sync_class, true, met);
    return met;
}

bool Puller::PollAcquireView(DmaPusher& dma_pusher) {
    // A poll of a suspended channel decided without draining the split (upstream ea4340a5b4,
    // MASTER_PLAN §6): until a release meets the acquire the channel stays suspended, and a
    // release landing latches the waiter. No 1 ms valve drain (Fix F3-03): a release the back
    // ran lands once its fence's submission is on the GPU; the fencing thread asks the back to
    // submit that tick (InnerFence::Wait), and an unmet poll with releases still to land asks
    // for a deferred submit too, never a drain.
    auto& split = VideoCommon::GpuSplit::Get();
    const bool verify = split.AcquireMode() == V2::Mode::Verify;
    PendingAcquire& acquire = *pending_acquire;
    const GPUVAddr address = acquire.address;
    const u32 value = acquire.value;
    const u32 operation = acquire.operation;
    const auto op = static_cast<V2::SemaphoreOp>(acquire.view_op);
    bool word_landed = false;
    if (acquire.readback_sent && !verify) {
        // Class C (Fix F3-06): the GPU-written word's readback, sent behind its writers.
        if (!word_readback->Ready()) {
            return false;
        }
        word_readback->Land();
        acquire.readback_sent = false;
        word_landed = true;
    }
    const ViewCheck check = CheckWithView(dma_pusher, address, value, op, verify, word_landed,
                                          acquire.qmd_target);
    bool decided = false;
    bool passes = false;
    if (Passes(check)) {
        decided = true;
        passes = true;
        NoteViewDecision(check);
    } else if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        decided = true;
        passes = true;
    } else if (!verify && SuspendsOnView(check)) {
        decided = true;
        const auto now = std::chrono::steady_clock::now();
        const bool gpu_written = check.reason == Reason::GpuWritten;
        // Fix F3h (F3-X1): no timed net. Which producers of the word land at a fence is asked
        // again with each submit request (at once, then once per millisecond): the back may
        // have run what was queued since, or landed it.
        const bool ask =
            !acquire.submit_requested || now - acquire.last_request >= std::chrono::milliseconds{1};
        if (ask && !gpu_written) {
            const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
            acquire.producer_lands =
                check.reason == Reason::Unlanded || !split.QmdTargetSettled(acquire.qmd_target) ||
                (device && split.QmdReleaseMayWrite(*device, sizeof(u32))) ||
                NotedReleaseOnItsWay(dma_pusher, address, value, op);
        }
        const bool queued = split.ConsumedNow() < split.WritePosition();
        // F3-X1 follow-up (8d31 deadlock: four channels on QMD reduction words 0x13 short, back
        // idle, nothing queued, no landing): an operation can sit at a fence nothing submits
        // (a reduction deferred behind a landing, an operation the front can't see). Once the
        // back is idle and nothing the polls asked for landed since the back last moved, the
        // waiter makes every pending fence land (the v1 acquire's ReleaseFences, on the back):
        // once per back position, so it repeats only after new work ran — never on a timer.
        const u64 landed_now = V2::Landing::landed.load(std::memory_order_acquire);
        const u64 untracked_landed_now =
            V2::Landing::untracked_landed.load(std::memory_order_acquire);
        const u64 landings = landed_now + untracked_landed_now;
        const bool landing_pending =
            V2::Landing::issued.load(std::memory_order_acquire) > landed_now ||
            V2::Landing::untracked_issued.load(std::memory_order_acquire) > untracked_landed_now;
        // Judged once per submit request cycle (`ask`, at most 1/ms), not per poll: landings
        // since the previous request, not since the previous poll (90k polls/s made "nothing
        // landed since" almost always true: d5e4 forced 10-14k times a second).
        const bool landed_since = landings != acquire.landings_seen;
        if (ask) {
            acquire.landings_seen = landings;
        }
        if (ask && V2::ForcePendingLandings(queued, gpu_written, acquire.submit_requested,
                                            acquire.producer_lands, landed_since,
                                            acquire.released_at == split.ConsumedNow() + 1,
                                            landing_pending)) {
            if (V2Stats::Enabled()) [[unlikely]] {
                V2Stats::NoteDecision(VideoCommon::DrainSite::Acquire, Decision::Waited,
                                      Reason::Forced);
            }
            LandEverythingPending(dma_pusher.rasterizer);
            // After it (a job adds no ring record, but other channels' work may have run):
            // the next force needs new work at the back.
            acquire.released_at = split.ConsumedNow() + 1;
            acquire.landings_seen = V2::Landing::landed.load(std::memory_order_acquire) +
                                    V2::Landing::untracked_landed.load(std::memory_order_acquire);
            if (Nx2SemaphoreWaiters::Get().Latched(this) ||
                V2::SemaphoreMet(op, dma_pusher.memory_manager.Read<u32>(address), value)) {
                return true; // NoteSyncAt not needed: nothing queued
            }
        }
        switch (V2::PlanSuspendedPoll(gpu_written, acquire.producer_lands, queued)) {
        case V2::SuspendedPoll::ReadBack:
            // Read the word back again once the back ran what was sent since (at most per
            // 200 us): no drain, no synchronous download on the back.
            if (now - acquire.last_request >= std::chrono::microseconds{200}) {
                acquire.last_request = now;
                SendWordReadback(dma_pusher);
            }
            break;
        case V2::SuspendedPoll::Submit:
            if (ask) {
                // Its fence may sit in a deferred submission: the wait causes its submit
                // (SESSION_HANDOFF §5g). The first ask is at once.
                const bool first_ask = !acquire.submit_requested;
                if (first_ask && V2Stats::Enabled()) [[unlikely]] {
                    // A wait on a known producer (counted once per suspension, F3h).
                    V2Stats::NoteDecision(VideoCommon::DrainSite::Acquire, Decision::Narrowed,
                                          Reason::QmdProducer);
                }
                acquire.submit_requested = true;
                acquire.last_request = now;
                // F4-X2 (R2-F4-01): urgently, for the tick of the newest fence still to land;
                // and the fence the back queues later for the producer is urgent at once (it
                // often does not exist yet at the first ask). Only the producer's position: the
                // known QMD producer, else what was sent when the channel suspended (019d asked
                // for everything sent so far every millisecond: 69 submits/f).
                const auto word_device = dma_pusher.memory_manager.GpuToCpuAddress(address);
                const u64 untracked_ticket =
                    word_device ? V2::Landing::UntrackedTargetFor(*word_device, sizeof(u32)) : 0;
                if (acquire.qmd_target != 0) {
                    V2::LandingFence::Want(acquire.qmd_target);
                } else if (untracked_ticket != 0) {
                    // F3h: the tracked QMD reduction/max release to this word (d5e4: reductions
                    // waited up to 91 ms for their fence): its entry's fence is urgent.
                    V2::LandingFence::Want(V2::Landing::UntrackedPosition(untracked_ticket));
                } else if (first_ask) {
                    V2::LandingFence::Want(split.WritePosition());
                }
                V2::LandingFence::Request(V2::LandingFence::TickForPending());
                split.RequestIdleHook();
            }
            break;
        case V2::SuspendedPoll::Wait:
            break;
        case V2::SuspendedPoll::OldRead:
            // No producer known and nothing queued: the old poll's read (nothing to drain; it
            // flushes and reads the word as the guest would). Counted once per suspension.
            decided = false;
            if (!acquire.old_read) {
                acquire.old_read = true;
                if (V2Stats::Enabled()) [[unlikely]] {
                    V2Stats::NoteDecision(VideoCommon::DrainSite::Acquire, Decision::Waited,
                                          Reason::Valve);
                }
            }
            break;
        }
        if (decided && V2Stats::Enabled()) [[unlikely]] {
            V2Stats::NoteDecision(VideoCommon::DrainSite::Acquire, Decision::Avoided,
                                  Reason::None);
        }
        if (!acquire.reported && now - acquire.since > std::chrono::seconds{2}) {
            acquire.reported = true;
            LOG_WARNING(HW_GPU,
                        "Channel {} waits on semaphore {:#x} (op {}, {:#x}, now {:#x}) for 2 s "
                        "(view: {})",
                        dma_pusher.channel_state.bind_id, address, operation, value,
                        dma_pusher.memory_manager.Read<u32>(address),
                        static_cast<u32>(check.reason));
            if (acquire.sem_execute) {
                // SEM_EXECUTE acquires never blocked before: keep a stuck one from hanging the
                // channel for good.
                return true;
            }
        }
    } else {
        NoteViewDecision(check);
    }
    if (!decided || verify) {
        const u32 view_word = dma_pusher.memory_manager.Read<u32>(address);
        const bool old_passed = PollAcquire(dma_pusher);
        if (acquire.old_read && old_passed && !verify) [[unlikely]] {
            // The old read passed what the view didn't: a release no front note or latch
            // covers (should not happen; the first 32 are logged).
            static std::atomic<u32> logged{};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 32) {
                const auto recent =
                    [&]() -> std::optional<VideoCommon::GpuSplit::RecentRelease> {
                    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(address);
                    return device ? split.FindRecentRelease(*device) : std::nullopt;
                }();
                LOG_INFO(HW_GPU,
                         "V2 acquire: the old read resumed channel {} semaphore {:#x} op {} "
                         "value {:#x}: view reason {}, word {:#x} at the view, {:#x} after, "
                         "waited {} us, newest noted release {} (kind {}, value {:#x})",
                         dma_pusher.channel_state.bind_id, address, operation, value,
                         static_cast<u32>(check.reason), view_word,
                         dma_pusher.memory_manager.Read<u32>(address),
                         std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - acquire.since)
                             .count(),
                         recent.has_value(),
                         recent ? VideoCommon::GpuSplit::ReleaseKindName(recent->kind) : "none",
                         recent ? recent->value : 0);
            }
        }
        if (verify && decided) {
            ReportViewMismatch(dma_pusher, address, value, operation, passes, old_passed,
                               "poll", check.landing, op);
        }
        return old_passed;
    }
    if (passes && check.verdict == V2::AcquireVerdict::MetByQueued) {
        // What the front reads after the acquire waits for the work sent before the release.
        split.NoteSyncAt(check.release);
    }
    return passes;
}

void Puller::SendWordReadback(DmaPusher& dma_pusher) {
    PendingAcquire& acquire = *pending_acquire;
    const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
    if (!device) {
        return;
    }
    if (!word_readback) {
        word_readback = std::make_shared<V2::WordReadback>();
    }
    auto& split = VideoCommon::GpuSplit::Get();
    const V2::Ir::ReadbackBody body{
        .op = V2::Ir::ReadbackOp::WordReadback,
        .object = split.Ref(dma_pusher.rasterizer),
        .object2 = split.Ref(word_readback.get()),
        .cache = 0,
        .address = *device,
        .bytes = sizeof(u32),
    };
    split.Emit(V2::Ir::Kind::Readback, body,
               [rasterizer = dma_pusher.rasterizer, slot = word_readback.get(), device = *device] {
                   slot->Run(*rasterizer, device, sizeof(u32));
               });
    split.Publish();
    acquire.readback_sent = true;
}

void Puller::SuspendOnView(DmaPusher& dma_pusher, PendingAcquire acquire, bool gpu_written,
                           u64 qmd_target) {
    acquire.qmd_target = qmd_target;
    // operation 0: a SEM_EXECUTE comparison the latch can't make (circular, mask zero); the polls
    // read the word.
    auto& waiters = Nx2SemaphoreWaiters::Get();
    if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address);
        device && acquire.operation != 0) {
        waiters.Add(this, *device, acquire.value, acquire.operation);
    }
    // Releases only what already signalled (nothing with async fences): the fencing thread lands
    // releases as the GPU passes them, and a release among them latches the waiter.
    OnBackBarrier(dma_pusher, VideoCommon::V2::Ir::BarrierOp::ReleaseFences,
                  [rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(false); },
                  VideoCommon::V2::Ir::ReleaseFencesUnforced);
    acquire.last_request = std::chrono::steady_clock::now();
    pending_acquire = acquire;
    if (gpu_written) {
        SendWordReadback(dma_pusher);
    }
    dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
    VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
}

bool Puller::PollAcquire(DmaPusher& dma_pusher) {
    if (!pending_acquire) {
        return true;
    }
    // The release may still be queued for the split GPU thread's back half (and the caches
    // ReadSemaphoreWord asks are its state).
    DrainBack();
    const u32 current = ReadSemaphoreWord(dma_pusher, pending_acquire->address);
    if (Nx2SemaphoreWaiters::Get().Latched(this)) {
        if (static std::atomic<u32> rescued{};
            !pending_acquire->Matches(current) && rescued.fetch_add(1) < 64) {
            LOG_INFO(HW_GPU, "Channel {} met semaphore {:#x} value {:#x} at its release (now {:#x})",
                     dma_pusher.channel_state.bind_id, pending_acquire->address,
                     pending_acquire->value, current);
        }
        return true;
    }
    if (pending_acquire->Matches(current)) {
        return true;
    }
    // With delayed fence operations (Balanced or Accurate fences, accurate GPU), a release
    // queued after the acquire released the fences waits in the caches for the next fence. With
    // every channel waiting none comes: release them again now and then while waiting.
    const bool delayed = Settings::IsGPUFenceBehaviorDefault()
                             ? Settings::IsGPULevelHigh()
                             : Settings::IsGPUFenceBehaviorBalanced() ||
                                   Settings::IsGPUFenceBehaviorAccurate();
    if (const auto now = std::chrono::steady_clock::now();
        delayed && now - pending_acquire->last_fences >= std::chrono::milliseconds{1}) {
        pending_acquire->last_fences = now;
        ReleaseFences(dma_pusher);
        if (Nx2SemaphoreWaiters::Get().Latched(this) ||
            pending_acquire->Matches(ReadSemaphoreWord(dma_pusher, pending_acquire->address))) {
            return true;
        }
    }
    if (!pending_acquire->reported &&
        std::chrono::steady_clock::now() - pending_acquire->since > std::chrono::seconds{2}) {
        pending_acquire->reported = true;
        const auto device = dma_pusher.memory_manager.GpuToCpuAddress(pending_acquire->address);
        LOG_WARNING(HW_GPU, "Channel {} waits on semaphore {:#x} (op {}, {:#x}, now {:#x}) for 2 s; "
                    "device {:#x}, buffer cache holds GPU data: {}, words {:08x} {:08x} {:08x} {:08x}",
                    dma_pusher.channel_state.bind_id, pending_acquire->address,
                    pending_acquire->operation, pending_acquire->value, current, device.value_or(0),
                    device && dma_pusher.rasterizer->MustFlushRegion(*device, 16, VideoCommon::CacheType::All),
                    dma_pusher.memory_manager.Read<u32>(pending_acquire->address & ~GPUVAddr{0xf}),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 4),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 8),
                    dma_pusher.memory_manager.Read<u32>((pending_acquire->address & ~GPUVAddr{0xf}) + 12));
    }
    return false;
}

bool Puller::ResumeAcquire(DmaPusher& dma_pusher) {
    if (!CanResumeAcquire(dma_pusher)) {
        return false;
    }
    // Only a resume from a suspension counts: DmaPusher::Step calls this before every entry, and
    // counting those made the count odd ("suspended") at random, so submit-time snapshots were
    // thrown away for about half of all entries (CommandList::snapshot_suspensions).
    if (pending_acquire) {
        pending_acquire.reset();
        dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
        Nx2SemaphoreWaiters::Get().Remove(this);
    }
    return true;
}

void Puller::ProcessBindMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_DEBUG(HW_GPU, "Binding subchannel {} to engine {:#x}", method_call.subchannel, method_call.argument);
    u32 engine = method_call.argument;
    if ((engine & ~Gf100BindClassMask) != 0 && (engine & ~Gf100BindValidMask) == 0) {
        engine &= Gf100BindClassMask;
    }
    // Switch 2 binds Ampere classes; run them on the Maxwell engines (same method layout for
    // most state). Class 0 unbinds the subchannel.
    switch (engine & 0xFFFF) {
    case 0xC797: // AMPERE_B
        engine = static_cast<u32>(EngineID::MAXWELL_B);
        break;
    case 0xC7C0: // AMPERE_COMPUTE_B
        engine = static_cast<u32>(EngineID::KEPLER_COMPUTE_B);
        break;
    case 0xC7B5: // AMPERE_DMA_COPY_B
        engine = static_cast<u32>(EngineID::MAXWELL_DMA_COPY_A);
        break;
    case 0x0:
        if (g_nx2_ampere) {
            LOG_DEBUG(HW_GPU, "Switch 2 unbinds subchannel {}", method_call.subchannel);
            return;
        }
        break;
    default:
        break;
    }

    const auto engine_id = static_cast<EngineID>(engine);
    bound_engines[method_call.subchannel] = engine_id;
    switch (engine_id) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->fermi_2d, method_call.subchannel, EngineTypes::Fermi2D);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_3d, method_call.subchannel, EngineTypes::Maxwell3D);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_compute, method_call.subchannel, EngineTypes::KeplerCompute);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->maxwell_dma, method_call.subchannel, EngineTypes::MaxwellDMA);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->kepler_memory, method_call.subchannel, EngineTypes::KeplerMemory);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.BindSubchannel(&dma_pusher.channel_state.payload->nv01_timer, method_call.subchannel, EngineTypes::Nv01Timer);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine {:04X}", engine_id);
        break;
    }
}

void Puller::ProcessFenceActionMethod(DmaPusher& dma_pusher) {
    switch (regs.fence_action.op) {
    case Puller::FenceOperation::Acquire:
        // UNIMPLEMENTED_MSG("Channel Scheduling pending.");
        // WaitFence(regs.fence_action.syncpoint_id, regs.fence_value);
        OnBackBarrier(dma_pusher, VideoCommon::V2::Ir::BarrierOp::ReleaseFences,
                      [rasterizer = dma_pusher.rasterizer] { rasterizer->ReleaseFences(); });
        VideoCommon::GpuSplit::Get().NoteSync();
        // A syncpoint wait orders the guest's CPU writes before it, as a semaphore acquire.
        VideoCommon::V2::Versions::HookEpoch(VideoCommon::V2::Versions::EpochReason::Acquire);
        break;
    case Puller::FenceOperation::Increment:
        if (auto& split = VideoCommon::GpuSplit::Get(); split.Active()) {
            const u32 id = static_cast<u32>(regs.fence_action.syncpoint_id.Value());
            const VideoCommon::V2::Ir::SyncPointBody body{
                .id = id,
                .value = 0,
                .op = VideoCommon::V2::Ir::SyncPointOp::Increment,
                .object = split.Ref(dma_pusher.rasterizer),
                .a = 0,
                .b = 0,
            };
            split.Emit(VideoCommon::V2::Ir::Kind::SyncPoint, body,
                       [rasterizer = dma_pusher.rasterizer, id] { rasterizer->SignalSyncPoint(id); });
        } else {
            dma_pusher.rasterizer->SignalSyncPoint(
                static_cast<u32>(regs.fence_action.syncpoint_id.Value()));
        }
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented operation {}", regs.fence_action.op.Value());
        break;
    }
}

void Puller::ProcessSemaphoreTriggerMethod(DmaPusher& dma_pusher) {
    const auto semaphoreOperationMask = 0xF;
    const auto op = GpuSemaphoreOperation(regs.semaphore_trigger & semaphoreOperationMask);
    // EDEN_NX2_SEM_WATCH=<gpu address>: log every release/acquire on the 16 bytes there.
    static const GPUVAddr watch = [] {
        const char* value = std::getenv("EDEN_NX2_SEM_WATCH");
        return value ? GPUVAddr{std::strtoull(value, nullptr, 16)} : GPUVAddr{};
    }();
    if (watch != 0 && (regs.semaphore_address.SemaphoreAddress() & ~GPUVAddr{0xf}) == watch) {
        LOG_INFO(HW_GPU, "Semaphore watch: channel {} as {} device {:#x} op {} {:#x} value {:#x} (now {:#x})",
                 dma_pusher.channel_state.bind_id, dma_pusher.memory_manager.GetID(),
                 dma_pusher.memory_manager.GpuToCpuAddress(regs.semaphore_address.SemaphoreAddress()).value_or(0),
                 static_cast<u32>(op),
                 regs.semaphore_address.SemaphoreAddress(), regs.semaphore_sequence,
                 dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress()));
    }
    if (op == GpuSemaphoreOperation::WriteLong) {
        const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
        const u32 payload = regs.semaphore_sequence;
        // RELEASE_SIZE (bit 24) set means a 4-byte release without the timestamp; writing 16
        // bytes there clobbers the words next to it (Switch 2 packs semaphores tightly).
        const auto properties = g_nx2_ampere && (regs.semaphore_trigger & (1U << 24)) != 0
                                    ? VideoCommon::QueryPropertiesFlags::IsAFence
                                    : VideoCommon::QueryPropertiesFlags::HasTimeout;
        OnBackRelease(dma_pusher, sequence_address, payload,
                      properties == VideoCommon::QueryPropertiesFlags::IsAFence ? 4 : 16,
                      [rasterizer = dma_pusher.rasterizer, sequence_address, properties, payload] {
            rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload, properties, payload, 0);
        });
        if (V2Stats::Enabled()) [[unlikely]] {
            V2Stats::NoteRelease(dma_pusher.memory_manager.GetID(), sequence_address, payload,
                                 V2Stats::ReleaseKind::Literal);
        }
    } else if (g_nx2_ampere) {
        // Switch 2 waits on semaphores the CPU (or another channel) releases, e.g. once it has
        // written the commands this channel runs next: suspend the channel until the value is
        // met instead of reading ahead (approach from changes.patch).
        PendingAcquire acquire{
            .address = regs.semaphore_address.SemaphoreAddress(),
            .value = regs.semaphore_sequence,
            .operation = static_cast<u32>(op),
            .since = std::chrono::steady_clock::now(),
            .last_fences = std::chrono::steady_clock::now(),
            .last_request = std::chrono::steady_clock::now(),
            .reported = false,
            .view_op = static_cast<u8>(ViewOp(static_cast<u32>(op))),
        };
        if (op != GpuSemaphoreOperation::AcquireEqual && op != GpuSemaphoreOperation::AcquireGequal &&
            op != GpuSemaphoreOperation::AcquireMask) {
            LOG_ERROR(HW_GPU, "Invalid semaphore operation {}", static_cast<u32>(op));
            return;
        }
        VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireChecks, 1);
        VideoCommon::V2::Resolve::MultiFront::Note(
            VideoCommon::V2::Resolve::MultiFront::EventKind::Acquire, acquire.address, acquire.value,
            op == GpuSemaphoreOperation::AcquireEqual
                ? VideoCommon::V2::Resolve::MultiFront::AcquireOp::Equal
                : VideoCommon::V2::Resolve::MultiFront::AcquireOp::GreaterEqual);
        auto& split = VideoCommon::GpuSplit::Get();
        // EDEN_NX2_V2_ACQUIRE: met by memory (class A) or by a release queued ahead (B) passes
        // without draining; anything else takes the path below, as before.
        bool view_passes = false;
        u64 view_landing = 0;
        const V2::Mode view_mode = split.AcquireMode();
        if (view_mode != V2::Mode::Off) [[unlikely]] {
            const ViewCheck view =
                CheckWithView(dma_pusher, acquire.address, acquire.value,
                              ViewOp(acquire.operation), view_mode != V2::Mode::On);
            view_passes = Passes(view);
            view_landing = view.landing;
            if (view_mode == V2::Mode::On && SuspendsOnView(view)) {
                // Not met yet (nothing sent to the word meets it, the newest write doesn't, or
                // only a counter/unmapped/GPU-written word decides): a drain can't settle it,
                // only a later release, its landing or the word's readback. Suspend at once:
                // latch, no forced fence, submit and back GPU wait (MASTER_PLAN §6, F3-05).
                if (V2Stats::Enabled()) [[unlikely]] {
                    V2Stats::NoteDecision(VideoCommon::DrainSite::Acquire, Decision::Avoided);
                }
                NoteAcquireRecord(dma_pusher, acquire.address, acquire.value, acquire.operation,
                                  static_cast<u32>(view.verdict));
                SuspendOnView(dma_pusher, acquire, view.reason == Reason::GpuWritten,
                              view.qmd_target);
                return;
            }
            NoteViewDecision(view);
            if (view_passes && view_mode == V2::Mode::On) {
                if (view.verdict == V2::AcquireVerdict::MetByQueued) {
                    // What the front reads after the acquire waits for the work sent before
                    // the release (GpuSplit::NoteSyncAt), not for what was sent after it.
                    split.NoteSyncAt(view.release);
                }
                NoteAcquireRecord(dma_pusher, acquire.address, acquire.value, acquire.operation,
                                  static_cast<u32>(view.verdict));
                return;
            }
        }
        NoteAcquireRecord(dma_pusher, acquire.address, acquire.value, acquire.operation, ~0U);
        // Verify: a pass of the view must be one of the old path too (no suspension).
        const auto verify_result = [&](bool old_passed) {
            if (view_passes) [[unlikely]] {
                ReportViewMismatch(dma_pusher, acquire.address, acquire.value,
                                   acquire.operation, true, old_passed, "acquire", view_landing,
                                   ViewOp(acquire.operation));
            }
        };
        const bool v2_stats = V2Stats::Enabled();
        const V2Stats::SyncClass sync_class =
            v2_stats ? V2Stats::ClassifyAcquire(dma_pusher.memory_manager.GetID(), acquire.address,
                                                acquire.value, StatsAcquireOp(acquire.operation))
                     : V2Stats::SyncClass::Count;
        const V2Stats::AcquireContext context{sync_class};
        DrainBack();
        const bool met = acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address));
        if (v2_stats) [[unlikely]] {
            V2Stats::NoteAcquire(sync_class, false, met);
        }
        if (met) {
            verify_result(true);
            return;
        }
        // Registered before the fences run, so a release among them that meets the value counts
        // even if a later one overwrites it.
        auto& waiters = Nx2SemaphoreWaiters::Get();
        if (const auto device = dma_pusher.memory_manager.GpuToCpuAddress(acquire.address)) {
            waiters.Add(this, *device, acquire.value, acquire.operation);
        }
        ReleaseFences(dma_pusher);
        if (!waiters.Latched(this) && !acquire.Matches(ReadSemaphoreWord(dma_pusher, acquire.address))) {
            pending_acquire = acquire;
            dma_pusher.suspensions.fetch_add(1, std::memory_order_release);
            VideoCommon::Nx2Profile::Add(VideoCommon::Nx2Profile::Counter::AcquireWaits, 1);
            verify_result(false);
        } else {
            waiters.Remove(this);
            verify_result(true);
        }
    } else {
        DrainBack();
        do {
            const u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
            regs.acquire_source = true;
            regs.acquire_value = regs.semaphore_sequence;
            if (op == GpuSemaphoreOperation::AcquireEqual) {
                regs.acquire_active = true;
                regs.acquire_mode = false;
                if (word != regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireGequal) {
                regs.acquire_active = true;
                regs.acquire_mode = true;
                if (word < regs.acquire_value) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else if (op == GpuSemaphoreOperation::AcquireMask) {
                if (word && regs.semaphore_sequence == 0) {
                    ReleaseFences(dma_pusher);
                    continue;
                }
            } else {
                LOG_ERROR(HW_GPU, "Invalid semaphore operation");
            }
        } while (false);
    }
}

void Puller::ProcessSemaphoreRelease(DmaPusher& dma_pusher) {
    const GPUVAddr sequence_address{regs.semaphore_address.SemaphoreAddress()};
    const u32 payload = regs.semaphore_release;
    OnBackRelease(dma_pusher, sequence_address, payload, 4,
                  [rasterizer = dma_pusher.rasterizer, sequence_address, payload] {
        rasterizer->Query(sequence_address, VideoCommon::QueryType::Payload,
                          VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
    });
    if (V2Stats::Enabled()) [[unlikely]] {
        V2Stats::NoteRelease(dma_pusher.memory_manager.GetID(), sequence_address, payload,
                             V2Stats::ReleaseKind::Literal);
    }
}

void Puller::ProcessSemaphoreAcquire(DmaPusher& dma_pusher) {
    VideoCommon::V2::Resolve::MultiFront::Note(
        VideoCommon::V2::Resolve::MultiFront::EventKind::Acquire,
        regs.semaphore_address.SemaphoreAddress(), regs.semaphore_acquire);
    DrainBack();
    // The CPU may rewrite what the next draws read once it sees the release this waits for.
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    VideoCommon::V2::Versions::HookEpoch(VideoCommon::V2::Versions::EpochReason::Acquire);
    u32 word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
    const auto value = regs.semaphore_acquire;
    while (word != value) {
        regs.acquire_active = true;
        regs.acquire_value = value;
        ReleaseFences(dma_pusher);
        word = dma_pusher.memory_manager.Read<u32>(regs.semaphore_address.SemaphoreAddress());
        // TODO(kemathe73) figure out how to do the acquire_timeout
        regs.acquire_mode = false;
        regs.acquire_source = false;
    }
}

// Ampere SEM_EXECUTE (method 0x1B): operation in bits 2:0 on the semaphore at
// SEM_ADDR_HI:SEM_ADDR_LO (methods 0x18/0x17) with payload SEM_PAYLOAD_LO (0x19).
void Puller::ProcessSemaphoreExecute(DmaPusher& dma_pusher, u32 argument) {
    const GPUVAddr address = (GPUVAddr{regs.reg_array[0x18] & 0x1FFFFFF} << 32) |
                             (regs.reg_array[0x17] & ~3U);
    const u32 payload = regs.reg_array[0x19];
    const u32 operation = argument & 7;
    if (operation == 1) { // RELEASE
        OnBackRelease(dma_pusher, address, payload, 4,
                      [rasterizer = dma_pusher.rasterizer, address, payload] {
            rasterizer->Query(address, VideoCommon::QueryType::Payload,
                              VideoCommon::QueryPropertiesFlags::IsAFence, payload, 0);
        });
        if (V2Stats::Enabled()) [[unlikely]] {
            V2Stats::NoteRelease(dma_pusher.memory_manager.GetID(), address, payload,
                                 V2Stats::ReleaseKind::Literal);
        }
        return;
    }
    // Acquires: check once after flushing pending fences; never block the GPU thread on them.
    const bool v2_stats = V2Stats::Enabled();
    // By operation, as `ready` below (1 is the release).
    static constexpr std::array<V2Stats::AcquireOp, 6> v2_ops{
        V2Stats::AcquireOp::Equal, V2Stats::AcquireOp::Equal,
        V2Stats::AcquireOp::Gequal, V2Stats::AcquireOp::CircularGequal,
        V2Stats::AcquireOp::Mask, V2Stats::AcquireOp::MaskZero,
    };
    const V2Stats::SyncClass sync_class =
        v2_stats && operation < v2_ops.size() && operation != 1
            ? V2Stats::ClassifyAcquire(dma_pusher.memory_manager.GetID(), address, payload,
                                       v2_ops[operation])
            : V2Stats::SyncClass::Count;
    const V2Stats::AcquireContext context{sync_class};
    bool view_passes = false;
    u64 view_landing = 0;
    static constexpr std::array<V2::SemaphoreOp, 6> view_ops{
        V2::SemaphoreOp::Equal,  V2::SemaphoreOp::Equal, V2::SemaphoreOp::Gequal,
        V2::SemaphoreOp::CircularGequal, V2::SemaphoreOp::Mask, V2::SemaphoreOp::MaskZero,
    };
    if (const V2::Mode view_mode = VideoCommon::GpuSplit::Get().AcquireMode();
        view_mode != V2::Mode::Off && operation < 6 && operation != 1) [[unlikely]] {
        // EDEN_NX2_V2_ACQUIRE: one the view meets skips the drain and the fence release; with
        // =1 one it doesn't suspends the channel.
        const bool on = view_mode == V2::Mode::On;
        const ViewCheck view =
            CheckWithView(dma_pusher, address, payload, view_ops[operation], !on);
        view_passes = Passes(view);
        view_landing = view.landing;
        if (on && SuspendsOnView(view)) {
            // Fix F3-07: suspended like a trigger acquire (MASTER_PLAN §6), not drained, fenced
            // and then run past unmet.
            if (V2Stats::Enabled()) [[unlikely]] {
                V2Stats::NoteDecision(VideoCommon::DrainSite::Acquire, Decision::Avoided);
            }
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
            VideoCommon::V2::Versions::HookEpoch(VideoCommon::V2::Versions::EpochReason::Acquire);
            // The latch's comparisons (GpuSemaphoreOperation); 0: none fits, the polls decide.
            static constexpr std::array<u32, 6> latch_ops{0x1, 0x1, 0x4, 0x0, 0x8, 0x0};
            const auto now = std::chrono::steady_clock::now();
            SuspendOnView(dma_pusher,
                          PendingAcquire{
                              .address = address,
                              .value = payload,
                              .operation = latch_ops[operation],
                              .since = now,
                              .last_fences = now,
                              .last_request = now,
                              .reported = false,
                              .view_op = static_cast<u8>(view_ops[operation]),
                              .sem_execute = true,
                          },
                          view.reason == Reason::GpuWritten, view.qmd_target);
            return;
        }
        NoteViewDecision(view);
        if (view_passes && on) {
            if (view.verdict == V2::AcquireVerdict::MetByQueued) {
                VideoCommon::GpuSplit::Get().NoteSyncAt(view.release);
            }
            VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    VideoCommon::V2::Versions::HookEpoch(VideoCommon::V2::Versions::EpochReason::Acquire);
            if (sync_class != V2Stats::SyncClass::Count) [[unlikely]] {
                V2Stats::NoteAcquire(sync_class, false, true);
            }
            return;
        }
    }
    DrainBack();
    VideoCommon::Nx2FastDraw::Break(VideoCommon::Nx2FastDraw::Reason::Fence);
    VideoCommon::V2::Versions::HookEpoch(VideoCommon::V2::Versions::EpochReason::Acquire);
    ReleaseFences(dma_pusher);
    const u32 word = dma_pusher.memory_manager.Read<u32>(address);
    const bool ready = operation == 0   ? word == payload
                       : operation == 2 ? word >= payload
                       : operation == 3 ? static_cast<s32>(word - payload) >= 0
                       : operation == 4 ? (word & payload) != 0
                       : operation == 5 ? (word & payload) == 0
                                        : true;
    if (sync_class != V2Stats::SyncClass::Count) [[unlikely]] {
        V2Stats::NoteAcquire(sync_class, false, ready);
    }
    if (view_passes) [[unlikely]] {
        ReportViewMismatch(dma_pusher, address, payload, operation, true, ready, "SEM_EXECUTE",
                           view_landing,
                           operation < 6 ? view_ops[operation] : V2::SemaphoreOp::Equal);
    }
    if (!ready) {
        static std::atomic<u32> reported{};
        if (reported.fetch_add(1) < 8) {
            LOG_WARNING(HW_GPU, "SEM_EXECUTE op {} at {:#x}: {:#x} vs {:#x} not met, continuing",
                        operation, address, word, payload);
        }
    }
}

/// Calls a GPU puller method.
void Puller::CallPullerMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    static const bool nx2_trace = std::getenv("EDEN_NX2_PULLER_TRACE") != nullptr;
    if (nx2_trace) {
        LOG_INFO(HW_GPU, "nx2 puller sub={} method={:#x} arg={:#x}", method_call.subchannel,
                 method_call.method, method_call.argument);
    }
    regs.reg_array[method_call.method] = method_call.argument;
    const auto method = static_cast<BufferMethods>(method_call.method);

    switch (method) {
    case BufferMethods::BindObject: {
        ProcessBindMethod(dma_pusher, method_call);
        break;
    }
    case BufferMethods::Nop:
    case BufferMethods::SemaphoreAddressHigh:
    case BufferMethods::SemaphoreAddressLow:
    case BufferMethods::SemaphoreSequencePayload:
    case BufferMethods::SyncpointPayload:
    case BufferMethods::WrcacheFlush:
        break;
    case BufferMethods::RefCnt:
        OnBackBarrier(dma_pusher, VideoCommon::V2::Ir::BarrierOp::SignalReference,
                      [rasterizer = dma_pusher.rasterizer] { rasterizer->SignalReference(); });
        break;
    case BufferMethods::SyncpointOperation:
        ProcessFenceActionMethod(dma_pusher);
        break;
    case BufferMethods::WaitForIdle:
        OnBackBarrier(dma_pusher, VideoCommon::V2::Ir::BarrierOp::WaitForIdle,
                      [rasterizer = dma_pusher.rasterizer] { rasterizer->WaitForIdle(); });
        // The channel's own engines go idle (W-01a: other channels' work isn't ordered by it).
        VideoCommon::GpuSplit::Get().NoteChannelSync();
        break;
    case BufferMethods::SemaphoreOperation: {
        ProcessSemaphoreTriggerMethod(dma_pusher);
        break;
    }
    case BufferMethods::NonStallInterrupt: {
        LOG_ERROR(HW_GPU, "Special puller engine method NonStallInterrupt not implemented");
        break;
    }
    case BufferMethods::MemOpA: {
        LOG_TRACE(HW_GPU, "Memory Operation A");
        break;
    }
    case BufferMethods::MemOpB: {
        // Implement this better.
        OnBackBarrier(dma_pusher, VideoCommon::V2::Ir::BarrierOp::InvalidateCaches,
                      [rasterizer = dma_pusher.rasterizer] { rasterizer->InvalidateGPUCache(); });

        break;
    }
    case BufferMethods::MemOpC:
    case BufferMethods::MemOpD: {
        LOG_TRACE(HW_GPU, "Memory Operation C,D");
        break;
    }
    case BufferMethods::SemaphoreAcquire: {
        // On Ampere host classes 0x17..0x1A are SEM_ADDR_LO/HI and SEM_PAYLOAD_LO/HI.
        if (!g_nx2_ampere) {
            ProcessSemaphoreAcquire(dma_pusher);
        }
        break;
    }
    case BufferMethods::SemaphoreRelease: {
        if (g_nx2_ampere) {
            ProcessSemaphoreExecute(dma_pusher, method_call.argument);
        } else {
            ProcessSemaphoreRelease(dma_pusher);
        }
        break;
    }
    case BufferMethods::Yield: {
        // TODO(Kmather73): Research and implement this method.
        LOG_ERROR(HW_GPU, "Special puller engine method Yield not implemented");
        break;
    }
    default:
        LOG_ERROR(HW_GPU, "Special puller engine method {:X} not implemented", method);
        if (static bool dumped = false; g_nx2_ampere && !dumped) {
            dumped = true;
            dma_pusher.DumpCurrentEntry("unknown host method");
        }
        break;
    }
}

/// Calls a GPU engine method.
static bool Nx2MethodTrace() {
    static const bool trace = std::getenv("EDEN_NX2_METHOD_TRACE") != nullptr;
    return trace;
}

void Puller::CallEngineMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    const EngineID engine = bound_engines[method_call.subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 method sub={} engine={:04X} method={:#x} arg={:#x}",
                 method_call.subchannel, static_cast<u32>(engine), method_call.method,
                 method_call.argument);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMethod(dma_pusher.system, method_call.method, method_call.argument, method_call.IsLastCall());
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU engine multivalue method.
void Puller::CallEngineMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    const EngineID engine = bound_engines[subchannel];
    if (Nx2MethodTrace()) {
        LOG_INFO(HW_GPU, "nx2 multimethod sub={} engine={:04X} method={:#x} amount={} first={:#x}",
                 subchannel, static_cast<u32>(engine), method, amount, base_start[0]);
    }
    switch (engine) {
    case EngineID::FERMI_TWOD_A:
        dma_pusher.channel_state.payload->fermi_2d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_B:
        dma_pusher.channel_state.payload->maxwell_3d.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_COMPUTE_B:
        dma_pusher.channel_state.payload->kepler_compute.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::MAXWELL_DMA_COPY_A:
        dma_pusher.channel_state.payload->maxwell_dma.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::KEPLER_INLINE_TO_MEMORY_B:
        dma_pusher.channel_state.payload->kepler_memory.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    case EngineID::NV01_TIMER:
        dma_pusher.channel_state.payload->nv01_timer.CallMultiMethod(dma_pusher.system, method, base_start, amount, methods_pending);
        break;
    default:
        UNIMPLEMENTED_MSG("Unimplemented engine");
        break;
    }
}

/// Calls a GPU method.
void Puller::CallMethod(DmaPusher& dma_pusher, const MethodCall& method_call) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method_call.method, method_call.subchannel);
    ASSERT(method_call.subchannel < bound_engines.size());

    if (ExecuteMethodOnEngine(dma_pusher, method_call.method)) {
        CallEngineMethod(dma_pusher, method_call);
    } else {
        CallPullerMethod(dma_pusher, method_call);
    }
}

/// Calls a GPU multivalue method.
void Puller::CallMultiMethod(DmaPusher& dma_pusher, u32 method, u32 subchannel, const u32* base_start, u32 amount, u32 methods_pending) {
    LOG_TRACE(HW_GPU, "Processing method {:08X} on subchannel {}", method, subchannel);
    ASSERT(subchannel < bound_engines.size());
    if (ExecuteMethodOnEngine(dma_pusher, method)) {
        CallEngineMultiMethod(dma_pusher, method, subchannel, base_start, amount, methods_pending);
    } else {
        for (u32 i = 0; i < amount; i++) {
            CallPullerMethod(dma_pusher, MethodCall{
                method,
                base_start[i],
                subchannel,
                methods_pending - i,
            });
        }
    }
}

/// Determines where the method should be executed.
[[nodiscard]] bool Puller::ExecuteMethodOnEngine(DmaPusher& dma_pusher, u32 method) {
    const auto buffer_method = BufferMethods(method);
    return buffer_method >= BufferMethods::NonPullerMethods;
}

} // namespace Tegra::Engines
