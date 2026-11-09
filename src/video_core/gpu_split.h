// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <functional>
#include <memory>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>

#include "common/common_types.h"
#include "video_core/v2/hazard.h"
#include "video_core/v2/ir.h"
#include "video_core/v2/ir_arena.h"
#include "video_core/v2/ir_trace.h"
#include "video_core/v2/landing.h"
#include "video_core/v2/v2_switches.h"

namespace Core {
class System;
}

namespace Tegra {
class MemoryManager;
}

namespace Tegra::Engines {
class EngineInterface;
}

namespace VideoCommon::V2::Resolve {
class SplitResolve;
struct SplitTraits;
} // namespace VideoCommon::V2::Resolve

namespace VideoCommon {

/// Why the split's front waited for its back (EDEN_NX2_GPU_PROFILE counts each).
enum class DrainSite : u32 {
    Other,
    FetchDrain,     ///< EDEN_NX2_FETCH_DRAIN: before every entry
    Acquire,        ///< A semaphore acquire reads the value the back may release
    SemaphoreFlush, ///< The acquired semaphore is in the buffer cache
    FenceRelease,   ///< The puller's acquires release the pending fences
    ListWriters,    ///< GPU-written command lists: the dispatches and copies sent before them
    ListSync,       ///< Command lists on draw-written pages: the work before the last host sync
    ListFlush,      ///< Command lists on GPU-written pages: drain and flush the pages
    MacroDmaRead,   ///< The front's macros read memory (MME DMA)
    MacroDmaFlush,  ///< Those words are in the buffer cache
    GpSyncFence,    ///< A GP entry's sync bit (sync_memory_operations)
    Map,            ///< GPU memory maps before later command reads
    FrontStart,     ///< The front takes an engine's macros over
    FrontVerify,    ///< EDEN_NX2_FRONT_VERIFY
    Count,
};

inline constexpr std::array<const char*, static_cast<size_t>(DrainSite::Count)> drain_site_names{
    "other",       "fetch drain",     "acquire",    "semaphore flush", "fence release",
    "list writers", "list sync",      "list flush", "macro dma read",  "macro dma flush",
    "gp sync fence", "map",           "front start", "front verify",
};

// Splits the GPU thread in two (NX2_GPU_THREAD_SPLIT.md). The GPU thread (the front) takes the
// CPU's commands, parses command lists and runs the puller; engine method calls and everything
// behind them (register state, macros, draws, the rasterizer and its caches) run in order on a
// second thread (the back), fed through a single-producer single-consumer ring. Work the front
// would do on back-owned state is deferred into the same ring; the front waits for the back to
// catch up (Drain) only where it needs a result from that state, e.g. a semaphore it acquires.
// On for Switch 2 titles with the 3D and compute engines' macros on the front, against copies of
// the registers they read (Maxwell3D and KeplerCompute Front*), which sends the writes they make.
// EDEN_NX2_SPLIT=1 keeps the macros on the back; EDEN_NX2_SPLIT=0 runs everything on one thread.
class GpuSplit {
public:
    static GpuSplit& Get() {
        return instance;
    }

    /// Whether the front sends work to the back (front thread only).
    [[nodiscard]] bool Active() const {
        return active;
    }
    /// How often the split started: state the front copied during an earlier start is stale.
    [[nodiscard]] u32 Starts() const {
        return starts;
    }
    /// Front thread: ring words sent so far, and executed by the back (v2 stats).
    [[nodiscard]] u64 WritePosition() const {
        return write_position;
    }
    [[nodiscard]] u64 ConsumedPosition() const {
        return consumed.load(std::memory_order_relaxed);
    }
    /// Front thread: ends the open IR StateRun (the next method starts a record) and returns
    /// the write position, which then follows the last method sent (the v2 stats' releases).
    [[nodiscard]] u64 CutPosition() {
        IrCut();
        return write_position;
    }
    /// Front thread: ring and payload words sent so far (grows with every method sent; the
    /// write position does not when a method joins an open IR StateRun).
    [[nodiscard]] u64 SentWords() const {
        return sent_words;
    }
    /// Whether the ring carries typed IR records (EDEN_NX2_V2_IR, fixed at each start).
    [[nodiscard]] bool TypedIr() const {
        return ir;
    }
    /// Front thread: the last IR record sent and the ring position after it. Waiting for the
    /// ticket (WaitFor) waits until the back executed that record. Without the IR the sequence
    /// stays 0 and only the position is meaningful.
    [[nodiscard]] V2::Ir::Ticket Mark() const {
        return {ir_sequence, write_position};
    }
    /// Any thread: the sequence number of the last IR record the back finished (updated when it
    /// hands ring space back, so it may lag).
    [[nodiscard]] u64 ExecutedSequence() const {
        return consumed_sequence.load(std::memory_order_acquire);
    }
    /// Whether the front runs the engines' macros (EDEN_NX2_SPLIT=2).
    [[nodiscard]] bool FrontMacros() const {
        return front_macros;
    }
    /// Whether the front sends Draw/Dispatch records before the methods that draw or launch
    /// compute work (EDEN_NX2_V2_IR_DRAWS with the typed IR and front macros; fixed per start).
    /// EDEN_NX2_V2_PARALLEL: the parallel resolve (front segmenter, shards, commit side), null
    /// when off. Fixed between a start and a stop.
    [[nodiscard]] V2::Resolve::SplitResolve* Parallel() const {
        return parallel;
    }
    [[nodiscard]] bool DrawRecords() const {
        return draw_records;
    }
    /// Fix F1 (S1 single owner, v2/s1_deltas.h; fixed per start): the 3D front sends classified
    /// StateRuns the back applies without executing the methods (EDEN_NX2_V2_REGSEM=1 with the
    /// typed IR, front macros and EDEN_NX2_V2_IR_DRAWS=1). S1Check: REGSEM or IR_DRAWS is
    /// verify, the back executes the method words and checks the classified entries sent with
    /// them. EDEN_NX2_V2_PARALLEL's shards apply the classified entries too.
    [[nodiscard]] bool S1Exec() const {
        return s1_exec;
    }
    [[nodiscard]] bool S1Check() const {
        return s1_check;
    }
    /// Front, S1: room for `words` of classified entries for `engine` in the open StateRun (a
    /// Classified one with S1Exec), opening one when needed. Fill them, then S1Commit(words).
    u64* S1Begin(Tegra::Engines::EngineInterface* engine, size_t words) {
        s1_cbuf = nullptr;
        // The open classified run's reserved words (S1Reserve): a pointer bump.
        if (engine == front_engine && static_cast<size_t>(s1_limit - s1_cursor) >= words)
            [[likely]] {
            u64* const entry = s1_cursor;
            s1_cursor += words;
            ir_run_words += words;
            return entry;
        }
        return S1BeginSlow(engine, words);
    }
    void S1Commit(size_t words) {
        if (s1_open_words != 0) [[unlikely]] {
            const size_t header = std::exchange(s1_open_words, size_t{0});
            Commit(header);
        }
        // The publish cadence counts payload words in batches (a publish closes the run, so
        // nothing written is ever left out of it).
        s1_unadvanced += words;
        if (s1_unadvanced >= S1AdvanceWords) [[unlikely]] {
            Advance(std::exchange(s1_unadvanced, size_t{0}));
        }
    }
    /// Front, S1: a one-word entry (Delta, Exec).
    void S1Word(Tegra::Engines::EngineInterface* engine, u64 word) {
        u64* const entry = S1Begin(engine, 1);
        entry[0] = word;
        S1Commit(1);
    }
    /// Front, S1: constant buffer words uploaded to `buffer` + `offset`, coalesced with the open
    /// Cbuf entry while they continue it (V2::S1 Cbuf). `slot`: const_buffer.buffer[slot] took
    /// the word (with `store`).
    void S1Cbuf(Tegra::Engines::EngineInterface* engine, bool store, u32 slot, u64 buffer,
                u32 offset, u32 word);
    /// Front, S1: inline index words for `method` (V2::S1 Inline entries).
    void S1Inline(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                  u32 count);

    /// Front, typed IR (tests, CI, v2/ir_trace.h): sends a traced record again, its payload
    /// copied to this split's arena; object references must name this split's object table.
    void ReplayIrRecord(std::span<const u64> record, std::span<const u64> payload);
    /// Whether Draw/Dispatch records carry their full bodies (group hashes, predicted draws):
    /// only when something reads them, EDEN_NX2_V2_PARALLEL or IR_DRAWS=verify. Otherwise they
    /// are V2::Ir::CompactBody records.
    [[nodiscard]] bool FullDrawRecords() const {
        return full_draw_records;
    }

    // Pipeline v2 P2 (fixed at each start): EDEN_NX2_V2_ACQUIRE, EDEN_NX2_V2_HAZARD.
    /// Whether the front records the guest memory writes it sends (either switch on).
    [[nodiscard]] bool TracksWrites() const {
        return tracks_writes;
    }
    [[nodiscard]] V2::Mode AcquireMode() const {
        return acquire_mode;
    }
    [[nodiscard]] V2::Mode HazardMode() const {
        return hazard_mode;
    }
    /// Front: the back's consumed position now.
    [[nodiscard]] u64 ConsumedNow() const {
        return consumed.load(std::memory_order_acquire);
    }
    /// Front, TracksWrites: the write just sent (its record ends at the write position) stores
    /// `size` known bytes, or unknown bytes, at a device address; or a range it can't bound.
    /// An open StateRun stays open: its record is already in the ring, so the write position is
    /// its end, the back stores `consumed` only after the whole run and tags every method of it
    /// with that end (Landing::tag). Waiting for the position covers the write (and at most the
    /// rest of the run). Only writes that can land at a fence cut the run, so that each one has
    /// its own landing position (NoteUnlandedWrite retires by landing).
    void NoteKnownWrite(u64 device_address, const void* bytes, u32 size, V2::Producer producer) {
        FlushCbWrite();
        CutForLanding(producer);
        PrepareHazardNote();
        hazards.NoteKnown(device_address, bytes, size, write_position, producer);
    }
    void NoteRangeWrite(u64 device_address, u64 size, V2::Producer producer) {
        if (producer != V2::Producer::CbData) {
            FlushCbWrite();
        }
        CutForLanding(producer);
        PrepareHazardNote();
        hazards.NoteRange(device_address, size, write_position, producer);
    }
    void NoteUnboundedWrite(V2::Producer producer) {
        hazards.NoteUnbounded(producer, write_position);
    }
    /// Front, TracksWrites: constant buffer words just sent to a GPU range. Words that continue
    /// the pending range on its GPU page join it (no translation, no table entry per word);
    /// the range is noted before any lookup, any other write note and a map (FlushCbWrite).
    void NoteCbWrite(const Tegra::MemoryManager& memory, u64 gpu_address, u64 size) {
        if (gpu_address == pending_cb.end && &memory == pending_cb.memory && size != 0 &&
            ((gpu_address + size - 1) >> 12) == (pending_cb.begin >> 12)) [[likely]] {
            pending_cb.end += size;
            return;
        }
        FlushCbWrite();
        pending_cb = {&memory, gpu_address, gpu_address + size};
    }
    void FlushCbWrite() {
        if (pending_cb.memory != nullptr) [[unlikely]] {
            FlushCbWriteSlow();
        }
    }
    /// Front: the draw just sent writes these device ranges (render and depth targets).
    void NoteDrawTargets(std::span<const std::pair<u64, u64>> ranges) {
        hazards.NoteDrawTargets(ranges, write_position);
    }
    /// Front: the queued writes to a device range that `filter` orders against (V2::HazardTable).
    [[nodiscard]] V2::HazardResult FindWrites(u64 device_address, u64 size,
                                              const V2::HazardFilter& filter, u8* out = nullptr) {
        FlushCbWrite();
        const u64 done = ConsumedNow();
        hazards.Retire(done);
        return hazards.Find(device_address, size, done, filter, out);
    }
    [[nodiscard]] u64 PendingUnboundedWrite(V2::Producer producer) const {
        return hazards.PendingUnbounded(producer, ConsumedNow());
    }
    /// Front: waits until the back consumed `position` (a HazardResult's). Returns whether any
    /// of it was still queued.
    bool WaitForPosition(u64 position, DrainSite site) {
        return WaitFor(std::min(position, write_position), site);
    }
    /// Front: moves at every map drain (the front's GPU→device translations may change).
    [[nodiscard]] u32 MapEpoch() const {
        return map_epoch;
    }
    /// Front: the newest compute/copy work sent before the last host-level sync (NoteSync).
    [[nodiscard]] u64 SyncedWriterPosition() const {
        return std::min(last_writer_position, last_sync_position);
    }
    /// Front: the compute/copy work sent so far (WaitForWriters' target).
    [[nodiscard]] u64 WriterPosition() const {
        return std::min(last_writer_position, write_position);
    }
    /// Front: the last host-level sync sent (NoteSync).
    [[nodiscard]] u64 SyncPosition() const {
        return std::min(last_sync_position, write_position);
    }
    [[nodiscard]] bool MayHoldReports(u64 device_address, u64 size) const {
        return hazards.MayHoldReports(device_address, size);
    }
    /// Front: a fence-flagged release of known bytes was just sent. It is in memory only once
    /// its fence ran (v2/landing.h); until then FindUnlanded forwards it.
    void NoteUnlandedWrite(u64 device_address, const void* bytes, u32 size) {
        IrCut();
        if (unlanded.Size() == V2::HazardTable::Capacity) [[unlikely]] {
            unlanded.Retire(V2::Landing::landed.load(std::memory_order_acquire));
        }
        unlanded.NoteKnown(device_address, bytes, size, write_position, V2::Producer::Release);
    }
    /// Front: a report whose bytes the front can't tell (a counter, a timestamp) was just sent;
    /// the query cache writes it when its operation runs (v2/landing.h). Until it lands,
    /// FindUnlanded answers Wait for its range: reads of it take the back's flushes, reads of
    /// the rest of its page don't (no sticky report pages).
    void NoteUnlandedRange(u64 device_address, u64 size) {
        if (unlanded.Size() == V2::HazardTable::Capacity) [[unlikely]] {
            unlanded.Retire(V2::Landing::landed.load(std::memory_order_acquire));
        }
        unlanded.NoteRange(device_address, size, write_position, V2::Producer::Report);
    }
    /// Front: the fence-flagged releases to a device range that are not in memory yet, queued
    /// or run by the back (Forward applies their bytes over `out`).
    [[nodiscard]] V2::HazardResult FindUnlanded(u64 device_address, u64 size, u8* out) {
        const u64 landed = V2::Landing::landed.load(std::memory_order_acquire);
        unlanded.Retire(landed);
        return unlanded.Find(device_address, size, landed, V2::HazardFilter{}, out);
    }
    /// Any thread, EDEN_NX2_V2_CPULANE: has the back run `hook` after its current entry (or at
    /// once if it waits for work).
    /// EDEN_NX2_V2_XFERQ: run on the back at every host-level sync position (NoteSync). Set by
    /// the renderer before the split starts.
    static void SetTransferJoinHook(void (*hook)(void*), void* argument) {
        transfer_join_hook = hook;
        transfer_join_argument = argument;
    }
    void SetCpuLaneHook(void (*hook)(void*), void* argument) {
        cpu_lane_hook = hook;
        cpu_lane_argument = argument;
    }
    /// Serving a lane request runs no idle hook (no deferred submit): a request that needs the
    /// GPU submits or waits for its own tick (WaitSubmitted asks for the idle hook).
    void RequestCpuLane() {
        cpu_lane_requested.store(true, std::memory_order_seq_cst);
        WakeBackForHook();
    }

    /// Whether the calling thread is the back.
    [[nodiscard]] static bool OnBackThread() {
        return on_back_thread;
    }
    /// Back thread: whether the front published more work past the batch the back is running,
    /// i.e. the back is the bottleneck right now (Perf fix C: the adaptive submission's idle
    /// and fast-flush rules stay off then).
    [[nodiscard]] bool BackBehind() const {
        return published.load(std::memory_order_relaxed) != back_batch_end;
    }
    /// Called on the back before it sleeps waiting for work (the rasterizer submits deferred work).
    void SetIdleHook(void (*hook)(void*), void* argument) {
        idle_hook = hook;
        idle_argument = argument;
    }
    /// Any thread: whether the back thread runs (false before it starts and once it stopped).
    [[nodiscard]] bool BackRunning() const {
        return back_running.load(std::memory_order_acquire);
    }
    /// Any thread: has the back call the idle hook soon, between entries or at once if it sleeps.
    void RequestIdleHook() {
        idle_hook_requested.store(true, std::memory_order_seq_cst);
        WakeBackForHook();
    }

    /// Starts the back thread once the split is wanted (EDEN_NX2_SPLIT=1, a Switch 2 title,
    /// commands read without cache flushes). Called by the front between commands.
    void MaybeStart(Core::System& system);
    void Stop();

    /// An engine method write, executed as DmaPusher would (method sink, ConsumeSink, call).
    /// `segment`: the engine's current_dma_segment for the call, when the engine reads it.
    void Method(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument, bool is_last,
                const GPUVAddr* segment) {
        if (ir) {
            IrMethod(engine,
                     static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                         (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                         (u64{argument} << 32),
                     segment);
            return;
        }
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        u64* const entry = Reserve(words);
        entry[0] = static_cast<u64>(Kind::Method) | (is_last ? IsLast : 0) |
                   (segment != nullptr ? HasSegment : 0) | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        if (segment != nullptr) {
            entry[1] = *segment;
        }
        Commit(words);
    }
    void MultiMethod(Tegra::Engines::EngineInterface* engine, u32 method, const u32* words,
                     u32 amount, u32 methods_pending, const GPUVAddr* segment);
    /// A write made by a macro: CallMethod without the method sink, as the engines' macros do.
    void MacroWrite(Tegra::Engines::EngineInterface* engine, u32 method, u32 argument) {
        if (ir) {
            IrMethod(engine,
                     static_cast<u64>(Kind::Method) | IsLast | Direct |
                         (u64{method & 0xffff} << 8) | (u64{argument} << 32),
                     nullptr);
            return;
        }
        SelectEngine(engine);
        u64* const entry = Reserve(1);
        entry[0] = static_cast<u64>(Kind::Method) | IsLast | Direct | (u64{method & 0xffff} << 8) |
                   (u64{argument} << 32);
        Commit(1);
    }

    /// Defer flag: the call doesn't end fast draw reuse (nx2_fast_draw.h).
    static constexpr u64 FastDrawNeutral = 1 << 7;

    /// Runs `func` on the back after everything sent before it. Must be trivially copyable.
    /// `flags`: FastDrawNeutral when `func` can't change what a draw resolves to. With the typed
    /// IR this is a Callback record: only debug hooks use it there (typed records: Emit).
    template <typename Func>
    void Defer(const Func& func, u64 flags = 0) {
        static_assert(std::is_trivially_copyable_v<Func> && alignof(Func) <= alignof(u64));
        constexpr size_t payload = (sizeof(Func) + sizeof(u64) - 1) / sizeof(u64);
        if (ir) {
            u64* const span = IrPayloadBegin(V2::Ir::Kind::Callback, nullptr, 0, 1 + payload,
                                             V2::Ir::HasThunk | (flags & FastDrawNeutral));
            span[0] = reinterpret_cast<u64>(&CallThunk<Func>);
            std::memcpy(span + 1, &func, sizeof(Func));
            IrPayloadCommit();
            return;
        }
        u64* const entry = Reserve(2 + payload);
        entry[0] = static_cast<u64>(Kind::Call) | (flags & FastDrawNeutral) | (u64{payload} << 8);
        entry[1] = reinterpret_cast<u64>(&CallThunk<Func>);
        std::memcpy(entry + 2, &func, sizeof(Func));
        Commit(2 + payload);
    }
    /// IR: the object table index of `object` for a record body (NoObject without the IR).
    [[nodiscard]] V2::Ir::ObjectRef Ref(const void* object) {
        return ir && !on_back_thread ? IrRef(object) : V2::Ir::NoObject;
    }
    /// A typed record the back executes itself (V2::IrExec), objects named with Ref. With the
    /// word stream, or when a Ref found the object table full, `legacy` (what the record does,
    /// as a trivially copyable callable) runs on the back as a Defer instead.
    template <typename Body, typename Legacy>
    void Emit(V2::Ir::Kind kind, const Body& body, const Legacy& legacy, u64 flags = 0) {
        static_assert(V2::Ir::IsBody<Body>);
        if (!ir || ir_ref_failed) {
            ir_ref_failed = false;
            Defer(legacy, flags);
            return;
        }
        constexpr size_t words = V2::Ir::HeaderWords + V2::Ir::BodyWordsOf<Body>;
        u64* const record =
            IrBegin(kind, V2::Ir::BodyWordsOf<Body>, words, flags & FastDrawNeutral);
        std::memcpy(record + V2::Ir::HeaderWords, &body, sizeof(Body));
        Commit(words);
        if constexpr (std::is_same_v<Body, V2::Ir::BarrierBody>) {
            if (parallel != nullptr) [[unlikely]] {
                ParallelBarrier(static_cast<u32>(body.op));
            }
        }
    }
    /// As Emit, with `payload_words` of variable data (to the payload arena).
    template <typename Body, typename Legacy>
    void EmitPayload(V2::Ir::Kind kind, const Body& body, const u64* payload,
                     size_t payload_words, const Legacy& legacy, u64 flags = 0) {
        static_assert(V2::Ir::IsBody<Body>);
        if (!ir || ir_ref_failed || payload_words == 0) {
            ir_ref_failed = false;
            Defer(legacy, flags);
            return;
        }
        u64* const span = IrPayloadBegin(kind, &body, V2::Ir::BodyWordsOf<Body>, payload_words,
                                         flags & FastDrawNeutral);
        std::memcpy(span, payload, payload_words * sizeof(u64));
        IrPayloadCommit();
    }
    /// IR only: a record that later stages and checks read; the back runs nothing for it and
    /// it doesn't end fast draw reuse.
    template <typename Body>
    void Note(V2::Ir::Kind kind, const Body& body) {
        static_assert(V2::Ir::IsBody<Body>);
        if (!ir) {
            return;
        }
        ir_ref_failed = false;
        constexpr size_t words = V2::Ir::HeaderWords + V2::Ir::BodyWordsOf<Body>;
        u64* const record = IrBegin(kind, V2::Ir::BodyWordsOf<Body>, words, FastDrawNeutral);
        std::memcpy(record + V2::Ir::HeaderWords, &body, sizeof(Body));
        Commit(words);
    }
    /// Run for typed records: Emit, then wait for the back to run it.
    template <typename Body, typename Legacy>
    void RunTyped(V2::Ir::Kind kind, const Body& body, const Legacy& legacy,
                  DrainSite site = DrainSite::Other) {
        if (on_back_thread || !active) {
            legacy();
            return;
        }
        Emit(kind, body, legacy);
        Drain(site);
    }
    /// RunTyped with a payload.
    template <typename Body, typename Legacy>
    void RunTypedPayload(V2::Ir::Kind kind, const Body& body, const u64* payload,
                         size_t payload_words, const Legacy& legacy, DrainSite site) {
        if (on_back_thread || !active) {
            legacy();
            return;
        }
        EmitPayload(kind, body, payload, payload_words, legacy);
        Drain(site);
    }
    /// Runs `func` on the back and waits for it: for work on back-owned state, such as a buffer
    /// cache download that records into the Vulkan scheduler. A drained back still submits from
    /// its idle hook, so the front never touches that state itself.
    template <typename Func>
    void Run(const Func& func, DrainSite site = DrainSite::Other) {
        if (on_back_thread || !active) {
            func();
            return;
        }
        Defer(func);
        Drain(site);
    }

    /// EDEN_NX2_SPLIT_CHECK=1 (debug): reports, with its stack, a thread other than the back
    /// using back-owned state (the Vulkan scheduler) while the back runs entries.
    static inline bool check_owner{};
    static void CheckOwner(const char* where);

    /// Makes what was sent visible to the back.
    void Publish();
    /// Waits until the back executed everything sent so far.
    void Drain(DrainSite site = DrainSite::Other);

    /// Compute or copy work was just sent: what can write GPU-written command lists.
    void NoteWriter() {
        last_writer_position = write_position;
    }
    /// Waits until the back executed the compute and copy work sent so far (not what came after).
    /// Returns whether any of it was still queued.
    bool WaitForWriters();
    /// A host-level wait was just sent (a syncpoint acquire, a host WFI, a GP entry with its sync
    /// bit; semaphore acquires drain): the guest's GPU work before it is complete before the
    /// commands after it are fetched.
    void NoteSync() {
        last_sync_position = write_position;
        if (transfer_join_hook != nullptr) [[unlikely]] {
            // EDEN_NX2_V2_XFERQ (MASTER_PLAN §6, multi-queue caveat): class B ordered work by the
            // one in-order host queue; with a transfer queue the back turns this position into a
            // timeline join (later transfer work waits for the graphics work before it).
            struct Join {
                void (*hook)(void*);
                void* argument;
                void operator()() const {
                    hook(argument);
                }
            };
            Defer(Join{transfer_join_hook, transfer_join_argument});
        }
        if (tracks_writes) [[unlikely]] {
            hazards.NoteSync(write_position);
        }
    }
    /// Waits until the back executed the work sent before the last NoteSync. Returns whether any
    /// of it was still queued.
    bool WaitForSync();
    /// Waits until the back executed everything up to `ticket` (Mark). Returns whether any of
    /// it was still queued.
    bool WaitFor(const V2::Ir::Ticket& ticket, DrainSite site) {
        return WaitFor(std::min(ticket.position, write_position), site);
    }

private:
    friend struct GpuSplitTestAccess;
    friend class V2::Resolve::SplitResolve;
    friend struct V2::Resolve::SplitTraits;

    enum class Kind : u64 {
        Wrap,   ///< The rest of the ring is unused: continue at its start
        Engine, ///< The engine the next methods go to: word 1
        Method, ///< Bits 8-23 method, 32-63 argument; flags IsLast, HasSegment (word 1)
        Multi,  ///< Bits 8-23 method, 32-63 amount; word 1 methods pending, [segment], words
        Call,   ///< Bits 8-31 payload size in words; word 1 thunk, then the payload
        Ir,     ///< A typed IR record (v2/ir.h, EDEN_NX2_V2_IR)
    };
    static_assert(static_cast<u64>(Kind::Ir) == V2::Ir::RingKind);
    static constexpr u64 IsLast = 1 << 4;
    static constexpr u64 HasSegment = 1 << 5;
    static constexpr u64 Direct = 1 << 6; ///< Method: call it even if not executable (macros)

    static constexpr size_t RingWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t RingMask = RingWords - 1;
    static constexpr u32 MaxMultiWords = 8192;
    static constexpr size_t PublishEvery = 256;
    /// Words a back waiting for work gets without a full publish_every (Commit).
    static constexpr size_t PublishAfterWait = 8;
    /// Most words in one IR StateRun (each publish closes the open run earlier).
    static constexpr size_t MaxRunWords = 1024;
    static constexpr size_t ArenaWords = size_t{1} << 20; // 8 MiB
    static constexpr size_t ArenaMarks = size_t{1} << 16;

    template <typename Func>
    static void CallThunk(const u64* payload) {
        (*reinterpret_cast<const Func*>(payload))();
    }

    /// Room for an entry of `words` at the write position (never across the ring's end).
    u64* Reserve(size_t words) {
        const size_t index = write_position & RingMask;
        if (index + words <= RingWords && write_position + words - consumed_cache <= RingWords)
            [[likely]] {
            return &ring[index];
        }
        return ReserveSlow(words);
    }
    u64* ReserveSlow(size_t words);
    void Commit(size_t words) {
        write_position += words;
        Advance(words);
    }
    /// `words` more were written (ring or payload arena): publishes every publish_every.
    void Advance(size_t words) {
        unpublished_words += words;
        sent_words += words;
        if (unpublished_words >= publish_every) {
            Publish();
            return;
        }
        // EDEN_NX2_V2_ACQUIRE: the back waits for work at the last position published: it gets
        // these words after a few (drains used to restart it like that; acquires no longer drain).
        if (feed_idle_back && unpublished_words >= PublishAfterWait &&
            back_idle_at.load(std::memory_order_relaxed) == published_local) [[unlikely]] {
            Publish();
        }
    }
    void FlushCbWriteSlow();
    void PrepareHazardNote() {
        if (hazards.Size() == V2::HazardTable::Capacity) [[unlikely]] {
            hazards.Retire(consumed.load(std::memory_order_acquire));
        }
    }
    void WaitForSpace(size_t words);
    void SelectEngine(Tegra::Engines::EngineInterface* engine) {
        if (engine != front_engine) [[unlikely]] {
            SwitchEngine(engine);
        }
    }
    void SwitchEngine(Tegra::Engines::EngineInterface* engine);
    bool WaitFor(u64 target, DrainSite site);

    // Typed IR (front). A StateRun stays open while method writes to one engine follow each
    // other: its record (header + payload reference) is in the ring, its entries go to the
    // payload arena, and the reference gets their count when it closes, which happens before
    // any other record and before every publish (the back never sees an open run).
    void IrMethod(Tegra::Engines::EngineInterface* engine, u64 word, const GPUVAddr* segment) {
        SelectEngine(engine);
        const size_t words = segment != nullptr ? 2 : 1;
        s1_cbuf = nullptr;
        if (ir_run != nullptr && !ir_run_classified && ir_run_words + words <= MaxRunWords)
            [[likely]] {
            if (u64* const entry = arena.Extend(words)) [[likely]] {
                entry[0] = word;
                if (segment != nullptr) {
                    entry[1] = *segment;
                }
                ir_run_words += words;
                Advance(words);
                return;
            }
        }
        IrOpenRun(word, segment);
    }
    void IrOpenRun(u64 word, const GPUVAddr* segment);
    void IrCloseRun() {
        ir_run[V2::Ir::HeaderWords] = V2::Ir::PackRef(ir_run_index, static_cast<u32>(ir_run_words));
        ir_run = nullptr;
        ir_run_classified = false;
        s1_cbuf = nullptr;
        s1_cursor = nullptr;
        s1_limit = nullptr;
    }
    u64* S1OpenRun(size_t words);
    u64* S1BeginSlow(Tegra::Engines::EngineInterface* engine, size_t words);
    /// S1Exec: reserves arena words after the open classified run for the next entries.
    void S1Reserve();
    static constexpr size_t S1ReserveWords = 64;
    static constexpr size_t S1AdvanceWords = 16;
    /// Closes the open StateRun (P2 notes a write at the position after it: the next method
    /// starts a new record, so every noted write ends its record).
    void IrCut() {
        if (ir_run != nullptr) [[unlikely]] {
            IrCloseRun();
        }
    }
    /// Releases and reports may land only when their fence runs (v2/landing.h): the methods
    /// after one start a new record, so a later landing never retires it early.
    void CutForLanding(V2::Producer producer) {
        if (producer == V2::Producer::Release || producer == V2::Producer::Report) {
            IrCut();
        }
    }
    /// Closes the open run and reserves a record of `words` with its header and sequence
    /// number written. The caller fills in the rest and commits `words`.
    u64* IrBegin(V2::Ir::Kind kind, size_t body_words, size_t words, u64 flags) {
        if (ir_run != nullptr) {
            IrCloseRun();
        }
        if (parallel != nullptr) [[unlikely]] {
            ParallelRecord(kind);
        }
        u64* const record = Reserve(words);
        record[0] = V2::Ir::MakeHeader(kind, body_words, words, flags);
        record[1] = ++ir_sequence;
        return record;
    }
    /// A record of `kind` with `body` (body_words, may be null) and a payload of
    /// `payload_words` in the arena; returns the payload span to fill in before
    /// IrPayloadCommit makes both visible.
    u64* IrPayloadBegin(V2::Ir::Kind kind, const void* body, size_t body_words,
                        size_t payload_words, u64 flags);
    void IrPayloadCommit() {
        Commit(ir_pending_words);
        Advance(ir_pending_payload);
    }
    /// Room in the payload arena for the record `sequence` ending at ring `position`; retires
    /// what the back executed and waits for it when the arena is full.
    u64* ArenaBegin(size_t words, u64 sequence, u64 position);
    /// The object table index of `object` (Engine records).
    V2::Ir::ObjectRef IrRef(const void* object);

    /// One step of a front wait for the back to consume `target`.
    void Backoff(u32 spins, u64 target);
    /// EDEN_NX2_V2_SPIN: whether a wait that started spinning at `start` (ns) still spins.
    [[nodiscard]] bool SpinWindowOpen(u32 spins, u64& start) const;
    void WakeBackForHook() {
        hook_requested.store(true, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }
    /// Back, hook_requested seen: serves the CPU lane, and runs the idle hook only if it was
    /// asked for (RequestIdleHook), not for a lane request alone.
    void ServeHookRequests() {
        hook_requested.store(false, std::memory_order_seq_cst);
        ServeCpuLane();
        if (idle_hook_requested.load(std::memory_order_relaxed) &&
            idle_hook_requested.exchange(false, std::memory_order_seq_cst) &&
            idle_hook != nullptr) {
            idle_hook(idle_argument);
        }
    }
    /// Back: runs the CPU lane hook if one was asked for (EDEN_NX2_V2_CPULANE).
    void ServeCpuLane() {
        if (cpu_lane_requested.load(std::memory_order_relaxed) &&
            cpu_lane_requested.exchange(false, std::memory_order_seq_cst) &&
            cpu_lane_hook != nullptr) {
            cpu_lane_hook(cpu_lane_argument);
        }
    }
    /// After the back stored `consumed`: wakes a front sleeping in Backoff.
    void NotifyFront() {
        if (front_waiting.load(std::memory_order_seq_cst)) {
            front_wake.fetch_add(1, std::memory_order_seq_cst);
            front_wake.notify_one();
        }
    }

    void BackLoop(std::stop_token stop_token);
    /// The size of the entry at `entry` in ring words (0: wrap).
    static size_t EntryWords(const u64* entry);
    /// Runs the entry at `entry` (of the `words` published before the ring's end), returns its size.
    size_t Execute(const u64* entry, size_t words);
    /// One method entry (the old Method entry, or one in an IR StateRun), returns its size.
    size_t ExecuteMethod(const u64* entry, size_t words);
    /// ExecuteMethod for a back engine with fast_maxwell3d (chosen once per StateRun).
    size_t ExecuteMethod3D(Tegra::Engines::EngineInterface* engine, const u64* entry,
                           size_t words);
    size_t ExecuteIr(const u64* entry);
    void CheckIrRecord(const u64* entry);
    void AllocateIr();
    // EDEN_NX2_V2_PARALLEL hooks (out of line: SplitResolve is incomplete here).
    void ParallelRecord(V2::Ir::Kind kind);
    void ParallelBarrier(u32 op);
    void ParallelPublish();
    void StartParallel();

    Core::System* system{};
    bool active{};
    u32 starts{};
    bool front_macros{};
    bool ir{};        ///< EDEN_NX2_V2_IR on at this start
    bool ir_verify{}; ///< EDEN_NX2_V2_IR=verify: the back checks every record header
    bool draw_records{}; ///< EDEN_NX2_V2_IR_DRAWS (DrawRecords)
    bool draw_verify{};  ///< EDEN_NX2_V2_IR_DRAWS=verify: the back checks draws (V2::IrCheck)
    bool full_draw_records{}; ///< FullDrawRecords
    bool ir_checks{}; ///< ir_verify, draw_verify or V2 stats: ExecuteIr's per-record checks
    bool s1_exec{};   ///< S1Exec
    bool s1_check{};  ///< S1Check
    /// EDEN_NX2_V2_IR_TRACE: the back's record of the frame it runs (null when off).
    std::unique_ptr<V2::IrTrace::Recorder> ir_trace;
    u64 ir_trace_frame{};
    void TraceIrRecord(const u64* entry);
    std::unique_ptr<u64[]> ring;
    V2::Resolve::SplitResolve* parallel{}; ///< EDEN_NX2_V2_PARALLEL (owned; deleted at Stop)

    // Front side.
    u64 write_position{};
    u64 last_writer_position{};
    u64 last_sync_position{};
    u64 published_local{};
    u64 consumed_cache{};
    /// Words between publishes: small right after a drain, while the back has nothing to do.
    size_t publish_every{PublishEvery};
    Tegra::Engines::EngineInterface* front_engine{};
    u64 ir_sequence{};
    u64* ir_run{}; ///< Header of the open StateRun
    size_t ir_run_words{};
    u32 ir_run_index{}; ///< Arena index of the open StateRun's entries
    bool ir_run_classified{}; ///< The open StateRun is an Ir::Classified one
    size_t s1_open_words{};   ///< S1OpenRun's record, committed by S1Commit
    u64* s1_cbuf{}; ///< The open run's last entry when it is a Cbuf entry S1Cbuf may extend
    u64* s1_cursor{}; ///< S1Reserve's words still free after the open classified run
    u64* s1_limit{};
    size_t s1_unadvanced{}; ///< S1 payload words not yet counted by Advance
    size_t ir_pending_words{};   ///< IrPayloadBegin's record, until IrPayloadCommit
    size_t ir_pending_payload{};
    bool ir_ref_failed{};        ///< IrRef found the object table full
    size_t unpublished_words{}; ///< Ring and payload words since the last publish
    u64 sent_words{};           ///< Ring and payload words sent since the start
    V2::Ir::PayloadArena arena;  ///< Variable record data (EDEN_NX2_V2_IR)
    std::unique_ptr<V2::Ir::ObjectTable> objects; ///< Object references (EDEN_NX2_V2_IR)
    // Pipeline v2 P2 (fixed at each start).
    bool tracks_writes{};
    bool feed_idle_back{};
    bool spin_window{};
    V2::Mode acquire_mode{};
    V2::Mode hazard_mode{};
    u64 spin_window_ns{};
    u64 idle_hook_ns{}; ///< EDEN_NX2_V2_SPIN: the back's idle hook after this long idle
    struct PendingCb {
        const Tegra::MemoryManager* memory;
        u64 begin;
        u64 end;
    };
    PendingCb pending_cb{}; ///< NoteCbWrite
    u32 map_epoch{};        ///< MapEpoch
    u64 front_spin_start{};
    V2::HazardTable hazards;
    V2::HazardTable unlanded; ///< Fence-flagged known releases until they land (Landing)
    u64 back_entry_position{}; ///< Back: ring position of the entry it runs (TracksWrites)

    // Back side.
    Tegra::Engines::EngineInterface* back_engine{};
    u64 back_sequence{};
    u64 ir_bad_records{};

    static GpuSplit instance;
    static inline thread_local bool on_back_thread{};
    void (*idle_hook)(void*){};
    void* idle_argument{};
    void (*cpu_lane_hook)(void*){};
    void* cpu_lane_argument{};
    std::atomic<bool> cpu_lane_requested{};
    alignas(64) std::atomic<bool> hook_requested{};
    std::atomic<bool> idle_hook_requested{};
    std::atomic<bool> back_running{};
    u64 back_batch_end{}; ///< Back thread: the published position of the batch it runs

    // 128 bytes: x86 spatial prefetchers pull 64-byte lines in pairs. back_waiting and wake are
    // always touched together by both threads, so they share a block.
    alignas(128) std::atomic<u64> published{};
    alignas(128) std::atomic<u64> consumed{};
    std::atomic<u64> consumed_sequence{}; ///< IR sequence of the last record before `consumed`

    /// A position the front waits for (WaitFor): the back stores `consumed` once past it.
    alignas(128) std::atomic<u64> wait_target{~u64{0}};
    alignas(128) std::atomic<bool> back_waiting{};
    std::atomic<u32> wake{};
    /// The front sleeps in Backoff (#161: instead of yielding).
    alignas(128) std::atomic<bool> front_waiting{};
    std::atomic<u32> front_wake{};
    /// The back's position when it last ran out of work (EDEN_NX2_V2_ACQUIRE, Commit). Port of
    /// the publish part of 62db242b34 (jpkto, PR #805) via a20471e48c.
    alignas(128) std::atomic<u64> back_idle_at{~u64{0}};
    std::jthread back_thread;
    static inline void (*transfer_join_hook)(void*) = nullptr;
    static inline void* transfer_join_argument = nullptr;
};

} // namespace VideoCommon
