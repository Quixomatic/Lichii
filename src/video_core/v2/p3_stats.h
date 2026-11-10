// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>

#include "common/common_types.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/v2_stats.h"

// Pipeline v2 P3 counters ("cheap draws"): one block per fast path, kept by the thread that
// draws and logged with the other EDEN_NX2_V2_STATS lines every 10 s ("V2 stats ... P3 ...").
// Verify mismatches are counted (and their first occurrences logged) with or without the stats.
namespace VideoCommon::V2::P3Stats {

struct Data {
    // EDEN_NX2_V2_PIPEKEY
    u64 key_lookups{};    ///< CurrentGraphicsPipeline calls on the v2 path
    u64 key_unchanged{};  ///< no block re-packed
    u64 key_blocks{};     ///< blocks re-packed (sum)
    u64 key_same{};       ///< same pipeline as the last lookup, no table lookup
    u64 key_variant_hits{};
    u64 key_slow{};       ///< graphics_cache lookup or creation
    u64 key_verify{};
    u64 key_mismatches{};
    // Relevance masking (any P3 tracker consumer): groups changed since the last resolved draw,
    // and those the bound pipeline can see.
    u64 relevance_draws{};
    std::array<u64, NumRegGroups> changed{};
    std::array<u64, NumRegGroups> relevant{};
    // EDEN_NX2_V2_CONTDRAW / MARKDIRTY
    u64 cont_candidates{}; ///< draws TakeReusableDraw looked at
    u64 cont_taken{};      ///< continued (delta) draws decided from the groups
    u64 cont_register_only{}; ///< of those, draws the old serial would have resolved in full
    u64 cont_heavy{};      ///< full: a heavy group changed
    u64 cont_pipeline{};   ///< full: the pipeline changed
    u64 cont_events{};     ///< full: a non-register break (event serial)
    u64 cont_verify{};
    u64 cont_mismatches{};
    u64 cont_across_submit{}; ///< MARKDIRTY: continued draws after a submit
    u64 full_ns{};
    u64 full_draws{};
    u64 cont_ns{};
    u64 cont_draws{};
    // EDEN_NX2_V2_DESCCACHE
    u64 desc_draws{};
    u64 desc_push_skipped{};
    u64 desc_set_hits{};
    u64 desc_buffer_hits{};
    u64 desc_verify{};
    u64 desc_mismatches{};
    // EDEN_NX2_V2_CBUFVER
    u64 cbuf_binds{};
    u64 cbuf_hits{};
    u64 cbuf_copies{};
    u64 cbuf_bytes_saved{};
    u64 cbuf_verify{};
    u64 cbuf_mismatches{};
    // EDEN_NX2_V2_CBUFRING (the back's binds; the front's counters are in CbufRing::Shared)
    u64 ring_binds{};
    u64 ring_hits{};
    u64 ring_none{};       ///< no version of the range
    u64 ring_none_absent{};  ///< ... no entry for the address (never published, or replaced)
    u64 ring_none_dropped{}; ///< ... the front dropped it (no version of its last writes)
    u64 ring_none_small{};   ///< ... the version is shorter than the bind
    u64 ring_old{};        ///< below the bind floor
    u64 ring_mismatches{}; ///< the version's bytes differ from guest memory
    // F2-03 (v2/cbuf_watch.h): how ring hits were validated
    u64 ring_trusted{};         ///< trusted, words unchanged: no compare
    u64 ring_memo{};            ///< compared earlier in this collect epoch, words unchanged
    u64 ring_compared_hits{};   ///< compared now
    u64 ring_compares{};        ///< byte compares run (verify compares every bind)
    u64 ring_trusted_differs{}; ///< verify: trusted and differs (must be 0)
    u64 ring_memo_differs{};    ///< verify: memo and differs (race class, ~0)
    // EDEN_NX2_V2_BINDLESS
    u64 heap_draws{};          ///< every image descriptor in the heap
    u64 heap_partial_draws{};  ///< some left out (heap_reasons)
    u64 heap_fallback_draws{}; ///< none in the heap
    u64 heap_dispatches{};
    u64 heap_partial_dispatches{};
    u64 heap_fallback_dispatches{};
    std::array<u64, 8> heap_reasons{}; ///< partial and fallback work by HeapReason bit
    u64 heap_slot_writes{};
    u64 heap_words{};
    u64 heap_verify{};
    u64 heap_mismatches{};
    // EDEN_NX2_V2_LEANDRAW
    u64 lean_draws{};
    u64 lean_cold{};
};

/// Descriptor set reuse happens on the VulkanWorker, which reports no stats of its own: it
/// counts here and the next report of a drawing thread takes the count.
inline std::atomic<u64> g_worker_set_hits{};

/// Counting is on with the stats (verify modes count their checks and mismatches anyway).
[[nodiscard]] inline bool Enabled() {
    return Stats::Enabled();
}

/// This thread's counters.
[[nodiscard]] Data& Get();

/// Logs and resets this thread's counters (called from Stats::ReportSlow).
void Report(const char* thread_name, u64 window_ns);

} // namespace VideoCommon::V2::P3Stats
