// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <memory>
#include <string>

#include <fmt/format.h>

#include "common/logging.h"
#include "video_core/v2/cbuf_ring.h"
#include "video_core/v2/p3_stats.h"

namespace VideoCommon::V2::P3Stats {

namespace {
thread_local std::unique_ptr<Data> t_data;

double PerSecond(u64 value, u64 window_ns) {
    return window_ns == 0 ? 0.0 : static_cast<double>(value) * 1e9 / static_cast<double>(window_ns);
}

double Percent(u64 part, u64 whole) {
    return whole == 0 ? 0.0 : static_cast<double>(part) * 100.0 / static_cast<double>(whole);
}

double Micros(u64 ns, u64 count) {
    return count == 0 ? 0.0 : static_cast<double>(ns) / 1e3 / static_cast<double>(count);
}
} // Anonymous namespace

Data& Get() {
    if (!t_data) [[unlikely]] {
        t_data = std::make_unique<Data>();
    }
    return *t_data;
}

void Report(const char* thread_name, u64 window_ns) {
    if (!t_data) {
        return;
    }
    Data& d = *t_data;
    d.desc_set_hits += g_worker_set_hits.exchange(0, std::memory_order_relaxed);
    if (d.key_lookups != 0) {
        LOG_INFO(HW_GPU,
                 "V2 stats {} P3 pipeline key: {:.0f} lookups/s, unchanged {:.1f}%, {:.2f} blocks "
                 "re-packed per lookup, same pipeline {:.1f}%, variant table hits {:.1f}%, slow "
                 "path {:.1f}% | verify {} checked, {} mismatches",
                 thread_name, PerSecond(d.key_lookups, window_ns),
                 Percent(d.key_unchanged, d.key_lookups),
                 d.key_lookups == 0 ? 0.0
                                    : static_cast<double>(d.key_blocks) /
                                          static_cast<double>(d.key_lookups),
                 Percent(d.key_same, d.key_lookups), Percent(d.key_variant_hits, d.key_lookups),
                 Percent(d.key_slow, d.key_lookups), d.key_verify, d.key_mismatches);
    }
    if (d.relevance_draws != 0) {
        std::string groups;
        for (size_t g = 0; g < NumRegGroups; ++g) {
            groups += fmt::format(" {} {:.1f}/{:.1f},", reg_group_names[g],
                                  Percent(d.changed[g], d.relevance_draws),
                                  Percent(d.relevant[g], d.relevance_draws));
        }
        groups.pop_back();
        LOG_INFO(HW_GPU,
                 "V2 stats {} P3 groups changed/relevant to the bound shaders (% of resolved "
                 "draws):{}",
                 thread_name, groups);
    }
    if (d.cont_candidates != 0 || d.full_draws != 0) {
        LOG_INFO(HW_GPU,
                 "V2 stats {} P3 continued draws: {:.1f}% of {:.0f} candidates/s ({:.1f}% only "
                 "register changes the old serial resolved in full, {:.0f}/s after a submit) | "
                 "full: heavy group {:.0f}/s, other pipeline {:.0f}/s, event {:.0f}/s | prepare "
                 "us per draw: full {:.2f}, continued {:.2f} | verify {} checked, {} mismatches",
                 thread_name, Percent(d.cont_taken, d.cont_candidates),
                 PerSecond(d.cont_candidates, window_ns),
                 Percent(d.cont_register_only, d.cont_taken),
                 PerSecond(d.cont_across_submit, window_ns), PerSecond(d.cont_heavy, window_ns),
                 PerSecond(d.cont_pipeline, window_ns), PerSecond(d.cont_events, window_ns),
                 Micros(d.full_ns, d.full_draws), Micros(d.cont_ns, d.cont_draws), d.cont_verify,
                 d.cont_mismatches);
    }
    if (d.desc_draws != 0) {
        LOG_INFO(HW_GPU,
                 "V2 stats {} P3 descriptor cache: {:.0f} draws/s, pushes skipped {:.1f}%, set "
                 "hits {:.1f}%, descriptor buffer hits {:.1f}% | verify {} checked, {} mismatches",
                 thread_name, PerSecond(d.desc_draws, window_ns),
                 Percent(d.desc_push_skipped, d.desc_draws), Percent(d.desc_set_hits, d.desc_draws),
                 Percent(d.desc_buffer_hits, d.desc_draws), d.desc_verify, d.desc_mismatches);
    }
    if (d.cbuf_binds != 0) {
        LOG_INFO(HW_GPU,
                 "V2 stats {} P3 constant buffer versions: {:.0f} small binds/s, version hits "
                 "{:.1f}%, copies {:.0f}/s, {:.0f} KiB/s not read or copied | verify {} checked, "
                 "{} mismatches",
                 thread_name, PerSecond(d.cbuf_binds, window_ns),
                 Percent(d.cbuf_hits, d.cbuf_binds), PerSecond(d.cbuf_copies, window_ns),
                 PerSecond(d.cbuf_bytes_saved, window_ns) / 1024.0, d.cbuf_verify,
                 d.cbuf_mismatches);
    }
    if (d.ring_binds != 0) {
        // The front's counters since the last line (one back thread reports them).
        auto& shared = CbufRing::Get();
        static u64 last_versions, last_guest, last_kept, last_none;
        const u64 versions = shared.versions.load(std::memory_order_relaxed);
        const u64 guest = shared.guest_bases.load(std::memory_order_relaxed);
        const u64 kept = shared.kept_bases.load(std::memory_order_relaxed);
        const u64 none = shared.no_version.load(std::memory_order_relaxed);
        LOG_INFO(HW_GPU,
                 "V2 stats {} F2-02 cbuf ring: {:.0f} small binds/s, ring hits {:.1f}% (no "
                 "version {:.1f}%, below floor {:.1f}%, bytes differ {:.1f}%) | front: versions "
                 "{:.0f}/s (from guest memory {:.0f}/s, kept {:.0f}/s), none {:.0f}/s, ring "
                 "retired {:#x} of head {:#x}",
                 thread_name, PerSecond(d.ring_binds, window_ns),
                 Percent(d.ring_hits, d.ring_binds), Percent(d.ring_none, d.ring_binds),
                 Percent(d.ring_old, d.ring_binds), Percent(d.ring_mismatches, d.ring_binds),
                 PerSecond(versions - last_versions, window_ns),
                 PerSecond(guest - last_guest, window_ns), PerSecond(kept - last_kept, window_ns),
                 PerSecond(none - last_none, window_ns), shared.ring.Retired(),
                 shared.ring.Head());
        last_versions = versions;
        last_guest = guest;
        last_kept = kept;
        last_none = none;
    }
    const u64 heap_work = d.heap_draws + d.heap_partial_draws + d.heap_fallback_draws +
                          d.heap_dispatches + d.heap_partial_dispatches +
                          d.heap_fallback_dispatches;
    if (heap_work != 0) {
        static constexpr std::array<const char*, 6> reason_names{
            "mesh emulation",      "texel buffers",     "byte-addressed images",
            "no storage image heap", "storage image type", "too many bindings",
        };
        std::string reasons;
        for (size_t reason = 0; reason < reason_names.size(); ++reason) {
            if (d.heap_reasons[reason] != 0) {
                reasons += fmt::format(" {} {:.0f}/s,", reason_names[reason],
                                       PerSecond(d.heap_reasons[reason], window_ns));
            }
        }
        if (!reasons.empty()) {
            reasons.pop_back();
        }
        LOG_INFO(HW_GPU,
                 "V2 stats {} P3 bindless heap (work with images): draws {:.0f}/s all in the heap, "
                 "{:.0f}/s partly, {:.0f}/s none; dispatches {:.0f}/s all, {:.0f}/s partly, {:.0f}/s "
                 "none | left out for:{} | {:.1f} words per heap draw, {:.0f} slot writes/s | "
                 "verify {} checked, {} mismatches",
                 thread_name, PerSecond(d.heap_draws, window_ns),
                 PerSecond(d.heap_partial_draws, window_ns),
                 PerSecond(d.heap_fallback_draws, window_ns),
                 PerSecond(d.heap_dispatches, window_ns),
                 PerSecond(d.heap_partial_dispatches, window_ns),
                 PerSecond(d.heap_fallback_dispatches, window_ns),
                 reasons.empty() ? std::string{" nothing"} : reasons,
                 heap_work == 0 ? 0.0
                                : static_cast<double>(d.heap_words) /
                                      static_cast<double>(heap_work),
                 PerSecond(d.heap_slot_writes, window_ns), d.heap_verify, d.heap_mismatches);
    }
    if (d.lean_draws != 0) {
        LOG_INFO(HW_GPU, "V2 stats {} P3 lean draws: {:.0f}/s, {:.1f}% took the debug hooks",
                 thread_name, PerSecond(d.lean_draws, window_ns),
                 Percent(d.lean_cold, d.lean_draws));
    }
    d = Data{};
}

} // namespace VideoCommon::V2::P3Stats
