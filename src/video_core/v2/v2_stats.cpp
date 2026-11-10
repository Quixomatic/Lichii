// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <memory>
#include <string>

#include "common/logging.h"
#include "video_core/engines/maxwell_3d.h"
#include "video_core/gpu_split.h"
#include "video_core/nx2_profile.h"
#include "video_core/v2/p5b_stats.h"
#include "video_core/v2/regsem.h"
#include "video_core/v2/ir.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/p3_stats.h"
#include "video_core/v2/resolve/draw_record.h"
#include "video_core/v2/v2_stats.h"

#define OFF(field_name) MAXWELL3D_REG_INDEX(field_name)

namespace VideoCommon::V2::Stats {

namespace {

using Tegra::Engines::Maxwell3D;

constexpr u64 WindowNs = 10'000'000'000ULL;

constexpr size_t NumSites = static_cast<size_t>(DrainSite::Count);
constexpr size_t NumClasses = static_cast<size_t>(SyncClass::Count);

constexpr std::array<const char*, NumClasses> class_names{"A", "B", "C", "D", "?", "S"};

u64 Mix(u64 value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

u64 MixRegister(u32 method, u32 value) {
    return Mix((u64{method} << 32) | value);
}

// ---------------------------------------------------------------------------------------------
// Front: releases it sent, acquires and waits for the back.

/// The class of a wait at a site outside an AcquireContext.
SyncClass SiteClass(DrainSite site) {
    switch (site) {
    case DrainSite::Acquire:
    case DrainSite::SemaphoreFlush:
    case DrainSite::FenceRelease:
        return SyncClass::Unknown;
    case DrainSite::ListWriters:
    case DrainSite::ListSync:
    case DrainSite::ListFlush:
    case DrainSite::MacroDmaRead:
    case DrainSite::MacroDmaFlush:
    case DrainSite::GpSyncFence:
        return SyncClass::C;
    default:
        return SyncClass::Structural;
    }
}

struct ReleaseEntry {
    GPUVAddr address{};
    u64 position{}; ///< GpuSplit::WritePosition() after the release was sent
    u32 payload{};
    u32 address_space{};
    ReleaseKind kind{};
    bool valid{};
};

struct FrontData {
    std::array<std::array<u64, NumClasses>, NumSites> drain_ticks{};
    std::array<std::array<u64, NumClasses>, NumSites> drain_count{};
    std::array<u64, NumClasses> acquires{};
    std::array<u64, NumClasses> acquires_met{};
    std::array<u64, NumClasses> polls{};
    std::array<u64, NumClasses> polls_met{};
    u64 mispredicted{}; ///< Classed A or B, yet not met
    u64 literal_releases{};
    u64 gpu_valued_releases{};
    u64 table_evictions{};
    // P2 decisions.
    std::array<std::array<u64, static_cast<size_t>(Decision::Count)>, NumSites> decisions{};
    std::array<std::array<u64, static_cast<size_t>(Reason::Count)>, NumSites> reasons{};
    std::array<std::array<u64, 8>, NumSites> narrowed_by{};
    std::array<u64, NumSites> mismatches{};
    std::array<u64, NumSites> benign_mismatches{};
    // SYNC Step 0.
    std::array<u64, static_cast<size_t>(WakeClass::Count)> suspended{};
    std::array<u64, static_cast<size_t>(WakeClass::Count)> suspended_ns{};
    std::array<u64, static_cast<size_t>(ReductionReject::Count)> reduction_rejects{};
    u64 forced_landings{};
    u64 forced_landing_ns{};
    std::array<u64, 8> sem_rows{};
    u64 sem_evictions{};
};

constexpr std::array<const char*, 7> sem_row_names{
    "map queued", "unmapped", "gpu-written", "no writer", "pending known", "unknown at execute",
    "unknown at fence",
};

constexpr std::array<const char*, static_cast<size_t>(WakeClass::Count)> wake_names{
    "release parsed", "consumed", "landing", "map", "readback", "cpu",
};
constexpr std::array<const char*, static_cast<size_t>(ReductionReject::Count)> reject_names{
    "unknown qmd may write", "eviction", "kind", "partly landed", "raced",
};

constexpr std::array<const char*, static_cast<size_t>(Decision::Count)> decision_names{
    "avoided", "forwarded", "narrowed", "waited",
};
constexpr std::array<const char*, static_cast<size_t>(Reason::Count)> reason_names{
    "-",         "settings",      "operation",   "unmapped", "gpu-written", "unknown write",
    "queued not met", "not met", "report page", "overflow", "valve", "unlanded", "qmd producer", "forced landing", "reduction view",
    "sem execute 2 s pass",
};

constexpr size_t NumReleases = 4096;

struct FrontState {
    FrontData data;
    /// Direct mapped by address: an approximation (a colliding release replaces the older one).
    std::array<ReleaseEntry, NumReleases> releases{};
};

// Allocated by the threads that use them: a large thread_local array is in every thread's TLS.
thread_local std::unique_ptr<FrontState> t_front_state;
thread_local SyncClass t_context{SyncClass::Count};

FrontState& Front() {
    if (!t_front_state) [[unlikely]] {
        t_front_state = std::make_unique<FrontState>();
    }
    return *t_front_state;
}

size_t ReleaseIndex(GPUVAddr address) {
    return static_cast<size_t>(Mix(address >> 2)) & (NumReleases - 1);
}

bool Satisfies(AcquireOp op, u32 word, u32 value) {
    switch (op) {
    case AcquireOp::Equal:
        return word == value;
    case AcquireOp::Gequal:
        return word >= value;
    case AcquireOp::CircularGequal:
        return static_cast<s32>(word - value) >= 0;
    case AcquireOp::Mask:
        return (word & value) != 0;
    case AcquireOp::MaskZero:
        return (word & value) == 0;
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// Draws.

// The groups are the register semantics table's (regsem.h).
using Group = RegGroup;
constexpr size_t NumGroups = NumRegGroups;
constexpr const auto& group_names = reg_group_names;

constexpr u32 Bit(Group group) {
    return 1U << static_cast<u32>(group);
}

constexpr u32 StateGroups = Bit(Group::Pipeline) | Bit(Group::Viewport) | Bit(Group::TexPools) |
                            Bit(Group::RenderTargets) | Bit(Group::Shaders);
constexpr u32 StreamGroups = Bit(Group::VertexStreams) | Bit(Group::IndexBuffer);
constexpr u32 IgnoredGroups = Bit(Group::DrawParams) | Bit(Group::Misc);
constexpr u32 LightGroups = IgnoredGroups | Bit(Group::CbufData);
constexpr u32 LightBindGroups = LightGroups | StreamGroups | Bit(Group::CbufBind);

using GroupTable = std::array<Group, RegSem::Table3D::NumRegs>;

const GroupTable& Groups() {
    return RegSem::GetTable3D().groups;
}

/// Simulated draw cache: direct mapped, fingerprint per slot.
template <size_t Size>
struct FingerprintCache {
    std::array<u64, Size> slots{};
    u64 hits{};

    void Look(u64 fingerprint) {
        u64& slot = slots[static_cast<size_t>(fingerprint >> 7) & (Size - 1)];
        if (slot == fingerprint) {
            ++hits;
        } else {
            slot = fingerprint;
        }
    }
};

/// Run lengths: 1, 2, 3-4, 5-8, ..., 65+.
constexpr size_t NumRunBuckets = 8;
constexpr std::array<const char*, NumRunBuckets> run_bucket_names{
    "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+",
};

size_t RunBucket(u64 length) {
    if (length <= 1) {
        return 0;
    }
    return std::min<size_t>(static_cast<size_t>(std::bit_width(length - 1)), NumRunBuckets - 1);
}

struct DrawData {
    u64 draws{};
    u64 indirect{};
    std::array<u64, NumGroups> changed{};
    u64 zero_change{};
    u64 light_only{};
    u64 light_bind{};
    u64 hash_drift{};
    std::array<u64, NumRunBuckets> runs{};
    std::array<u64, NumRunBuckets> run_draws{};
    // Kept across windows (the caches simulate a cache that lives on).
    u64 run_length{};
    FingerprintCache<256> state_256;
    FingerprintCache<4096> state_4096;
    FingerprintCache<256> full_256;
    FingerprintCache<4096> full_4096;
};

thread_local std::unique_ptr<DrawData> t_draw_data;

DrawData& Draws() {
    if (!t_draw_data) [[unlikely]] {
        t_draw_data = std::make_unique<DrawData>();
    }
    return *t_draw_data;
}
thread_local bool t_in_draw{};

/// IR records the back ran (EDEN_NX2_V2_IR), by kind.
struct IrData {
    std::array<u64, 32> records{};
    std::array<u64, 32> payload{};
    u64 ring_words{};
};
thread_local std::unique_ptr<IrData> t_ir_data;

void Rehash(const Maxwell3D& maxwell3d, u64& state_hash, u64& stream_hash) {
    const GroupTable& groups = Groups();
    const auto& reg_array = maxwell3d.regs.reg_array;
    state_hash = 0;
    stream_hash = 0;
    for (u32 method = 0; method < reg_array.size(); ++method) {
        const u32 bit = Bit(groups[method]);
        if ((bit & StateGroups) != 0) {
            state_hash ^= MixRegister(method, reg_array[method]);
        } else if ((bit & StreamGroups) != 0) {
            stream_hash ^= MixRegister(method, reg_array[method]);
        }
    }
}

u64 CbufHash(const Maxwell3D& maxwell3d) {
    u64 hash = 0;
    u64 key = 0;
    for (const auto& stage : maxwell3d.state.shader_stages) {
        for (const auto& buffer : stage.const_buffers) {
            ++key;
            if (buffer.enabled) {
                hash ^= Mix(Mix(buffer.address ^ (key << 56)) ^ buffer.size);
            }
        }
    }
    return hash;
}

// ---------------------------------------------------------------------------------------------
// Mutations.

constexpr size_t NumMutations = static_cast<size_t>(Mutation::Count);

constexpr std::array<const char*, NumMutations> mutation_names{
    "buffer uploads", "texture refreshes", "alias syncs", "rescales", "image deletes",
    "buffer deletes",
};
constexpr std::array<const char*, NumMutations> mutation_units{
    "KiB", "KiB", "copies", "", "", "",
};

struct MutationCounters {
    std::array<std::atomic<u64>, NumMutations> count{};
    std::array<std::atomic<u64>, NumMutations> amount{};
    std::array<std::atomic<u64>, NumMutations> draw_count{};
    std::array<std::atomic<u64>, NumMutations> draw_amount{};
};

MutationCounters g_mutations;

// ---------------------------------------------------------------------------------------------
// Report.

struct Window {
    u64 start_ns{};
    u64 start_ticks{};
};
thread_local Window t_window;

double PerSecond(u64 value, u64 window_ns) {
    return static_cast<double>(value) * 1e9 / static_cast<double>(window_ns);
}

double Percent(u64 part, u64 whole) {
    return whole == 0 ? 0.0 : static_cast<double>(part) * 100.0 / static_cast<double>(whole);
}

void ReportFront(const FrontData& data, const char* thread_name, u64 window_ns,
                 u64 window_ticks) {
    const auto ms = [window_ticks](u64 ticks) {
        return static_cast<double>(ticks) * 1000.0 / static_cast<double>(window_ticks);
    };
    std::array<u64, NumClasses> class_ticks{};
    std::array<u64, NumClasses> class_count{};
    std::string sites;
    for (size_t site = 0; site < NumSites; ++site) {
        std::string classes;
        for (size_t c = 0; c < NumClasses; ++c) {
            const u64 count = data.drain_count[site][c];
            if (count == 0) {
                continue;
            }
            class_ticks[c] += data.drain_ticks[site][c];
            class_count[c] += count;
            classes += fmt::format(" {} {:.1f} ms {:.0f}/s,", class_names[c],
                                   ms(data.drain_ticks[site][c]), PerSecond(count, window_ns));
        }
        if (!classes.empty()) {
            classes.pop_back();
            sites += fmt::format(" {}:{};", drain_site_names[site], classes);
        }
    }
    std::string classes;
    for (size_t c = 0; c < NumClasses; ++c) {
        classes += fmt::format(" {} {:.1f} ms {:.0f}/s,", class_names[c], ms(class_ticks[c]),
                               PerSecond(class_count[c], window_ns));
    }
    classes.pop_back();
    std::string acquires;
    for (size_t c = 0; c < NumClasses; ++c) {
        if (data.acquires[c] == 0 && data.polls[c] == 0) {
            continue;
        }
        acquires += fmt::format(" {} {:.0f}/s ({:.0f}% met), polls {:.0f}/s ({:.0f}% met);",
                                class_names[c], PerSecond(data.acquires[c], window_ns),
                                Percent(data.acquires_met[c], data.acquires[c]),
                                PerSecond(data.polls[c], window_ns),
                                Percent(data.polls_met[c], data.polls[c]));
    }
    if (!acquires.empty()) {
        acquires.pop_back();
    }
    LOG_INFO(HW_GPU,
             "V2 stats {} waits for the back by class (ms/s, waits/s):{} | by site:{} | "
             "acquires by class:{} | A/B not met at the first check {:.0f}/s | front releases literal {:.0f}/s, "
             "GPU-valued {:.0f}/s, table evictions {:.0f}/s",
             thread_name, classes, sites, acquires, PerSecond(data.mispredicted, window_ns),
             PerSecond(data.literal_releases, window_ns),
             PerSecond(data.gpu_valued_releases, window_ns),
             PerSecond(data.table_evictions, window_ns));
}

// SYNC_SUBMIT_SPEC Step 0: suspended acquire time by wake class, reduction view rejects,
// forced landings.
void ReportSync(const FrontData& data, const char* thread_name, u64 window_ns) {
    u64 sem_any = data.sem_evictions;
    std::string rows;
    for (size_t r = 0; r < sem_row_names.size(); ++r) {
        sem_any += data.sem_rows[r];
        rows += fmt::format(" {} {:.0f},", sem_row_names[r], PerSecond(data.sem_rows[r], window_ns));
    }
    if (sem_any != 0) {
        rows.pop_back();
        LOG_INFO(HW_GPU,
                 "V2 stats {} SYNC semaphore view decisions by row (per s):{} | slot evictions "
                 "{:.0f}/s",
                 thread_name, rows, PerSecond(data.sem_evictions, window_ns));
    }
    u64 any = data.forced_landings;
    std::string wakes;
    for (size_t w = 0; w < data.suspended.size(); ++w) {
        any += data.suspended[w];
        wakes += fmt::format(" {} {:.0f} ({:.1f} ms/s),", wake_names[w],
                             PerSecond(data.suspended[w], window_ns),
                             PerSecond(data.suspended_ns[w], window_ns) / 1e6);
    }
    std::string rejects;
    for (size_t r = 0; r < data.reduction_rejects.size(); ++r) {
        any += data.reduction_rejects[r];
        rejects += fmt::format(" {} {:.0f},", reject_names[r],
                               PerSecond(data.reduction_rejects[r], window_ns));
    }
    if (any == 0) {
        return;
    }
    wakes.pop_back();
    rejects.pop_back();
    LOG_INFO(HW_GPU,
             "V2 stats {} SYNC acquires resumed by wake class (per s):{} | reduction view "
             "rejects (per s):{} | forced landings {:.0f}/s, back blocked {:.1f} ms/s",
             thread_name, wakes, rejects, PerSecond(data.forced_landings, window_ns),
             PerSecond(data.forced_landing_ns, window_ns) / 1e6);
}

void ReportDecisions(const FrontData& data, const char* thread_name, u64 window_ns) {
    std::string sites;
    for (size_t site = 0; site < NumSites; ++site) {
        u64 total = data.mismatches[site] + data.benign_mismatches[site];
        for (const u64 count : data.decisions[site]) {
            total += count;
        }
        if (total == 0) {
            continue;
        }
        std::string line;
        for (size_t d = 0; d < data.decisions[site].size(); ++d) {
            line += fmt::format(" {} {:.0f},", decision_names[d],
                                PerSecond(data.decisions[site][d], window_ns));
        }
        line.pop_back();
        std::string reasons;
        for (size_t r = 1; r < data.reasons[site].size(); ++r) {
            if (data.reasons[site][r] != 0) {
                reasons += fmt::format(" {} {:.0f},", reason_names[r],
                                       PerSecond(data.reasons[site][r], window_ns));
            }
        }
        if (!reasons.empty()) {
            reasons.pop_back();
            line += fmt::format(" (waited for:{})", reasons);
        }
        static constexpr std::array<const char*, 8> producers{
            "release", "report", "cbuf data", "upload", "copy", "dispatch", "draw", "qmd release",
        };
        std::string narrowed;
        for (size_t p = 0; p < producers.size(); ++p) {
            if (data.narrowed_by[site][p] != 0) {
                narrowed += fmt::format(" {} {:.0f},", producers[p],
                                        PerSecond(data.narrowed_by[site][p], window_ns));
            }
        }
        if (!narrowed.empty()) {
            narrowed.pop_back();
            line += fmt::format(" (narrowed by:{})", narrowed);
        }
        if (data.mismatches[site] != 0) {
            line += fmt::format(", verify mismatches {}", data.mismatches[site]);
        }
        if (data.benign_mismatches[site] != 0) {
            line += fmt::format(", benign verify mismatches (new path waits) {}",
                                data.benign_mismatches[site]);
        }
        sites += fmt::format(" {}:{};", drain_site_names[site], line);
    }
    ReportSync(data, thread_name, window_ns);
    if (sites.empty()) {
        return;
    }
    sites.pop_back();
    LOG_INFO(HW_GPU, "V2 stats {} P2 wait decisions (per s):{}", thread_name, sites);
}

void ReportDraws(const DrawData& data, const char* thread_name, u64 window_ns) {
    std::string groups;
    for (size_t g = 0; g < NumGroups; ++g) {
        groups += fmt::format(" {} {:.1f},", group_names[g], Percent(data.changed[g], data.draws));
    }
    groups.pop_back();
    LOG_INFO(HW_GPU,
             "V2 stats {} draws {:.0f}/s ({:.1f}% indirect) | groups changed (% of draws):{} | "
             "zero-change {:.1f}% light-only {:.1f}% light+bind {:.1f}% | draw cache hit: state "
             "256 {:.1f}% 4096 {:.1f}%, state+bindings 256 {:.1f}% 4096 {:.1f}% | hash drift {}",
             thread_name, PerSecond(data.draws, window_ns), Percent(data.indirect, data.draws),
             groups, Percent(data.zero_change, data.draws), Percent(data.light_only, data.draws),
             Percent(data.light_bind, data.draws), Percent(data.state_256.hits, data.draws),
             Percent(data.state_4096.hits, data.draws), Percent(data.full_256.hits, data.draws),
             Percent(data.full_4096.hits, data.draws), data.hash_drift);
    std::string runs;
    u64 runs_total = 0;
    u64 run_draws_total = 0;
    for (size_t b = 0; b < NumRunBuckets; ++b) {
        runs_total += data.runs[b];
        run_draws_total += data.run_draws[b];
        runs += fmt::format(" {} {:.0f} ({:.1f}%),", run_bucket_names[b],
                            PerSecond(data.runs[b], window_ns),
                            Percent(data.run_draws[b], data.draws));
    }
    runs.pop_back();
    LOG_INFO(HW_GPU,
             "V2 stats {} zero-change runs (runs/s, % of draws):{} | draws in runs of 2+ {:.1f}%, "
             "mean run {:.2f}",
             thread_name, runs, Percent(run_draws_total - data.run_draws[0], data.draws),
             runs_total == 0 ? 0.0
                             : static_cast<double>(run_draws_total) /
                                   static_cast<double>(runs_total));
    std::string mutations;
    for (size_t m = 0; m < NumMutations; ++m) {
        const u64 count = g_mutations.count[m].exchange(0, std::memory_order_relaxed);
        const u64 amount = g_mutations.amount[m].exchange(0, std::memory_order_relaxed);
        const u64 draw_count = g_mutations.draw_count[m].exchange(0, std::memory_order_relaxed);
        const u64 draw_amount = g_mutations.draw_amount[m].exchange(0, std::memory_order_relaxed);
        mutations += fmt::format(" {} {:.0f}/{:.0f}", mutation_names[m],
                                 PerSecond(draw_count, window_ns), PerSecond(count, window_ns));
        if (mutation_units[m][0] != '\0') {
            const bool kib = mutation_units[m][0] == 'K';
            const double scale = kib ? 1.0 / 1024.0 : 1.0;
            mutations += fmt::format(" ({:.0f}/{:.0f} {})",
                                     PerSecond(draw_amount, window_ns) * scale,
                                     PerSecond(amount, window_ns) * scale, mutation_units[m]);
        }
        mutations += ',';
    }
    mutations.pop_back();
    LOG_INFO(HW_GPU, "V2 stats {} cache mutations per second (in draw resolve/all):{}",
             thread_name, mutations);
}

} // Anonymous namespace

// ---------------------------------------------------------------------------------------------

void NoteRelease(size_t address_space, GPUVAddr address, u32 payload, ReleaseKind kind) {
    FrontState& front = Front();
    ReleaseEntry& entry = front.releases[ReleaseIndex(address)];
    if (entry.valid && entry.address != address) {
        ++front.data.table_evictions;
    }
    entry = {
        .address = address,
        .position = GpuSplit::Get().CutPosition(),
        .payload = payload,
        .address_space = static_cast<u32>(address_space),
        .kind = kind,
        .valid = true,
    };
    if (kind == ReleaseKind::Literal) {
        ++front.data.literal_releases;
    } else {
        ++front.data.gpu_valued_releases;
    }
}

void NoteReportSemaphore(size_t address_space, const u32* regs) {
    const u32 query = regs[OFF(report_semaphore.query)];
    const u32 operation = query & 3;
    // Release (0) and ReportOnly (2) write; Acquire and Trap don't.
    if (operation != 0 && operation != 2) {
        return;
    }
    const GPUVAddr address = (GPUVAddr{regs[OFF(report_semaphore.address_high)]} << 32) |
                             regs[OFF(report_semaphore.address_low)];
    const u32 report = (query >> 23) & 0x1f;
    NoteRelease(address_space, address, regs[OFF(report_semaphore.payload)],
                report == 0 ? ReleaseKind::Literal : ReleaseKind::GpuValued);
}

SyncClass ClassifyAcquire(size_t address_space, GPUVAddr address, u32 value, AcquireOp op) {
    const ReleaseEntry& entry = Front().releases[ReleaseIndex(address)];
    if (!entry.valid || entry.address != address ||
        entry.address_space != static_cast<u32>(address_space)) {
        return SyncClass::Unknown;
    }
    if (entry.kind == ReleaseKind::GpuValued) {
        return SyncClass::D;
    }
    if (!Satisfies(op, entry.payload, value)) {
        return SyncClass::Unknown;
    }
    const auto& split = GpuSplit::Get();
    return split.Active() && entry.position > split.ConsumedPosition() ? SyncClass::B
                                                                       : SyncClass::A;
}

void NoteAcquire(SyncClass sync_class, bool poll, bool met) {
    FrontData& data = Front().data;
    const auto index = static_cast<size_t>(sync_class);
    if (poll) {
        ++data.polls[index];
        data.polls_met[index] += met ? 1 : 0;
    } else {
        ++data.acquires[index];
        data.acquires_met[index] += met ? 1 : 0;
    }
    // Fix pass 1 (B7): a suspended channel's polls repeat until met, so they are no prediction;
    // only an acquire's first check counts (it read 51-98k/s with every feature off).
    if (!poll && !met && (sync_class == SyncClass::A || sync_class == SyncClass::B)) {
        ++data.mispredicted;
    }
}

void NoteDrain(DrainSite site, u64 start) {
    SyncClass sync_class = SiteClass(site);
    if (sync_class == SyncClass::Unknown && t_context != SyncClass::Count) {
        sync_class = t_context;
    }
    const auto s = static_cast<size_t>(site);
    const auto c = static_cast<size_t>(sync_class);
    FrontData& data = Front().data;
    data.drain_ticks[s][c] += Nx2Profile::Ticks() - start;
    ++data.drain_count[s][c];
}

void NoteDecision(DrainSite site, Decision decision, Reason reason) {
    if (!Enabled()) [[likely]] {
        return;
    }
    FrontData& data = Front().data;
    const auto s = static_cast<size_t>(site);
    ++data.decisions[s][static_cast<size_t>(decision)];
    if (reason != Reason::None) {
        ++data.reasons[s][static_cast<size_t>(reason)];
    }
}

void NoteNarrowedBy(DrainSite site, u32 producers) {
    if (!Enabled()) [[likely]] {
        return;
    }
    auto& counts = Front().data.narrowed_by[static_cast<size_t>(site)];
    for (size_t p = 0; p < counts.size(); ++p) {
        counts[p] += (producers >> p) & 1;
    }
}

void NoteMismatch(DrainSite site, bool benign) {
    if (!Enabled()) [[likely]] {
        return;
    }
    auto& data = Front().data;
    ++(benign ? data.benign_mismatches : data.mismatches)[static_cast<size_t>(site)];
}

void NoteSuspended(WakeClass wake, u64 ns) {
    if (!Enabled()) [[likely]] {
        return;
    }
    auto& data = Front().data;
    ++data.suspended[static_cast<size_t>(wake)];
    data.suspended_ns[static_cast<size_t>(wake)] += ns;
}

void NoteReductionReject(ReductionReject reason) {
    if (!Enabled()) [[likely]] {
        return;
    }
    ++Front().data.reduction_rejects[static_cast<size_t>(reason)];
}

void NoteForcedLanding(u64 ns) {
    if (!Enabled()) [[likely]] {
        return;
    }
    auto& data = Front().data;
    ++data.forced_landings;
    data.forced_landing_ns += ns;
}

void NoteSemRow(u8 row) {
    if (!Enabled()) [[likely]] {
        return;
    }
    ++Front().data.sem_rows[std::min<size_t>(row, 7)];
}

void NoteSemEvictions(u64 count) {
    if (!Enabled()) [[likely]] {
        return;
    }
    Front().data.sem_evictions += count;
}

AcquireContext::AcquireContext(SyncClass sync_class) : enabled{Enabled()} {
    if (enabled) {
        previous = t_context;
        t_context = sync_class;
    }
}

AcquireContext::~AcquireContext() {
    if (enabled) {
        t_context = previous;
    }
}

void NoteWrite(DrawTracker& tracker, u32 method, u32 old_value, u32 value) {
    if (method >= Maxwell3D::Regs::NUM_REGS) {
        return;
    }
    const u32 bit = Bit(Groups()[method]);
    if (bit == Bit(Group::CbufData)) {
        tracker.pending |= bit;
        return;
    }
    if (old_value == value) {
        return;
    }
    tracker.pending |= bit;
    const u64 delta = MixRegister(method, old_value) ^ MixRegister(method, value);
    if ((bit & StateGroups) != 0) {
        tracker.state_hash ^= delta;
    } else if ((bit & StreamGroups) != 0) {
        tracker.stream_hash ^= delta;
    }
}

void NoteIrRecord(u32 kind, size_t words, size_t payload_words) {
    if (!t_ir_data) [[unlikely]] {
        t_ir_data = std::make_unique<IrData>();
    }
    IrData& data = *t_ir_data;
    const size_t index = std::min<size_t>(kind, data.records.size() - 1);
    ++data.records[index];
    data.payload[index] += payload_words;
    data.ring_words += words;
}

void NoteHashDrift() {
    ++Draws().hash_drift;
}

void NoteConstantData(DrawTracker& tracker) {
    tracker.pending |= Bit(Group::CbufData);
}

void NoteDraw(Maxwell3D& maxwell3d, u32 topology, bool indexed, bool indirect) {
    DrawTracker& tracker = maxwell3d.v2_draw_tracker;
    DrawData& data = Draws();
    // The incremental hashes miss writes made around ProcessDirtyRegisters (state loads,
    // EDEN_NX2_PERF_VERIFY's restores): recompute them now and then and count differences.
    if (!tracker.valid || ++tracker.since_rehash >= 4096) {
        u64 state_hash;
        u64 stream_hash;
        Rehash(maxwell3d, state_hash, stream_hash);
        if (tracker.valid &&
            (state_hash != tracker.state_hash || stream_hash != tracker.stream_hash)) {
            ++data.hash_drift;
        }
        if (!tracker.valid) {
            tracker.pending = ~0U >> (32 - NumGroups);
        }
        tracker.state_hash = state_hash;
        tracker.stream_hash = stream_hash;
        tracker.since_rehash = 0;
        tracker.valid = true;
    }
    const u64 cbuf_hash = CbufHash(maxwell3d);
    if (cbuf_hash != tracker.cbuf_hash) {
        tracker.pending |= Bit(Group::CbufBind);
        tracker.cbuf_hash = cbuf_hash;
    }
    if (topology != tracker.topology) {
        tracker.pending |= Bit(Group::Pipeline);
        tracker.topology = topology;
    }
    const u32 pending = tracker.pending;
    tracker.pending = 0;

    ++data.draws;
    data.indirect += indirect ? 1 : 0;
    for (u32 bits = pending; bits != 0; bits &= bits - 1) {
        ++data.changed[static_cast<size_t>(std::countr_zero(bits))];
    }
    const bool zero_change = (pending & ~IgnoredGroups) == 0;
    data.zero_change += zero_change ? 1 : 0;
    data.light_only += (pending & ~LightGroups) == 0 ? 1 : 0;
    data.light_bind += (pending & ~LightBindGroups) == 0 ? 1 : 0;
    if (zero_change && data.run_length != 0) {
        ++data.run_length;
    } else {
        if (data.run_length != 0) {
            const size_t bucket = RunBucket(data.run_length);
            ++data.runs[bucket];
            data.run_draws[bucket] += data.run_length;
        }
        data.run_length = 1;
    }

    const u64 state_fingerprint =
        Mix(tracker.state_hash ^ (u64{topology} << 48) ^ (indirect ? 1ULL << 40 : 0));
    const u64 full_fingerprint =
        Mix(state_fingerprint ^ std::rotl(tracker.stream_hash, 21) ^ std::rotl(cbuf_hash, 43) ^
            (indexed ? 1ULL << 41 : 0));
    data.state_256.Look(state_fingerprint);
    data.state_4096.Look(state_fingerprint);
    data.full_256.Look(full_fingerprint);
    data.full_4096.Look(full_fingerprint);
}

DrawScope::DrawScope(bool enabled_) : enabled{enabled_} {
    if (enabled) {
        t_in_draw = true;
    }
}

DrawScope::~DrawScope() {
    if (enabled) {
        t_in_draw = false;
    }
}

void CountMutationSlow(Mutation mutation, u64 amount) {
    const auto index = static_cast<size_t>(mutation);
    g_mutations.count[index].fetch_add(1, std::memory_order_relaxed);
    g_mutations.amount[index].fetch_add(amount, std::memory_order_relaxed);
    if (t_in_draw) {
        g_mutations.draw_count[index].fetch_add(1, std::memory_order_relaxed);
        g_mutations.draw_amount[index].fetch_add(amount, std::memory_order_relaxed);
    }
}

void ReportSlow(const char* thread_name) {
    auto& window = t_window;
    const u64 now = Nx2Profile::Now();
    if (window.start_ns == 0) {
        window.start_ns = now;
        window.start_ticks = Nx2Profile::Ticks();
        return;
    }
    const u64 window_ns = now - window.start_ns;
    if (window_ns < WindowNs) {
        return;
    }
    const u64 now_ticks = Nx2Profile::Ticks();
    const u64 window_ticks = std::max<u64>(now_ticks - window.start_ticks, 1);
    if (t_front_state) {
        FrontData& front = t_front_state->data;
        bool active = front.literal_releases != 0 || front.gpu_valued_releases != 0;
        for (size_t c = 0; c < NumClasses; ++c) {
            active |= front.acquires[c] != 0 || front.polls[c] != 0;
        }
        for (const auto& site : front.drain_count) {
            for (const u64 count : site) {
                active |= count != 0;
            }
        }
        if (active) {
            ReportFront(front, thread_name, window_ns, window_ticks);
        }
        ReportDecisions(front, thread_name, window_ns);
        front = {};
    }
    if (t_draw_data) {
        DrawData& draws = *t_draw_data;
        if (draws.draws != 0) {
            ReportDraws(draws, thread_name, window_ns);
        }
        draws.draws = 0;
        draws.indirect = 0;
        draws.changed = {};
        draws.zero_change = 0;
        draws.light_only = 0;
        draws.light_bind = 0;
        draws.hash_drift = 0;
        draws.runs = {};
        draws.run_draws = {};
        draws.state_256.hits = 0;
        draws.state_4096.hits = 0;
        draws.full_256.hits = 0;
        draws.full_4096.hits = 0;
    }
    if (t_ir_data) {
        IrData& ir = *t_ir_data;
        std::string kinds;
        u64 payload = 0;
        for (size_t kind = 1; kind < static_cast<size_t>(Ir::Kind::Count); ++kind) {
            payload += ir.payload[kind];
            if (ir.records[kind] == 0) {
                continue;
            }
            kinds += fmt::format(" {} {:.0f}", Ir::kind_names[kind],
                                 PerSecond(ir.records[kind], window_ns));
            if (ir.payload[kind] != 0) {
                kinds += fmt::format(" ({:.0f} payload words)",
                                     PerSecond(ir.payload[kind], window_ns));
            }
            kinds += ',';
        }
        if (!kinds.empty()) {
            kinds.pop_back();
            LOG_INFO(HW_GPU,
                     "V2 stats {} IR records (per s):{} | ring words {:.0f}/s, payload arena "
                     "words {:.0f}/s",
                     thread_name, kinds, PerSecond(ir.ring_words, window_ns),
                     PerSecond(payload, window_ns));
        }
        ir = {};
    }
    P3Stats::Report(thread_name, window_ns);
    Commit::Stats::Report(thread_name, window_ns);
    P5bStats::Report(thread_name, window_ns);
    if (On(Switch::DrawRec)) [[unlikely]] {
        const std::string text = Resolve::TakeRecordStatsText(static_cast<double>(window_ns) / 1e9);
        if (!text.empty()) {
            LOG_INFO(HW_GPU, "V2 stats {} P6 Step 0 | {}", thread_name, text);
        }
    }
    window.start_ns = now;
    window.start_ticks = now_ticks;
}

} // namespace VideoCommon::V2::Stats
