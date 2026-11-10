// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "video_core/nx2_profile.h"
#include "video_core/v2/commit.h"
#include "video_core/v2/resolve/draw_record.h"

namespace VideoCommon::V2::DrawCache {

void RecordTemplate::Deleter::operator()(Resolve::ResolvedDraw* pointer) const noexcept {
    delete pointer;
}

RecordTemplate::RecordTemplate(const RecordTemplate& other) {
    *this = other;
}

RecordTemplate& RecordTemplate::operator=(const RecordTemplate& other) {
    if (this == &other) {
        return *this;
    }
    valid = false;
    if (other.valid) {
        Store(*other.record);
    }
    return *this;
}

void RecordTemplate::Store(const Resolve::ResolvedDraw& source) {
    if (!record) {
        record.reset(new Resolve::ResolvedDraw{});
    }
    record->h = source.h;
    record->arena.assign(source.arena.begin(), source.arena.end());
    valid = true;
}

} // namespace VideoCommon::V2::DrawCache

namespace VideoCommon::V2::Resolve {

namespace {

/// Whether a resolved slot id matches an observed one: equal, or a proxy for a resource created
/// during the draw (dependent).
bool SameId(u32 resolved, u32 observed, bool created, u32& dependent) {
    if (resolved == observed) {
        return true;
    }
    if (created && Commit::IsProxy(resolved)) {
        ++dependent;
        return true;
    }
    return false;
}

bool SameRef(const Versions::WeakRef& resolved, const Versions::WeakRef& observed, bool created,
             u32& dependent) {
    if (Commit::IsProxy(resolved.id)) {
        return SameId(resolved.id, observed.id, created, dependent);
    }
    return resolved.id == observed.id;
}

/// SameRef for buffers: a buffer the resolve found and the configure deleted (a join into a
/// buffer it created) is dependent too: after a deletion the apply looks every binding up
/// again (RepeekRecordBuffers).
bool SameBuffer(const Versions::WeakRef& resolved, const Versions::WeakRef& observed,
                bool created, u32& dependent) {
    if (SameRef(resolved, observed, created, dependent)) {
        return true;
    }
    if (created && !Commit::IsProxy(resolved.id) &&
        Versions::Gens().Get(Versions::Kind::Buffer, resolved.id) != resolved.generation) {
        ++dependent;
        return true;
    }
    return false;
}

constexpr u32 Bit(RecordGroup group) {
    return 1U << static_cast<u32>(group);
}

template <class T, class Equal>
bool ListsEqual(const ResolvedDraw& lhs_record, Seg lhs_seg, const ResolvedDraw& rhs_record,
                Seg rhs_seg, Equal&& equal) {
    const auto lhs = lhs_record.Span<T>(lhs_seg);
    const auto rhs = rhs_record.Span<T>(rhs_seg);
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!equal(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

} // Anonymous namespace

bool HasProxies(const ResolvedDraw& record) {
    const DrawHeader& h = record.h;
    bool proxy = h.index.valid && Commit::IsProxy(h.index.buffer.id);
    for (const ResolvedView& entry : record.Span<const ResolvedView>(h.views)) {
        proxy |= Commit::IsProxy(entry.view.id);
    }
    for (const ResolvedSampler& entry : record.Span<const ResolvedSampler>(h.samplers)) {
        proxy |= Commit::IsProxy(entry.sampler.id);
    }
    for (const ResolvedVertex& entry : record.Span<const ResolvedVertex>(h.vertex)) {
        proxy |= Commit::IsProxy(entry.buffer.id);
    }
    for (const ResolvedUbo& entry : record.Span<const ResolvedUbo>(h.ubo)) {
        proxy |= Commit::IsProxy(entry.buffer.id);
    }
    for (const ResolvedSsbo& entry : record.Span<const ResolvedSsbo>(h.ssbo)) {
        proxy |= Commit::IsProxy(entry.buffer.id);
    }
    for (const ResolvedTbo& entry : record.Span<const ResolvedTbo>(h.tbo)) {
        proxy |= Commit::IsProxy(entry.buffer.id);
    }
    return proxy;
}

std::string DescribeRecordGroup(const ResolvedDraw& resolved, const ResolvedDraw& observed,
                                RecordGroup group) {
    const DrawHeader& r = resolved.h;
    const DrawHeader& o = observed.h;
    // The first entry of two lists whose formatted fields differ (or the counts).
    const auto lists = [&]<class T>(T*, Seg rs, Seg os, auto&& format) -> std::string {
        const auto lhs = resolved.Span<const T>(rs);
        const auto rhs = observed.Span<const T>(os);
        for (size_t i = 0; i < (std::min)(lhs.size(), rhs.size()); ++i) {
            std::string a = format(lhs[i]);
            std::string b = format(rhs[i]);
            if (a != b) {
                return fmt::format("[{}] record {{{}}} configure {{{}}}", i, a, b);
            }
        }
        return fmt::format("{} entries vs {}", lhs.size(), rhs.size());
    };
    const auto ref = [](const Versions::WeakRef& value) {
        return Commit::IsProxy(value.id) ? fmt::format("proxy {}", Commit::ProxyIndex(value.id))
                                         : fmt::format("{}", value.id);
    };
    switch (group) {
    case RecordGroup::Views:
        return lists(static_cast<ResolvedView*>(nullptr), r.views, o.views,
                     [&](const ResolvedView& v) {
                         return fmt::format("tic {} view {} stage {} kind {} blacklist {}",
                                            v.tic_index, ref(v.view), v.stage, v.kind,
                                            v.blacklist);
                     });
    case RecordGroup::Samplers:
        return lists(static_cast<ResolvedSampler*>(nullptr), r.samplers, o.samplers,
                     [&](const ResolvedSampler& v) {
                         return fmt::format("tsc {} sampler {}", v.tsc_index, ref(v.sampler));
                     });
    case RecordGroup::Vertex:
        return fmt::format("mask {:#x} vs {:#x}; ", r.vertex_mask, o.vertex_mask) +
               lists(static_cast<ResolvedVertex*>(nullptr), r.vertex, o.vertex,
                     [&](const ResolvedVertex& v) {
                         return fmt::format("slot {} buffer {} daddr {:#x} size {:#x} stride {}",
                                            v.slot, ref(v.buffer), v.daddr, v.size, v.stride);
                     });
    case RecordGroup::Index: {
        const auto index = [&](const ResolvedIndex& v) {
            return fmt::format("valid {} buffer {} daddr {:#x} size {:#x} format {} first {} "
                               "count {}",
                               v.valid, ref(v.buffer), v.daddr, v.size, v.format, v.first,
                               v.count);
        };
        return fmt::format("record {{{}}} configure {{{}}}", index(r.index), index(o.index));
    }
    case RecordGroup::Ubo:
        return lists(static_cast<ResolvedUbo*>(nullptr), r.ubo, o.ubo, [&](const ResolvedUbo& v) {
            return fmt::format("stage {} slot {} buffer {} daddr {:#x} size {:#x} version {}",
                               v.stage, v.slot, ref(v.buffer), v.daddr, v.size, v.version);
        });
    case RecordGroup::Ssbo:
        return lists(static_cast<ResolvedSsbo*>(nullptr), r.ssbo, o.ssbo,
                     [&](const ResolvedSsbo& v) {
                         return fmt::format("stage {} index {} written {} buffer {} daddr {:#x} "
                                            "va {:#x} size {:#x}",
                                            v.stage, v.index, v.written, ref(v.buffer), v.daddr,
                                            v.gpu_addr, v.size);
                     });
    case RecordGroup::Tbo:
        return lists(static_cast<ResolvedTbo*>(nullptr), r.tbo, o.tbo, [&](const ResolvedTbo& v) {
            return fmt::format("stage {} index {} written {} image {} buffer {} daddr {:#x} va "
                               "{:#x} size {:#x} format {}",
                               v.stage, v.index, v.written, v.image, ref(v.buffer), v.daddr,
                               v.gpu_addr, v.size, v.format);
        });
    case RecordGroup::Rt:
        return fmt::format("record {} configure {}", fmt::join(r.rt_ids, ","),
                           fmt::join(o.rt_ids, ","));
    case RecordGroup::RenderArea:
        return fmt::format("record {} {}x{} configure {} {}x{}", r.render_area.uses,
                           r.render_area.w, r.render_area.h, o.render_area.uses, o.render_area.w,
                           o.render_area.h);
    case RecordGroup::Dynamic: {
        const auto* lhs = reinterpret_cast<const u8*>(&r.dynamic);
        const auto* rhs = reinterpret_cast<const u8*>(&o.dynamic);
        for (size_t i = 0; i < sizeof(DynamicState); ++i) {
            if (lhs[i] != rhs[i]) {
                return fmt::format("first byte {} of {}: {:#x} vs {:#x}", i, sizeof(DynamicState),
                                   lhs[i], rhs[i]);
            }
        }
        return {};
    }
    case RecordGroup::Reads:
        return lists(static_cast<DrawCache::InputRead*>(nullptr), r.reads, o.reads,
                     [&](const DrawCache::InputRead& v) {
                         u32 word = 0;
                         std::memcpy(&word, v.bytes.data(), sizeof(word));
                         return fmt::format("va {:#x} size {} first word {:#x}", v.address, v.size,
                                            word);
                     });
    default:
        return {};
    }
}

u32 PredictKnownViewProxies(ResolvedDraw& record, std::span<const Commit::Request> textures) {
    u32 replaced = 0;
    for (ResolvedView& entry : record.Span<ResolvedView>(record.h.views)) {
        if (!Commit::IsProxy(entry.view.id)) {
            continue;
        }
        for (const Commit::Request& request : textures) {
            if (request.proxy != entry.view.id) {
                continue;
            }
            if (request.kind == Commit::Kind::ViewBind && request.slot != Commit::InvalidSlot &&
                !Commit::IsProxy(request.slot)) {
                entry.view = {request.slot, 0};
                ++replaced;
            }
            break;
        }
    }
    return replaced;
}

bool CompareRecords(const ResolvedDraw& resolved, const ResolvedDraw& observed, u32 created_mask,
                    std::string& reasons, u32* mismatched_groups, u32* dependent) {
    u32 groups = 0;
    u32 proxies = 0;
    const auto created = [&](RecordGroup group) { return (created_mask & Bit(group)) != 0; };
    const DrawHeader& r = resolved.h;
    const DrawHeader& o = observed.h;
    if (r.pipeline != o.pipeline) {
        groups |= Bit(RecordGroup::Pipeline);
    }
    const auto lists_equal = [&]<class T, class Equal>(T*, Seg rs, Seg os, Equal&& equal) {
        return ListsEqual<T>(resolved, rs, observed, os, equal);
    };
    if (!lists_equal(static_cast<ResolvedView*>(nullptr),
            r.views, o.views, [&](const ResolvedView& lhs, const ResolvedView& rhs) {
                return lhs.tic_index == rhs.tic_index && lhs.stage == rhs.stage &&
                       lhs.kind == rhs.kind && lhs.blacklist == rhs.blacklist &&
                       SameRef(lhs.view, rhs.view, created(RecordGroup::Views), proxies);
            })) {
        groups |= Bit(RecordGroup::Views);
    }
    if (!lists_equal(static_cast<ResolvedSampler*>(nullptr),
            r.samplers, o.samplers, [&](const ResolvedSampler& lhs, const ResolvedSampler& rhs) {
                return lhs.tsc_index == rhs.tsc_index &&
                       SameRef(lhs.sampler, rhs.sampler, created(RecordGroup::Samplers), proxies);
            })) {
        groups |= Bit(RecordGroup::Samplers);
    }
    if (!lists_equal(static_cast<DrawCache::InputRead*>(nullptr),
            r.reads, o.reads, [](const DrawCache::InputRead& lhs, const DrawCache::InputRead& rhs) {
                return lhs.address == rhs.address && lhs.size == rhs.size &&
                       lhs.what == rhs.what &&
                       std::memcmp(lhs.bytes.data(), rhs.bytes.data(), lhs.size) == 0;
            })) {
        groups |= Bit(RecordGroup::Reads);
    }
    const bool buffers_created = created(RecordGroup::Vertex);
    if (r.vertex_mask != o.vertex_mask ||
        !lists_equal(static_cast<ResolvedVertex*>(nullptr),
            r.vertex, o.vertex, [&](const ResolvedVertex& lhs, const ResolvedVertex& rhs) {
                return lhs.slot == rhs.slot && lhs.daddr == rhs.daddr && lhs.size == rhs.size &&
                       lhs.stride == rhs.stride &&
                       SameBuffer(lhs.buffer, rhs.buffer, buffers_created, proxies);
            })) {
        groups |= Bit(RecordGroup::Vertex);
    }
    if (r.index.valid != o.index.valid ||
        (r.index.valid &&
         (r.index.daddr != o.index.daddr || r.index.size != o.index.size ||
          r.index.format != o.index.format || r.index.first != o.index.first ||
          r.index.count != o.index.count ||
          !SameBuffer(r.index.buffer, o.index.buffer, buffers_created, proxies)))) {
        groups |= Bit(RecordGroup::Index);
    }
    if (r.ubo_masks != o.ubo_masks ||
        !lists_equal(static_cast<ResolvedUbo*>(nullptr),
            r.ubo, o.ubo, [&](const ResolvedUbo& lhs, const ResolvedUbo& rhs) {
                return lhs.stage == rhs.stage && lhs.slot == rhs.slot && lhs.daddr == rhs.daddr &&
                       lhs.size == rhs.size && lhs.version == rhs.version &&
                       SameBuffer(lhs.buffer, rhs.buffer, buffers_created, proxies);
            })) {
        groups |= Bit(RecordGroup::Ubo);
    }
    if (!lists_equal(static_cast<ResolvedSsbo*>(nullptr),
            r.ssbo, o.ssbo, [&](const ResolvedSsbo& lhs, const ResolvedSsbo& rhs) {
                return lhs.stage == rhs.stage && lhs.index == rhs.index &&
                       lhs.written == rhs.written && lhs.daddr == rhs.daddr &&
                       lhs.gpu_addr == rhs.gpu_addr && lhs.size == rhs.size &&
                       SameBuffer(lhs.buffer, rhs.buffer, buffers_created, proxies);
            })) {
        groups |= Bit(RecordGroup::Ssbo);
    }
    if (!lists_equal(static_cast<ResolvedTbo*>(nullptr),
            r.tbo, o.tbo, [&](const ResolvedTbo& lhs, const ResolvedTbo& rhs) {
                return lhs.stage == rhs.stage && lhs.index == rhs.index &&
                       lhs.written == rhs.written && lhs.image == rhs.image &&
                       lhs.gpu_addr == rhs.gpu_addr && lhs.daddr == rhs.daddr &&
                       lhs.size == rhs.size && lhs.format == rhs.format &&
                       SameBuffer(lhs.buffer, rhs.buffer, buffers_created, proxies);
            })) {
        groups |= Bit(RecordGroup::Tbo);
    }
    if (r.rt_known && o.rt_known) {
        for (size_t i = 0; i < NumRecordTargets; ++i) {
            if (!SameId(r.rt_ids[i], o.rt_ids[i], created(RecordGroup::Rt), proxies)) {
                groups |= Bit(RecordGroup::Rt);
                break;
            }
        }
    }
    if (std::memcmp(&r.dynamic, &o.dynamic, sizeof(DynamicState)) != 0) {
        groups |= Bit(RecordGroup::Dynamic);
    }
    if (r.render_area.uses != o.render_area.uses ||
        (r.render_area.uses &&
         (r.render_area.w != o.render_area.w || r.render_area.h != o.render_area.h))) {
        groups |= Bit(RecordGroup::RenderArea);
    }
    for (size_t group = 0; group < NumRecordGroups; ++group) {
        if ((groups >> group) & 1) {
            reasons += reasons.empty() ? "" : ", ";
            reasons += record_group_names[group];
        }
    }
    if (mismatched_groups != nullptr) {
        *mismatched_groups |= groups;
    }
    if (dependent != nullptr) {
        *dependent += proxies;
    }
    return groups == 0;
}

RecordStats& GetRecordStats() {
    static RecordStats stats;
    return stats;
}

std::string TakeRecordStatsText(double seconds) {
    // The time stamp counter's rate against the wall clock since the first report (S0-04).
    static const std::pair<u64, u64> origin{VideoCommon::Nx2Profile::Ticks(),
                                            VideoCommon::Nx2Profile::Now()};
    RecordStats& stats = GetRecordStats();
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    const auto rate = [&](u64 value) { return static_cast<double>(value) / seconds; };
    u64 any = stats.drawrec_draws.load(std::memory_order_relaxed) +
              stats.verify_draws.load(std::memory_order_relaxed);
    for (const auto& counter : stats.commit_only) {
        any += counter.load(std::memory_order_relaxed);
    }
    if (any == 0) {
        return {}; // another thread's window took them, or the path did not run
    }
    const u64 draws = take(stats.drawrec_draws);
    std::array<u64, NumCommitOnlyReasons> reasons{};
    u64 commit_only_total = 0;
    for (size_t i = 1; i < NumCommitOnlyReasons; ++i) {
        reasons[i] = take(stats.commit_only[i]);
        commit_only_total += reasons[i];
    }
    // The share of each reason among the draws the record path met (R-14, S0-03).
    const auto share = [&](u64 value) {
        const u64 total = draws + commit_only_total;
        return total != 0 ? 100.0 * static_cast<double>(value) / static_cast<double>(total) : 0.0;
    };
    std::string commit_only;
    if (commit_only_total != 0) {
        commit_only = fmt::format(" {:.1f}%", share(commit_only_total));
    }
    for (size_t i = 1; i < NumCommitOnlyReasons; ++i) {
        if (reasons[i] != 0) {
            commit_only += fmt::format(" {} {:.0f} ({:.1f}%)", commit_only_names[i],
                                       rate(reasons[i]), share(reasons[i]));
        }
    }
    std::string mismatches;
    for (size_t i = 0; i < NumRecordGroups; ++i) {
        if (const u64 value = take(stats.mismatches[i]); value != 0) {
            mismatches += fmt::format(" {} {}", record_group_names[i], value);
        }
    }
    // S0-04: microseconds per draw by path (whole draw / its resolve), the counter converted with the
    // window's length.
    std::array<u64, 3> paths{};
    std::array<u64, 3> path_ticks{};
    std::array<u64, 3> resolve_ticks{};
    for (size_t i = 0; i < paths.size(); ++i) {
        paths[i] = take(stats.by_path[i]);
        path_ticks[i] = take(stats.path_ticks[i]);
        resolve_ticks[i] = take(stats.resolve_ticks[i]);
    }
    const u64 now_ticks = VideoCommon::Nx2Profile::Ticks();
    const u64 now_ns = VideoCommon::Nx2Profile::Now();
    std::string timing;
    if (now_ticks > origin.first && now_ns > origin.second) {
        const double us_per_tick = static_cast<double>(now_ns - origin.second) / 1e3 /
                                   static_cast<double>(now_ticks - origin.first);
        const auto us = [&](u64 ticks, u64 count) {
            return count != 0 ? static_cast<double>(ticks) * us_per_tick / static_cast<double>(count)
                              : 0.0;
        };
        timing = fmt::format(
            ", us/draw full {:.2f} (resolve {:.2f}), delta {:.2f} ({:.2f}), template {:.2f} "
            "({:.2f})",
            us(path_ticks[0], paths[0]), us(resolve_ticks[0], paths[0]),
            us(path_ticks[1], paths[1]), us(resolve_ticks[1], paths[1]),
            us(path_ticks[2], paths[2]), us(resolve_ticks[2], paths[2]));
    }
    return fmt::format(
        "drawrec {:.0f}/s (full {:.0f}, delta {:.0f}, template {:.0f}){}, reused ubo {:.0f}/s tbo "
        "{:.0f}/s, nearest bases {:.0f}/s, commit-only/s:{}, stale {:.0f}/s, deleted loops "
        "{:.0f}/s, verify {} draws, mismatches:{}, dependent {}, purity violations {}",
        rate(draws), rate(paths[0]), rate(paths[1]), rate(paths[2]), timing,
        rate(take(stats.reused_ubos)), rate(take(stats.reused_tbos)),
        rate(take(stats.nearest_bases)),
        commit_only.empty() ? std::string{" none"} : commit_only, rate(take(stats.stale)),
        rate(take(stats.deleted_loops)), take(stats.verify_draws),
        mismatches.empty() ? std::string{" none"} : mismatches, take(stats.dependent),
        take(stats.purity_violations));
}

} // namespace VideoCommon::V2::Resolve
