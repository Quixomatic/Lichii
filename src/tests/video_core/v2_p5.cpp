// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/commit.h"
#include "video_core/v2/epoch_publish.h"
#include "video_core/v2/va_map.h"

namespace {

namespace Commit = VideoCommon::V2::Commit;
using Commit::Kind;
using Commit::Request;
using Commit::RequestList;
using Commit::Resource;

Request MakeRequest(Kind kind, u32 slot, u64 address = 0) {
    Request request{};
    request.kind = kind;
    request.slot = slot;
    request.address = address;
    request.resource = Resource::ImageView;
    return request;
}

} // Anonymous namespace

TEST_CASE("V2 P5: requests apply in sequence order and patch their proxies",
          "[video_core][v2_p5]") {
    Commit::Stage stage;
    RequestList list;
    list.Reset(7);
    const u32 created = list.Push(MakeRequest(Kind::ViewCreate, Commit::InvalidSlot, 3), true);
    REQUIRE(Commit::IsProxy(created));
    REQUIRE(list.Resolve(created) == Commit::InvalidSlot);
    // A prepare of the view the first request creates: it names the proxy.
    list.Push(MakeRequest(Kind::ImagePrepare, created));
    list.Push(MakeRequest(Kind::ImagePrepare, 42));
    REQUIRE(list.Requests()[1].sequence == Commit::MakeSequence(7, 1));
    REQUIRE(Commit::SegmentOf(list.Requests()[2].sequence) == 7);

    std::vector<std::pair<Kind, u32>> applied;
    stage.Apply(list, [&](const Request& request) {
        applied.emplace_back(request.kind, request.slot);
        return request.kind == Kind::ViewCreate ? 100U : request.slot;
    });
    REQUIRE(applied.size() == 3);
    REQUIRE(applied[0].first == Kind::ViewCreate);
    // The second request was patched with the first one's result before it was applied.
    REQUIRE(applied[1] == std::pair{Kind::ImagePrepare, 100U});
    REQUIRE(applied[2] == std::pair{Kind::ImagePrepare, 42U});
    REQUIRE(list.Resolve(created) == 100);
    REQUIRE(list.Resolve(42) == 42);
    REQUIRE(list.AllPatched());
    const auto counters = stage.TakeCounters();
    REQUIRE(counters.proxies_patched == 1);
    REQUIRE(counters.requests[static_cast<size_t>(Kind::ImagePrepare)] == 2);
}

TEST_CASE("V2 P5: segments from many shards commit strictly in segment order",
          "[video_core][v2_p5]") {
    constexpr size_t Segments = 400;
    constexpr size_t Shards = 4;
    Commit::Stage stage;
    std::vector<RequestList> lists(Segments);
    for (size_t segment = 0; segment < Segments; ++segment) {
        lists[segment].Reset(segment);
        for (u32 index = 0; index < 1 + segment % 5; ++index) {
            lists[segment].Push(MakeRequest(Kind::BufferUpload, index, segment), index == 0);
        }
    }
    // Shards finish segments in scrambled order and submit from their own threads.
    std::vector<std::thread> threads;
    for (size_t shard = 0; shard < Shards; ++shard) {
        threads.emplace_back([&, shard] {
            std::vector<size_t> mine;
            for (size_t segment = shard; segment < Segments; segment += Shards) {
                mine.push_back(segment);
            }
            std::mt19937 rng{static_cast<u32>(shard)};
            std::ranges::shuffle(mine, rng);
            for (const size_t segment : mine) {
                stage.Submit(&lists[segment]);
            }
        });
    }
    std::vector<u64> order;
    size_t committed = 0;
    while (committed < Segments) {
        for (RequestList* const list : stage.CommitReady([&](const Request& request) {
                 order.push_back(request.sequence);
                 return static_cast<u32>(request.address);
             })) {
            REQUIRE(list->Segment() == committed);
            REQUIRE(list->AllPatched());
            ++committed;
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(std::ranges::is_sorted(order));
    REQUIRE(stage.NextSegment() == Segments);
}

TEST_CASE("V2 P5: external queue keeps each producer's order and floods when full",
          "[video_core][v2_p5]") {
    Commit::ExternalQueue queue;
    REQUIRE(queue.Empty());
    constexpr u32 Producers = 4;
    constexpr u32 PerProducer = 20000;
    std::atomic<u32> done{0};
    std::vector<std::thread> threads;
    std::atomic<u64> flooded_ops{0};
    for (u32 producer = 0; producer < Producers; ++producer) {
        threads.emplace_back([&, producer] {
            for (u32 index = 0; index < PerProducer; ++index) {
                const Commit::ExternalOp op{
                    .address = (u64{producer} << 32) | index,
                    .size = 1,
                    .kind = Commit::ExternalKind::CpuWrite,
                    .arg = producer,
                };
                if (!queue.TryPush(op)) {
                    queue.Flood(op.address, op.size);
                    flooded_ops.fetch_add(1);
                }
            }
            done.fetch_add(1);
        });
    }
    std::array<s64, Producers> last;
    last.fill(-1);
    u64 received = 0;
    const auto drain = [&] {
        received += queue.Drain([&](const Commit::ExternalOp& op) {
            const s64 index = static_cast<s64>(op.address & 0xffffffff);
            // Per producer, pops come in push order (gaps only where a push flooded).
            REQUIRE(index > last[op.arg]);
            last[op.arg] = index;
        });
    };
    while (done.load() != Producers) {
        drain();
    }
    drain();
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(received + flooded_ops.load() == u64{Producers} * PerProducer);
    u64 address = 0;
    u64 size = 0;
    if (flooded_ops.load() != 0) {
        REQUIRE(queue.TakeFlood(address, size));
        REQUIRE(size != 0);
    }
    REQUIRE_FALSE(queue.TakeFlood(address, size));
    REQUIRE(queue.Empty());

    // A full queue rejects, the flood widens to the union of what didn't fit.
    for (size_t index = 0; index < Commit::ExternalQueue::Capacity; ++index) {
        REQUIRE(queue.TryPush({.address = index, .size = 1}));
    }
    REQUIRE_FALSE(queue.TryPush({.address = 0x5000, .size = 0x10}));
    queue.Flood(0x5000, 0x10);
    queue.Flood(0x1000, 0x10);
    REQUIRE_FALSE(queue.Empty());
    REQUIRE(queue.TakeFlood(address, size));
    REQUIRE(address == 0x1000);
    REQUIRE(size == 0x4010);
}

TEST_CASE("V2 P5: the bounded MPMC slot queue hands every value out once",
          "[video_core][v2_p5]") {
    Commit::BoundedMpmc<u32, 1024> queue;
    for (u32 slot = 0; slot < 1024; ++slot) {
        REQUIRE(queue.TryPush(slot));
    }
    REQUIRE_FALSE(queue.TryPush(5000));
    // Four threads take slots and give them back; every slot is held by one thread at a time.
    std::vector<std::atomic<u32>> holders(1024);
    std::atomic<u64> bad{0};
    std::vector<std::thread> threads;
    for (u32 index = 0; index < 4; ++index) {
        threads.emplace_back([&] {
            for (u32 round = 0; round < 50000; ++round) {
                u32 slot = 0;
                if (!queue.TryPop(slot)) {
                    continue;
                }
                if (holders[slot].fetch_add(1) != 0) {
                    bad.fetch_add(1);
                }
                holders[slot].fetch_sub(1);
                while (!queue.TryPush(slot)) {
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    REQUIRE(bad.load() == 0);
    std::vector<bool> seen(1024);
    u32 slot = 0;
    u32 count = 0;
    while (queue.TryPop(slot)) {
        REQUIRE_FALSE(seen[slot]);
        seen[slot] = true;
        ++count;
    }
    REQUIRE(count == 1024);
}

TEST_CASE("V2 P5: the first owner claim waits for direct external users",
          "[video_core][v2_p5]") {
    Commit::Owner owner;
    REQUIRE_FALSE(owner.HasOwner());
    REQUIRE(owner.BeginDirect());
    std::atomic<bool> claimed{false};
    std::thread claimer([&] {
        owner.Claim();
        claimed.store(true);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    // Still inside the direct section: the claim has not finished.
    REQUIRE_FALSE(claimed.load());
    owner.EndDirect();
    claimer.join();
    REQUIRE(claimed.load());
    REQUIRE(owner.HasOwner());
    // From now on external threads route through the owner.
    REQUIRE_FALSE(owner.BeginDirect());
}

TEST_CASE("V2 P5: the VA map serves lock-free readers while one writer remaps",
          "[video_core][v2_p5]") {
    // 64 KiB pages over 36 bits, as the buffer cache's page table.
    using Map = VideoCommon::V2::VaMap<16, 36>;
    Map map;
    REQUIRE(map.Load(0x123456) == 0);
    map.Store(0x10000, 0x30000, 5);
    REQUIRE(map.Load(0x10000) == 5);
    REQUIRE(map.Load(0x2ffff) == 5);
    REQUIRE(map.Load(0x30000) == 0);
    REQUIRE(map.Uniform(0x10000, 0x20000) == 5);
    REQUIRE(map.Uniform(0x10000, 0x20001) == 0);
    map.Store(0x10000, 0x30000, 0);
    REQUIRE(map.Load(0x10000) == 0);
    REQUIRE(map.AllocatedLeaves() == 1);

    // Stress: the writer maps ranges whose value names their first page and length; readers
    // must only ever see 0 or a value whose range covers the page they read.
    constexpr u64 Pages = 1 << 14;
    std::atomic<bool> stop{false};
    std::atomic<u64> checks{0};
    std::atomic<u64> bad{0};
    const auto encode = [](u64 first, u64 count) {
        return static_cast<u32>((first << 8) | count);
    };
    std::vector<std::thread> readers;
    for (u32 reader = 0; reader < 3; ++reader) {
        readers.emplace_back([&, reader] {
            std::mt19937_64 rng{reader + 1};
            while (!stop.load(std::memory_order_relaxed)) {
                const u64 page = rng() % Pages;
                const u32 value = map.Load(page << 16);
                if (value != 0) {
                    const u64 first = value >> 8;
                    const u64 count = value & 0xff;
                    if (page < first || page >= first + count) {
                        bad.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                checks.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    std::mt19937_64 rng{99};
    std::map<u64, u64> mapped; // first page -> count
    for (u32 step = 0; step < 200000; ++step) {
        const u64 first = rng() % (Pages - 255);
        const u64 count = 1 + rng() % 255;
        // Unmap whatever overlaps, then map the new range (the buffer cache's join/insert).
        for (auto it = mapped.begin(); it != mapped.end();) {
            if (it->first < first + count && first < it->first + it->second) {
                map.Store(it->first << 16, (it->first + it->second) << 16, 0);
                it = mapped.erase(it);
            } else {
                ++it;
            }
        }
        map.Store(first << 16, (first + count) << 16, encode(first, count));
        mapped.emplace(first, count);
    }
    stop.store(true);
    for (auto& thread : readers) {
        thread.join();
    }
    REQUIRE(bad.load() == 0);
    REQUIRE(checks.load() > 0);
}

TEST_CASE("V2 P5: the epoch domain frees retired objects only after readers leave",
          "[video_core][v2_p5]") {
    VideoCommon::V2::EpochDomain domain;
    const auto reader = domain.Register();
    REQUIRE(reader.Valid());
    static std::atomic<int> freed{0};
    freed = 0;
    int object = 0;
    domain.Enter(reader);
    domain.Retire(&object, [](void*) { freed.fetch_add(1); });
    REQUIRE(freed.load() == 0); // the reader may hold it
    domain.Exit(reader);
    REQUIRE(domain.Reclaim() == 1);
    REQUIRE(freed.load() == 1);
    // Retired while no reader is inside: freed at once.
    domain.Retire(&object, [](void*) { freed.fetch_add(1); });
    REQUIRE(freed.load() == 2);
    // A reader entering after the retirement doesn't hold it back.
    domain.Enter(reader);
    domain.Retire(&object, [](void*) { freed.fetch_add(1); });
    REQUIRE(freed.load() == 2);
    domain.Exit(reader);
    domain.Reclaim();
    REQUIRE(freed.load() == 3);
    domain.Unregister(reader);
}

TEST_CASE("V2 P5: the published map stays consistent for readers while the writer grows it",
          "[video_core][v2_p5]") {
    struct Key {
        std::array<u64, 4> words;
    };
    const auto hash = [](const Key& key) {
        u64 h = 0x9e3779b97f4a7c15ULL;
        for (const u64 word : key.words) {
            h = (h ^ word) * 0xff51afd7ed558ccdULL;
        }
        return h ^ (h >> 29);
    };
    const auto make_key = [](u64 id) { return Key{{id, id * 3, ~id, id ^ 0x55}}; };
    VideoCommon::V2::EpochDomain domain;
    VideoCommon::V2::PublishedMap<Key> map{domain, 16};
    const Key a = make_key(1);
    REQUIRE(map.Find(a, hash(a)) == map.Missing);
    map.Insert(a, hash(a), 10);
    REQUIRE(map.Find(a, hash(a)) == 10);
    REQUIRE(map.Erase(a, hash(a)));
    REQUIRE(map.Find(a, hash(a)) == map.Missing);
    map.Insert(a, hash(a), 11);
    REQUIRE(map.Find(a, hash(a)) == 11);
    REQUIRE(map.EraseIf([](u32 value) { return value == 11; }) == 1);
    REQUIRE(map.Size() == 0);

    // Stress: values are id * 2 + 1 for key id; readers must see Missing or exactly that.
    constexpr u64 Ids = 5000;
    std::atomic<bool> stop{false};
    std::atomic<u64> bad{0};
    std::atomic<u64> found{0};
    std::vector<std::thread> readers;
    for (u32 index = 0; index < 3; ++index) {
        readers.emplace_back([&, index] {
            const auto slot = domain.Register();
            std::mt19937_64 rng{index + 7};
            while (!stop.load(std::memory_order_relaxed)) {
                const u64 id = rng() % Ids;
                const Key key = make_key(id);
                VideoCommon::V2::EpochGuard guard{domain, slot};
                const u32 value = map.Find(key, hash(key));
                if (value != map.Missing) {
                    found.fetch_add(1, std::memory_order_relaxed);
                    if (value != id * 2 + 1) {
                        bad.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            domain.Unregister(slot);
        });
    }
    std::mt19937_64 rng{1234};
    for (u32 round = 0; round < 20; ++round) {
        for (u64 id = 0; id < Ids; ++id) {
            const Key key = make_key(id);
            map.Insert(key, hash(key), static_cast<u32>(id * 2 + 1));
        }
        for (u64 id = 0; id < Ids; ++id) {
            if (rng() % 3 == 0) {
                const Key key = make_key(id);
                map.Erase(key, hash(key));
            }
        }
        if (round % 5 == 4) {
            map.Clear();
        }
    }
    stop.store(true);
    for (auto& thread : readers) {
        thread.join();
    }
    domain.Reclaim();
    REQUIRE(bad.load() == 0);
    REQUIRE(domain.PendingRetired() == 0);
}

TEST_CASE("V2 P5: prepare prediction and the render target skip rule", "[video_core][v2_p5]") {
    using Commit::PredictPrepare;
    REQUIRE(PredictPrepare({.prepares = false, .cpu_modified = true}) == 0);
    REQUIRE(PredictPrepare({.prepares = true, .cpu_modified = true}) == Commit::EffectUpload);
    REQUIRE(PredictPrepare({.prepares = true, .cpu_modified = true, .newer_alias = true,
                            .invalidate = true}) == 0);
    REQUIRE(PredictPrepare({.prepares = true, .newer_alias = true, .modification = true}) ==
            (Commit::EffectAlias | Commit::EffectModify));

    Commit::RenderTargetState steady{
        .generation_matches = true,
        .cpu_modified = false,
        .newer_alias = false,
        .gpu_modified = true,
        .newest_modification = true,
        .special = false,
        .touched_this_frame = true,
    };
    REQUIRE_FALSE(Commit::RenderTargetNeedsPrepare(steady));
    for (int field = 0; field < 7; ++field) {
        Commit::RenderTargetState changed = steady;
        switch (field) {
        case 0:
            changed.generation_matches = false;
            break;
        case 1:
            changed.cpu_modified = true;
            break;
        case 2:
            changed.newer_alias = true;
            break;
        case 3:
            changed.gpu_modified = false;
            break;
        case 4:
            changed.newest_modification = false;
            break;
        case 5:
            changed.special = true;
            break;
        default:
            changed.touched_this_frame = false;
            break;
        }
        REQUIRE(Commit::RenderTargetNeedsPrepare(changed));
    }
}

namespace {

// A small model of the texture cache's alias resolution: images carry modification ticks and
// alias lists; preparing an image copies its newer aliases in tick order (SynchronizeAliases).
struct ModelImage {
    u64 tick = 0;
    bool cpu_modified = false;
    std::vector<u32> aliases;
};

struct Model {
    std::vector<ModelImage> images;
    std::vector<s32> views; ///< descriptor -> image, -1 = no view yet
    u64 clock = 0;
    std::vector<std::pair<u32, u32>> copies; ///< (destination, source)
    std::vector<Commit::LogEntry>* log = nullptr;

    bool NewerAlias(u32 image) const {
        return std::ranges::any_of(images[image].aliases,
                                   [&](u32 alias) { return images[alias].tick > images[image].tick; });
    }

    // The "old path" step: visit a descriptor (create on miss, which aliases the new image with
    // `alias_with`), then prepare it, optionally drawing into it (modification).
    u32 Visit(u32 descriptor, u32 alias_with, bool modification) {
        Commit::Kind kind = Commit::Kind::ImagePrepare;
        if (views[descriptor] < 0) {
            kind = Commit::Kind::ViewCreate;
            const u32 image = static_cast<u32>(images.size());
            images.push_back({.tick = 0, .cpu_modified = true, .aliases = {alias_with}});
            images[alias_with].aliases.push_back(image);
            views[descriptor] = static_cast<s32>(image);
        }
        const u32 image = static_cast<u32>(views[descriptor]);
        if (log != nullptr) {
            log->push_back({kind, Resource::ImageView, image, 0});
        }
        u32 effects = kind == Commit::Kind::ViewCreate ? u32{Commit::EffectCreate} : 0U;
        if (images[image].cpu_modified) {
            images[image].cpu_modified = false;
            effects |= Commit::EffectUpload;
        }
        std::vector<u32> newer;
        for (const u32 alias : images[image].aliases) {
            if (images[alias].tick > images[image].tick) {
                newer.push_back(alias);
            }
        }
        if (!newer.empty()) {
            effects |= Commit::EffectAlias;
            std::ranges::sort(newer, {}, [&](u32 alias) { return images[alias].tick; });
            for (const u32 alias : newer) {
                copies.emplace_back(image, alias);
            }
            images[image].tick = images[newer.back()].tick;
        }
        if (modification) {
            images[image].tick = ++clock;
            effects |= Commit::EffectModify;
        }
        if (log != nullptr) {
            log->back().effects = effects;
        }
        return image;
    }
};

struct Step {
    u32 descriptor;
    u32 alias_with;
    bool modification;
};

} // Anonymous namespace

TEST_CASE("V2 P5: alias resolution keeps its order through lookups and commit",
          "[video_core][v2_p5]") {
    std::mt19937 rng{5};
    for (int trial = 0; trial < 300; ++trial) {
        Model base;
        base.images.resize(4);
        base.views.assign(12, -1);
        for (u32 image = 0; image < 4; ++image) {
            base.views[image] = static_cast<s32>(image);
            base.images[image].tick = rng() % 4;
        }
        base.images[0].aliases = {1, 2};
        base.images[1].aliases = {0};
        base.images[2].aliases = {0, 3};
        base.images[3].aliases = {2};
        base.clock = 8;
        std::vector<Step> steps;
        for (int index = 0; index < 6; ++index) {
            steps.push_back({static_cast<u32>(rng() % 12), static_cast<u32>(rng() % 4),
                             rng() % 3 == 0});
        }

        // Old path: mutations inside each lookup.
        Model old_model = base;
        std::vector<Commit::LogEntry> old_log;
        old_model.log = &old_log;
        for (const Step& step : steps) {
            old_model.Visit(step.descriptor, step.alias_with, step.modification);
        }

        // New path: pure lookups predict a request per step (a proxy for a missing view), then
        // the commit applies them in order with the same ensure-semantics applier.
        Model new_model = base;
        RequestList list;
        list.Reset(1);
        std::vector<s32> seen = new_model.views; // what the lookups see: no mutation yet
        std::vector<u32> prepared; // images an earlier request of this list prepares
        for (const Step& step : steps) {
            Request request = MakeRequest(Commit::Kind::ImagePrepare, 0, step.descriptor);
            request.arg = step.modification ? Commit::ArgModification : 0;
            request.size = step.alias_with;
            bool proxy = false;
            if (seen[step.descriptor] < 0) {
                request.kind = Commit::Kind::ViewCreate;
                request.slot = Commit::InvalidSlot;
                proxy = true;
            } else {
                const u32 image = static_cast<u32>(seen[step.descriptor]);
                request.slot = image;
                // An image an earlier request prepares has nothing left to upload or copy in.
                const bool again = std::ranges::find(prepared, image) != prepared.end();
                request.effects = Commit::PredictPrepare({
                    .prepares = true,
                    .cpu_modified = !again && new_model.images[image].cpu_modified,
                    .newer_alias = !again && new_model.NewerAlias(image),
                    .modification = step.modification,
                });
                prepared.push_back(image);
            }
            list.Push(request, proxy);
        }
        Commit::Stage stage;
        std::vector<Commit::LogEntry> new_log;
        new_model.log = &new_log;
        stage.Apply(list, [&](const Request& request) {
            return new_model.Visit(static_cast<u32>(request.address),
                                   static_cast<u32>(request.size),
                                   (request.arg & Commit::ArgModification) != 0);
        });
        // Identical mutations in identical order.
        REQUIRE(new_model.copies == old_model.copies);
        REQUIRE(new_log.size() == old_log.size());
        for (size_t index = 0; index < new_log.size(); ++index) {
            REQUIRE(new_log[index].kind == old_log[index].kind);
            REQUIRE(new_log[index].slot == old_log[index].slot);
            REQUIRE(new_log[index].effects == old_log[index].effects);
        }
        // The predictions match the old path, except after a create or alias copy changed what
        // later lookups saw (counted as dependent, never as a mismatch).
        const Commit::CompareResult result = Commit::Compare(list.Requests(), old_log);
        REQUIRE(result.mismatches == 0);
        REQUIRE(result.compared == steps.size());
    }
}

TEST_CASE("V2 P5: verify compare classifies mismatches", "[video_core][v2_p5]") {
    std::vector<Request> predicted{MakeRequest(Kind::ImagePrepare, 1),
                                   MakeRequest(Kind::ImagePrepare, 2)};
    predicted[0].effects = Commit::EffectUpload;
    std::vector<Commit::LogEntry> actual{{Kind::ImagePrepare, Resource::ImageView, 1,
                                          Commit::EffectUpload},
                                         {Kind::ImagePrepare, Resource::ImageView, 2, 0}};
    auto result = Commit::Compare(predicted, actual);
    REQUIRE(result.mismatches == 0);
    // A wrong resource is a mismatch.
    actual[1].slot = 3;
    result = Commit::Compare(predicted, actual);
    REQUIRE(result.mismatches == 1);
    REQUIRE(result.first_mismatch == 1);
    REQUIRE(result.field == Commit::CompareResult::Field::Slot);
    // After a create, it is dependent.
    actual[0].effects |= Commit::EffectCreate;
    result = Commit::Compare(predicted, actual);
    REQUIRE(result.mismatches == 0);
    REQUIRE(result.dependent == 1);
    // A missing step is a count mismatch; a proxy matches any slot.
    predicted[1].slot = Commit::MakeProxy(0);
    actual[0].effects = Commit::EffectUpload;
    actual.pop_back();
    result = Commit::Compare(predicted, actual);
    REQUIRE(result.mismatches == 1);
    REQUIRE(result.field == Commit::CompareResult::Field::Count);
    // Effects that differ are a mismatch.
    actual.push_back({Kind::ImagePrepare, Resource::ImageView, 9, 0});
    actual[0].effects = 0;
    result = Commit::Compare(predicted, actual);
    REQUIRE(result.mismatches == 1);
    REQUIRE(result.field == Commit::CompareResult::Field::Effects);
}
