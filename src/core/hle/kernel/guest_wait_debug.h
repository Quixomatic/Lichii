// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "common/common_types.h"
#include "core/arm/debug.h"
#include "core/hle/kernel/guest_wait_debug_flag.h"
#include "core/hle/kernel/k_readable_event.h"
#include "core/hle/kernel/k_synchronization_object.h"
#include "core/hle/kernel/k_thread.h"
#include "core/hle/service/kernel_helpers.h"

// SYNC_SUBMIT_SPEC §14.8 H-14.8 step 1: which guest threads block, and on what, when the GPU front
// reports a 2 s semaphore waiter (an early-boot quiet deadlock whose producer is guest-side and
// unnamed: nothing is pending anywhere in the GPU pipeline). The blocking SVCs register the waiting
// thread for the duration of the wait; the dump names each one's SVC, its waited objects (an
// event's debug name, else its type) or address, its PC and how long it has waited. Debugging
// bookkeeping only, and only while `enabled` (guest_wait_debug_flag.h): no wait or wake changes.
namespace Kernel::GuestWaitDebug {

enum class Kind : u8 {
    Synchronization, ///< WaitSynchronization
    Address,         ///< WaitForAddress (address arbiter)
    ConditionVariable, ///< WaitProcessWideKeyAtomic
    Lock,            ///< ArbitrateLock
    Sleep,           ///< SleepThread
};

[[nodiscard]] constexpr const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Synchronization:
        return "WaitSynchronization";
    case Kind::Address:
        return "WaitForAddress";
    case Kind::ConditionVariable:
        return "WaitProcessWideKeyAtomic";
    case Kind::Lock:
        return "ArbitrateLock";
    case Kind::Sleep:
        return "SleepThread";
    }
    return "?";
}

struct Entry {
    KThread* thread{};
    Kind kind{};
    u64 address{};  ///< Address / condition variable / lock address, or the sleep's ns
    u32 num_objects{};
    std::array<KSynchronizationObject*, 4> objects{}; ///< The first waited objects
    std::chrono::steady_clock::time_point since{};
};

struct Registry {
    std::mutex mutex;
    std::vector<Entry> entries;
};

[[nodiscard]] inline Registry& Get() {
    static Registry registry;
    return registry;
}

/// Registers the calling guest thread's wait for the scope's life, only while `enabled` (one
/// relaxed load and no lock otherwise). Declared after the objects' close guard, so it is
/// removed before they are closed (a dump holding the lock keeps them alive).
class Scope {
public:
    Scope(KThread* thread, Kind kind, u64 address,
          std::span<KSynchronizationObject* const> objects = {}) {
        if (!enabled.load(std::memory_order_relaxed)) [[likely]] {
            return;
        }
        thread_ = thread;
        Entry entry{.thread = thread,
                    .kind = kind,
                    .address = address,
                    .num_objects = static_cast<u32>(objects.size()),
                    .since = std::chrono::steady_clock::now()};
        std::copy_n(objects.begin(), (std::min)(objects.size(), entry.objects.size()),
                    entry.objects.begin());
        auto& registry = Get();
        std::scoped_lock lock{registry.mutex};
        registry.entries.push_back(entry);
    }
    ~Scope() {
        if (thread_ == nullptr) [[likely]] {
            return;
        }
        auto& registry = Get();
        std::scoped_lock lock{registry.mutex};
        std::erase_if(registry.entries, [this](const Entry& entry) {
            return entry.thread == thread_;
        });
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    KThread* thread_{};
};

/// The registered waits now (tests, and the dump's source).
[[nodiscard]] inline std::vector<Entry> Entries() {
    auto& registry = Get();
    std::scoped_lock lock{registry.mutex};
    return registry.entries;
}

/// One line per waiting guest thread (oldest wait first).
[[nodiscard]] inline std::string Dump() {
    auto& registry = Get();
    std::scoped_lock lock{registry.mutex};
    std::vector<const Entry*> sorted;
    sorted.reserve(registry.entries.size());
    for (const Entry& entry : registry.entries) {
        sorted.push_back(&entry);
    }
    std::ranges::sort(sorted, [](const Entry* a, const Entry* b) { return a->since < b->since; });
    const auto now = std::chrono::steady_clock::now();
    std::string out;
    for (const Entry* entry : sorted) {
        const KThread* thread = entry->thread;
        const auto name = Core::GetThreadName(thread);
        out += fmt::format("\n  thread {} ({}) pc={:#x} {} for {} ms", thread->GetThreadId(),
                           name.value_or("?"), thread->GetContext().pc, KindName(entry->kind),
                           std::chrono::duration_cast<std::chrono::milliseconds>(now - entry->since)
                               .count());
        if (entry->kind == Kind::Synchronization) {
            out += fmt::format(" on {} object(s):", entry->num_objects);
            for (u32 i = 0; i < (std::min<u32>)(entry->num_objects, 4); ++i) {
                KSynchronizationObject* const object = entry->objects[i];
                if (const auto* event = object->DynamicCast<KReadableEvent*>()) {
                    out += fmt::format(" [event {}]",
                                       Service::KernelHelpers::DebugEventName(event));
                } else {
                    out += fmt::format(" [{}]", object->GetTypeName());
                }
            }
        } else if (entry->kind == Kind::Sleep) {
            out += fmt::format(" ({} ns)", static_cast<s64>(entry->address));
        } else {
            out += fmt::format(" at {:#x}", entry->address);
        }
    }
    return out.empty() ? std::string{" none"} : out;
}

} // namespace Kernel::GuestWaitDebug
