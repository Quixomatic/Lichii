// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <utility>
#include <vector>

// SYNC_SUBMIT_SPEC Step 9 (MASTER_PLAN §6 class D, §5.5 targeted barriers): work that must run
// outside a render pass but that nothing in the open pass reads (query result copies) waits for
// the pass's next natural end or the submission, in the order it was deferred, instead of ending
// the pass. Recording thread only.
namespace VideoCommon::V2 {

class AfterPassList {
public:
    using Job = void (*)(void*);

    /// Defers `job(argument)`; the same job deferred again before it ran is not repeated.
    void Defer(Job job, void* argument) {
        for (const auto& entry : jobs) {
            if (entry.first == job && entry.second == argument) {
                return;
            }
        }
        jobs.emplace_back(job, argument);
    }

    /// Runs every deferred job in order (jobs deferred while running run too, after them).
    void Run() {
        while (!jobs.empty()) [[unlikely]] {
            auto taken = std::move(jobs);
            jobs.clear();
            for (const auto& [job, argument] : taken) {
                job(argument);
            }
        }
    }

    [[nodiscard]] bool Empty() const noexcept {
        return jobs.empty();
    }

private:
    std::vector<std::pair<Job, void*>> jobs;
};

} // namespace VideoCommon::V2
