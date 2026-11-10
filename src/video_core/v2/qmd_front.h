// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <thread>
#include <utility>

#include "common/common_types.h"
#include "video_core/native_qmd.h"

// P5b (EDEN_NX2_V2_QMDSCHED): the split front's model of the back's native QMD scheduler (#805's
// FrontQmdView). It runs the same dependency logic (NativeQmdScheduler::SignalOn) on its own
// table, fed the same signals in the same order, so it knows on the front how many QMDs each
// signal launches (exact Dispatch record counts) and which QMDs they are (bounded hazard table
// writes). A QMD the front can't read for sure (bytes a queued write may change) makes the model
// unknown until the next drain, where it copies the back's table (the back is idle then).
namespace VideoCommon::V2::QmdFront {

using Scheduler = Tegra::NativeQmdScheduler;

struct SignalResult {
    bool valid = false;
    u32 dispatches = 0;
};

enum class ReadResult : u8 {
    Ok,
    Unmapped, ///< the back sees the same: an error that stops the chain there
    Unknown,  ///< the front can't know the words: the model becomes unknown
};

class Model {
public:
    /// Runs one QMD signal (method 0xae/0xaf/0xb0 and its argument, already filtered as the back
    /// filters it). `read(address, descriptor&) -> ReadResult`, `mapped(address, size) -> bool`,
    /// `on_dispatch(address, const Descriptor&)`, `on_release(address, payload, flags)`.
    template <typename Read, typename Mapped, typename OnDispatch, typename OnRelease>
    SignalResult Signal(u64 qmd_address, u32 argument, Read&& read, Mapped&& mapped,
                        OnDispatch&& on_dispatch, OnRelease&& on_release) {
        if (!valid) {
            return {};
        }
        bool unknown = false;
        u32 dispatches = 0;
        (void)runner.SignalOn(
            table, qmd_address, argument,
            [&](u64 address, Scheduler::Descriptor& descriptor) {
                switch (read(address, descriptor)) {
                case ReadResult::Ok:
                    return true;
                case ReadResult::Unmapped:
                    return false;
                case ReadResult::Unknown:
                    unknown = true;
                    return false;
                }
                return false;
            },
            [&](u64 address, u32 payload, u32 flags) {
                const bool four_words = (flags >> 30) == 0;
                if (!mapped(address, four_words ? 16 : 4)) {
                    return false;
                }
                on_release(address, payload, flags);
                return true;
            },
            [&](u64 address, const Scheduler::Descriptor& qmd) {
                ++dispatches;
                on_dispatch(address, qmd);
                return true;
            });
        if (unknown) {
            valid = false;
            ++invalidations;
            return {};
        }
        return {true, dispatches};
    }

    /// The front lost track (a read it can't trust): unknown until Resync.
    void Invalidate() noexcept {
        valid = false;
        ++invalidations;
    }

    /// Copies the back's table (call only while the back is idle: a drain).
    void Resync(const Scheduler::Table& back) {
        table = back;
        valid = true;
        ++resyncs;
    }
    /// Takes a copy of the back's table made when the back had run every signal the model saw
    /// (Fix F3h: a back job, no drain).
    void ResyncFrom(Scheduler::Table&& back) {
        table = std::move(back);
        valid = true;
        ++resyncs;
    }
    /// Front: the ring position after the newest signal the model saw was sent.
    void NoteSignalSent(u64 position) noexcept {
        last_signal_sent = position;
    }
    [[nodiscard]] u64 LastSignalSent() const noexcept {
        return last_signal_sent;
    }

    [[nodiscard]] bool Valid() const noexcept {
        return valid;
    }
    [[nodiscard]] const Scheduler::Table& Table() const noexcept {
        return table;
    }
    [[nodiscard]] u64 Invalidations() const noexcept {
        return invalidations;
    }
    [[nodiscard]] u64 Resyncs() const noexcept {
        return resyncs;
    }

private:
    Scheduler runner;
    Scheduler::Table table;
    bool valid = true;
    u64 invalidations = 0;
    u64 resyncs = 0;
    u64 last_signal_sent = 0;
};

/// Fix F3h (W-01c, R2-F3-02): whether a QMD read with a release still to land forwarded in
/// (`forwarded`) differs from memory's words (`memory`) only in the release slots (NVN2 patches
/// release payloads into scheduled QMDs): the scheduling is then the same for either.
[[nodiscard]] inline bool OnlyReleaseWordsDiffer(const Scheduler::Descriptor& memory,
                                                 const Scheduler::Descriptor& forwarded) {
    for (size_t word = 0; word < memory.size(); ++word) {
        if (memory[word] == forwarded[word]) {
            continue;
        }
        bool release_word = false;
        for (const auto base : Scheduler::ReleaseSlots) {
            release_word = release_word || (word >= base && word < base + 4);
        }
        if (!release_word) {
            return false;
        }
    }
    return true;
}

/// The release words `descriptor` names (GPU address and size), as the scheduler writes them.
template <typename Ranges>
inline void NoteBothReleases(const Scheduler::Descriptor& descriptor, Ranges& releases) {
    for (const auto base : Scheduler::ReleaseSlots) {
        const auto flags = descriptor[base + 1];
        if (!(flags & (1U << 23)) || !Scheduler::IsDecodableRelease(flags)) {
            continue;
        }
        const u64 destination = (u64{flags & 0xff} << 32) | descriptor[base];
        releases.emplace_back(destination, (flags >> 30) == 0 ? 16U : 4U);
    }
}

/// Inline QMDs (method 0x107) the front sent but the back may not have stored yet: the front
/// forwards their words to its own reads (store forwarding for 256-byte QMDs).
class InlineQmds {
public:
    static constexpr size_t Capacity = 64;

    void Note(u64 address, const Scheduler::Descriptor& words) {
        for (auto& entry : entries) {
            if (entry.used && entry.address == address) {
                entry.words = words;
                return;
            }
        }
        Entry& entry = entries[next];
        next = (next + 1) % Capacity;
        entry = {address, words, true};
    }

    [[nodiscard]] const Scheduler::Descriptor* Find(u64 address) const {
        for (const auto& entry : entries) {
            if (entry.used && entry.address == address) {
                return &entry.words;
            }
        }
        return nullptr;
    }

    void Clear() {
        entries = {};
        next = 0;
    }

private:
    struct Entry {
        u64 address = 0;
        Scheduler::Descriptor words{};
        bool used = false;
    };
    std::array<Entry, Capacity> entries{};
    size_t next = 0;
};

/// The split front's model and forwarded inline QMDs (front thread only; one table for every
/// channel, as the back's).
[[nodiscard]] inline Model& FrontModel() {
    static Model model;
    return model;
}
[[nodiscard]] inline InlineQmds& FrontInline() {
    static InlineQmds inline_qmds;
    return inline_qmds;
}

/// The front drained (the back is idle and has run everything): an unknown model resyncs from
/// the back's table, and every inline QMD sent is in memory now.
inline void OnDrain() {
    if (!FrontModel().Valid()) {
        FrontModel().Resync(Scheduler::BackTable());
    }
    FrontInline().Clear();
}

/// Front, between QMD writes (none of the current write's signals sent yet): the signals the
/// model saw were sent up to NoteSignalSent's position. `NoteSignalSent` via the model.
inline void NoteSignalSent(u64 position) {
    FrontModel().NoteSignalSent(position);
}

/// Front: an unknown model resyncs without a drain (the unmapped-acquire drain used to do it):
/// once the back consumed every signal the model saw, a back job (GpuSplit::RequestBackJob)
/// copies the back's table after the back's current entry. No signal is sent meanwhile (the
/// caller's aren't yet), so the copy is the table the model would hold. The front waits for the
/// job only (one entry at most), not for the queue. Split: GpuSplit.
template <typename Split>
inline bool ResyncByJob(Split& split) {
    Model& model = FrontModel();
    if (model.Valid() || split.ConsumedNow() < model.LastSignalSent()) {
        return false;
    }
    struct Job {
        Scheduler::Table table;
        std::atomic<bool> done{false};
    };
    // Static: a job a stopping back never ran must not point at a dead frame.
    static Job job;
    job.done.store(false, std::memory_order_relaxed);
    split.RequestBackJob(
        [](void* argument) {
            auto& self = *static_cast<Job*>(argument);
            self.table = Scheduler::BackTable();
            self.done.store(true, std::memory_order_release);
        },
        &job);
    while (!job.done.load(std::memory_order_acquire) && split.BackRunning()) {
        std::this_thread::yield();
    }
    if (!job.done.load(std::memory_order_acquire)) {
        return false;
    }
    model.ResyncFrom(std::move(job.table));
    job.table.clear();
    return true;
}

} // namespace VideoCommon::V2::QmdFront
