// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "common/common_types.h"
#include "video_core/v2/ir.h"

// Pipeline v2 IR traces (MASTER_PLAN §5.2 "replayable for tests and traces", §13 "IR traces
// recorded once and replayed deterministically"). EDEN_NX2_V2_IR_TRACE=<dir> makes the split's
// back append every IR record it executes, with its payload, and write one file per frame
// (`ir_frame_<n>.irt`, cut at Present records). A replayer (GpuSplit::ReplayIrRecord) puts the
// records back into a split's ring and arena for its back to execute; object references are the
// recording session's ObjectTable indices, so a replay interns the same objects in order.
//
// File format: u64 words. Per record: a header word (record words | payload words << 32), the
// record's ring words, then its payload words (its PayloadRef is rewritten at replay).
namespace VideoCommon::V2::IrTrace {

/// The directory from EDEN_NX2_V2_IR_TRACE (empty: off), read once.
[[nodiscard]] const std::string& Directory();

class Recorder {
public:
    /// One executed record: `record` (ring words, PayloadRef included) and its payload.
    void Add(std::span<const u64> record, std::span<const u64> payload) {
        words.push_back(u64{record.size()} | (u64{payload.size()} << 32));
        words.insert(words.end(), record.begin(), record.end());
        words.insert(words.end(), payload.begin(), payload.end());
        ++records;
    }
    [[nodiscard]] const std::vector<u64>& Words() const {
        return words;
    }
    [[nodiscard]] u64 Records() const {
        return records;
    }
    void Clear() {
        words.clear();
        records = 0;
    }
    /// Writes the words to `path`; returns whether it could.
    bool Write(const std::string& path) const;

private:
    std::vector<u64> words;
    u64 records{};
};

/// One record of a trace, decoded.
struct TraceRecord {
    std::span<const u64> record;
    std::span<const u64> payload;
};

/// Calls `visit(TraceRecord)` for every record of `words`; returns false on a malformed trace.
template <typename Visit>
bool ForEach(std::span<const u64> words, Visit&& visit) {
    size_t at = 0;
    while (at < words.size()) {
        const u64 header = words[at++];
        const size_t record_words = static_cast<size_t>(header & 0xffffffff);
        const size_t payload_words = static_cast<size_t>(header >> 32);
        if (record_words < Ir::HeaderWords || at + record_words + payload_words > words.size() ||
            Ir::RecordWords(words[at]) != record_words) {
            return false;
        }
        visit(TraceRecord{words.subspan(at, record_words),
                          words.subspan(at + record_words, payload_words)});
        at += record_words + payload_words;
    }
    return true;
}

/// Reads a trace file (empty on failure).
[[nodiscard]] std::vector<u64> Read(const std::string& path);

} // namespace VideoCommon::V2::IrTrace
