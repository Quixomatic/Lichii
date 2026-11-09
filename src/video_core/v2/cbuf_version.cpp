// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/v2/cbuf_version.h"

namespace VideoCommon::V2::CbufVer {

void WriteLog::NoteInline(DAddr begin, DAddr end) {
    if (begin >= end) {
        return;
    }
    const u64 index = sequence.load(std::memory_order_relaxed);
    ranges[index % LogSize] = Range{begin, end};
    sequence.store(index + 1, std::memory_order_release);
}

void WriteLog::NoteHost(DAddr address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 first = address >> PageBits;
    const u64 last = (address + size - 1) >> PageBits;
    if (last - first >= MaxHostPages) {
        NoteBoundary();
        return;
    }
    for (u64 page = first; page <= last; ++page) {
        pages[page % PageBuckets].fetch_add(1, std::memory_order_release);
    }
}

u64 WriteLog::PageSum(DAddr address, u32 size) const {
    if (size == 0) {
        return 0;
    }
    const u64 first = address >> PageBits;
    const u64 last = (address + size - 1) >> PageBits;
    u64 sum = 0;
    for (u64 page = first; page <= last; ++page) {
        // Mixed with the page so two buckets moving by the same amount don't cancel out.
        sum += (static_cast<u64>(pages[page % PageBuckets].load(std::memory_order_acquire)) + 1) *
               (page | 1);
    }
    return sum;
}

bool WriteLog::WrittenSince(u64 since, DAddr begin, DAddr end) const {
    const u64 now = sequence.load(std::memory_order_acquire);
    if (now - since > LogSize) {
        return true;
    }
    for (u64 index = since; index < now; ++index) {
        const Range& range = ranges[index % LogSize];
        if (range.begin < end && begin < range.end) {
            return true;
        }
    }
    return false;
}

void WriteLog::Reset() {
    sequence.store(0, std::memory_order_relaxed);
    boundary.fetch_add(1, std::memory_order_relaxed);
}

WriteLog& Log() {
    static WriteLog log;
    return log;
}

} // namespace VideoCommon::V2::CbufVer
