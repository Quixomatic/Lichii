// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cstdio>
#include <cstdlib>

#include "video_core/v2/ir_trace.h"

namespace VideoCommon::V2::IrTrace {

const std::string& Directory() {
    static const std::string directory = [] {
        const char* const value = std::getenv("EDEN_NX2_V2_IR_TRACE");
        return std::string{value != nullptr ? value : ""};
    }();
    return directory;
}

bool Recorder::Write(const std::string& path) const {
    std::FILE* const file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        return false;
    }
    const size_t written = std::fwrite(words.data(), sizeof(u64), words.size(), file);
    std::fclose(file);
    return written == words.size();
}

std::vector<u64> Read(const std::string& path) {
    std::vector<u64> words;
    std::FILE* const file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return words;
    }
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (bytes > 0 && bytes % sizeof(u64) == 0) {
        words.resize(static_cast<size_t>(bytes) / sizeof(u64));
        if (std::fread(words.data(), sizeof(u64), words.size(), file) != words.size()) {
            words.clear();
        }
    }
    std::fclose(file);
    return words;
}

} // namespace VideoCommon::V2::IrTrace
