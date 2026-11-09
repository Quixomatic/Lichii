// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <cctype>
#include <cstdlib>
#include <string>

#include "common/logging.h"
#include "video_core/v2/v2_switches.h"

namespace VideoCommon::V2 {

namespace {

std::string Lower(const char* text) {
    std::string value{text};
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

/// Whether the value parsed; `mode` is Off for anything else.
bool Parse(const char* text, Mode& mode) {
    mode = Mode::Off;
    if (text == nullptr) {
        return true;
    }
    const std::string value = Lower(text);
    if (value.empty() || value == "0" || value == "off" || value == "false") {
        return true;
    }
    if (value == "1" || value == "on" || value == "true") {
        mode = Mode::On;
        return true;
    }
    if (value == "verify" || value == "2") {
        mode = Mode::Verify;
        return true;
    }
    return false;
}

bool ForcedV1() {
    const char* pipeline = std::getenv("EDEN_NX2_PIPELINE");
    return pipeline != nullptr && Lower(pipeline) == "v1";
}

std::string VariableName(size_t index) {
    return std::string{"EDEN_NX2_V2_"} + switch_names[index];
}

} // Anonymous namespace

namespace Detail {
std::array<Mode, static_cast<size_t>(Switch::Count)> ReadAll() {
    std::array<Mode, static_cast<size_t>(Switch::Count)> modes{};
    if (ForcedV1()) {
        return modes;
    }
    for (size_t i = 0; i < modes.size(); ++i) {
        Parse(std::getenv(VariableName(i).c_str()), modes[i]);
    }
    return modes;
}
} // namespace Detail

void ReadSwitches() {
    const bool forced_v1 = ForcedV1();
    for (size_t i = 0; i < g_modes.size(); ++i) {
        const std::string name = VariableName(i);
        const char* text = std::getenv(name.c_str());
        Mode mode{};
        if (!Parse(text, mode)) {
            LOG_WARNING(HW_GPU, "{}={} is not 0, 1 or verify: off", name, text);
        }
        if (forced_v1 && mode != Mode::Off) {
            LOG_INFO(HW_GPU, "{}={} ignored: EDEN_NX2_PIPELINE=v1", name, text);
            mode = Mode::Off;
        }
        if (mode != Mode::Off) {
            LOG_INFO(HW_GPU, "Pipeline v2 switch {}={}", name,
                     mode == Mode::Verify ? "verify" : "1");
        }
        g_modes[i] = mode;
    }
    g_track_groups = Detail::TracksGroups(g_modes);
}

} // namespace VideoCommon::V2
