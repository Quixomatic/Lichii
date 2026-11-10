// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// P6 Step 0 (notes/STEP0_SPEC.md): draw records, ResolveDraw / ApplyDraw / EncodeDraw.

#include <cstdlib>
#include <string_view>

#include <catch2/catch_test_macros.hpp>

#include "video_core/v2/v2_switches.h"

namespace V2 = VideoCommon::V2;

namespace {

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // Anonymous namespace

TEST_CASE("V2 DRAWREC: the switch parses 0, 1 and verify", "[video_core][v2_drawrec]") {
    constexpr size_t index = static_cast<size_t>(V2::Switch::DrawRec);
    REQUIRE(std::string_view{V2::switch_names[index]} == "DRAWREC");
    const auto read = [&](const char* value) {
        SetEnv("EDEN_NX2_V2_DRAWREC", value);
        return V2::Detail::ReadAll()[index];
    };
    REQUIRE(read("0") == V2::Mode::Off);
    REQUIRE(read("1") == V2::Mode::On);
    REQUIRE(read("verify") == V2::Mode::Verify);
    REQUIRE(read("bogus") == V2::Mode::Off);
    SetEnv("EDEN_NX2_PIPELINE", "v1");
    REQUIRE(read("1") == V2::Mode::Off);
    SetEnv("EDEN_NX2_PIPELINE", "");
    REQUIRE(read("") == V2::Mode::Off);
}
