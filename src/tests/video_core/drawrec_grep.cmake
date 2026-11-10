# SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
# SPDX-License-Identifier: GPL-3.0-or-later

# Pipeline v2 P6 Step 0 (step 20): the resolve section of vk_draw_resolve.cpp (between
# "// RESOLVE BEGIN" and "// RESOLVE END") reads only the state view, guest memory and const
# cache lookups: no engine, no dirty flags, no recording.
# Usage: cmake -DSOURCE=<path to vk_draw_resolve.cpp> -P drawrec_grep.cmake

if(NOT DEFINED SOURCE)
    message(FATAL_ERROR "drawrec_grep: SOURCE is not set")
endif()
file(READ "${SOURCE}" text)
string(FIND "${text}" "// RESOLVE BEGIN" begin)
string(FIND "${text}" "// RESOLVE END" end)
if(begin EQUAL -1 OR end EQUAL -1 OR end LESS begin)
    message(FATAL_ERROR "drawrec_grep: the RESOLVE BEGIN/END markers are missing in ${SOURCE}")
endif()
math(EXPR length "${end} - ${begin}")
string(SUBSTRING "${text}" ${begin} ${length} section)
set(failed FALSE)
foreach(pattern "maxwell3d" "dirty.flags" "scheduler." "Record(")
    string(FIND "${section}" "${pattern}" found)
    if(NOT found EQUAL -1)
        message(SEND_ERROR "drawrec_grep: the resolve section contains \"${pattern}\"")
        set(failed TRUE)
    endif()
endforeach()
if(failed)
    message(FATAL_ERROR "drawrec_grep: the resolve section is not engine-free")
endif()
message(STATUS "drawrec_grep: the resolve section is engine-free")
