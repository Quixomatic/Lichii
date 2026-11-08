// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "system_monitor.h"

#include <algorithm>
#include <chrono>
#include <cwchar>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common/polyfill_thread.h"
#include "common/thread.h"

#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <dxgi.h>
#include <pdh.h>
#include <pdhmsg.h>
// clang-format on
#elif defined(__linux__)
#include <filesystem>
#include <fstream>
#include <sstream>
#endif

namespace {

constexpr auto SAMPLE_INTERVAL = std::chrono::milliseconds{500};

#ifdef _WIN32

/**
 * Combines a FILETIME's two 32-bit halves into one 64-bit tick count.
 */
u64 ToU64(const FILETIME& ft) {
    return (u64(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}

/**
 * Returns the part of a PDH GPU counter instance name from "luid_" up to `end_marker`, or to
 * the end of the name. Engine instances look like
 * "pid_1234_luid_0x00000000_0x0000D1B5_phys_0_eng_3_engtype_3D" and adapter memory instances
 * like "luid_0x00000000_0x0000D1B5_phys_0". The result views `name`, so nothing is allocated.
 * Returns an empty view if the name has no luid.
 */
std::wstring_view InstanceKey(std::wstring_view name, std::wstring_view end_marker) {
    const auto begin = name.find(L"luid_");
    if (begin == std::wstring_view::npos) {
        return {};
    }
    const auto end = name.find(end_marker, begin);
    return name.substr(begin, end == std::wstring_view::npos ? end : end - begin);
}

/**
 * Extracts the adapter key ("luid_0x..._0x...") from a PDH GPU counter instance name.
 */
std::wstring_view AdapterKey(std::wstring_view name) {
    return InstanceKey(name, L"_phys");
}

/**
 * Extracts the engine key ("luid_0x..._0x..._phys_0_eng_3") from a PDH GPU engine counter
 * instance name. Engines run independently, so process values are summed per engine and
 * never across engines.
 */
std::wstring_view EngineKey(std::wstring_view name) {
    return InstanceKey(name, L"_engtype");
}

/**
 * Compares two keys, ignoring case, since PDH and DXGI may format hex digits differently.
 */
bool SameKey(std::wstring_view a, std::wstring_view b) {
    return a.size() == b.size() && _wcsnicmp(a.data(), b.data(), a.size()) == 0;
}

/**
 * Formats a DXGI adapter LUID the same way PDH counter instance names do,
 * so DXGI adapters can be matched against PDH counters.
 */
std::wstring AdapterKey(const LUID& luid) {
    wchar_t buf[32];
    std::swprintf(buf, std::size(buf), L"luid_0x%08lX_0x%08lX",
                  static_cast<unsigned long>(luid.HighPart),
                  static_cast<unsigned long>(luid.LowPart));
    return buf;
}

/**
 * Windows sampler: CPU from GetSystemTimes, RAM from GlobalMemoryStatusEx, GPU load and VRAM
 * from the PDH counters Task Manager uses, and the VRAM total from DXGI.
 */
class Sampler {
public:
    /**
     * Opens the PDH query, adds the GPU counters and primes them, then finds the
     * adapter whose VRAM is reported. Missing counters leave their fields empty.
     */
    Sampler() {
        if (PdhOpenQueryW(nullptr, 0, &query) == ERROR_SUCCESS) {
            PdhAddEnglishCounterW(query, L"\\GPU Engine(*)\\Utilization Percentage", 0,
                                  &gpu_counter);
            PdhAddEnglishCounterW(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0,
                                  &vram_counter);
            PdhCollectQueryData(query);
        }
        FindMainAdapter();
    }

    /**
     * Closes the PDH query, which also frees its counters.
     */
    ~Sampler() {
        if (query) {
            PdhCloseQuery(query);
        }
    }

    /**
     * Fills in every measurement available. CPU and GPU load are rates, so they stay
     * empty on the first call and are computed against the previous call after that.
     */
    void Sample(SystemMonitor::Snapshot& out) {
        FILETIME idle, kernel, user;
        if (GetSystemTimes(&idle, &kernel, &user)) {
            const u64 i = ToU64(idle), k = ToU64(kernel), u = ToU64(user);
            const u64 total = (k - last_kernel) + (u - last_user);
            if (last_kernel != 0 && total > 0) {
                const u64 idle_delta = i - last_idle;
                const u64 busy = total - std::min(idle_delta, total);
                out.cpu_percent = 100.0 * double(busy) / double(total);
            }
            last_idle = i;
            last_kernel = k;
            last_user = u;
        }

        MEMORYSTATUSEX mem{};
        mem.dwLength = sizeof(mem);
        if (GlobalMemoryStatusEx(&mem)) {
            out.ram = SystemMonitor::Usage{mem.ullTotalPhys - mem.ullAvailPhys, mem.ullTotalPhys};
        }

        if (!query || PdhCollectQueryData(query) != ERROR_SUCCESS) {
            return;
        }

        if (gpu_counter) {
            SampleGpuLoad(out);
        }

        if (vram_counter && main_adapter_vram > 0) {
            ForEachItem(vram_counter, PDH_FMT_LARGE, [&](const PDH_FMT_COUNTERVALUE_ITEM_W& item) {
                if (SameKey(AdapterKey(item.szName), main_adapter_key)) {
                    out.vram =
                        SystemMonitor::Usage{u64(item.FmtValue.largeValue), main_adapter_vram};
                }
            });
        }
    }

private:
    /**
     * Computes GPU load the way Task Manager does: sums every process's utilization per
     * engine (3D, compute, copy, video...), then reports the busiest engine on the main
     * adapter, or on any adapter if the main one wasn't found. Keys view the PDH buffer
     * and the engine list keeps its capacity, so nothing is allocated per sample.
     */
    void SampleGpuLoad(SystemMonitor::Snapshot& out) {
        engine_loads.clear();
        ForEachItem(gpu_counter, PDH_FMT_DOUBLE | PDH_FMT_NOCAP100,
                    [&](const PDH_FMT_COUNTERVALUE_ITEM_W& item) {
                        const auto key = EngineKey(item.szName);
                        const auto it = std::find_if(
                            engine_loads.begin(), engine_loads.end(),
                            [&](const auto& entry) { return entry.first == key; });
                        if (it != engine_loads.end()) {
                            it->second += item.FmtValue.doubleValue;
                        } else {
                            engine_loads.emplace_back(key, item.FmtValue.doubleValue);
                        }
                    });
        if (engine_loads.empty()) {
            return;
        }

        std::optional<double> busiest_main;
        double busiest_any = 0.0;
        for (const auto& [key, load] : engine_loads) {
            busiest_any = std::max(busiest_any, load);
            if (!main_adapter_key.empty() && SameKey(AdapterKey(key), main_adapter_key)) {
                busiest_main = std::max(busiest_main.value_or(0.0), load);
            }
        }
        out.gpu_percent = std::min(busiest_main.value_or(busiest_any), 100.0);
    }

    /**
     * Picks the hardware adapter with the most dedicated memory, which is the one
     * games render on, and records its key and VRAM total.
     */
    void FindMainAdapter() {
        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                      reinterpret_cast<void**>(&factory)))) {
            return;
        }
        IDXGIAdapter1* adapter = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            if (SUCCEEDED(adapter->GetDesc1(&desc)) && !(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) &&
                desc.DedicatedVideoMemory > main_adapter_vram) {
                main_adapter_vram = desc.DedicatedVideoMemory;
                main_adapter_key = AdapterKey(desc.AdapterLuid);
            }
            adapter->Release();
        }
        factory->Release();
    }

    /**
     * Reads every instance of a (wildcard) counter in the given format and calls
     * func for each one with valid data. Reuses one buffer across calls.
     */
    template <typename F>
    void ForEachItem(PDH_HCOUNTER counter, DWORD format, F&& func) {
        DWORD size = 0;
        DWORD count = 0;
        if (PdhGetFormattedCounterArrayW(counter, format, &size, &count, nullptr) !=
            static_cast<PDH_STATUS>(PDH_MORE_DATA)) {
            return;
        }
        buffer.resize(size);
        auto* items = reinterpret_cast<PDH_FMT_COUNTERVALUE_ITEM_W*>(buffer.data());
        if (PdhGetFormattedCounterArrayW(counter, format, &size, &count, items) != ERROR_SUCCESS) {
            return;
        }
        for (DWORD i = 0; i < count; ++i) {
            const auto status = items[i].FmtValue.CStatus;
            if (status == PDH_CSTATUS_VALID_DATA || status == PDH_CSTATUS_NEW_DATA) {
                func(items[i]);
            }
        }
    }

    PDH_HQUERY query = nullptr;
    PDH_HCOUNTER gpu_counter = nullptr;
    PDH_HCOUNTER vram_counter = nullptr;
    std::vector<std::byte> buffer;
    std::vector<std::pair<std::wstring_view, double>> engine_loads;

    std::wstring main_adapter_key;
    u64 main_adapter_vram = 0;

    u64 last_idle = 0;
    u64 last_kernel = 0;
    u64 last_user = 0;
};

#elif defined(__linux__)

/**
 * Reads a single value from a text file such as a sysfs attribute.
 * Returns nothing if the file is missing or doesn't parse.
 */
template <typename T>
std::optional<T> ReadValue(const std::filesystem::path& path) {
    std::ifstream file(path);
    T value{};
    if (file >> value) {
        return value;
    }
    return std::nullopt;
}

/**
 * Linux sampler: CPU from /proc/stat, RAM from /proc/meminfo, and GPU load and VRAM
 * from amdgpu's sysfs attributes. Other GPU drivers leave GPU and VRAM empty.
 */
class Sampler {
public:
    /**
     * Finds the amdgpu device with the most VRAM, if any. Only amdgpu exposes
     * load and VRAM through sysfs.
     */
    Sampler() {
        std::error_code ec;
        u64 best_total = 0;
        for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            const auto device = entry.path() / "device";
            const auto total = ReadValue<u64>(device / "mem_info_vram_total");
            if (total && *total > best_total &&
                std::filesystem::exists(device / "gpu_busy_percent")) {
                best_total = *total;
                gpu_device = device;
            }
        }
    }

    /**
     * Fills in every measurement available. CPU load is a rate, so it stays empty
     * on the first call and is computed against the previous call after that.
     */
    void Sample(SystemMonitor::Snapshot& out) {
        if (std::ifstream stat("/proc/stat"); stat) {
            std::string cpu;
            u64 user, nice, system, idle, iowait, irq, softirq, steal;
            if (stat >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >>
                steal) {
                const u64 idle_all = idle + iowait;
                const u64 total = user + nice + system + idle_all + irq + softirq + steal;
                if (last_total != 0 && total > last_total) {
                    const u64 delta = total - last_total;
                    const u64 idle_delta = std::min(idle_all - last_idle, delta);
                    out.cpu_percent = 100.0 * double(delta - idle_delta) / double(delta);
                }
                last_total = total;
                last_idle = idle_all;
            }
        }

        if (std::ifstream meminfo("/proc/meminfo"); meminfo) {
            std::string key;
            u64 value;
            std::string unit;
            u64 total = 0, available = 0;
            while (meminfo >> key >> value >> unit) {
                if (key == "MemTotal:") {
                    total = value * 1024;
                } else if (key == "MemAvailable:") {
                    available = value * 1024;
                }
            }
            if (total) {
                out.ram = SystemMonitor::Usage{total - std::min(available, total), total};
            }
        }

        if (!gpu_device.empty()) {
            if (const auto busy = ReadValue<double>(gpu_device / "gpu_busy_percent")) {
                out.gpu_percent = *busy;
            }
            const auto used = ReadValue<u64>(gpu_device / "mem_info_vram_used");
            const auto total = ReadValue<u64>(gpu_device / "mem_info_vram_total");
            if (used && total) {
                out.vram = SystemMonitor::Usage{*used, *total};
            }
        }
    }

private:
    std::filesystem::path gpu_device;
    u64 last_total = 0;
    u64 last_idle = 0;
};

#else

/**
 * Fallback for platforms without a sampler: every measurement stays empty.
 */
class Sampler {
public:
    /**
     * Leaves the snapshot empty.
     */
    void Sample(SystemMonitor::Snapshot&) {}
};

#endif

} // namespace

SystemMonitor::SystemMonitor() = default;

SystemMonitor::~SystemMonitor() = default;

/**
 * Starts the sampling thread. Does nothing if it is already running.
 */
void SystemMonitor::Start() {
    if (!thread.joinable()) {
        thread = std::jthread([this](std::stop_token token) { Run(token); });
    }
}

/**
 * Stops and joins the sampling thread, then clears the last snapshot so stale
 * values aren't shown the next time sampling starts.
 */
void SystemMonitor::Stop() {
    thread = {};
    std::scoped_lock lock{mutex};
    snapshot = {};
}

/**
 * Returns a copy of the most recent snapshot.
 */
SystemMonitor::Snapshot SystemMonitor::Get() const {
    std::scoped_lock lock{mutex};
    return snapshot;
}

/**
 * Sampling thread body. Runs at low priority so it never competes with emulation threads,
 * creates the sampler here so its setup stays off the UI thread, then publishes a new
 * snapshot every SAMPLE_INTERVAL until stopped.
 */
void SystemMonitor::Run(std::stop_token token) {
    Common::SetCurrentThreadName("SystemMonitor");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);

    Sampler sampler;
    do {
        Snapshot next;
        sampler.Sample(next);
        std::scoped_lock lock{mutex};
        snapshot = next;
    } while (Common::StoppableTimedWait(token, SAMPLE_INTERVAL));
}
