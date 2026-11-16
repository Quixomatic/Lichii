// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <mutex>
#include <optional>
#include <thread>

#include "common/common_types.h"

/**
 * Samples host CPU, GPU and memory usage on a low-priority background thread.
 * Sampling only runs between Start() and Stop(); Get() is safe to call from any thread.
 */
class SystemMonitor {
public:
    /**
     * Used and total bytes of a memory pool (system RAM or dedicated VRAM).
     */
    struct Usage {
        u64 used = 0;
        u64 total = 0;

        /**
         * Returns how full the pool is, from 0.0 to 1.0.
         * Returns 0.0 when the total is unknown.
         */
        double Fraction() const {
            return total ? double(used) / double(total) : 0.0;
        }
    };

    /**
     * One sample of host usage. A field is empty when the platform or GPU
     * doesn't expose that measurement.
     */
    struct Snapshot {
        std::optional<double> cpu_percent;
        std::optional<double> gpu_percent;
        std::optional<Usage> ram;
        std::optional<Usage> vram;
    };

    SystemMonitor();
    ~SystemMonitor();

    /**
     * Starts the sampling thread. Does nothing if it is already running.
     */
    void Start();

    /**
     * Stops and joins the sampling thread, then clears the last snapshot.
     */
    void Stop();

    /**
     * Returns the most recent snapshot. Fields are empty until the first
     * sample completes, and CPU/GPU load needs two samples to compute.
     */
    Snapshot Get() const;

private:
    /**
     * Sampling thread body: sets up the platform sampler, then publishes a
     * new snapshot every interval until the stop token is triggered.
     */
    void Run(std::stop_token token);

    mutable std::mutex mutex;
    Snapshot snapshot;
    std::jthread thread;
};
