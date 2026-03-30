#pragma once
#include <nvml.h>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <numeric>
#include <stdexcept>

/* ============================================================
   EnergyMeter

   Measures GPU energy consumption during inference using NVML.
   Spawns a background thread that samples power every ~10ms.
   Energy is approximated by integrating power over time:

       E = sum(P_i * dt)

   Usage:
       EnergyMeter meter;
       meter.start();
       model.forward(input);
       meter.stop();
       double joules = meter.energyJ();
       double watts  = meter.avgPowerW();
   ============================================================ */

class EnergyMeter {
    nvmlDevice_t     device;
    std::thread      samplerThread;
    std::atomic<bool> running{false};

    std::vector<double> powerSamples_W;   // watts per sample
    std::vector<double> intervalSecs;     // time between samples (s)

    static constexpr int SAMPLE_INTERVAL_MS = 10;

public:
    EnergyMeter() {
        if (nvmlInit() != NVML_SUCCESS)
            throw std::runtime_error("NVML init failed");
        if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS)
            throw std::runtime_error("NVML device handle failed");
    }

    ~EnergyMeter() {
        if (running) stop();
        nvmlShutdown();
    }

    void start() {
        powerSamples_W.clear();
        intervalSecs.clear();
        running = true;

        samplerThread = std::thread([this]() {
            auto prev = std::chrono::high_resolution_clock::now();

            while (running) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(SAMPLE_INTERVAL_MS)
                );

                unsigned int powerMw = 0;
                if (nvmlDeviceGetPowerUsage(device, &powerMw) == NVML_SUCCESS) {
                    auto now = std::chrono::high_resolution_clock::now();
                    double dt = std::chrono::duration<double>(now - prev).count();
                    prev = now;

                    powerSamples_W.push_back(powerMw / 1000.0);  // mW → W
                    intervalSecs.push_back(dt);
                }
            }
        });
    }

    void stop() {
        running = false;
        if (samplerThread.joinable())
            samplerThread.join();
    }

    // Total energy in joules: sum(P_i * dt_i)
    double energyJ() const {
        double total = 0.0;
        for (std::size_t i = 0; i < powerSamples_W.size(); ++i)
            total += powerSamples_W[i] * intervalSecs[i];
        return total;
    }

    // Average power in watts
    double avgPowerW() const {
        if (powerSamples_W.empty()) return 0.0;
        double sum = 0.0;
        for (double p : powerSamples_W) sum += p;
        return sum / powerSamples_W.size();
    }

    // Number of samples collected
    int sampleCount() const {
        return static_cast<int>(powerSamples_W.size());
    }
};