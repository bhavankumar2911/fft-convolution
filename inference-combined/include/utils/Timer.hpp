#pragma once
#include <chrono>

class CpuTimer {
    std::chrono::high_resolution_clock::time_point start_;
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    double stop() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
};
