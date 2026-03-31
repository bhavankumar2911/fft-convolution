#pragma once
#include "core/IConvolution2D.hpp"
#include "cpu/FFTConvolution2D_CPU.hpp"
#include "cpu/NaiveCPUConvolution2D.hpp"

#include <memory>
#include <string>

/* ============================================================
   HybridConvolution2D_CPU

   CPU-only counterpart of HybridConvolution2D.
   Wraps FFTConvolution2D_CPU and NaiveCPUConvolution2D.

   Same spatial-to-kernel ratio decision rule:
     ratio = input.height / kernel_size
     if (ratio > ratioThreshold) → CPU FFT
     else                        → CPU Naive

   Empirical basis from STL-10 CPU benchmarks:
     Conv1: 96/5  = 19.2 → FFT wins on CPU
     Conv2: 48/5  =  9.6 → FFT wins on CPU
     Conv3: 24/3  =  8.0 → Naive wins on CPU
     Conv4: 12/3  =  4.0 → Naive wins on CPU

   Default threshold: 9.0 (same as GPU hybrid)
   ============================================================ */

template<typename T>
class HybridConvolution2D_CPU : public IConvolution2D<T> {

    std::unique_ptr<FFTConvolution2D_CPU<T>>  fftConv;
    std::unique_ptr<NaiveCPUConvolution2D<T>> naiveConv;

    int   k;
    float ratioThreshold;
    double lastMs{0.0};
    mutable bool lastUsedFFT{false};

public:
    /* --------------------------------------------------------
       Both backends constructed upfront — no first-call penalty
       ratioThreshold defaults to 9.0 (empirically derived)
    -------------------------------------------------------- */
    HybridConvolution2D_CPU(
        int inCh, int outCh, int ks, int p,
        const std::string& wPath,
        const std::string& bPath,
        float ratioThreshold = 9.0f
    )
        : fftConv       (std::make_unique<FFTConvolution2D_CPU<T>> (inCh, outCh, ks, p, wPath, bPath)),
          naiveConv     (std::make_unique<NaiveCPUConvolution2D<T>>(inCh, outCh, ks, p, wPath, bPath)),
          k             (ks),
          ratioThreshold(ratioThreshold)
    {}

    Tensor<T> forward(const Tensor<T>& input) override {
        float ratio = static_cast<float>(input.height) / static_cast<float>(k);

        if (ratio > ratioThreshold) {
            lastUsedFFT = true;
            auto result = fftConv->forward(input);
            lastMs = fftConv->lastExecutionTimeMs();
            return result;
        } else {
            lastUsedFFT = false;
            auto result = naiveConv->forward(input);
            lastMs = naiveConv->lastExecutionTimeMs();
            return result;
        }
    }

    double lastExecutionTimeMs() const override { return lastMs; }
    bool   usedFFT()             const          { return lastUsedFFT; }
};