#pragma once
#include "core/IConvolution2D.hpp"
#include "cuda/FFTConvolution2D_CUDA.hpp"
#include "cuda/NaiveCUDAConvolution2D.hpp"

#include <memory>
#include <string>

/* ============================================================
   HybridConvolution2D

   Wraps one FFT conv and one Naive CUDA conv for the same layer.
   At forward time, computes the spatial-to-kernel ratio of the
   input and delegates to the appropriate backend.

   Decision rule:
     ratio = input.height / kernel_size
     if (ratio > ratioThreshold) → FFT
     else                        → Naive CUDA

   Empirical basis (your STL-10 benchmark results):
     Conv1: 96/5  = 19.2 → FFT   (~11ms  vs ~42ms)
     Conv2: 48/5  =  9.6 → FFT   (~101ms vs ~221ms)
     Conv3: 24/3  =  8.0 → Naive (~286ms vs ~91ms)
     Conv4: 12/3  =  4.0 → Naive (~1050ms vs ~90ms)

   Crossover lies between 8.0 and 9.6.
   Default threshold of 9.0 cleanly separates all four layers:
     19.2 > 9.0 → FFT   ✓
      9.6 > 9.0 → FFT   ✓
      8.0 > 9.0 → Naive ✓
      4.0 > 9.0 → Naive ✓

   This ratio-based condition generalizes beyond this architecture:
   if kernel size or input resolution changes, the threshold
   remains physically meaningful.
   ============================================================ */

template<typename T>
class HybridConvolution2D : public IConvolution2D<T> {

    std::unique_ptr<FFTConvolution2D_CUDA<T>>  fftConv;
    std::unique_ptr<NaiveCUDAConvolution2D<T>> naiveConv;

    int   k;               // kernel size — needed for ratio computation
    float ratioThreshold;
    double lastMs{0.0};
    mutable bool lastUsedFFT{false};

public:
    /* --------------------------------------------------------
       Both backends are constructed upfront to avoid any
       first-call initialization penalty during benchmarking.
       ratioThreshold defaults to 9.0 (empirically derived).
    -------------------------------------------------------- */
    HybridConvolution2D(
        int inCh, int outCh, int ks, int p,
        const std::string& wPath,
        const std::string& bPath,
        float ratioThreshold = 9.0f
    )
        : fftConv        (std::make_unique<FFTConvolution2D_CUDA<T>> (inCh, outCh, ks, p, wPath, bPath)),
          naiveConv      (std::make_unique<NaiveCUDAConvolution2D<T>>(inCh, outCh, ks, p, wPath, bPath)),
          k              (ks),
          ratioThreshold (ratioThreshold)
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

    // Reports which backend was selected for the last forward call
    bool usedFFT() const { return lastUsedFFT; }
};