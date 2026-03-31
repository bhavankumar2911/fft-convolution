#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "core/IConvolution2D.hpp"

// -------------------------------------------------
// Backend-selected ReLU and Pool implementations
// -------------------------------------------------
#if defined(BACKEND_GPU_NAIVE) || defined(BACKEND_GPU_FFT) || defined(BACKEND_GPU_HYBRID)
    #include "cuda/ActivationReLU_CUDA.hpp"
    #include "cuda/MaxPool2D_CUDA.hpp"
    template<typename T> using ReLUImpl = ActivationReLU_CUDA<T>;
    template<typename T> using PoolImpl = MaxPool2D_CUDA<T>;
#else
    // CPU backends: CPU_NAIVE, CPU_FFT, CPU_HYBRID all use CPU ReLU/Pool
    #include "layers/ActivationReLU.hpp"
    #include "layers/MaxPool2D.hpp"
    template<typename T> using ReLUImpl = ActivationReLU<T>;
    template<typename T> using PoolImpl = MaxPool2D<T>;
#endif

template<typename T>
class FeatureExtractor {
    std::vector<std::unique_ptr<IConvolution2D<T>>> layers;

    double reluMs = 0.0;
    double poolMs = 0.0;

public:
    FeatureExtractor(std::vector<std::unique_ptr<IConvolution2D<T>>>&& l)
        : layers(std::move(l)) {}

    Tensor<T> forward(const Tensor<T>& input) {
        Tensor<T> x = input;
        reluMs = 0.0;
        poolMs = 0.0;

        for (auto& conv : layers) {
            x = conv->forward(x);

            auto t0 = std::chrono::high_resolution_clock::now();
            ReLUImpl<T>::apply(x);
            auto t1 = std::chrono::high_resolution_clock::now();
            reluMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            x = PoolImpl<T>::apply(x);
            t1 = std::chrono::high_resolution_clock::now();
            poolMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        return x;
    }

    const std::vector<std::unique_ptr<IConvolution2D<T>>>& convs() const {
        return layers;
    }

    double reluTimeMs() const { return reluMs; }
    double poolTimeMs() const { return poolMs; }
};