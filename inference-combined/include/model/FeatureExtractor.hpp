#pragma once
#include <memory>
#include <vector>
#include <chrono>

#include "core/IConvolution2D.hpp"
#include "layers/ActivationReLU.hpp"
#include "layers/MaxPool2D.hpp"

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
            ActivationReLU<T>::apply(x);
            auto t1 = std::chrono::high_resolution_clock::now();
            reluMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

            t0 = std::chrono::high_resolution_clock::now();
            x = MaxPool2D<T>::apply(x);
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
