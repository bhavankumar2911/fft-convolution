#pragma once
#include <chrono>
#include <algorithm>

#include "model/FeatureExtractor.hpp"
#include "layers/Linear.hpp"

template<typename T>
class STL10CNNModel {
    FeatureExtractor<T> features;
    Linear<T> fc1;
    Linear<T> fc2;

    double fcMs = 0.0;

public:
    STL10CNNModel(
        FeatureExtractor<T>&& f,
        Linear<T>&& l1,
        Linear<T>&& l2
    )
        : features(std::move(f)),
          fc1(std::move(l1)),
          fc2(std::move(l2)) {}

    std::vector<T> forward(const Tensor<T>& input) {
        auto x = features.forward(input);

        fcMs = 0.0;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto y = fc1.forward(x.data);
        auto t1 = std::chrono::high_resolution_clock::now();
        fcMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

        for (T& v : y)
            v = std::max(T(0), v);

        t0 = std::chrono::high_resolution_clock::now();
        auto out = fc2.forward(y);
        t1 = std::chrono::high_resolution_clock::now();
        fcMs += std::chrono::duration<double, std::milli>(t1 - t0).count();

        return out;
    }

    const FeatureExtractor<T>& featureExtractor() const {
        return features;
    }

    double fcTimeMs() const { return fcMs; }
};
