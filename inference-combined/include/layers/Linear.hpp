#pragma once
#include <vector>
#include <string>
#include "data/BinaryTensorLoader.hpp"

template<typename T>
class Linear {
    int inF, outF;
    std::vector<T> w, b;

public:
    Linear(int inFeatures, int outFeatures,
           const std::string& wPath,
           const std::string& bPath)
        : inF(inFeatures),
          outF(outFeatures),
          w(BinaryTensorLoader::loadVector<T>(wPath)),
          b(BinaryTensorLoader::loadVector<T>(bPath)) {}

    std::vector<T> forward(const std::vector<T>& input) {
        std::vector<T> out(outF, T(0));
        for (int o = 0; o < outF; ++o) {
            for (int i = 0; i < inF; ++i)
                out[o] += input[i] * w[o * inF + i];
            out[o] += b[o];
        }
        return out;
    }
};
