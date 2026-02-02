#pragma once
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "core/Tensor.hpp"

class BinaryTensorLoader {
public:
    template<typename T>
    static std::vector<T> loadVector(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open file: " + path);

        file.seekg(0, std::ios::end);
        size_t bytes = file.tellg();
        file.seekg(0, std::ios::beg);

        // Images may be float32 or float64
        if (bytes % sizeof(float) == 0) {
            size_t count = bytes / sizeof(float);
            std::vector<float> temp(count);
            file.read(reinterpret_cast<char*>(temp.data()), bytes);

            std::vector<T> out(count);
            for (size_t i = 0; i < count; ++i)
                out[i] = static_cast<T>(temp[i]);
            return out;
        }

        if (bytes % sizeof(double) == 0) {
            size_t count = bytes / sizeof(double);
            std::vector<double> temp(count);
            file.read(reinterpret_cast<char*>(temp.data()), bytes);

            std::vector<T> out(count);
            for (size_t i = 0; i < count; ++i)
                out[i] = static_cast<T>(temp[i]);
            return out;
        }

        throw std::runtime_error("Unsupported binary format: " + path);
    }

    template<typename T>
    static Tensor<T> loadImageCHW(
        const std::string& path,
        int c, int h, int w
    ) {
        auto raw = loadVector<T>(path);

        if (raw.size() != static_cast<size_t>(c * h * w)) {
            throw std::runtime_error(
                "Image size mismatch in " + path
            );
        }

        Tensor<T> t(c, h, w);

        // EXACT SAME normalization as PyTorch
        const T mean[3] = {T(0.4467), T(0.4398), T(0.4066)};
        const T stdv[3] = {T(0.2241), T(0.2215), T(0.2239)};

        for (int ch = 0; ch < c; ++ch)
            for (int i = 0; i < h * w; ++i)
                t.data[ch * h * w + i] =
                    (raw[ch * h * w + i] - mean[ch]) / stdv[ch];

        return t;
    }
};
