#pragma once
#include "../Matrix2D.hpp"
#include "../cnpy/cnpy.h"
#include <vector>
#include <string>

class NpyImageLoader
{
public:
    static std::vector<Matrix2D<double>> load(
        const std::string& path
    )
    {
        auto arr = cnpy::npy_load(path);
        const float* d = arr.data<float>();

        std::size_t C = arr.shape[0];
        std::size_t H = arr.shape[1];
        std::size_t W = arr.shape[2];

        std::vector<Matrix2D<double>> out;
        std::size_t off = 0;

        for (std::size_t c = 0; c < C; ++c)
        {
            Matrix2D<double> m(H, W);
            for (std::size_t y = 0; y < H; ++y)
                for (std::size_t x = 0; x < W; ++x)
                    m(y, x) = d[off++];
            out.push_back(std::move(m));
        }
        return out;
    }
};
