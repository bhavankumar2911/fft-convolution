#pragma once
#include "Conv2D_FFT.hpp"
#include "LinearLayer.hpp"
#include <algorithm>

class STL10CNN_Inference
{
    Conv2D_FFT c1, c2, c3, c4;
    LinearLayer f1, f2;

    static void relu(Matrix2D<double>& m)
    {
        for (std::size_t y = 0; y < m.rows(); ++y)
            for (std::size_t x = 0; x < m.cols(); ++x)
                m(y, x) = std::max(0.0, m(y, x));
    }

public:
    STL10CNN_Inference(
        Conv2D_FFT&& a, Conv2D_FFT&& b,
        Conv2D_FFT&& c, Conv2D_FFT&& d,
        LinearLayer&& e, LinearLayer&& f
    ) : c1(a), c2(b), c3(c), c4(d), f1(e), f2(f) {}

    std::vector<double> forward(
        std::vector<Matrix2D<double>> x
    )
    {
        x = c1.forward(x); for (auto& m : x) relu(m);
        x = c2.forward(x); for (auto& m : x) relu(m);
        x = c3.forward(x); for (auto& m : x) relu(m);
        x = c4.forward(x); for (auto& m : x) relu(m);

        std::vector<double> flat;
        for (const auto& c : x)
            for (std::size_t y = 0; y < c.rows(); ++y)
                for (std::size_t x2 = 0; x2 < c.cols(); ++x2)
                    flat.push_back(c(y, x2));

        auto h = f1.forward(flat);
        for (auto& v : h) v = std::max(0.0, v);
        return f2.forward(h);
    }
};
