#pragma once
#include "./Conv2D_FFT.hpp"
#include "../inference/LinearLayer.hpp"
#include "../inference/MaxPool2D.hpp"
#include "../inference/Normalization.hpp"
#include <algorithm>

class STL10CNN_Inference
{
    Conv2D_FFT c1, c2, c3, c4;
    LinearLayer f1, f2;

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
        Normalization::apply(x);

        auto convBlock = [&](Conv2D_FFT& conv)
        {
            x = conv.forward(x);
            for (auto& m : x)
            {
                for (std::size_t i = 0; i < m.rows() * m.cols(); ++i)
                    m.data()[i] = std::max(0.0, m.data()[i]);
                m = MaxPool2D::apply(m);
            }
        };

        convBlock(c1);
        convBlock(c2);
        convBlock(c3);
        convBlock(c4);

        std::vector<double> flat;
        flat.reserve(256 * 6 * 6);

        for (const auto& m : x)
            for (std::size_t i = 0; i < m.rows() * m.cols(); ++i)
                flat.push_back(m.data()[i]);

        auto h = f1.forward(flat);
        for (auto& v : h)
            v = std::max(0.0, v);

        return f2.forward(h);
    }
};
