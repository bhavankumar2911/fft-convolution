#pragma once
#include "Conv2D_FFT.hpp"
#include "LinearLayer.hpp"
#include "MaxPool2D.hpp"
#include "Normalization.hpp"
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
        /* -------- INPUT NORMALIZATION -------- */
        Normalization::apply(x);

        /* -------- CONV BLOCK 1 -------- */
        x = c1.forward(x);
        for (auto& m : x)
        {
            for (std::size_t y = 0; y < m.rows(); ++y)
                for (std::size_t z = 0; z < m.cols(); ++z)
                    m(y, z) = std::max(0.0, m(y, z));
            m = MaxPool2D::apply(m);
        }

        /* -------- CONV BLOCK 2 -------- */
        x = c2.forward(x);
        for (auto& m : x)
        {
            for (std::size_t y = 0; y < m.rows(); ++y)
                for (std::size_t z = 0; z < m.cols(); ++z)
                    m(y, z) = std::max(0.0, m(y, z));
            m = MaxPool2D::apply(m);
        }

        /* -------- CONV BLOCK 3 -------- */
        x = c3.forward(x);
        for (auto& m : x)
        {
            for (std::size_t y = 0; y < m.rows(); ++y)
                for (std::size_t z = 0; z < m.cols(); ++z)
                    m(y, z) = std::max(0.0, m(y, z));
            m = MaxPool2D::apply(m);
        }

        /* -------- CONV BLOCK 4 -------- */
        x = c4.forward(x);
        for (auto& m : x)
        {
            for (std::size_t y = 0; y < m.rows(); ++y)
                for (std::size_t z = 0; z < m.cols(); ++z)
                    m(y, z) = std::max(0.0, m(y, z));
            m = MaxPool2D::apply(m);
        }

        /* -------- FLATTEN -------- */
        std::vector<double> flat;
        flat.reserve(256 * 6 * 6);

        for (const auto& c : x)
            for (std::size_t y = 0; y < c.rows(); ++y)
                for (std::size_t z = 0; z < c.cols(); ++z)
                    flat.push_back(c(y, z));

        /* -------- FC -------- */
        auto h = f1.forward(flat);
        for (auto& v : h)
            v = std::max(0.0, v);

        return f2.forward(h);
    }
};
