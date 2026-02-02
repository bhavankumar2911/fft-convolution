#pragma once
#include "../Matrix2D.hpp"
#include "../FFTCrossCorrelation2D_CUDA.hpp"
#include "../PaddingMode.hpp"
#include <vector>

class Conv2D_FFT
{
    std::size_t inC, outC, k;
    std::vector<Matrix2D<double>> kernels;
    std::vector<double> bias;

public:
    Conv2D_FFT(
        std::size_t iC, std::size_t oC, std::size_t k_,
        const std::vector<double>& w,
        const std::vector<double>& b
    ) : inC(iC), outC(oC), k(k_), bias(b)
    {
        std::size_t off = 0;
        for (std::size_t oc = 0; oc < outC; ++oc)
            for (std::size_t ic = 0; ic < inC; ++ic)
            {
                Matrix2D<double> m(k, k);
                for (std::size_t y = 0; y < k; ++y)
                    for (std::size_t x = 0; x < k; ++x)
                        m(y, x) = w[off++];
                kernels.push_back(std::move(m));
            }
    }

    std::vector<Matrix2D<double>> forward(
        const std::vector<Matrix2D<double>>& in
    ) const
    {
        std::vector<Matrix2D<double>> out(outC);

        for (std::size_t oc = 0; oc < outC; ++oc)
        {
            Matrix2D<double> sum(in[0].rows(), in[0].cols());
            for (std::size_t ic = 0; ic < inC; ++ic)
            {
                auto c =
                    FFTCrossCorrelation2D_CUDA::compute(
                        in[ic],
                        kernels[oc * inC + ic],
                        PaddingMode::SAME
                    );

                for (std::size_t y = 0; y < sum.rows(); ++y)
                    for (std::size_t x = 0; x < sum.cols(); ++x)
                        sum(y, x) += c(y, x);
            }

            for (std::size_t y = 0; y < sum.rows(); ++y)
                for (std::size_t x = 0; x < sum.cols(); ++x)
                    sum(y, x) += bias[oc];

            out[oc] = std::move(sum);
        }
        return out;
    }
};
