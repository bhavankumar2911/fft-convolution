#include <torch/script.h>
#include "cnpy/cnpy.h"

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include "Matrix2D.hpp"
#include "PaddingMode.hpp"
#include "FFTCrossCorrelation2D_CUDA.hpp"

using Scalar = double;

/* ======================= ReLU ======================= */

void applyReLU(std::vector<Matrix2D<Scalar>>& maps)
{
    for (auto& m : maps)
        for (std::size_t y = 0; y < m.rows(); ++y)
            for (std::size_t x = 0; x < m.cols(); ++x)
                if (m(y, x) < 0.0)
                    m(y, x) = 0.0;
}

/* ===================== MaxPool2D ===================== */

std::vector<Matrix2D<Scalar>> maxPool2D(
    const std::vector<Matrix2D<Scalar>>& input
)
{
    std::vector<Matrix2D<Scalar>> output;
    output.reserve(input.size());

    for (const auto& in : input)
    {
        std::size_t outH = in.rows() / 2;
        std::size_t outW = in.cols() / 2;

        Matrix2D<Scalar> pooled(outH, outW);

        for (std::size_t y = 0; y < outH; ++y)
            for (std::size_t x = 0; x < outW; ++x)
            {
                Scalar m = in(2*y,     2*x);
                m = std::max(m, in(2*y + 1, 2*x));
                m = std::max(m, in(2*y,     2*x + 1));
                m = std::max(m, in(2*y + 1, 2*x + 1));
                pooled(y, x) = m;
            }

        output.push_back(std::move(pooled));
    }

    return output;
}

/* ======================= Flatten ===================== */

std::vector<Scalar> flatten(
    const std::vector<Matrix2D<Scalar>>& maps
)
{
    std::vector<Scalar> v;

    for (const auto& m : maps)
        for (std::size_t y = 0; y < m.rows(); ++y)
            for (std::size_t x = 0; x < m.cols(); ++x)
                v.push_back(m(y, x));

    return v;
}

/* ======================= Linear ====================== */

class LinearLayer
{
public:
    LinearLayer(
        std::vector<std::vector<Scalar>> w,
        std::vector<Scalar> b
    ) : weights(std::move(w)), bias(std::move(b)) {}

    std::vector<Scalar> forward(
        const std::vector<Scalar>& input
    ) const
    {
        std::vector<Scalar> out(weights.size());

        for (std::size_t o = 0; o < weights.size(); ++o)
        {
            Scalar sum = bias[o];
            for (std::size_t i = 0; i < input.size(); ++i)
                sum += weights[o][i] * input[i];
            out[o] = sum;
        }
        return out;
    }

private:
    std::vector<std::vector<Scalar>> weights;
    std::vector<Scalar> bias;
};

/* ======================== Main ======================== */

int main()
{
    torch::jit::Module model =
        torch::jit::load("stl10_cnn_same_stride1_cpu.pt");
    model.eval();

    /* ---------- Load convolution layers ---------- */

    std::vector<
        std::vector<std::vector<Matrix2D<Scalar>>>
    > convKernels;

    std::vector<std::vector<Scalar>> convBiases;

    for (int idx : {0, 4, 8, 12})
    {
        auto conv =
            model.attr("features").toModule()
                 .attr(std::to_string(idx)).toModule();

        auto w = conv.attr("weight").toTensor();
        auto b = conv.attr("bias").toTensor();

        int OC = w.size(0);
        int IC = w.size(1);
        int KH = w.size(2);
        int KW = w.size(3);

        auto wa = w.accessor<Scalar,4>();
        auto ba = b.accessor<Scalar,1>();

        convKernels.emplace_back(OC);
        convBiases.emplace_back(OC);

        for (int oc = 0; oc < OC; ++oc)
        {
            convKernels.back()[oc].resize(IC);
            convBiases.back()[oc] = ba[oc];

            for (int ic = 0; ic < IC; ++ic)
            {
                Matrix2D<Scalar> k(KH, KW);
                for (int y = 0; y < KH; ++y)
                    for (int x = 0; x < KW; ++x)
                        k(y, x) = wa[oc][ic][y][x];

                convKernels.back()[oc][ic] = std::move(k);
            }
        }
    }

    /* ---------- Load FC layers ---------- */

    auto loadLinear = [&](int index)
    {
        auto layer =
            model.attr("classifier").toModule()
                 .attr(std::to_string(index)).toModule();

        auto w = layer.attr("weight").toTensor();
        auto b = layer.attr("bias").toTensor();

        auto wa = w.accessor<Scalar,2>();
        auto ba = b.accessor<Scalar,1>();

        std::vector<std::vector<Scalar>> W(w.size(0));
        std::vector<Scalar> B(w.size(0));

        for (int o = 0; o < w.size(0); ++o)
        {
            B[o] = ba[o];
            W[o].resize(w.size(1));
            for (int i = 0; i < w.size(1); ++i)
                W[o][i] = wa[o][i];
        }

        return LinearLayer(W, B);
    };

    LinearLayer fc1 = loadLinear(0);
    LinearLayer fc2 = loadLinear(2);

    /* ---------- Inference ---------- */

    for (const auto& entry :
         std::filesystem::directory_iterator("./test_images"))
    {
        if (entry.path().extension() != ".npy") continue;

        cnpy::NpyArray arr = cnpy::npy_load(entry.path().string());
        float* data = arr.data<float>();

        std::vector<Matrix2D<Scalar>> maps;
        maps.reserve(3);

        for (int c = 0; c < 3; ++c)
        {
            Matrix2D<Scalar> m(96, 96);
            for (int y = 0; y < 96; ++y)
                for (int x = 0; x < 96; ++x)
                    m(y, x) =
                        static_cast<Scalar>(
                            data[c*96*96 + y*96 + x]
                        );
            maps.push_back(std::move(m));
        }

        for (std::size_t l = 0; l < convKernels.size(); ++l)
        {
            std::vector<Matrix2D<Scalar>> next;

            for (std::size_t oc = 0; oc < convKernels[l].size(); ++oc)
            {
                Matrix2D<Scalar> sum =
                    FFTCrossCorrelation2D_CUDA::compute(
                        maps[0],
                        convKernels[l][oc][0],
                        PaddingMode::SAME
                    );

                for (std::size_t ic = 1; ic < maps.size(); ++ic)
                {
                    Matrix2D<Scalar> tmp =
                        FFTCrossCorrelation2D_CUDA::compute(
                            maps[ic],
                            convKernels[l][oc][ic],
                            PaddingMode::SAME
                        );

                    for (std::size_t y = 0; y < sum.rows(); ++y)
                        for (std::size_t x = 0; x < sum.cols(); ++x)
                            sum(y, x) += tmp(y, x);
                }

                for (std::size_t y = 0; y < sum.rows(); ++y)
                    for (std::size_t x = 0; x < sum.cols(); ++x)
                        sum(y, x) += convBiases[l][oc];

                next.push_back(std::move(sum));
            }

            maps = std::move(next);
            applyReLU(maps);
            maps = maxPool2D(maps);
        }

        std::vector<Scalar> v = flatten(maps);
        v = fc1.forward(v);
        for (auto& x : v) if (x < 0.0) x = 0.0;
        v = fc2.forward(v);

        int pred =
            std::max_element(v.begin(), v.end()) - v.begin();

        std::cout << entry.path().filename()
                  << " -> " << pred << "\n";
    }

    return 0;
}
