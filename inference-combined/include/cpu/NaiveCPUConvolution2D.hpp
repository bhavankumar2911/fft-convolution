#pragma once
#include "core/IConvolution2D.hpp"
#include "data/BinaryTensorLoader.hpp"
#include "utils/Timer.hpp"

template<typename T>
class NaiveCPUConvolution2D : public IConvolution2D<T> {
    int inC, outC, k, pad;
    std::vector<T> w, b;
    double lastMs;

public:
    NaiveCPUConvolution2D(
        int inCh, int outCh, int ks, int p,
        const std::string& wPath,
        const std::string& bPath
    )
        : inC(inCh), outC(outCh), k(ks), pad(p),
          w(BinaryTensorLoader::loadVector<T>(wPath)),
          b(BinaryTensorLoader::loadVector<T>(bPath)),
          lastMs(0.0) {}

    Tensor<T> forward(const Tensor<T>& in) override {
        CpuTimer timer;
        timer.start();

        Tensor<T> out(outC, in.height, in.width);

        for (int oc = 0; oc < outC; ++oc)
            for (int ic = 0; ic < inC; ++ic)
                for (int y = 0; y < in.height; ++y)
                    for (int x = 0; x < in.width; ++x)
                        for (int ky = 0; ky < k; ++ky)
                            for (int kx = 0; kx < k; ++kx) {

                                int iy = y + ky - pad;
                                int ix = x + kx - pad;
                                if (iy >= 0 && ix >= 0 &&
                                    iy < in.height && ix < in.width) {

                                    out.data[
                                        oc * in.height * in.width +
                                        y * in.width + x
                                    ] +=
                                        in.data[
                                            ic * in.height * in.width +
                                            iy * in.width + ix
                                        ] *
                                        w[
                                            oc * inC * k * k +
                                            ic * k * k +
                                            ky * k + kx
                                        ];
                                }
                            }

        for (int oc = 0; oc < outC; ++oc)
            for (int i = 0; i < in.height * in.width; ++i)
                out.data[oc * in.height * in.width + i] += b[oc];

        lastMs = timer.stop();
        return out;
    }

    double lastExecutionTimeMs() const override { return lastMs; }
};
