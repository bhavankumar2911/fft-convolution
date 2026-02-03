#pragma once
#include "core/IConvolution2D.hpp"
#include "data/BinaryTensorLoader.hpp"
#include "utils/Timer.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cmath>
#include <type_traits>

/* ============================================================
   CUDA kernel: elementwise multiply with conjugate
   ============================================================ */

template<typename ComplexT>
__global__ void multiplyWithConjugate(
    ComplexT* imageFreq,
    const ComplexT* kernelFreq,
    std::size_t totalSize
) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize) {
        ComplexT a = imageFreq[idx];
        ComplexT b = kernelFreq[idx];

        imageFreq[idx].x = a.x * b.x + a.y * b.y;
        imageFreq[idx].y = a.y * b.x - a.x * b.y;
    }
}

/* ============================================================
   FFT resource cache
   ============================================================ */

template<typename ComplexT>
struct FFTResources {
    cufftHandle plan{};
    ComplexT* d_image{nullptr};
    ComplexT* d_kernel{nullptr};
    std::size_t totalSize{0};
};

static std::mutex fftCacheMutex;
static std::unordered_map<std::uint64_t, FFTResources<cufftComplex>> fftCacheF32;
static std::unordered_map<std::uint64_t, FFTResources<cufftDoubleComplex>> fftCacheF64;

inline std::uint64_t makeFFTKey(std::size_t h, std::size_t w) {
    return (static_cast<std::uint64_t>(h) << 32) | w;
}

/* ============================================================
   FFT Convolution Layer (CUDA)
   ============================================================ */

template<typename T>
class FFTConvolution2D_CUDA : public IConvolution2D<T> {
    int inC, outC, k, pad;
    std::vector<T> weights;
    std::vector<T> bias;
    double lastMs{0.0};

public:
    FFTConvolution2D_CUDA(
        int inCh, int outCh, int kernelSize, int padding,
        const std::string& wPath,
        const std::string& bPath
    )
        : inC(inCh), outC(outCh), k(kernelSize), pad(padding),
          weights(BinaryTensorLoader::loadVector<T>(wPath)),
          bias(BinaryTensorLoader::loadVector<T>(bPath)) {}

    Tensor<T> forward(const Tensor<T>& input) override
    {
        CpuTimer timer;
        timer.start();

        const int H = input.height;
        const int W = input.width;

        Tensor<T> output(outC, H, W);

        std::size_t fftH = 1, fftW = 1;
        while (fftH < H + k - 1) fftH <<= 1;
        while (fftW < W + k - 1) fftW <<= 1;

        std::size_t totalSize = fftH * fftW;
        std::uint64_t key = makeFFTKey(fftH, fftW);

        using ComplexT = std::conditional_t<
            std::is_same_v<T, double>,
            cufftDoubleComplex,
            cufftComplex
        >;

        FFTResources<ComplexT>* res = nullptr;

        /* ---------- Cache lookup / creation ---------- */
        {
            std::lock_guard<std::mutex> lock(fftCacheMutex);

            if constexpr (std::is_same_v<T, double>) {
                auto it = fftCacheF64.find(key);
                if (it == fftCacheF64.end()) {
                    FFTResources<cufftDoubleComplex> r;
                    r.totalSize = totalSize;
                    cudaMalloc(&r.d_image, totalSize * sizeof(cufftDoubleComplex));
                    cudaMalloc(&r.d_kernel, totalSize * sizeof(cufftDoubleComplex));
                    cufftPlan2d(&r.plan, fftH, fftW, CUFFT_Z2Z);
                    it = fftCacheF64.emplace(key, r).first;
                }
                res = reinterpret_cast<FFTResources<ComplexT>*>(&it->second);
            } else {
                auto it = fftCacheF32.find(key);
                if (it == fftCacheF32.end()) {
                    FFTResources<cufftComplex> r;
                    r.totalSize = totalSize;
                    cudaMalloc(&r.d_image, totalSize * sizeof(cufftComplex));
                    cudaMalloc(&r.d_kernel, totalSize * sizeof(cufftComplex));
                    cufftPlan2d(&r.plan, fftH, fftW, CUFFT_C2C);
                    it = fftCacheF32.emplace(key, r).first;
                }
                res = reinterpret_cast<FFTResources<ComplexT>*>(&it->second);
            }
        }

        std::vector<ComplexT> h_image(totalSize);
        std::vector<ComplexT> h_kernel(totalSize);

        for (int oc = 0; oc < outC; ++oc)
        {
            std::fill(output.data.begin() + oc * H * W,
                      output.data.begin() + (oc + 1) * H * W, T(0));

            for (int ic = 0; ic < inC; ++ic)
            {
                std::fill(h_image.begin(), h_image.end(), ComplexT{0, 0});
                std::fill(h_kernel.begin(), h_kernel.end(), ComplexT{0, 0});

                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        h_image[y * fftW + x].x =
                            input.data[ic * H * W + y * W + x];

                int kc = k / 2;
                for (int ky = 0; ky < k; ++ky)
                    for (int kx = 0; kx < k; ++kx) {
                        int sy = (ky + fftH - kc) % fftH;
                        int sx = (kx + fftW - kc) % fftW;
                        h_kernel[sy * fftW + sx].x =
                            weights[oc * inC * k * k + ic * k * k + ky * k + kx];
                    }

                cudaMemcpy(res->d_image, h_image.data(),
                           totalSize * sizeof(ComplexT), cudaMemcpyHostToDevice);
                cudaMemcpy(res->d_kernel, h_kernel.data(),
                           totalSize * sizeof(ComplexT), cudaMemcpyHostToDevice);

                if constexpr (std::is_same_v<T, double>) {
                    cufftExecZ2Z(res->plan, res->d_image, res->d_image, CUFFT_FORWARD);
                    cufftExecZ2Z(res->plan, res->d_kernel, res->d_kernel, CUFFT_FORWARD);
                } else {
                    cufftExecC2C(res->plan, res->d_image, res->d_image, CUFFT_FORWARD);
                    cufftExecC2C(res->plan, res->d_kernel, res->d_kernel, CUFFT_FORWARD);
                }

                int threads = 256;
                int blocks = (totalSize + threads - 1) / threads;
                multiplyWithConjugate<<<blocks, threads>>>(
                    res->d_image, res->d_kernel, totalSize
                );

                if constexpr (std::is_same_v<T, double>)
                    cufftExecZ2Z(res->plan, res->d_image, res->d_image, CUFFT_INVERSE);
                else
                    cufftExecC2C(res->plan, res->d_image, res->d_image, CUFFT_INVERSE);

                cudaMemcpy(h_image.data(), res->d_image,
                           totalSize * sizeof(ComplexT), cudaMemcpyDeviceToHost);

                T scale = static_cast<T>(fftH * fftW);
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        output.data[oc * H * W + y * W + x] +=
                            h_image[y * fftW + x].x / scale;
            }

            for (int i = 0; i < H * W; ++i)
                output.data[oc * H * W + i] += bias[oc];
        }

        lastMs = timer.stop();
        return output;
    }

    double lastExecutionTimeMs() const override { return lastMs; }
};
