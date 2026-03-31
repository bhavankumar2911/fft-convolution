#pragma once
#include "core/IConvolution2D.hpp"
#include "data/BinaryTensorLoader.hpp"
#include "utils/Timer.hpp"

#include <fftw3.h>

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cmath>

/* ============================================================
   FFTConvolution2D_CPU

   Identical algorithm to FFTConvolution2D_CUDA but runs
   entirely on CPU using FFTW3f (float precision).

   Steps per (oc, ic) pair — same as CUDA version:
     1. Zero-pad image to next power of 2 above H+K-1
     2. Place kernel at center-corrected position (circular shift)
     3. Forward FFT both
     4. Pointwise complex multiply (cross-correlation = conj multiply)
     5. Inverse FFT
     6. Scale by 1/(fftH*fftW)
     7. Accumulate into output

   Link with: -lfftw3f
   ============================================================ */

/* ============================================================
   FFTW plan cache — reuse plans across forward() calls
   Key: (fftH << 32) | fftW
   ============================================================ */
struct CPUFFTResources {
    fftwf_plan   planFwd{nullptr};
    fftwf_plan   planInv{nullptr};
    fftwf_complex* buf_image {nullptr};
    fftwf_complex* buf_kernel{nullptr};
    std::size_t  totalSize{0};
};

static std::mutex                                         cpuFftMutex;
static std::unordered_map<std::uint64_t, CPUFFTResources> cpuFftCache;

inline std::uint64_t makeCPUFFTKey(std::size_t h, std::size_t w) {
    return (static_cast<std::uint64_t>(h) << 32) | w;
}

/* ============================================================
   Pointwise complex multiply with conjugate (cross-correlation)
   a * conj(b):
     real = a.r*b.r + a.i*b.i
     imag = a.i*b.r - a.r*b.i
   ============================================================ */
inline void multiplyConjCPU(
    fftwf_complex*       a,
    const fftwf_complex* b,
    std::size_t          n
) {
    for (std::size_t i = 0; i < n; ++i) {
        float ar = a[i][0], ai = a[i][1];
        float br = b[i][0], bi = b[i][1];
        a[i][0] = ar * br + ai * bi;
        a[i][1] = ai * br - ar * bi;
    }
}

/* ============================================================
   FFTConvolution2D_CPU
   ============================================================ */
template<typename T>
class FFTConvolution2D_CPU : public IConvolution2D<T> {
    int inC, outC, k, pad;
    std::vector<T> weights;
    std::vector<T> bias;
    double lastMs{0.0};

public:
    FFTConvolution2D_CPU(
        int inCh, int outCh, int ks, int p,
        const std::string& wPath,
        const std::string& bPath
    )
        : inC(inCh), outC(outCh), k(ks), pad(p),
          weights(BinaryTensorLoader::loadVector<T>(wPath)),
          bias   (BinaryTensorLoader::loadVector<T>(bPath)) {}

    Tensor<T> forward(const Tensor<T>& input) override {
        CpuTimer timer;
        timer.start();

        const int H = input.height;
        const int W = input.width;

        // Next power of 2 above H+K-1
        std::size_t fftH = 1, fftW = 1;
        while (fftH < static_cast<std::size_t>(H + k - 1)) fftH <<= 1;
        while (fftW < static_cast<std::size_t>(W + k - 1)) fftW <<= 1;
        std::size_t totalSize = fftH * fftW;

        // Cache lookup / creation
        std::uint64_t key = makeCPUFFTKey(fftH, fftW);
        CPUFFTResources* res = nullptr;
        {
            std::lock_guard<std::mutex> lock(cpuFftMutex);
            auto it = cpuFftCache.find(key);
            if (it == cpuFftCache.end()) {
                CPUFFTResources r;
                r.totalSize  = totalSize;
                r.buf_image  = fftwf_alloc_complex(totalSize);
                r.buf_kernel = fftwf_alloc_complex(totalSize);
                // Plans are created once and reused
                r.planFwd = fftwf_plan_dft_2d(
                    fftH, fftW,
                    r.buf_image, r.buf_image,
                    FFTW_FORWARD, FFTW_ESTIMATE
                );
                r.planInv = fftwf_plan_dft_2d(
                    fftH, fftW,
                    r.buf_image, r.buf_image,
                    FFTW_BACKWARD, FFTW_ESTIMATE
                );
                it = cpuFftCache.emplace(key, r).first;
            }
            res = &it->second;
        }

        Tensor<T> output(outC, H, W);
        float scale = static_cast<float>(fftH * fftW);

        for (int oc = 0; oc < outC; ++oc) {
            std::fill(
                output.data.begin() + oc * H * W,
                output.data.begin() + (oc + 1) * H * W,
                T(0)
            );

            for (int ic = 0; ic < inC; ++ic) {

                // ----- Fill image buffer -----
                for (std::size_t i = 0; i < totalSize; ++i)
                    res->buf_image[i][0] = res->buf_image[i][1] = 0.0f;

                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        res->buf_image[y * fftW + x][0] =
                            static_cast<float>(
                                input.data[ic * H * W + y * W + x]
                            );

                // ----- Fill kernel buffer (circular shift) -----
                for (std::size_t i = 0; i < totalSize; ++i)
                    res->buf_kernel[i][0] = res->buf_kernel[i][1] = 0.0f;

                int kc = k / 2;
                for (int ky = 0; ky < k; ++ky)
                    for (int kx = 0; kx < k; ++kx) {
                        int sy = (ky + fftH - kc) % fftH;
                        int sx = (kx + fftW - kc) % fftW;
                        res->buf_kernel[sy * fftW + sx][0] =
                            static_cast<float>(
                                weights[oc * inC * k * k +
                                        ic * k * k +
                                        ky * k + kx]
                            );
                    }

                // ----- Forward FFT both -----
                fftwf_execute_dft(res->planFwd, res->buf_image,  res->buf_image);
                fftwf_execute_dft(res->planFwd, res->buf_kernel, res->buf_kernel);

                // ----- Pointwise multiply (cross-correlation) -----
                multiplyConjCPU(res->buf_image, res->buf_kernel, totalSize);

                // ----- Inverse FFT -----
                fftwf_execute_dft(res->planInv, res->buf_image, res->buf_image);

                // ----- Accumulate into output -----
                for (int y = 0; y < H; ++y)
                    for (int x = 0; x < W; ++x)
                        output.data[oc * H * W + y * W + x] +=
                            static_cast<T>(
                                res->buf_image[y * fftW + x][0] / scale
                            );
            }

            // Add bias
            for (int i = 0; i < H * W; ++i)
                output.data[oc * H * W + i] += bias[oc];
        }

        lastMs = timer.stop();
        return output;
    }

    double lastExecutionTimeMs() const override { return lastMs; }
};