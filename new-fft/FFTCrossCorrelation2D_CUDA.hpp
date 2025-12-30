#pragma once

#ifndef FFT_CROSS_CORRELATION_2D_CUDA_HPP
#define FFT_CROSS_CORRELATION_2D_CUDA_HPP

#include "Matrix2D.hpp"
#include "PaddingMode.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <vector>
#include <type_traits>
#include <stdexcept>

/* ================= CUDA kernel ================= */

__global__ void multiplyWithConjugate(
    cufftDoubleComplex* imageFreq,
    const cufftDoubleComplex* kernelFreq,
    std::size_t totalSize
)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize)
    {
        cufftDoubleComplex a = imageFreq[idx];
        cufftDoubleComplex b = kernelFreq[idx];

        imageFreq[idx].x = a.x * b.x + a.y * b.y;
        imageFreq[idx].y = a.y * b.x - a.x * b.y;
    }
}

/* ================= FFT Cross-Correlation (CUDA) ================= */

class FFTCrossCorrelation2D_CUDA
{
public:
    template<typename T>
    static Matrix2D<T> compute(
        const Matrix2D<T>& image,
        const Matrix2D<T>& kernel,
        PaddingMode paddingMode
    )
    {
        static_assert(std::is_same<T, double>::value,
                      "CUDA FFT version supports double only");

        std::size_t pad;
        if (paddingMode == PaddingMode::SAME)  pad = kernel.rows() / 2;
        if (paddingMode == PaddingMode::VALID) pad = 0;
        if (paddingMode == PaddingMode::FULL)  pad = kernel.rows() - 1;

        std::size_t paddedH = image.rows() + 2 * pad;
        std::size_t paddedW = image.cols() + 2 * pad;

        std::size_t fftH = 1, fftW = 1;
        while (fftH < paddedH + kernel.rows() - 1) fftH <<= 1;
        while (fftW < paddedW + kernel.cols() - 1) fftW <<= 1;

        std::size_t totalSize = fftH * fftW;
        std::size_t bytes = totalSize * sizeof(cufftDoubleComplex);

        std::vector<cufftDoubleComplex> h_image(totalSize, {0.0, 0.0});
        std::vector<cufftDoubleComplex> h_kernel(totalSize, {0.0, 0.0});

        for (std::size_t y = 0; y < image.rows(); ++y)
            for (std::size_t x = 0; x < image.cols(); ++x)
                h_image[(y + pad) * fftW + (x + pad)].x = image(y, x);

        std::size_t kCenterY = kernel.rows() / 2;
        std::size_t kCenterX = kernel.cols() / 2;

        for (std::size_t y = 0; y < kernel.rows(); ++y)
            for (std::size_t x = 0; x < kernel.cols(); ++x)
            {
                std::size_t sy = (y + fftH - kCenterY) % fftH;
                std::size_t sx = (x + fftW - kCenterX) % fftW;
                h_kernel[sy * fftW + sx].x = kernel(y, x);
            }

        cufftDoubleComplex *d_image, *d_kernel;
        cudaMalloc(&d_image, bytes);
        cudaMalloc(&d_kernel, bytes);

        cudaMemcpy(d_image, h_image.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, h_kernel.data(), bytes, cudaMemcpyHostToDevice);

        cufftHandle plan;
        cufftPlan2d(&plan, fftH, fftW, CUFFT_Z2Z);

        cufftExecZ2Z(plan, d_image, d_image, CUFFT_FORWARD);
        cufftExecZ2Z(plan, d_kernel, d_kernel, CUFFT_FORWARD);

        std::size_t threads = 256;
        std::size_t blocks = (totalSize + threads - 1) / threads;

        multiplyWithConjugate<<<blocks, threads>>>(
            d_image, d_kernel, totalSize
        );

        cufftExecZ2Z(plan, d_image, d_image, CUFFT_INVERSE);

        cudaMemcpy(h_image.data(), d_image, bytes, cudaMemcpyDeviceToHost);

        cufftDestroy(plan);
        cudaFree(d_image);
        cudaFree(d_kernel);

        std::size_t outH, outW, startY, startX;

        if (paddingMode == PaddingMode::SAME)
        {
            outH = image.rows();
            outW = image.cols();
            startY = pad;
            startX = pad;
        }
        else if (paddingMode == PaddingMode::VALID)
        {
            outH = image.rows() - kernel.rows() + 1;
            outW = image.cols() - kernel.cols() + 1;
            startY = kernel.rows() - 1;
            startX = kernel.cols() - 1;
        }
        else
        {
            outH = image.rows() + kernel.rows() - 1;
            outW = image.cols() + kernel.cols() - 1;
            startY = 0;
            startX = 0;
        }

        Matrix2D<T> output(outH, outW);
        double scale = static_cast<double>(fftH * fftW);

        for (std::size_t y = 0; y < outH; ++y)
            for (std::size_t x = 0; x < outW; ++x)
                output(y, x) =
                    static_cast<T>(
                        h_image[(y + startY) * fftW + (x + startX)].x / scale
                    );

        return output;
    }
};

#endif
