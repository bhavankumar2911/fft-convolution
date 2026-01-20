#pragma once
#ifndef FFT_CROSS_CORRELATION_2D_CUDA_HPP
#define FFT_CROSS_CORRELATION_2D_CUDA_HPP

#include "Matrix2D.hpp"
#include "../PaddingMode.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <unordered_map>
#include <stdexcept>
#include <mutex>

/* ============================================================
   CUDA kernel
   ============================================================ */

static __global__ void multiplyWithConjugate(
    cufftDoubleComplex* imageFreq,
    const cufftDoubleComplex* kernelFreq,
    std::size_t totalSize)
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

/* ============================================================
   FFT resource cache
   ============================================================ */

struct FFTResources
{
    cufftHandle plan;
    cufftDoubleComplex* d_image;
    cufftDoubleComplex* d_kernel;
    cudaStream_t stream;
    std::size_t totalSize;
};

static std::unordered_map<std::uint64_t, FFTResources> fftCache;
static std::mutex fftCacheMutex;

static std::uint64_t makeKey(std::size_t h, std::size_t w)
{
    return (static_cast<std::uint64_t>(h) << 32) | w;
}

/* ============================================================
   FFT Cross-Correlation (CUDA, PCIe-optimized)
   ============================================================ */

class FFTCrossCorrelation2D_CUDA
{
public:
    static Matrix2D<double> compute(
        const Matrix2D<double>& image,
        const Matrix2D<double>& kernel,
        PaddingMode paddingMode)
    {
        std::size_t pad = 0;
        if (paddingMode == PaddingMode::SAME)  pad = kernel.rows() / 2;
        if (paddingMode == PaddingMode::FULL)  pad = kernel.rows() - 1;

        std::size_t paddedH = image.rows() + 2 * pad;
        std::size_t paddedW = image.cols() + 2 * pad;

        std::size_t fftH = 1, fftW = 1;
        while (fftH < paddedH + kernel.rows() - 1) fftH <<= 1;
        while (fftW < paddedW + kernel.cols() - 1) fftW <<= 1;

        std::size_t totalSize = fftH * fftW;
        std::size_t bytes = totalSize * sizeof(cufftDoubleComplex);

        std::uint64_t key = makeKey(fftH, fftW);
        FFTResources* res = nullptr;

        {
            std::lock_guard<std::mutex> lock(fftCacheMutex);
            auto it = fftCache.find(key);
            if (it == fftCache.end())
            {
                FFTResources r{};
                r.totalSize = totalSize;

                cudaMalloc(&r.d_image, bytes);
                cudaMalloc(&r.d_kernel, bytes);

                cudaStreamCreate(&r.stream);
                cufftPlan2d(&r.plan, fftH, fftW, CUFFT_Z2Z);
                cufftSetStream(r.plan, r.stream);

                it = fftCache.emplace(key, r).first;
            }
            res = &it->second;
        }

        cufftDoubleComplex* h_image;
        cufftDoubleComplex* h_kernel;

        cudaMallocHost(&h_image, bytes);
        cudaMallocHost(&h_kernel, bytes);

        for (std::size_t i = 0; i < totalSize; ++i)
        {
            h_image[i] = {0.0, 0.0};
            h_kernel[i] = {0.0, 0.0};
        }

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

        cudaMemcpyAsync(res->d_image, h_image, bytes,
                        cudaMemcpyHostToDevice, res->stream);

        cudaMemcpyAsync(res->d_kernel, h_kernel, bytes,
                        cudaMemcpyHostToDevice, res->stream);

        cufftExecZ2Z(res->plan, res->d_image, res->d_image, CUFFT_FORWARD);
        cufftExecZ2Z(res->plan, res->d_kernel, res->d_kernel, CUFFT_FORWARD);

        std::size_t threads = 256;
        std::size_t blocks = (totalSize + threads - 1) / threads;

        multiplyWithConjugate<<<blocks, threads, 0, res->stream>>>(
            res->d_image, res->d_kernel, totalSize);

        cufftExecZ2Z(res->plan, res->d_image, res->d_image, CUFFT_INVERSE);

        cudaMemcpyAsync(h_image, res->d_image, bytes,
                        cudaMemcpyDeviceToHost, res->stream);

        cudaStreamSynchronize(res->stream);

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

        Matrix2D<double> output(outH, outW);
        double scale = static_cast<double>(fftH * fftW);

        for (std::size_t y = 0; y < outH; ++y)
            for (std::size_t x = 0; x < outW; ++x)
                output(y, x) =
                    h_image[(y + startY) * fftW + (x + startX)].x / scale;

        cudaFreeHost(h_image);
        cudaFreeHost(h_kernel);

        return output;
    }
};

#endif
