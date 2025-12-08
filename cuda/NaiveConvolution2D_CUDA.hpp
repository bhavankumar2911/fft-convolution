#ifndef NAIVE_CONVOLUTION2D_CUDA_HPP
#define NAIVE_CONVOLUTION2D_CUDA_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <iostream>
#include <chrono>
#include <type_traits>
#include "../Data2D.hpp"

inline void checkCuda(cudaError_t e, const char* where = "")
{
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << where << ": " << cudaGetErrorString(e) << "\n";
        std::terminate();
    }
}

enum OutputModeInt { MODE_VALID = 0, MODE_SAME = 1, MODE_FULL = 2 };

template<typename T>
__global__ void naiveConvolutionKernelGeneric(
    const T* __restrict__ d_image,
    int imageH, int imageW,
    const T* __restrict__ d_kernel,
    int kernelH, int kernelW,
    T* __restrict__ d_output,
    int outH, int outW,
    int mode)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    T sum = static_cast<T>(0);

    if (mode == MODE_VALID)
    {
        int baseY = oy;
        int baseX = ox;
        for (int ky = 0; ky < kernelH; ++ky)
        {
            int iy = baseY + ky;
            int rowIdx = iy * imageW;
            int kRowIdx = ky * kernelW;
            for (int kx = 0; kx < kernelW; ++kx)
            {
                sum += d_image[rowIdx + (baseX + kx)] * d_kernel[kRowIdx + kx];
            }
        }
    }
    else if (mode == MODE_FULL)
    {
        for (int ky = 0; ky < kernelH; ++ky)
        {
            int kRowIdx = ky * kernelW;
            for (int kx = 0; kx < kernelW; ++kx)
            {
                int iy = oy - ky;
                int ix = ox - kx;
                if (iy < 0 || ix < 0 || iy >= imageH || ix >= imageW) continue;
                sum += d_image[iy * imageW + ix] * d_kernel[kRowIdx + kx];
            }
        }
    }
    else // MODE_SAME
    {
        int kyCenter = kernelH / 2;
        int kxCenter = kernelW / 2;
        for (int ky = 0; ky < kernelH; ++ky)
        {
            int kRowIdx = ky * kernelW;
            for (int kx = 0; kx < kernelW; ++kx)
            {
                int iy = oy + ky - kyCenter;
                int ix = ox + kx - kxCenter;
                if (iy < 0 || ix < 0 || iy >= imageH || ix >= imageW) continue;
                sum += d_image[iy * imageW + ix] * d_kernel[kRowIdx + kx];
            }
        }
    }

    d_output[oy * outW + ox] = sum;
}

template<typename T>
class NaiveConvolution2D_CUDA
{
    static_assert(std::is_floating_point<T>::value, "NaiveConvolution2D_CUDA requires floating point type");

public:
    struct ConvolutionResult
    {
        Data2D<T> result;
        double elapsedMilliseconds; // kernel compute time (ms)
    };

private:
    const Data2D<T>& imageHost_;
    const Data2D<T>& kernelHost_;
    int imageH_, imageW_, kernelH_, kernelW_;

public:
    NaiveConvolution2D_CUDA(const Data2D<T>& image, const Data2D<T>& kernel)
        : imageHost_(image), kernelHost_(kernel)
    {
        imageH_ = imageHost_.getHeight();
        imageW_ = imageHost_.getWidth();
        kernelH_ = kernelHost_.getHeight();
        kernelW_ = kernelHost_.getWidth();
        if (imageH_ < 0 || imageW_ < 0 || kernelH_ < 0 || kernelW_ < 0)
            throw std::invalid_argument("Negative dimensions are not allowed");
    }

    ConvolutionResult computeConvolution(const std::string &outputMode = "valid")
    {
        int mode;
        if (outputMode == "valid") mode = MODE_VALID;
        else if (outputMode == "same") mode = MODE_SAME;
        else if (outputMode == "full") mode = MODE_FULL;
        else throw std::invalid_argument("outputMode must be 'valid', 'same' or 'full'");

        if (kernelH_ <= 0 || kernelW_ <= 0 || imageH_ <= 0 || imageW_ <= 0)
            return ConvolutionResult{ Data2D<T>(), 0.0 };

        int outH = 0, outW = 0;
        if (mode == MODE_VALID) {
            outH = imageH_ - kernelH_ + 1;
            outW = imageW_ - kernelW_ + 1;
            if (outH <= 0 || outW <= 0) return ConvolutionResult{ Data2D<T>(), 0.0 };
        }
        else if (mode == MODE_SAME) {
            outH = imageH_;
            outW = imageW_;
        }
        else { // full
            outH = imageH_ + kernelH_ - 1;
            outW = imageW_ + kernelW_ - 1;
        }

        Data2D<T> result(outH, outW);

        size_t imageBytes = static_cast<size_t>(imageH_) * imageW_ * sizeof(T);
        size_t kernelBytes = static_cast<size_t>(kernelH_) * kernelW_ * sizeof(T);
        size_t outBytes = static_cast<size_t>(outH) * outW * sizeof(T);

        T *d_image = nullptr, *d_kernel = nullptr, *d_output = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_image), imageBytes), "cudaMalloc d_image");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_kernel), kernelBytes), "cudaMalloc d_kernel");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_output), outBytes), "cudaMalloc d_output");

        checkCuda(cudaMemcpy(d_image, imageHost_.data().data(), imageBytes, cudaMemcpyHostToDevice), "H2D image");
        checkCuda(cudaMemcpy(d_kernel, kernelHost_.data().data(), kernelBytes, cudaMemcpyHostToDevice), "H2D kernel");

        dim3 block(16, 16);
        dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);

        cudaEvent_t startEvent, stopEvent;
        checkCuda(cudaEventCreate(&startEvent), "createEvent start");
        checkCuda(cudaEventCreate(&stopEvent), "createEvent stop");
        checkCuda(cudaEventRecord(startEvent), "record start");

        naiveConvolutionKernelGeneric<T><<<grid, block>>>(
            d_image, imageH_, imageW_,
            d_kernel, kernelH_, kernelW_,
            d_output, outH, outW, mode
        );
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaEventRecord(stopEvent), "record stop");
        checkCuda(cudaEventSynchronize(stopEvent), "synchronize stop");

        float ms = 0.0f;
        checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent), "elapsedTime");

        checkCuda(cudaMemcpy(result.data().data(), d_output, outBytes, cudaMemcpyDeviceToHost), "D2H output");

        checkCuda(cudaFree(d_image), "free d_image");
        checkCuda(cudaFree(d_kernel), "free d_kernel");
        checkCuda(cudaFree(d_output), "free d_output");
        checkCuda(cudaEventDestroy(startEvent), "destroy start");
        checkCuda(cudaEventDestroy(stopEvent), "destroy stop");

        return ConvolutionResult{ result, static_cast<double>(ms) };
    }
};

#endif // NAIVE_CONVOLUTION2D_CUDA_HPP
