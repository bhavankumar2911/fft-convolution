#pragma once
#include "core/IConvolution2D.hpp"
#include "data/BinaryTensorLoader.hpp"
#include "utils/Timer.hpp"

#include <cuda_runtime.h>
#include <vector>

/* ============================================================
   CUDA kernel: naive 2D convolution (cross-correlation)

   Each thread computes ONE output pixel for ONE (oc, y, x).
   Grid: (outC, ceil(H/16), ceil(W/16))
   Block: (1, 16, 16)
   ============================================================ */
template<typename T>
__global__ void naiveConv2dKernel(
    const T* __restrict__ input,   // [inC, H, W]
    const T* __restrict__ weights, // [outC, inC, k, k]
    const T* __restrict__ bias,    // [outC]
    T*       output,               // [outC, H, W]
    int inC, int outC,
    int H, int W,
    int k, int pad
) {
    int oc = blockIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int x  = blockIdx.z * blockDim.z + threadIdx.z;

    if (oc >= outC || y >= H || x >= W) return;

    T sum = bias[oc];

    for (int ic = 0; ic < inC; ++ic)
        for (int ky = 0; ky < k; ++ky)
            for (int kx = 0; kx < k; ++kx) {
                int iy = y + ky - pad;
                int ix = x + kx - pad;
                if (iy >= 0 && ix >= 0 && iy < H && ix < W) {
                    sum += input [ic * H * W + iy * W + ix] *
                           weights[oc * inC * k * k +
                                   ic * k * k +
                                   ky * k + kx];
                }
            }

    output[oc * H * W + y * W + x] = sum;
}

/* ============================================================
   NaiveCUDAConvolution2D
   - Weights and bias are uploaded to GPU once in constructor
   - forward() launches the kernel and copies result back
   ============================================================ */
template<typename T>
class NaiveCUDAConvolution2D : public IConvolution2D<T> {
    int inC, outC, k, pad;

    T* d_weights{nullptr};
    T* d_bias{nullptr};
    std::size_t wSize, bSize;

    double lastMs{0.0};

public:
    NaiveCUDAConvolution2D(
        int inCh, int outCh, int ks, int p,
        const std::string& wPath,
        const std::string& bPath
    )
        : inC(inCh), outC(outCh), k(ks), pad(p)
    {
        // Load weights once on host, upload to device
        auto hW = BinaryTensorLoader::loadVector<T>(wPath);
        auto hB = BinaryTensorLoader::loadVector<T>(bPath);

        wSize = hW.size() * sizeof(T);
        bSize = hB.size() * sizeof(T);

        cudaMalloc(&d_weights, wSize);
        cudaMalloc(&d_bias,    bSize);

        cudaMemcpy(d_weights, hW.data(), wSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias,    hB.data(), bSize, cudaMemcpyHostToDevice);
    }

    ~NaiveCUDAConvolution2D() {
        if (d_weights) cudaFree(d_weights);
        if (d_bias)    cudaFree(d_bias);
    }

    Tensor<T> forward(const Tensor<T>& in) override {
        CpuTimer timer;
        timer.start();

        const int H = in.height;
        const int W = in.width;
        const std::size_t inBytes  = inC  * H * W * sizeof(T);
        const std::size_t outBytes = outC * H * W * sizeof(T);

        // Upload input
        T* d_input{nullptr};
        T* d_output{nullptr};
        cudaMalloc(&d_input,  inBytes);
        cudaMalloc(&d_output, outBytes);
        cudaMemcpy(d_input, in.data.data(), inBytes, cudaMemcpyHostToDevice);

        // Launch kernel
        // Each block covers a 16x16 tile of (y,x) for one output channel
        dim3 block(1, 16, 16);
        dim3 grid(
            outC,
            (H + 15) / 16,
            (W + 15) / 16
        );

        naiveConv2dKernel<T><<<grid, block>>>(
            d_input, d_weights, d_bias, d_output,
            inC, outC, H, W, k, pad
        );
        cudaDeviceSynchronize();

        // Copy result back
        Tensor<T> out(outC, H, W);
        cudaMemcpy(out.data.data(), d_output, outBytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        lastMs = timer.stop();
        return out;
    }

    double lastExecutionTimeMs() const override { return lastMs; }
};