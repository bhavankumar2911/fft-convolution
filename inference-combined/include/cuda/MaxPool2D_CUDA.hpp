#pragma once
#include "core/Tensor.hpp"
#include <cuda_runtime.h>

/* ============================================================
   CUDA kernel: 2x2 MaxPool (stride 2)

   Each thread computes ONE output pixel for ONE (c, y, x).
   Grid: (C, ceil(outH/16), ceil(outW/16))
   Block: (1, 16, 16)
   ============================================================ */
template<typename T>
__global__ void maxPool2dKernel(
    const T* __restrict__ input,  // [C, inH, inW]
    T*                    output, // [C, outH, outW]
    int C,
    int inH,  int inW,
    int outH, int outW
) {
    int c = blockIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.z * blockDim.z + threadIdx.z;

    if (c >= C || y >= outH || x >= outW) return;

    int iy = y * 2;
    int ix = x * 2;

    T m = input[c * inH * inW + iy * inW + ix];  // (0,0)
    T v;

    v = input[c * inH * inW + iy       * inW + (ix + 1)]; if (v > m) m = v; // (0,1)
    v = input[c * inH * inW + (iy + 1) * inW +  ix     ]; if (v > m) m = v; // (1,0)
    v = input[c * inH * inW + (iy + 1) * inW + (ix + 1)]; if (v > m) m = v; // (1,1)

    output[c * outH * outW + y * outW + x] = m;
}

/* ============================================================
   MaxPool2D_CUDA
   - Same interface as MaxPool2D (static apply)
   - Returns a new Tensor with halved spatial dimensions
   ============================================================ */
template<typename T>
class MaxPool2D_CUDA {
public:
    static Tensor<T> apply(const Tensor<T>& input) {
        const int C    = input.channels;
        const int inH  = input.height;
        const int inW  = input.width;
        const int outH = inH / 2;
        const int outW = inW / 2;

        const std::size_t inBytes  = C * inH  * inW  * sizeof(T);
        const std::size_t outBytes = C * outH * outW * sizeof(T);

        T* d_input{nullptr};
        T* d_output{nullptr};
        cudaMalloc(&d_input,  inBytes);
        cudaMalloc(&d_output, outBytes);

        cudaMemcpy(d_input, input.data.data(), inBytes, cudaMemcpyHostToDevice);

        dim3 block(1, 16, 16);
        dim3 grid(
            C,
            (outH + 15) / 16,
            (outW + 15) / 16
        );

        maxPool2dKernel<T><<<grid, block>>>(
            d_input, d_output,
            C, inH, inW, outH, outW
        );
        cudaDeviceSynchronize();

        Tensor<T> out(C, outH, outW);
        cudaMemcpy(out.data.data(), d_output, outBytes, cudaMemcpyDeviceToHost);

        cudaFree(d_input);
        cudaFree(d_output);

        return out;
    }
};