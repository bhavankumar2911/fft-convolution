#pragma once
#include "core/Tensor.hpp"
#include <cuda_runtime.h>

/* ============================================================
   CUDA kernel: elementwise ReLU
   Each thread handles one element of the tensor.
   ============================================================ */
template<typename T>
__global__ void reluKernel(T* data, std::size_t n) {
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = data[idx] > T(0) ? data[idx] : T(0);
}

/* ============================================================
   ActivationReLU_CUDA
   - Accepts a Tensor whose data is on the HOST
   - Uploads, applies ReLU in-place on GPU, downloads back
   - Same interface as ActivationReLU (static apply)
   ============================================================ */
template<typename T>
class ActivationReLU_CUDA {
public:
    static void apply(Tensor<T>& t) {
        const std::size_t n     = t.data.size();
        const std::size_t bytes = n * sizeof(T);

        T* d_data{nullptr};
        cudaMalloc(&d_data, bytes);
        cudaMemcpy(d_data, t.data.data(), bytes, cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks  = (n + threads - 1) / threads;
        reluKernel<T><<<blocks, threads>>>(d_data, n);
        cudaDeviceSynchronize();

        cudaMemcpy(t.data.data(), d_data, bytes, cudaMemcpyDeviceToHost);
        cudaFree(d_data);
    }
};