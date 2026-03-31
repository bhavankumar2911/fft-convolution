/*
 * bench_conv.cpp
 *
 * Standalone convolution benchmark using the exact same backend classes
 * as the CNN inference pipeline. Compares all 4 backends for a single
 * (image_size, kernel_size) combination.
 *
 * Usage:
 *   ./bench_conv <image_size> <kernel_size> <data_dir> <out_csv>
 *
 * Example:
 *   ./bench_conv 256 7 ./bench_data ./bench_results/results.csv
 *
 * Output CSV columns:
 *   image_size, kernel_size, ratio,
 *   cpu_naive_ms, gpu_naive_ms, gpu_fft_ms, gpu_hybrid_ms, winner
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

#include "core/Tensor.hpp"
#include "data/BinaryTensorLoader.hpp"
#include "cpu/NaiveCPUConvolution2D.hpp"
#include "cuda/NaiveCUDAConvolution2D.hpp"
#include "cuda/FFTConvolution2D_CUDA.hpp"

// -------------------------------------------------
// Config
// -------------------------------------------------
static constexpr int   WARMUP_ITERS  = 3;
static constexpr int   TIMED_ITERS   = 10;
static constexpr int   IN_CHANNELS   = 1;
static constexpr int   OUT_CHANNELS  = 1;

using Real = float;

// -------------------------------------------------
// Helpers
// -------------------------------------------------
std::string imagePath(const std::string& dir, int H) {
    return dir + "/images/"
         + std::to_string(H) + "x" + std::to_string(H)
         + "_ic" + std::to_string(IN_CHANNELS) + ".bin";
}

std::string weightPath(const std::string& dir, int K) {
    return dir + "/weights/"
         + std::to_string(K) + "x" + std::to_string(K)
         + "_ic" + std::to_string(IN_CHANNELS)
         + "_oc" + std::to_string(OUT_CHANNELS) + ".bin";
}

std::string biasPath(const std::string& dir, int K) {
    return dir + "/biases/"
         + std::to_string(K) + "x" + std::to_string(K)
         + "_oc" + std::to_string(OUT_CHANNELS) + ".bin";
}

/* --------------------------------------------------------
   Run one backend for WARMUP_ITERS + TIMED_ITERS.
   Returns avg ms over timed iterations only.
-------------------------------------------------------- */
template<typename ConvT>
double benchmark(ConvT& conv, const Tensor<Real>& input) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; ++i)
        conv.forward(input);

    // Timed
    double total = 0.0;
    for (int i = 0; i < TIMED_ITERS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        conv.forward(input);
        auto t1 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return total / TIMED_ITERS;
}

// -------------------------------------------------
// Main
// -------------------------------------------------
int main(int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <image_size> <kernel_size> <data_dir> <out_csv>\n";
        return 1;
    }

    const int         H       = std::stoi(argv[1]);
    const int         K       = std::stoi(argv[2]);
    const std::string dataDir = argv[3];
    const std::string csvPath = argv[4];
    const int         pad     = K / 2;  // same padding
    const float       coverage = static_cast<float>(K) / H;

    std::cout << "Benchmarking: image=" << H << "x" << H
              << "  kernel=" << K << "x" << K
              << "  coverage=" << coverage << "\n";

    // -------------------------------------------------
    // Load input tensor from generated data
    // -------------------------------------------------
    auto raw = BinaryTensorLoader::loadVector<Real>(imagePath(dataDir, H));

    if (raw.size() != static_cast<std::size_t>(IN_CHANNELS * H * H)) {
        std::cerr << "ERROR: image size mismatch\n";
        return 1;
    }

    Tensor<Real> input(IN_CHANNELS, H, H);
    input.data = raw;  // no normalization for bench — raw values fine

    const std::string wPath = weightPath(dataDir, K);
    const std::string bPath = biasPath(dataDir, K);

    // -------------------------------------------------
    // Backend 1 — CPU Naive
    // -------------------------------------------------
    std::cout << "  [1/3] CPU Naive...  " << std::flush;
    NaiveCPUConvolution2D<Real> cpuNaive(
        IN_CHANNELS, OUT_CHANNELS, K, pad, wPath, bPath);
    double cpuNaiveMs = benchmark(cpuNaive, input);
    std::cout << cpuNaiveMs << " ms\n";

    // -------------------------------------------------
    // Backend 2 — GPU Naive
    // -------------------------------------------------
    std::cout << "  [2/3] GPU Naive...  " << std::flush;
    NaiveCUDAConvolution2D<Real> gpuNaive(
        IN_CHANNELS, OUT_CHANNELS, K, pad, wPath, bPath);
    double gpuNaiveMs = benchmark(gpuNaive, input);
    std::cout << gpuNaiveMs << " ms\n";

    // -------------------------------------------------
    // Backend 3 — GPU FFT
    // -------------------------------------------------
    std::cout << "  [3/3] GPU FFT...    " << std::flush;
    FFTConvolution2D_CUDA<Real> gpuFFT(
        IN_CHANNELS, OUT_CHANNELS, K, pad, wPath, bPath);
    double gpuFFTMs = benchmark(gpuFFT, input);
    std::cout << gpuFFTMs << " ms\n";

    // -------------------------------------------------
    // Determine winner among GPU backends
    // -------------------------------------------------
    std::string winner = (gpuNaiveMs <= gpuFFTMs) ? "gpu_naive" : "gpu_fft";
    std::cout << "  Winner (GPU): " << winner << "\n\n";

    // -------------------------------------------------
    // Append result to CSV
    // -------------------------------------------------
    bool writeHeader = !std::filesystem::exists(csvPath);
    std::ofstream csv(csvPath, std::ios::app);

    if (writeHeader)
        csv << "image_size,kernel_size,coverage,"
            << "cpu_naive_ms,gpu_naive_ms,gpu_fft_ms,"
            << "winner\n";

    csv << H << ","
        << K << ","
        << coverage << ","
        << cpuNaiveMs  << ","
        << gpuNaiveMs  << ","
        << gpuFFTMs << ","
        << winner   << "\n";

    csv.close();
    return 0;
}