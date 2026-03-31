/*
 * bench_conv.cpp
 *
 * Benchmarks 4 backends for a single (image_size, kernel_size) combination:
 *   1. CPU Naive
 *   2. CPU FFT  (FFTW)
 *   3. GPU Naive
 *   4. GPU FFT  (cuFFT)
 *
 * Also records theoretical operation counts for CPU backends:
 *   naive_ops  = H^2 * K^2          (multiply-adds, serial)
 *   fft_ops    = 6 * N^2 * log2(N)  (N = nextpow2(H+K-1))
 *
 * GPU theoretical costs are NOT recorded — GPU parallelism
 * discounts the serial op count making it misleading.
 *
 * Usage:
 *   ./bench_conv <image_size> <kernel_size> <data_dir> <out_csv>
 *
 * Output CSV columns:
 *   image_size, kernel_size, coverage,
 *   naive_ops, fft_ops,
 *   cpu_naive_ms, cpu_fft_ms,
 *   gpu_naive_ms, gpu_fft_ms,
 *   cpu_winner, gpu_winner
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <filesystem>

#include "core/Tensor.hpp"
#include "data/BinaryTensorLoader.hpp"
#include "cpu/NaiveCPUConvolution2D.hpp"
#include "cpu/FFTConvolution2D_CPU.hpp"
#include "cuda/NaiveCUDAConvolution2D.hpp"
#include "cuda/FFTConvolution2D_CUDA.hpp"

// -------------------------------------------------
// Config
// -------------------------------------------------
static constexpr int WARMUP_ITERS = 3;
static constexpr int TIMED_ITERS  = 10;
static constexpr int IN_CHANNELS  = 1;
static constexpr int OUT_CHANNELS = 1;

using Real = float;

// -------------------------------------------------
// Path helpers
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

// -------------------------------------------------
// Theoretical cost helpers
// -------------------------------------------------

// Naive: H^2 * K^2 multiply-adds
long long naiveOps(int H, int K) {
    return static_cast<long long>(H) * H * K * K;
}

// FFT: 6 * N^2 * log2(N) where N = nextpow2(H + K - 1)
long long fftOps(int H, int K) {
    std::size_t N = 1;
    while (N < static_cast<std::size_t>(H + K - 1)) N <<= 1;
    return static_cast<long long>(6.0 * N * N * std::log2(N));
}

// -------------------------------------------------
// Benchmark runner
// -------------------------------------------------
template<typename ConvT>
double benchmark(ConvT& conv, const Tensor<Real>& input) {
    for (int i = 0; i < WARMUP_ITERS; ++i)
        conv.forward(input);

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

    const int         H        = std::stoi(argv[1]);
    const int         K        = std::stoi(argv[2]);
    const std::string dataDir  = argv[3];
    const std::string csvPath  = argv[4];
    const int         pad      = K / 2;
    const float       coverage = static_cast<float>(K) / H;

    // Theoretical costs — CPU only
    long long nOps = naiveOps(H, K);
    long long fOps = fftOps(H, K);

    std::cout << "Benchmarking: image=" << H << "x" << H
              << "  kernel=" << K << "x" << K
              << "  coverage=" << coverage
              << "  naive_ops=" << nOps
              << "  fft_ops="   << fOps << "\n";

    // -------------------------------------------------
    // Load input
    // -------------------------------------------------
    auto raw = BinaryTensorLoader::loadVector<Real>(imagePath(dataDir, H));
    if (raw.size() != static_cast<std::size_t>(IN_CHANNELS * H * H)) {
        std::cerr << "ERROR: image size mismatch\n";
        return 1;
    }
    Tensor<Real> input(IN_CHANNELS, H, H);
    input.data = raw;

    const std::string wPath = weightPath(dataDir, K);
    const std::string bPath = biasPath(dataDir, K);

    // -------------------------------------------------
    // Backend 1 — CPU Naive
    // -------------------------------------------------
    std::cout << "  [1/4] CPU Naive...  " << std::flush;
    NaiveCPUConvolution2D<Real> cpuNaive(
        IN_CHANNELS, OUT_CHANNELS, K, pad, wPath, bPath);
    double cpuNaiveMs = benchmark(cpuNaive, input);
    std::cout << cpuNaiveMs << " ms\n";

    // -------------------------------------------------
    // Backend 2 — CPU FFT
    // -------------------------------------------------
    std::cout << "  [2/4] CPU FFT...    " << std::flush;
    FFTConvolution2D_CPU<Real> cpuFFT(
        IN_CHANNELS, OUT_CHANNELS, K, pad, wPath, bPath);
    double cpuFFTMs = benchmark(cpuFFT, input);
    std::cout << cpuFFTMs << " ms\n";

    // -------------------------------------------------
    // Backend 3 — GPU Naive
    // -------------------------------------------------
    std::cout << "  [3/4] GPU Naive...  " << std::flush;
    NaiveCUDAConvolution2D<Real> gpuNaive(
        IN_CHANNELS, OUT_CHANNELS, K, pad, wPath, bPath);
    double gpuNaiveMs = benchmark(gpuNaive, input);
    std::cout << gpuNaiveMs << " ms\n";

    // -------------------------------------------------
    // Backend 4 — GPU FFT
    // -------------------------------------------------
    std::cout << "  [4/4] GPU FFT...    " << std::flush;
    FFTConvolution2D_CUDA<Real> gpuFFT(
        IN_CHANNELS, OUT_CHANNELS, K, pad, wPath, bPath);
    double gpuFFTMs = benchmark(gpuFFT, input);
    std::cout << gpuFFTMs << " ms\n";

    // -------------------------------------------------
    // Winners
    // -------------------------------------------------
    std::string cpuWinner = (cpuNaiveMs <= cpuFFTMs) ? "cpu_naive" : "cpu_fft";
    std::string gpuWinner = (gpuNaiveMs <= gpuFFTMs) ? "gpu_naive" : "gpu_fft";

    std::cout << "  CPU winner: " << cpuWinner << "\n";
    std::cout << "  GPU winner: " << gpuWinner << "\n\n";

    // -------------------------------------------------
    // Append to CSV
    // -------------------------------------------------
    bool writeHeader = !std::filesystem::exists(csvPath);
    std::ofstream csv(csvPath, std::ios::app);

    if (writeHeader)
        csv << "image_size,kernel_size,coverage,"
            << "naive_ops,fft_ops,"
            << "cpu_naive_ms,cpu_fft_ms,"
            << "gpu_naive_ms,gpu_fft_ms,"
            << "cpu_winner,gpu_winner\n";

    csv << H          << ","
        << K          << ","
        << coverage   << ","
        << nOps       << ","
        << fOps       << ","
        << cpuNaiveMs << ","
        << cpuFFTMs   << ","
        << gpuNaiveMs << ","
        << gpuFFTMs   << ","
        << cpuWinner  << ","
        << gpuWinner  << "\n";

    csv.close();
    return 0;
}