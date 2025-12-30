#ifndef FFT_CROSS_CORRELATION_2D_HPP
#define FFT_CROSS_CORRELATION_2D_HPP

#include "Matrix2D.hpp"
#include "PaddingMode.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

namespace fft_internal
{
    inline void fft1D(std::vector<std::complex<double>>& data, bool inverse)
    {
        std::size_t n = data.size();

        for (std::size_t i = 1, j = 0; i < n; ++i)
        {
            std::size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j |= bit;
            if (i < j) std::swap(data[i], data[j]);
        }

        for (std::size_t len = 2; len <= n; len <<= 1)
        {
            double angle = 2 * M_PI / len * (inverse ? 1 : -1);
            std::complex<double> wlen(std::cos(angle), std::sin(angle));

            for (std::size_t i = 0; i < n; i += len)
            {
                std::complex<double> w(1);
                for (std::size_t j = 0; j < len / 2; ++j)
                {
                    auto u = data[i + j];
                    auto v = data[i + j + len / 2] * w;
                    data[i + j] = u + v;
                    data[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (inverse)
            for (auto& x : data) x /= n;
    }

    inline void fft2D(std::vector<std::complex<double>>& data,
                      std::size_t H,
                      std::size_t W,
                      bool inverse)
    {
        std::vector<std::complex<double>> temp(std::max(H, W));

        for (std::size_t y = 0; y < H; ++y)
        {
            for (std::size_t x = 0; x < W; ++x)
                temp[x] = data[y * W + x];

            temp.resize(W);
            fft1D(temp, inverse);

            for (std::size_t x = 0; x < W; ++x)
                data[y * W + x] = temp[x];
        }

        for (std::size_t x = 0; x < W; ++x)
        {
            for (std::size_t y = 0; y < H; ++y)
                temp[y] = data[y * W + x];

            temp.resize(H);
            fft1D(temp, inverse);

            for (std::size_t y = 0; y < H; ++y)
                data[y * W + x] = temp[y];
        }
    }
}

class FFTCrossCorrelation2D
{
public:
    template<typename T>
    static Matrix2D<T> compute(
        const Matrix2D<T>& image,
        const Matrix2D<T>& kernel,
        PaddingMode paddingMode
    )
    {
        std::size_t pad;
        if (paddingMode == PaddingMode::SAME)  pad = kernel.rows() / 2;
        if (paddingMode == PaddingMode::VALID) pad = 0;
        if (paddingMode == PaddingMode::FULL)  pad = kernel.rows() - 1;

        std::size_t paddedH = image.rows() + 2 * pad;
        std::size_t paddedW = image.cols() + 2 * pad;

        std::size_t fftH = 1;
        std::size_t fftW = 1;
        while (fftH < paddedH + kernel.rows() - 1) fftH <<= 1;
        while (fftW < paddedW + kernel.cols() - 1) fftW <<= 1;

        std::vector<std::complex<double>> imageFreq(fftH * fftW, 0.0);
        std::vector<std::complex<double>> kernelFreq(fftH * fftW, 0.0);

        // Image placement (with padding)
        for (std::size_t y = 0; y < image.rows(); ++y)
            for (std::size_t x = 0; x < image.cols(); ++x)
                imageFreq[(y + pad) * fftW + (x + pad)] = image(y, x);

        // ✅ Kernel center alignment (CRITICAL FIX)
        std::size_t kCenterY = kernel.rows() / 2;
        std::size_t kCenterX = kernel.cols() / 2;

        for (std::size_t y = 0; y < kernel.rows(); ++y)
        {
            for (std::size_t x = 0; x < kernel.cols(); ++x)
            {
                std::size_t shiftedY = (y + fftH - kCenterY) % fftH;
                std::size_t shiftedX = (x + fftW - kCenterX) % fftW;
                kernelFreq[shiftedY * fftW + shiftedX] = kernel(y, x);
            }
        }

        fft_internal::fft2D(imageFreq, fftH, fftW, false);
        fft_internal::fft2D(kernelFreq, fftH, fftW, false);

        // Cross-correlation
        for (std::size_t i = 0; i < imageFreq.size(); ++i)
            imageFreq[i] *= std::conj(kernelFreq[i]);

        fft_internal::fft2D(imageFreq, fftH, fftW, true);

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

        for (std::size_t y = 0; y < outH; ++y)
            for (std::size_t x = 0; x < outW; ++x)
                output(y, x) =
                    static_cast<T>(
                        imageFreq[(y + startY) * fftW + (x + startX)].real()
                    );

        return output;
    }
};

#endif
