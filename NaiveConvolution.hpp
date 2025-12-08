#ifndef NAIVE_CONVOLUTION_HPP
#define NAIVE_CONVOLUTION_HPP

#include "Data2D.hpp"
#include <chrono>
#include <stdexcept>
#include <string>
#include <type_traits>

template<typename T>
class NaiveConvolution2D
{
    static_assert(std::is_floating_point<T>::value, "NaiveConvolution2D requires floating point type");

public:
    struct ConvolutionResult
    {
        Data2D<T> result;
        double elapsedMilliseconds;
    };

private:
    const Data2D<T> &image_;
    const Data2D<T> &kernel_;
    int imageH_, imageW_, kernelH_, kernelW_;

    inline T getImageValueOrZero(int y, int x) const
    {
        if (y < 0 || x < 0 || y >= imageH_ || x >= imageW_) return static_cast<T>(0);
        return image_(y, x);
    }

public:
    NaiveConvolution2D(const Data2D<T>& img, const Data2D<T>& ker)
        : image_(img), kernel_(ker)
    {
        imageH_ = image_.getHeight(); imageW_ = image_.getWidth();
        kernelH_ = kernel_.getHeight(); kernelW_ = kernel_.getWidth();
        if (imageH_ < 0 || imageW_ < 0 || kernelH_ < 0 || kernelW_ < 0)
            throw std::invalid_argument("Negative dimensions are not allowed");
    }

    ConvolutionResult computeConvolution(const std::string &outputMode = "valid")
    {
        auto tStart = std::chrono::high_resolution_clock::now();

        if (kernelH_ <= 0 || kernelW_ <= 0 || imageH_ <= 0 || imageW_ <= 0)
            return ConvolutionResult{ Data2D<T>(), 0.0 };

        if (outputMode != "valid" && outputMode != "same" && outputMode != "full")
            throw std::invalid_argument("outputMode must be 'valid', 'same' or 'full'");

        if (outputMode == "valid")
        {
            int outH = imageH_ - kernelH_ + 1;
            int outW = imageW_ - kernelW_ + 1;
            if (outH <= 0 || outW <= 0) return ConvolutionResult{ Data2D<T>(), 0.0 };
            Data2D<T> res(outH, outW);

            for (int y = 0; y < outH; ++y)
            {
                for (int x = 0; x < outW; ++x)
                {
                    T s = static_cast<T>(0);
                    const int baseY = y;
                    const int baseX = x;
                    const std::vector<T>& imgData = image_.data();
                    const std::vector<T>& kerData = kernel_.data();
                    for (int ky = 0; ky < kernelH_; ++ky)
                    {
                        size_t rowOffset = static_cast<size_t>(baseY + ky) * imageW_;
                        size_t kOffset   = static_cast<size_t>(ky) * kernelW_;
                        for (int kx = 0; kx < kernelW_; ++kx)
                        {
                            s += imgData[rowOffset + (baseX + kx)] * kerData[kOffset + kx];
                        }
                    }
                    res(y, x) = s;
                }
            }

            auto tEnd = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            return ConvolutionResult{ res, elapsed };
        }
        else if (outputMode == "full")
        {
            int outH = imageH_ + kernelH_ - 1;
            int outW = imageW_ + kernelW_ - 1;
            Data2D<T> res(outH, outW);

            for (int y = 0; y < outH; ++y)
            {
                for (int x = 0; x < outW; ++x)
                {
                    T s = static_cast<T>(0);
                    for (int ky = 0; ky < kernelH_; ++ky)
                    {
                        for (int kx = 0; kx < kernelW_; ++kx)
                        {
                            int iy = y - ky;
                            int ix = x - kx;
                            if (iy < 0 || ix < 0 || iy >= imageH_ || ix >= imageW_) continue;
                            s += image_(iy, ix) * kernel_(ky, kx);
                        }
                    }
                    res(y, x) = s;
                }
            }

            auto tEnd = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            return ConvolutionResult{ res, elapsed };
        }
        else // "same"
        {
            int outH = imageH_, outW = imageW_;
            Data2D<T> res(outH, outW);

            for (int y = 0; y < outH; ++y)
            {
                for (int x = 0; x < outW; ++x)
                {
                    T s = static_cast<T>(0);
                    for (int ky = 0; ky < kernelH_; ++ky)
                    {
                        for (int kx = 0; kx < kernelW_; ++kx)
                        {
                            int iy = y + ky - (kernelH_/2);
                            int ix = x + kx - (kernelW_/2);
                            T iv = getImageValueOrZero(iy, ix);
                            s += iv * kernel_(ky, kx);
                        }
                    }
                    res(y, x) = s;
                }
            }

            auto tEnd = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
            return ConvolutionResult{ res, elapsed };
        }
    }
};

#endif // NAIVE_CONVOLUTION_HPP
