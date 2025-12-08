#ifndef FFT_CONVOLVER_HPP
#define FFT_CONVOLVER_HPP

#include "FftTransform.hpp"
#include "Data2D.hpp"
#include <chrono>
#include <stdexcept>
#include <string>
#include <type_traits>

template<typename T>
class FftConvolver
{
    static_assert(std::is_floating_point<T>::value, "FftConvolver requires floating point type");

public:
    enum class OperationMode { Convolution, Correlation };

    struct ConvolutionResult
    {
        Data2D<T> resultMatrix;
        double elapsedMilliseconds;
    };

    static ConvolutionResult convolve(const Data2D<T> &image,
                                      const Data2D<T> &kernel,
                                      const std::string &outputMode = "full",
                                      OperationMode opMode = OperationMode::Correlation,
                                      bool padToNextPowerOfTwo = false)
    {
        int imageH = image.getHeight();
        int imageW = image.getWidth();
        int kernelH = kernel.getHeight();
        int kernelW = kernel.getWidth();

        if (imageH <= 0 || imageW <= 0) throw std::invalid_argument("Image has non-positive dims");
        if (kernelH <= 0 || kernelW <= 0) throw std::invalid_argument("Kernel has non-positive dims");
        if (outputMode != "full" && outputMode != "same" && outputMode != "valid")
            throw std::invalid_argument("outputMode must be 'full' or 'same' or 'valid'");

        int convH = imageH + kernelH - 1;
        int convW = imageW + kernelW - 1;
        int minH = convH, minW = convW;

        auto tStart = std::chrono::high_resolution_clock::now();

        using FFT = Fft2DTransformer<T>;
        using ComplexArr = ComplexArray<T>;
        ComplexArr fftImage  = FFT::compute2D(image,  minH, minW, true,  padToNextPowerOfTwo);
        ComplexArr fftKernel = FFT::compute2D(kernel, minH, minW, true,  padToNextPowerOfTwo);

        bool conjKernel = (opMode == OperationMode::Correlation);
        if (fftImage.sizeH() != fftKernel.sizeH() || fftImage.sizeW() != fftKernel.sizeW())
            throw std::runtime_error("FFT padded dimensions mismatch");

        int H = fftImage.sizeH(), W = fftImage.sizeW();
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                fftImage(y, x) *= (conjKernel ? std::conj(fftKernel(y, x)) : fftKernel(y, x));

        FFT::inverse2DInPlace(fftImage);

        Data2D<T> out = buildOutputFromPadded(fftImage, imageH, imageW, kernelH, kernelW, outputMode);

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        return ConvolutionResult{ out, elapsed };
    }

private:
    static Data2D<T> buildOutputFromPadded(const ComplexArray<T> &padded,
                                           int imageH, int imageW,
                                           int kernelH, int kernelW,
                                           const std::string &outputMode)
    {
        int paddedH = padded.sizeH();
        int paddedW = padded.sizeW();
        if (paddedH <= 0 || paddedW <= 0) return Data2D<T>();

        int convH = imageH + kernelH - 1;
        int convW = imageW + kernelW - 1;
        if (paddedH < convH || paddedW < convW) throw std::runtime_error("padded size smaller than convolution size");

        if (outputMode == "full")
        {
            Data2D<T> out(convH, convW);
            for (int y = 0; y < convH; ++y)
                for (int x = 0; x < convW; ++x)
                    out(y, x) = padded(y, x).real();
            return out;
        }
        else if (outputMode == "same")
        {
            Data2D<T> out(imageH, imageW);
            for (int y = 0; y < imageH; ++y)
                for (int x = 0; x < imageW; ++x)
                    out(y, x) = padded(y, x).real();
            return out;
        }
        else // valid
        {
            int outH = imageH - kernelH + 1;
            int outW = imageW - kernelW + 1;
            if (outH <= 0 || outW <= 0) return Data2D<T>();
            Data2D<T> out(outH, outW);
            for (int y = 0; y < outH; ++y)
                for (int x = 0; x < outW; ++x)
                    out(y, x) = padded(y, x).real();
            return out;
        }
    }
};

#endif // FFT_CONVOLVER_HPP
