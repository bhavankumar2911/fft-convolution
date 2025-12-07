#ifndef FFT_CONVOLVER_HPP
#define FFT_CONVOLVER_HPP

#include "FftTransform.hpp"
#include "Data2D.hpp"
#include <chrono>
#include <stdexcept>
#include <string>
#include <complex>

class FftConvolver
{
public:
    enum class OperationMode
    {
        Convolution, // IFFT(FFT(image) * FFT(kernel)) -> flipped-kernel convolution
        Correlation  // IFFT(FFT(image) * conj(FFT(kernel))) -> cross-correlation (no flip)
    };

    struct ConvolutionResult
    {
        Data2D resultMatrix;
        double elapsedMilliseconds;
    };

    // outputMode: "full" | "same" | "valid"
    // operationMode: Convolution (mathematical) or Correlation (CNN-style)
    // padToNextPowerOfTwo: pad FFT sizes to next power of two for performance
    // replace existing convolve with this (top-left extraction semantics)
    static ConvolutionResult convolve(const Data2D &image,
                                    const Data2D &kernel,
                                    const std::string &outputMode = "full",
                                    OperationMode operationMode = OperationMode::Correlation, // correlation by default
                                    bool padToNextPowerOfTwo = false) // recommend false for debugging
    {
        int imageH = image.getHeight();
        int imageW = image.getWidth();
        int kernelH = kernel.getHeight();
        int kernelW = kernel.getWidth();

        if (imageH <= 0 || imageW <= 0) throw std::invalid_argument("Image has non-positive dimensions");
        if (kernelH <= 0 || kernelW <= 0) throw std::invalid_argument("Kernel has non-positive dimensions");
        if (outputMode != "full" && outputMode != "same" && outputMode != "valid")
            throw std::invalid_argument("outputMode must be 'full' or 'same' or 'valid'");

        int convH = imageH + kernelH - 1;
        int convW = imageW + kernelW - 1;

        int minHeight = convH;
        int minWidth  = convW;

        auto tStart = std::chrono::high_resolution_clock::now();

        // build padded matrices with top-left placement
        ComplexMatrix fftImage  = Fft2DTransformer::compute2D(image,  minHeight, minWidth, true,  padToNextPowerOfTwo);
        ComplexMatrix fftKernel = Fft2DTransformer::compute2D(kernel, minHeight, minWidth, true,  padToNextPowerOfTwo);

        // multiply: for correlation use conj(kernel)
        bool useConj = (operationMode == OperationMode::Correlation);
        multiplyInPlace(fftImage, fftKernel, useConj);

        // inverse 2D in place
        Fft2DTransformer::inverse2DInPlace(fftImage);

        // extract result using top-left semantics (see buildOutputFromPadded below)
        Data2D output = buildOutputFromPadded(fftImage, imageH, imageW, kernelH, kernelW, outputMode);

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        return ConvolutionResult{output, elapsedMs};
    }

private:
    // multiply matrices element-wise, optionally using conjugation of b (for correlation)
    static void multiplyInPlace(ComplexMatrix &a, const ComplexMatrix &b, bool useConjugateOfB)
    {
        int h = static_cast<int>(a.size());
        if (h == 0) return;
        int w = static_cast<int>(a[0].size());
        if (static_cast<int>(b.size()) != h || static_cast<int>(b[0].size()) != w)
            throw std::runtime_error("multiplyInPlace: dimension mismatch");

        if (!useConjugateOfB)
        {
            cout << "not using conj";
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    a[y][x] *= b[y][x];
        }
        else
        {
            cout << "using conj";
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    a[y][x] *= std::conj(b[y][x]);
        }
    }

    // replace existing buildOutputFromPadded with this
    static Data2D buildOutputFromPadded(const ComplexMatrix &padded,
                                        int imageH, int imageW,
                                        int kernelH, int kernelW,
                                        const std::string &outputMode)
    {
        int paddedH = static_cast<int>(padded.size());
        if (paddedH == 0) return Data2D();
        int paddedW = static_cast<int>(padded[0].size());

        int convH = imageH + kernelH - 1;
        int convW = imageW + kernelW - 1;

        if (paddedH < convH || paddedW < convW)
            throw std::runtime_error("buildOutputFromPadded: padded size smaller than convolution size");

        // Note: with top-left anchor semantics the linear cross-correlation / convolution result
        // occupies the top-left convH x convW region of the padded IFFT matrix.

        if (outputMode == "full")
        {
            Data2D out(convH, convW);
            for (int y = 0; y < convH; ++y)
                for (int x = 0; x < convW; ++x)
                    out.getMatrix()[y][x] = padded[y][x].real();
            return out;
        }
        else if (outputMode == "same")
        {
            Data2D out(imageH, imageW);
            for (int y = 0; y < imageH; ++y)
                for (int x = 0; x < imageW; ++x)
                    out.getMatrix()[y][x] = padded[y][x].real();
            return out;
        }
        else if (outputMode == "valid")
        {
            int outH = imageH - kernelH + 1;
            int outW = imageW - kernelW + 1;
            if (outH <= 0 || outW <= 0) return Data2D();

            Data2D out(outH, outW);
            for (int y = 0; y < outH; ++y)
                for (int x = 0; x < outW; ++x)
                    out.getMatrix()[y][x] = padded[y][x].real();
            return out;
        }
        else
        {
            throw std::invalid_argument("buildOutputFromPadded: unknown outputMode");
        }
    }
};

#endif // FFT_CONVOLVER_HPP
