#ifndef FFTTRANSFORM_HPP
#define FFTTRANSFORM_HPP

#include "Data2D.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <type_traits>

template<typename T>
struct ComplexArray
{
    using ComplexT = std::complex<T>;
    int height = 0;
    int width = 0;
    std::vector<ComplexT> data; // row-major

    ComplexArray() = default;
    ComplexArray(int h, int w) : height(h), width(w), data(static_cast<size_t>(h) * w, ComplexT(static_cast<T>(0), static_cast<T>(0))) {}

    inline ComplexT& operator()(int y, int x) { return data[static_cast<size_t>(y) * width + x]; }
    inline const ComplexT& operator()(int y, int x) const { return data[static_cast<size_t>(y) * width + x]; }

    int sizeH() const { return height; }
    int sizeW() const { return width; }
};

template<typename T>
class Fft1D
{
    static_assert(std::is_floating_point<T>::value, "Fft1D requires floating point type");
public:
    using ComplexT = std::complex<T>;

    static void transform(std::vector<ComplexT> &buf, bool forward)
    {
        int n = static_cast<int>(buf.size());
        if (n <= 1) return;
        int log2n = 0; while ((1 << log2n) < n) ++log2n;

        for (int i = 0, j = 0; i < n; ++i) {
            if (i < j) std::swap(buf[i], buf[j]);
            int bit = n >> 1;
            while (j & bit) { j ^= bit; bit >>= 1; }
            j ^= bit;
        }

        for (int s = 1; s <= log2n; ++s) {
            int m = 1 << s;
            T angleSign = forward ? static_cast<T>(-1) : static_cast<T>(1);
            T theta = angleSign * static_cast<T>(2) * static_cast<T>(M_PI) / static_cast<T>(m);
            ComplexT wm(std::cos(theta), std::sin(theta));
            for (int k = 0; k < n; k += m) {
                ComplexT w(static_cast<T>(1), static_cast<T>(0));
                int half = m >> 1;
                for (int j = 0; j < half; ++j) {
                    ComplexT t = w * buf[k + j + half];
                    ComplexT u = buf[k + j];
                    buf[k + j] = u + t;
                    buf[k + j + half] = u - t;
                    w *= wm;
                }
            }
        }

        if (!forward) {
            T inv = static_cast<T>(1) / static_cast<T>(n);
            for (int i = 0; i < n; ++i) buf[i] *= inv;
        }
    }
};

template<typename T>
class Fft2DTransformer
{
    static_assert(std::is_floating_point<T>::value, "Fft2DTransformer requires floating point type");

private:
    static int nextPowerOfTwo(int n)
    {
        if (n <= 1) return 1;
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

public:
    using ComplexT = std::complex<T>;
    using ComplexArr = ComplexArray<T>;
    using Fft1DType = Fft1D<T>;

    static ComplexArr buildPaddedComplexMatrix(const Data2D<T> &src, int minH, int minW, bool padToNextPowerOfTwo)
    {
        int srcH = src.getHeight();
        int srcW = src.getWidth();
        if (srcH <= 0 || srcW <= 0) throw std::runtime_error("Source has non-positive dimensions");

        int targetH = std::max(minH, srcH);
        int targetW = std::max(minW, srcW);
        if (padToNextPowerOfTwo) { targetH = nextPowerOfTwo(targetH); targetW = nextPowerOfTwo(targetW); }

        ComplexArr out(targetH, targetW);
        for (int y = 0; y < srcH; ++y)
            for (int x = 0; x < srcW; ++x)
                out(y, x) = ComplexT(src(y, x), static_cast<T>(0));
        return out;
    }

    static ComplexArr compute2D(const Data2D<T> &src, int minH, int minW, bool forward, bool padToNextPowerOfTwo)
    {
        ComplexArr mat = buildPaddedComplexMatrix(src, minH, minW, padToNextPowerOfTwo);
        int H = mat.sizeH();
        int W = mat.sizeW();

        std::vector<ComplexT> line;
        line.resize(W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) line[x] = mat(y, x);
            Fft1DType::transform(line, forward);
            for (int x = 0; x < W; ++x) mat(y, x) = line[x];
        }

        line.resize(H);
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) line[y] = mat(y, x);
            Fft1DType::transform(line, forward);
            for (int y = 0; y < H; ++y) mat(y, x) = line[y];
        }

        return mat;
    }

    static void inverse2DInPlace(ComplexArr &mat)
    {
        int H = mat.sizeH();
        if (H == 0) return;
        int W = mat.sizeW();
        std::vector<ComplexT> line;
        line.resize(H);
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) line[y] = mat(y, x);
            Fft1DType::transform(line, false);
            for (int y = 0; y < H; ++y) mat(y, x) = line[y];
        }
        line.resize(W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) line[x] = mat(y, x);
            Fft1DType::transform(line, false);
            for (int x = 0; x < W; ++x) mat(y, x) = line[x];
        }
    }

    static void printDifferenceMatrix(const Data2D<T> &original, const ComplexArr &inverse, int maxRows = -1, int maxCols = -1)
    {
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6);
        int h = original.getHeight();
        int w = original.getWidth();
        if (maxRows < 0 || maxRows > h) maxRows = h;
        if (maxCols < 0 || maxCols > w) maxCols = w;
        for (int y = 0; y < maxRows; ++y)
        {
            for (int x = 0; x < maxCols; ++x)
            {
                T invReal = inverse(y, x).real();
                T diff = static_cast<T>(std::abs(original(y, x) - invReal));
                std::cout << diff << "\t";
            }
            std::cout << "\n";
        }
    }
};

#endif // FFTTRANSFORM_HPP
