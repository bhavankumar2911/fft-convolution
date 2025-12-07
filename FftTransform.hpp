#ifndef FFT2DTRANSFORMER_H
#define FFT2DTRANSFORMER_H

#include "Data2D.hpp"
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <algorithm>

using Complex = std::complex<double>;
using ComplexMatrix = std::vector<std::vector<Complex>>;

class Fft1D
{
public:
    static void transform(std::vector<Complex> &data, bool forward)
    {
        int n = static_cast<int>(data.size());
        if (n <= 1) return;

        int log2n = 0;
        while ((1 << log2n) < n) ++log2n;

        for (int i = 0, j = 0; i < n; ++i) {
            if (i < j) std::swap(data[i], data[j]);
            int bit = n >> 1;
            while (j & bit) { j ^= bit; bit >>= 1; }
            j ^= bit;
        }

        for (int s = 1; s <= log2n; ++s) {
            int m = 1 << s;
            double angleSign = forward ? -1.0 : 1.0;
            double theta = angleSign * 2.0 * M_PI / m;
            Complex wm(std::cos(theta), std::sin(theta));
            for (int k = 0; k < n; k += m) {
                Complex w(1.0, 0.0);
                int half = m >> 1;
                for (int j = 0; j < half; ++j) {
                    Complex t = w * data[k + j + half];
                    Complex u = data[k + j];
                    data[k + j] = u + t;
                    data[k + j + half] = u - t;
                    w *= wm;
                }
            }
        }

        if (!forward) {
            for (int i = 0; i < n; ++i) data[i] /= static_cast<double>(n);
        }
    }
};

class Fft2DTransformer
{
private:
    static int nextPowerOfTwoInt(int n)
    {
        if (n <= 1) return 1;
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }

public:
    // print absolute difference between original real matrix and an inverse FFT result
    static void printDifferenceMatrix(const Data2D &original,
                                      const ComplexMatrix &inverse,
                                      int maxRows = -1,
                                      int maxCols = -1)
    {
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6);

        int h = original.getHeight();
        int w = original.getWidth();

        if (maxRows < 0 || maxRows > h) maxRows = h;
        if (maxCols < 0 || maxCols > w) maxCols = w;

        const auto &origMat = original.getMatrix();

        for (int y = 0; y < maxRows; ++y) {
            for (int x = 0; x < maxCols; ++x) {
                double invReal = inverse[y][x].real();
                double diff = std::abs(origMat[y][x] - invReal);
                std::cout << diff << "\t";
            }
            std::cout << "\n";
        }
    }

    // General-purpose builder:
    // - minHeight/minWidth: the minimum required size (e.g. A_h + B_h - 1 for convolution correctness)
    // - padToNextPowerOfTwo: if true, after ensuring the minimum size pad up to next power of two
    static ComplexMatrix buildPaddedComplexMatrix(const Data2D &source,
                                                  int minHeight,
                                                  int minWidth,
                                                  bool padToNextPowerOfTwo)
    {
        int sourceH = source.getHeight();
        int sourceW = source.getWidth();
        if (sourceH <= 0 || sourceW <= 0)
            throw std::runtime_error("Source has non-positive dimensions");

        int targetH = std::max(minHeight, sourceH);
        int targetW = std::max(minWidth, sourceW);

        if (padToNextPowerOfTwo) {
            targetH = nextPowerOfTwoInt(targetH);
            targetW = nextPowerOfTwoInt(targetW);

            cout << endl << "target height = " << targetH << endl << "target width = " << targetW;
        }

        ComplexMatrix out(targetH, std::vector<Complex>(targetW, Complex(0.0, 0.0)));

        const auto &mat = source.getMatrix();
        for (int y = 0; y < sourceH; ++y)
            for (int x = 0; x < sourceW; ++x)
                out[y][x] = Complex(mat[y][x], 0.0);

        return out;
    }

    // compute2D that allows specifying minHeight/minWidth for correctness-first padding,
    // and optionally pads further to next power of two for FFT efficiency.
    static ComplexMatrix compute2D(const Data2D &source,
                                   int minHeight,
                                   int minWidth,
                                   bool forward,
                                   bool padToNextPowerOfTwo)
    {
        ComplexMatrix buffer = buildPaddedComplexMatrix(source, minHeight, minWidth, padToNextPowerOfTwo);
        int paddedH = static_cast<int>(buffer.size());
        int paddedW = static_cast<int>(buffer[0].size());

        std::vector<Complex> row(paddedW);
        for (int y = 0; y < paddedH; ++y) {
            for (int x = 0; x < paddedW; ++x) row[x] = buffer[y][x];
            Fft1D::transform(row, forward);
            for (int x = 0; x < paddedW; ++x) buffer[y][x] = row[x];
        }

        std::vector<Complex> col(paddedH);
        for (int x = 0; x < paddedW; ++x) {
            for (int y = 0; y < paddedH; ++y) col[y] = buffer[y][x];
            Fft1D::transform(col, forward);
            for (int y = 0; y < paddedH; ++y) buffer[y][x] = col[y];
        }

        return buffer;
    }

    static void inverse2DInPlace(ComplexMatrix &paddedComplex)
    {
        int paddedH = static_cast<int>(paddedComplex.size());
        if (paddedH == 0) return;
        int paddedW = static_cast<int>(paddedComplex[0].size());

        std::vector<Complex> col(paddedH);
        for (int x = 0; x < paddedW; ++x) {
            for (int y = 0; y < paddedH; ++y) col[y] = paddedComplex[y][x];
            Fft1D::transform(col, false);
            for (int y = 0; y < paddedH; ++y) paddedComplex[y][x] = col[y];
        }

        std::vector<Complex> row(paddedW);
        for (int y = 0; y < paddedH; ++y) {
            for (int x = 0; x < paddedW; ++x) row[x] = paddedComplex[y][x];
            Fft1D::transform(row, false);
            for (int x = 0; x < paddedW; ++x) paddedComplex[y][x] = row[x];
        }
    }
};

static void printComplexMatrixSample(const ComplexMatrix &mat,
                                     int maxRows = -1,
                                     int maxCols = -1)
{
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6);

    int totalRows = static_cast<int>(mat.size());
    if (totalRows == 0) return;

    int totalCols = static_cast<int>(mat[0].size());

    if (maxRows < 0 || maxRows > totalRows) maxRows = totalRows;
    if (maxCols < 0 || maxCols > totalCols) maxCols = totalCols;

    for (int y = 0; y < maxRows; ++y) {
        for (int x = 0; x < maxCols; ++x) {
            const Complex &z = mat[y][x];
            std::cout << "(" << z.real() << "," << z.imag() << ")\t";
        }
        std::cout << "\n";
    }
}

#endif // FFT2DTRANSFORMER_H
