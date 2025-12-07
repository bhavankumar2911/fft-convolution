#include "Data2D.hpp"
#include "NaiveConvolution.hpp"
#include "FftTransform.hpp"
#include "FftConvolution.hpp"
#include "fftwConv.hpp"
#include <iostream>
#include <string>
#include <armadillo>

using namespace arma;
using namespace std;

arma::mat toArma(const vector<vector<double>>& v) {
    size_t R = v.size();
    size_t C = (R > 0 ? v[0].size() : 0);
    mat M(R, C);
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            M(i,j) = v[i][j];
    return M;
}

std::vector<std::vector<double>> armaMatToVector2D(const arma::mat & M) {
    size_t rows = M.n_rows;
    size_t cols = M.n_cols;
    std::vector<std::vector<double>> out;
    out.reserve(rows);

    for (size_t i = 0; i < rows; ++i) {
        // Use arma::conv_to to convert a row to std::vector<double>
        std::vector<double> row = arma::conv_to<std::vector<double>>::from(M.row(i));
        out.push_back(std::move(row));
    }
    return out;
}

int main(int argc, char* argv[])
{
    if (argc < 9)
    {
        cout << "Usage: ./program "
             << "<imageFolder> <imageH> <imageW> "
             << "<kernelFolder> <kernelH> <kernelW> "
             << "<fftOutputCropMode> <useNextPowerOf2>\n";
        return 1;
    }

    string imageFolder = argv[1];
    int imageH = stoi(argv[2]);
    int imageW = stoi(argv[3]);

    string kernelFolder = argv[4];
    int kernelH = stoi(argv[5]);
    int kernelW = stoi(argv[6]);

    string fftOutputCropMode = argv[7];
    // bool useNextPowerOf2 = (stoi(argv[8]) != 0);
    bool useNextPowerOf2 = true;

    Data2D image;
    Data2D kernel;

    image.loadFromBinary(imageFolder, imageH, imageW);
    kernel.loadFromBinary(kernelFolder, kernelH, kernelW);

    NaiveConvolution2D naiveConv(image.getMatrix(), kernel.getMatrix());
    auto naiveResult = naiveConv.computeConvolution(fftOutputCropMode);

    // armadillo lib naive conv
    mat A = toArma(image.getMatrix());
    mat K = toArma(kernel.getMatrix());

    mat armadilloConvResult = conv2(A, K, "valid");
    mat flippedArmadilloConvResult = conv2(A, flipud(fliplr(K)), fftOutputCropMode.c_str());

    auto fftResult = FftConvolver::convolve(
        image,
        kernel,
        fftOutputCropMode,
        FftConvolver::OperationMode::Correlation,
        useNextPowerOf2
    );

    cout << "Naive Convolution time: "
         << naiveResult.elapsedMilliseconds << " ms\n";

    cout << "FFT Convolution time: "
         << fftResult.elapsedMilliseconds << " ms\n";

    cout << "checking correctness naive and fft conv (between own implementation)" << endl;
    Data2D::printAbsoluteDifference(
        naiveResult.resultMatrix,
        fftResult.resultMatrix.getMatrix()
    );

    return 0;
}
