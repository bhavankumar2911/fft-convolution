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

    // cout << "\narmadillo conv matrix\n";
    // Data2D::printMatrix(armaMatToVector2D(armadilloConvResult));

    // cout << "\narmadillo flipped conv matrix\n";
    // Data2D::printMatrix(armaMatToVector2D(flippedArmadilloConvResult));

    // cout << "Naive Convolution time: "
    //      << naiveResult.elapsedMilliseconds << " ms\n";

    // cout << "FFT Convolution time: "
    //      << fftResult.elapsedMilliseconds << " ms\n";

    // cout << naiveResult.resultMatrix.size() << endl
    //      << naiveResult.resultMatrix[0].size() << endl
    //      << fftResult.resultMatrix.getHeight() << endl
    //      << fftResult.resultMatrix.getWidth() << endl;

    cout << "checking correctness of naive covn" << endl;
    cout << "\n diff between naive conv vs flipped armadillo conv\n";
    Data2D::printAbsoluteDifference(
        armaMatToVector2D(flippedArmadilloConvResult),
        naiveResult.resultMatrix
    );

    // cout << "checking correctness of fft covn" << endl;
    
    // cout << "\n diff between fft conv vs fftw conv (correlation)\n";
    auto fftResult = FftConvolver::convolve(
        image,
        kernel,
        fftOutputCropMode,
        FftConvolver::OperationMode::Correlation,
        useNextPowerOf2
    );
    auto fftwResult = fft_conv2d_full(image.getMatrix(), kernel.getMatrix(), true, useNextPowerOf2);
    // cout << fftResult.resultMatrix.getMatrix().size() << endl << fftResult.resultMatrix.getMatrix()[0].size() << endl
    //     << fftwResult.size() << endl << fftwResult[0].size() << endl;
    // Data2D::printAbsoluteDifference(
    //     fftResult.resultMatrix.getMatrix(),
    //     fftwResult
    // );

    // cout << "\n diff between fft conv vs fftw conv (convolution)\n";
    // auto fftResultConv = FftConvolver::convolve(
    //     image,
    //     kernel,
    //     fftOutputCropMode,
    //     FftConvolver::OperationMode::Convolution,
    //     useNextPowerOf2
    // );
    // auto fftwResultConv = fft_conv2d_full(image.getMatrix(), kernel.getMatrix(), false, useNextPowerOf2);
    // cout << fftResultConv.resultMatrix.getMatrix().size() << endl << fftResultConv.resultMatrix.getMatrix()[0].size() << endl
    // << fftwResultConv.size() << endl << fftwResultConv[0].size() << endl;
    // Data2D::printAbsoluteDifference(
    //     fftResultConv.resultMatrix.getMatrix(),
    //     fftwResultConv
    // );

    cout << "checking correctness marmadillo conv and fft conv (between own fft and armadillo implementation)" << endl;
    Data2D::printAbsoluteDifference(
        // naiveResult.resultMatrix,
        armaMatToVector2D(flippedArmadilloConvResult),
        fftResult.resultMatrix.getMatrix()
    );

    cout << "checking correctness naive and fft conv (between own implementation)" << endl;
    Data2D::printAbsoluteDifference(
        naiveResult.resultMatrix,
        // armaMatToVector2D(flippedArmadilloConvResult),
        fftResult.resultMatrix.getMatrix()
    );

    return 0;
}
