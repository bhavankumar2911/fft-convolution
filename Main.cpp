#include "Data2D.hpp"
#include "NaiveConvolution.hpp"
#include "FftTransform.hpp"
#include "FftConvolution.hpp"

#include <iostream>
#include <string>
#include <iomanip>

using namespace std;

void printUsageAndExit(const char* prog) {
    cout << "Usage: " << prog << " <imageBin> <imageH> <imageW> "
         << "<kernelBin> <kernelH> <kernelW> "
         << "<outputMode> <useNextPow2> <printDiff>\n\n"
         << "  <outputMode>     : full | same | valid\n"
         << "  <useNextPow2>    : 0 or 1\n"
         << "  <printDiff>      : 0 or 1\n";
    exit(1);
}

int main(int argc, char* argv[])
{
    if (argc < 10) printUsageAndExit(argv[0]);

    string imageBin  = argv[1];
    int imageH       = stoi(argv[2]);
    int imageW       = stoi(argv[3]);

    string kernelBin = argv[4];
    int kernelH      = stoi(argv[5]);
    int kernelW      = stoi(argv[6]);

    string outputMode = argv[7];
    bool useNextPow2  = (stoi(argv[8]) != 0);
    bool printDiff    = (stoi(argv[9]) != 0);

    if (outputMode != "full" && outputMode != "same" && outputMode != "valid") {
        cerr << "outputMode must be one of: full, same, valid\n";
        return 1;
    }

    // Load matrices
    Data2D image;
    Data2D kernel;
    image.loadFromBinary(imageBin, imageH, imageW);
    kernel.loadFromBinary(kernelBin, kernelH, kernelW);

    cout << fixed << setprecision(6);
    cout << "=== Convolution Comparison (Naive vs FFT) ===\n\n";

    cout << "Image:  " << imageBin 
         << " (" << imageH << "x" << imageW << ")\n";
    cout << "Kernel: " << kernelBin 
         << " (" << kernelH << "x" << kernelW << ")\n";
    cout << "Output mode: " << outputMode 
         << "   padToNextPow2: " << (useNextPow2 ? "true" : "false") << "\n\n";

    // Run naive convolution
    NaiveConvolution2D naiveConv(image.getMatrix(), kernel.getMatrix());
    auto naiveResult = naiveConv.computeConvolution(outputMode);

    // Run FFT convolution (own implementation)
    auto fftResult = FftConvolver::convolve(
        image,
        kernel,
        outputMode,
        FftConvolver::OperationMode::Correlation,  // cross-correlation (matching naive)
        useNextPow2
    );

    // Print timings
    cout << "Execution Times:\n";
    cout << "  Naive: " << naiveResult.elapsedMilliseconds << " ms\n";
    cout << "  FFT  : " << fftResult.elapsedMilliseconds << " ms\n\n";

    // Determine which is faster
    double naiveMs = naiveResult.elapsedMilliseconds;
    double fftMs   = fftResult.elapsedMilliseconds;

    if (naiveMs > 0 && fftMs > 0) {
        string winner = (fftMs < naiveMs) ? "FFT" : "Naive";
        double fasterBy = fabs(naiveMs - fftMs);
        double percent  = (winner == "FFT")
                            ? ((naiveMs - fftMs) / naiveMs * 100.0)
                            : ((fftMs - naiveMs) / fftMs * 100.0);

        cout << winner << " is faster by "
             << fasterBy << " ms  (" << percent << "%)\n\n";
    }

    // Print correctness via Data2D utility
    cout << "=== Numerical Difference (Naive vs FFT) ===\n";

    if (printDiff) {
        cout << "Absolute difference matrix:\n";
        Data2D::printAbsoluteDifference(
            naiveResult.resultMatrix,
            fftResult.resultMatrix.getMatrix()
        );
    } else {
        cout << "(Difference matrix not printed. Use <printDiff>=1 to enable.)\n\n";
    }

    double totalAbsDiff = Data2D::sumAbsoluteDifference(
        naiveResult.resultMatrix,
        fftResult.resultMatrix.getMatrix()
    );

    cout << "Total absolute difference = " << totalAbsDiff << "\n";

    cout << "\n=== END ===\n";
    return 0;
}
