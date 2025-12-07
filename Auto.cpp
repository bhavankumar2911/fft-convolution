// main_compare_no_args.cpp
#include "Data2D.hpp"
#include "NaiveConvolution.hpp"
#include "FftTransform.hpp"
#include "FftConvolution.hpp"

#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <exception>

using namespace std;

// ----------------- Configuration -----------------
static const string IMAGE_FOLDER   = "./test/images";
static const string KERNEL_FOLDER  = "./test/kernels";

const vector<int> IMAGE_SIZES  = {5, 20, 96, 256, 500};
const vector<int> KERNEL_SIZES = {3, 7, 9, 11, 15};

static const string OUTPUT_MODE   = "same";  // "full" | "same" | "valid"
static const bool   USE_NEXT_POW2 = true;
static const bool   PRINT_DIFF    = false;
// -------------------------------------------------

int main()
{
    if (OUTPUT_MODE != "full" && OUTPUT_MODE != "same" && OUTPUT_MODE != "valid") {
        cerr << "CONFIG ERROR: OUTPUT_MODE must be 'full','same' or 'valid'\n";
        return 1;
    }

    cout << fixed << setprecision(6);
    cout << "=== Fixed-size Batch Convolution Comparison (Naive vs FFT) ===\n\n";

    cout << "Image folder:  " << IMAGE_FOLDER << "\n";
    cout << "Kernel folder: " << KERNEL_FOLDER << "\n";
    cout << "Output mode:   " << OUTPUT_MODE
         << "   padToNextPow2=" << (USE_NEXT_POW2 ? "true" : "false") << "\n";
    cout << "Print diff?    " << (PRINT_DIFF ? "yes" : "no") << "\n\n";

    // Header formatting
    const int W1 = 14, W2 = 12, W3 = 8, W4 = 6, W5 = 12, W6 = 12, W7 = 12, W8 = 16;

    cout << left
         << setw(W1) << "Image"
         << setw(W2) << "Kernel"
         << setw(W3) << "Mode"
         << setw(W4) << "Pow2"
         << setw(W5) << "Naive(ms)"
         << setw(W6) << "FFT(ms)"
         << setw(W7) << "Faster"
         << setw(W8) << "AbsDiffSum"
         << "\n";

    cout << string(W1+W2+W3+W4+W5+W6+W7+W8, '-') << "\n";

    // Loop through valid combinations (kernel < image)
    for (int imgSize : IMAGE_SIZES)
    {
        for (int kerSize : KERNEL_SIZES)
        {
            if (kerSize >= imgSize)
                continue;  // <<--- ONLY VALID COMBINATIONS PRINTED

            string imageLabel  = to_string(imgSize) + "x" + to_string(imgSize);
            string kernelLabel = to_string(kerSize) + "x" + to_string(kerSize);

            Data2D image, kernel;
            bool ok = true;
            string errMsg;

            try {
                image.loadFromBinary(IMAGE_FOLDER, imgSize, imgSize);
            } catch (const exception &e) {
                ok = false;
                errMsg = "img-load: " + string(e.what());
            }

            if (ok) {
                try {
                    kernel.loadFromBinary(KERNEL_FOLDER, kerSize, kerSize);
                } catch (const exception &e) {
                    ok = false;
                    errMsg = "ker-load: " + string(e.what());
                }
            }

            if (!ok) {
                cout << left
                     << setw(W1) << imageLabel
                     << setw(W2) << kernelLabel
                     << setw(W3) << OUTPUT_MODE
                     << setw(W4) << (USE_NEXT_POW2 ? "1":"0")
                     << setw(W5) << "ERR"
                     << setw(W6) << "ERR"
                     << setw(W7) << "ERR"
                     << setw(W8) << errMsg
                     << "\n";
                continue;
            }

            // Run naive conv
            NaiveConvolution2D naiveConv(image.getMatrix(), kernel.getMatrix());
            auto naiveResult = naiveConv.computeConvolution(OUTPUT_MODE);

            // Run FFT conv
            auto fftResult = FftConvolver::convolve(
                image,
                kernel,
                OUTPUT_MODE,
                FftConvolver::OperationMode::Correlation,
                USE_NEXT_POW2
            );

            // Sum abs diff
            double absSum = 0.0;
            bool sumOk = true;
            try {
                absSum = Data2D::sumAbsoluteDifference(
                    naiveResult.resultMatrix,
                    fftResult.resultMatrix.getMatrix()
                );
            } catch (...) {
                sumOk = false;
            }

            // Determine which is faster
            string faster = "Tie";
            if (fftResult.elapsedMilliseconds < naiveResult.elapsedMilliseconds)
                faster = "FFT";
            else if (fftResult.elapsedMilliseconds > naiveResult.elapsedMilliseconds)
                faster = "Naive";

            // Print ONLY valid comparison rows
            cout << left
                 << setw(W1) << imageLabel
                 << setw(W2) << kernelLabel
                 << setw(W3) << OUTPUT_MODE
                 << setw(W4) << (USE_NEXT_POW2 ? "1":"0")
                 << setw(W5) << naiveResult.elapsedMilliseconds
                 << setw(W6) << fftResult.elapsedMilliseconds
                 << setw(W7) << faster
                 << setw(W8) << (sumOk ? to_string(absSum) : string("dim-mismatch"))
                 << "\n";

            if (PRINT_DIFF)
            {
                cout << "\nAbsolute difference matrix (" << imageLabel << " vs " << kernelLabel << "):\n";
                try {
                    Data2D::printAbsoluteDifference(
                        naiveResult.resultMatrix,
                        fftResult.resultMatrix.getMatrix()
                    );
                } catch (const exception &e) {
                    cout << "Could not print diff matrix: " << e.what() << "\n";
                }
                cout << "\n";
            }
        }
    }

    cout << "\n=== Fixed-size batch comparison complete ===\n";
    return 0;
}
