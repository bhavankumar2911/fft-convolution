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
#include <map>
#include <fstream>
#include <sstream>

using namespace std;

// choose float or double here
using T = double;

// ----------------- Configuration -----------------
static const string IMAGE_FOLDER        = "./test/images/double";
static const string KERNEL_FOLDER       = "./test/kernels/double";
static const string RESULTS_CSV_FILE    = "convolution_results_single_double.csv";

static const string OUTPUT_MODE   = "valid";  // "full" | "same" | "valid"
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

    // Header formatting for console
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

    // Map: image size -> list of kernel sizes (user-provided experiment set)
    map<int, vector<int>> imageKernelMap = {
        {256,  {3,   7,  21,  51, 101}},
        {512,  {5,  11,  31, 101, 151}},
        {1024, {11, 31, 101, 201, 301}},
        {2048, {21, 51, 151, 301, 501}}
    };

    // Open CSV file and write header
    ofstream csvOut(RESULTS_CSV_FILE);
    if (!csvOut.is_open()) {
        cerr << "ERROR: could not open CSV file '" << RESULTS_CSV_FILE << "' for writing\n";
        return 1;
    }

    // CSV header
    csvOut << "Image,Kernel,Mode,Pow2,Naive(ms),FFT(ms),Faster,AbsDiffSum\n";

    // Iterate map and run comparisons for each valid combination
    for (const auto &entry : imageKernelMap)
    {
        int imgSize = entry.first;
        const vector<int> &kernelsForImage = entry.second;

        string imageLabelBase = to_string(imgSize) + "x" + to_string(imgSize);

        for (int kerSize : kernelsForImage)
        {
            // sanity check (skip invalid combinations where kernel >= image)
            if (kerSize >= imgSize) {
                // Console print
                cout << left
                     << setw(W1) << imageLabelBase
                     << setw(W2) << (to_string(kerSize) + "x" + to_string(kerSize))
                     << setw(W3) << OUTPUT_MODE
                     << setw(W4) << (USE_NEXT_POW2 ? "1":"0")
                     << setw(W5) << "N/A"
                     << setw(W6) << "N/A"
                     << setw(W7) << "N/A"
                     << setw(W8) << "kernel>=image"
                     << "\n";

                // CSV write
                csvOut << imageLabelBase << ","
                       << to_string(kerSize) + "x" + to_string(kerSize) << ","
                       << OUTPUT_MODE << ","
                       << (USE_NEXT_POW2 ? "1" : "0") << ","
                       << "N/A,N/A,N/A,kernel>=image\n";
                continue;
            }

            string imageLabel  = imageLabelBase;
            string kernelLabel = to_string(kerSize) + "x" + to_string(kerSize);

            Data2D<T> image;
            Data2D<T> kernel;
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

                // CSV write error row
                string csvErrMsg = errMsg;
                for (char &c : csvErrMsg) if (c == ',') c = ';';

                csvOut << imageLabel << ","
                       << kernelLabel << ","
                       << OUTPUT_MODE << ","
                       << (USE_NEXT_POW2 ? "1" : "0") << ","
                       << "ERR,ERR,ERR," << csvErrMsg << "\n";
                continue;
            }

            // Run naive conv (templated NaiveConvolution2D<T>)
            NaiveConvolution2D<T> naiveConv(image, kernel);
            auto naiveResult = naiveConv.computeConvolution(OUTPUT_MODE);

            // Run FFT conv (templated FftConvolver<T>)
            auto fftResult = FftConvolver<T>::convolve(
                image,
                kernel,
                OUTPUT_MODE,
                FftConvolver<T>::OperationMode::Correlation,
                USE_NEXT_POW2
            );

            // Sum absolute difference using templated Data2D<T>
            T absSum = static_cast<T>(0);
            bool sumOk = true;
            try {
                absSum = Data2D<T>::sumAbsoluteDifference(
                    naiveResult.result,
                    fftResult.resultMatrix
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

            // Console print
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

            // CSV row: ensure formatting matches console (fixed, 6 decimals)
            ostringstream cellFormatter;
            cellFormatter << fixed << setprecision(6);

            string naiveStr, fftStr, absSumStr;
            cellFormatter.str(""); cellFormatter.clear();
            cellFormatter << naiveResult.elapsedMilliseconds;
            naiveStr = cellFormatter.str();

            cellFormatter.str(""); cellFormatter.clear();
            cellFormatter << fftResult.elapsedMilliseconds;
            fftStr = cellFormatter.str();

            if (sumOk) {
                cellFormatter.str(""); cellFormatter.clear();
                cellFormatter << absSum;
                absSumStr = cellFormatter.str();
            } else {
                absSumStr = "dim-mismatch";
            }

            // Write CSV (no extra quoting; values are simple)
            csvOut << imageLabel << ","
                   << kernelLabel << ","
                   << OUTPUT_MODE << ","
                   << (USE_NEXT_POW2 ? "1" : "0") << ","
                   << naiveStr << ","
                   << fftStr << ","
                   << faster << ","
                   << absSumStr << "\n";

            if (PRINT_DIFF)
            {
                cout << "\nAbsolute difference matrix (" << imageLabel << " vs " << kernelLabel << "):\n";
                Data2D<T>::printAbsoluteDifferenceMatrix(naiveResult.result, fftResult.resultMatrix);
                cout << "\n";
            }
        } // end kernels loop
    } // end map loop

    csvOut.close();
    cout << "\nCSV results written to '" << RESULTS_CSV_FILE << "'\n";
    cout << "\n=== Fixed-size batch comparison complete ===\n";
    return 0;
}
