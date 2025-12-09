// main_compare_cuda_vs_fft.cpp
#include "../Data2D.hpp"
#include "../FftConvolution.hpp"
#include "NaiveConvolution2D_CUDA.hpp"

#include <cuda_runtime.h>
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
static const string IMAGE_FOLDER        = "../test/images/double";
static const string KERNEL_FOLDER       = "../test/kernels/double";
static const string RESULTS_CSV_FILE    = "convolution_results_gpu_naive_cpu_fft.csv";

static const string OUTPUT_MODE   = "valid";  // "full" | "same" | "valid"
static const bool   USE_NEXT_POW2 = true;
// -------------------------------------------------

static bool isCudaAvailable()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) return false;
    return deviceCount > 0;
}

int main()
{
    if (OUTPUT_MODE != "full" && OUTPUT_MODE != "same" && OUTPUT_MODE != "valid") {
        cerr << "CONFIG ERROR: OUTPUT_MODE must be 'full','same' or 'valid'\n";
        return 1;
    }

    bool cudaPresent = isCudaAvailable();
    if (!cudaPresent) {
        cerr << "WARNING: No CUDA device detected. CUDA naive convolution will be skipped.\n";
    }

    cout << fixed << setprecision(6);
    cout << "=== Fixed-size Batch Convolution Comparison (Naive CUDA vs FFT CPU) ===\n\n";

    cout << "Image folder:  " << IMAGE_FOLDER << "\n";
    cout << "Kernel folder: " << KERNEL_FOLDER << "\n";
    cout << "Output mode:   " << OUTPUT_MODE
         << "   padToNextPow2=" << (USE_NEXT_POW2 ? "true" : "false") << "\n\n";

    const int W1 = 14, W2 = 12, W3 = 8, W4 = 6, W5 = 14, W6 = 14, W7 = 12, W8 = 16;

    cout << left
         << setw(W1) << "Image"
         << setw(W2) << "Kernel"
         << setw(W3) << "Mode"
         << setw(W4) << "Pow2"
         << setw(W5) << "NaiveCUDA(ms)"
         << setw(W6) << "FFTCPU(ms)"
         << setw(W7) << "Faster"
         << setw(W8) << "AbsDiffSum"
         << "\n";

    cout << string(W1+W2+W3+W4+W5+W6+W7+W8, '-') << "\n";

    map<int, vector<int>> imageKernelMap = {
        {256,  {3,   7,  21,  51, 101}},
        {512,  {5,  11,  31, 101, 151}},
        {1024, {11, 31, 101, 201, 301}},
        {2048, {21, 51, 151, 301, 501}}
    };

    ofstream csvOut(RESULTS_CSV_FILE);
    if (!csvOut.is_open()) {
        cerr << "ERROR: could not open CSV file '" << RESULTS_CSV_FILE << "' for writing\n";
        return 1;
    }

    csvOut << "Image,Kernel,Mode,Pow2,NaiveCUDA(ms),FFTCPU(ms),Faster,AbsDiffSum\n";

    for (const auto &entry : imageKernelMap)
    {
        int imgSize = entry.first;
        const vector<int> &kernelsForImage = entry.second;

        string imageLabelBase = to_string(imgSize) + "x" + to_string(imgSize);

        for (int kerSize : kernelsForImage)
        {
            if (kerSize >= imgSize) {
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

                string csvErrMsg = errMsg;
                for (char &c : csvErrMsg) if (c == ',') c = ';';

                csvOut << imageLabel << ","
                       << kernelLabel << ","
                       << OUTPUT_MODE << ","
                       << (USE_NEXT_POW2 ? "1" : "0") << ","
                       << "ERR,ERR,ERR," << csvErrMsg << "\n";
                continue;
            }

            double naiveCudaMs = -1.0;
            Data2D<T> naiveCudaResult;
            if (cudaPresent) {
                try {
                    NaiveConvolution2D_CUDA<T> cudaConv(image, kernel);
                    auto cudaRes = cudaConv.computeConvolution(OUTPUT_MODE);
                    naiveCudaResult = std::move(cudaRes.result);
                    naiveCudaMs = cudaRes.elapsedMilliseconds;
                } catch (const exception &e) {
                    // if CUDA failed, mark as unavailable for this test
                    naiveCudaMs = -1.0;
                }
            }

            // Run FFT conv on CPU
            FftConvolver<T>::ConvolutionResult fftResult;
            bool fftOk = true;
            try {
                fftResult = FftConvolver<T>::convolve(
                    image,
                    kernel,
                    OUTPUT_MODE,
                    FftConvolver<T>::OperationMode::Correlation,
                    USE_NEXT_POW2
                );
            } catch (const exception &e) {
                fftOk = false;
            }

            // Compute absolute sum difference if both produced valid outputs and dims match
            string absDiffStr = "dim-mismatch";
            bool computedAbs = false;
            if (cudaPresent && naiveCudaMs >= 0.0 && fftOk) {
                try {
                    T absSum = Data2D<T>::sumAbsoluteDifference(naiveCudaResult, fftResult.resultMatrix);
                    ostringstream ss; ss << fixed << setprecision(6) << absSum;
                    absDiffStr = ss.str();
                    computedAbs = true;
                } catch (...) {
                    absDiffStr = "dim-mismatch";
                }
            } else if (!cudaPresent && fftOk) {
                absDiffStr = "no-cuda";
            } else {
                absDiffStr = "err";
            }

            // Determine faster
            string faster = "Tie";
            if (!cudaPresent) {
                faster = "FFT";
            } else if (naiveCudaMs < 0.0) {
                faster = "FFT";
            } else {
                if (fftOk && fftResult.elapsedMilliseconds < naiveCudaMs) faster = "FFT";
                else if (fftOk && fftResult.elapsedMilliseconds > naiveCudaMs) faster = "NaiveCUDA";
                else if (fftOk) faster = "Tie";
                else faster = "NaiveCUDA";
            }

            // Console output
            cout << left
                 << setw(W1) << imageLabel
                 << setw(W2) << kernelLabel
                 << setw(W3) << OUTPUT_MODE
                 << setw(W4) << (USE_NEXT_POW2 ? "1":"0");

            if (cudaPresent) {
                if (naiveCudaMs >= 0.0) cout << setw(W5) << naiveCudaMs;
                else cout << setw(W5) << "ERR";
            } else {
                cout << setw(W5) << "NOCUDA";
            }

            if (fftOk) cout << setw(W6) << fftResult.elapsedMilliseconds;
            else cout << setw(W6) << "ERR";

            cout << setw(W7) << faster
                 << setw(W8) << absDiffStr
                 << "\n";

            // CSV formatting
            auto formatDouble = [&](double v) -> string {
                if (v < 0.0) return string("ERR");
                ostringstream os; os << fixed << setprecision(6) << v; return os.str();
            };

            string naiveCsv = cudaPresent ? (naiveCudaMs >= 0.0 ? formatDouble(naiveCudaMs) : string("ERR")) : string("NOCUDA");
            string fftCsv = fftOk ? formatDouble(fftResult.elapsedMilliseconds) : string("ERR");

            csvOut << imageLabel << ","
                   << kernelLabel << ","
                   << OUTPUT_MODE << ","
                   << (USE_NEXT_POW2 ? "1" : "0") << ","
                   << naiveCsv << ","
                   << fftCsv << ","
                   << faster << ","
                   << absDiffStr << "\n";

        } // end kernels loop
    } // end map loop

    csvOut.close();
    cout << "\nCSV results written to '" << RESULTS_CSV_FILE << "'\n";
    cout << "\n=== Fixed-size batch comparison complete ===\n";
    return 0;
}
