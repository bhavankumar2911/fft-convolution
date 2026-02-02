// main_compare_cpu_gpu_double.cpp
#include "../Data2D.hpp"
#include "../NaiveConvolution.hpp"
#include "NaiveConvolution2D_CUDA.hpp" // include CUDA TU so nvcc compiles it together

#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <exception>
#include <map>
#include <fstream>
#include <sstream>

using namespace std;
using T = double; // use double precision

// ----------------- Configuration -----------------
static const string IMAGE_FOLDER        = "../test/images/double";
static const string KERNEL_FOLDER       = "../test/kernels/double";
static const string RESULTS_CSV_FILE    = "convolution_results_cpu_vs_gpu_double.csv";

static const string OUTPUT_MODE   = "valid";  // "full" | "same" | "valid"
static const bool   PRINT_DIFF    = false;
// -------------------------------------------------

int main()
{
    if (OUTPUT_MODE != "full" && OUTPUT_MODE != "same" && OUTPUT_MODE != "valid") {
        cerr << "CONFIG ERROR: OUTPUT_MODE must be 'full','same' or 'valid'\n";
        return 1;
    }

    cout << fixed << setprecision(6);
    cout << "=== Fixed-size Batch Convolution Comparison (CPU Naive vs GPU Naive) ===\n\n";

    cout << "Image folder:  " << IMAGE_FOLDER << "\n";
    cout << "Kernel folder: " << KERNEL_FOLDER << "\n";
    cout << "Output mode:   " << OUTPUT_MODE << "\n\n";

    const int W1 = 14, W2 = 12, W3 = 8, W4 = 12, W5 = 12, W6 = 12, W7 = 16;

    cout << left
         << setw(W1) << "Image"
         << setw(W2) << "Kernel"
         << setw(W3) << "Mode"
         << setw(W4) << "NaiveCPU(ms)"
         << setw(W5) << "NaiveGPU_kern(ms)"
         << setw(W6) << "NaiveGPU_total(ms)"
         << setw(W7) << "AbsDiffSum"
         << "\n";

    cout << string(W1+W2+W3+W4+W5+W6+W7, '-') << "\n";

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
    csvOut << "Image,Kernel,Mode,NaiveCPU_ms,NaiveGPU_kernel_ms,NaiveGPU_total_ms,AbsDiffSum\n";

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
                     << setw(W4) << "N/A"
                     << setw(W5) << "N/A"
                     << setw(W6) << "N/A"
                     << setw(W7) << "kernel>=image"
                     << "\n";

                csvOut << imageLabelBase << ","
                       << to_string(kerSize) + "x" + to_string(kerSize) << ","
                       << OUTPUT_MODE << ","
                       << "N/A,N/A,N/A,kernel>=image\n";
                continue;
            }

            string imageLabel  = imageLabelBase;
            string kernelLabel = to_string(kerSize) + "x" + to_string(kerSize);

            Data2D<T> image, kernel;
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
                     << setw(W4) << "ERR"
                     << setw(W5) << "ERR"
                     << setw(W6) << "ERR"
                     << setw(W7) << errMsg
                     << "\n";

                string csvErrMsg = errMsg;
                for (char &c : csvErrMsg) if (c == ',') c = ';';

                csvOut << imageLabel << ","
                       << kernelLabel << ","
                       << OUTPUT_MODE << ","
                       << "ERR,ERR,ERR," << csvErrMsg << "\n";
                continue;
            }

            NaiveConvolution2D<T> cpuConv(image, kernel);

            auto t0 = chrono::high_resolution_clock::now();
            auto cpuRes = cpuConv.computeConvolution(OUTPUT_MODE);
            auto t1 = chrono::high_resolution_clock::now();
            double cpuMs = chrono::duration<double, milli>(t1 - t0).count();

            // GPU conv: instantiate GPU class and measure total host time (H2D + kernel + D2H)
            NaiveConvolution2D_CUDA<T> gpuConv(image, kernel);
            auto tg0 = chrono::high_resolution_clock::now();
            auto gpuRes = gpuConv.computeConvolution(OUTPUT_MODE); // returns kernel-only time in elapsedMilliseconds
            auto tg1 = chrono::high_resolution_clock::now();
            double gpuTotalMs = chrono::duration<double, milli>(tg1 - tg0).count();
            double gpuKernelMs = gpuRes.elapsedMilliseconds;

            // compute absolute sum difference if dims match
            T absSum = static_cast<T>(0);
            bool sumOk = true;
            try {
                absSum = Data2D<T>::sumAbsoluteDifference(cpuRes.result, gpuRes.result);
            } catch (...) {
                sumOk = false;
            }

            cout << left
                 << setw(W1) << imageLabel
                 << setw(W2) << kernelLabel
                 << setw(W3) << OUTPUT_MODE
                 << setw(W4) << cpuMs
                 << setw(W5) << gpuKernelMs
                 << setw(W6) << gpuTotalMs
                 << setw(W7) << (sumOk ? to_string(absSum) : string("dim-mismatch"))
                 << "\n";

            ostringstream cellFmt;
            cellFmt << fixed << setprecision(6);

            string cpuStr, gKerStr, gTotStr, absStr;
            cellFmt.str(""); cellFmt.clear(); cellFmt << cpuMs; cpuStr = cellFmt.str();
            cellFmt.str(""); cellFmt.clear(); cellFmt << gpuKernelMs; gKerStr = cellFmt.str();
            cellFmt.str(""); cellFmt.clear(); cellFmt << gpuTotalMs; gTotStr = cellFmt.str();
            if (sumOk) { cellFmt.str(""); cellFmt.clear(); cellFmt << absSum; absStr = cellFmt.str(); }
            else absStr = "dim-mismatch";

            csvOut << imageLabel << ","
                   << kernelLabel << ","
                   << OUTPUT_MODE << ","
                   << cpuStr << ","
                   << gKerStr << ","
                   << gTotStr << ","
                   << absStr << "\n";

            if (PRINT_DIFF && sumOk) {
                cout << "\nAbsolute difference matrix (" << imageLabel << " vs " << kernelLabel << "):\n";
                Data2D<T>::printAbsoluteDifferenceMatrix(cpuRes.result, gpuRes.result);
                cout << "\n";
            }
        } // kernels
    } // images

    csvOut.close();
    cout << "\nCSV results written to '" << RESULTS_CSV_FILE << "'\n";
    cout << "\n=== Done ===\n";
    return 0;
}
