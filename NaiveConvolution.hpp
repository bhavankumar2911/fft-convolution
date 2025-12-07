// #ifndef NAIVE_CONVOLUTION_HPP
// #define NAIVE_CONVOLUTION_HPP

// #include <vector>
// #include <chrono>
// using namespace std;

// class NaiveConvolution2D
// {
// public:
//     struct ConvolutionResult
//     {
//         vector<vector<double>> resultMatrix;
//         double elapsedMilliseconds;
//     };

// private:
//     const vector<vector<double>>& imageMatrix;
//     const vector<vector<double>>& kernelMatrix;

//     int imageHeight;
//     int imageWidth;
//     int kernelHeight;
//     int kernelWidth;

// public:
//     NaiveConvolution2D(const vector<vector<double>>& image,
//                        const vector<vector<double>>& kernel)
//         : imageMatrix(image),
//           kernelMatrix(kernel)
//     {
//         imageHeight  = imageMatrix.size();
//         imageWidth   = imageHeight ? imageMatrix[0].size() : 0;
//         kernelHeight = kernelMatrix.size();
//         kernelWidth  = kernelHeight ? kernelMatrix[0].size() : 0;
//     }

//     ConvolutionResult computeConvolution()
//     {
//         auto tStart = chrono::high_resolution_clock::now();

//         int outputHeight = imageHeight - kernelHeight + 1;
//         int outputWidth  = imageWidth  - kernelWidth  + 1;

//         if (outputHeight <= 0 || outputWidth <= 0)
//             return ConvolutionResult{ {}, 0.0 };

//         vector<vector<double>> result(outputHeight,
//                                       vector<double>(outputWidth, 0.0));

//         for (int row = 0; row < outputHeight; ++row)
//             for (int col = 0; col < outputWidth; ++col)
//             {
//                 double sum = 0.0;
//                 for (int kr = 0; kr < kernelHeight; ++kr)
//                     for (int kc = 0; kc < kernelWidth; ++kc)
//                         sum += imageMatrix[row + kr][col + kc]
//                              * kernelMatrix[kr][kc];
//                 result[row][col] = sum;
//             }

//         auto tEnd = chrono::high_resolution_clock::now();
//         double elapsedMs =
//             chrono::duration<double, milli>(tEnd - tStart).count();

//         return ConvolutionResult{ result, elapsedMs };
//     }
// };

// #endif

#ifndef NAIVE_CONVOLUTION_HPP
#define NAIVE_CONVOLUTION_HPP

#include <vector>
#include <chrono>
#include <string>
#include <stdexcept>
#include <cmath>
using namespace std;

class NaiveConvolution2D
{
public:
    struct ConvolutionResult
    {
        vector<vector<double>> resultMatrix;
        double elapsedMilliseconds;
    };

private:
    const vector<vector<double>>& imageMatrix;
    const vector<vector<double>>& kernelMatrix;

    int imageHeight;
    int imageWidth;
    int kernelHeight;
    int kernelWidth;

    // safe accessor returning 0 for out-of-bounds (zero padding)
    inline double getImageValueOrZero(int y, int x) const
    {
        if (y < 0 || x < 0 || y >= imageHeight || x >= imageWidth) return 0.0;
        return imageMatrix[y][x];
    }

public:
    NaiveConvolution2D(const vector<vector<double>>& image,
                       const vector<vector<double>>& kernel)
        : imageMatrix(image),
          kernelMatrix(kernel)
    {
        imageHeight  = static_cast<int>(imageMatrix.size());
        imageWidth   = imageHeight ? static_cast<int>(imageMatrix[0].size()) : 0;
        kernelHeight = static_cast<int>(kernelMatrix.size());
        kernelWidth  = kernelHeight ? static_cast<int>(kernelMatrix[0].size()) : 0;
        if (imageHeight < 0 || imageWidth < 0 || kernelHeight < 0 || kernelWidth < 0)
            throw invalid_argument("Negative dimensions are not allowed");
    }

    // outputMode: "valid" | "same" | "full"
    ConvolutionResult computeConvolution(const string &outputMode = "valid")
    {
        auto tStart = chrono::high_resolution_clock::now();

        if (kernelHeight <= 0 || kernelWidth <= 0 || imageHeight <= 0 || imageWidth <= 0)
            return ConvolutionResult{ {}, 0.0 };

        if (outputMode != "valid" && outputMode != "same" && outputMode != "full")
            throw invalid_argument("outputMode must be 'valid', 'same' or 'full'");

        if (outputMode == "valid")
        {
            int outputHeight = imageHeight - kernelHeight + 1;
            int outputWidth  = imageWidth  - kernelWidth  + 1;

            if (outputHeight <= 0 || outputWidth <= 0)
                return ConvolutionResult{ {}, 0.0 };

            vector<vector<double>> result(outputHeight, vector<double>(outputWidth, 0.0));

            for (int row = 0; row < outputHeight; ++row)
                for (int col = 0; col < outputWidth; ++col)
                {
                    double sum = 0.0;
                    for (int kr = 0; kr < kernelHeight; ++kr)
                        for (int kc = 0; kc < kernelWidth; ++kc)
                            sum += imageMatrix[row + kr][col + kc] * kernelMatrix[kr][kc];
                    result[row][col] = sum;
                }

            auto tEnd = chrono::high_resolution_clock::now();
            double elapsedMs = chrono::duration<double, milli>(tEnd - tStart).count();
            return ConvolutionResult{ result, elapsedMs };
        }
        else if (outputMode == "full")
        {
            int outH = imageHeight + kernelHeight - 1;
            int outW = imageWidth  + kernelWidth  - 1;

            vector<vector<double>> result(outH, vector<double>(outW, 0.0));

            int padY = kernelHeight - 1;
            int padX = kernelWidth  - 1;

            for (int y = 0; y < outH; ++y)
            {
                for (int x = 0; x < outW; ++x)
                {
                    double s = 0.0;

                    for (int kr = 0; kr < kernelHeight; ++kr)
                    {
                        int iy = y + kr - padY;  // correct alignment
                        if (iy < 0 || iy >= imageHeight) continue;

                        for (int kc = 0; kc < kernelWidth; ++kc)
                        {
                            int ix = x + kc - padX;
                            if (ix < 0 || ix >= imageWidth) continue;

                            s += imageMatrix[iy][ix] * kernelMatrix[kr][kc];
                        }
                    }
                    result[y][x] = s;
                }
            }

            auto tEnd = chrono::high_resolution_clock::now();
            double elapsedMs = chrono::duration<double, milli>(tEnd - tStart).count();
            return ConvolutionResult{ result, elapsedMs };
        }
        else // outputMode == "same"
        {
            // "same" produces output same size as image, kernel is centered over each pixel
            int outH = imageHeight;
            int outW = imageWidth;
            vector<vector<double>> result(outH, vector<double>(outW, 0.0));

            int padTop = kernelHeight / 2;   // floor center
            int padLeft = kernelWidth  / 2;

            for (int y = 0; y < outH; ++y)
            {
                for (int x = 0; x < outW; ++x)
                {
                    double s = 0.0;
                    // kernel coordinates mapped to image coords:
                    // image_y = y + kr - padTop  => kr = image_y - y + padTop
                    for (int kr = 0; kr < kernelHeight; ++kr)
                    {
                        int iy = y + kr - padTop;
                        for (int kc = 0; kc < kernelWidth; ++kc)
                        {
                            int ix = x + kc - padLeft;
                            double iv = getImageValueOrZero(iy, ix);
                            if (iv == 0.0 && (iy < 0 || ix < 0 || iy >= imageHeight || ix >= imageWidth))
                                continue; // out-of-bounds treated as zero
                            s += iv * kernelMatrix[kr][kc];
                        }
                    }
                    result[y][x] = s;
                }
            }

            auto tEnd = chrono::high_resolution_clock::now();
            double elapsedMs = chrono::duration<double, milli>(tEnd - tStart).count();
            return ConvolutionResult{ result, elapsedMs };
        }
    }
};

#endif
