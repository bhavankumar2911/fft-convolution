#include "Matrix2D.hpp"
#include "FFTCrossCorrelation2D.hpp"
#include "PaddingMode.hpp"

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

PaddingMode parsePaddingMode(const std::string& mode)
{
    if (mode == "same")  return PaddingMode::SAME;
    if (mode == "valid") return PaddingMode::VALID;
    if (mode == "full")  return PaddingMode::FULL;
    throw std::runtime_error("Unknown padding mode: " + mode);
}

int main(int argc, char** argv)
{
    std::string inputDir   = "../python/data64";
    std::string outputDir  = "./fft_outputs64";
    std::string dtypeStr   = "float64";
    std::string paddingStr = "same";
    std::string csvPath    = "./cpu_fft_results.csv";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--input_dir") inputDir = argv[++i];
        else if (arg == "--output_dir") outputDir = argv[++i];
        else if (arg == "--dtype") dtypeStr = argv[++i];
        else if (arg == "--padding_mode") paddingStr = argv[++i];
        else if (arg == "--csv_path") csvPath = argv[++i];
    }

    PaddingMode paddingMode = parsePaddingMode(paddingStr);

    std::map<int, std::vector<int>> imageKernelSizeMap = {
        {256,  {3, 7, 21, 51, 101}},
        {512,  {5, 11, 31, 101, 151}},
        {1024, {11, 31, 101, 201, 301}},
        {2048, {21, 51, 151, 301, 501}}
    };

    fs::create_directories(outputDir);

    bool csvExists = fs::exists(csvPath);
    std::ofstream csvFile(csvPath, std::ios::app);

    if (!csvExists)
    {
        csvFile
            << "image_size,"
            << "kernel_size,"
            << "padding_mode,"
            << "dtype,"
            << "operation_time_ms\n";
    }

    auto run = [&](auto dummy)
    {
        using T = decltype(dummy);

        for (const auto& [imageSize, kernelSizes] : imageKernelSizeMap)
        {
            Matrix2D<T> image(imageSize, imageSize);
            image.readFromBinaryFile(
                inputDir + "/images/" +
                std::to_string(imageSize) + "x" +
                std::to_string(imageSize) + ".bin"
            );

            for (int kernelSize : kernelSizes)
            {
                Matrix2D<T> kernel(kernelSize, kernelSize);
                kernel.readFromBinaryFile(
                    inputDir + "/kernels/" +
                    std::to_string(kernelSize) + "x" +
                    std::to_string(kernelSize) + ".bin"
                );

                auto startTime =
                    std::chrono::high_resolution_clock::now();

                Matrix2D<T> output =
                    FFTCrossCorrelation2D::compute(
                        image,
                        kernel,
                        paddingMode
                    );

                auto endTime =
                    std::chrono::high_resolution_clock::now();

                double operationTimeMs =
                    std::chrono::duration<double, std::milli>(
                        endTime - startTime
                    ).count();

                fs::path outputPath =
                    fs::path(outputDir) /
                    (std::to_string(imageSize) + "x" +
                     std::to_string(imageSize) + "_" +
                     std::to_string(kernelSize) + "x" +
                     std::to_string(kernelSize) + "_" +
                     paddingStr + ".bin");

                output.writeToBinaryFile(outputPath.string());

                csvFile
                    << imageSize << ","
                    << kernelSize << ","
                    << paddingStr << ","
                    << dtypeStr << ","
                    << operationTimeMs << "\n";

                std::cout
                    << "Saved: image=" << imageSize
                    << ", kernel=" << kernelSize
                    << ", mode=" << paddingStr
                    << ", time=" << operationTimeMs << " ms"
                    << std::endl;
            }
        }
    };

    if (dtypeStr == "float32") run(float{});
    else if (dtypeStr == "float64") run(double{});
    else throw std::runtime_error("Unsupported dtype");

    csvFile.close();

    std::cout << "FFT cross-correlation timing completed\n";
    return 0;
}
