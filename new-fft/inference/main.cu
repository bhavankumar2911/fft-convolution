#include "STL10CNN_Inference.hpp"
#include "NpyImageLoader.hpp"
#include "LabelExtractor.hpp"
#include "InferenceStatsWriter.hpp"
#include "WeightLoader.hpp"

#include <filesystem>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>

void writeInferenceSummary(
    const std::string& path,
    int samples,
    int correct,
    double totalTimeMs
)
{
    std::ofstream file(path);
    if (!file)
        throw std::runtime_error("Failed to open inference summary file");

    double accuracy = 100.0 * correct / samples;
    double avgTime = totalTimeMs / samples;

    file << "STL10 CNN Inference Statistics (only CUDA FFT Conv)\n";
    file << "-----------------------------\n";
    file << "Samples           : " << samples << "\n";
    file << "Accuracy (%)      : " << accuracy << "\n";
    file << "Total Time (ms)   : " << totalTimeMs << "\n";
    file << "Avg / Image (ms)  : " << avgTime << "\n";
}

int main()
{
    std::cout << "==============================\n";
    std::cout << "STL10 FFT Inference Started\n";
    std::cout << "==============================\n\n";
    std::cout << std::flush;

    /* =========================================================
       Load convolution weights (nn.Sequential indices: 0,3,6,9)
       ========================================================= */

    auto w1 = WeightLoader::load(
        "../../python/weights_bin/features_0_weight.bin",
        32 * 3 * 5 * 5
    );
    auto b1 = WeightLoader::load(
        "../../python/weights_bin/features_0_bias.bin",
        32
    );

    auto w2 = WeightLoader::load(
        "../../python/weights_bin/features_3_weight.bin",
        64 * 32 * 5 * 5
    );
    auto b2 = WeightLoader::load(
        "../../python/weights_bin/features_3_bias.bin",
        64
    );

    auto w3 = WeightLoader::load(
        "../../python/weights_bin/features_6_weight.bin",
        128 * 64 * 3 * 3
    );
    auto b3 = WeightLoader::load(
        "../../python/weights_bin/features_6_bias.bin",
        128
    );

    auto w4 = WeightLoader::load(
        "../../python/weights_bin/features_9_weight.bin",
        256 * 128 * 3 * 3
    );
    auto b4 = WeightLoader::load(
        "../../python/weights_bin/features_9_bias.bin",
        256
    );

    /* =========================================================
       Load classifier weights
       ========================================================= */

    auto wfc1 = WeightLoader::load(
        "../../python/weights_bin/classifier_0_weight.bin",
        512 * 256 * 6 * 6
    );
    auto bfc1 = WeightLoader::load(
        "../../python/weights_bin/classifier_0_bias.bin",
        512
    );

    auto wfc2 = WeightLoader::load(
        "../../python/weights_bin/classifier_2_weight.bin",
        10 * 512
    );
    auto bfc2 = WeightLoader::load(
        "../../python/weights_bin/classifier_2_bias.bin",
        10
    );

    std::cout << "Weights loaded successfully\n\n";
    std::cout << std::flush;

    /* =========================================================
       Build network
       ========================================================= */

    Conv2D_FFT c1(3,   32, 5, w1, b1);
    Conv2D_FFT c2(32,  64, 5, w2, b2);
    Conv2D_FFT c3(64, 128, 3, w3, b3);
    Conv2D_FFT c4(128,256, 3, w4, b4);

    LinearLayer f1(256 * 6 * 6, 512, wfc1, bfc1);
    LinearLayer f2(512, 10, wfc2, bfc2);

    STL10CNN_Inference model(
        std::move(c1),
        std::move(c2),
        std::move(c3),
        std::move(c4),
        std::move(f1),
        std::move(f2)
    );

    std::cout << "Model constructed\n\n";
    std::cout << std::flush;

    /* =========================================================
       Count test images
       ========================================================= */

    const std::string testDir = "../../python/test_images";

    int totalImages = 0;
    for (const auto& _ : std::filesystem::directory_iterator(testDir))
        totalImages++;

    std::cout << "Found " << totalImages << " test images\n\n";
    std::cout << std::flush;

    /* =========================================================
       Inference loop (WITH VISIBILITY)
       ========================================================= */

    InferenceStatsWriter csv("inference_stats.csv");

    int idx = 0;
    int correct = 0;
    double totalTimeMs = 0.0;

    for (const auto& entry :
         std::filesystem::directory_iterator(testDir))
    {
        const std::string path = entry.path().string();

        std::cout << "[IMAGE " << (idx + 1)
                  << " / " << totalImages << "] loading\n";
        std::cout << std::flush;

        auto input = NpyImageLoader::load(path);
        int gt = LabelExtractor::from(path);

        std::cout << "  running inference...\n";
        std::cout << std::flush;

        auto t0 = std::chrono::high_resolution_clock::now();
        auto output = model.forward(input);
        auto t1 = std::chrono::high_resolution_clock::now();

        double timeMs =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        int pred =
            std::max_element(output.begin(), output.end()) - output.begin();

        csv.write(idx, gt, pred, timeMs);

        if (pred == gt)
            correct++;

        totalTimeMs += timeMs;

        std::cout << "  done in " << timeMs << " ms"
                  << " | pred=" << pred
                  << " | gt=" << gt << "\n\n";
        std::cout << std::flush;

        idx++;
    }

    /* =========================================================
       Summary
       ========================================================= */

    std::cout << "==============================\n";
    std::cout << "Inference completed\n";
    std::cout << "Accuracy: "
              << (100.0 * correct / totalImages) << " %\n";
    std::cout << "Average inference time: "
              << (totalTimeMs / totalImages) << " ms\n";
    std::cout << "CSV written to inference_stats.csv\n";
    std::cout << "==============================\n";

    writeInferenceSummary(
        "inference_summary.txt",
        totalImages,
        correct,
        totalTimeMs
    );


    return 0;
}
