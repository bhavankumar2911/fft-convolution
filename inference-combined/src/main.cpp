#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <string>

#include "data/BinaryTensorLoader.hpp"
#include "cpu/NaiveCPUConvolution2D.hpp"
#include "model/STL10CNNModel.hpp"

int main() {
    using Real = float;   // FP32 default (matches PyTorch)

    // -------------------------------------------------
    // Ensure output directories exist
    // -------------------------------------------------
    std::filesystem::create_directories("../results/naive");

    // -------------------------------------------------
    // Build convolution layers
    // -------------------------------------------------
    std::vector<std::unique_ptr<IConvolution2D<Real>>> convLayers;

    convLayers.push_back(std::make_unique<NaiveCPUConvolution2D<Real>>(
        3, 32, 5, 2,
        "../trained_weights/features_0_weight.bin",
        "../trained_weights/features_0_bias.bin"
    ));
    convLayers.push_back(std::make_unique<NaiveCPUConvolution2D<Real>>(
        32, 64, 5, 2,
        "../trained_weights/features_3_weight.bin",
        "../trained_weights/features_3_bias.bin"
    ));
    convLayers.push_back(std::make_unique<NaiveCPUConvolution2D<Real>>(
        64, 128, 3, 1,
        "../trained_weights/features_6_weight.bin",
        "../trained_weights/features_6_bias.bin"
    ));
    convLayers.push_back(std::make_unique<NaiveCPUConvolution2D<Real>>(
        128, 256, 3, 1,
        "../trained_weights/features_9_weight.bin",
        "../trained_weights/features_9_bias.bin"
    ));

    FeatureExtractor<Real> extractor(std::move(convLayers));

    STL10CNNModel<Real> model(
        std::move(extractor),
        Linear<Real>(
            256 * 6 * 6, 512,
            "../trained_weights/classifier_0_weight.bin",
            "../trained_weights/classifier_0_bias.bin"
        ),
        Linear<Real>(
            512, 10,
            "../trained_weights/classifier_2_weight.bin",
            "../trained_weights/classifier_2_bias.bin"
        )
    );

    // -------------------------------------------------
    // Output files
    // -------------------------------------------------
    std::ofstream csv("../results/naive/layerwise_timing.csv");
    std::ofstream summary("../results/naive/summary.txt");

    if (!csv || !summary) {
        std::cerr << "ERROR: Failed to open output files\n";
        return 1;
    }

    csv << "image_name,"
        << "conv1_ms,conv2_ms,conv3_ms,conv4_ms,"
        << "relu_ms,pool_ms,fc_ms,"
        << "total_conv_ms,total_infer_ms\n";

    // -------------------------------------------------
    // Stats
    // -------------------------------------------------
    int totalSamples = 0;
    int correct = 0;

    double totalInferenceMs = 0.0;
    double totalConvMs = 0.0;

    // Count total images (for progress)
    int totalImages = 0;
    for (const auto& e : std::filesystem::directory_iterator("../test_images_bin"))
        if (e.path().extension() == ".bin")
            totalImages++;

    // -------------------------------------------------
    // Inference loop
    // -------------------------------------------------
    int imageIndex = 0;

    for (const auto& entry :
         std::filesystem::directory_iterator("../test_images_bin")) {

        if (entry.path().extension() != ".bin")
            continue;

        imageIndex++;

        const std::string imagePath = entry.path().string();
        const std::string imageName = entry.path().filename().string();

        // Load + normalize (matches PyTorch exactly)
        Tensor<Real> input =
            BinaryTensorLoader::loadImageCHW<Real>(
                imagePath, 3, 96, 96
            );

        auto inferStart = std::chrono::high_resolution_clock::now();
        std::vector<Real> logits = model.forward(input);
        auto inferEnd = std::chrono::high_resolution_clock::now();

        double inferMs =
            std::chrono::duration<double, std::milli>(
                inferEnd - inferStart
            ).count();

        totalInferenceMs += inferMs;

        // -----------------------------
        // Convolution timing
        // -----------------------------
        double imageConvMs = 0.0;
        csv << imageName;

        for (const auto& conv : model.featureExtractor().convs()) {
            double t = conv->lastExecutionTimeMs();
            csv << "," << t;
            imageConvMs += t;
        }

        totalConvMs += imageConvMs;

        // -----------------------------
        // Other layers timing
        // -----------------------------
        double reluMs = model.featureExtractor().reluTimeMs();
        double poolMs = model.featureExtractor().poolTimeMs();
        double fcMs   = model.fcTimeMs();

        csv << "," << reluMs
            << "," << poolMs
            << "," << fcMs
            << "," << imageConvMs
            << "," << inferMs
            << "\n";

        // -----------------------------
        // Prediction / accuracy
        // -----------------------------
        int predicted = 0;
        Real maxVal = logits[0];
        for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                predicted = i;
            }
        }

        // filename format: image_XXX_label_Y.bin
        std::size_t pos = imageName.find("_label_");
        int trueLabel = std::stoi(
            imageName.substr(
                pos + 7,
                imageName.find(".bin") - (pos + 7)
            )
        );

        bool ok = (predicted == trueLabel);
        if (ok) correct++;

        totalSamples++;

        // -----------------------------
        // Progress print
        // -----------------------------
        std::cout << "[" << imageIndex << "/" << totalImages << "] "
                  << imageName
                  << " | pred=" << predicted
                  << " | true=" << trueLabel
                  << " | " << (ok ? "OK" : "WRONG")
                  << std::endl;
    }

    // -------------------------------------------------
    // Summary
    // -------------------------------------------------
    double avgInferMs = totalInferenceMs / totalSamples;
    double avgConvMs  = totalConvMs / totalSamples;
    double accuracy   = 100.0 * correct / totalSamples;

    summary << "STL10 Inference Summary\n";
    summary << "----------------------\n";
    summary << "Samples                     : " << totalSamples << "\n";
    summary << "Accuracy (%)                : " << accuracy << "\n\n";

    summary << "Total inference time (ms)   : " << totalInferenceMs << "\n";
    summary << "Avg inference / image (ms)  : " << avgInferMs << "\n\n";

    summary << "Total convolution time (ms) : " << totalConvMs << "\n";
    summary << "Avg convolution / image(ms): " << avgConvMs << "\n";

    csv.close();
    summary.close();

    std::cout << "\nInference complete.\n";
    std::cout << "Results written to:\n";
    std::cout << "  ../results/naive/layerwise_timing.csv\n";
    std::cout << "  ../results/naive/summary.txt\n";

    return 0;
}
