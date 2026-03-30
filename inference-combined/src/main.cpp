#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <chrono>
#include <string>
#include <ctime>

#include "data/BinaryTensorLoader.hpp"
#include "model/STL10CNNModel.hpp"

// -------------------------------------------------
// Compile-time datatype selection
// -------------------------------------------------
#ifdef USE_FP64
    using Real = double;
    constexpr const char* DTYPE_NAME = "fp64";
#else
    using Real = float;
    constexpr const char* DTYPE_NAME = "fp32";
#endif

// -------------------------------------------------
// Compile-time backend selection
//
// Four backends:
//   BACKEND_CPU_NAIVE  — all layers on CPU (default)
//   BACKEND_GPU_NAIVE  — naive conv + ReLU + pool on GPU
//   BACKEND_GPU_FFT    — FFT conv + ReLU + pool on GPU
//   BACKEND_GPU_HYBRID — FFT for large maps, naive for small
//                        ReLU + pool on GPU
// -------------------------------------------------
#if defined(BACKEND_GPU_FFT)
    #include "cuda/FFTConvolution2D_CUDA.hpp"
    template<typename T> using Conv2D = FFTConvolution2D_CUDA<T>;
    constexpr const char* BACKEND_NAME = "gpu_fft";

#elif defined(BACKEND_GPU_NAIVE)
    #include "cuda/NaiveCUDAConvolution2D.hpp"
    template<typename T> using Conv2D = NaiveCUDAConvolution2D<T>;
    constexpr const char* BACKEND_NAME = "gpu_naive";

#elif defined(BACKEND_GPU_HYBRID)
    #include "hybrid/HybridConvolution2D.hpp"
    template<typename T> using Conv2D = HybridConvolution2D<T>;
    constexpr const char* BACKEND_NAME = "gpu_hybrid";

#else
    // BACKEND_CPU_NAIVE (default — no macro needed)
    #include "cpu/NaiveCPUConvolution2D.hpp"
    template<typename T> using Conv2D = NaiveCPUConvolution2D<T>;
    constexpr const char* BACKEND_NAME = "cpu_naive";
#endif

int main() {
    // -------------------------------------------------
    // Output directory: results/<timestamp>/<backend>_<dtype>
    // Timestamp format: YYYYMMDD_HHMMSS
    // -------------------------------------------------
    std::time_t now = std::time(nullptr);
    char tsBuf[32];
    std::strftime(tsBuf, sizeof(tsBuf), "%Y-%m-%d-%H-%M", std::localtime(&now));

    std::string resultDir =
        std::string("../results/") + BACKEND_NAME + "_" + DTYPE_NAME + "/" + tsBuf;
    std::filesystem::create_directories(resultDir);

    // -------------------------------------------------
    // Build convolution layers
    // All four backends use identical constructor arguments —
    // the type alias Conv2D handles backend dispatch.
    // -------------------------------------------------
    std::vector<std::unique_ptr<IConvolution2D<Real>>> convLayers;

    convLayers.push_back(std::make_unique<Conv2D<Real>>(
        3, 32, 5, 2,
        "../trained_weights_fp32/features_0_weight.bin",
        "../trained_weights_fp32/features_0_bias.bin"
    ));
    convLayers.push_back(std::make_unique<Conv2D<Real>>(
        32, 64, 5, 2,
        "../trained_weights_fp32/features_3_weight.bin",
        "../trained_weights_fp32/features_3_bias.bin"
    ));
    convLayers.push_back(std::make_unique<Conv2D<Real>>(
        64, 128, 3, 1,
        "../trained_weights_fp32/features_6_weight.bin",
        "../trained_weights_fp32/features_6_bias.bin"
    ));
    convLayers.push_back(std::make_unique<Conv2D<Real>>(
        128, 256, 3, 1,
        "../trained_weights_fp32/features_9_weight.bin",
        "../trained_weights_fp32/features_9_bias.bin"
    ));

    FeatureExtractor<Real> extractor(std::move(convLayers));

    STL10CNNModel<Real> model(
        std::move(extractor),
        Linear<Real>(
            256 * 6 * 6, 512,
            "../trained_weights_fp32/classifier_0_weight.bin",
            "../trained_weights_fp32/classifier_0_bias.bin"
        ),
        Linear<Real>(
            512, 10,
            "../trained_weights_fp32/classifier_2_weight.bin",
            "../trained_weights_fp32/classifier_2_bias.bin"
        )
    );

    // -------------------------------------------------
    // Output files
    // -------------------------------------------------
    std::ofstream csv(resultDir + "/layerwise_timing.csv");
    std::ofstream summary(resultDir + "/summary.txt");

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
    int    totalSamples     = 0;
    int    correct          = 0;
    double totalInferenceMs = 0.0;
    double totalConvMs      = 0.0;

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

        Tensor<Real> input =
            BinaryTensorLoader::loadImageCHW<Real>(imagePath, 3, 96, 96);

        auto inferStart = std::chrono::high_resolution_clock::now();
        std::vector<Real> logits = model.forward(input);
        auto inferEnd   = std::chrono::high_resolution_clock::now();

        double inferMs =
            std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
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
        csv << "," << model.featureExtractor().reluTimeMs()
            << "," << model.featureExtractor().poolTimeMs()
            << "," << model.fcTimeMs()
            << "," << imageConvMs
            << "," << inferMs
            << "\n";

        // -----------------------------
        // Accuracy
        // -----------------------------
        int  predicted = 0;
        Real maxVal    = logits[0];
        for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
            if (logits[i] > maxVal) {
                maxVal    = logits[i];
                predicted = i;
            }
        }

        std::size_t pos = imageName.find("_label_");
        int trueLabel = std::stoi(
            imageName.substr(pos + 7, imageName.find(".bin") - (pos + 7))
        );

        bool ok = (predicted == trueLabel);
        if (ok) correct++;
        totalSamples++;

        // -----------------------------
        // Progress
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
    summary << "STL10 Inference Summary\n";
    summary << "----------------------\n";
    summary << "Backend                     : " << BACKEND_NAME << "\n";
    summary << "Datatype                    : " << DTYPE_NAME   << "\n";
    summary << "Samples                     : " << totalSamples << "\n";
    summary << "Accuracy (%)                : "
            << (100.0 * correct / totalSamples) << "\n\n";

    summary << "Total inference time (ms)   : " << totalInferenceMs << "\n";
    summary << "Avg inference / image (ms)  : "
            << (totalInferenceMs / totalSamples) << "\n\n";

    summary << "Total convolution time (ms) : " << totalConvMs << "\n";
    summary << "Avg convolution / image(ms) : "
            << (totalConvMs / totalSamples) << "\n";

    csv.close();
    summary.close();

    std::cout << "\nInference complete.\n";
    std::cout << "Backend : " << BACKEND_NAME << "\n";
    std::cout << "Datatype: " << DTYPE_NAME   << "\n";
    std::cout << "Results : " << resultDir    << "\n";

    return 0;
}