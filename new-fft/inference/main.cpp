#include "STL10CNN_Inference.hpp"
#include "NpyImageLoader.hpp"
#include "LabelExtractor.hpp"
#include "InferenceStatsWriter.hpp"

#include <filesystem>
#include <chrono>
#include <iostream>

int main()
{
    STL10CNN_Inference model = /* constructed using weights */;

    InferenceStatsWriter csv("inference_stats.csv");

    int idx = 0;
    for (const auto& e : std::filesystem::directory_iterator("./test_images"))
    {
        auto x = NpyImageLoader::load(e.path().string());
        int gt = LabelExtractor::from(e.path().string());

        auto t0 = std::chrono::high_resolution_clock::now();
        auto y = model.forward(x);
        auto t1 = std::chrono::high_resolution_clock::now();

        int pred = std::max_element(y.begin(), y.end()) - y.begin();
        double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        csv.write(idx++, gt, pred, ms);
    }
}
