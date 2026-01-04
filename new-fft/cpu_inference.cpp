#include <filesystem>
#include <iostream>
#include "STL10CNN.hpp"
#include "InferenceEngine.hpp"

namespace fs = std::filesystem;

int extractLabel(const std::string& filename)
{
    auto pos = filename.find("_label_");
    auto end = filename.find(".npy", pos);
    return std::stoi(filename.substr(pos + 7, end - pos - 7));
}

int main()
{
    InferenceEngine engine("stl10_cnn_same_stride1_mps.pth");

    std::string dataDir = "./test_numpy";

    int correct = 0;
    int total = 0;

    for (const auto& entry : fs::directory_iterator(dataDir))
    {
        const std::string path = entry.path().string();
        if (path.find(".npy") == std::string::npos)
            continue;

        auto input = engine.loadImage(path);
        int predicted = engine.infer(input);
        int label = extractLabel(path);

        if (predicted == label)
            correct++;

        total++;
    }

    std::cout << "Accuracy: "
              << (100.0 * correct / total)
              << "% (" << correct << "/" << total << ")\n";

    return 0;
}
