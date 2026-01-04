#include <torch/script.h>
#include <torch/torch.h>
#include <cnpy.h>

#include <filesystem>
#include <iostream>
#include <vector>
#include <chrono>

namespace fs = std::filesystem;

int extractLabel(const std::string& filename)
{
    auto pos = filename.find("_label_");
    return std::stoi(filename.substr(pos + 7));
}

int main()
{
    torch::jit::Module model =
        torch::jit::load("stl10_cnn_same_stride1_cpu.pt", torch::kCPU);
    model.eval();

    torch::Tensor mean = torch::tensor({0.4467, 0.4398, 0.4066}).view({1,3,1,1});
    torch::Tensor std  = torch::tensor({0.2241, 0.2215, 0.2239}).view({1,3,1,1});

    std::string inputDir = "../python/test_images";

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(inputDir))
        if (e.path().extension() == ".npy")
            files.push_back(e.path().string());

    std::sort(files.begin(), files.end());

    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& path : files)
    {
        cnpy::NpyArray arr = cnpy::npy_load(path);
        float* data = arr.data<float>();

        torch::Tensor image =
            torch::from_blob(data, {1,3,96,96}, torch::kFloat32).clone();

        image = (image - mean) / std;

        std::vector<torch::jit::IValue> inputs{image};
        torch::Tensor out = model.forward(inputs).toTensor();

        int pred = out.argmax(1).item<int>();
        int label = extractLabel(path);

        correct += (pred == label);
        std::cout << fs::path(path).filename()
                  << " | pred=" << pred
                  << " | label=" << label << "\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration<double>(end - start).count();

    std::cout << "\nAccuracy: " << (100.0 * correct / files.size()) << "%\n";
    std::cout << "Total time: " << t << " s\n";
    std::cout << "Avg/image: " << t / files.size() << " s\n";
}
