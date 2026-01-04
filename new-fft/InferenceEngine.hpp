#pragma once
#include <torch/torch.h>
#include <cnpy.h>
#include <string>

class InferenceEngine
{
public:
    InferenceEngine(const std::string& modelPath)
    {
        model = STL10CNN();
        torch::load(model, modelPath);
        model->eval();
    }

    torch::Tensor loadImage(const std::string& path)
    {
        cnpy::NpyArray array = cnpy::npy_load(path);
        float* data = array.data<float>();

        auto tensor = torch::from_blob(
            data,
            {1, 3, 96, 96},
            torch::kFloat32
        ).clone();

        const double mean[3] = {0.4467, 0.4398, 0.4066};
        const double std[3]  = {0.2241, 0.2215, 0.2239};

        for (int c = 0; c < 3; ++c)
            tensor[0][c] = tensor[0][c].sub(mean[c]).div(std[c]);

        return tensor;
    }

    int infer(const torch::Tensor& input)
    {
        torch::NoGradGuard noGrad;
        auto output = model->forward(input);
        return output.argmax(1).item<int>();
    }

private:
    STL10CNN model;
};
