#pragma once
#include <torch/torch.h>

class STL10CNNImpl : public torch::nn::Module
{
public:
    STL10CNNImpl()
    {
        features = torch::nn::Sequential(
            torch::nn::Conv2d({3, 32, 5, 1, 2}),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(true),
            torch::nn::MaxPool2d(2),

            torch::nn::Conv2d({32, 64, 5, 1, 2}),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(true),
            torch::nn::MaxPool2d(2),

            torch::nn::Conv2d({64, 128, 3, 1, 1}),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(true),
            torch::nn::MaxPool2d(2),

            torch::nn::Conv2d({128, 256, 3, 1, 1}),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(true),
            torch::nn::MaxPool2d(2)
        );

        classifier = torch::nn::Sequential(
            torch::nn::Linear(256 * 6 * 6, 512),
            torch::nn::ReLU(true),
            torch::nn::Dropout(0.5),
            torch::nn::Linear(512, 10)
        );

        register_module("features", features);
        register_module("classifier", classifier);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = features->forward(x);
        x = x.flatten(1);
        return classifier->forward(x);
    }

private:
    torch::nn::Sequential features{nullptr};
    torch::nn::Sequential classifier{nullptr};
};

TORCH_MODULE(STL10CNN);
