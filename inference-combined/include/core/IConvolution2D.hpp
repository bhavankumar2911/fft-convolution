#pragma once
#include "Tensor.hpp"

template<typename T>
class IConvolution2D {
public:
    virtual Tensor<T> forward(const Tensor<T>& input) = 0;
    virtual double lastExecutionTimeMs() const = 0;
    virtual ~IConvolution2D() = default;
};
