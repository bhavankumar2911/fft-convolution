#pragma once
#include "core/Tensor.hpp"
#include <algorithm>

template<typename T>
class ActivationReLU {
public:
    static void apply(Tensor<T>& t) {
        for (T& v : t.data)
            v = std::max(T(0), v);
    }
};
