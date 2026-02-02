#pragma once
#include "core/Tensor.hpp"
#include <algorithm>

template<typename T>
class MaxPool2D {
public:
    static Tensor<T> apply(const Tensor<T>& input) {
        Tensor<T> out(
            input.channels,
            input.height / 2,
            input.width / 2
        );

        for (int c = 0; c < input.channels; ++c)
            for (int y = 0; y < out.height; ++y)
                for (int x = 0; x < out.width; ++x) {

                    T m = T(-1e9);
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            int iy = y * 2 + dy;
                            int ix = x * 2 + dx;
                            m = std::max(
                                m,
                                input.data[
                                    c * input.height * input.width +
                                    iy * input.width + ix
                                ]
                            );
                        }

                    out.data[
                        c * out.height * out.width +
                        y * out.width + x
                    ] = m;
                }

        return out;
    }
};
