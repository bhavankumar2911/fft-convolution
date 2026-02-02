#pragma once
#include <vector>

template<typename T>
class Tensor {
public:
    int channels;
    int height;
    int width;
    std::vector<T> data;

    Tensor(int c, int h, int w)
        : channels(c), height(h), width(w),
          data(c * h * w, static_cast<T>(0)) {}
};
