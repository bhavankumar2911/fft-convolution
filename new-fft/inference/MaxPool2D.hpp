#ifndef MAXPOOL2D_HPP
#define MAXPOOL2D_HPP

#include "../Matrix2D.hpp"
#include <algorithm>

class MaxPool2D
{
public:
    static Matrix2D<double> apply(const Matrix2D<double>& input)
    {
        std::size_t outH = input.rows() / 2;
        std::size_t outW = input.cols() / 2;

        Matrix2D<double> output(outH, outW);

        for (std::size_t y = 0; y < outH; ++y)
            for (std::size_t x = 0; x < outW; ++x)
            {
                double m = input(2*y, 2*x);
                m = std::max(m, input(2*y, 2*x + 1));
                m = std::max(m, input(2*y + 1, 2*x));
                m = std::max(m, input(2*y + 1, 2*x + 1));
                output(y, x) = m;
            }

        return output;
    }
};

#endif
