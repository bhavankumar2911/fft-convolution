#pragma once
#include <vector>

class LinearLayer
{
    std::size_t inF, outF;
    std::vector<double> w, b;

public:
    LinearLayer(
        std::size_t i, std::size_t o,
        const std::vector<double>& W,
        const std::vector<double>& B
    ) : inF(i), outF(o), w(W), b(B) {}

    std::vector<double> forward(
        const std::vector<double>& x
    ) const
    {
        std::vector<double> y(outF, 0.0);
        for (std::size_t o = 0; o < outF; ++o)
        {
            double s = b[o];
            for (std::size_t i = 0; i < inF; ++i)
                s += x[i] * w[o * inF + i];
            y[o] = s;
        }
        return y;
    }
};
