#ifndef NORMALIZATION_HPP
#define NORMALIZATION_HPP

#include "../Matrix2D.hpp"
#include <vector>

class Normalization
{
public:
    static void apply(
        std::vector<Matrix2D<double>>& channels
    )
    {
        static const double mean[3] = {
            0.4467, 0.4398, 0.4066
        };
        static const double std[3] = {
            0.2241, 0.2215, 0.2239
        };

        for (std::size_t c = 0; c < 3; ++c)
            for (std::size_t y = 0; y < channels[c].rows(); ++y)
                for (std::size_t x = 0; x < channels[c].cols(); ++x)
                    channels[c](y, x) =
                        (channels[c](y, x) - mean[c]) / std[c];
    }
};

#endif
