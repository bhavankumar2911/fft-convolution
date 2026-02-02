#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

class WeightLoader
{
public:
    static std::vector<double> load(
        const std::string& path,
        std::size_t expected
    )
    {
        std::vector<double> data(expected);
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("Weight open failed");

        f.read(reinterpret_cast<char*>(data.data()),
               expected * sizeof(double));

        if (!f) throw std::runtime_error("Weight read failed");
        return data;
    }
};
