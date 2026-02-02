#pragma once
#include <fstream>

class InferenceStatsWriter
{
    std::ofstream f;
public:
    InferenceStatsWriter(const std::string& p)
        : f(p) { f << "id,gt,pred,time_ms\n"; }

    void write(int i, int g, int p, double t)
    {
        f << i << "," << g << "," << p << "," << t << "\n";
    }
};
