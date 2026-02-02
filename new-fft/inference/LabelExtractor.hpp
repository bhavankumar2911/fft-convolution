#pragma once
#include <string>
#include <cstdlib>

class LabelExtractor
{
public:
    static int from(const std::string& s)
    {
        auto p = s.find("_label_");
        return std::atoi(s.c_str() + p + 7);
    }
};
