#ifndef MATRIX2D_HPP
#define MATRIX2D_HPP

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <type_traits>

template<typename T>
class Matrix2D
{
    static_assert(std::is_floating_point<T>::value,
                  "Matrix2D requires float or double");

private:
    std::size_t height;
    std::size_t width;
    std::vector<T> buffer;

public:
    Matrix2D(std::size_t rows, std::size_t cols)
        : height(rows), width(cols), buffer(rows * cols) {}

    std::size_t rows() const { return height; }
    std::size_t cols() const { return width; }

    T& operator()(std::size_t y, std::size_t x)
    {
        return buffer[y * width + x];
    }

    const T& operator()(std::size_t y, std::size_t x) const
    {
        return buffer[y * width + x];
    }

    T* data()
    {
        return buffer.data();
    }

    const T* data() const
    {
        return buffer.data();
    }

    void readFromBinaryFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open file for reading: " + path);

        file.read(reinterpret_cast<char*>(buffer.data()),
                  buffer.size() * sizeof(T));

        if (!file)
            throw std::runtime_error("Binary read failed: " + path);
    }

    void writeToBinaryFile(const std::string& path) const
    {
        std::ofstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open file for writing: " + path);

        file.write(reinterpret_cast<const char*>(buffer.data()),
                   buffer.size() * sizeof(T));

        if (!file)
            throw std::runtime_error("Binary write failed: " + path);
    }
};

#endif
