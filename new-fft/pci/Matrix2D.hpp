#ifndef MATRIX2D_HPP
#define MATRIX2D_HPP

#include <cuda_runtime.h>
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
    std::size_t elementCount;
    T* buffer;

public:
    Matrix2D()
        : height(0), width(0), elementCount(0), buffer(nullptr) {}

    Matrix2D(std::size_t rows, std::size_t cols)
        : height(rows), width(cols), elementCount(rows * cols), buffer(nullptr)
    {
        cudaError_t status =
            cudaMallocHost(reinterpret_cast<void**>(&buffer),
                           elementCount * sizeof(T));

        if (status != cudaSuccess)
            throw std::runtime_error("cudaMallocHost failed");
    }

    ~Matrix2D()
    {
        if (buffer)
            cudaFreeHost(buffer);
    }

    Matrix2D(const Matrix2D&) = delete;
    Matrix2D& operator=(const Matrix2D&) = delete;

    Matrix2D(Matrix2D&& other) noexcept
        : height(other.height),
          width(other.width),
          elementCount(other.elementCount),
          buffer(other.buffer)
    {
        other.buffer = nullptr;
        other.height = other.width = other.elementCount = 0;
    }

    Matrix2D& operator=(Matrix2D&& other) noexcept
    {
        if (this != &other)
        {
            if (buffer)
                cudaFreeHost(buffer);

            height = other.height;
            width = other.width;
            elementCount = other.elementCount;
            buffer = other.buffer;

            other.buffer = nullptr;
            other.height = other.width = other.elementCount = 0;
        }
        return *this;
    }

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
        return buffer;
    }

    const T* data() const
    {
        return buffer;
    }

    std::size_t sizeInBytes() const
    {
        return elementCount * sizeof(T);
    }

    void readFromBinaryFile(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error(
                "Failed to open file for reading: " + path
            );

        file.read(reinterpret_cast<char*>(buffer),
                  sizeInBytes());

        if (!file)
            throw std::runtime_error(
                "Binary read failed: " + path
            );
    }

    void writeToBinaryFile(const std::string& path) const
    {
        std::ofstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error(
                "Failed to open file for writing: " + path
            );

        file.write(reinterpret_cast<const char*>(buffer),
                   sizeInBytes());

        if (!file)
            throw std::runtime_error(
                "Binary write failed: " + path
            );
    }
};

#endif
