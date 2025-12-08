#ifndef DATA2D_HPP
#define DATA2D_HPP

#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <cmath>

template<typename T>
class Data2D
{
    static_assert(std::is_floating_point<T>::value, "Data2D requires a floating point type");

private:
    int height_;
    int width_;
    std::vector<T> storage_; // row-major: index = y*width + x

    static std::string buildFilePath(const std::string& folder, int h, int w)
    {
        std::string path = folder;
#ifdef _WIN32
        if (!path.empty() && path.back() != '\\') path += '\\';
#else
        if (!path.empty() && path.back() != '/') path += '/';
#endif
        path += std::to_string(h) + "x" + std::to_string(w) + ".bin";
        return path;
    }

public:
    Data2D(int h = 0, int w = 0)
        : height_(h), width_(w), storage_()
    {
        if (h < 0 || w < 0) throw std::invalid_argument("Dimensions must be non-negative");
        if (h > 0 && w > 0) storage_.assign(static_cast<size_t>(h) * w, static_cast<T>(0));
    }

    void resize(int h, int w)
    {
        if (h <= 0 || w <= 0) throw std::invalid_argument("Dimensions must be positive");
        height_ = h; width_ = w;
        storage_.assign(static_cast<size_t>(h) * w, static_cast<T>(0));
    }

    int getHeight() const { return height_; }
    int getWidth()  const { return width_;  }
    const std::vector<T>& data() const { return storage_; }
    std::vector<T>& data() { return storage_; }

    inline T operator()(int y, int x) const { return storage_[static_cast<size_t>(y) * width_ + x]; }
    inline T& operator()(int y, int x) { return storage_[static_cast<size_t>(y) * width_ + x]; }

    T at(int y, int x) const
    {
        if (y < 0 || x < 0 || y >= height_ || x >= width_) throw std::out_of_range("Data2D::at out of range");
        return storage_[static_cast<size_t>(y) * width_ + x];
    }

    T& atRef(int y, int x)
    {
        if (y < 0 || x < 0 || y >= height_ || x >= width_) throw std::out_of_range("Data2D::atRef out of range");
        return storage_[static_cast<size_t>(y) * width_ + x];
    }

    void fillRandom(T minVal = static_cast<T>(0), T maxVal = static_cast<T>(1))
    {
        if (height_ <= 0 || width_ <= 0) throw std::runtime_error("Matrix not initialized");
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<T> d(minVal, maxVal);
        for (T &v : storage_) v = d(gen);
    }

    void saveToBinary(const std::string& folderPath) const
    {
        if (height_ <= 0 || width_ <= 0) throw std::runtime_error("saveToBinary: invalid dims");
        std::string file = buildFilePath(folderPath, height_, width_);
        std::cout << "Saving matrix to: " << file << '\n';
        std::ofstream out(file, std::ios::binary);
        if (!out) throw std::runtime_error("Failed to open output file: " + file);
        out.write(reinterpret_cast<const char*>(&height_), sizeof(int));
        out.write(reinterpret_cast<const char*>(&width_), sizeof(int));
        out.write(reinterpret_cast<const char*>(storage_.data()), static_cast<std::streamsize>(storage_.size() * sizeof(T)));
    }

    void loadFromBinary(const std::string& folderPath, int h, int w)
    {
        if (h <= 0 || w <= 0) throw std::invalid_argument("loadFromBinary: dimensions must be positive");
        std::string file = buildFilePath(folderPath, h, w);
        std::ifstream in(file, std::ios::binary);
        if (!in) throw std::runtime_error("Failed to open input file: " + file);
        int fh = 0, fw = 0;
        in.read(reinterpret_cast<char*>(&fh), sizeof(int));
        in.read(reinterpret_cast<char*>(&fw), sizeof(int));
        if (fh != h || fw != w) throw std::runtime_error("loadFromBinary: file header dims do not match provided dims");
        height_ = h; width_ = w;
        storage_.assign(static_cast<size_t>(height_) * width_, static_cast<T>(0));
        in.read(reinterpret_cast<char*>(storage_.data()), static_cast<std::streamsize>(storage_.size() * sizeof(T)));
    }

    static T sumAbsoluteDifference(const Data2D& A, const Data2D& B)
    {
        if (A.getHeight() != B.getHeight() || A.getWidth() != B.getWidth())
            throw std::runtime_error("sumAbsoluteDifference: dimensions do not match");
        T sum = static_cast<T>(0);
        const size_t n = A.data().size();
        const T* aPtr = A.data().data();
        const T* bPtr = B.data().data();
        for (size_t i = 0; i < n; ++i) sum += std::abs(aPtr[i] - bPtr[i]);
        return sum;
    }

    // Print absolute difference between two matrices (must be same dimensions)
    static void printAbsoluteDifferenceMatrix(const Data2D& A, const Data2D& B, int maxRows = -1, int maxCols = -1)
    {
        if (A.getHeight() == 0 || B.getHeight() == 0) {
            std::cout << "Matrix is empty\n";
            return;
        }

        if (A.getHeight() != B.getHeight() || A.getWidth() != B.getWidth()) {
            std::cout << "Dimension mismatch\n";
            return;
        }

        int H = A.getHeight();
        int W = A.getWidth();

        if (maxRows < 0 || maxRows > H) maxRows = H;
        if (maxCols < 0 || maxCols > W) maxCols = W;

        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6);

        for (int y = 0; y < maxRows; ++y)
        {
            for (int x = 0; x < maxCols; ++x)
            {
                T diff = static_cast<T>(std::abs(A(y, x) - B(y, x)));
                std::cout << diff << " ";
            }
            std::cout << "\n";
        }
    }

    static void printMatrixSample(const Data2D& M, int maxRows = -1, int maxCols = -1)
    {
        if (M.getHeight() == 0 || M.getWidth() == 0) { std::cout << "Matrix is empty\n"; return; }
        if (maxRows < 0 || maxRows > M.getHeight()) maxRows = M.getHeight();
        if (maxCols < 0 || maxCols > M.getWidth())  maxCols = M.getWidth();
        std::cout.setf(std::ios::fixed);
        std::cout << std::setprecision(6);
        for (int y = 0; y < maxRows; ++y)
        {
            for (int x = 0; x < maxCols; ++x)
                std::cout << M(y, x) << " ";
            std::cout << "\n";
        }
    }
};

#endif // DATA2D_HPP
