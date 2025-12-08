#ifndef DATA2D_H
#define DATA2D_H

#include <vector>
#include <random>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>

using namespace std;

class Data2D
{
private:
    int matrixHeight;
    int matrixWidth;
    vector<vector<double>> matrix;

    string buildFilePath(const string& folder, int h, int w) const
    {
        string path = folder;
#ifdef _WIN32
        if (!path.empty() && path.back() != '\\') path += '\\';
#else
        if (!path.empty() && path.back() != '/') path += '/';
#endif
        path += to_string(h) + "x" + to_string(w) + ".bin";
        return path;
    }

public:
    static double sumAbsoluteDifference(
            const vector<vector<double>>& firstMatrix,
            const vector<vector<double>>& secondMatrix)
    {
        const int heightA = firstMatrix.size();
        const int heightB = secondMatrix.size();

        if (heightA == 0 || heightB == 0)
            throw runtime_error("Matrices must not be empty");

        const int widthA = firstMatrix[0].size();
        const int widthB = secondMatrix[0].size();

        if (heightA != heightB || widthA != widthB)
            throw runtime_error("Matrix dimensions do not match for sumAbsoluteDifference");

        double sum = 0.0;

        for (int y = 0; y < heightA; ++y)
            for (int x = 0; x < widthA; ++x)
                sum += std::abs(firstMatrix[y][x] - secondMatrix[y][x]);

        return sum;
    }

    static void printAbsoluteDifference(
            const vector<vector<double>>& firstMatrix,
            const vector<vector<double>>& secondMatrix)
    {
        const int heightA = firstMatrix.size();
        const int heightB = secondMatrix.size();

        if (heightA == 0 || heightB == 0)
            throw runtime_error("Matrices must not be empty");

        const int widthA = firstMatrix[0].size();
        const int widthB = secondMatrix[0].size();

        if (heightA != heightB || widthA != widthB)
            throw runtime_error("Matrix dimensions do not match for absolute difference");

        cout.setf(std::ios::fixed);
        cout << std::setprecision(6);

        for (int y = 0; y < heightA; ++y)
        {
            for (int x = 0; x < widthA; ++x)
            {
                double elementDifference = std::abs(firstMatrix[y][x] - secondMatrix[y][x]);
                cout << elementDifference << " ";
            }
            cout << '\n';
        }
    }

    static void printMatrix(const vector<vector<double>>& matrix2d)
    {
        const int height = matrix2d.size();
        if (height == 0)
        {
            cout << "Matrix is empty\n";
            return;
        }

        const int width = matrix2d[0].size();

        cout.setf(std::ios::fixed);
        cout << std::setprecision(6);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
                cout << matrix2d[y][x] << " ";
            cout << '\n';
        }
    }

    // Construct optionally empty matrix with given dims
    Data2D(int height = 0, int width = 0)
        : matrixHeight(height), matrixWidth(width), matrix()
    {
        if (height < 0 || width < 0)
            throw invalid_argument("Dimensions must be non-negative");
        if (height > 0 && width > 0)
            matrix.assign(height, vector<double>(width));
    }

    // Resize internal storage to height x width (keeps contents undefined)
    void resize(int height, int width)
    {
        if (height <= 0 || width <= 0)
            throw invalid_argument("Dimensions must be positive");
        matrixHeight = height;
        matrixWidth = width;
        matrix.assign(height, vector<double>(width));
    }

    // Fill current matrix with random values
    void fillRandom(double minVal = 0.0, double maxVal = 1.0)
    {
        if (matrixHeight <= 0 || matrixWidth <= 0)
            throw runtime_error("Matrix not initialized (height/width == 0)");
        mt19937 generator(random_device{}());
        uniform_real_distribution<double> distribution(minVal, maxVal);
        for (int y = 0; y < matrixHeight; ++y)
            for (int x = 0; x < matrixWidth; ++x)
                matrix[y][x] = distribution(generator);
    }

    // Save to folder using provided height/width (caller supplies dims)
    // Caller must provide the same dims as the in-memory matrix.
    void saveToBinary(const string& folderPath, int height, int width) const
    {
        if (height != matrixHeight || width != matrixWidth)
            throw runtime_error("saveToBinary: provided dimensions do not match in-memory matrix");

        string file = buildFilePath(folderPath, height, width);
        cout << "Saving matrix to: " << file << '\n';

        ofstream out(file, ios::binary);
        if (!out) throw runtime_error("Failed to open output file: " + file);

        out.write(reinterpret_cast<const char*>(&height), sizeof(int));
        out.write(reinterpret_cast<const char*>(&width), sizeof(int));

        for (int y = 0; y < height; ++y)
            out.write(reinterpret_cast<const char*>(matrix[y].data()),
                      static_cast<std::streamsize>(width * sizeof(double)));
    }

    // Load from folder using provided height/width.
    // This resizes the internal matrix to the provided dims and fills it with data from file.
    void loadFromBinary(const string& folderPath, int height, int width)
    {
        if (height <= 0 || width <= 0)
            throw invalid_argument("loadFromBinary: dimensions must be positive");

        string file = buildFilePath(folderPath, height, width);

        ifstream in(file, ios::binary);
        if (!in) throw runtime_error("Failed to open input file: " + file);

        int h = 0, w = 0;
        in.read(reinterpret_cast<char*>(&h), sizeof(int));
        in.read(reinterpret_cast<char*>(&w), sizeof(int));

        if (h != height || w != width)
            throw runtime_error("loadFromBinary: file header dims do not match provided dims");

        // resize storage and read
        matrixHeight = height;
        matrixWidth = width;
        matrix.assign(matrixHeight, vector<double>(matrixWidth));

        for (int y = 0; y < matrixHeight; ++y)
            in.read(reinterpret_cast<char*>(matrix[y].data()),
                    static_cast<std::streamsize>(matrixWidth * sizeof(double)));

        // cout << "Loaded matrix dims: " << matrix.size() << " x "
        //      << (matrix.empty() ? 0 : matrix[0].size()) << '\n';
    }

    // Accessors
    const vector<vector<double>>& getMatrix() const { return matrix; }
    std::vector<std::vector<double>>& getMatrix() { return matrix; }
    int getHeight() const { return matrixHeight; }
    int getWidth() const { return matrixWidth; }
};

#endif // DATA2D_H
