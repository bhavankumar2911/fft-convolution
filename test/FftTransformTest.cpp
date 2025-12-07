#include "../Data2D.hpp"
#include "../FftTransform.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <folderPath> <height> <width>\n";
        return 1;
    }

    std::string folder = argv[1];
    int height = std::stoi(argv[2]);
    int width  = std::stoi(argv[3]);

    try {
        // Load the matrix
        Data2D inputMatrix;
        inputMatrix.loadFromBinary(folder, height, width);

        std::cout << "\n--- Input matrix ---\n";
        Data2D::printMatrix(inputMatrix);

        // Forward 2D FFT (padded)
        ComplexMatrix forwardFFT =
            Fft2DTransformer::compute2D(inputMatrix, inputMatrix.getHeight(), inputMatrix.getWidth(), true, true);

        std::cout << "\n--- Forward 2D FFT (padded) ---\n";
        printComplexMatrixSample(forwardFFT, 8, 8);

        // Inverse 2D FFT
        ComplexMatrix recovered = forwardFFT;
        Fft2DTransformer::inverse2DInPlace(recovered);

        std::cout << "\n--- Recovered Spatial Matrix (real parts) ---\n";
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x)
                std::cout << recovered[y][x].real() << "\t";
            std::cout << "\n";
        }

        std::cout << "\n--- Error matrix ---\n";
        Fft2DTransformer::printDifferenceMatrix(inputMatrix, recovered, 8, 8);
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 2;
    }

    return 0;
}
