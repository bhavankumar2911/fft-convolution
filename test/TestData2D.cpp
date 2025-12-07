#include "../Data2D.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <cstring>

using namespace std;
namespace fs = filesystem;

static void printUsage(const char* prog)
{
    cout << "Usage: " << prog << " [height] [width] [folderPath]\n"
         << "  height     : matrix height (default 15)\n"
         << "  width      : matrix width  (default 15)\n"
         << "  folderPath : output folder  (default ./kernels)\n";
}

class Data2DTester
{
public:
    static void runTest(int height, int width, const string& folderPath)
    {
        if (height <= 0 || width <= 0)
            throw runtime_error("Height and width must be positive integers");

        // ensure folder exists
        if (!fs::exists(folderPath))
        {
            if (!fs::create_directories(folderPath))
                throw runtime_error("Failed to create data folder: " + folderPath);
        }

        // prepare data (single unified matrix)
        Data2D original(height, width);
        original.fillRandom();

        // save to binary (uses automatic filename pattern data<height>x<width>.bin)
        auto t0 = chrono::high_resolution_clock::now();
        original.saveToBinary(folderPath, height, width);
        auto t1 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> saveTime = t1 - t0;

        cout << "Binary save completed in "
             << fixed << setprecision(3)
             << saveTime.count() << " ms\n";

        // load into a fresh object (demonstrates other classes/programs can call load)
        Data2D loaded; // default construct and let loadFromBinary resize
        t0 = chrono::high_resolution_clock::now();
        loaded.loadFromBinary(folderPath, height, width);
        t1 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> loadTime = t1 - t0;

        cout << "Binary load completed in "
             << fixed << setprecision(3)
             << loadTime.count() << " ms\n";

        if (verify(original, loaded))
            cout << "Data2D binary serialization test SUCCESS\n";
        else
            cout << "Data2D binary serialization test FAILED\n";
    }

private:
    static bool verify(const Data2D& a, const Data2D& b)
    {
        const double epsilon = 1e-12;

        const auto& matA = a.getMatrix();
        const auto& matB = b.getMatrix();

        int h = a.getHeight();
        int w = a.getWidth();

        if (h != b.getHeight() || w != b.getWidth())
            return false;

        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                if (fabs(matA[y][x] - matB[y][x]) > epsilon)
                    return false;

        return true;
    }
};

int main(int argc, char** argv)
{
    // defaults
    int height = 15;
    int width  = 15;
    string folder = "./kernels";

    if (argc > 1)
    {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (argc >= 2)
    {
        try { height = stoi(argv[1]); }
        catch (...) { cerr << "Invalid height: " << argv[1] << '\n'; printUsage(argv[0]); return 1; }
    }
    if (argc >= 3)
    {
        try { width = stoi(argv[2]); }
        catch (...) { cerr << "Invalid width: " << argv[2] << '\n'; printUsage(argv[0]); return 1; }
    }
    if (argc >= 4)
    {
        folder = argv[3];
    }

    try
    {
        Data2DTester::runTest(height, width, folder);
    }
    catch (const exception& ex)
    {
        cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
