#include "../Data2D.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <cstring>

using namespace std;
namespace fs = filesystem;

// Choose data type here:
using T = float;

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

        if (!fs::exists(folderPath))
        {
            if (!fs::create_directories(folderPath))
                throw runtime_error("Failed to create data folder: " + folderPath);
        }

        Data2D<T> original(height, width);
        original.fillRandom();

        auto t0 = chrono::high_resolution_clock::now();
        // Data2D<T>::saveToBinary takes only folderPath in this API
        original.saveToBinary(folderPath);
        auto t1 = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> saveTime = t1 - t0;

        cout << "Binary save completed in "
             << fixed << setprecision(3)
             << saveTime.count() << " ms\n";

        Data2D<T> loaded; // default construct and let loadFromBinary resize
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
    static bool verify(const Data2D<T>& a, const Data2D<T>& b)
    {
        // tolerance scaled with type
        const long double eps_base = std::is_same<T, float>::value ? 1e-6L : 1e-12L;
        long double epsilon = eps_base;

        if (a.getHeight() != b.getHeight() || a.getWidth() != b.getWidth())
            return false;

        const size_t n = static_cast<size_t>(a.getHeight()) * static_cast<size_t>(a.getWidth());
        const T* pa = a.data().data();
        const T* pb = b.data().data();

        for (size_t i = 0; i < n; ++i)
        {
            long double da = static_cast<long double>(pa[i]);
            long double db = static_cast<long double>(pb[i]);
            if (fabsl(da - db) > epsilon) return false;
        }
        return true;
    }
};

int main(int argc, char** argv)
{
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
