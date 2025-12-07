#include "Data2D.hpp"
#include "NaiveConvolution.hpp"
#include <iostream>
using namespace std;

int main()
{
    Data2D image;
    Data2D kernel;

    image.loadFromBinary("./test/images", 500, 500);
    kernel.loadFromBinary("./test/kernels", 15, 15);

    NaiveConvolution2D conv(image.getMatrix(), kernel.getMatrix());

    auto result = conv.computeConvolution();

    cout << "Convolution time: " 
         << result.elapsedMilliseconds 
         << " ms\n";
}
