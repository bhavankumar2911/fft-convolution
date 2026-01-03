#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "Matrix2D.hpp"
#include "PaddingMode.hpp"
#include "FFTCrossCorrelation2D_CUDA.hpp"

namespace py = pybind11;

py::array_t<double> fft_conv2d_forward(
    py::array_t<double> input,
    py::array_t<double> kernel,
    int padding_mode
)
{
    auto in = input.unchecked<2>();
    auto k  = kernel.unchecked<2>();

    Matrix2D<double> image(in.shape(0), in.shape(1));
    Matrix2D<double> filt(k.shape(0), k.shape(1));

    for (ssize_t i = 0; i < in.shape(0); ++i)
        for (ssize_t j = 0; j < in.shape(1); ++j)
            image(i, j) = in(i, j);

    for (ssize_t i = 0; i < k.shape(0); ++i)
        for (ssize_t j = 0; j < k.shape(1); ++j)
            filt(i, j) = k(i, j);

    auto out = FFTCrossCorrelation2D_CUDA::compute(
        image,
        filt,
        static_cast<PaddingMode>(padding_mode)
    );

    py::array_t<double> output({out.rows(), out.cols()});
    auto o = output.mutable_unchecked<2>();

    for (size_t i = 0; i < out.rows(); ++i)
        for (size_t j = 0; j < out.cols(); ++j)
            o(i, j) = out(i, j);

    return output;
}

PYBIND11_MODULE(fft_conv_cuda, m)
{
    m.def("forward", &fft_conv2d_forward, "FFT Conv2D CUDA");
}
