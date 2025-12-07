#include <vector>
#include <fftw3.h>
#include <cmath>

static int nextPowerOf2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

std::vector<std::vector<double>>
fft_conv2d_full(const std::vector<std::vector<double>>& img,
                const std::vector<std::vector<double>>& ker,
                bool correlationMode,
                bool useNextPowerOf2)     // <--- NEW
{
    int H  = img.size();
    int W  = img[0].size();
    int Kh = ker.size();
    int Kw = ker[0].size();

    int Hf = H + Kh - 1;
    int Wf = W + Kw - 1;

    int Hpad = useNextPowerOf2 ? nextPowerOf2(Hf) : Hf;
    int Wpad = useNextPowerOf2 ? nextPowerOf2(Wf) : Wf;

    int complex_cols = Wpad / 2 + 1;

    double* A = (double*)fftw_malloc(sizeof(double) * Hpad * Wpad);
    double* B = (double*)fftw_malloc(sizeof(double) * Hpad * Wpad);
    double* C = (double*)fftw_malloc(sizeof(double) * Hpad * Wpad);

    fftw_complex* FA = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Hpad * complex_cols);
    fftw_complex* FB = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Hpad * complex_cols);
    fftw_complex* FC = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * Hpad * complex_cols);

    // zero pad
    for (int i = 0; i < Hpad * Wpad; ++i) { A[i] = 0; B[i] = 0; }

    // copy image
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            A[r * Wpad + c] = img[r][c];

    // copy kernel
    for (int r = 0; r < Kh; ++r)
        for (int c = 0; c < Kw; ++c)
            B[r * Wpad + c] = ker[r][c];

    // plans
    fftw_plan pA = fftw_plan_dft_r2c_2d(Hpad, Wpad, A, FA, FFTW_ESTIMATE);
    fftw_plan pB = fftw_plan_dft_r2c_2d(Hpad, Wpad, B, FB, FFTW_ESTIMATE);
    fftw_plan pC = fftw_plan_dft_c2r_2d(Hpad, Wpad, FC, C, FFTW_ESTIMATE);

    fftw_execute(pA);
    fftw_execute(pB);

    // convolution or correlation
    for (int i = 0; i < Hpad * complex_cols; ++i)
    {
        double ar = FA[i][0], ai = FA[i][1];
        double br = FB[i][0], bi = FB[i][1];

        if (!correlationMode)
        {
            FC[i][0] = ar * br - ai * bi;
            FC[i][1] = ar * bi + ai * br;
        }
        else
        {
            FC[i][0] = ar * br + ai * bi;
            FC[i][1] = ai * br - ar * bi;
        }
    }

    fftw_execute(pC);

    // scale
    double scale = 1.0 / (Hpad * Wpad);
    for (int i = 0; i < Hpad * Wpad; ++i)
        C[i] *= scale;

    // crop back to original full convolution size
    std::vector<std::vector<double>> out(Hf, std::vector<double>(Wf));
    for (int r = 0; r < Hf; ++r)
        for (int c = 0; c < Wf; ++c)
            out[r][c] = C[r * Wpad + c];

    fftw_destroy_plan(pA);
    fftw_destroy_plan(pB);
    fftw_destroy_plan(pC);

    fftw_free(A); fftw_free(B); fftw_free(C);
    fftw_free(FA); fftw_free(FB); fftw_free(FC);

    return out;
}
