#!/bin/bash
# run_all_backends.sh
# Compiles and runs inference for all 6 backends sequentially.
# Run from the project root directory.

set -e

# -------------------------------------------------
# FFTW runtime library path — self-contained
# -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/fftw-3.3.10/.libs:${LD_LIBRARY_PATH}"

# -------------------------------------------------
# Configuration
# -------------------------------------------------
SRC="src/main.cpp"
INCLUDE_DIR="include"
BINARY_DIR="src"

FFTW_INC="-I${SCRIPT_DIR}/fftw-3.3.10/api"
FFTW_LIB="-L${SCRIPT_DIR}/fftw-3.3.10/.libs -lfftw3f"

# CPU backends — need FFTW, no NVML
NVCC_FLAGS_CPU="-O3 -std=c++17 -x cu -I${INCLUDE_DIR} ${FFTW_INC} ${FFTW_LIB} -lcufft"

# GPU backends — need NVML, no FFTW
NVCC_FLAGS_GPU="-O3 -std=c++17 -x cu -I${INCLUDE_DIR} -lcufft -lnvidia-ml"

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
compile_cpu() {
    local binary=$1
    local macro=$2
    echo ""
    echo "============================================"
    echo " Compiling: ${binary}"
    echo "============================================"
    if [ -z "$macro" ]; then
        nvcc ${NVCC_FLAGS_CPU} -o ${BINARY_DIR}/${binary} ${SRC}
    else
        nvcc ${NVCC_FLAGS_CPU} -D${macro} -o ${BINARY_DIR}/${binary} ${SRC}
    fi
    echo " Done: ${BINARY_DIR}/${binary}"
}

compile_gpu() {
    local binary=$1
    local macro=$2
    echo ""
    echo "============================================"
    echo " Compiling: ${binary}"
    echo "============================================"
    nvcc ${NVCC_FLAGS_GPU} -D${macro} -o ${BINARY_DIR}/${binary} ${SRC}
    echo " Done: ${BINARY_DIR}/${binary}"
}

run() {
    local binary=$1
    echo ""
    echo "============================================"
    echo " Running: ${binary}"
    echo "============================================"
    cd src
    ./${binary}
    cd ..
    echo " Done: ${binary}"
}

# -------------------------------------------------
# Backend 1 — CPU Naive
# -------------------------------------------------
compile_cpu "inference_cpu_naive"  ""
run         "inference_cpu_naive"

# -------------------------------------------------
# Backend 2 — CPU FFT
# -------------------------------------------------
compile_cpu "inference_cpu_fft"    "BACKEND_CPU_FFT"
run         "inference_cpu_fft"

# -------------------------------------------------
# Backend 3 — CPU Hybrid
# -------------------------------------------------
compile_cpu "inference_cpu_hybrid" "BACKEND_CPU_HYBRID"
run         "inference_cpu_hybrid"

# -------------------------------------------------
# Backend 4 — GPU Naive
# -------------------------------------------------
compile_gpu "inference_gpu_naive"  "BACKEND_GPU_NAIVE"
run         "inference_gpu_naive"

# -------------------------------------------------
# Backend 5 — GPU FFT
# -------------------------------------------------
compile_gpu "inference_gpu_fft"    "BACKEND_GPU_FFT"
run         "inference_gpu_fft"

# -------------------------------------------------
# Backend 6 — GPU Hybrid
# -------------------------------------------------
compile_gpu "inference_gpu_hybrid" "BACKEND_GPU_HYBRID"
run         "inference_gpu_hybrid"

# -------------------------------------------------
# Done
# -------------------------------------------------
echo ""
echo "============================================"
echo " All 6 backends complete."
echo " Results saved in: results/"
echo "============================================"