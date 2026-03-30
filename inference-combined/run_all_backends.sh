#!/bin/bash
# run_all_backends.sh
# Compiles and runs inference for all 4 backends sequentially.
# Run from the project root directory.

set -e  # exit immediately on any error

# -------------------------------------------------
# Configuration — adjust paths if needed
# -------------------------------------------------
SRC="src/main.cpp"
INCLUDE_DIR="include"
WEIGHTS_DIR="trained_weights_fp32"
BINARY_DIR="src"

NVCC_FLAGS="-O3 -std=c++17 -x cu -I${INCLUDE_DIR} -lcufft"

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
compile() {
    local binary=$1
    local macro=$2

    echo ""
    echo "============================================"
    echo " Compiling: ${binary}"
    echo "============================================"

    if [ -z "$macro" ]; then
        # CPU naive — no macro needed
        nvcc ${NVCC_FLAGS} -o ${BINARY_DIR}/${binary} ${SRC}
    else
        nvcc ${NVCC_FLAGS} -D${macro} -o ${BINARY_DIR}/${binary} ${SRC}
    fi

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
#compile "inference_cpu_naive"  ""
#run     "inference_cpu_naive"

# -------------------------------------------------
# Backend 2 — GPU Naive
# -------------------------------------------------
compile "inference_gpu_naive"  "BACKEND_GPU_NAIVE"
run     "inference_gpu_naive"

# -------------------------------------------------
# Backend 3 — GPU FFT
# -------------------------------------------------
compile "inference_gpu_fft"    "BACKEND_GPU_FFT"
run     "inference_gpu_fft"

# -------------------------------------------------
# Backend 4 — GPU Hybrid
# -------------------------------------------------
compile "inference_gpu_hybrid" "BACKEND_GPU_HYBRID"
run     "inference_gpu_hybrid"

# -------------------------------------------------
# Done
# -------------------------------------------------
echo ""
echo "============================================"
echo " All backends complete."
echo " Results saved in: results/"
echo "============================================"