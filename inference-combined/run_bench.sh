#!/bin/bash
# run_bench.sh
# Compiles bench_conv and runs all image/kernel combinations.
# Run generate_data.py separately before running this script.
# Run from project root directory.

set -e

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_DIR="./bench_data"
RESULTS_DIR="./bench_results"
CSV="${RESULTS_DIR}/bench_results.csv"
BINARY="./bench/bench_conv"
SRC="./bench/bench_conv.cpp"
INCLUDE_DIR="include"

# Image size → kernel sizes (must match generate_data.py)
declare -A COMBOS
COMBOS[28]="3 5 7 9 11"
COMBOS[56]="3 5 7 9 11 15 21"
COMBOS[96]="3 5 7 9 11 15 21 31"
COMBOS[112]="3 5 7 9 11 15 21 31 51"
COMBOS[224]="3 5 7 9 11 15 21 31 51 71 101"
COMBOS[256]="3 5 7 9 11 15 21 31 51 71 101"
COMBOS[512]="3 5 7 9 11 15 21 31 51 71 101"
COMBOS[1024]="3 5 7 9 11 15 21 31 51 71 101"
COMBOS[2048]="3 5 7 9 11 15 21 31 51 71 101"

IMAGE_SIZES="28 56 96 112 224 256 512 1024 2048"

# -------------------------------------------------
# Step 1 — Compile
# -------------------------------------------------
echo "============================================"
echo " Step 1: Compiling bench_conv"
echo "============================================"
mkdir -p bench
nvcc -O3 -std=c++17 -x cu \
    -I${INCLUDE_DIR} \
    -lcufft \
    -o ${BINARY} ${SRC}
echo " Done: ${BINARY}"

# -------------------------------------------------
# Step 2 — Run all combinations
# -------------------------------------------------
echo ""
echo "============================================"
echo " Step 2: Running benchmarks"
echo "============================================"
mkdir -p ${RESULTS_DIR}

# Remove old CSV so header is written fresh
rm -f ${CSV}

for IMAGE_SIZE in ${IMAGE_SIZES}; do
    for KERNEL_SIZE in ${COMBOS[$IMAGE_SIZE]}; do
        ${BINARY} ${IMAGE_SIZE} ${KERNEL_SIZE} ${DATA_DIR} ${CSV}
    done
done

# -------------------------------------------------
# Done
# -------------------------------------------------
echo ""
echo "============================================"
echo " All benchmarks complete."
echo " Results: ${CSV}"
echo "============================================"