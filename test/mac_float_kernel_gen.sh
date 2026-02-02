#!/bin/bash

executable="./TestData2DFloatMac"
kernelFolder="./kernels/float"

kernelSizes=(3 5 7 11 21 31 51 101 151 201 301 501)

for kernelSize in "${kernelSizes[@]}"; do
    echo "Running for kernel size: $kernelSize"
    $executable $kernelSize $kernelSize "$kernelFolder"
done
