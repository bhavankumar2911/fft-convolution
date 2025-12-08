#!/bin/bash

executable="./TestData2DMac"
kernelFolder="./images/double"

kernelSizes=(256 512 1024 2028)

for kernelSize in "${kernelSizes[@]}"; do
    echo "Running for kernel size: $kernelSize"
    $executable $kernelSize $kernelSize "$kernelFolder"
done
