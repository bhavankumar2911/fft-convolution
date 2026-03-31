import numpy as np
import os
import argparse


def generate_and_save_matrices(
    parent_directory: str,
    image_kernel_size_map: dict,
    dtype: np.dtype,
    base_seed: int,
    in_channels: int = 1,
    out_channels: int = 1
):
    """
    Generates random image and convolution weight/bias data in the exact
    binary format expected by BinaryTensorLoader and the CNN conv classes.

    Layout on disk:
      images/   {H}x{W}_ic{inC}.bin         — flat float32, shape [inC, H, W]
      weights/  {K}x{K}_ic{inC}_oc{outC}.bin — flat float32, shape [outC, inC, K, K]
      biases/   {K}x{K}_oc{outC}.bin         — flat float32, shape [outC]

    These files can be passed directly to:
      NaiveCPUConvolution2D(inC, outC, K, pad, wPath, bPath)
      NaiveCUDAConvolution2D(inC, outC, K, pad, wPath, bPath)
      FFTConvolution2D_CUDA(inC, outC, K, pad, wPath, bPath)
      HybridConvolution2D(inC, outC, K, pad, wPath, bPath)
    """

    images_dir  = os.path.join(parent_directory, "images")
    weights_dir = os.path.join(parent_directory, "weights")
    biases_dir  = os.path.join(parent_directory, "biases")

    os.makedirs(images_dir,  exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(biases_dir,  exist_ok=True)

    seed_offset = 0

    for image_size, kernel_sizes in image_kernel_size_map.items():

        # -------------------------------------------------
        # Image: shape [inC, H, W]
        # -------------------------------------------------
        np.random.seed(base_seed + seed_offset)

        image = np.random.rand(
            in_channels, image_size, image_size
        ).astype(dtype)

        assert image.flags["C_CONTIGUOUS"]
        assert image.dtype == dtype

        image_filename = f"{image_size}x{image_size}_ic{in_channels}.bin"
        image.tofile(os.path.join(images_dir, image_filename))

        print(f"  Image  [{in_channels}, {image_size}, {image_size}] → {image_filename}")

        for kernel_size in kernel_sizes:

            # -------------------------------------------------
            # Weights: shape [outC, inC, K, K]
            # -------------------------------------------------
            np.random.seed(base_seed + seed_offset + kernel_size)

            weights = np.random.randn(
                out_channels, in_channels, kernel_size, kernel_size
            ).astype(dtype)

            assert weights.flags["C_CONTIGUOUS"]
            assert weights.dtype == dtype

            weight_filename = f"{kernel_size}x{kernel_size}_ic{in_channels}_oc{out_channels}.bin"
            weights.tofile(os.path.join(weights_dir, weight_filename))

            # -------------------------------------------------
            # Bias: shape [outC] — zeros for clean benchmark
            # -------------------------------------------------
            bias = np.zeros(out_channels, dtype=dtype)
            bias_filename = f"{kernel_size}x{kernel_size}_oc{out_channels}.bin"
            bias.tofile(os.path.join(biases_dir, bias_filename))

            ratio = image_size / kernel_size
            print(f"  Weight [{out_channels}, {in_channels}, {kernel_size}, {kernel_size}] "
                  f"→ {weight_filename}  (ratio={ratio:.1f})")

        seed_offset += 1

    print("\nGeneration complete.")
    print(f"  Images  → {images_dir}")
    print(f"  Weights → {weights_dir}")
    print(f"  Biases  → {biases_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="./bench_data",
        help="Parent directory for images/, weights/, biases/"
    )
    parser.add_argument(
        "--dtype", type=str, choices=["float32", "float64"], default="float32"
    )
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--in_channels", type=int, default=1,
                        help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=1,
                        help="Number of output channels")

    args = parser.parse_args()

    dtype_map = {"float32": np.float32, "float64": np.float64}

    # Realistic combinations from well-known models and datasets
    # Covers spatial-to-kernel ratios from 4.0 to 37
    # Includes STL-10 layers as reference points
    # Systematic grid: common image sizes 28→2048, kernels 3→101
    # Filter: K < H//2 to keep kernel smaller than half the image
    all_image_sizes  = [28, 56, 96, 112, 224, 256, 512, 1024, 2048]
    all_kernel_sizes = [3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 101]

    image_kernel_size_map = {
        H: [K for K in all_kernel_sizes if K < H // 2]
        for H in all_image_sizes
    }

    print(f"Generating benchmark data...")
    print(f"  dtype       : {args.dtype}")
    print(f"  in_channels : {args.in_channels}")
    print(f"  out_channels: {args.out_channels}")
    print(f"  output_dir  : {args.output_dir}")
    print()

    generate_and_save_matrices(
        parent_directory=args.output_dir,
        image_kernel_size_map=image_kernel_size_map,
        dtype=dtype_map[args.dtype],
        base_seed=args.seed,
        in_channels=args.in_channels,
        out_channels=args.out_channels
    )