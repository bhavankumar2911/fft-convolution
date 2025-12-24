import numpy as np
import os
import argparse

def generate_and_save_matrices(
    parent_directory: str,
    image_kernel_size_map: dict,
    dtype: np.dtype,
    base_seed: int
):
    images_directory = os.path.join(parent_directory, "images")
    kernels_directory = os.path.join(parent_directory, "kernels")

    os.makedirs(images_directory, exist_ok=True)
    os.makedirs(kernels_directory, exist_ok=True)

    seed_offset = 0

    for image_size, kernel_sizes in image_kernel_size_map.items():
        np.random.seed(base_seed + seed_offset)

        image_matrix = np.random.rand(
            image_size,
            image_size
        ).astype(dtype)

        assert image_matrix.flags["C_CONTIGUOUS"]
        assert image_matrix.dtype == dtype

        image_matrix.tofile(
            os.path.join(images_directory, f"{image_size}x{image_size}.bin")
        )

        for kernel_size in kernel_sizes:
            np.random.seed(base_seed + seed_offset + kernel_size)

            kernel_matrix = np.random.randn(
                kernel_size,
                kernel_size
            ).astype(dtype)

            assert kernel_matrix.flags["C_CONTIGUOUS"]
            assert kernel_matrix.dtype == dtype

            kernel_matrix.tofile(
                os.path.join(kernels_directory, f"{kernel_size}x{kernel_size}.bin")
            )

        seed_offset += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Parent directory for images/ and kernels/"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64
    }

    image_kernel_size_map = {
        256:  [3,   7,   21,  51,  101],
        512:  [5,   11,  31,  101, 151],
        1024: [11,  31,  101, 201, 301],
        2048: [21,  51,  151, 301, 501]
    }

    generate_and_save_matrices(
        parent_directory=args.output_dir,
        image_kernel_size_map=image_kernel_size_map,
        dtype=dtype_map[args.dtype],
        base_seed=args.seed
    )

    print("Matrix generation completed")
