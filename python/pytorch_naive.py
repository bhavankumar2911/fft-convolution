import os
import argparse
import csv
import time
import numpy as np
import torch
import torch.nn.functional as F


def read_bin_matrix(path, size, dtype):
    matrix = np.fromfile(path, dtype=dtype)
    return matrix.reshape(size, size)


def save_bin_matrix(path, matrix):
    assert matrix.flags["C_CONTIGUOUS"]
    matrix.tofile(path)


def get_padding(kernel_size, padding_mode):
    if padding_mode == "same":
        return kernel_size // 2
    if padding_mode == "valid":
        return 0
    if padding_mode == "full":
        return kernel_size - 1
    raise ValueError(f"Unknown padding mode: {padding_mode}")


def perform_cross_correlation(
    input_parent_directory: str,
    output_parent_directory: str,
    image_kernel_size_map: dict,
    dtype: np.dtype,
    padding_mode: str,
    csv_path: str
):
    images_directory = os.path.join(input_parent_directory, "images")
    kernels_directory = os.path.join(input_parent_directory, "kernels")

    os.makedirs(output_parent_directory, exist_ok=True)

    torch_dtype = torch.float32 if dtype == np.float32 else torch.float64

    csv_file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)

        if not csv_file_exists:
            csv_writer.writerow([
                "image_size",
                "kernel_size",
                "padding_mode",
                "dtype",
                "operation_time_ms"
            ])

        for image_size, kernel_sizes in image_kernel_size_map.items():
            image_path = os.path.join(
                images_directory,
                f"{image_size}x{image_size}.bin"
            )

            image_matrix = read_bin_matrix(
                image_path,
                image_size,
                dtype
            )

            image_tensor = (
                torch.from_numpy(image_matrix)
                .to(dtype=torch_dtype)
                .unsqueeze(0)
                .unsqueeze(0)
            )

            for kernel_size in kernel_sizes:
                kernel_path = os.path.join(
                    kernels_directory,
                    f"{kernel_size}x{kernel_size}.bin"
                )

                kernel_matrix = read_bin_matrix(
                    kernel_path,
                    kernel_size,
                    dtype
                )

                kernel_tensor = (
                    torch.from_numpy(kernel_matrix)
                    .to(dtype=torch_dtype)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

                padding = get_padding(kernel_size, padding_mode)

                with torch.no_grad():
                    start_time = time.perf_counter()

                    output_tensor = F.conv2d(
                        image_tensor,
                        kernel_tensor,
                        bias=None,
                        stride=1,
                        padding=padding
                    )

                    end_time = time.perf_counter()

                operation_time_ms = (end_time - start_time) * 1000.0

                output_matrix = (
                    output_tensor
                    .squeeze(0)
                    .squeeze(0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )

                output_path = os.path.join(
                    output_parent_directory,
                    f"{image_size}x{image_size}_"
                    f"{kernel_size}x{kernel_size}_"
                    f"{padding_mode}.bin"
                )

                save_bin_matrix(output_path, output_matrix)

                csv_writer.writerow([
                    image_size,
                    kernel_size,
                    padding_mode,
                    str(dtype),
                    f"{operation_time_ms:.6f}"
                ])

                print(
                    f"Saved: image={image_size}, "
                    f"kernel={kernel_size}, "
                    f"mode={padding_mode}, "
                    f"time={operation_time_ms:.3f} ms"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data64")
    parser.add_argument("--output_dir", type=str, default="./naive_outputs64")
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64")
    parser.add_argument("--padding_mode", type=str, choices=["same", "valid", "full"], default="same")
    parser.add_argument("--csv_path", type=str, default="./naive_conv_results.csv")

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

    perform_cross_correlation(
        input_parent_directory=args.input_dir,
        output_parent_directory=args.output_dir,
        image_kernel_size_map=image_kernel_size_map,
        dtype=dtype_map[args.dtype],
        padding_mode=args.padding_mode,
        csv_path=args.csv_path
    )

    print("Timing CSV generation completed")
