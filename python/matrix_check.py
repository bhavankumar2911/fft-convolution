import os
import argparse
import numpy as np


def read_bin_vector(path, dtype):
    return np.fromfile(path, dtype=dtype)


def compare_vectors(a, b):
    diff = a - b
    abs_diff = np.abs(diff)

    max_abs_error = np.max(abs_diff)
    mean_abs_error = np.mean(abs_diff)
    sum_abs_error = np.sum(abs_diff)
    l2_relative_error = (
        np.linalg.norm(diff) /
        (np.linalg.norm(a) + 1e-12)
    )

    return (
        max_abs_error,
        mean_abs_error,
        sum_abs_error,
        l2_relative_error
    )


def compare_directories(
    dir_a,
    dir_b,
    dtype,
    atol,
    rtol
):
    files = sorted(
        f for f in os.listdir(dir_a)
        if f.endswith(".bin")
    )

    for filename in files:
        path_a = os.path.join(dir_a, filename)
        path_b = os.path.join(dir_b, filename)

        if not os.path.exists(path_b):
            raise FileNotFoundError(f"Missing file: {filename}")

        a = read_bin_vector(path_a, dtype)
        b = read_bin_vector(path_b, dtype)

        if a.size != b.size:
            raise ValueError(
                f"Size mismatch for {filename}: {a.size} vs {b.size}"
            )

        max_err, mean_err, sum_err, rel_err = compare_vectors(a, b)

        passed = (max_err <= atol) or (rel_err <= rtol)
        status = "PASS" if passed else "FAIL"

        print(f"\n[{status}] {filename}")
        print(f"Elements           : {a.size}")
        print(f"Max abs error      : {max_err:.10f}")
        print(f"Mean abs error     : {mean_err:.10f}")
        print(f"Sum abs error      : {sum_err:.10f}")
        print(f"Relative L2 error  : {rel_err:.12f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", type=str, default="./naive_outputs")
    parser.add_argument("--dir_b", type=str, default="../new-fft/fft_outputs")
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32"
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)

    args = parser.parse_args()

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64
    }

    compare_directories(
        dir_a=args.dir_a,
        dir_b=args.dir_b,
        dtype=dtype_map[args.dtype],
        atol=args.atol,
        rtol=args.rtol
    )
