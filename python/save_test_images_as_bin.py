import os
import numpy as np
from torchvision.datasets import STL10
from torchvision import transforms
import argparse


def save_stl10_images_as_bin(
    output_directory: str,
    dtype: np.dtype,
    max_images: int
):
    os.makedirs(output_directory, exist_ok=True)

    transform = transforms.ToTensor()  # C x H x W, float32 [0,1]

    dataset = STL10(
        root="./stldata",
        split="test",
        download=False,
        transform=transform
    )

    metadata_lines = []

    for index in range(max_images):
        image_tensor, label = dataset[index]

        image_numpy = (
            image_tensor
            .numpy()
            .astype(dtype, copy=False)
        )

        assert image_numpy.flags["C_CONTIGUOUS"]
        assert image_numpy.dtype == dtype

        filename = f"image_{index:03d}_label_{label}.bin"
        filepath = os.path.join(output_directory, filename)

        image_numpy.tofile(filepath)

        metadata_lines.append(
            f"{filename} shape={image_numpy.shape} dtype={dtype}"
        )

    with open(os.path.join(output_directory, "metadata.txt"), "w") as f:
        for line in metadata_lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../inference-combined/test_images_bin"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64
    }

    save_stl10_images_as_bin(
        output_directory=args.output_dir,
        dtype=dtype_map[args.dtype],
        max_images=args.num_images
    )

    print(f"Saved {args.num_images} STL10 images as raw .bin files")


if __name__ == "__main__":
    main()
