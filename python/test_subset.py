import os
import numpy as np
from torchvision.datasets import STL10
from torchvision import transforms

output_directory = "./test_images"
os.makedirs(output_directory, exist_ok=True)

transform = transforms.ToTensor()  # C x H x W, float32 [0,1]

dataset = STL10(
    root="./stldata",
    split="test",
    download=False,
    transform=transform
)

for index in range(100):
    image_tensor, label = dataset[index]
    image_numpy = image_tensor.numpy()

    np.save(
        f"{output_directory}/image_{index:03d}_label_{label}.npy",
        image_numpy
    )
