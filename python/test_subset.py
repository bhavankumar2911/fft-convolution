import os
from torchvision.datasets import STL10

output_directory = "./test_images"
os.makedirs(output_directory, exist_ok=True)

dataset = STL10(
    root="./stldata",
    split="test",
    download=False
)

for image_index in range(100):
    image, image_label = dataset[image_index]
    image.save(f"{output_directory}/image_{image_index:03d}_label_{image_label}.png")
