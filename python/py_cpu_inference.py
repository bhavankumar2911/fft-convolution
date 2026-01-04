import os
import time
import numpy as np
import torch
import torch.nn as nn

# -----------------------------
# Force CPU
# -----------------------------
device = torch.device("cpu")
torch.set_default_device("cpu")

# -----------------------------
# Model Definition (exact match)
# -----------------------------
class STL10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

# -----------------------------
# Load Model (CPU)
# -----------------------------
model = STL10CNN()
model.load_state_dict(
    torch.load("stl10_cnn_same_stride1_mps.pth", map_location="cpu")
)
model.eval()

# -----------------------------
# Normalization (test-time)
# -----------------------------
mean = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
std  = torch.tensor([0.2241, 0.2215, 0.2239]).view(1, 3, 1, 1)

# -----------------------------
# Dataset Directory
# -----------------------------
input_directory = "./test_images"
image_files = sorted(
    f for f in os.listdir(input_directory) if f.endswith(".npy")
)

# -----------------------------
# Inference
# -----------------------------
correct = 0
total = 0

start_time = time.perf_counter()

with torch.no_grad():
    for file_name in image_files:
        image_path = os.path.join(input_directory, file_name)

        image_numpy = np.load(image_path)                # C x H x W
        image_tensor = torch.from_numpy(image_numpy)     # float32
        image_tensor = image_tensor.unsqueeze(0)         # 1 x C x H x W

        image_tensor = (image_tensor - mean) / std

        outputs = model(image_tensor)
        predicted_class = outputs.argmax(dim=1).item()

        true_label = int(file_name.split("_label_")[1].split(".")[0])

        is_correct = predicted_class == true_label
        correct += int(is_correct)
        total += 1

        print(
            f"{file_name} | "
            f"Predicted: {predicted_class} | "
            f"True: {true_label} | "
            f"{'OK' if is_correct else 'WRONG'}"
        )

elapsed_time = time.perf_counter() - start_time
accuracy = 100.0 * correct / total

print("\n-----------------------------")
print(f"Samples      : {total}")
print(f"Accuracy     : {accuracy:.2f}%")
print(f"Total Time   : {elapsed_time:.4f} s")
print(f"Avg / Image  : {elapsed_time / total:.6f} s")
