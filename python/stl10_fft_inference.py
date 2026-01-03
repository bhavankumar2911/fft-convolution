import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fft_conv_layer import FFTConv2D
import time

# -----------------------------
# Force CPU everywhere
# -----------------------------
device = torch.device("cpu")
torch.set_default_device("cpu")

# -----------------------------
# Load trained weights
# -----------------------------
state = torch.load(
    "stl10_cnn_same_stride1_mps.pth",
    map_location="cpu"
)

# -----------------------------
# FFT-based CNN (CPU graph, CUDA conv inside)
# -----------------------------
class STL10_FFT_CNN(torch.nn.Module):
    def __init__(self, state):
        super().__init__()

        self.conv1 = FFTConv2D(
            state["features.0.weight"],
            state["features.0.bias"]
        )
        self.conv2 = FFTConv2D(
            state["features.4.weight"],
            state["features.4.bias"]
        )
        self.conv3 = FFTConv2D(
            state["features.8.weight"],
            state["features.8.bias"]
        )
        self.conv4 = FFTConv2D(
            state["features.12.weight"],
            state["features.12.bias"]
        )

        self.pool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(256 * 6 * 6, 512)
        self.fc2 = torch.nn.Linear(512, 10)

        self.fc1.weight.data.copy_(state["classifier.0.weight"])
        self.fc1.bias.data.copy_(state["classifier.0.bias"])
        self.fc2.weight.data.copy_(state["classifier.3.weight"])
        self.fc2.bias.data.copy_(state["classifier.3.bias"])

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = STL10_FFT_CNN(state).eval()

# -----------------------------
# Dataset (CPU)
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4467, 0.4398, 0.4066],
        std=[0.2241, 0.2215, 0.2239]
    )
])

test_set = datasets.STL10(
    root="./stldata",
    split="test",
    download=True,
    transform=transform
)

loader = DataLoader(
    test_set,
    batch_size=1,
    shuffle=False,
    pin_memory=False
)

# -----------------------------
# Inference timing
# -----------------------------
correct = 0
total = 0

start = time.perf_counter()

with torch.no_grad():
    for x, y in loader:
        # remain on CPU
        out = model(x)
        pred = out.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += 1

end = time.perf_counter()

print(f"Accuracy: {100 * correct / total:.2f}%")
print(f"Inference Time (CPU + CUDA conv only): {end - start:.2f}s")
