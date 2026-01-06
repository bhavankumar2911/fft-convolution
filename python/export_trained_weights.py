import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import csv
import numpy as np
import os


# --------------------------------------------------
# Save model weights as raw float64 .bin files
# --------------------------------------------------
def save_model_weights_as_bin(
    model: nn.Module,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)

    metadata_lines = []

    for name, param in model.state_dict().items():
        param_numpy = (
            param.detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )

        assert param_numpy.flags["C_CONTIGUOUS"]
        assert param_numpy.dtype == np.float64

        filename = name.replace(".", "_") + ".bin"
        filepath = os.path.join(output_dir, filename)

        param_numpy.tofile(filepath)

        metadata_lines.append(
            f"{filename} shape={param_numpy.shape} dtype=float64"
        )

    with open(os.path.join(output_dir, "metadata.txt"), "w") as f:
        for line in metadata_lines:
            f.write(line + "\n")


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # -----------------------------
    # Device (MPS preferred)
    # -----------------------------
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Using device: {device}")

    # -----------------------------
    # Config
    # -----------------------------
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-3
    num_workers = 0

    csv_file = "training_stats.csv"
    weights_bin_dir = "./trained_weights"

    # -----------------------------
    # Force float64 globally
    # -----------------------------
    torch.set_default_dtype(torch.float64)

    # -----------------------------
    # Transforms
    # -----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float64),
        transforms.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2241, 0.2215, 0.2239]
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float64),
        transforms.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2241, 0.2215, 0.2239]
        )
    ])

    # -----------------------------
    # Datasets & Loaders
    # -----------------------------
    train_dataset = datasets.STL10(
        root="./stldata",
        split="train",
        download=False,
        transform=transform_train
    )

    test_dataset = datasets.STL10(
        root="./stldata",
        split="test",
        download=False,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # -----------------------------
    # Model
    # -----------------------------
    class STL10CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 5, padding=2, dtype=torch.float64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, 5, padding=2, dtype=torch.float64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, 3, padding=1, dtype=torch.float64),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, 3, padding=1, dtype=torch.float64),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 6 * 6, 512, dtype=torch.float64),
                nn.ReLU(),
                nn.Linear(512, 10, dtype=torch.float64)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(1)
            return self.classifier(x)

    model = STL10CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # -----------------------------
    # CSV setup
    # -----------------------------
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "test_accuracy",
            "epoch_time_sec"
        ])

    # -----------------------------
    # Training + Evaluation
    # -----------------------------
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.perf_counter()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_accuracy = 100.0 * correct / total
        epoch_time = time.perf_counter() - epoch_start_time

        print(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Test Acc: {test_accuracy:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                train_accuracy,
                test_accuracy,
                epoch_time
            ])

    # -----------------------------
    # Save raw float64 weights
    # -----------------------------
    model_cpu = model.to("cpu")
    model_cpu.eval()

    save_model_weights_as_bin(
        model=model_cpu,
        output_dir=weights_bin_dir
    )

    print(f"Saved float64 weights to: {weights_bin_dir}")

    # -----------------------------
    # TorchScript Export (optional)
    # -----------------------------
    example_input = torch.randn(
        1, 3, 96, 96,
        dtype=torch.float64
    )

    traced = torch.jit.trace(model_cpu, example_input)
    traced.save("stl10_cnn_same_stride1_cpu.pt")

    print("Saved TorchScript model: stl10_cnn_same_stride1_cpu.pt")
    print(f"Training stats saved to: {csv_file}")


if __name__ == "__main__":
    main()
