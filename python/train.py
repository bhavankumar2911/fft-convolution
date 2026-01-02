import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time


# -----------------------------
# Configuration
# -----------------------------
batch_size = 128
num_epochs = 50
learning_rate = 1e-3
num_workers = 2   # macOS safe

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

print(f"Using device: {device}")


# -----------------------------
# Dataset & Transforms
# -----------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(96, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4467, 0.4398, 0.4066],
        std=[0.2241, 0.2215, 0.2239]
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4467, 0.4398, 0.4066],
        std=[0.2241, 0.2215, 0.2239]
    )
])

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
    num_workers=num_workers,
    pin_memory=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=False
)


# -----------------------------
# CNN Model (SAME padding, stride=1)
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


model = STL10CNN().to(device)


# -----------------------------
# Loss, Optimizer, Scheduler
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=20,
    gamma=0.1
)


# -----------------------------
# Training Loop
# -----------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.perf_counter()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{num_epochs}",
        leave=False
    )

    for images, labels in progress_bar:
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

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%"
        )

    epoch_time = time.perf_counter() - start_time
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    print(
        f"Epoch [{epoch}] "
        f"Loss: {avg_loss:.4f} "
        f"Accuracy: {accuracy:.2f}% "
        f"Time: {epoch_time:.2f}s"
    )


# -----------------------------
# Evaluation
# -----------------------------
def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(
            test_loader,
            desc="Evaluating",
            leave=False
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(
                acc=f"{100.0 * correct / total:.2f}%"
            )

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# -----------------------------
# Main (macOS multiprocessing fix)
# -----------------------------
if __name__ == "__main__":

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        evaluate()
        scheduler.step()

    torch.save(
        model.state_dict(),
        "stl10_cnn_same_stride1_mps.pth"
    )

    print("Saved model: stl10_cnn_same_stride1_mps.pth")
