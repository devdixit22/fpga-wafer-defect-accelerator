"""
Step 2: Train the CNN model on the prepared dataset.

Reads data from data/train/ and data/val/.
Saves the trained model to wafer_model.pth.

Usage:
    python software/2_train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import N, IMAGE_SIZE, NUM_CLASSES

EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001


class WaferCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers: N filters in conv1, 2*N in conv2
        self.conv1 = nn.Conv2d(1, N, 3, padding=1)
        self.conv2 = nn.Conv2d(N, N*2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # FC input size
        spatial = IMAGE_SIZE // 4
        fc_in = (N * 2) * spatial * spatial
        self.fc1 = nn.Linear(fc_in, N * 4)
        self.fc2 = nn.Linear(N * 4, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} (N={N}, Image={IMAGE_SIZE}x{IMAGE_SIZE})")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    train_dir = ROOT / "data" / "train"
    val_dir = ROOT / "data" / "val"
    assert train_dir.exists(), "Run 1_prepare_data.py first!"

    train_ds = datasets.ImageFolder(str(train_dir), transform=transform)
    val_ds = datasets.ImageFolder(str(val_dir), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WaferCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                v_correct += outputs.max(1)[1].eq(labels).sum().item()
                v_total += labels.size(0)
        
        val_acc = 100. * v_correct / v_total
        best_acc = max(best_acc, val_acc)

        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}%")

    out_path = ROOT / "wafer_model.pth"
    torch.save(model.state_dict(), str(out_path))
    print(f"\nModel saved to {out_path.name} (Best Val Acc: {best_acc:.1f}%)")


if __name__ == "__main__":
    main()
