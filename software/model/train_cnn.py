import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

IMAGE_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 5
NUM_CLASSES = 9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.FakeData(
    size=3000,
    image_size=(1, IMAGE_SIZE, IMAGE_SIZE),
    num_classes=NUM_CLASSES,
    transform=transform
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

class WaferCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(32*8*8,64)
        self.fc2 = nn.Linear(64,NUM_CLASSES)

    def forward(self,x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0),-1)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = WaferCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(EPOCHS):

    total_loss = 0

    for images,labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

torch.save(model.state_dict(),"wafer_model.pth")

print("Model saved as wafer_model.pth")
