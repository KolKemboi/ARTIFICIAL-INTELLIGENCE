import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2



device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.455], std=[0.250, 0.252, 0.251]),
])

class CustomLoader(Dataset):
    def __init__(self, root_dir, transform):
        self.dir = root_dir
        self.class_to_idx = {}
        self.image_path = []
        self.labels = []
        self.transform = transform

        for idx, subdir in enumerate(sorted(os.listdir(self.dir))):
            subdir_path = os.path.join(self.dir, subdir)
            if os.path.isdir(subdir_path):
                self.class_to_idx[subdir] = idx
                for fname in os.listdir(subdir_path):
                    if fname.endswith((".png", ".jpeg", ".jpg")):
                        self.image_path.append(os.path.join(subdir_path, fname))
                        self.labels.append(idx)

        
    def __len__(self) -> int:
        return len(self.image_path)
    
    def __getitem__(self, index):
        img_path = self.image_path[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = self.labels[index]

        return image, label
    

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size= 3, stride= 1, padding= 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 3, stride= 1, padding= 1)
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.flatten = self._get_flatten_size()
        self.fc1 = nn.Linear(self.flatten, 128)
        self.fc2 = nn.Linear(128, 3)

    def _get_flatten_size(self):
        with torch.no_grad():
            dummy_data = torch.zeros(1, 3, 128, 128)
            x = self.pool(F.relu(self.conv1(dummy_data)))
            x = self.pool(F.relu(self.conv2(x)))

            return x.numel()
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

TRAIN_FOLDER = os.path.join("dataset", "Images", "Train") 
TEST_FOLDER = os.path.join("dataset", "Images", "Test")

# Initialize datasets and loaders
train_data = CustomLoader(root_dir=TRAIN_FOLDER, transform = transform)
test_data = CustomLoader(root_dir=TEST_FOLDER, transform = transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

model = CNN()

epoch_count = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(epoch_count):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print("Training")
        break

