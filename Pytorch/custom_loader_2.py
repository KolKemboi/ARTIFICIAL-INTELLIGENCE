import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import os

device = torch.device("cpu")

class CustomDataLoader(Dataset):
    def __init__(self, root_dir):
        self.dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        for idx, subdir in enumerate(sorted(os.listdir(root_dir))):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                self.class_to_idx[subdir] = idx
                for fname in os.listdir(subdir_path):
                    if fname.endswith((".png", ".jpeg", ".jpg")):
                        self.image_paths.append(os.path.join(subdir_path, fname))
                        self.labels.append(idx)


        self.transfrom = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.228]),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transfrom(image)
        label = self.labels[idx]

        return image, label

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size= 3, stride = 1, padding= 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 3, stride = 1, padding= 1)
        self.maxpool = nn.MaxPool2d(kernel_size= 2, stride = 2)
        self.flatten = self._get_flatten_size()
        self.fc1 = nn.Linear(self.flatten, 128)
        self.fc2 = nn.Linear(128, 3)


    def _get_flatten_size(self):
        with torch.no_grad():
            dummy_data = torch.zeros(1, 3, 128, 128)
            x = self.maxpool(F.relu(self.conv1(dummy_data)))
            x = self.maxpool(F.relu(self.conv2(x)))
            return x.numel()
        
    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    

TRAIN_FOLDER = os.path.join("dataset", "Images", "Train") 
TEST_FOLDER = os.path.join("dataset", "Images", "Test")

# Initialize datasets and loaders
train_data = CustomDataLoader(root_dir=TRAIN_FOLDER)
test_data = CustomDataLoader(root_dir=TEST_FOLDER)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


model =  CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

epoch_count = 1
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

        