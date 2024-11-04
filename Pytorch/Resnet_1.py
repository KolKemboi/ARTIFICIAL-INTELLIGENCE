import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import tranforms
import os
import cv2 


device = torch.device("cpu")


tranform = tranforms.Compose([
    tranforms.ToPILImage(),
    tranforms.Resize((128, 128)),
    tranforms.ToTensor(),
    tranforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomLoader(Dataset):
    def __init__(self, root_dir, transform):
        self.path = root_dir
        self.transform = transform
        self.image_paths = list()
        self.labels = list()
        self.class_to_idx = dict()

        for idx, subdir in enumerate(sorted(os.listdir(self.path))):
            subdir_path = os.path.join(self.path, subdir)
            if os.path.isdir(subdir_path):
                self.class_to_idx[idx] = subdir
                for fname in os.listdir(subdir_path):
                    if fname.endswith((".jpg", ".png", ".jpeg")):
                        image_path = os.path.join(subdir_path, fname)
                        self.image_paths.append(image_path)
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index) :
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = self.transform(image)
        label = self.labels[index]

        return image, label
    
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = false)
        self.bn2 = nn.BatchNorm(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)

        return out