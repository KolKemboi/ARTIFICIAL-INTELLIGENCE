import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class CustomDataLoader(Dataset):
    def __init__(self, image_dir, masks_dir, transform = None):
        self.images_dir = image_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.images[index])
        mask_path = os.path.join(self.masks_dir, self.images[index].replace(".png", "_mask.png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (np.array(mask) > 0).astype(np.float32)
        mask = torch.tensor(mask, dtype = torch.float).unsqueeze(0)


        return image, mask
    