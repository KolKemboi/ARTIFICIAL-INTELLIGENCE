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
            subdir_path = os.path.join