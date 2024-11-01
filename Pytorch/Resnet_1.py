import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import tranforms
import os
import cv2 


device = torch.device("cpu")

class CustomLoader(Dataset):
	def __init__(self, root_dir, tranform):
		self.path = root_dir
		self.tranform = tranform
		self.image_paths = []
		self.labels = []
		self.class_to_idx = {}

		for idx, subdir in enumerate(sorted(os.listdir(self.path))):
			subdir_path = os.path.join(self.path, subdir)
			if os.path.isdir(subdir_path):
				self.class_to_idx[idx] = subdir
				

	def __len__(self):
		return len(self.image_paths)


	def __getitem__(self, idx):
