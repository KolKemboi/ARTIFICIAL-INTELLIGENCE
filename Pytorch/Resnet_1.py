import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import tranforms
import os
import cv2 


device = torch.device("cpu")

