import torchvision.transforms as transforms

class ImageTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust based on your data
        ])
        
    def __call__(self, x):
        return self.transform(x)