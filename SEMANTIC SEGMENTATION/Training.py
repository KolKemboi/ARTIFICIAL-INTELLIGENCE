import torch
from torch import optim
from Dataloader import CustomDataLoader
from torchvision import transforms
from model import UNet
from torch.utils.data import DataLoader
from Transforms import ImageTransform

IMAGES_PATH = "C:\\Users\\kolke\\Documents\\DATASETS\\maskRCNN\\PNGImages"
MASK_PATH = "C:\\Users\\kolke\\Documents\\DATASETS\\maskRCNN\\PedMasks"

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

dataset = CustomDataLoader(IMAGES_PATH, MASK_PATH, transform)
trainloader = DataLoader(dataset, batch_size = 4, shuffle = True)

model = UNet()

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


epochs = 1

for epoch in range(epochs):
    for images, masks in trainloader:
        optimizer.zero_grad()

        outputs = model(images)
        masks = masks.squeeze(2)
        loss = loss_fn(outputs, masks)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


torch.save(model.state_dict(), "unet_penn_fudan.pth")