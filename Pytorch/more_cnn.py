import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cpu")

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5), (0.5)),

])

train_data = datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transform)
test_data = datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transform)


train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
		self.fc1 = nn.Linear(64 * 7 * 7, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x)))
		x = self.pool(torch.relu(self.conv2(x)))
		x = x.view(-1, 64 * 7 * 7)
		x = torch.relu(self.fc1(x))
		x = self.fc2(x)

		return x

model = CNN()

loss_fun = nn.CrossEntropyLoss()
adam_optim = optim.Adam(model.parameters(), lr = 0.001)

epoch_count = 1

for epoch in range(epoch_count):
	model.train()
	running_loss = 0.0
	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)

		loss = loss_fun(outputs, labels)

		adam_optim.zero_grad()
		loss.backward()
		adam_optim.step()

		running_loss += loss.item()
		print("Training")
		break

model.eval()
correct, total = 0, 0

with torch.no_grad():
	for images, labels in test_loader:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, pred = torch.max(outputs.data, 1)

		total += labels.size(0)
		correct += (pred == labels).sum().item()
		break
		
