##make some imports
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

##choose the device
device = torch.device("cpu")

##make the transforms
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5), (0.5)),
	])

##get the data
train_data = datasets.FashionMNIST(root  = "./data", download = True, train = True, transform = transform)
test_data = datasets.FashionMNIST("./data", download = True, train = False, transform = transform)

##make the data loader
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data , batch_size = 64, shuffle = True)

##make the class, its torch.relu
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
		self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0);
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

##make the optim, and loss function
loss_fn = nn.CrossEntropyLoss()
adam_optim = optim.Adam(model.parameters(), lr = 0.001)
epoch_count = 1

"""
so, the loop goes this way,
loop over the epoch count
call the model to train
init the running loss, 
iterate over the images and labels in the train_loader
cast it to device
then give the model to pred, then get the loss
then zero the optimizer
calculate the loss
take a step with the optimizer
"""
for epoch in range(epoch_count):
	model.train()
	running_loss = 0.0

	for images, labels in train_loader:
		images, labels = images.to(device), labels.to(device)

		outputs = model(images)
		loss = loss_fn(outputs, labels)

		adam_optim.zero_grad()
		loss.backward()
		adam_optim.step()

		running_loss += loss.item()
		print(loss.item())

		print("trainin")
		break




model.eval()
correct, total = 0, 0
with torch.no_grad():
	for images, labels in test_loader:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, pred = torch.max(outputs.data, 1)
		print(outputs.data)
		print("-"* 10)
		print(pred)
		total = labels.size(0)
		correct += (pred == labels).sum().item()

		print("evalling")
		break








