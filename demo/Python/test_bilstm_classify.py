import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import colorama
import argparse

# Blog: https://blog.csdn.net/fengbingchun/article/details/144138306

def parse_args():
	parser = argparse.ArgumentParser(description="BiLSTM classify")
	parser.add_argument("--epochs", type=int, default=10, help="number of training")
	parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size during training")
	parser.add_argument("--hidden_size", type=int, default=128, help="hidden state size")
	parser.add_argument("--num_layers", type=int, default=2, help="number of recurrent layers")

	args = parser.parse_args()
	return args

def load_data(batch_size):
	train_dataset = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
	test_dataset = torchvision.datasets.MNIST(root='../../data/', train=False, transform=transforms.ToTensor())

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	return train_loader, test_loader

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
		super(BiRNN, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
		self.fc = nn.Linear(hidden_size*2, num_classes) # 2 for bidirection
		self.device = device

	def forward(self, x):
		# Set initial states
		h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device) # 2 for bidirection, x.size(0)=batch_size
		c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(self.device)

		# Forward propagate LSTM
		out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
		# Decode the hidden state of the last time step
		out = self.fc(out[:, -1, :]) # out: tensor of shape (batch_size, num_classes)

		return out

def train(epochs, lr, batch_size, hidden_size, num_layers, device, input_size, sequence_length, num_classes):
	train_loader, test_loader = load_data(batch_size)

	model = BiRNN(input_size, hidden_size, num_layers, num_classes, device).to(device)

	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
	# Train the model
	total_step = len(train_loader)
	for epoch in range(epochs):
		model.train()
		for i, (images, labels) in enumerate(train_loader):
			images = images.reshape(-1, sequence_length, input_size).to(device)
			labels = labels.to(device)

			# Forward pass
			outputs = model(images)
			loss = criterion(outputs, labels)

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if (i+1) % 100 == 0:
				print ("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch+1, epochs, i+1, total_step, loss.item()))

		# Test the model
		model.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			for images, labels in test_loader:
				images = images.reshape(-1, sequence_length, input_size).to(device)
				labels = labels.to(device)
				outputs = model(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

			print("Test Accuracy of the model on the 10000 test images: {} %".format(100 * correct / total))

	# Save the model checkpoint
	torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
	# reference: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/bidirectional_recurrent_neural_network/main.py
	colorama.init(autoreset=True)
	args = parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	sequence_length = 28
	input_size = 28
	num_classes = 10

	train(args.epochs, args.lr, args.batch_size, args.hidden_size, args.num_layers, device, input_size, sequence_length, num_classes)

	print(colorama.Fore.GREEN + "====== execution completed ======")
