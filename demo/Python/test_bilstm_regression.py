import os
import time
import torch
import torch.nn as nn
import colorama
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import csv

def parse_args():
	parser = argparse.ArgumentParser(description="BiLSTM regression")
	parser.add_argument("--task", required=True, type=str, choices=["train", "predict"], help="specify what kind of task")
	parser.add_argument("--epochs", type=int, default=100, help="number of training")
	parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size during training")
	parser.add_argument("--hidden_size", type=int, default=128, help="hidden state size")
	parser.add_argument("--num_layers", type=int, default=2, help="number of recurrent layers")
	parser.add_argument("--bidirectional", type=bool, default=True, help="specify whether to use LSTM or BiLSTM")
	parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate after first fc")
	parser.add_argument("--first_fc_features_length", type=int, default=512, help="the features length of the first layer fc")
	parser.add_argument("--loss_delta", type=float, default=0.2, help="huber loss delta value")
	parser.add_argument("--train_csv_file", type=str, help="train csv file")
	parser.add_argument("--val_csv_file", type=str, help="val csv file")
	parser.add_argument("--threshold", type=float, default=0.5, help="error margin")
	parser.add_argument("--model_name", type=str, help="the model generated during training or the model loaded during prediction")
	parser.add_argument("--pretrained_model", type=str, default="", help="pretrained model loaded during training")
	parser.add_argument("--gpu", type=str, default="0", help="set which graphics card to use. it can also support multiple graphics cards at the same time, for example 0,1")

	args = parser.parse_args()
	return args

class BiLSTM(nn.Module):
	def __init__(self,
		hidden_size: int = 256,
		num_layers: int = 2,
		bidirectional: bool = True,
		drop_rate: float = 0.0,
		first_fc_features_length: int = 512,
		device: str = "cpu"
	) -> None:
		super().__init__()
		self.num_directions = 2 if bidirectional else 1
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.device = device

		self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

		self.fc1 = nn.Linear(hidden_size*self.num_directions, first_fc_features_length)
		self.fc2 = nn.Linear(first_fc_features_length, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=drop_rate)

	def forward(self, x: Tensor) -> Tensor:
		# print(f"x shape: {x.shape}; type: {x.dtype}")
		batch_size = x.shape[0]
		h_0 = Variable(torch.zeros(self.num_directions*self.num_layers, batch_size, self.hidden_size).to(device))
		c_0 = Variable(torch.zeros(self.num_directions*self.num_layers, batch_size, self.hidden_size).to(device))

		out, _ = self.lstm(x, (h_0, c_0))
		out = out[:, -1, :]
		# out = self.sigmoid(out)
		out = self.relu(out)
		# print(f"relu shape: {out.shape}")
		out = self.fc1(out)
		# print(f"fc1 shape: {out.shape}")
		# out = self.sigmoid(out)
		out = self.relu(out)
		out = self.dropout(out)
		out = self.fc2(out)
		return out

class CustomDataset(Dataset):
	def __init__(self, csv_file):
		self.values = pd.read_csv(csv_file, header=None)

	def __len__(self):
		return len(self.values)

	def __getitem__(self, index):
		value = torch.tensor([self.values.iloc[index, 7], self.values.iloc[index, 9], self.values.iloc[index, 11], self.values.iloc[index, 12], self.values.iloc[index, 13], self.values.iloc[index, 14]]) # 7,9,11,12,13,14
		value = value.unsqueeze(0) # [6] ==> [1, 6]
		value = value.to(torch.float32) # torch.float64 ==> torch.float32
		label = self.values.iloc[index, 1]
		return value, label

def load_dataset(train_csv_file, val_csv_file, batch_size):
	train_dataset = CustomDataset(train_csv_file)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	val_dataset = CustomDataset(val_csv_file)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	return len(train_dataset), len(val_dataset), train_loader, val_loader

def get_model_parameters(model):
	print("model:", model)

	total_params = sum(p.numel() for p in model.parameters())
	print(f"total parameters: {total_params}")
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"training parameters: {total_trainable_params}")

	tensor = torch.rand(5, 1, 6)
	output = model(tensor)
	print(f"output: {output}; output.shape: {output.shape}; output.type: {output.type()}")
	raise ValueError(colorama.Fore.YELLOW + "for testing purposes")

def calculate_hit_rate(labels_src, labels_dst, length, threshold):
	assert len(labels_src) == len(labels_dst) and len(labels_src) == length, f"they must be of equal length: {len(labels_src)}, {len(labels_dst)}, {length}"

	count = 0
	for i in range(length):
		if abs(labels_src[i] - labels_dst[i]) <= threshold:
			count = count + 1
	return float(count) / length

def draw_graph(train_losses, train_accuracies, val_losses, val_accuracies):
	plt.subplot(1, 2, 1) # loss
	plt.title("Loss curve")
	plt.xlabel("Epoch Number")
	plt.ylabel("Loss")
	plt.plot(train_losses, color="blue")
	plt.plot(val_losses, color="red")
	plt.legend(["Train Loss", "Val Loss"])

	plt.subplot(1, 2, 2) # accuracy
	plt.title("Accuracy curve")
	plt.xlabel("Epoch Number")
	plt.ylabel("Accuracy")
	plt.plot(train_accuracies, color="blue")
	plt.plot(val_accuracies, color="red")
	plt.legend(["Train Accuracy", "Val Accuracy"])

	now = datetime.now()
	formatted_now = now.strftime("%Y-%m-%d-%H-%M-%S")
	# plt.savefig("regression_"+formatted_now+".png")
	plt.show()

def train(train_csv_file, val_csv_file, epochs, lr, batch_size, hidden_size, num_layers, bidirectional, drop_rate, first_fc_features_length, threshold, model_name, pretrained_model, loss_delta, device):
	train_dataset_num, val_dataset_num, train_loader, val_loader = load_dataset(train_csv_file, val_csv_file, batch_size)
	print(f"train dataset num: {train_dataset_num}; val dataset num: {val_dataset_num}")

	model = BiLSTM(hidden_size, num_layers, bidirectional, drop_rate, first_fc_features_length, device)
	if pretrained_model != "":
		model.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
	# get_model_parameters(model)
	model.to(device)

	# optimizer = optim.Adadelta(model.parameters(), lr=lr)
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
	# criterion = nn.MSELoss()
	criterion = nn.HuberLoss(delta=loss_delta)

	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	highest_accuracy_train = 0.
	minimum_loss_train = 10000.
	highest_accuracy_val = 0.
	minimum_loss_val = 10000.
	best_epoch_accuracy_train = 0
	best_epoch_loss_train = 0
	best_epoch_accuracy_val = 0
	best_epoch_loss_val = 0

	for epoch in range(epochs):
		epoch_start = time.time()

		train_loss = 0.0 # loss
		val_loss = 0.0
		labels_src = []
		labels_dst = []

		model.train() # set to training mode
		for _, (inputs, labels) in enumerate(train_loader):
			# print(f"type: {inputs.type()}; {labels.type()}; shape: {inputs.shape}; {labels.shape}"); raise
			inputs = inputs.to(device)
			labels = labels.to(device, dtype=torch.float).view(-1,1)
			flat = [item[0] for item in labels.tolist()]
			labels_src = labels_src + flat

			outputs = model(inputs) # forward pass
			flat = [item[0] for item in outputs.tolist()]
			labels_dst = labels_dst + flat
			# print(f"shape: {outputs.shape}; {labels.shape}")
			loss = criterion(outputs, labels) # compute loss

			optimizer.zero_grad() # clean existing gradients
			loss.backward() # backpropagate the gradients
			optimizer.step() # update the parameters

			train_loss += loss.item() * inputs.size(0) # compute the total loss

		avg_train_loss = train_loss / train_dataset_num # average training loss
		train_losses.append(avg_train_loss)
		avg_train_acc = calculate_hit_rate(labels_src, labels_dst, train_dataset_num, threshold)
		train_accuracies.append(avg_train_acc)

		labels_src = []
		labels_dst = []

		model.eval() # set to evaluation mode
		with torch.no_grad():
			for _, (inputs, labels) in enumerate(val_loader):
				inputs = inputs.to(device)
				labels = labels.to(device, dtype=torch.float).view(-1,1)
				flat = [item[0] for item in labels.tolist()]
				labels_src = labels_src + flat

				outputs = model(inputs) # forward pass
				flat = [item[0] for item in outputs.tolist()]
				labels_dst = labels_dst + flat
				loss = criterion(outputs, labels) # compute loss

				val_loss += loss.item() * inputs.size(0) # compute the total loss

		avg_val_loss = val_loss / val_dataset_num # average validation loss
		val_losses.append(avg_val_loss)
		avg_val_acc = calculate_hit_rate(labels_src, labels_dst, val_dataset_num, threshold) # average validation accuracy
		val_accuracies.append(avg_val_acc)

		epoch_end = time.time()
		print(f"epoch:{epoch+1}/{epochs}; train loss:{avg_train_loss:.6f}, accuracy:{avg_train_acc:.6f}; validation loss:{avg_val_loss:.6f}, accuracy:{avg_val_acc:.6f}; time:{epoch_end-epoch_start:.2f}s")

		if highest_accuracy_val < avg_val_acc: #and minimum_loss_val > avg_val_loss:
			torch.save(model.state_dict(), model_name)

		if highest_accuracy_val < avg_val_acc:
			highest_accuracy_val = avg_val_acc
			best_epoch_accuracy_val = epoch + 1
		if minimum_loss_val > avg_val_loss:
			minimum_loss_val = avg_val_loss
			best_epoch_loss_val = epoch + 1
		if highest_accuracy_train < avg_train_acc:
			highest_accuracy_train = avg_train_acc
			best_epoch_accuracy_train = epoch + 1
		if minimum_loss_train > avg_train_loss:
			minimum_loss_train = avg_train_loss
			best_epoch_loss_train = epoch + 1

		if highest_accuracy_train > 0.99 and highest_accuracy_val < 0.5:
			print(colorama.Fore.YELLOW + "overfitting")
			break

	print(f"train: loss:{minimum_loss_train:.6f}, epoch:{best_epoch_loss_train}, acc:{highest_accuracy_train:.6f}, epoch:{best_epoch_accuracy_train};  val: loss:{minimum_loss_val:.6f}, epoch:{best_epoch_loss_val}, acc:{highest_accuracy_val:.6f}, epoch:{best_epoch_accuracy_val}")
	# draw_graph(train_losses, train_accuracies, val_losses, val_accuracies)

def predict(model_name, csv_file, threshold, hidden_size, num_layers, bidirectional, first_fc_features_length, device):
	model = BiLSTM(hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, first_fc_features_length=first_fc_features_length, device=device)
	model.load_state_dict(torch.load(model_name))
	model.to(device)

	csv_values = []
	with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			csv_values.append(row)
	print(f"csv file length: {len(csv_values)}; value: {csv_values[0]}")
	print("\tname\tground truch\tpredict value\tresult")
	count = 0

	model.eval()
	with torch.no_grad():
		for value in csv_values:
			input = torch.tensor([float(value[7]), float(value[9]), float(value[11]), float(value[12]), float(value[13]), float(value[14])]) # 7,9,11,12,13,14
			input = input.unsqueeze(0) # [6] ==> [1, 6]
			input = input.unsqueeze(0) # [1, 6] ==> [1, 1, 6]
			input = input.to(torch.float32) # torch.float64 ==> torch.float32
			input = input.to(device)
			# print(f"input shape: {input.shape}"); raise

			output = model(input)
			# print(f"output shape: {output.shape}; value: {output}"); raise

			value1 = output[0,0].item()
			value2 = float(value[1])
			result = 0
			if abs(value2 - value1) <= threshold:
				result = 1
				count += 1
			result = str(result) + "(" + f"{value2-value1:.2f}" + ")"
			print(f"{value[2]}\t{value2}\t{value1:.2f}\t{result}")

	hit_rate = float(count) / len(csv_values)
	print(f"total number of predict: {len(csv_values)}, hit rate: {hit_rate:.6f}")


if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # set which graphics card to use: 0,1,2..., default is 0

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if args.task == "train":
		train(args.train_csv_file, args.val_csv_file, args.epochs, args.lr, args.batch_size, args.hidden_size, args.num_layers, args.bidirectional, args.drop_rate, args.first_fc_features_length, args.threshold, args.model_name, args.pretrained_model, args.loss_delta, device)
	else:
		predict(args.model_name, args.val_csv_file, args.threshold, args.hidden_size, args.num_layers, args.bidirectional, args.first_fc_features_length, device)

	print(colorama.Fore.GREEN + "====== execution completed ======")
