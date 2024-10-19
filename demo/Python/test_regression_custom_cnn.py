import colorama
import argparse
import time
import ast
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torch.utils.checkpoint as cp
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import split_regression_dataset
from typing import Type, Any, List, Optional, Tuple
from pathlib import Path
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import torchvision.models as models
import torch.nn.functional as F
from collections import OrderedDict

def parse_args():
	parser = argparse.ArgumentParser(description="Modified ResNet18/DenseNet image regression")
	parser.add_argument("--task", required=True, type=str, choices=["split", "train", "predict"], help="specify what kind of task")
	parser.add_argument("--src_dataset_path", type=str, help="source dataset path")
	parser.add_argument("--csv_file", type=str, help="source csv file")
	parser.add_argument("--dst_dataset_path", type=str, help="the path of the destination dataset after split")
	parser.add_argument("--resize", default=(64,512), help="the size to which images are resized when split the dataset, if(0,0),no scaling is done")
	parser.add_argument("--fill_value", default=(114,114,114), help="image fill value")
	parser.add_argument("--ratios", default=(0.8,0.1,0.1), help="the ratio of split the data set(train set, validation set, test set), the test set can be 0, but their sum must be 1")
	parser.add_argument("--net", type=str, choices=["resnet18", "densenet"], help="specifies which network to use for training and prediction")
	parser.add_argument("--epochs", type=int, help="number of training")
	parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
	parser.add_argument("--loss_delta", type=float, default=0.2, help="huber loss delta value")
	parser.add_argument("--batch_size", type=int, default=32, help="batch size during training")
	parser.add_argument("--drop_rate", type=float, default=0, help="dropout rate after each dense layer")
	parser.add_argument("--drop_rate2", type=float, default=0, help="dropout rate after fc")
	parser.add_argument("--last_fc_features_length", type=int, default=512, help="the features length of the last layer fc")
	parser.add_argument("--threshold", type=float, default=0.5, help="error margin")
	parser.add_argument("--mean", type=str, help="the mean of the training set of images")
	parser.add_argument("--std", type=str, help="the standard deviation of the training set of images")
	parser.add_argument("--model_name", type=str, help="the model generated during training or the model loaded during prediction")
	parser.add_argument("--pretrained_model", type=str, default="", help="pretrained model loaded during training")
	parser.add_argument("--predict_images_path", type=str, help="predict images path")
	parser.add_argument("--gpu", type=str, default="0", help="set which graphics card to use")

	args = parser.parse_args()
	return args

class BasicBlock(nn.Module):
	"""residual building block"""
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		stride: int = 1,
		downsample: nn.Module = None
	) -> None:
		super(BasicBlock, self).__init__()
		self.downsample = downsample
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x: Tensor) -> Tensor:
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class ResNet18(nn.Module):
	def __init__(
		self,
		block: Type[BasicBlock],
		init_weights: bool = False
	) -> None:
		super(ResNet18, self).__init__()
		layers = [2, 2, 2, 2]
		self.in_channels = 64
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=3, stride=(1,2), padding=1, bias=False)
		self.conv2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1, bias=False)
		self.conv3 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1, bias=False)
		self.bn = nn.BatchNorm2d(self.in_channels)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512, 1) # image regression

		if init_weights:
			self._init_weights()

	def _make_layer(
		self,
		block: Type[BasicBlock],
		out_channels: int,
		blocks: int,
		stride: int = 1
	) -> nn.Sequential:
		downsample = None
		if stride != 1:
			downsample = nn.Sequential(
				nn.AvgPool2d(kernel_size=2, stride=2),
				nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(out_channels),
			)

		layers = []
		layers.append(block(self.in_channels, out_channels, stride, downsample))
		self.in_channels = out_channels

		for _ in range(1, blocks):
			layers.append(block(self.in_channels, out_channels))

		return nn.Sequential(*layers)

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
				if m.bias is not None:
					nn.init.constant_(m.bias, 0.000001)
			# elif isinstance(m, nn.BatchNorm2d):
			# 	nn.init.uniform_(m.weight, 0.000001, 1.0)
			# 	if m.bias is not None:
			# 		nn.init.constant_(m.bias, 0.000001)
			elif isinstance(m, nn.Linear):
				# nn.init.normal_(m.weight, 0, 0.000001)
				nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
				nn.init.constant_(m.bias, 0.000001)

	def forward(self, x: Tensor) -> Tensor:
		# print("input x shape:", x.shape)
		x = self.conv1(x)
		# print("self.conv1 shape:", x.shape)
		x = self.bn(x)
		# print("self.bn1(x) shape:", x.shape)
		x = self.relu(x)
		# print("self.relu(x) shape:", x.shape)
		x = self.conv2(x)
		# print("self.conv2 shape:", x.shape)
		x = self.bn(x)
		# print("self.bn1(x) shape:", x.shape)
		x = self.relu(x)
		# print("self.relu(x) shape:", x.shape)
		x = self.conv3(x)
		# print("self.conv3 shape:", x.shape)
		x = self.bn(x)
		# print("self.bn1(x) shape:", x.shape)
		x = self.relu(x)
		# print("self.relu(x) shape:", x.shape)
		x = self.maxpool(x)
		# print("self.maxpool(x) shape:", x.shape)

		x = self.layer1(x)
		# print("self.layer1(x) shape:", x.shape)
		x = self.layer2(x)
		# print("self.layer2(x) shape", x.shape)
		x = self.layer3(x)
		# print("self.layer3(x) shape:", x.shape)
		x = self.layer4(x)
		# print("Dimensions of the last convolutional feature map:", x.shape)
		# print("self.layer4(x) shape:", x.shape)

		x = self.avgpool(x)
		# print("self.avgpool(x) shape:", x.shape)
		x = torch.flatten(x, 1)
		# print("torch.flatten(x,1) shape:", x.shape)
		x = self.fc(x)
		# print("self.fc(x) shape:", x.shape)
		# raise

		return x


class _DenseLayer(nn.Module):
	def __init__(
		self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, index: int
	) -> None:
		super().__init__()
		self.norm1 = nn.BatchNorm2d(num_input_features)
		self.relu1 = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False) # 1x1 conv

		self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
		self.relu2 = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False) # 3x3 conv

		if index == 0:
			self.norm3 = nn.BatchNorm2d(growth_rate)
			self.relu3 = nn.ReLU(inplace=True)
			self.conv3 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False) # 3x3 conv

			self.norm4 = nn.BatchNorm2d(growth_rate)
			self.relu4 = nn.ReLU(inplace=True)
			self.conv4 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False) # 3x3 conv

		self.drop_rate = float(drop_rate)
		self.index = index

	def bn_function(self, inputs: List[Tensor]) -> Tensor:
		concated_features = torch.cat(inputs, 1)
		bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
		return bottleneck_output

	@torch.jit._overload_method  # noqa: F811
	def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
		pass

	@torch.jit._overload_method  # noqa: F811
	def forward(self, input: Tensor) -> Tensor:  # noqa: F811
		pass

	# torchscript does not yet support *args, so we overload method
	# allowing it to take either a List[Tensor] or single Tensor
	def forward(self, input: Tensor) -> Tensor:  # noqa: F811
		bottleneck_output = self.bn_function(input)

		new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
		if self.index == 0:
			new_features = self.conv3(self.relu3(self.norm3(new_features)))
			new_features = self.conv4(self.relu4(self.norm4(new_features)))

		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
		return new_features

class _DenseBlock(nn.ModuleDict):
	def __init__(
		self,
		num_layers: int,
		num_input_features: int,
		bn_size: int,
		growth_rate: int,
		drop_rate: float,
		index: int
	) -> None:
		super().__init__()
		for i in range(num_layers):
			layer = _DenseLayer(
				num_input_features + i * growth_rate,
				growth_rate=growth_rate,
				bn_size=bn_size,
				drop_rate=drop_rate,
				index=index,
			)
			self.add_module("denselayer%d" % (i + 1), layer)

	def forward(self, init_features: Tensor) -> Tensor:
		features = [init_features]
		# print(f"init_features shape: {init_features.shape}")
		for name, layer in self.items():
			new_features = layer(features)
			# print(f"name: {name}; new_features shape: {new_features.shape}")
			features.append(new_features)
			# print(f"features shape: {torch.cat(features,1).shape}")
		return torch.cat(features, 1)

class _Transition(nn.Sequential):
	def __init__(self, num_input_features: int, num_output_features: int) -> None:
		super().__init__()
		self.norm = nn.BatchNorm2d(num_input_features)
		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False) # 1x1 conv
		self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class DenseNet(nn.Module):
	def __init__(
		self,
		growth_rate: int = 32,
		block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
		num_init_features: int = 64,
		bn_size: int = 4,
		drop_rate: float = 0,
		num_classes: int = 1,
		init_weights: bool = False,
		drop_rate2: float = 0,
		last_fc_features_length: int = 512
	) -> None:
		super().__init__()

		# First convolution
		self.features = nn.Sequential(
			OrderedDict(
				[
					("conv0", nn.Conv2d(3, num_init_features, kernel_size=3, stride=(1,2), padding=1, bias=False)),
					("norm0", nn.BatchNorm2d(num_init_features)),
					("relu0", nn.ReLU(inplace=True)),
					("conv1", nn.Conv2d(num_init_features, num_init_features, kernel_size=3, padding=1, bias=False)),
					("norm1", nn.BatchNorm2d(num_init_features)),
					("relu1", nn.ReLU(inplace=True)),
					("conv2", nn.Conv2d(num_init_features, num_init_features, kernel_size=3, padding=1, bias=False)),
					("norm2", nn.BatchNorm2d(num_init_features)),
					("relu2", nn.ReLU(inplace=True)),
					("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
				]
			)
		)

		# Each denseblock
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			# print(f"num_layers: {num_layers}; num_input_features: {num_features}; bn_size: {bn_size}; growth_rate: {growth_rate}")
			block = _DenseBlock(
				num_layers=num_layers,
				num_input_features=num_features,
				bn_size=bn_size,
				growth_rate=growth_rate,
				drop_rate=drop_rate,
				index=i,
			)
			self.features.add_module("denseblock%d" % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			# print(f"i: {i}; num_features: {num_features}; num_layers: {num_layers}")
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
				self.features.add_module("transition%d" % (i + 1), trans)
				num_features = num_features // 2

		# Final batch norm
		self.features.add_module("norm5", nn.BatchNorm2d(num_features))

		# Linear layer
		self.fc1 = nn.Linear(num_features, last_fc_features_length)
		self.fc2 = nn.Linear(last_fc_features_length, num_classes)
		self.dropout = nn.Dropout(p=drop_rate2)

		# Official init from torch repo.
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.kaiming_normal_(m.weight)
		# 	elif isinstance(m, nn.BatchNorm2d):
		# 		nn.init.constant_(m.weight, 1)
		# 		nn.init.constant_(m.bias, 0)
		# 	elif isinstance(m, nn.Linear):
		# 		nn.init.constant_(m.bias, 0)

		if init_weights:
			self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
				if m.bias is not None:
					nn.init.constant_(m.bias, 0.000001)
			# elif isinstance(m, nn.BatchNorm2d):
			# 	nn.init.uniform_(m.weight, 0.000001, 1.0)
			# 	if m.bias is not None:
			# 		nn.init.constant_(m.bias, 0.000001)
			elif isinstance(m, nn.Linear):
				# nn.init.normal_(m.weight, 0, 0.000001)
				nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
				nn.init.constant_(m.bias, 0.000001)

	def forward(self, x: Tensor) -> Tensor:
		# print("input x shape:", x.shape)
		features = self.features(x)
		# print("features shape:", features.shape)
		out = F.relu(features, inplace=True)
		# print("out shape F.relu:", out.shape)
		out = F.adaptive_avg_pool2d(out, (1, 1))
		# print("out shape F.adaptive_avg_pool2d:", out.shape)
		out = torch.flatten(out, 1)
		# print("out shape torch.flatten:", out.shape)
		out = self.fc1(out)
		# print("out shape self.fc1:", out.shape)
		out = self.dropout(out)
		# print("out shape self.dropout:", out.shape)
		out = self.fc2(out)
		# print("out shape self.fc2:", out.shape)
		return out


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
	plt.savefig("regression_"+formatted_now+".png")
	# plt.show()

class CustomImageDataset(Dataset):
	def __init__(self, img_dir, csv_file, transform=None):
		self.img_labels = pd.read_csv(csv_file, header=None) # the first row is not a column name
		# print(f"len: {len(self.img_labels)}")
		self.img_dir = img_dir
		self.transform = transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, index):
		img_name = os.path.join(self.img_dir, self.img_labels.iloc[index, 2]) # column 3 is the image file name
		# print("image name:", img_name)
		image = Image.open(img_name) # RGB
		label = self.img_labels.iloc[index, 1] # column 2 is the regression value

		if self.transform:
			image = self.transform(image)

		# print(f"type: {type(image)}; {type(label)}")
		return image, label

def load_dataset(dataset_path, mean, std, batch_size):
	mean = ast.literal_eval(mean) # str to tuple
	std = ast.literal_eval(std)

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std), # RGB
	])

	train_dataset = CustomImageDataset(dataset_path+"/train", dataset_path+"/train.csv", transform)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	val_dataset = CustomImageDataset(dataset_path+"/val", dataset_path+"/val.csv", transform)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	# print(f"train dataset length: {len(train_dataset)}; val dataset length: {len(val_dataset)}")
	return len(train_dataset), len(val_dataset), train_loader, val_loader

def get_model_parameters(model):
	print("model:", model)

	total_params = sum(p.numel() for p in model.parameters())
	print(f"total parameters: {total_params}")
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"training parameters: {total_trainable_params}")

	tensor = torch.rand(1, 3, 64, 512) # (n,c,h,w)
	output = model(tensor)
	print(f"output: {output}; output.shape: {output.shape}; output.type: {output.type()}")
	raise ValueError(colorama.Fore.YELLOW + "for testing purposes")

def calculate_hit_rate(labels_src, labels_dst, length, threshold):
	assert len(labels_src) == len(labels_dst) and len(labels_src) == length, f"they must be of equal length: {len(labels_src)}, {len(labels_dst)}, {length}"

	count = 0
	for i in range(length):
		if abs(labels_src[i] - labels_dst[i]) < threshold:
			count = count + 1
	return float(count) / length

def train(dataset_path, epochs, mean, std, model_name, net, pretrained_model, batch_size, threshold, lr, drop_rate, drop_rate2, loss_delta, last_fc_features_length):
	train_dataset_num, val_dataset_num, train_loader, val_loader = load_dataset(dataset_path, mean, std, batch_size)

	if net == "resnet18":
		model = ResNet18(block=BasicBlock, init_weights=True)
	elif net == "densenet":
		# model = models.DenseNet(growth_rate=32, block_config=(6,12,24,16), num_init_features=64, num_classes=1)
		model = DenseNet(growth_rate=32, block_config=(6,12,24,16), num_init_features=64, num_classes=1, init_weights=True, drop_rate=drop_rate, drop_rate2=drop_rate2, last_fc_features_length=last_fc_features_length)
	if pretrained_model != "":
		model.load_state_dict(torch.load(pretrained_model, map_location="cpu"))
	# get_model_parameters(model)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8) # set the optimizer
	criterion = nn.HuberLoss(delta=loss_delta) # nn.MSELoss() # set the loss

	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	highest_accuracy_train = 0.
	minimum_loss_train = 100.
	highest_accuracy_val = 0.
	minimum_loss_val = 100.

	for epoch in range(epochs):
		epoch_start = time.time()

		train_loss = 0.0 # loss
		val_loss = 0.0
		labels_src = []
		labels_dst = []

		model.train() # set to training mode
		for _, (inputs, labels) in enumerate(train_loader):
			inputs = inputs.to(device)
			labels = labels.to(device, dtype=torch.float).view(-1,1)
			# print(f"type: {inputs.type()}; {labels.type()}")
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
		if minimum_loss_val > avg_val_loss:
			minimum_loss_val = avg_val_loss
		if highest_accuracy_train < avg_train_acc:
			highest_accuracy_train = avg_train_acc
		if minimum_loss_train > avg_train_loss:
			minimum_loss_train = avg_train_loss

		if highest_accuracy_train > 0.99 and highest_accuracy_val < 0.5:
			print(colorama.Fore.YELLOW + "overfitting")
			break

		# if avg_val_loss < 0.00001 and avg_val_acc > 0.99999:
		# 	print(colorama.Fore.YELLOW + "stop training early")
		# 	torch.save(model.state_dict(), model_name)
		# 	break

	print(f"train: loss:{minimum_loss_train:.6f}, acc:{highest_accuracy_train:.6f};  val: loss:{minimum_loss_val:.6f}, acc:{highest_accuracy_val:.6f}")
	# draw_graph(train_losses, train_accuracies, val_losses, val_accuracies)


def get_images_list(images_path):
	image_names = []

	p = Path(images_path)
	for subpath in p.rglob("*"):
		if subpath.is_file():
			image_names.append(subpath)
	return image_names

def get_ground_truth(csv_file):
	ground_truth = {}

	img_labels = pd.read_csv(csv_file, header=None)
	for i in range(len(img_labels)):
		img_name = img_labels.iloc[i, 2]
		label = img_labels.iloc[i, 1]

		ground_truth[img_name] = label
	return ground_truth

def predict(model_name, images_path, mean, std, net, threshold, csv_file, last_fc_features_length):
	image_names = get_images_list(images_path)
	assert len(image_names) != 0, "no images found"

	ground_truth = get_ground_truth(csv_file)
	# print(ground_truth)

	if net == "resnet18":
		model = ResNet18(block=BasicBlock)
	elif net == "densenet":
		model = DenseNet(growth_rate=32, block_config=(6,12,24,16), num_init_features=64, num_classes=1, last_fc_features_length=last_fc_features_length)
	model.load_state_dict(torch.load(model_name))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	mean = ast.literal_eval(mean) # str to tuple
	std = ast.literal_eval(std)

	print("image name\t\t\tground truth\t\tpredict value\t\tresult")
	count = 0

	model.eval()
	with torch.no_grad():
		for image_name in image_names:
			input_image = Image.open(image_name)
			preprocess = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std) # RGB
			])

			input_tensor = preprocess(input_image) # (c,h,w)
			input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model, (1,c,h,w)
			input_batch = input_batch.to(device)
			output = model(input_batch)

			value1 = output[0,0].item()
			value2 = ground_truth[os.path.basename(image_name)]
			result = 0
			if abs(value1 - value2) < threshold:
				result = 1
				count = count + 1
			print(f"{os.path.basename(image_name)}\t\t\t{value2:.2f}\t\t{value1:.4f}\t\t{result}")

	hit_rate = float(count) / len(image_names)
	print(f"total number of predict images: {len(image_names)}, hit rate: {hit_rate:.6f}")


def set_gpu(id):
	os.environ["CUDA_VISIBLE_DEVICES"] = id # set which graphics card to use: 0,1,2..., default is 0

	print("available gpus:", torch.cuda.device_count())
	print("current gpu device:", torch.cuda.current_device())


if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()
	set_gpu(args.gpu)

	if args.task == "split":
		# python test_regression_custom_cnn.py --task split --src_dataset_path ../../data/database/regression/FeO --dst_dataset_path datasets/regression --csv_file ../../data/database/regression/FeO.csv --resize (64,512) --ratios (0.9,0.05,0.05)
		split_regression_dataset(args.src_dataset_path, args.dst_dataset_path, args.csv_file, args.resize, args.fill_value, args.ratios)
	elif args.task == "train":
		# python test_regression_custom_cnn.py --task train --src_dataset_path datasets/regression --epochs 100 --mean (0.51105501,0.2900612,0.45467574) --std (0.27224947583159903,0.28995317923225,0.13527405631842085) --model_name best.pth --net resnet18 --batch_size 2 --lr 0.0005
		train(args.src_dataset_path, args.epochs, args.mean, args.std, args.model_name, args.net, args.pretrained_model, args.batch_size, args.threshold, args.lr, args.drop_rate, args.drop_rate2, args.loss_delta, args.last_fc_features_length)
	else: # predict
		# python test_regression_custom_cnn.py --task predict --predict_images_path datasets/regression/test --mean (0.51105501,0.2900612,0.45467574) --std (0.27224947583159903,0.28995317923225,0.13527405631842085) --model_name best.pth --net resnet18 --csv_file datasets/regression/test.csv
		predict(args.model_name, args.predict_images_path, args.mean, args.std, args.net, args.threshold, args.csv_file, args.last_fc_features_length)

	print(colorama.Fore.GREEN + "====== execution completed ======")
