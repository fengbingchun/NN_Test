import colorama
import argparse
import time
import ast
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
from utils import split_dataset
from typing import Type
from pathlib import Path
from PIL import Image
import torchvision.models as models

def parse_args():
	parser = argparse.ArgumentParser(description="Modified AlexNet/ResNet18/DenseNet image classification")
	parser.add_argument("--task", required=True, type=str, choices=["split", "train", "predict"], help="specify what kind of task")
	parser.add_argument("--src_dataset_path", type=str, help="source dataset path")
	parser.add_argument("--dst_dataset_path", type=str, help="the path of the destination dataset after split")
	parser.add_argument("--resize", default=(256,256), help="the size to which images are resized when split the dataset, if(0,0),no scaling is done")
	parser.add_argument("--ratios", default=(0.8,0.1,0.1), help="the ratio of split the data set(train set, validation set, test set), the test set can be 0, but their sum must be 1")
	parser.add_argument("--net", type=str, choices=["alexnet", "resnet18"], help="specifies which network to use for training and prediction")
	parser.add_argument("--epochs", type=int, help="number of training")
	parser.add_argument("--mean", type=str, help="the mean of the training set of images")
	parser.add_argument("--std", type=str, help="the standard deviation of the training set of images")
	parser.add_argument("--model_name", type=str, help="the model generated during training or the model loaded during prediction")
	parser.add_argument("--pretrained_model", type=str, default="", help="pretrained model loaded during training")
	parser.add_argument("--labels_file", type=str, help="one category per line, the format is: index class_name")
	parser.add_argument("--predict_images_path", type=str, help="predict images path")

	args = parser.parse_args()
	return args

class AlexNet(nn.Module):
	def __init__(self, num_classes: int = 1000, dropout: float = 0.5, init_weights: bool = False) -> None:
		super().__init__()
		# self.features = nn.Sequential( # 1: 37%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((4, 24))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(256 * 4 * 24, 4096),
		# 	nn.ReLU(inplace=True),
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(4096, 4096),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(4096, num_classes),
		# )

		# self.features = nn.Sequential( # 2: 35%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((4, 24))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(256 * 4 * 24, 4096),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(4096, num_classes),
		# )

		# self.features = nn.Sequential( # 3: 35%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((8, 48))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(128 * 8 * 48, 1024),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(1024, num_classes),
		# )

		# self.features = nn.Sequential( # 4: 34%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,1), stride=1, padding=(1,0)),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((8, 48))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(64 * 8 * 48, 512),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(512, num_classes),
		# )

		# self.features = nn.Sequential( # 5: 34%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,1), stride=1, padding=(1,0)),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((8, 48))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(64 * 8 * 48, 512),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(512, num_classes),
		# )

		# self.features = nn.Sequential( # 6: 39%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
		# 	nn.ReLU(inplace=True),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((4, 24))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(32 * 4 * 24, 256),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(256, num_classes),
		# )

		# self.features = nn.Sequential( # 7: 37%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1,3), stride=1, padding=(0,1)),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,3), stride=1, padding=(0,1)),
		# 	nn.ReLU(inplace=True),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(32 * 8 * 8, 256),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(256, num_classes),
		# )

		# self.features = nn.Sequential( # 8: 34%, overfitting
		# 	nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1,3), stride=1, padding=(0,1)),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,3), stride=1, padding=(0,1)),
		# 	nn.ReLU(inplace=True),
		# )
		# self.avgpool = nn.AdaptiveAvgPool2d((8, 48))
		# self.classifier = nn.Sequential(
		# 	nn.Dropout(p=dropout),
		# 	nn.Linear(32 * 8 * 48, 512),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(512, num_classes),
		# )

		self.features = nn.Sequential( # 9: 
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1,3), stride=1, padding=(0,1)),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,3), stride=1, padding=(0,1)),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
		)
		self.avgpool = nn.AdaptiveAvgPool2d((4, 8))
		self.classifier = nn.Sequential(
			nn.Dropout(p=dropout),
			nn.Linear(128 * 4 * 8, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, num_classes),
		)

		if init_weights:
			self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


# class BasicBlock(nn.Module):
# 	"""residual building block"""
# 	def __init__(
# 		self,
# 		in_channels: int,
# 		out_channels: int,
# 		stride: int = 1,
# 		downsample: nn.Module = None
# 	) -> None:
# 		super(BasicBlock, self).__init__()
# 		self.downsample = downsample
# 		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
# 		self.bn1 = nn.BatchNorm2d(out_channels)
# 		self.relu = nn.ReLU(inplace=True)
# 		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
# 		self.bn2 = nn.BatchNorm2d(out_channels)

# 	def forward(self, x: Tensor) -> Tensor:
# 		identity = x

# 		out = self.conv1(x)
# 		out = self.bn1(out)
# 		out = self.relu(out)

# 		out = self.conv2(out)
# 		out = self.bn2(out)

# 		if self.downsample is not None:
# 			identity = self.downsample(x)

# 		out += identity
# 		out = self.relu(out)

# 		return out

# class ResNet18(nn.Module):
# 	def __init__(
# 		self,
# 		block: Type[BasicBlock],
# 		num_classes: int = 1000,
# 		init_weights: bool = False
# 	) -> None:
# 		super(ResNet18, self).__init__()
# 		layers = [2, 2, 2, 2]
# 		self.in_channels = 64
# 		self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
# 		self.bn1 = nn.BatchNorm2d(self.in_channels)
# 		self.relu = nn.ReLU(inplace=True)
# 		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

# 		self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
# 		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
# 		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
# 		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

# 		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
# 		self.fc = nn.Linear(512, num_classes)

# 		if init_weights:
# 			self._init_weights()

# 	def _make_layer(
# 		self,
# 		block: Type[BasicBlock],
# 		out_channels: int,
# 		blocks: int,
# 		stride: int = 1
# 	) -> nn.Sequential:
# 		downsample = None
# 		if stride != 1:
# 			downsample = nn.Sequential(
# 				nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
# 				nn.BatchNorm2d(out_channels),
# 			)

# 		layers = []
# 		layers.append(block(self.in_channels, out_channels, stride, downsample))
# 		self.in_channels = out_channels

# 		for _ in range(1, blocks):
# 			layers.append(block(self.in_channels, out_channels))

# 		return nn.Sequential(*layers)

# 	def _init_weights(self):
# 		for m in self.modules():
# 			if isinstance(m, nn.Conv2d):
# 				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
# 				if m.bias is not None:
# 					nn.init.constant_(m.bias, 0.000001)
# 			elif isinstance(m, nn.BatchNorm2d):
# 				nn.init.uniform_(m.weight, 0.000001, 1.0)
# 				if m.bias is not None:
# 					nn.init.constant_(m.bias, 0.000001)
# 			elif isinstance(m, nn.Linear):
# 				nn.init.normal_(m.weight, 0, 0.000001)
# 				nn.init.constant_(m.bias, 0.000001)

# 	def forward(self, x: Tensor) -> Tensor:
# 		print("input x shape:", x.shape)
# 		x = self.conv1(x)
# 		print("self.conv1 shape:", x.shape)
# 		x = self.bn1(x)
# 		print("self.bn1(x) shape:", x.shape)
# 		x = self.relu(x)
# 		print("self.relu(x) shape:", x.shape)
# 		x = self.maxpool(x)
# 		print("self.maxpool(x) shape:", x.shape)

# 		x = self.layer1(x)
# 		print("self.layer1(x) shape:", x.shape)
# 		x = self.layer2(x)
# 		print("self.layer2(x) shape", x.shape)
# 		x = self.layer3(x)
# 		print("self.layer3(x) shape:", x.shape)
# 		x = self.layer4(x)
# 		# print("Dimensions of the last convolutional feature map:", x.shape)
# 		print("self.layer4(x) shape:", x.shape)

# 		x = self.avgpool(x)
# 		print("self.avgpool(x) shape:", x.shape)
# 		x = torch.flatten(x, 1)
# 		print("torch.flatten(x,1) shape:", x.shape)
# 		x = self.fc(x)
# 		print("self.fc(x) shape:", x.shape)
# 		raise

# 		return x

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
		num_classes: int = 1000,
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
		self.fc = nn.Linear(512, num_classes)

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
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.uniform_(m.weight, 0.000001, 1.0)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0.000001)
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

	plt.show()


def write_labels(class_to_idx, labels_file):
	# print("class_to_idx:", class_to_idx)
	with open(labels_file, "w") as file:
		for key, val in class_to_idx.items():
			file.write("%d %s\n" % (int(val), key))

def load_dataset(dataset_path, mean, std, labels_file):
	mean = ast.literal_eval(mean) # str to tuple
	std = ast.literal_eval(std)
	# print(f"type: {type(mean)}, {type(std)}")

	train_transform = transforms.Compose([
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std), # RGB
	])

	train_dataset = ImageFolder(root=dataset_path+"/train", transform=train_transform)
	print(f"train dataset length: {len(train_dataset)}; classes: {train_dataset.class_to_idx}; number of categories: {len(train_dataset.class_to_idx)}")

	train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

	val_transform = transforms.Compose([
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std), # RGB
	])

	val_dataset = ImageFolder(root=dataset_path+"/val", transform=val_transform)
	print(f"val dataset length: {len(val_dataset)}; classes: {val_dataset.class_to_idx}")
	assert len(train_dataset.class_to_idx) == len(val_dataset.class_to_idx), f"the number of categories int the train set must be equal to the number of categories in the validation set: {len(train_dataset.class_to_idx)} : {len(val_dataset.class_to_idx)}"

	val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=0)

	write_labels(train_dataset.class_to_idx, labels_file)

	return len(train_dataset.class_to_idx), len(train_dataset), len(val_dataset), train_loader, val_loader

def get_model_parameters(model):
	print("model:", model)

	total_params = sum(p.numel() for p in model.parameters())
	print(f"total parameters: {total_params}")
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"training parameters: {total_trainable_params}")

	tensor = torch.rand(1, 3, 64, 512) # (n,c,h,w)
	output = model(tensor)
	raise ValueError(colorama.Fore.YELLOW + "for testing purposes")

def train(dataset_path, epochs, mean, std, model_name, labels_file, net, pretrained_model):
	classes_num, train_dataset_num, val_dataset_num, train_loader, val_loader = load_dataset(dataset_path, mean, std, labels_file)

	if net == "alexnet":
		model = AlexNet(num_classes=classes_num, init_weights=True)
	else:
		model = ResNet18(block=BasicBlock, num_classes=classes_num, init_weights=True)
	if pretrained_model != "":
		model.load_state_dict(torch.load(pretrained_model))
	# if net == "resnet18":
	# 	model = models.ResNet(block=models.resnet.BasicBlock, layers=[2,2,2,2], num_classes=classes_num)
	# get_model_parameters(model)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=0.0001) # set the optimizer
	criterion = nn.CrossEntropyLoss() #nn.HuberLoss() # nn.CrossEntropyLoss() # set the loss

	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	highest_accuracy = 0.
	minimum_loss = 100.

	for epoch in range(epochs):
		epoch_start = time.time()

		train_loss = 0.0 # loss
		train_acc = 0.0 # accuracy
		val_loss = 0.0
		val_acc = 0.0

		model.train() # set to training mode
		for i, (inputs, labels) in enumerate(train_loader):
			inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad() # clean existing gradients
			outputs = model(inputs) # forward pass
			loss = criterion(outputs, labels) # compute loss
			loss.backward() # backpropagate the gradients
			optimizer.step() # update the parameters

			train_loss += loss.item() * inputs.size(0) # compute the total loss
			_, predictions = torch.max(outputs.data, 1) # compute the accuracy
			correct_counts = predictions.eq(labels.data.view_as(predictions))
			acc = torch.mean(correct_counts.type(torch.FloatTensor)) # convert correct_counts to float
			train_acc += acc.item() * inputs.size(0) # compute the total accuracy
			# print(f"train batch number: {i}; train loss: {loss.item():.4f}; accuracy: {acc.item():.4f}")

		model.eval() # set to evaluation mode
		with torch.no_grad():
			for i, (inputs, labels) in enumerate(val_loader):
				inputs = inputs.to(device)
				labels = labels.to(device)

				outputs = model(inputs) # forward pass
				loss = criterion(outputs, labels) # compute loss
				val_loss += loss.item() * inputs.size(0) # compute the total loss
				_, predictions = torch.max(outputs.data, 1) # compute validation accuracy
				correct_counts = predictions.eq(labels.data.view_as(predictions))
				acc = torch.mean(correct_counts.type(torch.FloatTensor)) # convert correct_counts to float
				val_acc += acc.item() * inputs.size(0) # compute the total accuracy
				# print(f"val batch number: {i}; validation loss: {loss.item():.4f}; accuracy: {acc.item():.4f}")

		avg_train_loss = train_loss / train_dataset_num # average training loss
		avg_train_acc = train_acc / train_dataset_num # average training accuracy
		avg_val_loss = val_loss / val_dataset_num # average validation loss
		avg_val_acc = val_acc / val_dataset_num # average validation accuracy
		train_losses.append(avg_train_loss)
		train_accuracies.append(avg_train_acc)
		val_losses.append(avg_val_loss)
		val_accuracies.append(avg_val_acc)
		epoch_end = time.time()
		print(f"epoch:{epoch+1}/{epochs}; train loss:{avg_train_loss:.6f}, accuracy:{avg_train_acc:.6f}; validation loss:{avg_val_loss:.6f}, accuracy:{avg_val_acc:.6f}; time:{epoch_end-epoch_start:.2f}s")

		if highest_accuracy < avg_val_acc and minimum_loss > avg_val_loss:
			torch.save(model.state_dict(), model_name)
			highest_accuracy = avg_val_acc
			minimum_loss = avg_val_loss

		if avg_val_loss < 0.00001 and avg_val_acc > 0.99999:
			print(colorama.Fore.YELLOW + "stop training early")
			torch.save(model.state_dict(), model_name)
			break

		if avg_val_acc > 0.98:
			torch.save(model.state_dict(), model_name)
			break

	# draw_graph(train_losses, train_accuracies, val_losses, val_accuracies)

def parse_labels_file(labels_file):
	classes = {}

	with open(labels_file, "r") as file:
		for line in file:
			# print(f"line: {line}")
			idx_value = []
			for v in line.split(" "):
				idx_value.append(v.replace("\n", "")) # remove line breaks(\n) at the end of the line
			assert len(idx_value) == 2, f"the length must be 2: {len(idx_value)}"
			classes[int(idx_value[0])] = idx_value[1]

	# print(f"clases: {classes}; length: {len(classes)}")
	return classes

def get_images_list(images_path):
	image_names = []

	p = Path(images_path)
	for subpath in p.rglob("*"):
		if subpath.is_file():
			image_names.append(subpath)

	return image_names

def predict(model_name, labels_file, images_path, mean, std, net):
	classes = parse_labels_file(labels_file)
	assert len(classes) != 0, "the number of categories can't be 0"

	image_names = get_images_list(images_path)
	assert len(image_names) != 0, "no images found"

	mean = ast.literal_eval(mean) # str to tuple
	std = ast.literal_eval(std)

	if net == "alexnet":
		model = AlexNet(num_classes=len(classes))
	else:
		model = ResNet18(block=BasicBlock, num_classes=len(classes))
	model.load_state_dict(torch.load(model_name))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	print("image name\t\t\t\t\t\tclass\tprobability")

	model.eval()
	with torch.no_grad():
		for image_name in image_names:
			input_image = Image.open(image_name)
			preprocess = transforms.Compose([
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std) # RGB
			])

			input_tensor = preprocess(input_image) # (c,h,w)
			input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model, (1,c,h,w)
			input_batch = input_batch.to(device)

			output = model(input_batch)
			# print(f"output.shape: {output.shape}")
			probabilities = torch.nn.functional.softmax(output[0], dim=0) # the output has unnormalized scores, to get probabilities, you can run a softmax on it
			max_value, max_index = torch.max(probabilities, dim=0)
			print(f"{image_name}\t\t\t\t\t\t{classes[max_index.item()]}\t{max_value.item():.4f}")


if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "split":
		# python test_classify_custom_cnn.py --task split --src_dataset_path ../../data/database/classify/melon --dst_dataset_path datasets/melon_new_classify --resize (32,192) --ratios (0.9,0.05,0.05)
		split_dataset(args.src_dataset_path, args.dst_dataset_path, args.resize, args.ratios)
	elif args.task == "train":
		# python test_classify_custom_cnn.py --task train --src_dataset_path datasets/melon_new_classify --epochs 100 --mean (0.53087615,0.23997033,0.45703197) --std (0.29807151489753686,0.3128615049442739,0.15151863355831655) --labels_file classes.txt --model_name best.pth --net alexnet
		# 64x512: --mean (0.51225255,0.29016045,0.45465541) --std (0.2718249152287802,0.28992280900329326,0.1353453804698747)
		train(args.src_dataset_path, args.epochs, args.mean, args.std, args.model_name, args.labels_file, args.net, args.pretrained_model)
	else: # predict
		# python test_classify_custom_cnn.py --task predict --predict_images_path datasets/melon_new_classify/test --mean (0.53087615,0.23997033,0.45703197) --std (0.29807151489753686,0.3128615049442739,0.15151863355831655) --labels_file classes.txt --model_name best.pth --net alexnet
		predict(args.model_name, args.labels_file, args.predict_images_path, args.mean, args.std, args.net)

	print(colorama.Fore.GREEN + "====== execution completed ======")
