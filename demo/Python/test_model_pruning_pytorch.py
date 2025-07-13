import argparse
import colorama
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from utils import SplitClassifyDataset
import ast
import time
from pathlib import Path

# Blog: https://blog.csdn.net/fengbingchun/article/details/149307432

def parse_args():
	parser = argparse.ArgumentParser(description="model pruning: pytorch densenet")
	parser.add_argument("--task", required=True, type=str, choices=["split", "train", "predict", "prune"], help="specify what kind of task")
	parser.add_argument("--src_dataset_path", type=str, help="source dataset path")
	parser.add_argument("--dst_dataset_path", type=str, help="the path of the destination dataset after split")
	parser.add_argument("--resize", default=(224,224), help="the size to which images are resized when split the dataset, if(0,0),no scaling is done")
	parser.add_argument("--ratios", default=(0.8,0.1,0.1), help="the ratio of split the data set(train set, validation set, test set), the test set can be 0, but their sum must be 1")
	parser.add_argument("--epochs", type=int, default=1000, help="number of training")
	parser.add_argument("--mean", type=str, help="the mean of the training set of images")
	parser.add_argument("--std", type=str, help="the standard deviation of the training set of images")
	parser.add_argument("--model_name", type=str, help="the model generated during training or the model loaded during prediction")
	parser.add_argument("--labels_file", type=str, help="one category per line, the format is: index class_name")
	parser.add_argument("--images_path", type=str, help="predict images path")
	parser.add_argument("--prune_type", type=str, choices=["unstructured", "structured"], help="prune type")
	parser.add_argument("--prune_amount", type=float, default=0.3, help="prune amount")

	args = parser.parse_args()
	return args

def _str2tuple(value):
	if not isinstance(value, tuple):
		value = ast.literal_eval(value) # str to tuple
	return value

def split_dataset(src_dataset_path, dst_dataset_path, resize, ratios):
	split = SplitClassifyDataset(path_src=src_dataset_path, path_dst=dst_dataset_path, ratios=_str2tuple(ratios))

	if resize != "(0,0)":
		split.resize(shape=_str2tuple(resize))

	split()
	mean, std = split.get_mean_std()
	print(f"mean: {mean}; std: {std}")

def _write_labels(class_to_idx, labels_file):
	with open(labels_file, "w") as file:
		for key, val in class_to_idx.items():
			file.write("%d %s\n" % (int(val), key))

def _load_dataset(dataset_path, mean, std, labels_file, batch_size):
	mean = _str2tuple(mean)
	std = _str2tuple(std)

	train_transform = transforms.Compose([
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std), # RGB
	])

	train_dataset = ImageFolder(root=dataset_path+"/train", transform=train_transform)
	print(f"train dataset length: {len(train_dataset)}; classes: {train_dataset.class_to_idx}; number of categories: {len(train_dataset.class_to_idx)}")

	train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=0)

	val_transform = transforms.Compose([
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std), # RGB
	])

	val_dataset = ImageFolder(root=dataset_path+"/val", transform=val_transform)
	print(f"val dataset length: {len(val_dataset)}; classes: {val_dataset.class_to_idx}")
	assert len(train_dataset.class_to_idx) == len(val_dataset.class_to_idx), f"the number of categories int the train set must be equal to the number of categories in the validation set: {len(train_dataset.class_to_idx)} : {len(val_dataset.class_to_idx)}"

	val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=0)

	_write_labels(train_dataset.class_to_idx, labels_file)

	return len(train_dataset.class_to_idx), len(train_dataset), len(val_dataset), train_loader, val_loader

def _get_model_parameters(model):
	print("model:", model)

	total_params = sum(p.numel() for p in model.parameters())
	print(f"total parameters: {total_params}")
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"training parameters: {total_trainable_params}")

	tensor = torch.rand(1, 3, 224, 224)
	output = model(tensor)
	raise ValueError(colorama.Fore.YELLOW + "for testing purposes")

def train(dataset_path, epochs, mean, std, model_name, labels_file):
	classes_num, train_dataset_num, val_dataset_num, train_loader, val_loader = _load_dataset(dataset_path, mean, std, labels_file, 16)

	model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT) # densenet121-a639ec97.pth
	model.classifier = nn.Linear(model.classifier.in_features, classes_num)
	# _get_model_parameters(model)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=0.00001) # set the optimizer
	criterion = nn.CrossEntropyLoss() # set the loss

	highest_accuracy = 0.
	minimum_loss = 100.

	for epoch in range(epochs):
		epoch_start = time.time()

		train_loss = 0.0
		train_acc = 0.0
		val_loss = 0.0
		val_acc = 0.0

		model.train() # set to training mode
		for _, (inputs, labels) in enumerate(train_loader):
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
			for _, (inputs, labels) in enumerate(val_loader):
				inputs = inputs.to(device)
				labels = labels.to(device)

				outputs = model(inputs) # forward pass
				loss = criterion(outputs, labels) # compute loss
				val_loss += loss.item() * inputs.size(0) # compute the total loss
				_, predictions = torch.max(outputs.data, 1) # compute validation accuracy
				correct_counts = predictions.eq(labels.data.view_as(predictions))
				acc = torch.mean(correct_counts.type(torch.FloatTensor)) # convert correct_counts to float
				val_acc += acc.item() * inputs.size(0) # compute the total accuracy

		avg_train_loss = train_loss / train_dataset_num # average training loss
		avg_train_acc = train_acc / train_dataset_num # average training accuracy
		avg_val_loss = val_loss / val_dataset_num # average validation loss
		avg_val_acc = val_acc / val_dataset_num # average validation accuracy

		epoch_end = time.time()
		print(f"epoch:{epoch+1}/{epochs}; train loss:{avg_train_loss:.6f}, accuracy:{avg_train_acc:.6f}; validation loss:{avg_val_loss:.6f}, accuracy:{avg_val_acc:.6f}; time:{epoch_end-epoch_start:.2f}s")

		if highest_accuracy < avg_val_acc and minimum_loss > avg_val_loss:
			torch.save(model.state_dict(), model_name)
			highest_accuracy = avg_val_acc
			minimum_loss = avg_val_loss

		if avg_val_loss < 0.0001 or avg_val_acc > 0.9999:
			print(colorama.Fore.YELLOW + "stop training early")
			torch.save(model.state_dict(), model_name)
			break

def _parse_labels_file(labels_file):
	classes = {}

	with open(labels_file, "r") as file:
		for line in file:
			idx_value = []
			for v in line.split(" "):
				idx_value.append(v.replace("\n", "")) # remove line breaks(\n) at the end of the line
			assert len(idx_value) == 2, f"the length must be 2: {len(idx_value)}"
			classes[int(idx_value[0])] = idx_value[1]

	return classes

def _get_images_list(images_path):
	image_names = []

	p = Path(images_path)
	for subpath in p.rglob("*"):
		if subpath.is_file():
			image_names.append(subpath)

	return image_names

def predict(model_name, labels_file, images_path, mean, std):
	classes = _parse_labels_file(labels_file)
	assert len(classes) != 0, "the number of categories can't be 0"

	image_names = _get_images_list(images_path)
	assert len(image_names) != 0, "no images found"

	mean = _str2tuple(mean)
	std = _str2tuple(std)

	model = models.densenet121(weights=None)
	model.classifier = nn.Linear(model.classifier.in_features, len(classes))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.load_state_dict(torch.load(model_name, weights_only=False, map_location="cpu"))
	model.to(device)

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
			probabilities = torch.nn.functional.softmax(output[0], dim=0) # the output has unnormalized scores, to get probabilities, you can run a softmax on it
			max_value, max_index = torch.max(probabilities, dim=0)
			print(f"{image_name.name}\t{classes[max_index.item()]}\t{max_value.item():.4f}")

def _remove_pruned_weights(model):
	...

def model_pruning(model_name, labels_file, prune_type, prune_amount):
	classes = _parse_labels_file(labels_file)
	assert len(classes) != 0, "the number of categories can't be 0"

	model = models.densenet121(weights=None)
	model.classifier = nn.Linear(model.classifier.in_features, len(classes))
	model.load_state_dict(torch.load(model_name, weights_only=False, map_location="cpu"))
	model.eval()
	# _get_model_parameters(model)

	if prune_type == "structured":
		new_model_name = "structured_prune_melon_classify.pt"
		transition_conv = model.features.transition1.conv
		prune.ln_structured(transition_conv, name="weight", amount=prune_amount, n=1, dim=0) # n=1: L1 norm; n=2: L2 norm
		prune.remove(transition_conv, "weight")

		dense_conv = model.features.denseblock4.denselayer1.conv1
		prune.random_structured(dense_conv, name="weight", amount=prune_amount, dim=0)
		prune.remove(dense_conv,"weight")
	else:
		new_model_name = "unstructured_prune_melon_classify.pt"
		parameters_to_prune = [(module, "weight") for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
		prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount)
		for module, _ in parameters_to_prune:
			prune.remove(module, "weight")

	# model = _remove_pruned_weights(model)
	torch.save(model.state_dict(), new_model_name)

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "split":
		split_dataset(args.src_dataset_path, args.dst_dataset_path, args.resize, args.ratios)
	elif args.task == "train":
		train(args.src_dataset_path, args.epochs, args.mean, args.std, args.model_name, args.labels_file)
	elif args.task == "predict":
		predict(args.model_name, args.labels_file, args.images_path, args.mean, args.std)
	else:
		model_pruning(args.model_name, args.labels_file, args.prune_type, args.prune_amount)

	print(colorama.Fore.GREEN + "====== execution completed ======")
