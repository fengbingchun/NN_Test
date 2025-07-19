import argparse
import colorama
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch_pruning as tp
from pathlib import Path
import ast
from PIL import Image
import torch.optim as optim
import time

# Blog: https://blog.csdn.net/fengbingchun/article/details/149468652

def parse_args():
	parser = argparse.ArgumentParser(description="model pruning: torch-pruning densenet")
	parser.add_argument("--task", required=True, type=str, choices=["prune", "fine-tuning", "predict"], help="specify what kind of task")
	parser.add_argument("--model_name", type=str, help="the model generated during training")
	parser.add_argument("--classes_number", type=int, default=2, help="classes number")
	parser.add_argument("--prune_amount", type=float, default=0.3, help="prune amount")
	parser.add_argument("--mean", type=str, help="the mean of the training set of images")
	parser.add_argument("--std", type=str, help="the standard deviation of the training set of images")
	parser.add_argument("--labels_file", type=str, help="one category per line, the format is: index class_name")
	parser.add_argument("--images_path", type=str, help="predict images path")
	parser.add_argument("--epochs", type=int, default=1000, help="number of training")
	parser.add_argument("--dataset_path", type=str, help="source dataset path")

	args = parser.parse_args()
	return args

def _str2tuple(value):
	if not isinstance(value, tuple):
		value = ast.literal_eval(value) # str to tuple
	return value

def model_pruning(model_name, classes_number, prune_amount):
	# https://github.com/VainF/Torch-Pruning/blob/master/examples/torchvision_models/torchvision_global_pruning.py
	model = models.densenet121(weights=None)
	model.classifier = nn.Linear(model.classifier.in_features, classes_number)
	# print("before pruning, model:", model)

	model.load_state_dict(torch.load(model_name, weights_only=False, map_location="cpu"))

	orininal_size = tp.utils.count_params(model)

	model.cpu().eval()

	for p in model.parameters():
		p.requires_grad_(True)

	ignored_layers = []
	for m in model.modules():
		if isinstance(m, nn.Linear):
			ignored_layers.append(m)
	print(f"ignored_layers: {ignored_layers}")

	example_inputs = torch.randn(1, 3, 224, 224)

	# build network pruners
	importance = tp.importance.MagnitudeImportance(p=1)
	pruner = tp.pruner.MagnitudePruner(
		model,
		example_inputs=example_inputs,
		importance=importance,
		iterative_steps=1,
		pruning_ratio=prune_amount,
		global_pruning=True,
		round_to=None,
		unwrapped_parameters=None,
		ignored_layers=ignored_layers,
		channel_groups={}
	)

	# pruning
	layer_channel_cfg = {}
	for module in model.modules():
		if module not in pruner.ignored_layers:
			if isinstance(module, nn.Conv2d):
				layer_channel_cfg[module] = module.out_channels
			elif isinstance(module, nn.Linear):
				layer_channel_cfg[module] = module.out_features

	pruner.step()
	# print("after pruning, model", model)

	result_size = tp.utils.count_params(model)
	print(f"model: original size: {orininal_size}; result_size: {result_size}")

	# testing
	with torch.no_grad():
		out = model(example_inputs)
		print("test out:", out)

	torch.save(model, "new_structured_prune_melon_classify.pt") # cann't bu used: torch.save(model.state_dict(), "")

def _load_dataset(dataset_path, mean, std, batch_size):
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

	return len(train_dataset), len(val_dataset), train_loader, val_loader

def fine_tuning(dataset_path, epochs, mean, std, model_name):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = torch.load(model_name, weights_only=False)
	model.to(device)

	train_dataset_num, val_dataset_num, train_loader, val_loader = _load_dataset(dataset_path, mean, std, 4)

	optimizer = optim.Adam(model.parameters(), lr=0.00001) # set the optimizer
	criterion = nn.CrossEntropyLoss() # set the loss

	highest_accuracy = 0.
	minimum_loss = 100.
	new_model_name = "fine_tuning_melon_classify.pt"

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
			torch.save(model, new_model_name)
			highest_accuracy = avg_val_acc
			minimum_loss = avg_val_loss

		if avg_val_loss < 0.0001 or avg_val_acc > 0.9999:
			print(colorama.Fore.YELLOW + "stop training early")
			torch.save(model, new_model_name)
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

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = torch.load(model_name, weights_only=False)
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

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "prune":
		model_pruning(args.model_name, args.classes_number, args.prune_amount)
	elif args.task == "fine-tuning":
		fine_tuning(args.dataset_path, args.epochs, args.mean, args.std, args.model_name)
	else:
		predict(args.model_name, args.labels_file, args.images_path, args.mean, args.std)

	print(colorama.Fore.GREEN + "====== execution completed ======")
