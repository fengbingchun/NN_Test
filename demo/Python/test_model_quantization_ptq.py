import argparse
import colorama
import ast
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.ao.quantization as quantization
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
import copy
from PIL import Image
from pathlib import Path

# Blog: https://blog.csdn.net/fengbingchun/article/details/150703580

def parse_args():
	parser = argparse.ArgumentParser(description="model quantization")
	parser.add_argument("--task", required=True, type=str, choices=["quantize", "predict"], help="specify what kind of task")
	parser.add_argument("--src_model", type=str, help="source model name")
	parser.add_argument("--dst_model", type=str, help="quantized model name")
	parser.add_argument("--classes_number", type=int, default=2, help="classes number")
	parser.add_argument("--mean", type=str, help="the mean of the training set of images")
	parser.add_argument("--std", type=str, help="the standard deviation of the training set of images")
	parser.add_argument("--labels_file", type=str, help="one category per line, the format is: index class_name")
	parser.add_argument("--images_path", type=str, help="predict images path")
	parser.add_argument("--dataset_path", type=str, help="source dataset path")

	args = parser.parse_args()
	return args

def _str2tuple(value):
	if not isinstance(value, tuple):
		value = ast.literal_eval(value) # str to tuple
	return value

def _load_dataset(dataset_path, mean, std, batch_size):
	mean = _str2tuple(mean)
	std = _str2tuple(std)

	calibration_transform = transforms.Compose([
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std), # RGB
	])

	calibration_dataset = ImageFolder(root=dataset_path+"/train", transform=calibration_transform)
	print(f"calibration dataset length: {len(calibration_dataset)}; classes: {calibration_dataset.class_to_idx}; number of categories: {len(calibration_dataset.class_to_idx)}")

	calibration_loader = DataLoader(calibration_dataset, batch_size, shuffle=True, num_workers=0)
	return calibration_loader

def _calibrate(model, calibration_loader, num_batches=20):
	model.eval()

	with torch.no_grad():
		for i, (x, _) in enumerate(calibration_loader):
			x = x.to(torch.device("cpu"))
			_ = model(x)
			if i + 1 >= num_batches:
				break

def quantize(src_model, device, classes_number, dataset_path, mean, std, dst_model):
	# load model
	model = models.densenet121(weights=None)
	model.classifier = nn.Linear(model.classifier.in_features, classes_number)
	model.load_state_dict(torch.load(src_model, weights_only=False, map_location="cpu"))
	model.to(device)
	model.eval()

	# prepare quantization: fx
	qconfig_mapping = quantization.get_default_qconfig_mapping('x86')
	model_prepared = prepare_fx(copy.deepcopy(model), qconfig_mapping, example_inputs=torch.randn(1, 3, 224, 224))
	model_prepared.eval()

	# load dataset
	calibration_loader = _load_dataset(dataset_path, mean, std, 4)

	# calibration
	_calibrate(model_prepared, calibration_loader)

	# quantize: INT8
	quantized_model = convert_fx(model_prepared)
	quantized_model.eval()

	# save model
	scripted_model = torch.jit.script(quantized_model)
	scripted_model.save(dst_model)

def _get_images_list(images_path):
	image_names = []

	p = Path(images_path)
	for subpath in p.rglob("*"):
		if subpath.is_file():
			image_names.append(subpath)

	return image_names

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

def predict(model_name, device, labels_file, images_path, mean, std):
	model = torch.jit.load(model_name, map_location="cpu")
	model.to(device)
	model.eval()

	mean = _str2tuple(mean)
	std = _str2tuple(std)

	image_names = _get_images_list(images_path)
	assert len(image_names) != 0, "no images found"

	classes = _parse_labels_file(labels_file)
	assert len(classes) != 0, "the number of categories can't be 0"

	model.eval()
	with torch.no_grad():
		for image_name in image_names:
			input_image = Image.open(image_name)
			preprocess = transforms.Compose([
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std)
			])

			input_tensor = preprocess(input_image) # (c,h,w)
			input_batch = input_tensor.unsqueeze(0) # (1,c,h,w)
			input_batch = input_batch.to(device)

			output = model(input_batch)
			probabilities = torch.nn.functional.softmax(output[0], dim=0)
			max_value, max_index = torch.max(probabilities, dim=0)
			print(f"{image_name.name}\t{classes[max_index.item()]}\t{max_value.item():.4f}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	device = torch.device("cpu")

	if args.task == "quantize":
		quantize(args.src_model, device, args.classes_number, args.dataset_path, args.mean, args.std, args.dst_model)
	elif args.task == "predict":
		predict(args.dst_model, device, args.labels_file, args.images_path, args.mean, args.std)

	print(colorama.Fore.GREEN + "====== execution completed ======")
