import colorama
import argparse
import os
import time
import ast
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from utils import SplitClassifyDataset
import matplotlib.pyplot as plt
from pathlib import Path

# Blog: https://blog.csdn.net/fengbingchun/article/details/141635132

def parse_args():
	parser = argparse.ArgumentParser(description="AlexNet image classification")
	parser.add_argument("--task", required=True, type=str, choices=["split", "train", "predict"], help="specify what kind of task")
	parser.add_argument("--src_dataset_path", type=str, help="source dataset path")
	parser.add_argument("--dst_dataset_path", type=str, help="the path of the destination dataset after split")
	parser.add_argument("--resize", default=(256,256), help="the size to which images are resized when split the dataset")
	parser.add_argument("--ratios", default=(0.8,0.1,0.1), help="the ratio of split the data set(train set, validation set, test set), the test set can be 0, but their sum must be 1")
	parser.add_argument("--epochs", type=int, help="number of training")
	parser.add_argument("--mean", type=str, help="the mean of the training set of images")
	parser.add_argument("--std", type=str, help="the standard deviation of the training set of images")
	parser.add_argument("--model_name", type=str, help="the model generated during training or the model loaded during prediction")
	parser.add_argument("--labels_file", type=str, help="one category per line, the format is: index class_name")
	parser.add_argument("--predict_images_path", type=str, help="predict images path")

	args = parser.parse_args()
	return args


def split_dataset(src_dataset_path, dst_dataset_path, resize, ratios):
	split = SplitClassifyDataset(path_src=src_dataset_path, path_dst=dst_dataset_path, ratios=ast.literal_eval(ratios))

	# print("resize:", type(ast.literal_eval(resize))) # str to tuple
	split.resize(shape=ast.literal_eval(resize))

	split()
	mean, std = split.get_mean_std()
	print(f"mean: {mean}; std: {std}")


def load_pretraind_model():
	model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1) # the first execution will download model: alexnet-owt-7be5be79.pth, pos: C:\Users\xxxxxx/.cache\torch\hub\checkpoints\alexnet-owt-7be5be79.pth
	# print("model:", model)

	return model

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
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])

	train_dataset = ImageFolder(root=dataset_path+"/train", transform=train_transform)
	print(f"train dataset length: {len(train_dataset)}; classes: {train_dataset.class_to_idx}; number of categories: {len(train_dataset.class_to_idx)}")

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)

	val_transform = transforms.Compose([
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std),
	])

	val_dataset = ImageFolder(root=dataset_path+"/val", transform=val_transform)
	print(f"val dataset length: {len(val_dataset)}; classes: {val_dataset.class_to_idx}")
	assert len(train_dataset.class_to_idx) == len(val_dataset.class_to_idx), f"the number of categories int the train set must be equal to the number of categories in the validation set: {len(train_dataset.class_to_idx)} : {len(val_dataset.class_to_idx)}"

	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)

	write_labels(train_dataset.class_to_idx, labels_file)

	return len(train_dataset.class_to_idx), len(train_dataset), len(val_dataset), train_loader, val_loader

def train(dataset_path, epochs, mean, std, model_name, labels_file):
	classes_num, train_dataset_num, val_dataset_num, train_loader, val_loader = load_dataset(dataset_path, mean, std, labels_file)

	model = load_pretraind_model()

	in_features = model.classifier[6].in_features
	# print(f"in_features: {in_features}")
	model.classifier[6] = nn.Linear(in_features, classes_num) # modify the number of categories

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=0.0002) # set the optimizer
	criterion = nn.CrossEntropyLoss() # set the loss

	train_losses = []
	train_accuracies = []
	val_losses = []
	val_accuracies = []

	highest_accuracy = 0.
	minimum_loss = 100.

	for epoch in range(epochs):
		# reference: https://learnopencv.com/image-classification-using-transfer-learning-in-pytorch/
		epoch_start = time.time()
		# print(colorama.Fore.CYAN + f"epoch: {epoch+1}/{epochs}")

		train_loss = 0.0 # loss
		train_acc = 0.0 # accuracy
		val_loss = 0.0
		val_acc = 0.0

		model.train() # set to training mode
		for i, (inputs, labels) in enumerate(train_loader):
			inputs = inputs.to(device)
			labels = labels.to(device)
			# print("inputs.size(0):", inputs.size(0))

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
		print(f"epoch:{epoch+1}/{epochs}; train loss:{avg_train_loss:.4f}, accuracy:{avg_train_acc:.4f}; validation loss:{avg_val_loss:.4f}, accuracy:{avg_val_acc:.4f}; time:{epoch_end-epoch_start:.2f}s")

		if highest_accuracy < avg_val_acc and minimum_loss > avg_val_loss:
			torch.save(model.state_dict(), model_name)
			highest_accuracy = avg_val_acc
			minimum_loss = avg_val_loss

		if avg_val_loss < 0.00001 and avg_val_acc > 0.99999:
			print(colorama.Fore.YELLOW + "stop training early")
			torch.save(model.state_dict(), model_name)
			break

	draw_graph(train_losses, train_accuracies, val_losses, val_accuracies)

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

def save_features(model, input_batch, image_name):
	features = model.features(input_batch) # shape: torch.Size([1, 256, 6, 6])
	features = model.avgpool(features)
	features = torch.flatten(features, 1) # shape: torch.Size([1, 9216])

	if torch.cuda.is_available():
		features = features.squeeze().detach().cpu().numpy() # shape: (9216,)
	else:
		features = features.queeeze().detach().numpy()
	# print(f"features: {features}; shape: {features.shape}")

	dir_name = "tmp"
	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	file_name = Path(image_name)
	file_name = file_name.name
	# print(f"file name: {file_name}")
	features.tofile(dir_name+"/"+file_name+".bin")

def predict(model_name, labels_file, images_path, mean, std):
	classes = parse_labels_file(labels_file)
	assert len(classes) != 0, "the number of categories can't be 0"

	image_names = get_images_list(images_path)
	assert len(image_names) != 0, "no images found"

	mean = ast.literal_eval(mean) # str to tuple
	std = ast.literal_eval(std)

	model = models.alexnet(weights=None)
	in_features = model.classifier[6].in_features
	model.classifier[6] = nn.Linear(in_features, len(classes)) # modify the number of categories
	# print("alexnet model:", model)
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
				transforms.Normalize(mean=mean, std=std)
			])

			input_tensor = preprocess(input_image) # (c,h,w)
			input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model, (1,c,h,w)
			input_batch = input_batch.to(device)

			output = model(input_batch)
			# print(f"output.shape: {output.shape}")
			probabilities = torch.nn.functional.softmax(output[0], dim=0) # the output has unnormalized scores, to get probabilities, you can run a softmax on it
			max_value, max_index = torch.max(probabilities, dim=0)
			print(f"{image_name}\t\t\t\t\t\t{classes[max_index.item()]}\t{max_value.item():.4f}")

			save_features(model, input_batch, image_name)


if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "split":
		# python test_alexnet.py --task split --src_dataset_path ../../data/database/classify/melon --dst_dataset_path datasets/melon_new_classify --resize (256,256) --ratios (0.7,0.2,0.1)
		split_dataset(args.src_dataset_path, args.dst_dataset_path, args.resize, args.ratios)
	elif args.task == "train":
		# python test_alexnet.py --task train --dst_dataset_path datasets/melon_new_classify --epochs 100 --mean (0.52817206,0.60931162,0.59818634) --std (0.2533697287956878,0.22790271847362834,0.2380239874816262) --model_name best.pth --labels_file classes.txt
		train(args.dst_dataset_path, args.epochs, args.mean, args.std, args.model_name, args.labels_file)
	else: # predict
		# python test_alexnet.py --task predict --predict_images_path datasets/melon_new_classify/test --mean (0.52817206,0.60931162,0.59818634) --std (0.2533697287956878,0.22790271847362834,0.2380239874816262) --model_name best.pth --labels_file classes.txt
		predict(args.model_name, args.labels_file, args.predict_images_path, args.mean, args.std)

	print(colorama.Fore.GREEN + "====== execution completed ======")
