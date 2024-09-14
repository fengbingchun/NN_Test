import colorama
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models

# Blog: https://blog.csdn.net/fengbingchun/article/details/142261770

def parse_args():
	parser = argparse.ArgumentParser(description="learning rate warm up")
	parser.add_argument("--epochs", required=True, type=int, help="number of training")
	parser.add_argument("--dataset_path", required=True, type=str, help="source dataset path")
	parser.add_argument("--model_name", required=True, type=str, help="the model generated during training or the model loaded during prediction")
	parser.add_argument("--pretrained_model", type=str, default="", help="pretrained model loaded during training")
	parser.add_argument("--batch_size", type=int, default=2, help="specify the batch size")

	args = parser.parse_args()
	return args

def load_dataset(dataset_path, batch_size):
	mean = (0.53087615, 0.23997033, 0.45703197)
	std = (0.29807151489753686, 0.3128615049442739, 0.15151863355831655)

	transform = transforms.Compose([
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=mean, std=std), # RGB
	])

	train_dataset = ImageFolder(root=dataset_path+"/train", transform=transform)
	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	val_dataset = ImageFolder(root=dataset_path+"/val", transform=transform)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	assert len(train_dataset.class_to_idx) == len(val_dataset.class_to_idx), f"the number of categories int the train set must be equal to the number of categories in the validation set: {len(train_dataset.class_to_idx)} : {len(val_dataset.class_to_idx)}"

	return len(train_dataset.class_to_idx), len(train_dataset), len(val_dataset), train_loader, val_loader

def train(model, train_loader, device, optimizer, criterion, train_loss, train_acc):
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

	return train_loss, train_acc

def validate(model, val_loader, device, criterion, val_loss, val_acc):
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

	return val_loss, val_acc

def training(epochs, dataset_path, model_name, pretrained_model, batch_size):
	classes_num, train_dataset_num, val_dataset_num, train_loader, val_loader = load_dataset(dataset_path, batch_size)
	model = models.ResNet(block=models.resnet.BasicBlock, layers=[2,2,2,2], num_classes=classes_num) # ResNet18

	if pretrained_model != "":
		model.load_state_dict(torch.load(pretrained_model))

	optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99), eps=1e-7) # set the optimizer
	scheduler = optim.lr_scheduler.ConstantLR(optimizer, factor=0.2, total_iters=10)
	# scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=0.8, total_iters=5)
	# assert len(optimizer.param_groups) == 1, f"optimizer.param_groups's length must be equal to 1: {len(optimizer.param_groups)}"
	# lr_lambda = lambda epoch: 0.95 ** epoch
	# scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
	# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
	# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
	# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.2)
	# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.05)
	# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
	# scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.05)
	# scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.)
	print(f"epoch: 0/{epochs}: learning rate: {scheduler.get_last_lr()}")

	criterion = nn.CrossEntropyLoss() # set the loss

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	highest_accuracy = 0.
	minimum_loss = 100.

	for epoch in range(epochs):
		epoch_start = time.time()
		train_loss = 0.0 # loss
		train_acc = 0.0 # accuracy
		val_loss = 0.0
		val_acc = 0.0

		train_loss, train_acc = train(model, train_loader, device, optimizer, criterion, train_loss, train_acc)
		val_loss, val_acc = validate(model, val_loader, device, criterion, val_loss, val_acc)
		# scheduler.step(val_loss) # update lr, ReduceLROnPlateau
		scheduler.step() # update lr

		avg_train_loss = train_loss / train_dataset_num # average training loss
		avg_train_acc = train_acc / train_dataset_num # average training accuracy
		avg_val_loss = val_loss / val_dataset_num # average validation loss
		avg_val_acc = val_acc / val_dataset_num # average validation accuracy
		epoch_end = time.time()
		print(f"epoch:{epoch+1}/{epochs}; learning rate: {scheduler.get_last_lr()}, train loss:{avg_train_loss:.6f}, accuracy:{avg_train_acc:.6f}; validation loss:{avg_val_loss:.6f}, accuracy:{avg_val_acc:.6f}; time:{epoch_end-epoch_start:.2f}s")

		if highest_accuracy < avg_val_acc and minimum_loss > avg_val_loss:
			torch.save(model.state_dict(), model_name)
			highest_accuracy = avg_val_acc
			minimum_loss = avg_val_loss

		if avg_val_loss < 0.00001 and avg_val_acc > 0.9999:
			print(colorama.Fore.YELLOW + "stop training early")
			torch.save(model.state_dict(), model_name)
			break

if __name__ == "__main__":
	# python test_learning_rate_warmup.py --epochs 1000 --dataset_path datasets/melon_new_classify --pretrained_model pretrained.pth --model_name best.pth
	colorama.init(autoreset=True)
	args = parse_args()

	training(args.epochs, args.dataset_path, args.model_name, args.pretrained_model, args.batch_size)

	print(colorama.Fore.GREEN + "====== execution completed ======")
