import argparse
import colorama
from ultralytics import YOLO
import torch
import os

# Blog:
# 	https://blog.csdn.net/fengbingchun/article/details/139203567
#	https://blog.csdn.net/fengbingchun/article/details/140691177
#	https://blog.csdn.net/fengbingchun/article/details/140850285

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 train")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment", "classify"], help="specify what kind of task")
	parser.add_argument("--yaml", required=True, type=str, help="yaml file or datasets path(classify)")
	parser.add_argument("--epochs", type=int, default=1000, help="number of training")
	parser.add_argument("--imgsz", type=int, default=640, help="input net image size")
	parser.add_argument("--patience", type=int, default=100, help="number of epochs to wait without improvement in validation metrics before early stopping the training")
	parser.add_argument("--batch", type=int, default=16, help="batch size")
	parser.add_argument("--optimizer", type=str, default="auto", help="choice of optimizer for training")
	parser.add_argument("--lr0", type=float, default=0.01, help="initial learning rate")
	parser.add_argument("--lrf", type=float, default=0.01, help="final learning rate as a fraction of the initial rate=(lr0*lrf)")
	parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for regularization in classification tasks")
	parser.add_argument("--pretrained_model", type=str, default="", help="pretrained model loaded during training")
	parser.add_argument("--gpu", type=str, default="0", help="set which graphics card to use. it can also support multiple graphics cards at the same time, for example 0,1")
	parser.add_argument("--augment", action="store_true", help="augment inference")

	args = parser.parse_args()
	return args

def train(task, yaml, epochs, imgsz, patience, batch, optimizer, lr0, lrf, dropout, pretrained_model, augment):
	if pretrained_model != "":
		model = YOLO(pretrained_model)
	else:
		if task == "detect":
			model = YOLO("yolov8n.pt") # load a pretrained model, should be a *.pt PyTorch model to run this method
		elif task == "segment":
			model = YOLO("yolov8n-seg.pt") # load a pretrained model, should be a *.pt PyTorch model to run this method
		elif task == "classify":
			model = YOLO("yolov8n-cls.pt") # n/s/m/l/x
		else:
			raise ValueError(colorama.Fore.RED + f"Error: unsupported task: {task}")

	# petience: Training stopped early as no improvement observed in last patience epochs, use patience=0 to disable EarlyStopping
	results = model.train(data=yaml, epochs=epochs, imgsz=imgsz, patience=patience, batch=batch, optimizer=optimizer, lr0=lr0, lrf=lrf, dropout=dropout, augment=augment) # train the model, supported parameter reference, for example: runs/segment(detect)/train3/args.yaml

	metrics = model.val() # It'll automatically evaluate the data you trained, no arguments needed, dataset and settings remembered
	if task == "classify":
		print(f"Top-1 Accuracy:{metrics.top1:.6f}") # top1 accuracy
		print(f"Top-5 Accuracy: {metrics.top5:.6f}") # top5 accuracy

	model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=imgsz) # onnx, export the model, cannot specify dynamic=True, opencv does not support
	# model.export(format="torchscript", imgsz=imgsz) # libtorch
	# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2) # tensorrt fp32
	# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2, half=True) # tensorrt fp16
	# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2, int8=True, data=yaml) # tensorrt int8
	# model.export(format="openvino", imgsz=imgsz) # openvino fp32
	# model.export(format="openvino", imgsz=imgsz, half=True) # openvino fp16
	# model.export(format="openvino", imgsz=imgsz, int8=True, data=yaml) # openvino int8, INT8 export requires 'data' arg for calibration

def set_gpu(id):
	os.environ["CUDA_VISIBLE_DEVICES"] = id # set which graphics card to use: 0,1,2..., default is 0

	print("available gpus:", torch.cuda.device_count())
	print("current gpu device:", torch.cuda.current_device())

if __name__ == "__main__":
	# python test_yolov8_train.py --yaml datasets/melon_new_detect/melon_new_detect.yaml --epochs 1000 --task detect --imgsz 640
	colorama.init(autoreset=True)
	args = parse_args()
	set_gpu(args.gpu)

	print("Running on GPU") if torch.cuda.is_available() else print("Running on CPU")

	train(args.task, args.yaml, args.epochs, args.imgsz, args.patience, args.batch, args.optimizer, args.lr0, args.lrf, args.dropout, args.pretrained_model, args.augment)

	print(colorama.Fore.GREEN + "====== execution completed ======")
