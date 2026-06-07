import argparse
import colorama
from ultralytics import YOLO
import torch
import os

import numpy as np
np.bool = np.bool_ # Fix Error: AttributeError: module 'numpy' has no attribute 'bool'. OR: downgrade numpy: pip unistall numpy; pip install numpy==1.23.1

# Blog:
# 	https://blog.csdn.net/fengbingchun/article/details/139203567
#	https://blog.csdn.net/fengbingchun/article/details/140691177
#	https://blog.csdn.net/fengbingchun/article/details/140850285
#	https://blog.csdn.net/fengbingchun/article/details/157615429
# 	https://blog.csdn.net/fengbingchun/article/details/139377787
#	https://blog.csdn.net/fengbingchun/article/details/141931184
#	https://blog.csdn.net/fengbingchun/article/details/161764208

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8/YOLO11/YOLO26 train and predict")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment", "classify", "obb"], help="specify what kind of task")
	parser.add_argument("--task2", required=True, type=str, choices=["train", "predict"], help="train or predict")
	parser.add_argument("--model_name", required=True, type=str, help="model name")
	parser.add_argument("--yaml", type=str, help="yaml file or datasets path(classify)")
	parser.add_argument("--epochs", type=int, default=1000, help="number of training")
	parser.add_argument("--imgsz", type=int, default=640, help="input net image size")
	parser.add_argument("--patience", type=int, default=100, help="number of epochs to wait without improvement in validation metrics before early stopping the training")
	parser.add_argument("--batch", type=int, default=16, help="batch size")
	parser.add_argument("--optimizer", type=str, default="auto", help="choice of optimizer for training")
	parser.add_argument("--lr0", type=float, default=0.01, help="initial learning rate")
	parser.add_argument("--lrf", type=float, default=0.01, help="final learning rate as a fraction of the initial rate=(lr0*lrf)")
	parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for regularization in classification tasks")
	parser.add_argument("--gpu", type=str, default="0", help="set which graphics card to use. it can also support multiple graphics cards at the same time, for example 0,1")
	parser.add_argument("--augment", action="store_true", help="augment inference")
	parser.add_argument("--dir_images", type=str, default="", help="directory of test images")
	parser.add_argument("--verbose", action="store_true", help="whether to output detailed information")
	parser.add_argument("--dir_result", type=str, default="", help="directory where the image or video results are saved")

	args = parser.parse_args()
	return args

def train(task, model_name, yaml, epochs, imgsz, patience, batch, optimizer, lr0, lrf, dropout, augment):
	model = YOLO(model_name) # load a pretrained model, should be a *.pt PyTorch model to run this method, n/s/m/l/x

	# petience: Training stopped early as no improvement observed in last patience epochs, use patience=0 to disable EarlyStopping
	model.train(data=yaml, epochs=epochs, imgsz=imgsz, patience=patience, batch=batch, optimizer=optimizer, lr0=lr0, lrf=lrf, dropout=dropout, augment=augment) # train the model, supported parameter reference, for example: runs/segment(detect)/train3/args.yaml

	metrics = model.val() # It'll automatically evaluate the data you trained, no arguments needed, dataset and settings remembered
	if task == "classify":
		print(f"Top-1 Accuracy:{metrics.top1:.6f}") # top1 accuracy
		print(f"Top-5 Accuracy: {metrics.top5:.6f}") # top5 accuracy
	else:
		print(f"map50-95(B):", metrics.box.map)
		print(f"map50(B):", metrics.box.map50)
		print(f"map75(B):", metrics.box.map75)

	model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=imgsz) # onnx, export the model, cannot specify dynamic=True, opencv does not support
	model.export(format="torchscript", imgsz=imgsz) # libtorch
	model.export(format="openvino", imgsz=imgsz, dynamic=False, half=False, int8=False) # openvino fp32
	# model.export(format="openvino", imgsz=imgsz, dynamic=False, half=True) # openvino fp16
	# model.export(format="openvino", imgsz=imgsz, dynamic=False, int8=True, data=yaml) # openvino int8, INT8 export requires 'data' arg for calibration
	if torch.cuda.is_available():
		model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2) # tensorrt fp32
		# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2, half=True) # tensorrt fp16
		# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2, int8=True, data=yaml) # tensorrt int8

def _get_images(dir):
	# supported image formats
	img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
	images = []

	for file in os.listdir(dir):
		if os.path.isfile(os.path.join(dir, file)):
			# print(file)
			_, extension = os.path.splitext(file)
			for format in img_formats:
				if format == extension.lower():
					images.append(file)
					break

	return images

def predict(task, model_name, device, verbose, dir_images, dir_result):
	model = YOLO(model_name) # load an model, support format: *.pt, *.onnx, *.torchscript, *.engine, openvino_model
	# model.info() # display model information # only *.pt format support

	if task == "detect" or task =="segment" or task == "obb":
		os.makedirs(dir_result, exist_ok=True)

	images = _get_images(dir_images)

	for image in images:
		results = model.predict(dir_images+"/"+image, verbose=verbose, device=device)

		if task == "detect" or task =="segment" or task == "obb":
			for result in results:
				# print("result:", result)
				result.save(dir_result+"/"+image)
		else:
			print(f"class names:{results[0].names}: top5: {results[0].probs.top5}; conf:{results[0].probs.top5conf}")

def _set_gpu(id):
	os.environ["CUDA_VISIBLE_DEVICES"] = id # set which graphics card to use: 0,1,2..., default is 0

	print("available gpus:", torch.cuda.device_count())
	print("current gpu device:", torch.cuda.current_device())

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if torch.cuda.is_available():
		print("Running on GPU")
		_set_gpu(args.gpu)
		device = "cuda"
	else:
		print("Running on CPU")
		device = "cpu"

	if args.task2 == "train":
		train(args.task, args.model_name, args.yaml, args.epochs, args.imgsz, args.patience, args.batch, args.optimizer, args.lr0, args.lrf, args.dropout, args.augment)
	else:
		predict(args.task, args.model_name, device, args.verbose, args.dir_images, args.dir_result)

	print(colorama.Fore.GREEN + "====== execution completed ======")
