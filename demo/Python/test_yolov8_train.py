import argparse
import colorama
from ultralytics import YOLO
import torch

# Blog:
# 	https://blog.csdn.net/fengbingchun/article/details/139203567
#	https://blog.csdn.net/fengbingchun/article/details/140691177
#	https://blog.csdn.net/fengbingchun/article/details/140850285

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 train")
	parser.add_argument("--yaml", required=True, type=str, help="yaml file")
	parser.add_argument("--epochs", required=True, type=int, help="number of training")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment"], help="specify what kind of task")

	args = parser.parse_args()
	return args

def train(task, yaml, epochs):
	if task == "detect":
		model = YOLO("yolov8n.pt") # load a pretrained model
	elif task == "segment":
		model = YOLO("yolov8n-seg.pt") # load a pretrained model
	else:
		print(colorama.Fore.RED + "Error: unsupported task:", task)
		raise

	results = model.train(data=yaml, epochs=epochs, imgsz=640) # train the model

	metrics = model.val() # It'll automatically evaluate the data you trained, no arguments needed, dataset and settings remembered

	model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640) # onnx, export the model, cannot specify dynamic=True, opencv does not support
	model.export(format="torchscript", imgsz=640) # libtorch
	model.export(format="engine", imgsz=640, dynamic=False, verbose=False, batch=1, workspace=2) # tensorrt fp32
	# model.export(format="engine", imgsz=640, dynamic=True, verbose=True, batch=4, workspace=2, half=True) # tensorrt fp16
	# model.export(format="engine", imgsz=640, dynamic=True, verbose=True, batch=4, workspace=2, int8=True, data=yaml) # tensorrt int8
	# model.export(format="openvino", imgsz=640) # openvino fp32
	# model.export(format="openvino", imgsz=640, half=True) # openvino fp16
	model.export(format="openvino", imgsz=640, int8=True, data=yaml) # openvino int8, INT8 export requires 'data' arg for calibration


if __name__ == "__main__":
	# python test_yolov8_train.py --yaml datasets/melon_new_detect/melon_new_detect.yaml --epochs 1000 --task detect
	colorama.init()
	args = parse_args()

	if torch.cuda.is_available():
		print("Runging on GPU")
	else:
		print("Runting on CPU")

	train(args.task, args.yaml, args.epochs)

	print(colorama.Fore.GREEN + "====== execution completed ======")
