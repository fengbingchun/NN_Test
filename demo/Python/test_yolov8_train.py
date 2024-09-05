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
	parser.add_argument("--yaml", required=True, type=str, help="yaml file or datasets path(classify)")
	parser.add_argument("--epochs", required=True, type=int, help="number of training")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment", "classify"], help="specify what kind of task")
	parser.add_argument("--imgsz", type=int, default=640, help="input net image size")

	args = parser.parse_args()
	return args

def train(task, yaml, epochs, imgsz):
	if task == "detect":
		model = YOLO("yolov8n.pt") # load a pretrained model, should be a *.pt PyTorch model to run this method
	elif task == "segment":
		model = YOLO("yolov8n-seg.pt") # load a pretrained model, should be a *.pt PyTorch model to run this method
	elif task == "classify":
		model = YOLO("yolov8n-cls.pt") # n/s/m/l/x
	else:
		raise ValueError(colorama.Fore.RED + f"Error: unsupported task: {task}")

	# petience: Training stopped early as no improvement observed in last patience epochs, use patience=0 to disable EarlyStopping
	results = model.train(data=yaml, epochs=epochs, imgsz=imgsz, patience=150, augment=True) # train the model, supported parameter reference, for example: runs/segment(detect)/train3/args.yaml

	metrics = model.val() # It'll automatically evaluate the data you trained, no arguments needed, dataset and settings remembered
	if task == "classify":
		print("Top-1 Accuracy:", metrics.top1) # top1 accuracy
		print("Top-5 Accuracy:", metrics.top5) # top5 accuracy

	model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=imgsz) # onnx, export the model, cannot specify dynamic=True, opencv does not support
	# model.export(format="torchscript", imgsz=imgsz) # libtorch
	# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2) # tensorrt fp32
	# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2, half=True) # tensorrt fp16
	# model.export(format="engine", imgsz=imgsz, dynamic=False, verbose=False, batch=1, workspace=2, int8=True, data=yaml) # tensorrt int8
	# model.export(format="openvino", imgsz=imgsz) # openvino fp32
	# model.export(format="openvino", imgsz=imgsz, half=True) # openvino fp16
	# model.export(format="openvino", imgsz=imgsz, int8=True, data=yaml) # openvino int8, INT8 export requires 'data' arg for calibration


if __name__ == "__main__":
	# python test_yolov8_train.py --yaml datasets/melon_new_detect/melon_new_detect.yaml --epochs 1000 --task detect --imgsz 640
	colorama.init(autoreset=True)
	args = parse_args()

	print("Runging on GPU") if torch.cuda.is_available() else print("Runting on CPU")

	train(args.task, args.yaml, args.epochs, args.imgsz)

	print(colorama.Fore.GREEN + "====== execution completed ======")
