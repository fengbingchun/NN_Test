import colorama
import argparse
from ultralytics import YOLO
import os
import torch

import numpy as np
np.bool = np.bool_ # Fix Error: AttributeError: module 'numpy' has no attribute 'bool'. OR: downgrade numpy: pip unistall numpy; pip install numpy==1.23.1

# Blog:
# 	https://blog.csdn.net/fengbingchun/article/details/139377787
#	https://blog.csdn.net/fengbingchun/article/details/140691177
#	https://blog.csdn.net/fengbingchun/article/details/141931184

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 predict")
	parser.add_argument("--model", required=True, type=str, help="model file")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment", "classify"], help="specify what kind of task")
	parser.add_argument("--dir_images", required=True, type=str, help="directory of test images")
	parser.add_argument("--dir_result", type=str, default="", help="directory where the image results are saved")

	args = parser.parse_args()
	return args

def get_images(dir):
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

def predict(model, task, dir_images, dir_result):
	model = YOLO(model) # load an model, support format: *.pt, *.onnx, *.torchscript, *.engine, openvino_model
	# model.info() # display model information # only *.pt format support

	images = get_images(dir_images)
	# print("images:", images)

	if task == "detect" or task =="segment":
		os.makedirs(dir_result) #, exist_ok=True)

	for image in images:
		device = "cuda" if torch.cuda.is_available() else "cpu"
		results = model.predict(dir_images+"/"+image, verbose=True, device=device)
		# print("results:", results)

		if task == "detect" or task =="segment":
			for result in results:
				result.save(dir_result+"/"+image)
		else:
			print(f"class names:{results[0].names}: top5: {results[0].probs.top5}; conf:{results[0].probs.top5conf}")

if __name__ == "__main__":
	# python test_yolov8_predict.py --model runs/detect/train10/weights/best_int8.engine --dir_images datasets/melon_new_detect/images/test --dir_result result_detect_engine_int8 --task classify
	colorama.init(autoreset=True)
	args = parse_args()

	print("Runging on GPU") if torch.cuda.is_available() else print("Runting on CPU")

	predict(args.model, args.task, args.dir_images, args.dir_result)

	print(colorama.Fore.GREEN + "====== execution completed ======")
