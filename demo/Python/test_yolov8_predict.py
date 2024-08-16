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

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 predict")
	parser.add_argument("--model", required=True, type=str, help="model file")
	parser.add_argument("--dir_images", required=True, type=str, help="directory of test images")
	parser.add_argument("--dir_result", required=True, type=str, help="directory where the image results are saved")

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

def predict(model, dir_images, dir_result):
	model = YOLO(model) # load an model, support format: *.pt, *.onnx, *.torchscript, *.engine, openvino_model
	# model.info() # display model information # only *.pt format support

	images = get_images(dir_images)
	# print("images:", images)

	os.makedirs(dir_result) #, exist_ok=True)

	for image in images:
		if torch.cuda.is_available():
			results = model.predict(dir_images+"/"+image, verbose=True, device="cuda")
		else:
			results = model.predict(dir_images+"/"+image, verbose=True)
		for result in results:
			# print("result:", result)
			result.save(dir_result+"/"+image)
			
if __name__ == "__main__":
	# python test_yolov8_predict.py --model runs/detect/train10/weights/best_int8.engine --dir_images datasets/melon_new_detect/images/test --dir_result result_detect_engine_int8
	colorama.init()
	args = parse_args()

	if torch.cuda.is_available():
		print("Runging on GPU")
	else:
		print("Runting on CPU")

	predict(args.model, args.dir_images, args.dir_result)

	print(colorama.Fore.GREEN + "====== execution completed ======")
