import colorama
import argparse
from ultralytics import YOLO
import os
import torch
import cv2
from datetime import datetime, timedelta
import time
from pathlib import Path
import csv

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
	parser.add_argument("--dir_images", type=str, default="", help="directory of test images")
	parser.add_argument("--video_file", type=str, default="", help="video file")
	parser.add_argument("--dir_result", type=str, default="", help="directory where the image or video results are saved")

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

def print_boxes_info(predict_result):
	print(f"orig img shape: {predict_result.orig_img.shape}") # (h,w,c)
	print(f"box result:")
	for i in range(len(predict_result.boxes.data)):
		data = predict_result.boxes.data[i].tolist()
		print(f"  cls:{int(data[-1])}, conf:{data[-2]:.2f}, rect(xyxy):{data[0]:.2f},{data[1]:.2f},{data[2]:.2f},{data[3]:.2f}")

def linear_func(x):
	return 0.2376 * x - 14.4752 # <====== modify according to actual situation

def draw_cross_sectional(predict_result, src_image_name, dst_image_name, buckle, count):
	# <====== modify according to actual situation
	min_conf = 0.85
	min_width = 100
	min_height = 10
	ystart = 170; yend = 187 # ROI
	min_count = 10
	min_seconds = 10

	count[0] += 1
	best_box = []

	if len(predict_result.boxes.data) != 0:
		for i in range(len(predict_result.boxes.data)):
			data = predict_result.boxes.data[i].tolist()
			conf = data[-2]
			width = data[2] - data[0]
			height = data[3] - data[1]
			ypos = data[1]
			if conf > min_conf and width > min_width and height > min_height and ypos > ystart and ypos < yend:
				if len(best_box) == 0:
					best_box.extend([conf, width, height, ypos, datetime.now()])
				else:
					if best_box[0] < conf:
						best_box.clear()
						best_box.extend([conf, width, height, ypos, datetime.now()])

	if best_box:
		if isinstance(best_box[0], list) or len(best_box) != 5:
			print(f"best_box: {best_box}")
			raise ValueError(colorama.Fore.RED + f"the length of the best_box must be equal to 5")

		if len(buckle) == 0:
			buckle.extend(best_box)
			buckle.extend([src_image_name, dst_image_name])
		else:
			if (best_box[4] - buckle[4]).total_seconds() > min_seconds and count[0] > min_count: # write
				image = cv2.imread(buckle[5])
				if image is None:
					raise FileNotFoundError(colorama.Fore.RED + f"could not load image: {buckle[5]}")

				h = int(linear_func(buckle[3]) + 0.5)
				w = h * 8
				x = int((image.shape[1] - w) / 2 + 0.5)
				y = int(buckle[3] + 0.5) - h - 1
				if x < 0 or y < 0 or x+w>image.shape[1] or y+h>image.shape[0]:
					raise ValueError(colorama.Fore.RED + f"the size of the rectangle is out of range: {x},{y},{w},{h}")

				# image size:64x512
				roi = image[y:y+h, x:x+w]
				scaled = cv2.resize(roi, (512,64))
				cv2.imwrite(buckle[6]+".jpg", scaled)

				cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 1)
				cv2.imwrite(buckle[6], image)

				count[0] = 0
				buckle.clear()
				buckle.extend(best_box)
				buckle.extend([src_image_name, dst_image_name])
			else: # compare
				if best_box[0] > buckle[0]:
					buckle.clear()
					buckle.extend(best_box)
					buckle.extend([src_image_name, dst_image_name])

def write_rect(name, x1, y1, x2, y2):
	image = cv2.imread(name)
	path = Path(name)
	name = str(path.name)

	with open("result.csv", mode="a", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow([name, x1, y1, x2, y2])

	# cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 1)
	# cv2.imwrite(name, image)

def draw_cross_sectional(image, rect, name, src_image_name):
	maxh, maxw = image.shape[:2]
	h = int((rect[2] - rect[0]) / 8 + 0.5)
	w = h * 8
	xcenter = rect[0] + int((rect[2] - rect[0]) / 2 + 0.5)

	x1 = max(0, int(xcenter - w / 2))
	y1 = max(0, rect[1] - h - 1)
	x2 = min(maxw, x1+w)
	y2 = min(maxh, y1+h)
	cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 1)

	image2 = cv2.imread(src_image_name)
	roi = image2[y1:y2, x1:x2]
	resized = cv2.resize(roi, (512, 64))
	cv2.imwrite(name+".png", resized)

	write_rect(src_image_name, x1, y1, x2, y2)

def save_image(predict_result, dst_image_name, image, src_image_name):
	dir = os.path.dirname(dst_image_name)
	name = os.path.basename(dst_image_name)
	if len(predict_result.boxes.data) == 0:
		cv2.imwrite(dir+"/abnormal/"+name, image)
	else:
		rects = []
		# <====== modify according to actual situation
		roi = [35, 115, 340, 205] # left top, right bottom
		minw = 155
		conf = 0.85

		for i in range(len(predict_result.boxes.data)):
			data = predict_result.boxes.data[i].tolist()
			rect = [int(data[0]+0.5), int(data[1]+0.5), int(data[2]+0.5), int(data[3]+0.5), data[-2]]
			if not (roi[0] < rect[0] < roi[2]) or not (roi[0] < rect[2] < roi[2]) or not (roi[1] < rect[1] < roi[3]) or not (roi[1] < rect[3] < roi[3]):
				continue
			rects.append(rect)

		if len(rects) > 2 or len(rects) == 0:
			cv2.imwrite(dir+"/abnormal/"+name, image)
		elif len(rects) == 1:
			if rects[0][2] - rects[0][0] > minw:
				if rects[0][4] > conf:
					draw_cross_sectional(image, rects[0], dir+"/good/"+name, src_image_name)
					cv2.imwrite(dir+"/good/"+name, image)
				else:
					draw_cross_sectional(image, rects[0], dir+"/normal/"+name, src_image_name)
					cv2.imwrite(dir+"/normal/"+name, image)
			else:
				cv2.imwrite(dir+"/obscured/"+name, image)
		else:
			cv2.imwrite(dir+"/obscured/"+name, image)

def rect_color(index):
	colors = [(255,0,0), (255,255,0), (128,255,128), (0,255,0)]
	return colors[index]

def draw_rect(predict_result, src_image_name, dst_image_name, buckle, count):
	# print_boxes_info(predict_result)
	image = cv2.imread(src_image_name)
	if image is None:
		raise FileNotFoundError(colorama.Fore.RED + f"could not load image: {src_image_name}")

	for i in range(len(predict_result.boxes.data)):
		data = predict_result.boxes.data[i].tolist()
		cv2.rectangle(image, (int(data[0]+0.5), int(data[1]+0.5)), (int(data[2]+0.5), int(data[3]+0.5)), rect_color(int(data[-1])), 1)
		cv2.putText(image, f"{int(data[-1])},{data[-2]:.2f}", (int(data[0]+0.5), int(data[3]+0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color(int(data[-1])), 1, cv2.LINE_AA)
	cv2.imwrite(dst_image_name, image)
	# save_image(predict_result, dst_image_name, image, src_image_name)

	# draw_cross_sectional(predict_result, dst_image_name, dst_image_name, buckle, count)

def parse_result(result): # result = results[0]
	boxes = []
	for i in range(len(result.boxes.data)):
		data = result.boxes.data[i].tolist()
		box = {}
		# box["valid"] = False
		box["label"] = int(data[-1])
		box["pos"] = [int(data[0]), int(data[1]), int(data[2]), int(data[3])] # left, top, right, bottom
		box["confidence"] = float(data[-2])
		boxes.append(box)

	return boxes


def predict(task, model, dir_images, video_file, dir_result):
	model = YOLO(model) # load an model, support format: *.pt, *.onnx, *.torchscript, *.engine, openvino_model
	# model.info() # display model information # only *.pt format support

	if task == "detect" or task =="segment":
		os.makedirs(dir_result, exist_ok=True)
		os.makedirs(dir_result+"/good", exist_ok=True)
		os.makedirs(dir_result+"/normal", exist_ok=True)
		os.makedirs(dir_result+"/obscured", exist_ok=True)
		os.makedirs(dir_result+"/abnormal", exist_ok=True)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	buckle = []
	count = [0]

	if dir_images != "":
		images = get_images(dir_images)
		# print("images:", images)

		for image in images:
			# time.sleep(0.95) # <====== comment out or modify
			results = model.predict(dir_images+"/"+image, verbose=True, device=device)
			# print("results:", results)

			if task == "detect" or task =="segment":
				for result in results:
					# result.save(dir_result+"/"+image)
					# boxes = parse_result(result); print(boxes); raise
					draw_rect(result, dir_images+"/"+image, dir_result+"/"+image, buckle, count)
			else:
				print(f"class names:{results[0].names}: top5: {results[0].probs.top5}; conf:{results[0].probs.top5conf}")

if __name__ == "__main__":
	# python test_yolov8_predict.py --model runs/detect/train10/weights/best_int8.engine --dir_images datasets/melon_new_detect/images/test --dir_result result_detect_engine_int8 --task detect
	colorama.init(autoreset=True)
	args = parse_args()
	if args.dir_images == "" and args.video_file == "":
		raise ValueError(colorama.Fore.RED + f"dir_images and video file cannot be empty at the same time:{args.dir_images}, {args.video}")

	print("Runging on GPU") if torch.cuda.is_available() else print("Runting on CPU")

	predict(args.task, args.model, args.dir_images, args.video_file, args.dir_result)

	print(colorama.Fore.GREEN + "====== execution completed ======")
