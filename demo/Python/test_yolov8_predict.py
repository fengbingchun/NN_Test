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
import copy

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
	parser.add_argument("--verbose", action="store_true", help="whether to output detailed information")
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
class SaveResult:
	def __init__(self, result_dir):
		self.frame_difference = 20
		self.average_pixel_value = 200
		self.roi = (92, 40, 200, 100) # x,y,width,height
		self.confidence = 0.8
		self.distance = 4
		self.continuous_number = 3
		self.cycle_min_time = 35
		self.max_frames = 500
		self.conf_diff = 0.05
		self.frames_mininum = 10
		self.result_dir = result_dir
		self.frames_diff = []
		self.frames_value = []
		self.frames_predict = []
		self.first_frame = True
		self.cycle_start = False
		self.cycle_end = False
		self.frame_start_time = None
		self.frame_end_time = None

	def _pixels_average(self, image):
		roi = image[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]
		mean = np.mean(roi)
		# print(f"mean: {mean}"); raise
		return mean

	def _reset(self):
		self.frames_diff[:] = self.frames_diff[-self.continuous_number:]
		self.frames_value[:] = self.frames_value[-self.continuous_number:]
		self.frames_predict.clear()
		self.cycle_start = False
		self.cycle_end = False
		self.frame_start_time = None
		self.frame_end_time = None
		# print(f"frames diff: {self.frames_diff}; frames value: {self.frames_value}"); raise

	def _str_to_datetime(self, name):
		return datetime.strptime(name[:-4], "%Y%m%d%H%M%S")

	def _parse_predict_result(self, predict_result):
		boxes = []
		for i in range(len(predict_result.boxes.data)):
			data = predict_result.boxes.data[i].tolist()
			box = {}
			box["label"] = int(data[-1])
			box["pos"] = [int(data[0]+0.5), int(data[1]+0.5), int(data[2]+0.5), int(data[3]+0.5)] # left, top, right, bottom
			box["confidence"] = float(data[-2])
			boxes.append(box)

		if len(boxes) == 2:
			# print(f"boxes: {boxes}"); raise
			longstrip = {}
			hotzone = {}
			for box in boxes:
				if box["label"] == 0:
					longstrip = box.copy()
				if box["label"] == 2:
					hotzone = box.copy()

			if bool(longstrip) and bool(hotzone) and \
				longstrip["confidence"] > self.confidence and hotzone["confidence"] > self.confidence and \
				abs(longstrip["pos"][1]-hotzone["pos"][3]) < self.distance:
				return [longstrip["confidence"], longstrip["pos"][0], longstrip["pos"][1], longstrip["pos"][2], longstrip["pos"][3], hotzone["confidence"]]

		return None

	def _draw_cross_sectional(self, info):
		name = info[0][:-4]+".png"
		name = self.result_dir+"/"+name
		# print(f"name: {name}"); raise

		maxh, maxw = info[1].shape[:2]
		rect = [info[3], info[4], info[5], info[6]]
		h = int((rect[2] - rect[0]) / 8)# + 0.5)
		w = h * 8
		xcenter = rect[0] + int((rect[2] - rect[0]) / 2 + 0.5)

		x1 = max(0, int(xcenter - w / 2))
		y1 = max(0, rect[1] - h - 1)
		x2 = min(maxw, x1+w)
		y2 = min(maxh, y1+h)
		assert (x2-x1) == (y2-y1)*8, f"the width must be 8 times the height: (x1,y1,x2,y2): {x1},{y1},{x2},{y2}"
		roi = info[1][y1:y2, x1:x2]
		resized = cv2.resize(roi, (512, 64))
		cv2.imwrite(name, resized)

	def _save_image(self):
		if len(self.frames_predict) < self.frames_mininum:
			return

		red_values =[]
		for idx in range(len(self.frames_predict) // 2):
			info = self.frames_predict[idx]
			# print(f"info: name: {info[0]}; longstrip conf: {info[2]:.4f}; hotzone conf: {info[7]:.4f}; red value: {info[8]}")
			red_values.append(info[8])

		# print(f"red values: {red_values}; min value: {min(red_values)}; index: {red_values.index(min(red_values))}")
		# print(f"info: {self.frames_predict[red_values.index(min(red_values))]}")
		index = red_values.index(min(red_values))
		info = self.frames_predict[index]
		cv2.imwrite(self.result_dir+"/"+info[0], info[1])

		self._draw_cross_sectional(info)

	def load_image(self, image, predict_result, name):
		# print(f"name: {name}"); raise
		if len(self.frames_diff) >= self.max_frames:
			print(colorama.Fore.YELLOW + f"the length of frames_diff exceeds the maximum supported number: {self.max_frames}, reset")
			self._reset()

		_, _, red = cv2.split(image)
		self.frames_value.append(self._pixels_average(red))
		if self.first_frame:
			self.frames_diff.append(0)
			self.first_frame = False
		else:
			self.frames_diff.append(self.frames_value[-1]-self.frames_value[-2])

		if len(self.frames_diff) > self.continuous_number and self.frames_diff[-(self.continuous_number+1)] > self.frame_difference:
			flag = True
			for idx in range(self.continuous_number):
				if self.frames_value[-(idx+1)] < self.average_pixel_value:
					flag = False
					break

			if flag:
				# print(f"cycle start: frames diff: {self.frames_diff}; frames value: {self.frames_value}; length: {len(self.frames_diff)}"); raise
				if not self.cycle_start:
					self.cycle_start = True
					self.frame_start_time = self._str_to_datetime(name)
					# print(f"frame start time: {self.frame_start_time}"); raise
				else:
					cycle_end_time = self._str_to_datetime(name)
					if cycle_end_time - self.frame_start_time > timedelta(seconds=self.cycle_min_time):
						self.frame_end_time = cycle_end_time
						self.cycle_end = True
						# print(f"start time: {self.frame_start_time}; end time: {self.frame_end_time}"); raise

		if self.cycle_start:
			ret = self._parse_predict_result(predict_result)
			if ret is not None and self._pixels_average(red) < self.average_pixel_value:
				self.frames_predict.append([name, image, ret[0], ret[1], ret[2], ret[3], ret[4], ret[5], self._pixels_average(red)])
				# print(f"frames predict: {self.frames_predict}"); raise

		if self.cycle_start and self.cycle_end:
			print(f"end time: {self.frame_end_time}; start time: {self.frame_start_time}; cycle: {self.frame_end_time-self.frame_start_time}; len frames predict: {len(self.frames_predict)}")
			self._save_image()

			self.frames_diff[:] = self.frames_diff[-self.continuous_number:]
			self.frames_value[:] = self.frames_value[-self.continuous_number:]
			self.frames_predict.clear()
			self.cycle_end = False
			self.frame_start_time = self.frame_end_time
			self.frame_end_time = None


def draw_rect(predict_result, src_image_name, dst_image_name, buckle, count, save_result):
	# print_boxes_info(predict_result)
	image = cv2.imread(src_image_name)
	if image is None:
		raise FileNotFoundError(colorama.Fore.RED + f"could not load image: {src_image_name}")
	image2 = np.copy(image)

	for i in range(len(predict_result.boxes.data)):
		data = predict_result.boxes.data[i].tolist()
		cv2.rectangle(image2, (int(data[0]+0.5), int(data[1]+0.5)), (int(data[2]+0.5), int(data[3]+0.5)), rect_color(int(data[-1])), 1)
		cv2.putText(image2, f"{int(data[-1])},{data[-2]:.2f}", (int(data[0]+0.5), int(data[3]+0.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color(int(data[-1])), 1, cv2.LINE_AA)
	cv2.imwrite(dst_image_name, image2)
	# save_image(predict_result, dst_image_name, image, src_image_name)
	save_result.load_image(image, predict_result, str(Path(src_image_name).name))

	# draw_cross_sectional(predict_result, dst_image_name, dst_image_name, buckle, count)

def predict(task, model, verbose, dir_images, video_file, dir_result):
	model = YOLO(model) # load an model, support format: *.pt, *.onnx, *.torchscript, *.engine, openvino_model
	# model.info() # display model information # only *.pt format support

	if task == "detect" or task =="segment":
		os.makedirs(dir_result, exist_ok=True)
		os.makedirs(dir_result+"/result", exist_ok=True)
		# os.makedirs(dir_result+"/good", exist_ok=True)
		# os.makedirs(dir_result+"/normal", exist_ok=True)
		# os.makedirs(dir_result+"/obscured", exist_ok=True)
		# os.makedirs(dir_result+"/abnormal", exist_ok=True)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	buckle = []
	count = [0]

	if dir_images != "":
		images = get_images(dir_images)
		# print("images:", images)
		save_result = SaveResult(dir_result+"/result")

		for image in images:
			# time.sleep(0.95) # <====== comment out or modify
			results = model.predict(dir_images+"/"+image, verbose=verbose, device=device)
			# print("results:", results)

			if task == "detect" or task =="segment":
				for result in results:
					result.save(dir_result+"/"+image)
					# boxes = parse_result(result); print(boxes); raise
					# draw_rect(result, dir_images+"/"+image, dir_result+"/"+image, buckle, count, save_result)
			else:
				print(f"class names:{results[0].names}: top5: {results[0].probs.top5}; conf:{results[0].probs.top5conf}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()
	if args.dir_images == "" and args.video_file == "":
		raise ValueError(colorama.Fore.RED + f"dir_images and video file cannot be empty at the same time:{args.dir_images}, {args.video}")

	print("Running on GPU") if torch.cuda.is_available() else print("Running on CPU")

	predict(args.task, args.model, args.verbose, args.dir_images, args.video_file, args.dir_result)

	print(colorama.Fore.GREEN + "====== execution completed ======")
