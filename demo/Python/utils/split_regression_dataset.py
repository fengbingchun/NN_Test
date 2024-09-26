import cv2
import os
import random
import shutil
import numpy as np
import ast
import csv
from pathlib import Path

class SplitRegressionDataset:
	"""split the regression dataset"""

	def __init__(self, path_src, path_dst, csv_file, ratios=(0.8, 0.1, 0.1)):
		"""
		path_src: source dataset path
		path_dst: the path to the split dataset
		csv_file: csv file
		ratios: they are the ratios of train set, validation set, and test set, respectively
		"""
		assert len(ratios) == 3, f"the length of ratios is not 3: {len(ratios)}"
		assert abs(ratios[0] + ratios[1] + ratios[2] - 1) < 1e-05, f"ratios sum must be 1: {ratios[0]}, {ratios[1]}, {ratios[2]}"
		assert csv_file.lower().endswith(".csv") == True, f"{csv_file} must be a csv file"

		self.path_src = path_src
		self.path_dst = path_dst
		self.csv_file = csv_file
		self.ratio_train = ratios[0]
		self.ratio_val = ratios[1]
		self.ratio_test = ratios[2]

		self.is_resize = False
		self.fill_value = None
		self.shape = None

		self.length_total = None

		self.mean = None
		self.std = None

		self.supported_img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")

	def resize(self, value=(114,114,114), shape=(256,256)):
		"""
		value: fill value
		shape: the scaled shape:(height, width)
		"""
		self.is_resize = True
		self.fill_value = value
		self.shape = shape

	def _create_dir(self):
		directory = self.path_dst + "/train"
		if os.path.exists(directory):
			raise ValueError(f"{directory} directory already exists, delete it")
		os.makedirs(directory, exist_ok=True)

		directory = self.path_dst + "/val"
		if os.path.exists(directory):
			raise ValueError(f"{directory} directory already exists, delete it")
		os.makedirs(directory, exist_ok=True)

		if self.ratio_test != 0:
			directory = self.path_dst + "/test"
			if os.path.exists(directory):
				raise ValueError(f"{directory} directory already exists, delete it")
			os.makedirs(directory, exist_ok=True)

	def _get_images(self):
		image_names = []

		directory = Path(self.path_src)
		for file in directory.iterdir(): # subdirectories are not traversed
			if file.is_file():
				_, extension = os.path.splitext(file)
				if extension in self.supported_img_formats:
					image_names.append(file.name)
				else:
					print(f"Warning: {file} is an unsupported file")

		self.length_total = len(image_names)
		return image_names

	def _get_random_sequence(self, image_names):
		length = len(image_names)
		numbers = list(range(0, length))
		train_sequence = random.sample(numbers, int(length*self.ratio_train))
		# print("train_sequence:", train_sequence)

		val_sequence = [x for x in numbers if x not in train_sequence]

		if self.ratio_test != 0:
			val_sequence = random.sample(val_sequence, int(length*self.ratio_val))
			# print("val_sequence:", val_sequence)

			test_sequence = [x for x in numbers if x not in train_sequence and x not in val_sequence]
			# print("test_sequence:", test_sequence)
		else:
			test_sequence = []

		image_sequences = [train_sequence, val_sequence, test_sequence]
		assert self.length_total == len(train_sequence) + len(val_sequence) + len(test_sequence), f"they are not equal: {self.length_total}:{len(train_sequence)}+{len(val_sequence)}+{len(test_sequence)}"
		return image_sequences

	def _letterbox(self, img):
		shape = img.shape[:2] # current shape: [height, width, channel]
		new_shape = [self.shape[0], self.shape[1]]

		# scale ratio (new / old)
		r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

		# compute padding
		new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
		dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] # wh padding
		dw /= 2 # divide padding into 2 sides
		dh /= 2

		if shape[::-1] != new_unpad: # resize
			img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

		top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
		left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
		img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill_value) # add border

		return img

	def _parse_csv(self, image_names):
		csv_contents = {}

		with open(self.csv_file, mode="r", encoding="utf-8") as file:
			csv_reader = csv.reader(file)

			for row in csv_reader:
				if row[2] in image_names:
					csv_contents[row[2]] = row
				else:
					raise FileNotFoundError(f"image not found: {row[2]}")

		assert len(csv_contents) == len(image_names), f"length mismatch: csv:{len(csv_contents)}; images:{len(image_names)}"
		return csv_contents

	def _copy_image(self):
		image_names = self._get_images()
		image_sequences = self._get_random_sequence(image_names) # train, val, test
		csv_contents = self._parse_csv(image_names)

		dirname = ["train", "val", "test"]
		index = [0, 1, 2]
		if self.ratio_test == 0:
			index = [0, 1]

		for idx in index:
			with open(self.path_dst+"/"+dirname[idx]+".csv", mode="w", newline="", encoding="utf-8") as file:
				writer = csv.writer(file)

				for i in image_sequences[idx]:
					writer.writerow(csv_contents[image_names[i]])

					image_name = self.path_src + "/" + image_names[i]
					dst_dir_name =self.path_dst + "/" + dirname[idx]
					# print(f"image_name:{image_name}; dst_dir_name:{dst_dir_name}")

					if not self.is_resize: # only copy
						shutil.copy(image_name, dst_dir_name)
					else: # resize, scale the image proportionally
						img = cv2.imread(image_name) # BGR
						if img is None:
							raise FileNotFoundError(f"image not found: {image_name}")

						img = self._letterbox(img)
						cv2.imwrite(dst_dir_name+"/"+image_names[i], img)

	def _cal_mean_std(self):
		imgs = []
		std_reds = []
		std_greens = []
		std_blues = []

		directory = Path(self.path_dst + "/train/")
		for file in directory.iterdir(): # subdirectories are not traversed
			if file.is_file():
				img = cv2.imread(str(file))
				if img is None:
					raise FileNotFoundError(f"image not found: {file}")
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # bgr -> rgb
				imgs.append(img)

				img_array = np.array(img)
				std_reds.append(np.std(img_array[:,:,0]))
				std_greens.append(np.std(img_array[:,:,1]))
				std_blues.append(np.std(img_array[:,:,2]))

		arr = np.array(imgs)
		# print("arr.shape:", arr.shape)
		self.mean = np.mean(arr, axis=(0, 1, 2)) / 255
		self.std = [np.mean(std_reds) / 255, np.mean(std_greens) / 255, np.mean(std_blues) / 255] # R,G,B

	def __call__(self):
		self._create_dir()
		self._copy_image()
		self._cal_mean_std()

	def get_mean_std(self):
		"""get the mean and variance"""
		return self.mean, self.std

def split_regression_dataset(src_dataset_path, dst_dataset_path, csv_file, resize, fill_value, ratios):
	split = SplitRegressionDataset(path_src=src_dataset_path, path_dst=dst_dataset_path, csv_file=csv_file, ratios=ast.literal_eval(ratios))

	if resize != "(0,0)":
		# print("resize:", type(ast.literal_eval(resize))) # str to tuple
		split.resize(shape=ast.literal_eval(resize), value=ast.literal_eval(fill_value))

	split()
	mean, std = split.get_mean_std()
	print(f"mean: {mean}; std: {std}")

if __name__ == "__main__":
	split = SplitRegressionDataset(path_src="../../data/database/regression/FeO", path_dst="datasets/regression", csv_file="../../data/database/regression/FeO.csv", ratios=(0.9, 0.05, 0.05))
	split.resize(shape=(64,512))

	split()

	mean, std = split.get_mean_std()
	print(f"mean: {mean}; std: {std}")
	print("====== execution completed ======")
