import cv2
import os
import random
import shutil
import numpy as np
import ast

class SplitClassifyDataset:
	"""split the classification dataset"""

	def __init__(self, path_src, path_dst, ratios=(0.8, 0.1, 0.1)):
		"""
		path_src: source dataset path
		path_dst: the path to the split dataset
		ratios: they are the ratios of train set, validation set, and test set, respectively 
		"""
		assert len(ratios) == 3, f"the length of ratios is not 3: {len(ratios)}"
		assert abs(ratios[0] + ratios[1] + ratios[2] - 1) < 1e-05, f"ratios sum must be 1: {ratios[0]}, {ratios[1]}, {ratios[2]}"

		self.path_src = path_src
		self.path_dst = path_dst
		self.ratio_train = ratios[0]
		self.ratio_val = ratios[1]
		self.ratio_test = ratios[2]

		self.is_resize = False
		self.fill_value = None
		self.shape = None

		self.length_total = None
		self.classes = None

		self.mean = None
		self.std = None

		self.supported_img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")

	def resize(self, value=(114,114,114), shape=(256,256)):
		"""
		value: fill value
		shape: the scaled shape
		"""
		self.is_resize = True
		self.fill_value = value
		self.shape = shape

	def _create_dir(self):
		self.classes = [name for name in os.listdir(self.path_src) if os.path.isdir(os.path.join(self.path_src, name))]

		for name in self.classes:
			directory = self.path_dst + "/train/" + name
			if os.path.exists(directory):
				raise ValueError(f"{directory} directory already exists, delete it")
			os.makedirs(directory, exist_ok=True)

			directory = self.path_dst + "/val/" + name
			if os.path.exists(directory):
				raise ValueError(f"{directory} directory already exists, delete it")
			os.makedirs(directory, exist_ok=True)

			if self.ratio_test != 0:
				directory = self.path_dst + "/test/" + name
				if os.path.exists(directory):
					raise ValueError(f"{directory} directory already exists, delete it")
				os.makedirs(directory, exist_ok=True)

	def _get_images(self):
		image_names = {}
		self.length_total = 0

		for class_name in self.classes:
			imgs = []
			for root, dirs, files in os.walk(os.path.join(self.path_src, class_name)):
				for file in files:
					_, extension = os.path.splitext(file)
					if extension in self.supported_img_formats:
						imgs.append(file)
					else:
						print(f"Warning: {self.path_src+'/'+class_name+'/'+file} is an unsupported file")

			image_names[class_name] = imgs
			self.length_total += len(imgs)

		return image_names

	def _get_random_sequence(self, image_names):
		image_sequences = {}

		for name in self.classes:
			length = len(image_names[name])
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

			image_sequences[name] = [train_sequence, val_sequence, test_sequence]

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

	def _copy_image(self):
		image_names = self._get_images()
		image_sequences = self._get_random_sequence(image_names) # train, val, test

		sum = 0
		for name in self.classes:
			for i in range(3):
				sum += len(image_sequences[name][i])
		assert self.length_total == sum, f"the length before and afeter the split must be equal: {self.length_total}:{sum}"

		for name in self.classes:
			dirname = ["train", "val", "test"]
			index = [0, 1, 2]
			if self.ratio_test == 0:
				index = [0, 1]

			for idx in index:
				for i in image_sequences[name][idx]:
					image_name = self.path_src + "/" + name + "/" + image_names[name][i]
					dst_dir_name =self.path_dst + "/" + dirname[idx] + "/" + name
					# print(image_name)
					if not self.is_resize: # only copy
						shutil.copy(image_name, dst_dir_name)
					else: # resize, scale the image proportionally
						img = cv2.imread(image_name) # BGR
						if img is None:
							raise FileNotFoundError(f"image not found: {image_name}")

						img = self._letterbox(img)
						cv2.imwrite(dst_dir_name+"/"+image_names[name][i], img)

	def _cal_mean_std(self):
		imgs = []
		std_reds = []
		std_greens = []
		std_blues = []

		for name in self.classes:
			dst_dir_name = self.path_dst + "/train/" + name + "/"

			for root, dirs, files in os.walk(dst_dir_name):
				for file in files:
					# print("file:", dst_dir_name+file)
					img = cv2.imread(dst_dir_name+file)
					if img is None:
						raise FileNotFoundError(f"image not found: {dst_dir_name}{file}")
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

def split_classify_dataset(src_dataset_path, dst_dataset_path, resize, fill_value, ratios):
	split = SplitClassifyDataset(path_src=src_dataset_path, path_dst=dst_dataset_path, ratios=ast.literal_eval(ratios))

	if resize != "(0,0)":
		# print("resize:", type(ast.literal_eval(resize))) # str to tuple
		split.resize(shape=ast.literal_eval(resize), value=ast.literal_eval(fill_value))

	split()
	mean, std = split.get_mean_std()
	print(f"mean: {mean}; std: {std}")

if __name__ == "__main__":
	split = SplitClassifyDataset(path_src="../../data/database/classify/melon", path_dst="datasets/melon_new_classify")
	split.resize(shape=(256,256))

	split()
	mean, std = split.get_mean_std()
	print(f"mean: {mean}; std: {std}")
	print("====== execution completed ======")
