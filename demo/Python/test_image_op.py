import cv2
import os
from pathlib import Path
import argparse
import colorama
import shutil
import matplotlib.pyplot as plt
import numpy as np
import ast
from datetime import datetime, timedelta

def parse_args():
	parser = argparse.ArgumentParser(description="image related operations")
	parser.add_argument("--src_path", required=True, type=str, help="images src path")
	parser.add_argument("--dst_path", type=str, help="images dst/src path")
	parser.add_argument("--interval", type=int, help="specify the interval between images to be taken")
	parser.add_argument("--suffix", type=str, help="image suffix")
	parser.add_argument("--rect", type=str, help="rect pos: x,y,width,height")

	args = parser.parse_args()
	return args

def get_images_min_max_shape(src_path):
	min_width = 1000; max_width = 0
	min_height = 1000; max_height = 0
	count = 0

	directory = Path(src_path)
	for item in directory.iterdir(): # subdirectories are not traversed
		if item.is_dir():
			print(colorama.Fore.YELLOW + f"{item} is a directory")
			continue

		if item.is_file():
			img = cv2.imread(str(item))
			if img is None:
				print(colorama.Fore.YELLOW + f"{item} is not an image")
				continue

			count += 1
			height, width = img.shape[:2]

			if min_width > width:
				min_width = width
				image_name_min_width = str(item)
			if max_width < width:
				max_width = width
				image_name_max_width = str(item)
			if min_height > height:
				min_height = height
				image_name_min_height = str(item)
			if max_height < height:
				max_height = height
				image_name_max_height = str(item)

	print(f"min width: {min_width}, name: {image_name_min_width}; max width: {max_width}, name: {image_name_max_width};"
	   f"min height: {min_height}, name: {image_name_min_height}; max height: {max_height}, name: {image_name_max_height}; image count: {count}")

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

def copy_images(src_path, dst_path, interval):
	images = _get_images(src_path)
	# print(f"len: {len(images)}; name: {images[0]}"); raise
	result = [images[i] for i in range(0, len(images), interval)]

	if os.path.exists(dst_path) and os.path.isdir(dst_path):
		print(colorama.Fore.YELLOW + f"the specified directory already exists: {dst_path}")
	os.makedirs(dst_path, exist_ok=True)

	for img in result:
		shutil.copy(src_path+"/"+img, dst_path)
	print(f"total number of images: {len(images)}; number of images to copy: {len(result)}")

def find_duplicate_images(src_path1, src_path2):
	imgs1 = _get_images(src_path1)
	imgs2 = _get_images(src_path2)

	count = 0
	for img1 in imgs1:
		for img2 in imgs2:
			if img1 == img2:
				count += 1
				print("name:", img1)
				break

	print("duplicate images count:", count)

def rename_images_prefix_dirname(src_path, dst_path):
	if os.path.exists(dst_path) and os.path.isdir(dst_path):
		print(colorama.Fore.YELLOW + f"the specified directory already exists: {dst_path}")
	os.makedirs(dst_path, exist_ok=True)

	path = Path(src_path)
	_, last_dir_name = os.path.split(path)
	print(f"last dir name: {last_dir_name}")

	images = _get_images(src_path)
	for name in images:
		shutil.copy(src_path+"/"+name, dst_path)
		rename = dst_path + "/" + last_dir_name + "_" + name
		os.rename(dst_path+"/"+name, rename)

	print(f"copy images count: {len(images)}")

def multidirs_copy_images(src_path, dst_path, interval):
	dirs = [str(d) for d in Path(src_path).glob("*") if d.is_dir() ]
	print(f" dirs count: {len(dirs)}; dir: {dirs[0]}")

	for dir in dirs:
		print(f"current dir: {dir}")
		copy_images(dir, dst_path, interval)

def _diff_pixels(pixel_values):
	diff = []
	for idx in range(1, len(pixel_values)):
		ret = pixel_values[idx] - pixel_values[idx-1]
		diff.append(ret)

	print(f"red values: {pixel_values}\ndiff: {diff}")
	return diff

def _get_key_points(image_names, pixel_values_red, diff_red):
	key_points = []
	points_index = []
	for idx in range(0, len(diff_red)):
		if diff_red[idx] > 20:
			key_points.append(image_names[idx+1])
			points_index.append(idx+1)
			# start_point = image_names[idx+1][:-4]
			# print(f"start point: {start_point}"); raise
	if len(key_points) == 0 or len(key_points) == 1:
		return

	results = []
	for idx in range(0, len(key_points)):
		if points_index[idx] + 3 < len(pixel_values_red):
			if pixel_values_red[points_index[idx]+1] > 200 and pixel_values_red[points_index[idx]+2] > 200 and pixel_values_red[points_index[idx]+3] > 200: # pixel_values_red[points_index[idx]] > 200 and and pixel_values_red[points_index[idx]+4] > 200:
				results.append(key_points[idx])

	if len(results) == 0 or len(results) == 1:
		return

	results2 = []
	results2.append(results[0])

	for idx in range(1, len(results)):
		start_time = datetime.strptime(results2[-1][:-4], "%Y%m%d%H%M%S")
		end_time = datetime.strptime(results[idx][:-4], "%Y%m%d%H%M%S")

		if end_time - start_time > timedelta(seconds=35):
			results2.append(results[idx])

	print(f"result points: {len(results2)}, {results2}")

def _draw_graph(image_names, pixel_values_gray, pixel_values_blue, pixel_values_green, pixel_values_red, name, dst_path):
	length_match = len(pixel_values_gray) == len(pixel_values_blue) == len(pixel_values_green) == len(pixel_values_red)
	# if not length_match:
	# 	raise ValueError(colorama.Fore.RED + f"they must be the same length: {len(pixel_values_gray)}, {len(pixel_values_blue)}, {len(pixel_values_green)}, {len(pixel_values_red)}")
	# print(f"length: {len(pixel_values_gray)}"); raise

	x_values = list(range(1, len(pixel_values_red))) # len(pixel_values_gray) + 1

	# diff_gray = _diff_pixels(pixel_values_gray)
	# diff_blue = _diff_pixels(pixel_values_blue)
	# diff_green = _diff_pixels(pixel_values_green)
	diff_red = _diff_pixels(pixel_values_red)

	_get_key_points(image_names, pixel_values_red, diff_red)

	fig = plt.figure()
	ax = fig.add_subplot()

	# ax.plot(x_values, diff_gray, label="gray", color="black") # pixel_values_gray
	# ax.plot(x_values, diff_blue, label="blue", color="blue")
	# ax.plot(x_values, diff_green, label="green", color="green")
	ax.plot(x_values, diff_red, "-o", label="red", color="red")
	ax.legend()

	ax.set_title("Comparison of pixel values in each channel")
	ax.set_xlabel("X-axis")
	ax.set_ylabel("Y-axis")
	plt.savefig(dst_path+"/"+name)
	# plt.show()
	plt.close(fig)

def _pixels_average(image, rect):
	roi = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
	mean = int(np.mean(roi))
	# print(f"mean: {mean}"); raise

	return mean

def images_split(src_path, suffix, rect, figure_name, dst_path):
	# if os.path.exists(dst_path) and os.path.isdir(dst_path):
	# 	print(colorama.Fore.YELLOW + f"the specified directory already exists: {dst_path}")
	os.makedirs(dst_path, exist_ok=True)

	# dir_blue = dst_path + "/blue"
	# dir_green = dst_path + "/green"
	# dir_red = dst_path + "/red"

	# os.makedirs(dir_blue, exist_ok=True)
	# os.makedirs(dir_green, exist_ok=True)
	# os.makedirs(dir_red, exist_ok=True)

	pixel_values_gray = []
	pixel_values_blue = []
	pixel_values_green = []
	pixel_values_red = []

	rect = ast.literal_eval(rect)
	image_names = []

	for file in Path(src_path).rglob("*."+suffix):
		name = file.name
		img = cv2.imread(str(file))
		if img is None:
			raise FileNotFoundError(colorama.Fore.RED + f"image not found: {file}")

		blue, green, red = cv2.split(img)
		# cv2.imwrite(dir_blue+"/"+name, blue)
		# cv2.imwrite(dir_green+"/"+name, green)
		# cv2.imwrite(dir_red+"/"+name, red)

		# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if rect[0] == 0 and rect[1] == 0 and rect[2] == 0 and rect[3] == 0:
			rect = list(rect)
			rect[2] = img.shape[1]
			rect[3] = img.shape[0]

		# pixel_values_gray.append(_pixels_average(gray, rect))
		# pixel_values_blue.append(_pixels_average(blue, rect))
		# pixel_values_green.append(_pixels_average(green, rect))
		pixel_values_red.append(_pixels_average(red, rect))

		image_names.append(str(name))

	_draw_graph(image_names, pixel_values_gray, pixel_values_blue, pixel_values_green, pixel_values_red, figure_name, dst_path)

def dir_images_split(src_path, suffix, rect, dst_path):
	root_path = Path(src_path)
	count = 0
	for dir in root_path.rglob("*"):
		if dir.is_dir() and dir != root_path:
			print(f"dir name: {dir}")
			name = str(dir.name)
			name += ".png"
			# print(f"name: {name}"); raise
			images_split(str(dir), suffix, rect, name, dst_path)
			count += 1

	if count == 0:
		name = str(Path(src_path).name)
		name += ".png"
		images_split(src_path, suffix, rect, name, dst_path)


if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	directory = Path(args.src_path)
	if not directory.is_dir():
		raise FileNotFoundError(colorama.Fore.RED + f"the specified directory does not exist: {args.src_path}")

	dir_images_split(args.src_path, args.suffix, args.rect, args.dst_path)

	print(colorama.Fore.GREEN + "====== execution completed ======")
