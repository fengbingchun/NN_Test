import cv2
import os
from pathlib import Path
import argparse
import colorama
import shutil

def parse_args():
	parser = argparse.ArgumentParser(description="image related operations")
	parser.add_argument("--src_path", required=True, type=str, help="images src path")
	parser.add_argument("--dst_path", type=str, help="images dst/src path")
	parser.add_argument("--interval", type=int, help="specify the interval between images to be taken")

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
	# print(f"len: {len(images)}; name: {images[0]}")
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

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	directory = Path(args.src_path)
	if not directory.is_dir():
		raise FileNotFoundError(colorama.Fore.RED + f"the specified directory does not exist: {args.src_path}")

	rename_images_prefix_dirname(args.src_path, args.dst_path)

	print(colorama.Fore.GREEN + "====== execution completed ======")
