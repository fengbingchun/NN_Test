import colorama
import argparse
import os
from pathlib import Path
import csv
import shutil

def parse_args():
	parser = argparse.ArgumentParser(description="parse csv file")
	parser.add_argument("--src_dataset_path", type=str, help="source dataset path")
	parser.add_argument("--dst_dataset_path", type=str, help="the path of the destination dataset after split")

	args = parser.parse_args()
	return args

def create_dirs(dst_dataset_path, feo_dirs_name):
	for _, value in feo_dirs_name.items():
		os.makedirs(dst_dataset_path+"/"+value, exist_ok=True)

def extract_feo(csv_name, dst_dataset_path):
	# csv fields: date feo1 image_name feo2
	feos_images = {"6.5":[], "7.0":[], "7":[], "7.5":[], "8.0":[], "8":[], "8.5":[], "9.0":[], "9":[], "9.5":[], "10.0":[], "10":[], "10.5":[], "11.0":[], "11":[]}
	with open(csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			feos_images[row[3]].append(row[2])

	return feos_images

def copy_images(feo_dirs_name, feos_images, src_dataset_path, dst_dataset_path):
	for feo, images in feos_images.items():
		save_image_path = dst_dataset_path + "/" + feo_dirs_name[feo]
		for image in images:
			src_image_name = src_dataset_path+"/0/"+image
			shutil.copy(src_image_name, save_image_path)

			# src_image_name = image[:-len("_0.jpg")]+"_1.jpg"
			# src_image_name = src_dataset_path+"/1/"+src_image_name
			# shutil.copy(src_image_name, save_image_path)

def count_images(dataset_path):
	count = 0
	path = Path(dataset_path)
	for file in path.rglob("*.jpg"):
		count += 1

	return count

def parse_csv(src_dataset_path, dst_dataset_path):
	feo_dirs_name = {"6.5":"six_five", "7.0":"seven_zero", "7":"seven_zero", "7.5":"seven_five", "8.0":"eight_zero", "8":"eight_zero", "8.5":"eight_five", "9.0":"nine_zero", "9":"nine_zero", "9.5":"nine_five", "10.0":"ten_zero", "10":"ten_zero", "10.5":"ten_five", "11.0":"eleven_zero", "11":"eleven_zero"}
	create_dirs(dst_dataset_path, feo_dirs_name)

	feos1_images = extract_feo(src_dataset_path+"/YuKun1_CSV/YunKun1.csv", dst_dataset_path)
	feos2_images = extract_feo(src_dataset_path+"/YuKun2_CSV/YunKun2.csv", dst_dataset_path)

	copy_images(feo_dirs_name, feos1_images, src_dataset_path+"/YuKun1_CSV", dst_dataset_path)
	copy_images(feo_dirs_name, feos2_images, src_dataset_path+"/YuKun2_CSV", dst_dataset_path)

	src_images_count1 = count_images(src_dataset_path+"/YuKun1_CSV/0")
	src_images_count2 = count_images(src_dataset_path+"/YuKun2_CSV/0")
	src_images_count = src_images_count1 + src_images_count2
	dst_images_count = count_images(dst_dataset_path)
	assert src_images_count == dst_images_count, f"the number of images in the source path and dst path must be equal: {src_images_count} : {dst_images_count}"

	for feo, path in feo_dirs_name.items():
		if feo in {"7", "8", "9", "10", "11"}:
			continue
		count = count_images(dst_dataset_path+"/"+path)
		print(f"{feo} images count: {count}")
	print(f"total number of images: {dst_images_count}")

if __name__ == "__main__":
    # python test_parse_csv.py --src_dataset_path ../../data/database/YuKun --dst_dataset_path ../../data/database/YuKun_FeO
	colorama.init(autoreset=True)
	args = parse_args()

	parse_csv(args.src_dataset_path, args.dst_dataset_path)

	print(colorama.Fore.GREEN + "====== execution completed ======")
