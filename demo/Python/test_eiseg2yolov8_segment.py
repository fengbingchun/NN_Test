import os
import json
import argparse
import colorama
import random
import shutil
import cv2

# Blog: https://blog.csdn.net/fengbingchun/article/details/139377787

# supported image formats
img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")

def parse_args():
	parser = argparse.ArgumentParser(description="json(EISeg) to txt(YOLOv8)")

	parser.add_argument("--dir", required=True, type=str, help="images directory, all json files are in the label directory, and generated txt files are also in the label directory")
	parser.add_argument("--labels", required=True, type=str, help="txt file that hold indexes and labels, one label per line, for example: face 0")
	parser.add_argument("--val_size", default=0.2, type=float, help="the proportion of the validation set to the overall dataset:[0., 0.5]")
	parser.add_argument("--name", required=True, type=str, help="the name of the dataset")

	args = parser.parse_args()
	return args

def get_labels_index(name):
	labels = {} # key,value
	with open(name, "r") as file:
		for line in file:
			# print("line:", line)

			key_value = []
			for v in line.split(" "):
				# print("v:", v)
				key_value.append(v.replace("\n", "")) # remove line breaks(\n) at the end of the line
			if len(key_value) != 2:
				print(colorama.Fore.RED + "Error: each line should have only two values(key value):", len(key_value))
				continue

			labels[key_value[0]] = key_value[1]
		
	with open(name, "r") as file:
		line_num = len(file.readlines())

	if line_num != len(labels):
		print(colorama.Fore.RED + "Error: there may be duplicate lables:", line_num, len(labels))

	return labels

def get_json_files(dir):
	jsons = []
	for x in os.listdir(dir+"/label"):
		if x.endswith(".json"):
			jsons.append(x)

	return jsons

def parse_json(name_json, name_image):
	img = cv2.imread(name_image)
	if img is None:
		print(colorama.Fore.RED + "Error: unable to load image:", name_image)
		raise
	height, width = img.shape[:2]

	with open(name_json, "r") as file:
		data = json.load(file)

		objects=[]
		for i in range(0, len(data)):
			object = []
			object.append(data[i]["name"])
			object.append(data[i]["points"])
			objects.append(object)

	return width, height, objects

def write_to_txt(name_json, width, height, objects, labels):
	name_txt = name_json[:-len(".json")] + ".txt"
	# print("name txt:", name_txt)

	with open(name_txt, "w") as file:
		for obj in objects: # 0: name; 1: points
			if len(obj[1]) < 3:
				print(colorama.Fore.RED + "Error: must be at least 3 pairs:", len(obj[1]), name_json)
				raise
			
			if obj[0] not in labels:
				print(colorama.Fore.RED + "Error: unsupported label:", obj[0], labels)
				raise

			string = ""
			for pt in obj[1]:
				string = string + " " + str(round(pt[0] / width, 6)) + " " + str(round(pt[1] / height, 6))
			
			string = labels[obj[0]] + string + "\r"
			file.write(string)

def json_to_txt(dir, jsons, labels):
	for json in jsons:
		name_json = dir + "/label/" + json
		name_image = ""

		for format in img_formats:
			file = dir + "/" + json[:-len(".json")] + format
			if os.path.isfile(file):
				name_image = file
				break

		if not name_image:
			print(colorama.Fore.RED + "Error: required image does not exist:", json[:-len(".json")])
			raise
		# print("name image:", name_image)

		width, height, objects = parse_json(name_json, name_image)
		# print(f"width: {width}; height: {height}; objects: {objects}")

		write_to_txt(name_json, width, height, objects, labels)


def get_random_sequence(length, val_size):
	numbers = list(range(0, length))
	val_sequence = random.sample(numbers, int(length*val_size))
	# print("val_sequence:", val_sequence)

	train_sequence = [x for x in numbers if x not in val_sequence]
	# print("train_sequence:", train_sequence)

	return train_sequence, val_sequence

def get_files_number(dir):
	count = 0
	for file in os.listdir(dir):
		if os.path.isfile(os.path.join(dir, file)):
			count += 1

	return count

def split_train_val(dir, jsons, name, val_size):
	if val_size > 0.5 or val_size < 0.01:
		print(colorama.Fore.RED + "Error: the interval for val_size should be:[0.01, 0.5]:", val_size)
		raise

	dst_dir_images_train = "datasets/" + name + "/images/train"
	dst_dir_images_val = "datasets/" + name + "/images/val"
	dst_dir_labels_train = "datasets/" + name + "/labels/train"
	dst_dir_labels_val = "datasets/" + name + "/labels/val"

	try:
		os.makedirs(dst_dir_images_train) #, exist_ok=True
		os.makedirs(dst_dir_images_val)
		os.makedirs(dst_dir_labels_train)
		os.makedirs(dst_dir_labels_val)
	except OSError as e:
		print(colorama.Fore.RED + "Error: cannot create directory:", e.strerror)
		raise

	# print("jsons:", jsons)
	train_sequence, val_sequence = get_random_sequence(len(jsons), val_size)

	for index in train_sequence:
		for format in img_formats:
			file = dir + "/" + jsons[index][:-len(".json")] + format
			# print("file:", file)
			if os.path.isfile(file):
				shutil.copy(file, dst_dir_images_train)
				break

		file = dir + "/label/" + jsons[index][:-len(".json")] + ".txt"
		if os.path.isfile(file):
			shutil.copy(file, dst_dir_labels_train)

	for index in val_sequence:
		for format in img_formats:
			file = dir + "/" + jsons[index][:-len(".json")] + format
			if os.path.isfile(file):
				shutil.copy(file, dst_dir_images_val)
				break

		file = dir + "/label/" + jsons[index][:-len(".json")] + ".txt"
		if os.path.isfile(file):
			shutil.copy(file, dst_dir_labels_val)

	num_images_train = get_files_number(dst_dir_images_train)
	num_images_val = get_files_number(dst_dir_images_val)
	num_labels_train = get_files_number(dst_dir_labels_train)
	num_labels_val = get_files_number(dst_dir_labels_val)

	if  num_images_train + num_images_val != len(jsons) or num_labels_train + num_labels_val != len(jsons):
		print(colorama.Fore.RED + "Error: the number of files is inconsistent:", num_images_train, num_images_val, num_labels_train, num_labels_val, len(jsons))
		raise


def generate_yaml_file(labels, name):
	path = os.path.join("datasets", name, name+".yaml")
	# print("path:", path)
	with open(path, "w") as file:
		file.write("path: ../datasets/%s # dataset root dir\n" % name)
		file.write("train: images/train # train images (relative to 'path')\n")
		file.write("val: images/val  # val images (relative to 'path')\n")
		file.write("test: # test images (optional)\n\n")

		file.write("# Classes\n")
		file.write("names:\n")
		for key, value in labels.items():
			# print(f"key: {key}; value: {value}")
			file.write("  %d: %s\n" % (int(value), key))


if __name__ == "__main__":
	colorama.init()
	args = parse_args()

	# 1. parse JSON file and write it to a TXT file
	labels = get_labels_index(args.labels)
	# print("labels:", labels)
	jsons = get_json_files(args.dir)
	# print(f"jsons: {jsons}; number: {len(jsons)}")
	json_to_txt(args.dir, jsons, labels)

	# 2. split the dataset
	split_train_val(args.dir, jsons, args.name, args.val_size)

	# 3. generate a YAML file
	generate_yaml_file(labels, args.name)

	print(colorama.Fore.GREEN + "====== execution completed ======")
