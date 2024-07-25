import os
import json
import argparse
import colorama
import random
import shutil

# Blog: https://blog.csdn.net/fengbingchun/article/details/139012796

def parse_args():
	parser = argparse.ArgumentParser(description="json(LabelMe) to txt(YOLOv8)")

	parser.add_argument("--dir", required=True, type=str, help="images, json files, and generated txt files, all in the same directory")
	parser.add_argument("--labels", required=True, type=str, help="txt file that hold indexes and labels, one label per line, for example: face 0")
	parser.add_argument("--val_size", default=0.2, type=float, help="the proportion of the validation set to the overall dataset:[0., 0.2]; the proportion of the test set is the same as the validataion set")
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
	for x in os.listdir(dir):
		if x.endswith(".json"):
			jsons.append(x)

	return jsons

def parse_json(name):
	with open(name, "r") as file:
		data = json.load(file)

	width = data["imageWidth"]
	height = data["imageHeight"]
	# print(f"width: {width}; height: {height}")

	objects=[]
	for shape in data["shapes"]:
		if shape["shape_type"] != "rectangle":
			print(colorama.Fore.YELLOW + "Warning: only the rectangle type is supported:", shape["shape_type"])
			continue

		object = []
		object.append(shape["label"])
		object.append(shape["points"])
		objects.append(object)

	return width, height, objects

def get_box_width_height(box):
	dist = lambda val: max(val) - min(val)

	x = [pt[0] for pt in box]
	y = [pt[1] for pt in box]

	return min(x), min(y), dist(x), dist(y)

def bounding_box_normalization(width, height, objects, labels):
	boxes = []
	for object in objects:
		box = [] # class x_center y_center width height
		box.append(labels[object[0]])

		# print("point:", object[1])
		x_min, y_min, box_w, box_h = get_box_width_height(object[1])
		box.append(round((float(x_min + box_w / 2.0) / width), 6))
		box.append(round((float(y_min + box_h / 2.0) / height), 6))
		box.append(round(float(box_w / width), 6))
		box.append(round(float(box_h / height), 6))

		boxes.append(box)
	
	return boxes	

def write_to_txt(dir, json, width, height, objects, labels):
	boxes = bounding_box_normalization(width, height, objects, labels)
	# print("boxes:", boxes)

	name = json[:-len(".json")] + ".txt"
	# print("name:", name)
	with open(dir + "/" + name, "w") as file:
		for item in boxes:
			# print("item:", item)
			if len(item) != 5:
				print(colorama.Fore.RED + "Error: the length must be 5:", len(item))
				continue
			string = item[0] + " " + str(item[1]) + " " + str(item[2]) + " " + str(item[3]) + " " + str(item[4]) + "\r"
			file.write(string)

def json_to_txt(dir, jsons, labels):
	for json in jsons:
		name = dir + "/" + json
		# print("name:", name)

		width, height, objects = parse_json(name)
		# print(f"width: {width}; height: {height}; objects: {objects}")

		write_to_txt(dir, json, width, height, objects, labels)


def is_in_range(value, a, b):
	return a <= value <= b

def get_random_sequence(length, val_size):
	numbers = list(range(0, length))
	val_sequence = random.sample(numbers, int(length*val_size))
	# print("val_sequence:", val_sequence)

	test_sequence = [x for x in numbers if x not in val_sequence]
	test_sequence = random.sample(test_sequence, int(length*val_size))
	# print("test_sequence:", test_sequence)

	train_sequence = [x for x in numbers if x not in val_sequence and x not in test_sequence]
	# print("train_sequence:", train_sequence)

	return train_sequence, val_sequence, test_sequence

def get_files_number(dir):
	count = 0
	for file in os.listdir(dir):
		if os.path.isfile(os.path.join(dir, file)):
			count += 1

	return count

def split_train_val(dir, jsons, name, val_size):
	if is_in_range(val_size, 0., 0.2) is False:
		print(colorama.Fore.RED + "Error: the interval for val_size should be:[0., 0.2]:", val_size)
		raise

	dst_dir_images_train = "datasets/" + name + "/images/train"
	dst_dir_images_val = "datasets/" + name + "/images/val"
	dst_dir_images_test = "datasets/" + name + "/images/test"
	dst_dir_labels_train = "datasets/" + name + "/labels/train"
	dst_dir_labels_val = "datasets/" + name + "/labels/val"
	dst_dir_labels_test = "datasets/" + name + "/labels/test"

	try:
		os.makedirs(dst_dir_images_train) #, exist_ok=True
		os.makedirs(dst_dir_images_val)
		os.makedirs(dst_dir_images_test)
		os.makedirs(dst_dir_labels_train)
		os.makedirs(dst_dir_labels_val)
		os.makedirs(dst_dir_labels_test)
	except OSError as e:
		print(colorama.Fore.RED + "Error: cannot create directory:", e.strerror)
		raise

	# supported image formats
	img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")

	# print("jsons:", jsons)
	train_sequence, val_sequence, test_sequence = get_random_sequence(len(jsons), val_size)

	for index in train_sequence:
		for format in img_formats:
			file = dir + "/" + jsons[index][:-len(".json")] + format
			# print("file:", file)
			if os.path.isfile(file):
				shutil.copy(file, dst_dir_images_train)
				break

		file = dir + "/" + jsons[index][:-len(".json")] + ".txt"
		if os.path.isfile(file):
			shutil.copy(file, dst_dir_labels_train)

	for index in val_sequence:
		for format in img_formats:
			file = dir + "/" + jsons[index][:-len(".json")] + format
			if os.path.isfile(file):
				shutil.copy(file, dst_dir_images_val)
				break

		file = dir + "/" + jsons[index][:-len(".json")] + ".txt"
		if os.path.isfile(file):
			shutil.copy(file, dst_dir_labels_val)

	for index in test_sequence:
		for format in img_formats:
			file = dir + "/" + jsons[index][:-len(".json")] + format
			if os.path.isfile(file):
				shutil.copy(file, dst_dir_images_test)
				break
		file = dir + "/" + jsons[index][:-len(".json")] + ".txt"
		if os.path.isfile(file):
			shutil.copy(file, dst_dir_labels_test)

	num_images_train = get_files_number(dst_dir_images_train)
	num_images_val = get_files_number(dst_dir_images_val)
	num_images_test = get_files_number(dst_dir_images_test)
	num_labels_train = get_files_number(dst_dir_labels_train)
	num_labels_val = get_files_number(dst_dir_labels_val)
	num_labels_test = get_files_number(dst_dir_labels_test)

	if  num_images_train + num_images_val + num_images_test != len(jsons) or num_labels_train + num_labels_val + num_labels_test != len(jsons):
		print(colorama.Fore.RED + "Error: the number of files is inconsistent:", num_images_train, num_images_val, num_images_test, num_labels_train, num_labels_val, num_labels_test, len(jsons))
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
	# python test_labelme2yolov8.py --dir ../../data/database/melon_new_detect --labels ../../data/images/labels.txt --val_size 0.15 --name melon_new_detect
	colorama.init()
	args = parse_args()

	# 1. parse JSON file and write it to a TXT file
	labels = get_labels_index(args.labels)
	# print("labels:", labels)
	jsons = get_json_files(args.dir)
	# print("jsons:", jsons)
	json_to_txt(args.dir, jsons, labels)

	# 2. split the dataset
	split_train_val(args.dir, jsons, args.name, args.val_size)

	# 3. generate a YAML file
	generate_yaml_file(labels, args.name)

	print(colorama.Fore.GREEN + "====== execution completed ======")
