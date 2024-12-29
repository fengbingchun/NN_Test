import argparse
import colorama
import os
from pathlib import Path
import csv
import shutil
import cv2
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="parse txt file")
	parser.add_argument("--src_path", type=str, help="source txt path")
	parser.add_argument("--src_txt_name", type=str, help="source txt file name")
	parser.add_argument("--src_csv_name", type=str, help="source csv file name")
	parser.add_argument("--dst_path", type=str, help="the path of the destination file")
	parser.add_argument("--dst_csv_name", type=str, help="destination csv file name")
	parser.add_argument("--suffix", type=str, help="file name suffix")
	parser.add_argument("--prefix", type=str, help="file name prefix")
	parser.add_argument("--line_counts", type=int, help="specify how many lines the txt file has")
	parser.add_argument("--column_counts", type=int, help="specify how many columns the txt file has")
	parser.add_argument("--category_names", type=str, help="category names, separated by commas")

	args = parser.parse_args()
	return args

def parse_txt1(src_path):
	path = Path(src_path)

	for file in path.rglob("*.txt"):
		with open(file, "r") as f:
			lines_seen = set()
			duplicates = []

			for line in f:
				line = line.strip().lower()
				# print("line:", line)

				if line in lines_seen:
					duplicates.append(line)
				else:
					lines_seen.add(line)

		print("duplicates count:", len(duplicates))

def parse_txt2(src_txt_name, src_csv_name, dst_path):
	images_name = []
	with open(src_txt_name, "r") as file:
		for line in file:
			line = line.strip()
			# print(f"line: {line}"); raise
			images_name.append(line)
	print(f"txt file lines: {len(images_name)}, name: {images_name[0]}")

	dates_name = []
	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			dates_name.append(row)
	print(f"csv file lines: {len(dates_name)}; date: {dates_name[0]}")

	os.makedirs(dst_path, exist_ok=True)

	path = Path(src_csv_name)
	path_name = path.parent
	csv_name = path.name
	results = []
	count = 0

	for row1 in dates_name:
		flag = False
		for row2 in images_name:
			if row1[2] == row2 + ".png":
				# print(f"remove {row2}")
				flag = True
				count += 1
				break

		if not flag:
			results.append(row1)

	if count + len(results) != len(dates_name):
		raise ValueError(colorama.Fore.RED + f"length mismatch: {count} + {len(results)} != {len(dates_name)}")
	print(f"results count: {len(results)}; remove count: {count}")

	with open(dst_path+"/"+csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in results:
			writer.writerow(row)

			shutil.copy(str(path_name)+"/"+row[2], dst_path)

def parse_txt3(src_path):
	if not Path(src_path).exists():
		raise FileNotFoundError(colorama.Fore.RED + f"{src_path} doesn't exist")
	if not Path(src_path).is_dir():
		raise Exception(colorama.Fore.RED + f"{src_path} is not a directory")

	for name in Path(src_path).rglob("*.txt"):
		max_value = 0
		min_value = 9999999

		with open(name, "r", encoding="utf-8") as file:
			for line in file:
				columns = [int(column.strip()) for column in line.strip().split(',')]

				for value in columns:
					if value > max_value:
						max_value = value
					if value < min_value:
						min_value = value

		print(f"name: {name.name}; max vale: {max_value}; min value: {min_value}")

def _get_txt_list(name, line_counts, column_counts):
	temp = []

	with open(name, "r", encoding="utf-8") as file:
		lines = file.readlines()
		if len(lines) != line_counts:
			raise Exception(colorama.Fore.RED + f"name: {name.name}, mismatch line counts: {line_counts}:{len(lines)}")

		for line in lines:
			columns = [int(column.strip()) for column in line.strip().split(',')]
			if len(columns) != column_counts:
				raise Exception(colorama.Fore.RED + f"name: {name.name}, mismatch column counts: {column_counts}:{len(lines)}")
			temp.append(columns)

	assert len(temp) == line_counts and len(temp[0]) == column_counts, colorama.Fore.RED + f"length mismatch: {len(temp)}:{line_counts}; {len(temp[0])}:{column_counts}"
	return temp

def _txt2image(name, temp, line_counts, column_counts, dst_path):
	name = name[:-4] + ".png"

	data = np.array(temp)
	data //= 8000 # 1000 * 8
	data = data.astype(np.int8)
	data = data[::-1, :]

	height, width = data.shape
	assert height == line_counts and width == column_counts, colorama.Fore.RED + f"length mismatch: {height}:{line_counts}; {width}:{column_counts}"

	os.makedirs(dst_path, exist_ok=True)
	cv2.imwrite(dst_path+"/"+name, data)

def _crop(name, temp, src_csv_name, dst_path):
	name = name[:-4] + ".jpg"

	data = np.array(temp).astype(np.float32)
	data = data[::-1, :]
	data /= 1000
	# print(f"data type: {data.dtype}, {data[0]}")

	infos = []
	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			infos.append(row)
	# print(f"infos length: {len(infos)}; value: {infos[0]}")

	flag = False
	for info in infos:
		if info[2] == name:
			flag = True
			x1 = int(info[3]); y1 = int(info[4])
			x2 = int(info[5]); y2 = int(info[6])
			# print(f"x1:{x1}; y1:{y1}; x2:{x2}; y2:{y2}")
			assert x2 - x1 == (y2 - y1) * 8, colorama.Fore.RED + f"rect mismatch: left top: ({x1}, {y1}); right bottom: ({x2},{y2})"
			break

	if not flag:
		raise Exception(colorama.Fore.RED + f"file mismatch: {name}")

	crop = data[y1:y2, x1:x2]
	# print(f"crop shape: {crop.shape}")
	os.makedirs(dst_path, exist_ok=True)
	crop2 = crop / 8
	crop2 = crop2.astype(np.int8)
	cv2.imwrite(dst_path+"/"+name, crop2)

	return crop

def _cal_proportion(src_csv_name, temp_dict, dst_csv_name):
	infos = []
	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			infos.append(row)
	print(f"infos length: {len(infos)}; value: {infos[0]}")

	results = []
	for info in infos:
		flag = False
		for key, value in temp_dict.items():
			if info[2][:-4] == key[:-4]:
				height, width = value.shape
				# print(f"height: {height}, width: {width}"); raise
				max_value = np.max(value)
				min_value = np.min(value)
				mean_value = np.mean(value)
				# print(f"max value: {max_value}; min value: {min_value}; mean value: {mean_value}"); raise

				bins = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]).astype(np.float32)
				digitized = np.digitize(value, bins) - 1
				counts = np.bincount(digitized.flatten(), minlength=len(bins) - 1)
				if np.sum(counts) != height*width:
					raise Exception(colorama.Fore.RED + f"sum mismatch: {np.sum(counts)} : {height*width}")

				counts = counts.astype(np.float32)
				counts /= (height * width)

				result = info
				result.extend([max_value, min_value, mean_value])
				for i in range(len(counts)):
					result.extend([counts[i]])

				flag = True
				break

		if not flag:
			raise Exception(colorama.Fore.RED + f"file mismatch: {info[2]}")
		results.append(result)

	with open(dst_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerows(results)

def parse_txt4(src_path, src_csv_name, line_counts, column_counts, dst_path, dst_csv_name):
	if not Path(src_path).exists():
		raise FileNotFoundError(colorama.Fore.RED + f"{src_path} doesn't exist")
	if not Path(src_path).is_dir():
		raise Exception(colorama.Fore.RED + f"{src_path} is not a directory")

	temp_dict = {}
	for file in Path(src_path).rglob("*.txt"):
		# print(f"file: {file}"); raise
		temperature = _get_txt_list(file, line_counts, column_counts)
		_txt2image(file.name, temperature, line_counts, column_counts, dst_path)
		crop = _crop(file.name, temperature, src_csv_name, dst_path)
		temp_dict[file.name] = crop

	_cal_proportion(src_csv_name, temp_dict, dst_csv_name)

def parse_txt5(src_path, category_names):
	category_list = category_names.split(",")
	# print(f"category list: {category_list}")
	category_dict = {item: 0 for item in category_list}
	# print(f"category dict: {category_dict}")

	for file in Path(src_path).rglob("*.txt"):
		with open(file, "r") as f:
			for line in f:
				line = line.strip()
				# print(f"value: {line.split()[0]}"); raise
				category_dict[line.split()[0]] += 1

	for category in category_list:
		print(f"{category}: {category_dict[category]}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	parse_txt5(args.src_path, args.category_names)

	print(colorama.Fore.GREEN + "====== execution completed ======")
