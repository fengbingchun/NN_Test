import colorama
import argparse
import os
from pathlib import Path
import csv
import shutil
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from scipy.interpolate import interp1d
import cv2

def parse_args():
	parser = argparse.ArgumentParser(description="parse csv file")
	parser.add_argument("--src_dataset_path1", type=str, help="source dataset path1")
	parser.add_argument("--src_dataset_path2", type=str, help="source dataset path2")
	parser.add_argument("--src_csv_file1", type=str, help="source csv file1")
	parser.add_argument("--src_csv_file2", type=str, help="source csv file1")
	parser.add_argument("--dst_csv_file", type=str, help="destination csv file")
	parser.add_argument("--dst_dataset_path1", type=str, default="", help="the path of the destination dataset1 after split")
	parser.add_argument("--dst_dataset_path2", type=str, default="", help="the path of the destination dataset2 after split")
	parser.add_argument("--prefix", type=str, help="file name prefix")
	parser.add_argument("--suffix", type=str, help="file name suffix")

	args = parser.parse_args()
	return args

def create_dirs(dst_dataset_path, dirs_name):
	for _, value in dirs_name.items():
		os.makedirs(dst_dataset_path+"/"+value, exist_ok=True)

def extract_images(csv_name):
	images = {"6.5":[], "7.0":[], "7":[], "7.5":[], "8.0":[], "8":[], "8.5":[], "9.0":[], "9":[], "9.5":[], "10.0":[], "10":[], "10.5":[], "11.0":[], "11":[]}
	with open(csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			images[row[3]].append(row[2])

	return images

def copy_images(dirs_name, target_images, src_dataset_path, dst_dataset_path):
	for key, images in target_images.items():
		save_image_path = dst_dataset_path + "/" + dirs_name[key]
		for image in images:
			src_image_name = src_dataset_path+"/target/"+image
			shutil.copy(src_image_name, save_image_path)

def count_images(dataset_path):
	count = 0
	path = Path(dataset_path)
	for _ in path.rglob("*.jpg"):
		count += 1

	return count

def parse_csv(src_dataset_path, dst_dataset_path):
	dirs_name = {"6.5":"six_five", "7.0":"seven_zero", "7":"seven_zero", "7.5":"seven_five", "8.0":"eight_zero", "8":"eight_zero", "8.5":"eight_five", "9.0":"nine_zero", "9":"nine_zero", "9.5":"nine_five", "10.0":"ten_zero", "10":"ten_zero", "10.5":"ten_five", "11.0":"eleven_zero", "11":"eleven_zero"}
	create_dirs(dst_dataset_path, dirs_name)

	target_images = extract_images(src_dataset_path+"/target.sort.csv")

	copy_images(dirs_name, target_images, src_dataset_path, dst_dataset_path)

	src_images_count = count_images(src_dataset_path+"/target")
	dst_images_count = count_images(dst_dataset_path)
	assert src_images_count == dst_images_count, f"the number of images in the source path and dst path must be equal: {src_images_count} : {dst_images_count}"

	for key, path in dirs_name.items():
		if key in {"7", "8", "9", "10", "11"}:
			continue
		count = count_images(dst_dataset_path+"/"+path)
		print(f"{key} images count: {count}")
	print(f"total number of images: {dst_images_count}")


def parse_csv2(src_dataset_path, dst_dataset_path):
	src_csv_name = src_dataset_path + "/target.csv"
	src_images_name = src_dataset_path + "/target/"
	dst_csv_name = dst_dataset_path + "/target.csv"
	dst_images_name = dst_dataset_path + "/target/"

	os.makedirs(dst_images_name)

	dates = set()
	lists = []

	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			if row[0] not in dates:
				dates.add(row[0])
				lists.append(row) # first

				shutil.copy(src_images_name+"/"+row[2], dst_images_name)

	if len(dates) != len(lists):
		raise ValueError(f"their length must be equal: {len(dates)} : {len(lists)}")

	with open(dst_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for l in lists:
			writer.writerow(l)

	print(f"number fo images to copy: {len(lists)}")

def parse_csv3(csv1, csv2):
	dates1 = set()
	dates2 = set()

	with open(csv1, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			if row[0] not in dates1:
				dates1.add(row[0])

	with open(csv2, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			if row[0] not in dates2:
				dates2.add(row[0])

	print(f"dates1 length:{len(dates1)}; dates2 length:{len(dates2)}")

	for item1 in dates1:
		for item2 in dates2:
			if item1 == item2:
				print(item1) # same item

def parse_csv4(src_dataset_path, dst_dataset_path):
	src_csv_name = src_dataset_path + "/target.csv"
	src_images_name = src_dataset_path + "/target/"
	dst_csv_name = dst_dataset_path + "/target.csv"
	dst_images_name = dst_dataset_path + "/target/"

	os.makedirs(dst_images_name)

	dates = set()
	list_src = []
	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			list_src.append(row)
			if row[0] not in dates:
				dates.add(row[0])

	print(f"dates length:{len(dates)}; list src length: {len(list_src)}")

	list_dst = []
	for item1 in dates:
		tmp = []
		for item2 in list_src:
			if item1 == item2[0]:
				tmp.append(item2)

		if len(tmp) == 1:
			list_dst.append(tmp[0]) # [[]] ==> []
		else:
			index = (len(tmp) + 1) // 2 - 1
			list_dst.append(tmp[index])
			# print(f"tmp: {tmp}; length: {len(tmp)};index: {index}; {tmp[index]}"); raise

	print(f"list dst length:{len(list_dst)}")

	with open(dst_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in list_dst:
			writer.writerow(row)
			shutil.copy(src_images_name+"/"+row[2], dst_images_name)

def parse_csv5(src_dataset_path, dst_dataset_path): # first run parse_csv4
	src_csv_name = src_dataset_path + "/target.csv"
	src_images_name = src_dataset_path + "/target/"

	list_src = []
	list_dst = []

	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			list_src.append(row)

	date_src = []
	date_dict = {}
	index = 0
	for row in list_src:
		date = datetime.strptime(row[0], "%Y/%m/%d %H:%M")
		date_dict[date] = index
		index += 1
		date_src.append(date)

	date_src.sort()

	for row in date_src:
		list_dst.append(list_src[date_dict[row]])

	list_dst2 = []
	remove_imgs = []
	for row in list_dst:
		date1 = datetime.strptime(row[0], "%Y/%m/%d %H:%M")
		name = row[2][:12]
		date2 = datetime.strptime(name, "%Y%m%d%H%M")

		if date1 < date2:
			print(colorama.Fore.RED, f"Error: date: {date1}, {date2}")
		diff = date1 - date2
		hours, remainder = divmod(diff.seconds, 3600)
		minutes, seconds = divmod(remainder, 60)
		# if hours > 2:
		# 	print(colorama.Fore.YELLOW, f"Warning: date: {date1}, {date2}: {hours}:{minutes}:{seconds}")

		if 1*60+59 <= hours*60+minutes <= 2*60+1:
			list_dst2.append(row)
		else:
			# print(f"{date1}; {date2}; {hours}:{minutes}:{seconds}")
			remove_imgs.append(row[2])

	for name in remove_imgs:
		file = Path(src_images_name+name)
		if file.exists():
			try:
				file.unlink()
			except Exception as e:
				raise f"delete image error: {file}"
		else:
			raise FileNotFoundError(f"image no exist: {file}")

	dst_csv_name = src_dataset_path + "/target.sort.csv"
	with open(dst_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in list_dst2:
			writer.writerow(row)

	print(f"length: list dst:{len(list_dst)}; list dst2:{len(list_dst2)}; list remove imgs:{len(remove_imgs)}")

def parse_csv6(src_dataset_path, dst_dataset_path): # first: run parse_csv5, second:remove some abnormal images
	src_csv_name = src_dataset_path + "/target.sort.csv"
	src_images_name = src_dataset_path + "/target/"

	path = Path(src_images_name)
	count = 0
	images_name = []

	for image in path.rglob("*.jpg"):
		image_name = image.name
		# print(image_name)
		images_name.append(image_name)
		count += 1
	print(f"images count:{count}")

	list_src = []
	list_dst = []

	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			list_src.append(row)

	count = 0
	for row in list_src:
		if row[2] in images_name:
			count += 1
			list_dst.append(row)
	print(f"result csv length:{count}")

	with open(src_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in list_dst:
			writer.writerow(row)


def parse_csv7(src_csv_name1, src_csv_name2):
	count1 = 0; count2 = 0
	list_src1 = []
	list_src2 = []

	with open(src_csv_name1, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			count1 += 1
			list_src1.append(row)

	with open(src_csv_name2, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			count2 += 1
			list_src2.append(row)

	print(f"count1: {count1}; count2: {count2}")
	# print(f"src1:{list_src1[0]}, {list_src1[1]}; src2:{list_src2[0]}, {list_src2[1]}")

	dir_name = "tmp3"
	os.makedirs(dir_name, exist_ok=True)

	for row1 in list_src1: # list_src1[1:]
		# print(i)
		csv_name = row1[0]
		csv_name = csv_name.replace(" ", "-").replace(":", "-") + ".csv"

		datetime1_1 = datetime.strptime(row1[0], "%Y-%m-%d %H:%M:%S")
		datetime1_2 = datetime.strptime(row1[1], "%Y-%m-%d %H:%M:%S")
		# print(f"time1:{datetime1_1}; time2:{datetime1_2}")

		with open(dir_name+"/"+csv_name, mode="w", newline="", encoding="utf-8") as file:
			writer = csv.writer(file)
			writer.writerow(["vtime", "vangle"])

			for row2 in list_src2:
				datetime2 = datetime.strptime(row2[1], "%Y-%m-%d %H:%M:%S")

				if datetime1_2 >= datetime2 and datetime1_1 <= datetime2:
					writer.writerow([row2[1], row2[0]])

def parse_csv8(src_csv_path, dst_png_path):
	os.makedirs(dst_png_path, exist_ok=True)
	matplotlib.rcParams['font.sans-serif'] = ['SimHei'] # show chinese

	times_new_csv = []
	with open("../../data/time-new.csv", mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			times_new_csv.append(row)
	print(f"times new csv length: {len(times_new_csv)}; {times_new_csv[0][0]} 到 {times_new_csv[0][1]}")

	path = Path(src_csv_path)
	for name in path.rglob("*.csv"):
		print(f"name: {name}")
		list_src = []
		with open(name, mode="r", newline="", encoding="utf-8") as file:
			csv_reader = csv.reader(file)
			for row in csv_reader:
				list_src.append(row)
		# print(f"value: {list_src[1]}")

		t = os.path.splitext(os.path.basename(name))[0]
		t = t[:10] + " " + t[11:].replace("-", ":")
		# print(f"t: {t}")
		datetime1 = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")

		for row in times_new_csv:
			datetime2 = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
			if datetime1 == datetime2:
				title = row
				break

		times = []
		values = []
		for row in list_src[1:]:
			times.append(datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S"))
			values.append(row[1])
		times = list(reversed(times))
		values = list(reversed(values))
		values_int = [int(item) for item in values]
		# print(f"time: {times[0]}; value: {values[0]}")

		fig, ax = plt.subplots()
		ax.plot(times, values_int, marker="o")
		ax.set_xlabel("Time")
		ax.set_ylabel("Value")
		diff = datetime.strptime(title[1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(title[0], '%Y-%m-%d %H:%M:%S')
		hours = int(diff.total_seconds() // 3600)
		miuntes = int((diff.total_seconds() % 3600) // 60)
		seconds = int(diff.total_seconds() % 60)
		# print(f"hours:{hours}; minutes:{miuntes}; seconds:{seconds}")
		if hours == 0:
			plt.title(str(title[0]) + " 到 " + str(title[1]) + "\n用时: " + str(miuntes) + "分钟 " + str(seconds) + "秒")
		else:
			plt.title(str(title[0]) + " 到 " + str(title[1]) + "\n用时: " + str(hours) + "小时 " + str(miuntes) + "分钟 " + str(seconds) + "秒")

		fig.autofmt_xdate()

		plt.savefig(dst_png_path+"/"+os.path.basename(name)+".png")
		# plt.show()
		plt.close(fig)

def parse_csv9(src_csv_name, dst_csv_name):
	times_new_csv = []
	with open(src_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			times_new_csv.append(row)
	print(f"times new csv length: {len(times_new_csv)}; {times_new_csv[0][0]} 到 {times_new_csv[0][1]}")

	with open(dst_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["开始时间", "结束时间", "用时", "是否正常", "图像名"])

		for row in times_new_csv:
			diff = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S') - datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
			hours = int(diff.total_seconds() // 3600)
			miuntes = int((diff.total_seconds() % 3600) // 60)
			seconds = int(diff.total_seconds() % 60)
			if hours == 0:
				times = str(miuntes) + "分钟" + str(seconds) + "秒"
			else:
				times = str(hours) + "小时" + str(miuntes) + "分钟" + str(seconds) + "秒"
			# print(f"times: {times}")

			name = row[0].replace(" ", "-").replace(":", "-") + ".csv.png"
			# print(f"name: {name}")

			writer.writerow([row[0], row[1], times, "", name])

	# csv to excel
	df = pd.read_csv(dst_csv_name)
	df.to_excel(dst_csv_name+".xlsx", index=False)

def write_new_csv(image_name, time_start, time_end, lists_angle, save_path):
	csv_name = image_name[:-4]
	print(f"csv name: {csv_name}")

	time_start = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S")
	time_end = datetime.strptime(time_end, "%Y-%m-%d %H:%M:%S")
	time_consuming = str(int((time_end - time_start).total_seconds()))
	# print(f"time start: {time_start}; time end: {time_end}; time-consuming: {time_consuming}")

	with open(save_path+"/"+csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)
		writer.writerow(["开始时间", "结束时间", "用时(秒)", "角度"])

		count = 0
		for row in lists_angle:
			tmp = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
			if time_end >= tmp and time_start <= tmp:
				if count == 0:
					writer.writerow([time_start, time_end, time_consuming, row[0]])
					count += 1
				else:
					writer.writerow(["", "", "", row[0]])

def parse_csv10(src_excel_name, dst_png_path):
	df = pd.read_excel(src_excel_name)
	csv_name = src_excel_name + ".csv"
	df.to_csv(csv_name, index=False)

	lists_csv = []
	with open(csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			lists_csv.append(row)
	print(f"len: {len(lists_csv)}; value: {lists_csv[1]}")

	normal = dst_png_path + "/正常"
	abnormal = dst_png_path + "/异常"
	other = dst_png_path + "/其它"
	other2 = dst_png_path + "/未标注"
	os.makedirs(normal, exist_ok=True)
	os.makedirs(abnormal, exist_ok=True)
	os.makedirs(other, exist_ok=True)
	os.makedirs(other2, exist_ok=True)

	angle_csv_name = "../../data/value-new.csv"
	lists_angle = []
	with open(angle_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			lists_angle.append(row)
	print(f"lists angle len: {len(lists_angle)}; value: {lists_angle[0]}")

	src_images_path = "tmp7"
	for row in lists_csv[1:]:
		if row[3] == "正常":
			shutil.copy(src_images_path+"/"+row[4], normal)
			write_new_csv(row[4], row[0], row[1], lists_angle, normal)
		elif row[3] == "异常":
			shutil.copy(src_images_path+"/"+row[4], abnormal)
			write_new_csv(row[4], row[0], row[1], lists_angle, abnormal)
		elif row[3] == "":
			shutil.copy(src_images_path+"/"+row[4], other2)
			# write_new_csv(row[4], row[0], row[1], lists_angle, other2)
		else:
			shutil.copy(src_images_path+"/"+row[4], other)
			write_new_csv(row[4], row[0], row[1], lists_angle, other)

def shorten_list(lst):
	lst1 = [0] + lst
	lst2 = lst + [0]

	lst3 = [(a + b) / 2 for a, b in zip(lst1, lst2)]
	lst4 = lst3[1:-1]
	# print(f"lst1: {lst1}; lst2:{lst2}, lst3:{lst3}, lst4:{lst4}")

	return lst4

def expand_list(lst):
	result = []

	for i in range(len(lst)-1):
		result.append(lst[i])
		average = (lst[i]+lst[i+1]) / 2
		result.append(average)

	result.append(lst[-1])
	# print(f"result: {result}")

	return result

def adjust_list_length():
	# shorten
	lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	target_length = 7
	while len(lst) > target_length:
		lst = shorten_list(lst)

	lst = [int(round(x)) for x in lst]
	# lst = [int(x+0.5) for x in lst]
	print(f"lst: {lst}")

	# expand
	lst = [1, 2, 3, 4, 5]
	while len(lst) < target_length:
		lst = expand_list(lst)

	while len(lst) > target_length:
		lst = shorten_list(lst)

	lst = [int(round(x)) for x in lst]
	# lst = [int(x+0.5) for x in lst]
	print(f"lst: {lst}")

def get_same_length(lists, target_length):
	lists = [np.array(lst) for lst in lists]
	adjusted_lists = []

	for lst in lists:
		length = len(lst)

		if length == target_length:
			adjusted_lists.append(lst)
		elif length < target_length:
			x = np.linspace(0, 1, length)
			x_new = np.linspace(0, 1, target_length)
			f = interp1d(x, lst, fill_value="extrapolate")
			lst_new = [int(v+0.5) for v in f(x_new)]
			# print(f"lst_new: {lst_new}")
			adjusted_lists.append(lst_new)
		else:
			indices = np.linspace(0, length-1, target_length, dtype=int)
			indices = np.clip(indices, 0, length-1)
			lst_new = lst[indices]
			# print(f"lst_new: {lst_new}")
			adjusted_lists.append(lst_new.tolist())

	return adjusted_lists

def parse_csv11(src_csv_path, dst_csv_path):
	path = Path(src_csv_path)
	for name in path.rglob("*.csv"):
		# print(f"name: {name}")
		lists_csv = []
		with open(name, mode="r", newline="", encoding="utf-8") as file:
			csv_reader = csv.reader(file)
			for row in csv_reader:
				lists_csv.append(row)
		print(f"csv: {lists_csv[0]}, {lists_csv[1]}")
		raise

	# lists = [
	# 	[53, 47, 53, 59, 65, 71, 77, 83, 89, 95, 99, 99],
	# 	[49, 55, 61, 67, 73, 73, 85, 98, 54, 49, 45],
	# 	[47, 54, 59, 90, 98, 98, 89, 51]
	# ]

	# adjusted_lists = get_same_length(lists, 10)
	# for i in range(len(lists)):
	# 	print(f"value1: {lists[i]}")
	# 	print(f"value2: {adjusted_lists[i]}")

def copy_csv_file(dst_csv_path, lists_csv_src1, lists_csv_src2):
	os.makedirs(dst_csv_path, exist_ok=True)

	for row1 in lists_csv_src1:
		name1 = os.path.splitext(os.path.basename(row1))[0]
		time1 = datetime.strptime(name1, "%Y%m%d%H%M%S")
		# print(f"time1: {time1}")

		for row2 in lists_csv_src2:
			name2 = os.path.splitext(os.path.basename(row2))[0]
			time2 = datetime.strptime(name2, "%Y-%m-%d-%H-%M-%S")
			# print(f"time2: {time2}")

			if time1 == time2:
				shutil.copy(row1, dst_csv_path)
				os.rename(dst_csv_path+"/"+name1+".csv", dst_csv_path+"/"+name2+"_temperature.csv")
				break

def parse_csv12(src_csv_path, dst_csv_path):
	csv_temperature = src_csv_path + "/temperature"
	csv_normal = src_csv_path + "/normal"
	csv_abnormal = src_csv_path + "/abnormal"

	path = Path(csv_normal)
	lists_normal = []
	for name in path.rglob("*.csv"):
		lists_normal.append(name)
	# print(f"name: {lists_normal[0]}, {lists_normal[1]}")

	path = Path(csv_abnormal)
	lists_abnormal = []
	for name in path.rglob("*.csv"):
		lists_abnormal.append(name)

	path = Path(csv_temperature)
	lists_temperature = []
	for name in path.rglob("*.csv"):
		lists_temperature.append(name)

	copy_csv_file(dst_csv_path+"/正常_温度_CSV", lists_temperature, lists_normal)
	copy_csv_file(dst_csv_path+"/异常_温度_CSV", lists_temperature, lists_abnormal)

def _get_suitable_name(time1, candidate):
	if len(candidate) == 1:
		return candidate[0][0]

	minseconds = 10000
	name = candidate[0][0]
	for v in candidate:
		diff = abs((time1 - v[1]).total_seconds() - 2*60*60)
		if diff < minseconds:
			minseconds = diff
			name = v[0]

	return name

def parse_csv13(src_csv_file, src_dataset_path, dst_dataset_path, prefix):
	values = []
	with open(src_csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			values.append(row)
	print(f"length: {len(values)}; value: {values[0]}")

	images_name = []
	for name in Path(src_dataset_path).rglob("*.jpg"):
		images_name.append(os.path.splitext(os.path.basename(name))[0])
	print(f"images count: {len(images_name)}; name: {images_name[0]}")

	os.makedirs(dst_dataset_path, exist_ok=True)
	minseconds = 2*60*60-1*60
	maxseconds = 2*60*60+5*60

	results = []

	for value in values:
		time1 = datetime.strptime(value[0], "%Y/%m/%d %H:%M")

		candidate = []
		for name in images_name:
			name1 = f"{name[:4]}-{name[4:6]}-{name[6:8]} {name[8:10]}:{name[10:12]}:{name[12:]}"
			time2 = datetime.strptime(name1, "%Y-%m-%d %H:%M:%S")
			diff = (time1 - time2).total_seconds()
			# print(f"time1: {time1}; time2: {time2}; diff: {diff}")
			if diff < 0:
				break
			if diff > minseconds and diff < maxseconds:
				candidate.append([name, time2])

		if len(candidate) == 0:
			continue

		name = _get_suitable_name(time1, candidate)
		# print(f"time1: {time1}; name: {name}")
		results.append([value[0], value[1], name])

	print(f"results length: {len(results)}")
	with open(dst_dataset_path+"/result.csv", mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in results:
			writer.writerow([row[0], row[1], prefix+"_"+row[2]+".jpg.png"])

			shutil.copy(src_dataset_path+"/"+row[2]+".jpg", dst_dataset_path)
			os.rename(dst_dataset_path+"/"+row[2]+".jpg", dst_dataset_path+"/"+prefix+"_"+row[2]+".jpg")
			shutil.copy(src_dataset_path+"/"+row[2]+".jpg.png", dst_dataset_path)
			os.rename(dst_dataset_path+"/"+row[2]+".jpg.png", dst_dataset_path+"/"+prefix+"_"+row[2]+".jpg.png")

def parse_csv14(src_csv_file):
	values = []
	with open(src_csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			values.append(row)
	print(f"length: {len(values)}; value: {values[0]}")

	flag = False
	minseconds = 10*60
	results = []
	for i in range(len(values)-1):
		if flag:
			flag = False
			continue

		time1 = datetime.strptime(values[i][0], "%Y/%m/%d %H:%M")
		time2 = datetime.strptime(values[i+1][0], "%Y/%m/%d %H:%M")
		diff = (time2 - time1).total_seconds()
		if diff < minseconds:
			flag = True
			print(f"time1: {time1}, value: {values[i][1]}; time2: {time2}, value: {values[i+1][1]}; diff: {diff}")
		else:
			flag = False
			results.append(values[i])

	if not flag:
		results.append(values[len(values)-1])
	print(f"results length: {len(results)}")

	path = Path(src_csv_file)
	path_name = path.parent
	file_name = path.name
	print(f"path name: {path_name}; file name: {file_name}")

	with open(str(path_name)+"/result_"+str(file_name), mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in results:
			writer.writerow(row)

def _cal_mean_std(dataset_path):
	imgs = []
	std_reds = []
	std_greens = []
	std_blues = []

	directory = Path(dataset_path)
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
	mean = np.mean(arr, axis=(0, 1, 2)) / 255
	std = [np.mean(std_reds) / 255, np.mean(std_greens) / 255, np.mean(std_blues) / 255] # R,G,B
	print(f"mean: {mean}; std: {std}")

def parse_csv15(src_dataset_path1, src_dataset_path2, dst_dataset_path1, dst_dataset_path2):
	src_train_csv_name = src_dataset_path1 + "/train.csv"
	src_val_csv_name = src_dataset_path1 + "/val.csv"

	src_train_values = []
	with open(src_train_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			src_train_values.append(row)
	print(f"train length: {len(src_train_values)}; value: {src_train_values[0]}")

	src_val_values = []
	with open(src_val_csv_name, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			src_val_values.append(row)
	print(f"val length: {len(src_val_values)}; value: {src_val_values[0]}")

	path = Path(src_dataset_path2)
	count = 0
	images_name = []

	for image in path.rglob("*.jpg.png"):
		image_name = image.name
		# print(image_name)
		images_name.append(image_name)
		count += 1
	print(f"images count:{count}")

	count = 0
	src_train_values_new = []
	for img1 in src_train_values:
		flag = False
		for img2 in images_name:
			if img1[2][:-6] == img2[:-8]:
				flag = True
				break

		if not flag:
			count += 1
			# print(f"train image not found: {img1[2]}")
		else:
			src_train_values_new.append(img1)
	print(f"train image not found count: {count}; src train values new: {len(src_train_values_new)}")

	count = 0
	src_val_values_new =[]
	for img1 in src_val_values:
		flag = False
		for img2 in images_name:
			if img1[2][:-6] == img2[:-8]:
				flag = True
				break

		if not flag:
			count += 1
			# print(f"val image not found: {img1[2]}")
		else:
			src_val_values_new.append(img1)
	print(f"val image not found count: {count}, src val values new: {len(src_val_values_new)}")

	os.makedirs(dst_dataset_path1+"/train", exist_ok=True)
	os.makedirs(dst_dataset_path1+"/val", exist_ok=True)

	dst1_train_csv_name = dst_dataset_path1 + "/train.csv"
	dst1_val_csv_name = dst_dataset_path1 + "/val.csv"

	with open(dst1_train_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in src_train_values_new:
			writer.writerow(row)
			shutil.copy(src_dataset_path1+"/train/"+row[2], dst_dataset_path1+"/train")

	with open(dst1_val_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in src_val_values_new:
			writer.writerow(row)
			shutil.copy(src_dataset_path1+"/val/"+row[2], dst_dataset_path1+"/val")

	os.makedirs(dst_dataset_path2+"/train", exist_ok=True)
	os.makedirs(dst_dataset_path2+"/val", exist_ok=True)

	dst2_train_csv_name = dst_dataset_path2 + "/train.csv"
	dst2_val_csv_name = dst_dataset_path2 + "/val.csv"

	with open(dst2_train_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for src_row in src_train_values_new:
			img_name = src_row[2]
			img_name = img_name[:-6]
			img_name += ".jpg.png"
			# print(f"img_name: {img_name}"); raise

			dst_row = [src_row[0], src_row[1], img_name]
			writer.writerow(dst_row)
			shutil.copy(src_dataset_path2+"/"+img_name, dst_dataset_path2+"/train")

	with open(dst2_val_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for src_row in src_val_values_new:
			img_name = src_row[2]
			img_name = img_name[:-6]
			img_name += ".jpg.png"
			# print(f"img_name: {img_name}"); raise

			dst_row = [src_row[0], src_row[1], img_name]
			writer.writerow(dst_row)
			shutil.copy(src_dataset_path2+"/"+img_name, dst_dataset_path2+"/val")

	_cal_mean_std(dst_dataset_path1+"/train")
	_cal_mean_std(dst_dataset_path2+"/train")

def parse_csv16(src_csv_file, src_dataset_path, suffix, dst_dataset_path):
	images_name1 = []
	with open(src_csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			images_name1.append(row[2])
	print(f"length: {len(images_name1)}; value: {images_name1[0]}")

	images_name2 = []
	count = 0
	path = Path(src_dataset_path)
	for img in path.rglob("*."+suffix):
		# print(f"img: {img}")
		images_name2.append(img)
		count += 1
	print(f"images count: {count}, name: {images_name2[0]}")

	os.makedirs(dst_dataset_path, exist_ok=True)

	for img1 in images_name1:
		flag = False
		for img2 in images_name2:
			name = str(img2.name)
			if str(img1)[:-8] == name[:-4]:
				# print(f"name: {name}")
				shutil.copy(img2, dst_dataset_path)
				flag = True
				break

		if not flag:
			raise ValueError(f"does't exist: {img1}")

def parse_csv17(src_csv_file, src_dataset_path, suffix, dst_datset_path):
	images_name1 = []
	count = 0
	path = Path(src_dataset_path)
	for img in path.rglob("*."+suffix):
		# print(f"img: {img}")
		images_name1.append(img)
		count += 1
	print(f"images count: {count}, name: {images_name1[0]}")

	images_name2 = []
	with open(src_csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			images_name2.append(row)
	print(f"length: {len(images_name2)}; value: {images_name2[0]}")

	path = Path(src_csv_file)
	dst_csv_file = dst_datset_path + "/" + str(path.name)
	print(f"dst csv file: {dst_csv_file}")

	with open(dst_csv_file, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for img1 in images_name1:
			name1 = str(img1.name)[:-4]
			flag = False

			for img2 in images_name2:
				name2 = img2[2][:-8]
				# print(f"name1: {name1}; name2: {name2}"); raise
				if name1 == name2:
					flag = True
					writer.writerow(img2)
					break

			if not flag:
				raise ValueError(f"mismatch: {img1}")

def parse_csv18(src_csv_file, src_dataset_path, suffix, dst_dataset_path):
	images_name1 = []
	with open(src_csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			images_name1.append(row[2])
	print(f"length: {len(images_name1)}; value: {images_name1[0]}")

	images_name2 = []
	count = 0
	path = Path(src_dataset_path)
	for img in path.rglob("*."+suffix):
		# print(f"img: {img}")
		images_name2.append(img)
		count += 1
	print(f"images count: {count}, name: {images_name2[0]}")

	os.makedirs(dst_dataset_path, exist_ok=True)

	for img1 in images_name1:
		flag = False
		for img2 in images_name2:
			name = str(img2.name)
			# print(f"name: {str(img1)[:-4]}; {name}"); raise
			if str(img1)[:-4] == name:
				# print(f"name: {name}")
				shutil.copy(img2, dst_dataset_path)
				flag = True
				break

		if not flag:
			print(colorama.Fore.YELLOW, f"does't exist: {str(img1)[:-4]}")

def parse_csv19(src_csv_file1, src_csv_file2, dst_csv_file):
	if not Path(src_csv_file1).exists() or not Path(src_csv_file2).exists():
		raise FileNotFoundError(f"file doesn't exist: {src_csv_file1} or {src_csv_file2}")

	images_name1 = []
	with open(src_csv_file1, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			images_name1.append(row)
	print(f"length: {len(images_name1)}; value: {images_name1[0]}")

	images_name2 = []
	with open(src_csv_file2, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			images_name2.append(row)
	print(f"length: {len(images_name2)}; value: {images_name2[0]}")

	if len(images_name1) != len(images_name2):
		raise ValueError(f"they must be of equal length: {len(images_name1)}:{len(images_name2)}")

	with open(dst_csv_file, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for i in range(len(images_name1)):
			if images_name1[i][2][:-4] != images_name2[i][0]:
				raise ValueError(f"mismatch: {images_name1[i][2][:-4]}:{images_name2[i][0]}")
			writer.writerow(images_name1[i][:-1] + images_name2[i])

def parse_csv20(src_csv_file, src_dataset_path, suffix, dst_dataset_path):
	path = Path(src_dataset_path)
	video_names = []
	for v in path.rglob("*."+suffix):
		video_names.append(v)
	print(f"video count: {len(video_names)}, name: {video_names[0]}")

	time_names = []
	with open(src_csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			time_names.append(row[0])
	print(f"time count: {len(time_names)}, name: {time_names[0]}")

	os.makedirs(dst_dataset_path, exist_ok=True)

	count = 0
	for name1 in time_names:
		flag = False
		time1 = datetime.strptime(name1, "%Y/%m/%d %H:%M")
		time11 = time1.replace(minute=0, second=0, microsecond=0)
		# print(f"time1: {time1}")

		for name2 in video_names:
			time2 = name2.name
			time2 = time2[2:-4]
			time2 = datetime.strptime(time2, "%Y%m%d%H%M%S")
			time21 = time2.replace(minute=0, second=0, microsecond=0)
			# print(f"time2: {time2}"); raise

			# if time11 == time21 and time1 >= time2:
			# 	# shutil.copy(name2, dst_dataset_path)
			# 	flag = True
			# 	break

			time22 = time2 + timedelta(hours=1)
			# print(f"time2: {time2}; time22: {time22}"); raise
			if time1 >= time2 and time1 <= time22:
				shutil.copy(name2, dst_dataset_path)
				flag = True
				break

		if not flag:
			print(f"missing video file: {name1}")
			count += 1

	print(f"found video file count: {len(time_names)-count}; missing video file count: {count}")

def parse_csv21(src_dataset_path, src_csv_file, suffix, dst_csv_file, dst_dataset_path):
	dates1 = []
	with open(src_csv_file, mode="r", newline="", encoding="utf-8") as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			dates1.append(row)
	print(f"date1 length: {len(dates1)}, row0: {dates1[0]}")

	dates2 = []
	for name in Path(src_dataset_path).rglob("*."+suffix):
		dates2.append(str(name.name))
	print(f"date2 length: {len(dates2)}, row0: {dates2[0]}")

	os.makedirs(dst_dataset_path, exist_ok=True)
	minseconds = 2*60*60-5*60 # 5
	maxseconds = 2*60*60+5*60
	count = 0
	results = []

	for date1 in dates1:
		time1 = datetime.strptime(date1[0], "%Y/%m/%d %H:%M")

		candidate = []
		for date2 in dates2:
			time2 = date2[:-(len(suffix)+1)]
			time2 = f"{time2[:4]}-{time2[4:6]}-{time2[6:8]} {time2[8:10]}:{time2[10:12]}:{time2[12:]}"
			# print(f"time2: {time2}"); raise
			time2 = datetime.strptime(time2, "%Y-%m-%d %H:%M:%S")
			diff = (time1 - time2).total_seconds()
			if diff <= 0:
				break
			if diff > minseconds and diff < maxseconds:
				candidate.append([date2, time2])

		if len(candidate) == 0:
			continue
		count += 1
		print(f"len candidate: {len(candidate)}, count: {count}")

		name = _get_suitable_name(time1, candidate)
		results.append([date1[0], date1[1], name])

	print(f"results length: {len(results)}; row0: {results[0]}")
	with open(dst_csv_file, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in results:
			writer.writerow(row)

			shutil.copy(src_dataset_path+"/"+row[2], dst_dataset_path)


if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	parse_csv21(args.src_dataset_path1, args.src_csv_file1, args.suffix, args.dst_csv_file, args.dst_dataset_path1)

	print(colorama.Fore.GREEN + "====== execution completed ======")
