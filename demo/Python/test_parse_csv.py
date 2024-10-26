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

def parse_args():
	parser = argparse.ArgumentParser(description="parse csv file")
	parser.add_argument("--src_dataset_path", type=str, help="source dataset path")
	parser.add_argument("--dst_dataset_path", type=str, default="", help="the path of the destination dataset after split")

	args = parser.parse_args()
	return args

def create_dirs(dst_dataset_path, feo_dirs_name):
	for _, value in feo_dirs_name.items():
		os.makedirs(dst_dataset_path+"/"+value, exist_ok=True)

def extract_feo(csv_name):
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
			src_image_name = src_dataset_path+"/FeO/"+image
			shutil.copy(src_image_name, save_image_path)

def count_images(dataset_path):
	count = 0
	path = Path(dataset_path)
	for _ in path.rglob("*.jpg"):
		count += 1

	return count

def parse_csv(src_dataset_path, dst_dataset_path):
	feo_dirs_name = {"6.5":"six_five", "7.0":"seven_zero", "7":"seven_zero", "7.5":"seven_five", "8.0":"eight_zero", "8":"eight_zero", "8.5":"eight_five", "9.0":"nine_zero", "9":"nine_zero", "9.5":"nine_five", "10.0":"ten_zero", "10":"ten_zero", "10.5":"ten_five", "11.0":"eleven_zero", "11":"eleven_zero"}
	create_dirs(dst_dataset_path, feo_dirs_name)

	feo_images = extract_feo(src_dataset_path+"/FeO.sort.csv")

	copy_images(feo_dirs_name, feo_images, src_dataset_path, dst_dataset_path)

	src_images_count = count_images(src_dataset_path+"/FeO")
	dst_images_count = count_images(dst_dataset_path)
	assert src_images_count == dst_images_count, f"the number of images in the source path and dst path must be equal: {src_images_count} : {dst_images_count}"

	for feo, path in feo_dirs_name.items():
		if feo in {"7", "8", "9", "10", "11"}:
			continue
		count = count_images(dst_dataset_path+"/"+path)
		print(f"{feo} images count: {count}")
	print(f"total number of images: {dst_images_count}")


def parse_csv2(src_dataset_path, dst_dataset_path):
	src_csv_name = src_dataset_path + "/FeO.csv"
	src_images_name = src_dataset_path + "/FeO/"
	dst_csv_name = dst_dataset_path + "/FeO.csv"
	dst_images_name = dst_dataset_path + "/FeO/"

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
	src_csv_name = src_dataset_path + "/FeO.csv"
	src_images_name = src_dataset_path + "/FeO/"
	dst_csv_name = dst_dataset_path + "/FeO.csv"
	dst_images_name = dst_dataset_path + "/FeO/"

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
	src_csv_name = src_dataset_path + "/FeO.csv"
	src_images_name = src_dataset_path + "/FeO/"

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

	dst_csv_name = src_dataset_path + "/FeO.sort.csv"
	with open(dst_csv_name, mode="w", newline="", encoding="utf-8") as file:
		writer = csv.writer(file)

		for row in list_dst2:
			writer.writerow(row)

	print(f"length: list dst:{len(list_dst)}; list dst2:{len(list_dst2)}; list remove imgs:{len(remove_imgs)}")

def parse_csv6(src_dataset_path, dst_dataset_path): # first: run parse_csv5, second:remove some abnormal images
	src_csv_name = src_dataset_path + "/FeO.sort.csv"
	src_images_name = src_dataset_path + "/FeO/"

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

if __name__ == "__main__":
    # python test_parse_csv.py --src_dataset_path ../../data/database/regression --dst_dataset_path ../../data/database/regression_small
	colorama.init(autoreset=True)
	args = parse_args()

	parse_csv9(args.src_dataset_path, args.dst_dataset_path)

	print(colorama.Fore.GREEN + "====== execution completed ======")
