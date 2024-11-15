import argparse
import colorama
import os
from pathlib import Path
import csv
import shutil

def parse_args():
	parser = argparse.ArgumentParser(description="parse txt file")
	parser.add_argument("--src_path", type=str, help="source txt path")
	parser.add_argument("--src_txt_name", type=str, help="source txt file name")
	parser.add_argument("--src_csv_name", type=str, help="source csv file name")
	parser.add_argument("--dst_path", type=str, help="the path of the destination file")
	parser.add_argument("--suffix", type=str, help="file name suffix")
	parser.add_argument("--prefix", type=str, help="file name prefix")

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

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	parse_txt2(args.src_txt_name, args.src_csv_name, args.dst_path)

	print(colorama.Fore.GREEN + "====== execution completed ======")
