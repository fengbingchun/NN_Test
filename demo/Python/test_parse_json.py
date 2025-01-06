import argparse
import colorama
import os
from pathlib import Path
import shutil
import json

def parse_args():
	parser = argparse.ArgumentParser(description="parse json file")
	parser.add_argument("--src_path", required=True, type=str, help="source json path")
	parser.add_argument("--dst_path", type=str, help="the path of the destination json")
	parser.add_argument("--suffix", type=str, help="file name suffix")
	parser.add_argument("--rename", action="store_true", help="whether to rename the file")
	parser.add_argument("--category_names", type=str, help="category names, separated by commas")

	args = parser.parse_args()
	return args

def parse_json1(src_path, dst_path, img_suffix, rename):
	os.makedirs(dst_path, exist_ok=True)

	path = Path(src_path)
	_, last_dir_name = os.path.split(path)
	print(f"last dir name: {last_dir_name}")
	count = 0

	for file in path.rglob("*.json"):
		shutil.copy(file, dst_path)

		file2 = str(file)[:-4] + img_suffix
		# print(f"file2: {file2}")
		shutil.copy(file2, dst_path)

		if rename == True:
			name_json = os.path.basename(file)
			name_img = str(name_json)[:-4] + img_suffix
			# print(f"name: {name_json}; {name_img}")

			new_name_json = dst_path+"/"+last_dir_name+"_"+name_json
			new_name_img = dst_path+"/"+last_dir_name+"_"+name_img
			os.rename(dst_path+"/"+name_json, new_name_json)
			os.rename(dst_path+"/"+name_img, new_name_img)

			with open(new_name_json, "r", encoding="utf-8") as f:
				data = json.load(f)
				data["imagePath"] = last_dir_name+"_"+name_img

			with open(new_name_json, "w", encoding="utf-8") as f:
				json.dump(data, f, ensure_ascii=False, indent=2)

		count += 2

	print(f"number fo files copied: {count}")

def parse_json2(src_path, category_names):
	category_list = category_names.split(",")
	# print(f"category list: {category_list}")
	category_dict = {item: 0 for item in category_list}
	# print(f"category dict: {category_dict}")

	for file in Path(src_path).rglob("*.json"):
		with open(str(file), "r", encoding="utf-8") as f:
			data = json.load(f)
			# print(data["shapes"][0]); raise
			for value in data["shapes"]:
				# print(value["label"])
				category_dict[value["label"]] += 1

	for category in category_list:
		print(f"{category}: {category_dict[category]}")

def parse_json3(src_path):
	names = []
	for file in Path(src_path).rglob("*.json"):
		with open(str(file), "r", encoding="utf-8") as f:
			data = json.load(f)
			# if "shapes" not in data:
			if len(data["shapes"]) == 0:
				names.append(file)

	print(f"names: {names}; length: {len(names)}")
	for name in names:
		name.unlink() # delete file

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	parse_json3(args.src_path)

	print(colorama.Fore.GREEN + "====== execution completed ======")
