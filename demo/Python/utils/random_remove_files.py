from pathlib import Path
import os
import random

def random_remove_files(path, remove_files_number, suffix="*"):
	_path = Path(path)
	if not _path.exists():
		raise ValueError(f"the specified path does't exist: {path}")
	if not _path.is_dir():
		raise ValueError(f"the specified path is not a directory: {path}")

	files = [file for file in _path.rglob(suffix) if os.path.isfile(file)]
	if len(files) < remove_files_number:
		raise ValueError(f"the number of files to be removed exceeds the actual number of files: {len(files)}:{remove_files_number}")

	removed_files = random.sample(files, remove_files_number)
	for file in removed_files:
		try:
			os.remove(file)
		except Exception as e:
			raise OSError(f"unable to remove file: {file}")

if __name__ == "__main__":
	random_remove_files("../../data/database/eight_five", 10, "*.jpg")
	print("====== execution completed ======")
