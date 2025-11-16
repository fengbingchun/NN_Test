import argparse
import colorama
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Blog: https://blog.csdn.net/fengbingchun/article/details/154904636

def parse_args():
	parser = argparse.ArgumentParser(description="labelme to coco")
	parser.add_argument("--dataset_path", type=str, help="image dataset path")
	parser.add_argument("--json_file", type=str, help="image dataset json file, coco format")

	args = parser.parse_args()
	return args

class CocoDataset(Dataset):
	def __init__(self, json_file, image_dir, transforms=None):
		self.image_dir = image_dir
		self.transforms = transforms

		with open(json_file, mode="r", encoding="utf-8") as f:
			coco = json.load(f)

		self.image_infos = {img["id"]: img for img in coco["images"]}
		self.image_ids = list(self.image_infos.keys())

		self.annotations = {}
		for ann in coco["annotations"]:
			img_id = ann["image_id"]
			self.annotations.setdefault(img_id, []).append(ann)

	def __len__(self):
		return len(self.image_ids)

	def __getitem__(self, idx):
		img_id = self.image_ids[idx]
		img_info = self.image_infos[img_id]

		file_path = Path(self.image_dir) / Path(img_info["file_name"]).name
		if not file_path.exists():
			raise FileNotFoundError(colorama.Fore.RED + f"image not found: {file_path}")

		image = Image.open(file_path).convert("RGB")

		anns = self.annotations.get(img_id, [])
		boxes, labels, areas, iscrowd = [], [], [], []

		for ann in anns:
			x, y, w, h = ann["bbox"]
			if w <= 0 or h <= 0:
				continue

			# COCO bbox:[xmin, ymin, width, height]; PyTorch need:[xmin, ymin, xmax, ymax]
			boxes.append([x, y, x + w, y + h])
			labels.append(ann["category_id"])
			areas.append(ann.get("area", w * h))
			iscrowd.append(ann.get("iscrowd", 0))

		target = {
			"boxes": boxes,
			"labels": labels,
			"image_id": img_id,
			"areas": areas,
			"iscrowd": iscrowd,
		}

		if self.transforms is not None:
			image = self.transforms(image)

		return image, target

 # custom batch processing function to process different numbers of targets
def collate_fn(batch):
	return tuple(zip(*batch))

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	transform = transforms.Compose([
		transforms.ToTensor()
	])

	dataset = CocoDataset(args.json_file, args.dataset_path, transforms=transform)
	data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)

	for idx, (images, targets) in enumerate(data_loader):
		print(f"idx: {idx}")
		for i in range(len(images)):
			print(f"image{i} shape: {images[i].shape}")
			print(f"boxes{i}: {targets[i]['boxes']}")
			print(f"labels{i}: {targets[i]['labels']}")

	print(colorama.Fore.GREEN + "====== execution completed ======")
