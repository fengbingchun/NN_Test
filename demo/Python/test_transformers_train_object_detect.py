import argparse
import colorama
import json
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor, TrainingArguments, Trainer, AutoModelForObjectDetection, AutoImageProcessor
import cv2

# Blog: https://blog.csdn.net/fengbingchun/article/details/156825572

def parse_args():
	parser = argparse.ArgumentParser(description="transformers train object detect test code")
	parser.add_argument("--task", required=True, type=str, choices=["train", "predict"], help="specify what kind of task")
	parser.add_argument("--pretrained_model", type=str, default="facebook/detr-resnet-50", help="pretrained model loaded during training")
	parser.add_argument("--image_name", type=str, help="image name")
	parser.add_argument("--threshold", type=float, default=0.1, help="threshold")
	parser.add_argument("--train_dataset_path", type=str, default="melon_detect/train", help="train dataset path")
	parser.add_argument("--train_dataset_json", type=str, default="melon_detect/train.json", help="train dataset json file, coco format")
	parser.add_argument("--val_dataset_path", type=str, default="melon_detect/val", help="val dataset path")
	parser.add_argument("--val_dataset_json", type=str, default="melon_detect/val.json", help="val dataset json file, coco format")
	parser.add_argument("--num_classes", type=int, default=2, help="number of categories")
	parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
	parser.add_argument("--batch_size", type=int, default=4, help="batch size")
	parser.add_argument("--epochs", type=int, default=15, help="epochs")
	parser.add_argument("--output_dir", type=str, default="result", help="output dir")

	args = parser.parse_args()
	return args

class CocoDataset(Dataset):
	def __init__(self, json_file, image_dir, processor):
		self.image_dir = image_dir
		self.processor = processor
		
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
		
		image_path = Path(self.image_dir) / Path(img_info["file_name"]).name
		if not image_path.exists():
			raise FileNotFoundError(colorama.Fore.RED + f"Image not found: {image_path}")
		
		image = Image.open(image_path).convert("RGB")
		image_width, image_height = image.size
		
		anns = self.annotations.get(img_id, [])
		
		coco_annotations = []
		for ann in anns:
			x, y, w, h = ann["bbox"]
			
			if w <= 0 or h <= 0:
				continue

			x = max(0, min(x, image_width))
			y = max(0, min(y, image_height))
			w = min(w, image_width - x)
			h = min(h, image_height - y)
			
			if w <= 0 or h <= 0:
				continue
			
			coco_annotations.append({
				"bbox": [x, y, w, h],
				"category_id": ann["category_id"],
				"area": ann.get("area", w * h),
				"iscrowd": ann.get("iscrowd", 0)
			})
		
		target = {
			"image_id": img_id,
			"annotations": coco_annotations
		}
		
		encoding = self.processor(images=image, annotations=target, return_tensors="pt")
		
		pixel_values = encoding["pixel_values"].squeeze(0)
		labels = encoding["labels"][0]
		return pixel_values, labels

def collate_fn(batch):
	pixel_values = [item[0] for item in batch]
	labels = [item[1] for item in batch]

	max_height = max([pv.shape[1] for pv in pixel_values])
	max_width = max([pv.shape[2] for pv in pixel_values])

	padded_pixel_values = []
	for pv in pixel_values:
		c, h, w = pv.shape
		padded = torch.zeros((c, max_height, max_width), dtype=pv.dtype, device=pv.device)
		padded[:, :h, :w] = pv
		padded_pixel_values.append(padded)

	pixel_values_batch = torch.stack(padded_pixel_values)
	return {"pixel_values": pixel_values_batch, "labels": labels}

def train(pretrained_model, train_dataset_path, train_dataset_json, val_dataset_path, val_dataset_json, num_classes, epochs, batch_size, lr, output_dir):
	class_names = ["N/A", "watermelon", "wintermelon"]
	id2label = {i: name for i, name in enumerate(class_names)}
	label2id = {name: i for i, name in enumerate(class_names)}	

	model = DetrForObjectDetection.from_pretrained(
		pretrained_model_name_or_path=pretrained_model,
		num_labels=len(id2label),
		ignore_mismatched_sizes=True,
		id2label=id2label,
		label2id=label2id  
	)

	processor = DetrImageProcessor.from_pretrained(pretrained_model_name_or_path=pretrained_model)

	train_dataset = CocoDataset(train_dataset_json, train_dataset_path, processor=processor)
	val_dataset = CocoDataset(val_dataset_json, val_dataset_path, processor=processor)

	train_args = TrainingArguments(
		output_dir=output_dir,
		num_train_epochs=epochs,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		learning_rate=lr,
		eval_strategy="epoch",
		save_strategy="epoch",
		remove_unused_columns=False,
		save_total_limit=3,
		load_best_model_at_end=True,
		report_to="none"
	)

	trainer = Trainer(
		model=model,
		args=train_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		data_collator=collate_fn,
		processing_class=processor
	)

	trainer.train()

	final_path = f"{output_dir}/final"
	trainer.save_model(final_path)
	processor.save_pretrained(final_path)

def predict(pretrained_model, image_name, threshold):
	model = AutoModelForObjectDetection.from_pretrained(pretrained_model)
	processor = AutoImageProcessor.from_pretrained(pretrained_model)

	bgr = cv2.imread(image_name)
	if bgr is None:
		raise FileNotFoundError(colorama.Fore.RED + f"unable to load image: {image_name}")
	rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

	inputs = processor(images=rgb, return_tensors="pt")
	outputs = model(**inputs)

	# convert outputs (bounding boxes and class logits) to COCO API
	target_sizes = torch.tensor([rgb.shape[:2]])
	results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)
	result = results[0] # single image

	for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
		box = [int(i) for i in box.tolist()] # boxes are the original image size
		print(f"result: id: {label}; label: {model.config.id2label[label.item()]}; confidence: {round(score.item(), 3)}; box: {box}")

		cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
		cv2.putText(bgr, f"{model.config.id2label[label.item()]},{round(score.item(), 3)}", (box[0]+2, box[3]-2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

	cv2.imwrite("../../data/result.png", bgr)

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "train":
		train(args.pretrained_model, args.train_dataset_path, args.train_dataset_json, args.val_dataset_path, args.val_dataset_json, args.num_classes, args.epochs, args.batch_size, args.lr, args.output_dir)
	elif args.task == "predict":
		predict(args.pretrained_model, args.image_name, args.threshold)

	print(colorama.Fore.GREEN + "====== execution completed ======")
