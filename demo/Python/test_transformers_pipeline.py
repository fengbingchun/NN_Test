from transformers import pipeline
from transformers.utils import logging
import argparse
import colorama
import torch
from pathlib import Path

# Blog: https://blog.csdn.net/fengbingchun/article/details/154282204

def parse_args():
	parser = argparse.ArgumentParser(description="transformers test")
	parser.add_argument("--task", required=True, type=str, choices=["document-question-answering", "image-classification", "image-feature-extraction", "image-segmentation", "object-detection"], help="specify what kind of task")
	parser.add_argument("--model", type=str, help="model name, for example: naver-clova-ix/donut-base-finetuned-docvqa")
	parser.add_argument("--file_name", type=str, help="image or pdf file name")
	parser.add_argument("--text", type=str, help="text")

	args = parser.parse_args()
	return args

def document_question_answering(model, file_name, text):
	pipe = pipeline("document-question-answering", model=model)
	result = pipe(image=file_name, question=text)
	if result is not None and isinstance(result, list) and len(result) > 0:
		print(f'model name: {model}; file name {Path(file_name).name}; answer: {result[0]["answer"]}')

def image_classification(model, file_name):
	pipe = pipeline("image-classification", model=model)
	result = pipe(file_name)
	if result is not None and isinstance(result, list) and len(result) > 0:
		print(f'model name: {model}; file name: {Path(file_name).name}; label: {result[0]["label"]}; score: {result[0]["score"]:.4f}')

def image_feature_extraction(model, file_name):
	pipe = pipeline("image-feature-extraction", model=model)
	result = pipe(file_name)
	if result is not None and isinstance(result, list) and len(result) > 0:
		features = torch.tensor(result[0]).mean(dim=0)
		print(f"features length: {len(features)}; features[0:5]: {features[0:5]}")

def image_segmentation(model, file_name):
	pipe = pipeline("image-segmentation", model=model, trust_remote_code=True)
	result = pipe(file_name)
	if result is not None:
		result.save("result_image_segmentation.png")

def object_detection(model, file_name):
	pipe = pipeline("object-detection", model=model)
	result = pipe(file_name)
	if result is not None and isinstance(result, list) and len(result) > 0:
		print(f'label: {result[0]["label"]}, score: {result[0]["score"]:.4f}, box: {result[0]["box"]}')

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	logging.set_verbosity_error()

	if args.task == "document-question-answering":
		document_question_answering(args.model, args.file_name, args.text)
	elif args.task == "image-classification":
		image_classification(args.model, args.file_name)
	elif args.task == "image-feature-extraction":
		image_feature_extraction(args.model, args.file_name)
	elif args.task == "image-segmentation":
		image_segmentation(args.model, args.file_name)
	elif args.task == "object-detection":
		object_detection(args.model, args.file_name)

	print(colorama.Fore.GREEN + "====== execution completed ======")
