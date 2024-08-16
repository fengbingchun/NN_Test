import colorama
import argparse
import os
import cv2
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 detect/segment preprocess")
	parser.add_argument("--src_image", required=True, type=str, help="source image")
	parser.add_argument("--dst_image", required=True, type=str, help="result image after drawing the box")
	parser.add_argument("--net_output_file", required=True, type=str, help="output result after inference")
	parser.add_argument("--conf", type=float, default=0.25, help="the confidence threshold below which boxes will be filtered out, valid values are between 0.0 and 1.0")
	parser.add_argument("--iou", type=float, default=0.7, help="the IoU threshold below which boxes will be filtered out during NMS, valid values are between 0.0 and 1.0")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment"], help="specify what kind of task")

	args = parser.parse_args()
	return args

def load_net_output_data(data, task):
	if task == "detect":
		shape = (1, 6, 8400)
		dtype = np.float32
		preds = np.fromfile(data, dtype=dtype)
		preds = preds.reshape(shape)
		# print("preds:", preds)
		return preds

def non_max_suppression(preds, conf, iou):
	# reference: ultralytics/utils/ops.py
	...

if __name__ == "__main__":
	# python test_yolov8_postprocess.py --src_image datasets/melon_new_detect/images/test/27702418.webp --dst_image=result_postprocess.png --net_output_file 27702418.webp.output.bin --task detect
	colorama.init()
	args = parse_args()

	preds = load_net_output_data(args.net_output_file, args.task)

	non_max_suppression(preds, args.conf, args.iou)

	print(colorama.Fore.GREEN + "====== execution completed ======")
