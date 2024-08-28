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
	parser.add_argument("--max_det", type=int, default=300, help="the maximum number of boxes to keep after NMS")
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

def xywh2xyxy(x):
	'''Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner'''
	assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"

	y = np.empty_like(x) # faster than clone/copy
	dw = x[..., 2] / 2 # half-width
	dh = x[..., 3] / 2 # half-height
	y[..., 0] = x[..., 0] - dw # top left x
	y[..., 1] = x[..., 1] - dh # top left y
	y[..., 2] = x[..., 0] + dw # bottom right x
	y[..., 3] = x[..., 1] + dh # bottom right y

	return y

def non_max_suppression(prediction, conf_thres, iou_thres, task):
	# reference: ultralytics/utils/ops.py
	assert 0 <= conf_thres <= 1, f"invalid confidence threshold: {conf_thres}, valid values are between 0.0 and 1.0"
	assert 0 <= iou_thres <= 1, f"invalid iou threshold: {iou_thres}, valid values are between 0.0 and 1.0"

	bs = prediction.shape[0] # batch size
	nc = prediction.shape[1] - 4 # number of classes
	nm = prediction.shape[1] - nc - 4
	mi = 4 + nc # mask start index
	xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres # candidates
	print(f"bs: {bs}; nc:{nc}; nm:{nm}; mi:{mi}; prediction shape: {prediction.shape}; xc shape: {xc.shape}; xc: {xc}")

	prediction = prediction.transpose(0,2,1)  # shape(1,6,8400) to shape(1,8400,6)
	print("prediction shape:", prediction.shape)

	prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

	for xi, x in enumerate(prediction): # image index, image inference
		x = x[xc[xi]] # confidence
		print(f"x shape: {x.shape}; x: {x}")

		# if none remain process next image
		if not x.shape[0]:
			continue

		# detections matrix nx6 (xyxy, conf, cls)
		if task == "detect":
			box, cls = np.hsplit(x, [4])
		print(f"box:{box}; cls:{cls}")

		conf = np.amax(cls, axis=1)
		conf = conf[:, np.newaxis]
		j = np.argmax(cls, axis=1)
		j = j[:, np.newaxis]
		j = j.astype(float)
		print(f"conf:{conf}; conf shape:{conf.shape} j:{j}; j shape:{j.shape}")

		x = np.concatenate((box, conf, j), axis=1)
		print(f"x:{x}; x.shape:{x.shape}")

		n = x.shape[0] # number of boxes
		if not n: # no boxes
			continue

		# if n > 30000: # excess boxes

		max_wh=7680
		c = x[:, 5:6] * max_wh # classes
		scores = x[:, 4] # scores
		print(f"c:{c}; c.shape:{c.shape}; scores:{scores}; scores.shape:{scores.shape}")

		boxes = x[:, :4] + c # boxes(offset by class)
		print(f"boxes:{boxes}; boxes.shape:{boxes.shape}")



if __name__ == "__main__":
	# python test_yolov8_postprocess.py --src_image datasets/melon_new_detect/images/test/27702418.webp --dst_image=result_postprocess.png --net_output_file 27702418.webp.output.bin --task detect
	colorama.init(autoreset=True)
	args = parse_args()

	preds = load_net_output_data(args.net_output_file, args.task)

	non_max_suppression(preds, args.conf, args.iou, args.task)

	print(colorama.Fore.GREEN + "====== execution completed ======")
