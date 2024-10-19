import colorama
import argparse
import os
import cv2
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 detect/segment postprocess")
	parser.add_argument("--src_image", required=True, type=str, help="source image")
	parser.add_argument("--dst_image", required=True, type=str, help="result image after drawing the box")
	parser.add_argument("--net_output_file", required=True, type=str, help="output result after inference")
	parser.add_argument("--conf", type=float, default=0.25, help="the confidence threshold below which boxes will be filtered out, valid values are between 0.0 and 1.0")
	parser.add_argument("--iou", type=float, default=0.7, help="the IoU threshold below which boxes will be filtered out during NMS, valid values are between 0.0 and 1.0")
	parser.add_argument("--max_det", type=int, default=300, help="the maximum number of boxes to keep after NMS")
	parser.add_argument("--task", required=True, type=str, choices=["detect", "segment"], help="specify what kind of task")
	parser.add_argument("--imgsz", type=int, default=640, help="input net image size")

	args = parser.parse_args()
	return args

def load_net_output_data_detect(data):
	shape = (1, 6, 8400)
	dtype = np.float32
	preds = np.fromfile(data, dtype=dtype)
	preds = preds.reshape(shape)

	# print("preds:", preds)
	return preds

def load_net_output_data_segment(data):
	output0_shape = (1,38,8400)
	output1_shape = (1,32,160,160)
	dtype = np.float32

	preds = np.fromfile(data+".0", dtype=dtype).reshape(output0_shape)
	preds2 = np.fromfile(data+".1", dtype=dtype).reshape(output1_shape)

	# print(f"preds.shape:{preds.shape}; preds:{preds}; preds2.shape:{preds2.shape}; preds2:{preds2}")
	return preds, preds2

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
	nc = 2 #prediction.shape[1] - 4 # number of classes
	nm = prediction.shape[1] - nc - 4
	mi = 4 + nc # mask start index
	xc = np.max(prediction[:, 4:mi], axis=1) > conf_thres # candidates

	prediction = prediction.transpose(0,2,1)  # detect: shape(1,6,8400) to shape(1,8400,6)

	prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

	output = []

	for xi, x in enumerate(prediction): # image index, image inference
		x = x[xc[xi]] # confidence

		# if none remain process next image
		if not x.shape[0]:
			continue

		# detections matrix nx6 (xyxy, conf, cls)
		if task == "detect":
			box, cls = np.hsplit(x, [4])
		else:
			box, cls, mask = np.hsplit(x, [4,6])
			# print(f"box:{box}; cls:{cls}; mask:{mask}")

		conf = np.amax(cls, axis=1)
		conf = conf[:, np.newaxis]
		j = np.argmax(cls, axis=1)
		j = j[:, np.newaxis]
		j = j.astype(float)

		if task == "detect":
			x = np.concatenate((box, conf, j), axis=1)
		else:
			x = np.concatenate((box, conf, j, mask), axis=1)

		n = x.shape[0] # number of boxes
		if not n: # no boxes
			continue

		# if n > 30000: # excess boxes

		max_wh = 7680
		c = x[:, 5:6] * max_wh # classes
		scores = x[:, 4] # scores
		# print(f"c:{c}; c.shape:{c.shape}; scores:{scores}; scores.shape:{scores.shape}; conf_thres:{conf_thres}; iou_thres:{iou_thres}")

		boxes = x[:, :4] #+ c # boxes(offset by class)

		indexes = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)

		output = [x[i] for i in indexes.flatten()]
		output = np.array(output)

		return output

def clip_boxes(boxes, shape):
	boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
	boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

	return boxes

def scale_boxes(img1_shape, boxes, img0_shape):
	# calculate from img0_shape
	gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
	pad = (
		round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
		round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
	)  # wh padding

	boxes[..., 0] -= pad[0]  # x padding
	boxes[..., 1] -= pad[1]  # y padding
	boxes[..., 2] -= pad[0]  # x padding
	boxes[..., 3] -= pad[1]  # y padding
	boxes[..., :4] /= gain
	# print(f"boxes: {boxes}")

	return clip_boxes(boxes, img0_shape)

def draw_rect(boxes, img0, masks=None):
	classes = ["watermelon", "wintermelon"]
	for box in boxes:
		cv2.rectangle(img0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
		text = classes[int(box[5])] + f" {box[4]:.2f} "
		# print(f"text: {text}")
		org = (int(box[0]), int(box[1]-10))
		cv2.putText(img0, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

	frame = np.copy(img0)
	if masks is not None:
		assert img0.shape[0] == masks.shape[0] and img0.shape[1] == masks.shape[1], f"the size of the masks must be consistent with the original image size"
		for i in range(masks.shape[2]):
			frame[masks[:,:,i] > 0] = (33,145,237)

	img0 = cv2.addWeighted(img0, 0.6, frame, 0.4, 0)
	cv2.imwrite("result_postprocess.png", img0)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def crop_mask(masks, boxes):
	_, h, w = masks.shape
	x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
	x1 = np.expand_dims(x1, axis=2) # x1 shape(n,1,1)
	y1 = np.expand_dims(y1, axis=2)
	x2 = np.expand_dims(x2, axis=2)
	y2 = np.expand_dims(y2, axis=2)
	# print(f"x1:{x1.shape}; y1{y1}; x2:{x2}; y2:{y2}")

	r = np.arange(w) # rows shape(1,1,w)
	r = r.reshape(1, 1, w)
	c = np.arange(h) # cols shape(1,h,1)
	c = c.reshape(1, h, 1)

	return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, preds, shape):
	masks_in = preds[:,6:]
	bboxes = preds[:,:4]

	c, mh, mw = protos.shape  # CHW
	ih, iw = shape
	# masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
	masks = masks_in @ protos.reshape(32,160*160)
	masks = sigmoid(masks)
	masks = masks.reshape(masks.size//mh//mw,mh,mw)
	# print(f"masks:{masks}; masks.shape:{masks.shape}")

	width_ratio = mw / iw
	height_ratio = mh / ih

	downsampled_bboxes = bboxes.copy()
	downsampled_bboxes[:, 0] *= width_ratio
	downsampled_bboxes[:, 2] *= width_ratio
	downsampled_bboxes[:, 3] *= height_ratio
	downsampled_bboxes[:, 1] *= height_ratio
	# print("downsampled_bboxes:", downsampled_bboxes)

	masks = crop_mask(masks, downsampled_bboxes)  # CHW

	arrays = np.zeros((masks.shape[0], ih, iw), dtype=np.float32)
	for i in range(masks.shape[0]):
		resized = cv2.resize(masks[i,:,:], (iw,ih), interpolation=cv2.INTER_LINEAR)
		arrays[i,:,:] = resized
	masks = arrays > 0.5
	masks = masks.astype(np.uint8)

	return masks

def resize_mask(masks, shape):
	img1_shape = masks.shape[1:]
	img0_shape = shape
	gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
	pad = (
		round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), # width
		round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1), # height
	)  # wh padding
	# print(f"gain:{gain}; pad:{pad}")

	crop_x = round(max(pad[0], 0))
	crop_y = round(max(pad[1], 0))
	crop_w = round(min(img1_shape[1], img0_shape[1]*gain))
	crop_h = round(min(img1_shape[0], img0_shape[0]*gain))
	masks = masks.transpose((1,2,0))
	crop = masks[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]
	# print(f"crop.shape: {crop.shape}, crop.dtype:{crop.dtype}")

	resize = np.zeros((img0_shape[0], img0_shape[1], masks.shape[2]), dtype=crop.dtype)
	for i in range(masks.shape[2]):
		resize[:,:,i] = cv2.resize(crop[:,:,i], (img0_shape[1], img0_shape[0]))
	# print("resize.shape:", resize.shape)

	return resize


if __name__ == "__main__":
	# python test_yolov8_postprocess.py --src_image datasets/melon_new_detect/images/test/27702418.webp --dst_image=result_postprocess.png --net_output_file 27702418.webp.output.bin --task detect
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "detect":
		preds = load_net_output_data_detect(args.net_output_file)
	else:
		preds, preds2 = load_net_output_data_segment(args.net_output_file)

	preds = non_max_suppression(preds, args.conf, args.iou, args.task)

	img0 = cv2.imread(args.src_image)
	if img0 is None:
		raise FileNotFoundError(f"image not found: {args.src_image}")

	img1_shape = [args.imgsz, args.imgsz] # [height, width]
	img0_shape = [img0.shape[0], img0.shape[1]] # [height, width]

	if args.task == "segment":
		masks = process_mask(preds2[0,:,:,:], preds, img1_shape)
		masks = resize_mask(masks, img0_shape)
	else:
		masks = None

	preds = scale_boxes(img1_shape, preds, img0_shape)
	# print("preds:", preds)

	draw_rect(preds, img0, masks)

	print(colorama.Fore.GREEN + "====== execution completed ======")
