import colorama
import argparse
import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Colors

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 segment onnxruntime predict")
	parser.add_argument("--model", required=True, type=str, help="model file")
	parser.add_argument("--dir_images", required=True, type=str, help="directory of test images")
	parser.add_argument("--dir_result", required=True, type=str, help="directory where the image results are saved")
	parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
	parser.add_argument("--iou", type=float, default=0.45, help="NMS iou threshold")
	parser.add_argument("--yaml", required=True, type=str, help="classes yamal file")

	args = parser.parse_args()
	return args

def get_images_name(dir):
	# supported image formats
	img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
	images_name = []

	for name in os.listdir(dir):
		file = os.path.join(dir, name)
		# print("file:", file)
		if os.path.isfile(file):
			_, extension = os.path.splitext(file)
			# print(extension)
			for format in img_formats:
				if format == extension:
					images_name.append(name)
					break

	return images_name

def preprocess(img, model_height, model_width, ndtype):
	# resize and pad input image using letterbox() (borrowed from Ultralytics)
	shape = img.shape[:2] # original image shape
	new_shape = (model_height, model_width)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	ratio = r, r
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
	if shape[::-1] != new_unpad: # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
	left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

	# transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
	img = np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=ndtype) / 255.0
	img_process = img[None] if len(img.shape) == 3 else img

	return img_process, ratio, (pad_w, pad_h)

def scale_mask(masks, img_shape, ratio_pad=None):
	im1_shape = masks.shape[:2]
	if ratio_pad is None: # calculate from img_shape
		gain = min(im1_shape[0] / img_shape[0], im1_shape[1] / img_shape[1]) # gain  = old / new
		pad = (im1_shape[1] - img_shape[1] * gain) / 2, (im1_shape[0] - img_shape[0] * gain) / 2 # wh padding
	else:
		pad = ratio_pad[1]

	# calculate tlbr of mask
	top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1)) # y, x
	bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
	if len(masks.shape) < 2:
		raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
	masks = masks[top:bottom, left:right]
	masks = cv2.resize(
		masks, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
	)  # INTER_CUBIC would be better
	if len(masks.shape) == 2:
		masks = masks[:, :, None]

	return masks

def crop_mask(masks, boxes):
	n, h, w = masks.shape
	x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
	r = np.arange(w, dtype=x1.dtype)[None, None, :]
	c = np.arange(h, dtype=x1.dtype)[None, :, None]

	return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, img_shape):
	c, mh, mw = protos.shape
	masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0) # HWN
	masks = np.ascontiguousarray(masks)
	masks = scale_mask(masks, img_shape) # re-scale mask from P3 shape to original input image shape
	masks = np.einsum("HWN -> NHW", masks) # HWN -> NHW
	masks = crop_mask(masks, bboxes)

	return np.greater(masks, 0.5)

def masks2segments(masks):
	segments = []
	for x in masks.astype("uint8"):
		c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0] # CHAIN_APPROX_SIMPLE
		if c:
			c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
		else:
			c = np.zeros((0, 2)) # no segments found
		segments.append(c.astype("float32"))

	return segments

def postprocess(predictions, img, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
	x, protos = predictions[0], predictions[1] # two outputs: predictions and protos

	# transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
	x = np.einsum("bcn->bnc", x)

	# predictions filtering by conf-threshold
	x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]

	# create a new matrix which merge these(box, score, cls, nm) into one
	x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]

	# NMS filtering
	x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]

	# decode and return
	if len(x) > 0:
		# bounding boxes format change: cxcywh -> xyxy
		x[..., [0, 1]] -= x[..., [2, 3]] / 2
		x[..., [2, 3]] += x[..., [0, 1]]

		# rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
		x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
		x[..., :4] /= min(ratio)

		# bounding boxes boundary clamp
		x[..., [0, 2]] = x[:, [0, 2]].clip(0, img.shape[1])
		x[..., [1, 3]] = x[:, [1, 3]].clip(0, img.shape[0])

		# process masks
		masks = process_mask(protos[0], x[:, 6:], x[:, :4], img.shape)

		# Masks -> Segments(contours)
		segments = masks2segments(masks)

		return x[..., :6], segments, masks  # boxes, segments, masks
	else:
		return [], [], []
	
def draw(name, img, bboxes, segments, dir_result, classes, color_palette):
	img_canvas = img.copy()
	for (*box, conf, cls_), segment in zip(bboxes, segments):
		# draw contour and fill mask
		cv2.polylines(img, np.int32([segment]), True, (255, 255, 255), 2) # white borderline
		cv2.fillPoly(img_canvas, np.int32([segment]), color_palette(int(cls_), bgr=True))

		 # draw bbox rectangle
		cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_palette(int(cls_), bgr=True), 1, cv2.LINE_AA)
		cv2.putText(img, f"{classes[cls_]}: {conf:.3f}", (int(box[0]), int(box[1] - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_palette(int(cls_), bgr=True), 2, cv2.LINE_AA)

		# mix image
		img = cv2.addWeighted(img_canvas, 0.3, img, 0.7, 0)
		cv2.imwrite(dir_result+"/"+name, img)

def predict(model, dir_images, dir_result, conf_threshold, iou_threshold, classes_yaml):
	# ort session
	session = ort.InferenceSession(
		model,
		providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
	)

	# numpy dtype: support both FP32 and FP16 onnx model
	ndtype = np.half if session.get_inputs()[0].type == "tensor(float16)" else np.single
	# print(ndtype)

	# get model width and height(YOLOv8-seg only has one input)
	model_height, model_width = [x.shape for x in session.get_inputs()][0][-2:]
	# print(f"model height: {model_height}; model width: {model_width}")

	# load class names
	classes = yaml_load(check_yaml(classes_yaml))["names"]
	# print(classes)

	# create color palette
	color_palette = Colors()

	os.makedirs(dir_result) #, exist_ok=True

	images_name = get_images_name(dir_images)
	# print(images_name)

	for name in images_name:
		img = cv2.imread(dir_images+"/"+name)
		img2, ratio, (pad_w, pad_h) = preprocess(img, model_height, model_width, ndtype)
		# print(f"img shape:{img.shape}; img2 shape:{img2.shape}; ratio:{ratio}; pad_w:{pad_w}; pad_h:{pad_h}")

		# ort inference
		predictions = session.run(None, {session.get_inputs()[0].name: img2})

		bboxes, segments, _ = postprocess(predictions, img, ratio, pad_w, pad_h, conf_threshold, iou_threshold)

		if len(bboxes) > 0:
			draw(name, img, bboxes, segments, dir_result, classes, color_palette)

if __name__ == "__main__":
	# reference: ultralytics/examples/YOLOv8-Segmentation-ONNXRuntime-Python
	colorama.init()
	args = parse_args()

	predict(args.model, args.dir_images, args.dir_result, args.conf, args.iou, args.yaml)

	print(colorama.Fore.GREEN + "====== execution completed ======")
