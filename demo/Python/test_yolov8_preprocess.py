import colorama
import argparse
import os
import cv2
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 detect/segment preprocess")
	parser.add_argument("--dir_images", required=True, type=str, help="directory of source images")
	parser.add_argument("--dir_result", required=True, type=str, help="directory where the preprocess image results are saved")
	parser.add_argument("--imgsz", type=int, default=640, help="image size that is input to the network, consistent with the trained model, image size must be a multiple of the given stride in each dimension")
	parser.add_argument("--stride", type=int, default=32, help="stride value, consistent with the trained model")

	args = parser.parse_args()
	return args

def check_imgsz(imgsz, stride):
	if imgsz % stride != 0:
		print(colorama.Fore.RED + "Error: image size must be a multiple of the given stride in each dimension:", imgsz, stride)
		raise

def get_image_names(dir):
	# supported image formats
	img_formats = (".bmp", ".jpeg", ".jpg", ".png", ".webp")
	names = []

	for file in os.listdir(dir):
		if os.path.isfile(os.path.join(dir, file)):
			# print(file)
			_, extension = os.path.splitext(file)
			for format in img_formats:
				if format == extension.lower():
					names.append(file)
					break

	return names

def letterbox(img, imgsz):
	# reference: ultralytics/data/augment.py
	shape = img.shape[:2] # current shape: [height, width, channel]
	new_shape = [imgsz, imgsz]

	# scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

	# compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1] # wh padding
	dw /= 2 # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad: # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)) # add border

	return img

def preprocess(dir_images, dir_result, imgsz):
	# reference: ultralytics/engine/predictor.py
	os.makedirs(dir_result, exist_ok=True)

	image_names = get_image_names(dir_images)
	# print("image name:", image_names)

	for name in image_names:
		im0 = cv2.imread(dir_images+"/"+name) # BGR
		if im0 is None:
			raise FileNotFoundError(colorama.Fore.RED + f"Error: Image Not Found: {dir_images}/{name}")

		img = letterbox(im0, imgsz) # [640, 640, 3]
		img = np.stack([img]) # [1, 640, 640, 3]
		img = img[..., ::-1].transpose((0, 3, 1, 2)) # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
		img = np.ascontiguousarray(img) # contiguous
		img = img.astype(np.float32) / 255. # 0 - 255 to 0.0 - 1.0

		with open(dir_result+"/"+name+".data", "wb") as f:
			img.tofile(f)

if __name__ == "__main__":
	# python test_yolov8_preprocess.py --dir_images datasets/melon_new_segment/images/test --dir_result result_preprocess
	colorama.init(autoreset=True)
	args = parse_args()

	check_imgsz(args.imgsz, args.stride)

	preprocess(args.dir_images, args.dir_result, args.imgsz)

	print(colorama.Fore.GREEN + "====== execution completed ======")
