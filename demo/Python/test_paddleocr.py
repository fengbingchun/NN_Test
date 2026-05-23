import colorama
import argparse
from pathlib import Path

import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
import warnings
warnings.filterwarnings("ignore") # UserWarning: No ccache found

import paddle
import paddleocr
from paddleocr import PaddleOCR

# Blog: https://blog.csdn.net/fengbingchun/article/details/161339162

def parse_args():
	parser = argparse.ArgumentParser(description="test PaddleOCR")
	parser.add_argument("--task", required=True, type=str, choices=["version", "ocr_predict"], help="specify what kind of task")
	parser.add_argument("--detection_model_name", type=str, default="PP-OCRv5_mobile_det", help="detection model name")
	parser.add_argument("--detection_model_dir", type=str, default="PP-OCRv5_mobile_det", help="detection model directory")
	parser.add_argument("--recognition_model_name", type=str, default="PP-OCRv5_mobile_rec", help="recognition model name")
	parser.add_argument("--recognition_model_dir", type=str, default="PP-OCRv5_mobile_rec", help="recognition model directory")
	parser.add_argument("--image_name", type=str, help="image name")

	args = parser.parse_args()
	return args

def version():
	print(f"Paddle Version: {paddle.__version__}")
	print(f"GPU available: {paddle.is_compiled_with_cuda()}")
	print(f"GPU count: {paddle.device.cuda.device_count()}")

	print(f"PaddleOCR Version: {paddleocr.__version__}")

def ocr_predict(detection_model_name, detection_model_dir, recognition_model_name, recognition_model_dir, image_name):
	if detection_model_dir is None or not detection_model_dir or not Path(detection_model_dir).is_dir():
		raise FileNotFoundError(colorama.Fore.RED + f"detection model dir must be specified: {detection_model_dir}")
	if recognition_model_dir is None or not recognition_model_dir or not Path(recognition_model_dir).is_dir():
		raise FileNotFoundError(colorama.Fore.RED + f"recognition model dir must be specified: {recognition_model_dir}")

	if image_name is None or not image_name or not Path(image_name).is_file():
		raise FileNotFoundError(colorama.Fore.RED + f"image file must be specified: {image_name}")

	ocr = PaddleOCR(
		text_detection_model_name=detection_model_name,
		text_detection_model_dir=detection_model_dir,
		text_recognition_model_name=recognition_model_name,
		text_recognition_model_dir=recognition_model_dir,
		use_doc_orientation_classify=False,
		use_doc_unwarping=False,
		use_textline_orientation=False,
		enable_mkldnn=False
	)

	result = ocr.predict(input=image_name)
	for res in result:
		res.print()
		res.save_to_img(save_path="output")
		res.save_to_json(save_path="output")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "version":
		version()
	elif args.task == "ocr_predict":
		ocr_predict(args.detection_model_name, args.detection_model_dir, args.recognition_model_name, args.recognition_model_dir, args.image_name)

	print(colorama.Fore.GREEN + "====== execution completed ======")
