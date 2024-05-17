import argparse
import colorama
from ultralytics import YOLO

def parse_args():
	parser = argparse.ArgumentParser(description="YOLOv8 object detect")
	parser.add_argument("--yaml", required=True, type=str, help="yaml file")
	parser.add_argument("--epochs", required=True, type=int, help="number of training")

	args = parser.parse_args()
	return args

def train(yaml, epochs):
	model = YOLO("yolov8n.pt") # load a pretrained model
	results = model.train(data=yaml, epochs=epochs, imgsz=640) # train the model

	metrics = model.val() # It'll automatically evaluate the data you trained, no arguments needed, dataset and settings remembered

	model.export(format="onnx", dynamic=True) # export the model

if __name__ == "__main__":
	colorama.init()
	args = parse_args()

	train(args.yaml, args.epochs)

	print(colorama.Fore.GREEN + "====== execution completed ======")
