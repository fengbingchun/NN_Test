from ultralytics import YOLO
from ultralytics import settings
from ultralytics.utils.benchmarks import benchmark
import torch
import colorama
import ultralytics

# Blog: https://blog.csdn.net/fengbingchun/article/details/139031247

def checks():
	ultralytics.checks() # check software and hardware

def cuda_available():
	print("cuda is available:", torch.cuda.is_available())

def predict(model):
	model.info() # display model information
	results = model.predict("../../data/images/face")
	for result in results:
		result.show()

def yolov8_segment():
	model = YOLO("yolov8n-seg.pt") # load an official model
	predict(model)
	
def yolov8_classify():
	model = YOLO("yolov8n-cls.pt") # load an official model
	predict(model)

def yolov8_pose():
	model = YOLO("yolov8n-pose.pt") # load an official model
	predict(model)

def yolov8_obb():
	model = YOLO("yolov8n-obb.pt") # load an official model
	predict(model)

def yolov8_detect():
	# model = YOLO("yolov8n.pt") # load a pretrained YOLOv8n model
	model = YOLO("best.pt")  # best.pt comes from the training results in the yolov8_detect_train function

	# predict with the model
	# results = model("../../data/images/face/1.jpg")
	# all Ultralytics predict() calls will return a list of Results objects
	results = model.predict("../../data/images/predict")

	name = 1
	for result in results:
		result.show()
		print("orig shape:", result.orig_shape) # (height, width)
		print("boxes xyxy:", result.boxes.xyxy) # tensor:left-top, right-bottm
		print("boxes cls:", result.boxes.cls) # tensor:object categories
		print("boxes conf:", result.boxes.conf) # tensor:confidence
		# result.save(filename="../../data/result_3.png")
		result.save(filename="../../data/result_"+str(name)+".png")
		name = name + 1

def yolov8_settings():
	# view all settings
	print(settings)

def yolov8_tune():
	model = YOLO("yolov8n.pt") # load a pretrained model (recommended for training)

	# tune hyperparameters on COCO8 for 10 epochs
	model.tune(data="coco8.yaml", epochs=100, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)

def yolov8_detect_train():
	# load a model
	# model = YOLO("yolov8n.yaml") # build a new model form YAML
	model = YOLO("yolov8n.pt") # load a pretrained model (recommended for training)
	# model = YOLO("yolov8n.yaml").load("yolov8n.pt") # build from YAML and transfer weights

	# train the model
	results = model.train(data="coco8.yaml", epochs=5, imgsz=640)

	# validate the model
	metrics = model.val() # It'll automatically evaluate the data you trained, no arguments needed, dataset and settings remembered
	# print("metrics.box.map: ", metrics.box.map)      # map50-95
	# print("metrics.box.map50: ", metrics.box.map50)  # map50
	# print("metrics.box.map75: ", metrics.box.map75)  # map75
	# print("metrics.box.maps: ", metrics.box.maps)    # a list contains map50-95 of each category

	# export the model
	model.export(format="onnx")#, dynamic=True) # cannot specify dynamic=True, opencv does not support

def yolov8_benchmark():
	if torch.cuda.is_available():
		benchmark(model="best.onnx", data="coco8.yaml", imgsz=640, half=False, device=0)
	else: # benchmark on CPU
		benchmark(model="best.onnx", data="coco8.yaml", imgsz=640, half=False, device="cpu")

if __name__ == "__main__":
	# cuda_available()
	# yolov8_detect()
	# yolov8_settings()
	yolov8_detect_train()
	# yolov8_benchmark()
	# yolov8_segment()
	# yolov8_classify()
	# yolov8_pose()
	# yolov8_obb()
	# yolov8_tune()
	# checks()

	print(colorama.Fore.GREEN + "====== execution completed ======")
