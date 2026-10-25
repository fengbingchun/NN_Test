import colorama
import argparse
from insightface.app import FaceAnalysis
import insightface.utils
import onnxruntime as ort
import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Blog: https://blog.csdn.net/fengbingchun/article/details/169047552

def parse_args():
	parser = argparse.ArgumentParser(description="test InsightFace")
	parser.add_argument("--task", required=True, type=str, choices=["inspect_model", "face_detect", "face_align", "face_landmarks", "extract_features", "face_recognize", "gender_age"], help="specify what kind of task")
	parser.add_argument("--model_name", type=str, help="model name")
	parser.add_argument("--model_dir_name", type=str, help="model directory name")
	parser.add_argument("--image_name", type=str, help="image name")
	parser.add_argument("--json_dir_name", type=str, help="json dir name")
	parser.add_argument("--index", type=int, default=1, help="start index")

	args = parser.parse_args()
	return args

def inspect_model(model_name):
	session = ort.InferenceSession(model_name, providers=["CPUExecutionProvider"])

	print("inputs:")
	for item in session.get_inputs():
		print(f"  name={item.name}, shape={item.shape}, type={item.type}")

	print("outputs:")
	for item in session.get_outputs():
		print(f"  name={item.name}, shape={item.shape}, type={item.type}")

def face_detect(model_dir_name, image_name):
	app = FaceAnalysis(name=model_dir_name, allowed_modules=["detection"], providers=["CPUExecutionProvider"]) # det_10g.onnx
	app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)

	img = cv2.imread(image_name)
	if img is None:
		raise FileNotFoundError(colorama.Fore.RED + f"failed to read image: {image_name}")

	faces = app.get(img)

	for face in faces:
		x1, y1, x2, y2 = face.bbox.astype(int)
		score = float(face.det_score)

		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(img, f"{score:.3f}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

	cv2.imshow("face detect", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("result_face_detect.png", img)

def face_align(model_dir_name, image_name):
	app = FaceAnalysis(name=model_dir_name, allowed_modules=["detection"], providers=["CPUExecutionProvider"]) # det_10g.onnx
	app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)

	img = cv2.imread(image_name)
	if img is None:
		raise FileNotFoundError(colorama.Fore.RED + f"failed to read image: {image_name}")

	faces = app.get(img)

	for i, face in enumerate(faces):
		x1, y1, x2, y2 = face.bbox.astype(int)

		kps = face.kps # det_10g.onnx: 5-point landmarks
		if kps is None or kps.shape != (5, 2):
			print(colorama.Fore.YELLOW + f"face {i}: missing 5-point landmarks")
			continue

		aligned = insightface.utils.face_align.norm_crop(img, landmark=kps, image_size=112)
		name = "result_face_aligned_" + f"{i}" + ".png"
		if not cv2.imwrite(name, aligned):
			raise RuntimeError(colorama.Fore.RED + f"failed to save: {name}")

		print(f"face {i}: bbox: {[x1, y1, x2, y2]}; landmarks shape: {kps.shape}; aligned image shape: {aligned.shape}");

		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
		for x, y in kps:
			cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

	cv2.imshow("face detect", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("result_face_detect.png", img)

def face_landmarks(model_dir_name, image_name):
	app = FaceAnalysis(name=model_dir_name, allowed_modules=["detection", "landmark_2d_106"], providers=["CPUExecutionProvider"]) # det_10g.onnx, 2d106det.onnx
	app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)

	img = cv2.imread(image_name)
	if img is None:
		raise FileNotFoundError(colorama.Fore.RED + f"failed to read image: {image_name}")

	faces = app.get(img)

	for i, face in enumerate(faces):
		x1, y1, x2, y2 = face.bbox.astype(int)
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

		landmarks = face.landmark_2d_106 # 2d106det.onnx: 106-point landmarks
		if landmarks is None or landmarks.shape != (106, 2):
			print(colorama.Fore.YELLOW + f"face {i}: missing 106-point landmarks")
			continue

		for x, y in landmarks:
			cv2.circle(img, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)

	cv2.imshow("face detect", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("result_face_detect.png", img)

def extract_features(model_dir_name, image_name, index, json_dir_name):
	app = FaceAnalysis(name=model_dir_name, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"]) # det_10g.onnx, w600k_r50.onnx
	app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)

	img = cv2.imread(image_name)
	if img is None:
		raise FileNotFoundError(colorama.Fore.RED + f"failed to read image: {image_name}")

	faces = app.get(img)
	if not faces:
		raise RuntimeError(colorama.Fore.RED + f"no face detected: {image_name}")

	if Path(json_dir_name).exists():
		print(colorama.Fore.YELLOW + f"specified directory already exists: {json_dir_name}")
	else:
		Path(json_dir_name).mkdir(parents=True)

	for i, face in enumerate(faces):
		embedding = np.asarray(face.normed_embedding, dtype=np.float32) # w600k_r50.onnx: 512-dimensional feature vector
		if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
			raise RuntimeError("invalid face embedding")

		database = {
			"embedding":embedding.tolist(),
			"dimension": int(embedding.size),
			"det_score": float(f"{face.det_score:.4f}"),
			"created_at": datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
		}

		name = json_dir_name + "/" + str(i + index) + ".json"
		with open(name, "w", encoding="utf-8") as f:
			json.dump(database, f, ensure_ascii=False, indent=4)

def face_recognize(model_dir_name, image_name, json_dir_name):
	app = FaceAnalysis(name=model_dir_name, allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"]) # det_10g.onnx, w600k_r50.onnx
	app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)

	img = cv2.imread(image_name)
	if img is None:
		raise FileNotFoundError(colorama.Fore.RED + f"failed to read image: {image_name}")

	faces = app.get(img)
	if not faces:
		raise RuntimeError(colorama.Fore.RED + f"no face detected: {image_name}")

	json_files = [f for f in Path(json_dir_name).glob("*.json") if f.is_file()]
	if len(json_files) == 0:
		raise FileNotFoundError(colorama.Fore.RED + f"there are no matching json file in this directory: {json_dir_name}")

	for i, face in enumerate(faces):
		query = np.asarray(face.normed_embedding, dtype=np.float32) # w600k_r50.onnx: 512-dimensional feature vector
		if query.ndim != 1 or not np.all(np.isfinite(query)):
			raise RuntimeError("invalid face embedding")

		for name in json_files:
			with open(name, "r", encoding="utf-8") as f:
				database = json.load(f)

			known = np.asarray(database["embedding"], dtype=np.float32)
			score = float(np.dot(query, known)) # both vectors have been normalized; the dot product is equal to the cosine similarity
			print(f"index {i+1} facial similarity score with {name.name} is: {score:.4f}")

def gender_age(model_dir_name, image_name):
	app = FaceAnalysis(name=model_dir_name, allowed_modules=["detection", "genderage"], providers=["CPUExecutionProvider"]) # det_10g.onnx, genderage.onnx
	app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.5)

	img = cv2.imread(image_name)
	if img is None:
		raise FileNotFoundError(colorama.Fore.RED + f"failed to read image: {image_name}")

	faces = app.get(img)
	if not faces:
		raise RuntimeError(colorama.Fore.RED + f"no face detected: {image_name}")

	for i, face in enumerate(faces):
		x1, y1, x2, y2 = face.bbox.astype(int)
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

		gender = "Male" if int(face.gender) == 1 else "Female" # genderage.onnx
		age = int(face.age)

		print(f"index {i+1}: gender: {gender}; age: {age}")

		label = f"{gender}, {age}"
		cv2.putText(img, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

	cv2.imshow("gender age", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.imwrite("result_gender_age.png", img)

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "inspect_model":
		inspect_model(args.model_name)
	elif args.task == "face_detect":
		face_detect(args.model_dir_name, args.image_name)
	elif args.task == "face_align":
		face_align(args.model_dir_name, args.image_name)
	elif args.task == "face_landmarks":
		face_landmarks(args.model_dir_name, args.image_name)
	elif args.task == "extract_features":
		extract_features(args.model_dir_name, args.image_name, args.index, args.json_dir_name)
	elif args.task == "face_recognize":
		face_recognize(args.model_dir_name, args.image_name, args.json_dir_name)
	elif args.task == "gender_age":
		gender_age(args.model_dir_name, args.image_name)

	print(colorama.Fore.GREEN + "====== execution completed ======")
