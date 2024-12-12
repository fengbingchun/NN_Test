import colorama
import argparse
import os
import cv2
from pathlib import Path
from moviepy.editor import VideoFileClip

def parse_args():
	parser = argparse.ArgumentParser(description="video related operations")
	parser.add_argument("--video_file", type=str, help="video file")
	parser.add_argument("--images_path", type=str, help="images path")
	parser.add_argument("--frame_interval", type=int, default=25, help="specifies how many frames to extract a frame")
	parser.add_argument("--frame_rate", type=int, default=25, help="frame rate")
	parser.add_argument("--first_image_name", type=int, default=0, help="the name of the first image")
	parser.add_argument("--src_path", type=str, help="source video file path")
	parser.add_argument("--dst_path", type=str, help="destination path")
	parser.add_argument("--suffix", type=str, help="file name suffix")

	args = parser.parse_args()
	return args


def extract_frame(video_file, images_path, frame_interval, first_image_name):
	os.makedirs(images_path, exist_ok=True)

	cap = cv2.VideoCapture(video_file)
	if not cap.isOpened():
		raise FileNotFoundError(colorama.Fore.RED + f"file does not exist: {video_file}")

	frame_count = 0
	frame_id = first_image_name

	while True:
		ret, frame = cap.read()
		if not ret:
			break

		if frame_count % frame_interval == 0:
			image_name = os.path.join(images_path, f"{frame_id:08d}.png")
			cv2.imwrite(image_name, frame)
			frame_id += 1

		frame_count += 1


def generate_video(images_path, frame_rate, video_file):
	images_name = sorted([f for f in os.listdir(images_path) if f.endswith("png")])
	if len(images_name) == 0:
		raise FileNotFoundError(colorama.Fore.RED + f"there are no images in this directory that match the criteria: {images_path}")
	# print(f"images name: {images_name}")

	frame = cv2.imread(os.path.join(images_path, images_name[0]))
	height, width, channel = frame.shape

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(video_file, fourcc, frame_rate, (width, height))

	for image_name in images_name:
		frame = cv2.imread(os.path.join(images_path, image_name))
		if frame is not None:
			out.write(frame)
		else:
			print(f"Error: unable to read image: {os.path.join(images_path, image_name)}")

	out.release()

def video_info(src_path, suffix):
	names = []
	for name in Path(src_path).rglob("*."+suffix):
		names.append(name)
	print(f"video files count: {len(names)}, name: {names[0]}")

	del_names = []
	for name in names:
		try:
			video = VideoFileClip(str(name))
		except Exception as e:
			# raise IOError(colorama.Fore.RED + f"cann't read file: {name}: {e}")
			print(colorama.Fore.RED + f"cann't read file: {name}: {e}")
			del_names.append(name)
			continue

		duration = video.duration # second
		minutes = int(duration // 60)
		seconds = int(duration % 60)
		fps = video.fps
		size = video.size
		width, height = size[0], size[1]
		print(f"name: {name.name}: duration:{minutes}:{seconds}; fps:{fps}; width:{width}; height:{height}")
		if minutes < 47:
			del_names.append(name)

	for name in del_names:
		print(f"delete file: {str(name)}")

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	video_info(args.src_path, args.suffix)

	print(colorama.Fore.GREEN + "====== execution completed ======")
