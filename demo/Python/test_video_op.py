import colorama
import argparse
import os
import cv2
from pathlib import Path
from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta
import shutil

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
	parser.add_argument("--duration", type=int, default=3600, help="the duration(seconds) of the video file")

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

def video_interval(src_path, suffix):
	names = []
	for name in Path(src_path).rglob("*."+suffix):
		names.append(name)
	print(f"video files count: {len(names)}, name: {names[0]}")

	times = []
	for name in names:
		name = name.name
		name = name[2:-4]

		time = datetime.strptime(name, "%Y%m%d%H%M%S")
		# print(f"name: {name}; time: {time}"); raise
		times.append(time)

	for i in range(len(times) - 1):
		diff = times[i+1] - times[i]
		hours, remainder = divmod(diff.seconds, 3600)
		minutes, seconds = divmod(remainder, 60)
		print(f"diff: {diff}")

		if hours == 0:
			print(colorama.Fore.YELLOW + f"name: {names[i+1].name}, diff: {diff}")

def extract_frame(src_path, suffix, duration, dst_path):
	names = []
	for name in Path(src_path).rglob("*."+suffix):
		names.append(name)
	print(f"video files count: {len(names)}, name: {names[0]}")

	current_video = 1
	for name in names:
		print(f"current video: {str(name)}, {current_video}/{len(names)}")
		current_video += 1
		dir_name = name.name
		dir_name = dir_name[2:-4]

		start_time = datetime.strptime(dir_name, "%Y%m%d%H%M%S")

		dir_name = dst_path + "/" + dir_name
		if os.path.exists(dir_name) and os.path.isdir(dir_name):
			shutil.rmtree(dir_name)
		os.makedirs(dir_name)

		video = VideoFileClip(str(name))
		src_duration = video.duration
		fps = video.fps
		total_frames = int(src_duration * fps)
		video.close()
		# print(f"duratoin src: {duration_src}; fps: {fps}; total frames: {total_frames}"); raise

		frames = []
		frame_rate = total_frames / duration
		# print(f"frame rate: {frame_rate}"); raise
		for i in range(duration):
			frames.append(int(i * frame_rate))

		cap = cv2.VideoCapture(str(name))
		if not cap.isOpened():
			raise FileNotFoundError(colorama.Fore.RED + f"file does not exist: {name}")

		frame_count = 0
		count = 0
		while True:
			ret, frame = cap.read()
			if not ret:
				break

			if frame_count == frames[count]:
				image_name = start_time.strftime("%Y%m%d%H%M%S")
				image_name = dir_name + "/" + str(image_name) + ".jpg"
				# print(f"image name: {image_name}"); raise
				cv2.imwrite(image_name, frame)
				count += 1
				start_time += timedelta(seconds=1)
				if count == len(frames):
					break

			frame_count += 1


if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	extract_frame(args.src_path, args.suffix, args.duration, args.dst_path)

	print(colorama.Fore.GREEN + "====== execution completed ======")
