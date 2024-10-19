import colorama
import argparse
import os
import cv2

def parse_args():
	parser = argparse.ArgumentParser(description="extract frames from a video")
	parser.add_argument("--video_file", type=str, required=True, help="video file")
	parser.add_argument("--dst_images_path", type=str, required=True, help="the location where the image is stored")
	parser.add_argument("--frame_interval", type=int, default=25, help="specifies how many frames to extract a frame")
	parser.add_argument("--first_image_name", type=int, default=0, help="the name of the first image")

	args = parser.parse_args()
	return args

def extract_frame(video_file, dst_images_path, frame_interval, first_image_name):
	os.makedirs(dst_images_path, exist_ok=True)

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
			image_name = os.path.join(dst_images_path, f"{frame_id:08d}.png")
			cv2.imwrite(image_name, frame)
			frame_id += 1

		frame_count += 1

if __name__ == "__main__":
    # python test_video_to_images.py --video_file ../../data/1_20210115183444.avi --dst_images_path ../../data/database/detect
	colorama.init(autoreset=True)
	args = parse_args()

	extract_frame(args.video_file, args.dst_images_path, args.frame_interval, args.first_image_name)

	print(colorama.Fore.GREEN + "====== execution completed ======")
