import colorama
import argparse
import os
import cv2

def parse_args():
	parser = argparse.ArgumentParser(description="extract frames from a video or generate a video from a collection of images")
	parser.add_argument("--task", required=True, type=str, choices=["images", "video"], help="specifies whether to generate images or videos")
	parser.add_argument("--video_file", required=True, type=str, help="video file")
	parser.add_argument("--images_path", required=True, type=str, help="images path")
	parser.add_argument("--frame_interval", type=int, default=25, help="specifies how many frames to extract a frame")
	parser.add_argument("--frame_rate", type=int, default=25, help="frame rate")
	parser.add_argument("--first_image_name", type=int, default=0, help="the name of the first image")

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


if __name__ == "__main__":
    # python test_video_to_images.py --task images --video_file ../../data/1_20210115183444.avi --images_path ../../data/database/detect
	colorama.init(autoreset=True)
	args = parse_args()

	if args.task == "images":
		extract_frame(args.video_file, args.images_path, args.frame_interval, args.first_image_name)
	else:
		generate_video(args.images_path, args.frame_rate, args.video_file)

	print(colorama.Fore.GREEN + "====== execution completed ======")
