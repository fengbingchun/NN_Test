import argparse
import colorama
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Blog: https://blog.csdn.net/fengbingchun/article/details/150451109

def parse_args():
	parser = argparse.ArgumentParser(description="summary of image similarity algorithms")
	parser.add_argument("--algorithm", required=True, type=str, choices=["mse", "psnr", "ssim", "color_histogram", "hog", "sift", "orb", "lbp", "template_matching", "glcm", "hausdorff", "fft", "dct", "phash", "densenet", "resnet"], help="specify what kind of algorithm")
	parser.add_argument("--src_path1", required=True, type=str, help="source images directory or image name")
	parser.add_argument("--src_path2", required=True, type=str, help="source images directory or image name")
	parser.add_argument("--suffix1", type=str, default="jpg", help="image suffix")
	parser.add_argument("--suffix2", type=str, default="jpg", help="image suffix")

	args = parser.parse_args()
	return args

def _get_images(src_path, suffix):
	images = []
	path = Path(src_path)
	if path.is_dir():
		images = [img for img in path.rglob("*."+suffix)]
	else:
		if path.exists() and path.name.split(".")[1] == suffix:
			images.append(path)

	if not images:
		raise FileNotFoundError(colorama.Fore.RED + f"there are no matching images: src_path:{src_path}; suffix:{suffix}")
	return images

def _check_images(images1, images2):
	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_UNCHANGED)
		if img1 is None:
			raise FileNotFoundError(colorama.Fore.RED + f"image not found: {name1}")

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_UNCHANGED)
			if img1 is None:
				raise FileNotFoundError(colorama.Fore.RED + f"image not found: {name2}")

			if img1.shape != img2.shape:
				raise ValueError(colorama.Fore.RED + f"images must have the same dimensions: {img1.shape}:{img2.shape}")

def _print_results(results):
	for ret in results:
		print(f"{ret[0]} ---- {ret[1]}\t: {ret[2]}")

def mse(images1, images2):
	results = []

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)

			mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
			results.append((name1.name, name2.name, f"{mse:.2f}"))

	return results

def psnr(images1, images2):
	results = []
	max_pixel = 255.0

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)

			mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
			if mse == 0:
				results.append((name1.name, name2.name, float("inf")))
			else:
				psnr = 10 * np.log10((max_pixel ** 2) / mse)
				results.append((name1.name, name2.name, f"{psnr:.2f}"))

	return results

def ssim(images1, images2):
	results = []
	L = 255
	K1 = 0.01; K2 = 0.03
	C1 = (K1 * L) ** 2; C2 = (K2 * L) ** 2
	kernel = (11, 11)
	sigma = 1.5

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE).astype(np.float32)
		mu1 = cv2.GaussianBlur(img1, kernel, sigma)
		mu1_sq = mu1 ** 2
		sigmal_sq = cv2.GaussianBlur(img1 ** 2, kernel, sigma) - mu1_sq

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE).astype(np.float32)
			mu2 = cv2.GaussianBlur(img2, kernel, sigma)
			mu2_sq = mu2 ** 2
			sigmal2_sq = cv2.GaussianBlur(img2 ** 2, kernel, sigma) - mu2_sq

			mu1_mu2 = mu1 * mu2
			sigmal2 = cv2.GaussianBlur(img1 * img2, kernel, sigma) - mu1_mu2

			ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / \
					   ((mu1_sq + mu2_sq + C1) * (sigmal_sq + sigmal2_sq + C2))
			results.append((name1.name, name2.name, f"{np.mean(ssim_map):.4f}"))

	return results

def color_histogram(images1, images2):
	results = []
	bins = [50, 60]

	for name1 in images1:
		img1 = cv2.imread(str(name1))
		hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
		hist1 = cv2.calcHist([hsv1], [0, 1], None, bins, [0, 180, 0, 256])
		cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

		for name2 in images2:
			img2 = cv2.imread(str(name2))
			hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
			hist2 = cv2.calcHist([hsv2], [0, 1], None, bins, [0, 180, 0, 256])
			cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

			score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
			results.append((name1.name, name2.name, f"{score:.4f}"))

	return results

def _resize_padding(img, size, color = (0,0,0)):
	h, w = img.shape[:2]
	ratio = min(size[0] / w, size[1] / h)
	neww, newh = int(w*ratio), int(h*ratio)
	resized = cv2.resize(img, (neww, newh))

	padw = size[0] - neww
	padh = size[1] - newh

	top, bottom = padh // 2, padh - padh // 2
	left, right = padw // 2, padw - padw //2

	return cv2.copyMakeBorder(resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=color)

def _cosine_similarity(feature1, feature2):
	dot = np.dot(feature1, feature2)
	norm1 = np.linalg.norm(feature1)
	norm2 = np.linalg.norm(feature2)

	if norm1 == 0 or norm2 == 0:
		return 0.0

	return dot / (norm1 * norm2)

def hog(images1, images2):
	results = []
	win_size = (488, 488)
	cell_size = (8, 8)
	block_size = (16, 16)
	block_stride = (8, 8)
	nbins = 9

	hog = cv2.HOGDescriptor(_winSize=win_size, _blockSize=block_size, _blockStride=block_stride, _cellSize=cell_size, _nbins=nbins)

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		img1 = _resize_padding(img1, win_size)

		feature1 = hog.compute(img1).flatten()

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			img2 = _resize_padding(img2, win_size)

			feature2 = hog.compute(img2).flatten()

			results.append((name1.name, name2.name, f"{_cosine_similarity(feature1, feature2):.4f}"))

	return results

def sift(images1, images2):
	results = []
	ratio_threshold = 0.75
	sift = cv2.SIFT_create()
	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		kp1, des1 = sift.detectAndCompute(img1, None)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			kp2, des2 = sift.detectAndCompute(img2, None)

			if des1 is None or des2 is None:
				results.append((name1.name, name2.name, 0.0))
				continue

			total_features = min(len(kp1), len(kp2))
			if total_features == 0:
				results.append((name1.name, name2.name, 0.0))
				continue

			matches = bf.knnMatch(des1, des2, k=2)

			good_matches = []
			for match_pair in matches:
				if len(match_pair) == 2:
					m, n = match_pair
					if m.distance < ratio_threshold * n.distance:
						good_matches.append(m)

			results.append((name1.name, name2.name, f"{len(good_matches)/total_features:.4f}"))

	return results

def orb(images1, images2):
	results = []
	nfeatures = 1000
	orb = cv2.ORB_create(nfeatures=nfeatures)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		kp1, des1 = orb.detectAndCompute(img1, None)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			kp2, des2 = orb.detectAndCompute(img2, None)

			if des1 is None or des2 is None:
				results.append((name1.name, name2.name, 0.0))
				continue

			matches = bf.match(des1, des2)
			if len(matches) == 0:
				results.append((name1.name, name2.name, 0.0))
				continue

			good_matches = sorted(matches, key=lambda x: x.distance)
			results.append((name1.name, name2.name, f"{len(good_matches)/min(len(kp1), len(kp2)):.4f}"))

	return results

def _lbp_histogram(img, P=8, R=1):
	lbp = local_binary_pattern(img, P, R, method="uniform")
	hist = cv2.calcHist([lbp.astype(np.float32)], [0], None, [P+2], [0, P+2])
	hist = cv2.normalize(hist, hist).flatten()
	return hist

def lbp(images1, images2):
	results = []

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		hist1 = _lbp_histogram(img1)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			hist2 = _lbp_histogram(img2)

			results.append((name1.name, name2.name, f"{_cosine_similarity(hist1, hist2):.4f}"))

	return results

def template_matching(images1, images2):
	results = []
	method = cv2.TM_CCOEFF_NORMED

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)

			res = cv2.matchTemplate(img2, img1, method)
			min_val, max_val, _, _ = cv2.minMaxLoc(res)

			if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
				results.append((name1.name, name2.name, f"{1-min_val:.4f}"))
			else:
				results.append((name1.name, name2.name, f"{max_val:.4f}"))

	return results

def _get_glcm_features(img):
	distances = [1]
	angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
	features = ["contrast", "correlation", "energy", "homogeneity"]

	img = cv2.equalizeHist(img)
	glcm = graycomatrix(img, distances=distances, angles=angles, symmetric=True, normed=True)
	feat = np.hstack([graycoprops(glcm, f).flatten() for f in features])
	return feat

def glcm(images1, images2):
	results = []

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		feat1 = _get_glcm_features(img1)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			feat2 = _get_glcm_features(img2)

			results.append((name1.name, name2.name, f"{_cosine_similarity(feat1, feat2):.4f}"))

	return results

def _get_hausdorff_contours(img):
	_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	contour = max(contours, key=cv2.contourArea)
	return contour

def hausdorff(images1, images2):
	results = []
	extractor = cv2.createHausdorffDistanceExtractor()

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		contour1 = _get_hausdorff_contours(img1)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			contour2 = _get_hausdorff_contours(img2)

			results.append((name1.name, name2.name, f"{extractor.computeDistance(contour1, contour2):.4f}"))

	return results

def _get_fft_features(img):
	f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
	mag = cv2.magnitude(f[:,:,0], f[:,:,1])
	mag = np.log1p(mag)
	mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
	return mag.flatten()

def fft(images1, images2):
	results = []

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		v1 = _get_fft_features(img1)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			v2 = _get_fft_features(img2)

			results.append((name1.name, name2.name, f"{_cosine_similarity(v1, v2):.4f}"))

	return results

def _get_dct_features(img, low_freq_size):
	dct = cv2.dct(np.float32(img))
	dct = dct[:low_freq_size, :low_freq_size]
	v = dct.flatten()
	v = (v - np.min(v)) / (np.max(v) - np.min(v) + 1e-8)
	return v

def dct(images1, images2):
	results = []
	low_freq_size = 8

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		v1 = _get_dct_features(img1, low_freq_size)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			v2 = _get_dct_features(img2, low_freq_size)

			results.append((name1.name, name2.name, f"{_cosine_similarity(v1, v2):.4f}"))

	return results

def _get_hash_value(img, low_freq_size):
	dct = cv2.dct(np.float32(img))
	dct = dct[:low_freq_size, :low_freq_size]
	mean = np.mean(dct)
	value = (dct > mean).astype(np.uint8).flatten()
	return value

def _hamming_distance(hash1, hash2):
	return np.count_nonzero(hash1 != hash2)

def phash(images1, images2):
	results = []
	hash_size = 32
	low_freq_size = 8

	for name1 in images1:
		img1 = cv2.imread(str(name1), cv2.IMREAD_GRAYSCALE)
		img1 = cv2.resize(img1, (hash_size, hash_size))
		hash_value1 = _get_hash_value(img1, low_freq_size)

		for name2 in images2:
			img2 = cv2.imread(str(name2), cv2.IMREAD_GRAYSCALE)
			img2 = cv2.resize(img2, (hash_size, hash_size))
			hash_value2 = _get_hash_value(img2, low_freq_size)

			dist = _hamming_distance(hash_value1, hash_value2)
			results.append((name1.name, name2.name, f"{1 - dist / hash_value1.size:.4f}"))

	return results

def cnn(images1, images2, net_name):
	if net_name == "densenet":
		model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT) # densenet121-a639ec97.pth
	else:
		model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT) # resnet101-cd907fc2.pth
	model = nn.Sequential(*list(model.children())[:-1])
	# print(f"model: {model}")
	model.eval()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	def _extract_features(name):
		img = Image.open(name).convert("RGB")
		tensor = transform(img).unsqueeze(0).to(device) # [1, 3, 224, 224]
		with torch.no_grad():
			features = model(tensor) # densenet: [1, 1024, 7, 7]; resnet: [1, 2048, 1, 1]
			if net_name == "densenet":
				features = F.adaptive_avg_pool2d(features, (1, 1)) # densenet: [1, 1024, 1, 1]
		return features.cpu().numpy().flatten()

	results = []
	for name1 in images1:
		features1 = _extract_features(str(name1))
		for name2 in images2:
			features2 = _extract_features(str(name2))

			results.append((name1.name, name2.name, f"{_cosine_similarity(features1, features2):.4f}"))

	return results

if __name__ == "__main__":
	colorama.init(autoreset=True)
	args = parse_args()

	images1 = _get_images(args.src_path1, args.suffix1)
	images2 = _get_images(args.src_path2, args.suffix2)
	_check_images(images1, images2)

	if args.algorithm == "mse": # Mean Squared Error
		results = mse(images1, images2)
	elif args.algorithm == "psnr": # Peak Signal-to Noise Ratio
		results = psnr(images1, images2)
	elif args.algorithm == "ssim": # Structural Similarity Index Measure
		results = ssim(images1, images2)
	elif args.algorithm == "color_histogram":
		results = color_histogram(images1, images2)
	elif args.algorithm == "hog": # Histogram of Oriented Gradients
		results = hog(images1, images2)
	elif args.algorithm == "sift": # Scale-Invariant Feature Transform
		results = sift(images1, images2)
	elif args.algorithm == "orb": # Oriented FAST and Rotated BRIEF
		results = orb(images1, images2)
	elif args.algorithm == "lbp": # Local Binary Patten
		results = lbp(images1, images2)
	elif args.algorithm == "template_matching":
		results = template_matching(images1, images2)
	elif args.algorithm == "glcm": # Gray-Level Co-occurrence Matrix
		results = glcm(images1, images2)
	elif args.algorithm == "hausdorff": # Hausdorff distance
		results = hausdorff(images1, images2)
	elif args.algorithm == "fft": # Fast Fourier Transform
		results = fft(images1, images2)
	elif args.algorithm == "dct": # Discrete Cosine Transform
		results = dct(images1, images2)
	elif args.algorithm == "phash": # Perceptual Hash
		results = phash(images1, images2)
	elif args.algorithm == "densenet" or args.algorithm == "resnet":
		results = cnn(images1, images2, args.algorithm)

	_print_results(results)

	print(colorama.Fore.GREEN + "====== execution completed ======")
