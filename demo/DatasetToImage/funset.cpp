#include "funset.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

///////////////////////// ORL Face /////////////////////////////
int ORLFacestoImage()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/79008891
#ifdef _MSC_VER
	const std::string path{ "../../../data/database/ORL_Faces/" };
#else
	const std::string path{ "data/database/ORL_Faces/" };
#endif
	cv::Mat dst;
	int height, width;

	for (int i = 1; i <= 40; ++i) {
		std::string directory = path + "s" + std::to_string(i) + "/";

		for (int j = 1; j <= 10; ++j) {
			std::string image_name = directory + std::to_string(j) + ".pgm";
			cv::Mat mat = cv::imread(image_name, 0);
			if (!mat.data) {
				fprintf(stderr, "read image fail: %s\n", image_name.c_str());
			}

			//std::string save_image_name = directory + std::to_string(j) + ".png";
			//cv::imwrite(save_image_name, mat);

			if (i == 1 && j == 1) {
				height = mat.rows;
				width = mat.cols;
				dst = cv::Mat(height * 20, width * 20, CV_8UC1);
			}

			int y_start = (i - 1) / 2 * height;
			int y_end = y_start + height;
			int x_start = (i - 1) % 2 * 10 * width + (j - 1) * width;
			int x_end = x_start + width;
			cv::Mat copy = dst(cv::Range(y_start, y_end), cv::Range(x_start, x_end));
			mat.copyTo(copy);
		}
	}

	int new_width = 750;
	float factor = dst.cols * 1.f / new_width;
	int new_height = dst.rows / factor;
	cv::resize(dst, dst, cv::Size(new_width, new_height));
#ifdef _MSC_VER
	cv::imwrite("../../../data/orl_faces_dataset.png", dst);
#else
	cv::imwrite("data/orl_faces_dataset.png", dst);
#endif

	return 0;
}

////////////////////////// MNIST /////////////////////////////////
namespace {
int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(std::string filename, std::vector<cv::Mat> &vec)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for (int i = 0; i < number_of_images; ++i) {
			cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
			for (int r = 0; r < n_rows; ++r) {
				for (int c = 0; c < n_cols; ++c) {
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					tp.at<uchar>(r, c) = (int)temp;
				}
			}
			vec.push_back(tp);
		}

		file.close();
	}
}

void read_Mnist_Label(std::string filename, std::vector<int> &vec)
{
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		for (int i = 0; i < number_of_images; ++i) {
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			vec[i] = (int)temp;
		}

		file.close();
	}
}

std::string GetImageName(int number, int arr[])
{
	std::string str1, str2;

	for (int i = 0; i < 10; i++) {
		if (number == i) {
			arr[i]++;
			str1 = std::to_string(arr[i]);

			if (arr[i] < 10) {
				str1 = "0000" + str1;
			} else if (arr[i] < 100) {
				str1 = "000" + str1;
			} else if (arr[i] < 1000) {
				str1 = "00" + str1;
			} else if (arr[i] < 10000) {
				str1 = "0" + str1;
			}

			break;
		}
	}

	str2 = std::to_string(number) + "_" + str1;

	return str2;
}

int write_images_to_file(const std::string& file_name, const std::vector<cv::Mat>& image_data,
	int magic_number, int image_number, int image_rows, int image_cols)
{
	if (image_number > image_data.size()) {
		fprintf(stderr, "Error: image_number > image_data.size(): \
			image_number: %d, image_data.size: %d", image_number, image_data.size());
		return -1;
	}

	std::ofstream file(file_name, std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "Error: open file fail: %s\n", file_name.c_str());
		return -1;
	}

	int tmp = ReverseInt(magic_number);
	file.write((char*)&tmp, sizeof(int));
	tmp = ReverseInt(image_number);
	file.write((char*)&tmp, sizeof(int));
	tmp = ReverseInt(image_rows);
	file.write((char*)&tmp, sizeof(int));
	tmp = ReverseInt(image_cols);
	file.write((char*)&tmp, sizeof(int));

	int size = image_rows * image_cols;
	for (int i = 0; i < image_number; ++i) {
		file.write((char*)image_data[i].data, sizeof(unsigned char) * size);
	}

	file.close();
	return 0;
}

int write_labels_to_file(const std::string& file_name, const std::vector<int>& label_data,
	int magic_number, int label_number)
{
	if (label_number > label_data.size()) {
		fprintf(stderr, "Error: label_number > label_data.size(): \
			label_number: %d, label_data.size: %d", label_number, label_data.size());
		return -1;
	}

	std::ofstream file(file_name, std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "Error: open file fail: %s\n", file_name.c_str());
		return -1;
	}

	int tmp = ReverseInt(magic_number);
	file.write((char*)&tmp, sizeof(int));
	tmp = ReverseInt(label_number);
	file.write((char*)&tmp, sizeof(int));

	std::unique_ptr<unsigned char[]> labels(new unsigned char[label_number]);
	for (int i = 0; i < label_number; ++i) {
		labels[i] = static_cast<unsigned char>(label_data[i]);
	}
	file.write((char*)labels.get(), sizeof(unsigned char) * label_number);

	file.close();
	return 0;
}
} // namespace //mnist

int ImageToMNIST()
{
	// Blog: https://blog.csdn.net/fengbingchun/article/details/87890701
	// read images
#ifdef _MSC_VER
	std::string filename_test_images = "../../../data/database/MNIST/t10k-images.idx3-ubyte";
#else
	std::string filename_test_images = "data/database/MNIST/t10k-images.idx3-ubyte";
#endif
	const int number_of_test_images = 10000;
	std::vector<cv::Mat> vec_test_images;

	read_Mnist(filename_test_images, vec_test_images);
	if (vec_test_images.size() != number_of_test_images) {
		fprintf(stderr, "Error: fail to parse t10k-images.idx3-ubyte file: %d\n", vec_test_images.size());
		return -1;
	}

	// read labels
#ifdef _MSC_VER
	std::string filename_test_labels = "../../../data/database/MNIST/t10k-labels.idx1-ubyte";
#else
	std::string filename_test_labels = "data/database/MNIST/t10k-labels.idx1-ubyte";
#endif
	std::vector<int> vec_test_labels(number_of_test_images);

	read_Mnist_Label(filename_test_labels, vec_test_labels);

	// write images
	const int image_magic_number = 2051; // 0x00000803
	const int image_number = 10000;
	const int image_rows = 28;
	const int image_cols = 28;
#ifdef _MSC_VER
	const std::string images_save_file_name = "../../../data/new_t10k-images.idx3-ubyte";
#else
	const std::string images_save_file_name = "data/new_t10k-images.idx3-ubyte";
#endif

	if (write_images_to_file(images_save_file_name, vec_test_images, image_magic_number,
		image_number, image_rows, image_cols) != 0) {
		fprintf(stderr, "Error: write images to file fail\n");
		return -1;
	}

	// write labels
	const int label_magic_number = 2049; // 0x00000801
	const int label_number = 10000;
#ifdef _MSC_VER
	const std::string labels_save_file_name = "../../../data/new_t10k-labels.idx1-ubyte";
#else
	const std::string labels_save_file_name = "data/new_t10k-labels.idx1-ubyte";
#endif

	if (write_labels_to_file(labels_save_file_name, vec_test_labels, label_magic_number, label_number) != 0) {
		fprintf(stderr, "Error: write labels to file fail\n");
		return -1;
	}

	return 0;
}

int MNISTtoImage()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/49611549
	// reference: http://eric-yuan.me/cpp-read-mnist/
	// test images and test labels
	// read MNIST image into OpenCV Mat vector
#ifdef _MSC_VER
	std::string filename_test_images = "../../../data/database/MNIST/t10k-images.idx3-ubyte";
#else
	std::string filename_test_images = "data/database/MNIST/t10k-images.idx3-ubyte";
#endif
	int number_of_test_images = 10000;
	std::vector<cv::Mat> vec_test_images;

	read_Mnist(filename_test_images, vec_test_images);

	// read MNIST label into int vector
#ifdef _MSC_VER
	std::string filename_test_labels = "../../../data/database/MNIST/t10k-labels.idx1-ubyte";
#else
	std::string filename_test_labels = "data/database/MNIST/t10k-labels.idx1-ubyte";
#endif
	std::vector<int> vec_test_labels(number_of_test_images);

	read_Mnist_Label(filename_test_labels, vec_test_labels);

	if (vec_test_images.size() != vec_test_labels.size()) {
		std::cout << "parse MNIST test file error" << std::endl;
		return -1;
	}

	// save test images
	int count_digits[10];
	std::fill(&count_digits[0], &count_digits[0] + 10, 0);

	fprintf(stdout, "##### Warning: need to create ../../../data/tmp/MNIST/test_images directory by yourself in windows.\n");
#ifdef _MSC_VER
	std::string save_test_images_path = "../../../data/tmp/MNIST/test_images/";
#else
	std::string save_test_images_path = "data/tmp/MNIST/test_images/";
#endif

	for (int i = 0; i < vec_test_images.size(); i++) {
		int number = vec_test_labels[i];
		std::string image_name = GetImageName(number, count_digits);
		image_name = save_test_images_path + image_name + ".jpg";

		cv::imwrite(image_name, vec_test_images[i]);
	}

	// train images and train labels
	// read MNIST image into OpenCV Mat vector
#ifdef _MSC_VER
	std::string filename_train_images = "../../../data/database/MNIST/train-images.idx3-ubyte";
#else
	std::string filename_train_images = "data/database/MNIST/train-images.idx3-ubyte";
#endif
	int number_of_train_images = 60000;
	std::vector<cv::Mat> vec_train_images;

	read_Mnist(filename_train_images, vec_train_images);

	// read MNIST label into int vector
#ifdef _MSC_VER
	std::string filename_train_labels = "../../../data/database/MNIST/train-labels.idx1-ubyte";
#else
	std::string filename_train_labels = "data/database/MNIST/train-labels.idx1-ubyte";
#endif
	std::vector<int> vec_train_labels(number_of_train_images);

	read_Mnist_Label(filename_train_labels, vec_train_labels);

	if (vec_train_images.size() != vec_train_labels.size()) {
		std::cout << "parse MNIST train file error" << std::endl;
		return -1;
	}

	// save train images
	std::fill(&count_digits[0], &count_digits[0] + 10, 0);

	fprintf(stdout, "##### Warning: need to create ../../../data/tmp/MNIST/train_images directory by yourself in windows.\n");
#ifdef _MSC_VER
	std::string save_train_images_path = "../../../data/tmp/MNIST/train_images/";
#else
	std::string save_train_images_path = "data/tmp/MNIST/train_images/";
#endif

	for (int i = 0; i < vec_train_images.size(); i++) {
		int number = vec_train_labels[i];
		std::string image_name = GetImageName(number, count_digits);
		image_name = save_train_images_path + image_name + ".jpg";

		cv::imwrite(image_name, vec_train_images[i]);
	}

	// save big imags
#ifdef _MSC_VER
	std::string images_path = "../../../data/tmp/MNIST/train_images/";
#else
	std::string images_path = "data/tmp/MNIST/train_images/";
#endif
	int width = 28 * 20;
	int height = 28 * 10;
	cv::Mat dst(height, width, CV_8UC1);

	for (int i = 0; i < 10; i++) {
		for (int j = 1; j <= 20; j++) {
			int x = (j-1) * 28;
			int y = i * 28;
			cv::Mat part = dst(cv::Rect(x, y, 28, 28));

			std::string str = std::to_string(j);
			if (j < 10)
				str = "0000" + str;
			else
				str = "000" + str;

			str = std::to_string(i) + "_" + str + ".jpg";
			std::string input_image = images_path + str;

			cv::Mat src = cv::imread(input_image, 0);
			if (src.empty()) {
				fprintf(stderr, "read image error: %s\n", input_image.c_str());
				return -1;
			}

			src.copyTo(part);
		}
	}

	std::string output_image = images_path + "result.png";
	cv::imwrite(output_image, dst);

	return 0;
}

////////////////////// CIFAR /////////////////////////////
namespace {
void write_image_cifar(const cv::Mat& bgr, const std::string& image_save_path, const std::vector<int>& label_count, int label_class)
{
	std::string str = std::to_string(label_count[label_class]);

	if (label_count[label_class] < 10) {
		str = "0000" + str;
	} else if (label_count[label_class] < 100) {
		str = "000" + str;
	} else if (label_count[label_class] < 1000) {
		str = "00" + str;
	} else if (label_count[label_class] < 10000) {
		str = "0" + str;
	} else {
		fprintf(stderr, "save image name fail\n");
		return;
	}

	str = std::to_string(label_class) + "_" + str + ".png";
	str = image_save_path + str;

	cv::imwrite(str, bgr);
}

void read_cifar_10(const std::string& bin_name, const std::string& image_save_path, int image_count, std::vector<int>& label_count)
{
	int image_width = 32;
	int image_height = 32;

	std::ifstream file(bin_name, std::ios::binary);
	if (file.is_open()) {
		for (int i = 0; i < image_count; ++i) {
			cv::Mat red = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat green = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat blue = cv::Mat::zeros(image_height, image_width, CV_8UC1);

			int label_class = 0;
			file.read((char*)&label_class, 1);
			label_count[label_class]++;

			file.read((char*)red.data, 1024);
			file.read((char*)green.data, 1024);
			file.read((char*)blue.data, 1024);

			std::vector<cv::Mat> tmp{ blue, green, red };
			cv::Mat bgr;
			cv::merge(tmp, bgr);

			write_image_cifar(bgr, image_save_path, label_count, label_class);
		}

		file.close();
	}
}

void write_image_cifar(const cv::Mat& bgr, const std::string& image_save_path,
	const std::vector<std::vector<int>>& label_count, int label_class_coarse, int label_class_fine)
{
	std::string str = std::to_string(label_count[label_class_coarse][label_class_fine]);

	if (label_count[label_class_coarse][label_class_fine] < 10) {
		str = "0000" + str;
	} else if (label_count[label_class_coarse][label_class_fine] < 100) {
		str = "000" + str;
	} else if (label_count[label_class_coarse][label_class_fine] < 1000) {
		str = "00" + str;
	} else if (label_count[label_class_coarse][label_class_fine] < 10000) {
		str = "0" + str;
	} else {
		fprintf(stderr, "save image name fail\n");
		return;
	}

	str = std::to_string(label_class_coarse) + "_" + std::to_string(label_class_fine) + "_" + str + ".png";
	str = image_save_path + str;

	cv::imwrite(str, bgr);
}

void read_cifar_100(const std::string& bin_name, const std::string& image_save_path, int image_count, std::vector<std::vector<int>>& label_count)
{
	int image_width = 32;
	int image_height = 32;

	std::ifstream file(bin_name, std::ios::binary);
	if (file.is_open()) {
		for (int i = 0; i < image_count; ++i) {
			cv::Mat red = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat green = cv::Mat::zeros(image_height, image_width, CV_8UC1);
			cv::Mat blue = cv::Mat::zeros(image_height, image_width, CV_8UC1);

			int label_class_coarse = 0;
			file.read((char*)&label_class_coarse, 1);
			int label_class_fine = 0;
			file.read((char*)&label_class_fine, 1);
			label_count[label_class_coarse][label_class_fine]++;

			file.read((char*)red.data, 1024);
			file.read((char*)green.data, 1024);
			file.read((char*)blue.data, 1024);

			std::vector<cv::Mat> tmp{ blue, green, red };
			cv::Mat bgr;
			cv::merge(tmp, bgr);

			write_image_cifar(bgr, image_save_path, label_count, label_class_coarse, label_class_fine);
		}

		file.close();
	}
}
} // namespace // cifar

int CIFAR10toImage()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/53560637
#ifdef _MSC_VER
	std::string images_path = "../../../data/database/CIFAR/CIFAR-10/";
#else
	std::string images_path = "data/database/CIFAR/CIFAR-10/";
#endif
	
	fprintf(stdout, "##### Warning: need to create ../../../data/tmp/cifar-10_train directory by yourself in windows.\n");
	// train image
	std::vector<int> label_count(10, 0);
	for (int i = 1; i <= 5; i++) {
		std::string bin_name = images_path + "data_batch_" + std::to_string(i) + ".bin";
#ifdef _MSC_VER
		std::string image_save_path = "../../../data/tmp/cifar-10_train/";
#else
		std::string image_save_path = "data/tmp/cifar-10_train/";
#endif
		int image_count = 10000;

		read_cifar_10(bin_name, image_save_path, image_count, label_count);
	}

	// test image
	std::fill(&label_count[0], &label_count[0] + 10, 0);
	std::string bin_name = images_path + "test_batch.bin";

	fprintf(stdout, "##### Warning: need to create ../../../data/tmp/cifar-10_test directory by yourself in windows.\n");
#ifdef _MSC_VER
	std::string image_save_path = "../../../data/tmp/cifar-10_test/";
#else
	std::string image_save_path = "data/tmp/cifar-10_test/";
#endif
	int image_count = 10000;

	read_cifar_10(bin_name, image_save_path, image_count, label_count);

	// save big imags
#ifdef _MSC_VER
	images_path = "../../../data/tmp/cifar-10_train/";
#else
	images_path = "data/tmp/cifar-10_train/";
#endif
	int width = 32 * 20;
	int height = 32 * 10;
	cv::Mat dst(height, width, CV_8UC3);

	for (int i = 0; i < 10; i++) {
		for (int j = 1; j <= 20; j++) {
			int x = (j - 1) * 32;
			int y = i * 32;
			cv::Mat part = dst(cv::Rect(x, y, 32, 32));

			std::string str = std::to_string(j);
			if (j < 10)
				str = "0000" + str;
			else
				str = "000" + str;

			str = std::to_string(i) + "_" + str + ".png";
			std::string input_image = images_path + str;

			cv::Mat src = cv::imread(input_image, 1);
			if (src.empty()) {
				fprintf(stderr, "read image error: %s\n", input_image.c_str());
				return -1;
			}

			src.copyTo(part);
		}
	}

	std::string output_image = images_path + "result.png";
	cv::imwrite(output_image, dst);

	return 0;
}

int CIFAR100toImage()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/53560637
#ifdef _MSC_VER
	std::string images_path = "../../../data/database/CIFAR/CIFAR-100/";
#else
	std::string images_path = "data/database/CIFAR/CIFAR-100/";
#endif
	// train image
	std::vector<std::vector<int>> label_count;
	label_count.resize(20);
	for (int i = 0; i < 20; i++) {
		label_count[i].resize(100);
		std::fill(&label_count[i][0], &label_count[i][0] + 100, 0);
	}

	std::string bin_name = images_path + "train.bin";
	fprintf(stdout, "##### Warning: need to create ../../../data/tmp/cifar-100_train directory by yourself in windows.\n");
	fprintf(stdout, "##### Warning: need to decompress ../../../data/database/CIFAR/CIFAR-100/train.7z.* files by yourself in windows.\n");
#ifdef _MSC_VER
	std::string image_save_path = "../../../data/tmp/cifar-100_train/";
#else
	std::string image_save_path = "data/tmp/cifar-100_train/";
#endif
	int image_count = 50000;

	read_cifar_100(bin_name, image_save_path, image_count, label_count);

	// test image
	for (int i = 0; i < 20; i++) {
		label_count[i].resize(100);
		std::fill(&label_count[i][0], &label_count[i][0] + 100, 0);
	}
	bin_name = images_path + "test.bin";
	fprintf(stdout, "##### Warning: need to create ../../../data/tmp/cifar-100_test directory by yourself in windows.\n");
#ifdef _MSC_VER
	image_save_path = "../../../data/tmp/cifar-100_test/";
#else
	image_save_path = "data/tmp/cifar-100_test/";
#endif
	image_count = 10000;

	read_cifar_100(bin_name, image_save_path, image_count, label_count);

	// save big imags
#ifdef _MSC_VER
	images_path = "../../../data/tmp/cifar-100_train/";
#else
	images_path = "data/tmp/cifar-100_train/";
#endif
	int width = 32 * 20;
	int height = 32 * 100;
	cv::Mat dst(height, width, CV_8UC3);
	std::vector<std::string> image_names;

	for (int j = 0; j < 20; j++) {
		for (int i = 0; i < 100; i++) {
			std::string str1 = std::to_string(j);
			std::string str2 = std::to_string(i);
			std::string str = images_path + str1 + "_" + str2 + "_00001.png";
			cv::Mat src = cv::imread(str, 1);
			if (src.data) {
				for (int t = 1; t < 21; t++) {
					if (t < 10)
						str = "0000" + std::to_string(t);
					else
						str = "000" + std::to_string(t);

					str = images_path + str1 + "_" + str2 + "_" + str + ".png";
					image_names.push_back(str);
				}
			}
		}
	}

	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 20; j++) {
			int x = j * 32;
			int y = i * 32;
			cv::Mat part = dst(cv::Rect(x, y, 32, 32));
			cv::Mat src = cv::imread(image_names[i * 20 + j], 1);
			if (src.empty()) {
				fprintf(stderr, "read image fail: %s\n", image_names[i * 20 + j].c_str());
				return -1;
			}

			src.copyTo(part);
		}
	}

	std::string output_image = images_path + "result.png";
	cv::imwrite(output_image, dst);

	cv::Mat src = cv::imread(output_image, 1);
	if (src.empty()) {
		fprintf(stderr, "read result image fail: %s\n", output_image.c_str());
		return -1;
	}
	for (int i = 0; i < 4; i++) {
		cv::Mat dst = src(cv::Rect(0, i * 800, 640, 800));
		std::string str = images_path + "result_" + std::to_string(i + 1) + ".png";
		cv::imwrite(str, dst);
	}

	return 0;
}
