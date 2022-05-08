#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "common.hpp"

int save_images(const std::vector<cv::Mat>& src, const std::string& name, int row_image_count)
{
	int rows = ((src.size() + row_image_count - 1) / row_image_count);
	int width = src[0].cols, height = src[0].rows;
	cv::Mat dst(height * rows, width * row_image_count, CV_8UC1);

	for (int i = 0; i < src.size(); ++i) {
		int row_start = (i / row_image_count) * height;
		int row_end = row_start + height;
		int col_start = i % row_image_count * width;
		int col_end = col_start + width;
		cv::Mat part = dst(cv::Range(row_start, row_end), cv::Range(col_start, col_end));
		src[i].copyTo(part);
	}

	cv::imwrite(name, dst);

	return 0;
}

int mat_horizontal_concatenate()
{
#ifdef _MSC_VER
	const std::string path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string path{ "data/images/digit/handwriting_0_and_1/" };
#endif

	std::vector<std::string> prefix{ "0_", "1_", "2_", "3_" };
	const int every_class_number{ 20 };
	const int category_number{ (int)prefix.size() };
	std::vector<std::vector<cv::Mat>> mats(category_number);

	cv::Mat mat = cv::imread(path+"0_1.jpg", 0);
	CHECK(!mat.empty());

	const int width{ mat.cols }, height{ mat.rows };

	int count{ 0 };
	for (const auto& value : prefix) {
		for (int i = 1; i <= every_class_number; ++i) {
			std::string name = path + value + std::to_string(i) + ".jpg";
			cv::Mat mat = cv::imread(name, 0);
			if (mat.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}
			if (width != mat.cols || height != mat.rows) {
				fprintf(stderr, "image size not equal\n");
				return -1;
			}

			mats[count].push_back(mat);
		}

		++count;
	}

	std::vector<cv::Mat> middle(category_number);
	for (int i = 0; i < category_number; ++i) {
		cv::hconcat(mats[i].data(), mats[i].size(), middle[i]);
	}

	cv::Mat dst;
	cv::vconcat(middle.data(), middle.size(), dst);
#ifdef _MSC_VER
	cv::imwrite("E:/GitCode/NN_Test/data/result.jpg", dst);
#else
	cv::imwrite("data/result.jpg", dst);
#endif

	return 0;
}

int compare_file(const std::string& name1, const std::string& name2)
{
	std::ifstream infile1;
	infile1.open(name1.c_str(), std::ios::in | std::ios::binary);
	if (!infile1.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	std::ifstream infile2;
	infile2.open(name2.c_str(), std::ios::in | std::ios::binary);
	if (!infile2.is_open()) {
		fprintf(stderr, "failed to open file\n");
		return -1;
	}

	size_t length1 = 0, length2 = 0;

	infile1.read((char*)&length1, sizeof(size_t));
	infile2.read((char*)&length2, sizeof(size_t));

	if (length1 != length2) {
		fprintf(stderr, "their length is mismatch: required length: %d, actual length: %d\n", length1, length2);
		return -1;
	}

	double* data1 = new double[length1];
	double* data2 = new double[length2];

	for (int i = 0; i < length1; i++) {
		infile1.read((char*)&data1[i], sizeof(double));
		infile2.read((char*)&data2[i], sizeof(double));

		if (data1[i] != data2[i]) {
			fprintf(stderr, "no equal: %d: %f, %f\n", i, data1[i], data2[i]);
		}
	}

	delete[] data1;
	delete[] data2;

	infile1.close();
	infile2.close();
}

template<typename T>
void generator_real_random_number(T* data, int length, T a, T b, bool default_random)
{
	if (default_random) { // 每次产生固定的不同的值
		std::default_random_engine generator;

		std::uniform_real_distribution<T> distribution(a, b);
		for (int i = 0; i < length; ++i)
			data[i] = distribution(generator);
	} else { // 每次产生不固定的不同的值
		std::random_device rd;
		std::mt19937 generator(rd());

		std::uniform_real_distribution<T> distribution(a, b);
		for (int i = 0; i < length; ++i)
			data[i] = distribution(generator);
	}
}

template<typename T>
int read_txt_file(const char* name, std::vector<std::vector<T>>& data, const char separator, const int rows, const int cols)
{
	if (typeid(float).name() != typeid(T).name()) {
		fprintf(stderr, "string convert to number only support float type\n");
		return -1;	
	}

	std::ifstream fin(name, std::ios::in);
	if (!fin.is_open()) {
		fprintf(stderr, "open file fail: %s\n", name);
		return -1;
	}
 
	std::string line, cell;
	int col_count = 0, row_count = 0;
	data.clear();
	
	while (std::getline(fin, line)) {
		col_count = 0;
		++row_count;	
		std::stringstream line_stream(line);
		std::vector<T> vec;

		while (std::getline(line_stream, cell, separator)) {
			++col_count;
			vec.emplace_back(std::stof(cell));
		}

		CHECK(cols == col_count);
		data.emplace_back(vec);
	}
	
	CHECK(rows == row_count);

	fin.close();
	return 0;
}

template void generator_real_random_number<float>(float*, int, float, float, bool);
template void generator_real_random_number<double>(double*, int, double, double, bool);
//template int read_txt_file<int>(const char*, std::vector<std::vector<int>>&, const char, const int, const int);
template int read_txt_file<float>(const char*, std::vector<std::vector<float>>&, const char, const int, const int);
//template int read_txt_file<double>(const char*, std::vector<std::vector<double>>&, const char, const int, const int);



