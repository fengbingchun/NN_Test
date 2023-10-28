#include <assert.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>
#ifdef _MSC_VER
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "BP.hpp"

namespace ANN {

BP::BP()
{
	data_input_train = NULL;
	data_output_train = NULL;
	data_input_test = NULL;
	data_output_test = NULL;
}

BP::~BP()
{
	release();
}

void BP::release()
{
	if (data_input_train) {
		delete[] data_input_train;
	}
	if (data_output_train) {
		delete[] data_output_train;
	}
	if (data_input_test) {
		delete[] data_input_test;
	}
	if (data_output_test) {
		delete[] data_output_test;
	}
}

bool BP::initWeightThreshold()
{
	srand(time(0) + rand());

	for (int i = 0; i < num_node_input_BP; i++) {
		for (int j = 0; j < num_node_hidden_BP; j++) {
			weight1[i][j] = -1 + 2 * ((float)rand()) / RAND_MAX; //[-1, 1]
		}
	}

	for (int i = 0; i < num_node_hidden_BP; i++) {
		for (int j = 0; j < num_node_output_BP; j++) {
			weight2[i][j] = -1 + 2 * ((float)rand()) / RAND_MAX;
		}
	}

	for (int i = 0; i < num_node_hidden_BP; i++) {
		threshold1[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
	}

	for (int i = 0; i < num_node_output_BP; i++) {
		threshold2[i] = -1 + 2 * ((float)rand()) / RAND_MAX;
	}

	return true;
}

static int reverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

static void readMnistImages(std::string filename, int* data_dst, int num_image)
{
	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open());

	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	assert(number_of_images == num_image);
	file.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	file.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);
	assert(n_rows == height_image_BP && n_cols == width_image_BP);

	for (int i = 0; i < number_of_images; ++i) {
		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; ++c) {
				unsigned char temp = 0;
				file.read((char*)&temp, sizeof(temp));
				//data_dst[i * num_node_input_BP + r * n_cols + c] = (int)temp; //formula[1]
				if (temp > 128) {
					data_dst[i * num_node_input_BP + r * n_cols + c] = 1;
				} else {
					data_dst[i * num_node_input_BP + r * n_cols + c] = 0;
				}
			}
		}
	}
}

static void readMnistLabels(std::string filename, int* data_dst, int num_image)
{
	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open());

	int magic_number = 0;
	int number_of_images = 0;
	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);
	assert(number_of_images == num_image);

	for (int i = 0; i < number_of_images; ++i) {
		unsigned char temp = 0;
		file.read((char*)&temp, sizeof(temp));
		data_dst[i * num_node_output_BP + temp] = 1; //formula[2]
	}
}

bool BP::getSrcData()
{
	assert(data_input_train && data_output_train && data_input_test && data_output_test);

#ifdef _MSC_VER
	std::string filename_train_images = "../../../data/database/MNIST/train-images.idx3-ubyte";
	std::string filename_train_labels = "../../../data/database/MNIST/train-labels.idx1-ubyte";
#else
	std::string filename_train_images = "data/database/MNIST/train-images.idx3-ubyte";
	std::string filename_train_labels = "data/database/MNIST/train-labels.idx1-ubyte";
#endif
	readMnistImages(filename_train_images, data_input_train, patterns_train_BP);
	/*unsigned char* p = new unsigned char[784];
	memset(p, 0, sizeof(unsigned char) * 784);
	for (int j = 0, i = 59998 * 784; j< 784; j++, i++) {
		p[j] = (unsigned char)data_input_train[i];
	}
	delete[] p;*/
	readMnistLabels(filename_train_labels, data_output_train, patterns_train_BP);
	/*int* q = new int[10];
	memset(q, 0, sizeof(int) * 10);
	for (int j = 0, i = 59998 * 10; j < 10; j++, i++) {
		q[j] = data_output_train[i];
	}
	delete[] q;*/

#ifdef _MSC_VER
	std::string filename_test_images = "../../../data/database/MNIST/t10k-images.idx3-ubyte";
	std::string filename_test_labels = "../../../data/database/MNIST/t10k-labels.idx1-ubyte";
#else
	std::string filename_test_images = "data/database/MNIST/t10k-images.idx3-ubyte";
	std::string filename_test_labels = "data/database/MNIST/t10k-labels.idx1-ubyte";
#endif
	readMnistImages(filename_test_images, data_input_test, patterns_test_BP);
	readMnistLabels(filename_test_labels, data_output_test, patterns_test_BP);

	return true;
}

void BP::init()
{
	data_input_train = new int[patterns_train_BP * num_node_input_BP];
	memset(data_input_train, 0, sizeof(int) * patterns_train_BP * num_node_input_BP);
	data_output_train = new int[patterns_train_BP * num_node_output_BP];
	memset(data_output_train, 0, sizeof(int) * patterns_train_BP * num_node_output_BP);
	data_input_test = new int[patterns_test_BP * num_node_input_BP];
	memset(data_input_test, 0, sizeof(int) * patterns_test_BP * num_node_input_BP);
	data_output_test = new int[patterns_test_BP * num_node_output_BP];
	memset(data_output_test, 0, sizeof(int) * patterns_test_BP * num_node_output_BP);

	initWeightThreshold();
	getSrcData();
}

float BP::calcActivationFunction(float x)
{
	return 1.0 / (1.0 + std::exp(-x)); //formula[4] formula[5] formula[7]
}

void BP::calcHiddenLayer(const int* data)
{
	for (int i = 0; i < num_node_hidden_BP; i++) {
		float tmp = 0;
		for (int j = 0; j < num_node_input_BP; j++) {
			tmp += data[j] * weight1[j][i];
		}

		tmp -= threshold1[i]; //formula[3]
		output_hiddenLayer[i] = calcActivationFunction(tmp);
	}
}

void BP::calcOutputLayer()
{
	for (int i = 0; i < num_node_output_BP; i++) {
		float tmp = 0;
		for (int j = 0; j < num_node_hidden_BP; j++) {
			tmp += output_hiddenLayer[j] * weight2[j][i];
		}

		tmp -= threshold2[i]; //formula[6]
		output_outputLayer[i] = calcActivationFunction(tmp);
	}
}

void BP::calcAdjuctOutputLayer(const int* data)
{
	for (int i = 0; i < num_node_output_BP; i++) {
		adjust_error_outputLayer[i] = (data[i] - output_outputLayer[i]) *
			output_outputLayer[i] * (1.0 - output_outputLayer[i]); //formula[8], f'(x)= f(x)*(1. - f(x))
	}
}

void BP::calcAdjuctHiddenLayer()
{
	for (int i = 0; i < num_node_hidden_BP; i++) {
		float tmp = 0;
		for (int j = 0; j < num_node_output_BP; j++) {
			tmp += weight2[i][j] * adjust_error_outputLayer[j];
		}

		adjust_error_hiddenLayer[i] = tmp * (output_hiddenLayer[i] * (1.0 - output_hiddenLayer[i])); //formula[9]
	}
}

void BP::updateWeightThresholdOutputLayer()
{
	for (int i = 0; i < num_node_output_BP; i++) {
		for (int j = 0; j < num_node_hidden_BP; j++) {
			weight2[j][i] += (alpha_learning_BP * adjust_error_outputLayer[i] * output_hiddenLayer[j]); //formula[10]
		}

		threshold2[i] += (alpha_learning_BP * adjust_error_outputLayer[i]); //formula[11]
	}
}

void BP::updateWeightThresholdHiddenLayer(const int* data)
{
	for (int i = 0; i < num_node_hidden_BP; i++) {
		for (int j = 0; j < num_node_input_BP; j++) {
			weight1[j][i] += (beta_learning_BP * adjust_error_hiddenLayer[i] * data[j]); //formula[12]
		}

		threshold1[i] += (beta_learning_BP * adjust_error_hiddenLayer[i]); //formula[13]
	}
}

float BP::test()
{
	int count_accuracy = 0;

	for (int num = 0; num < patterns_test_BP; num++) {
		int* p1 = data_input_test + num * num_node_input_BP;
		calcHiddenLayer(p1);
		calcOutputLayer();

		float max_value = -9999;
		int pos = -1;

		for (int i = 0; i < num_node_output_BP; i++) {
			if (output_outputLayer[i] > max_value) {
				max_value = output_outputLayer[i];
				pos = i;
			}
		}

		int* p2 = data_output_test + num * num_node_output_BP;
		if (p2[pos] == 1) {
			count_accuracy++;
		}
#ifdef _MSC_VER
		Sleep(1);
#else
		usleep(1); // 1000
#endif
	}

	return (count_accuracy * 1.0 / patterns_test_BP);
}

bool BP::saveModelFile(const char* name)
{
	FILE* fp = fopen(name, "wb");
	if (fp == NULL) {
		return false;
	}

	int num_node_input = num_node_input_BP;
	int num_node_hidden = num_node_hidden_BP;
	int num_node_output = num_node_output_BP;
	fwrite(&num_node_input, sizeof(int), 1, fp);
	fwrite(&num_node_hidden, sizeof(int), 1, fp);
	fwrite(&num_node_output, sizeof(int), 1, fp);
	fwrite(weight1, sizeof(weight1), 1, fp);
	fwrite(threshold1, sizeof(threshold1), 1, fp);
	fwrite(weight2, sizeof(weight2), 1, fp);
	fwrite(threshold2, sizeof(threshold2), 1, fp);

	fflush(fp);
	fclose(fp);

	return true;
}

bool BP::readModelFile(const char* name)
{
	FILE* fp = fopen(name, "rb");
	if (fp == NULL) {
		return false;
	}

	int num_node_input, num_node_hidden, num_node_output;

	fread(&num_node_input, sizeof(int), 1, fp);
	assert(num_node_input == num_node_input_BP);
	fread(&num_node_hidden, sizeof(int), 1, fp);
	assert(num_node_hidden == num_node_hidden_BP);
	fread(&num_node_output, sizeof(int), 1, fp);
	assert(num_node_output == num_node_output_BP);
	fread(weight1, sizeof(weight1), 1, fp);
	fread(threshold1, sizeof(threshold1), 1, fp);
	fread(weight2, sizeof(weight2), 1, fp);
	fread(threshold2, sizeof(threshold2), 1, fp);

	fflush(fp);
	fclose(fp);

	return true;
}

int BP::predict(const int* data, int width, int height)
{
	assert(data && width == width_image_BP && height == height_image_BP);

	const int* p = data;
	calcHiddenLayer(p);
	calcOutputLayer();

	float max_value = -9999;
	int ret = -1;

	for (int i = 0; i < num_node_output_BP; i++) {
		if (output_outputLayer[i] > max_value) {
			max_value = output_outputLayer[i];
			ret = i;
		}
	}

	return ret;
}

bool BP::train()
{
	int i = 0;
	for (i = 0; i < iterations_BP; i++) {
		std::cout << "iterations : " << i;

		float accuracyRate = test();
		std::cout << ",    accuray rate: " << accuracyRate << std::endl;
		if (accuracyRate > accuracy_rate_BP) {
#ifdef _MSC_VER
			saveModelFile("../../../data/bp.model");
#else	
			saveModelFile("data/bp.model");
#endif
			std::cout << "generate bp model" << std::endl;
			break;
		}

		for (int j = 0; j < patterns_train_BP; j++) {
			int* p1 = data_input_train + j * num_node_input_BP;
			calcHiddenLayer(p1);
			calcOutputLayer();

			int* p2 = data_output_train + j * num_node_output_BP;
			calcAdjuctOutputLayer(p2);
			calcAdjuctHiddenLayer();

			updateWeightThresholdOutputLayer();
			int* p3 = data_input_train + j * num_node_input_BP;
			updateWeightThresholdHiddenLayer(p3);
		}
	}

	if (i == iterations_BP) {
#ifdef _MSC_VER
		saveModelFile("../../../data/bp.model");
#else
		saveModelFile("data/bp.model");
#endif
		std::cout << "generate bp model" << std::endl;
	}

	return true;
}

}
