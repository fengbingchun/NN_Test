#include "funset.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "perceptron.hpp"
#include "BP.hpp"
#include "CNN.hpp"
#include "linear_regression.hpp"
#include "naive_bayes_classifier.hpp"
#include "logistic_regression.hpp"
#include "common.hpp"
#include "knn.hpp"
#include "decision_tree.hpp"
#include "pca.hpp"
#include "logistic_regression2.hpp"
#include "single_hidden_layer.hpp"
#include "kmeans.hpp"
#include "lrn.hpp"
#include "batch_normalization.hpp"

// ========================= Logistic Regression: Gradient Descent ====================
namespace {

int read_images(const std::string& path, int samples_single_class_num, int image_size, std::unique_ptr<ANN::Database>& data)
{
	const std::vector<std::string> prefix_name {"0_", "1_"};

	for (int i = 1; i <= samples_single_class_num; ++i) {
		for (const auto& prefix : prefix_name) {
			std::string name = std::to_string(i);
			if (i < 10) {
				name = "0000" + name;
			} else if (i < 100) {
				name = "000" + name;
			} else if (i < 1000) {
				name = "00" + name;
			} else if (i < 10000) {
				name = "0" + name;
			}
			name = path + prefix + name + ".jpg";

			cv::Mat mat = cv::imread(name, 0);
			if (mat.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}
			if (mat.cols * mat.rows != image_size || mat.channels() != 1) {
				fprintf(stderr, "image size fail: width: %d, height: %d, channels: %d\n", mat.cols, mat.rows, mat.channels());
				return -1;
			}

			mat.convertTo(mat, CV_32F);
			int index = prefix == "0_" ? (i-1)*2+0 : (i-1)*2+1;
			data->samples[index].resize(image_size);
			memcpy(data->samples[index].data(), mat.data, image_size*sizeof(float));
			data->labels[index] = prefix == "0_" ? 0 : 1;
		}
	}

	return 0;
}

} // namespace

int test_logistic_regression2_gradient_descent()
{
	fprintf(stdout,"Warning: first generate test images: execute demo/DatasetToImage/DatasetToImage: MNISTtoImage\n");

	fprintf(stdout, "load train images ...\n");
#ifdef _MSC_VER
	const std::vector<std::string> image_path{ "E:/GitCode/NN_Test/data/tmp/MNIST/train_images/", "E:/GitCode/NN_Test/data/tmp/MNIST/test_images/"};
	const std::string model{ "E:/GitCode/NN_Test/data/logistic_regression2.model" };
#else
	const std::vector<std::string> image_path{ "data/tmp/MNIST/train_images/", "data/tmp/MNIST/test_images/"};
	const std::string model{ "data/logistic_regression2.model" };
#endif
	const int image_size = 28*28;
	const int samples_single_class_num = 5000;
	auto data1 = std::make_unique<ANN::Database>();
	data1->samples.resize(samples_single_class_num*2);
	data1->labels.resize(samples_single_class_num*2);
	if (read_images(image_path[0], samples_single_class_num, image_size, data1) == -1) return -1;

	ANN::Optimization optim = ANN::Optimization::SGD_Momentum;
	int batch_size = 128;
	float learning_rate = 0.0001;
	int epochs = 100;
	float error = 0.00002;
	float mu = 0.7;
	fprintf(stdout, "optimization method: %d, batch size: %d, learning rate: %f, epochs: %d, error: %f, mu: %f\n",
			optim, batch_size, learning_rate, epochs, error, mu);

	fprintf(stdout, "start train ...\n");
	auto start = std::chrono::steady_clock::now();
	//ANN::LogisticRegression2 lr(ANN::Optimization::BGD, samples_single_class_num * 2); // Batch Gradient Descent, epochs = 10000, correct rete: 0.997778
	ANN::LogisticRegression2 lr(optim, batch_size);
	lr.set_error(error);
	lr.set_mu(mu);
	int ret = lr.init(std::move(data1), image_size, learning_rate, epochs);
	if (ret != 0) {
		fprintf(stderr, "logistic regression init fail: %d\n", ret);
		return -1;
	}

	ret = lr.train(model);
	if (ret != 0) {
		fprintf(stderr, "logistic regression train fail: %d\n", ret);
		return -1;
	}
	auto end = std::chrono::steady_clock::now();
	fprintf(stdout, "train elapsed time: %d seconds\n", std::chrono::duration_cast<std::chrono::seconds>(end - start).count());

	fprintf(stdout, "start predict ...\n");
	const int test_single_class_num = 900;
	const std::vector<std::string> prefix_name {"0_", "1_"};
	ANN::LogisticRegression2 lr2;
	lr2.load_model(model);
	int count = 0;

	for (int i = 1; i <= test_single_class_num; ++i) {
		for (const auto& prefix : prefix_name) {
			std::string name = std::to_string(i);
			if (i < 10) {
				name = "0000" + name;
			} else if (i < 100) {
				name = "000" + name;
			} else if (i < 1000) {
				name = "00" + name;
			}
			name = image_path[1] + prefix + name + ".jpg";

			cv::Mat mat = cv::imread(name, 0);
			if (mat.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}
			if (mat.cols * mat.rows != image_size || mat.channels() != 1) {
				fprintf(stderr, "image size fail: width: %d, height: %d, channels: %d\n", mat.cols, mat.rows, mat.channels());
				return -1;
			}

			mat.convertTo(mat, CV_32F);
			float probability = lr2.predict((float*)mat.data, image_size);
			int label = prefix == "0_" ? 0 : 1;
			if ((probability > 0.5 &&  label == 1) || (probability < 0.5 && label == 0)) ++count;
		}
	}

	float correct_rate = count / (test_single_class_num * 2.);
	fprintf(stdout, "correct rate: %f\n", correct_rate);
	return 0;
}

// ========================= Batch Normalization ====================
int test_batch_normalization()
{
	const std::vector<float> data = { 11.1, -2.2, 23.3, 54.4, 58.5, -16.6,
									-97.7, -28.8, 49.9, -61.3, 52.6, -33.9,
									-2.45, -15.7, 72.4, 9.1, 47.2, 21.7};
	const int number = 3, channels = 1, height = 1, width = 6;

	ANN::BatchNorm bn(number, channels, height, width);
	bn.LoadData(data.data(), data.size());

	std::unique_ptr<float[]> output = bn.Run();

	fprintf(stdout, "result:\n");
	for (int n = 0; n < number; ++n) {
		const float* p = output.get() + n * (channels * height * width);
		for (int c = 0; c < channels; ++c) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					fprintf(stdout, "%f, ", p[c * (height * width) + h * width + w]);
				}
				fprintf(stdout, "\n");
			}
		}
	}

	return 0;
}

// ========================= LRN(Local Response Normalization) ====================
int test_lrn()
{
	int batch = 2, channel = 4, height = 2, width = 2;
	std::vector<float> input{ 1., 5., 9., 13., 2., 6., 10., 14., 3., 7., 11., 15., 4., 8., 12., 16.,
			17., 21., 25., 29., 18., 22., 26., 30., 19., 23., 27., 31., 20., 24., 28., 32.};
	CHECK(batch * channel * height * width == input.size());

	std::unique_ptr<float[]> output(new float[input.size()]);
	ANN::LRN<> lrn;
	lrn.run(input.data(), batch, channel, height, width, output.get());

	auto print = [height, width](const float* data, int length) {
		int size = height * width;
		for (int i = 0; i < length / size; ++i) {
			const float* p = data + i * size;

			for (int j = 0; j < size; ++j) {
				fprintf(stdout, "  %f", p[j]);
			}

			fprintf(stdout, "\n");
		}
	};

	fprintf(stdout, "input:\n"); print(input.data(), input.size());
	fprintf(stdout, "output:\n"); print(output.get(), input.size());

	return 0;
}

// ================================= K-Means ===============================
int test_kmeans()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat tmp = cv::imread(image_path + "0_1.jpg", 0);
	CHECK(tmp.data != nullptr && tmp.channels() == 1);
	const int samples_number{ 80 }, every_class_number{ 20 }, categories_number{ samples_number / every_class_number };

	cv::Mat samples_data(samples_number, tmp.rows * tmp.cols, CV_32FC1);
	cv::Mat labels(samples_number, 1, CV_32FC1);
	float* p1 = reinterpret_cast<float*>(labels.data);

	for (int i = 1; i <= every_class_number; ++i) {
		static const std::vector<std::string> digit{ "0_", "1_", "2_", "3_" };
		CHECK(digit.size() == categories_number);
		static const std::string suffix{ ".jpg" };

		for (int j = 0; j < categories_number; ++j) {
			std::string image_name = image_path + digit[j] + std::to_string(i) + suffix;
			cv::Mat image = cv::imread(image_name, 0);
			CHECK(!image.empty() && image.channels() == 1);
			image.convertTo(image, CV_32FC1);

			image = image.reshape(0, 1);
			tmp = samples_data.rowRange((i - 1) * categories_number + j, (i - 1) * categories_number + j + 1);
			image.copyTo(tmp);

			p1[(i - 1) * categories_number + j] = j;
		}
	}

	std::vector<std::vector<float>> data(samples_data.rows);
	for (int i = 0; i < samples_data.rows; ++i) {
		data[i].resize(samples_data.cols);
		const float* p = (const float*)samples_data.ptr(i);
		memcpy(data[i].data(), p, sizeof(float)* samples_data.cols);
	}

	const int K{ 4 }, attemps{ 100 }, max_iter_count{ 100 };
	const double epsilon{ 0.001 };
	const int flags = ANN::KMEANS_RANDOM_CENTERS;

	std::vector<int> best_labels;
	double compactness_measure{ 0. };
	std::vector<std::vector<float>> centers;

	ANN::kmeans<float>(data, K, best_labels, centers, compactness_measure, max_iter_count, epsilon, attemps, flags);
	fprintf(stdout, "K = %d, attemps = %d, iter count = %d, compactness measure =  %f\n",
		K, attemps, max_iter_count, compactness_measure);

	CHECK(best_labels.size() == samples_number);
	const auto* p2 = best_labels.data();
	for (int i = 1; i <= every_class_number; ++i) {
		for (int j = 0; j < categories_number; ++j) {
			fprintf(stdout, "  %d  ", *p2++);
		}
		fprintf(stdout, "\n");
	}

	return 0;
}

// ====================== single hidden layer(two categories) ===============
int test_single_hidden_layer_train()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat data, labels;

	for (int i = 1; i < 11; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}
	data.convertTo(data, CV_32F);

	std::unique_ptr<float[]> tmp(new float[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0.f;
		else tmp[i] = 1.f;
	}
	labels = cv::Mat(20, 1, CV_32FC1, tmp.get());

	ANN::SingleHiddenLayer<float> shl;
	const float learning_rate{ 0.00001f };
	const int iterations{ 10000 };
	const int hidden_layer_node_num{ static_cast<int>(std::log2(data.cols)) };
	const int hidden_layer_activation_type{ ANN::SingleHiddenLayer<float>::ReLU };
	const int output_layer_activation_type{ ANN::SingleHiddenLayer<float>::Sigmoid };
	int ret = shl.init((float*)data.data, (float*)labels.data, data.rows, data.cols,
		hidden_layer_node_num, learning_rate, iterations, hidden_layer_activation_type, output_layer_activation_type);
	if (ret != 0) {
		fprintf(stderr, "single_hidden_layer(two categories) init fail: %d\n", ret);
		return -1;
	}

#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/single_hidden_layer.model" };
#else
	const std::string model{ "data/single_hidden_layer.model" };
#endif

	ret = shl.train(model);
	if (ret != 0) {
		fprintf(stderr, "single_hidden_layer(two categories) train fail: %d\n", ret);
		return -1;
	}

	return 0;
}

int test_single_hidden_layer_predict()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat data, labels, result;

	for (int i = 11; i < 21; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}
	data.convertTo(data, CV_32F);

	std::unique_ptr<int[]> tmp(new int[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0;
		else tmp[i] = 1;
	}
	labels = cv::Mat(20, 1, CV_32SC1, tmp.get());

	CHECK(data.rows == labels.rows);

#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/single_hidden_layer.model" };
#else
	const std::string model{ "data/single_hidden_layer.model" };
#endif

	ANN::SingleHiddenLayer<float> shl;
	int ret = shl.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load single_hidden_layer(two categories) model fail: %d\n", ret);
		return -1;
	}

	for (int i = 0; i < data.rows; ++i) {
		float probability = shl.predict((float*)(data.row(i).data), data.cols);

		fprintf(stdout, "probability: %.6f, ", probability);
		if (probability > 0.5) fprintf(stdout, "predict result: 1, ");
		else fprintf(stdout, "predict result: 0, ");
		fprintf(stdout, "actual result: %d\n", ((int*)(labels.row(i).data))[0]);
	}

	return 0;
}

// ================================ logistic regression =====================
int test_logistic_regression2_train()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	const std::vector<std::string> prefix_name{ "0_", "1_" };
	const int image_size = 28*28;
	const int samples_single_class_num = 10;
	auto data = std::make_unique<ANN::Database>();
	data->samples.resize(samples_single_class_num*2);
	data->labels.resize(samples_single_class_num*2);

	for (int i = 1; i < 11; ++i) {
		for (const auto& value : prefix_name) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}
			if (image.cols*image.rows != image_size) {
				fprintf(stderr, "image size is not supported: %d, %d\n", image.cols, image.rows);
				return -1;
			}

			image.convertTo(image, CV_32F);
			int index = value == "0_" ? (i-1)*2+0 : (i-1)*2+1;
			data->samples[index].resize(image_size);
			memcpy(data->samples[index].data(), image.data, image_size*sizeof(float));
			data->labels[index] = value == "0_" ? 0 : 1;
		}
	}

	ANN::LogisticRegression2 lr;
	int ret = lr.init(std::move(data), image_size);
	if (ret != 0) {
		fprintf(stderr, "logistic regression init fail: %d\n", ret);
		return -1;
	}

#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/logistic_regression2.model" };
#else
	const std::string model{ "data/logistic_regression2.model" };
#endif
	ret = lr.train(model);
	if (ret != 0) {
		fprintf(stderr, "logistic regression train fail: %d\n", ret);
		return -1;
	}

	return 0;
}

int test_logistic_regression2_predict()
{
#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/logistic_regression2.model" };
#else
	const std::string model{ "data/logistic_regression2.model" };
#endif
	ANN::LogisticRegression2 lr;
	int ret = lr.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load logistic regression model fail: %d\n", ret);
		return -1;
	}

#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	const std::vector<std::string> prefix_name{ "0_", "1_" };
	const int image_size = 28*28;
	const int samples_single_class_num = 10;
	std::vector<std::vector<float>> data(samples_single_class_num*2);
	std::vector<int> labels(samples_single_class_num*2);

	for (int i = 11; i < 21; ++i) {
		for (const auto& value : prefix_name) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}
			if (image.cols*image.rows != image_size) {
				fprintf(stderr, "image size is not supported: %d, %d\n", image.cols, image.rows);
				return -1;
			}

			image.convertTo(image, CV_32F);
			int index = value == "0_" ? (i-11)*2+0 : (i-11)*2+1;
			data[index].resize(image_size);
			memcpy(data[index].data(), image.data, image_size*sizeof(float));
			labels[index] = value == "0_" ? 0 : 1;
		}
	}

	for (int i = 0; i < samples_single_class_num*2; ++i) {
		float probability = lr.predict(data[i].data(), image_size);

		fprintf(stdout, "probability: %.6f, ", probability);
		if (probability > 0.5) fprintf(stdout, "predict result: 1, ");
		else fprintf(stdout, "predict result: 0, ");
		fprintf(stdout, "actual result: %d\n", labels[i]);
	}

	return 0;
}

// =============================== PCA(Principal Components Analysis) ===================
namespace {
void normalize(const std::vector<float>& src, std::vector<unsigned char>& dst)
{
	dst.resize(src.size());
	double dmin = 0, dmax = 255;
	double smin = src[0], smax = smin;

	for (int i = 1; i < src.size(); ++i) {
		if (smin > src[i]) smin = src[i];
		if (smax < src[i]) smax = src[i];
	}

	double scale = (dmax - dmin) * (smax - smin > DBL_EPSILON ? 1. / (smax - smin) : 0);
	double shift = dmin - smin * scale;

	for (int i = 0; i < src.size(); ++i) {
		dst[i] = static_cast<unsigned char>(src[i] * scale + shift);
	}
}
} // namespace

int test_pca()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/database/ORL_Faces/" };
#else
	const std::string image_path{ "data/database/ORL_Faces/" };
#endif
	const std::string image_name{ "1.pgm" };

	std::vector<cv::Mat> images;
	for (int i = 1; i <= 15; ++i) {
		std::string name = image_path + "s" + std::to_string(i) + "/" + image_name;
		cv::Mat mat = cv::imread(name, 0);
		if (!mat.data) {
			fprintf(stderr, "read image fail: %s\n", name.c_str());
			return -1;
		}

		images.emplace_back(mat);
	}
#ifdef _MSC_VER
	save_images(images, "E:/GitCode/NN_Test/data/pca_src.jpg", 5);
#else
	save_images(images, "data/pca_src.jpg", 5);
#endif

	cv::Mat data(images.size(), images[0].rows * images[0].cols, CV_32FC1);
	for (int i = 0; i < images.size(); ++i) {
		cv::Mat image_row = images[i].clone().reshape(1, 1);
		cv::Mat row_i = data.row(i);
		image_row.convertTo(row_i, CV_32F);
	}

	int features_length = images[0].rows * images[0].cols;
	std::vector<std::vector<float>> data_(images.size());
	std::vector<float> labels(images.size(), 0.f);
	for (int i = 0; i < images.size(); ++i) {
		data_[i].resize(features_length);
		memcpy(data_[i].data(), data.row(i).data, sizeof(float)* features_length);
	}

#ifdef _MSC_VER	
	const std::string save_model_file{ "E:/GitCode/NN_Test/data/pca.model" };
#else
	const std::string save_model_file{ "data/pca.model" };
#endif
	ANN::PCA<float> pca;
	pca.load_data(data_, labels);
	double retained_variance{ 0.95 };
	pca.set_retained_variance(retained_variance);
	pca.train(save_model_file);

	const std::string read_model_file{ save_model_file };
	ANN::PCA<float> pca2;
	pca2.load_model(read_model_file);

	std::vector<cv::Mat> result(images.size());
	for (int i = 0; i < images.size(); ++i) {
		std::vector<float> point, reconstruction;
		pca2.project(data_[i], point);
		pca2.back_project(point, reconstruction);
		std::vector<unsigned char> dst;
		normalize(reconstruction, dst);
		cv::Mat tmp(images[i].rows, images[i].cols, CV_8UC1, dst.data());
		tmp.copyTo(result[i]);
	}
#ifdef _MSC_VER
	save_images(result, "E:/GitCode/NN_Test/data/pca_result.jpg", 5);
#else
	save_images(result, "data/pca_result.jpg", 5);
#endif

	return 0;
}

// =============================== decision tree ==============================
int test_decision_tree_train()
{
	// small dataset test
	/*const std::vector<std::vector<float>> data{ { 2.771244718f, 1.784783929f, 0.f },
					{ 1.728571309f, 1.169761413f, 0.f },
					{ 3.678319846f, 2.81281357f, 0.f },
					{ 3.961043357f, 2.61995032f, 0.f },
					{ 2.999208922f, 2.209014212f, 0.f },
					{ 7.497545867f, 3.162953546f, 1.f },
					{ 9.00220326f, 3.339047188f, 1.f },
					{ 7.444542326f, 0.476683375f, 1.f },
					{ 10.12493903f, 3.234550982f, 1.f },
					{ 6.642287351f, 3.319983761f, 1.f } };

	const std::vector<float> classes{ 0.f, 1.f };

	ANN::DecisionTree<float> dt;
	dt.init(data, classes);
	dt.set_max_depth(3);
	dt.set_min_size(1);

	dt.train();
#ifdef _MSC_VER
	const char* model_name = "E:/GitCode/NN_Test/data/decision_tree.model";
#else
	const char* model_name = "data/decision_tree.model";
#endif
	dt.save_model(model_name);

	ANN::DecisionTree<float> dt2;
	dt2.load_model(model_name);
	const std::vector<std::vector<float>> test{{0.6f, 1.9f, 0.f}, {9.7f, 4.3f, 1.f}};
	for (const auto& row : test) {
		float ret = dt2.predict(row);
		fprintf(stdout, "predict result: %.1f, actural value: %.1f\n", ret, row[2]);
	} */

	// banknote authentication dataset
#ifdef _MSC_VER
	const char* file_name = "E:/GitCode/NN_Test/data/database/BacknoteDataset/data_banknote_authentication.txt";
#else
	const char* file_name = "data/database/BacknoteDataset/data_banknote_authentication.txt";
#endif

	std::vector<std::vector<float>> data;
	int ret = read_txt_file<float>(file_name, data, ',', 1372, 5);
	if (ret != 0) {
		fprintf(stderr, "parse txt file fail: %s\n", file_name);
		return -1;
	}

	//fprintf(stdout, "data size: rows: %d\n", data.size());

	const std::vector<float> classes{ 0.f, 1.f };
	ANN::DecisionTree<float> dt;
	dt.init(data, classes);
	dt.set_max_depth(6);
	dt.set_min_size(10);
	dt.train();
#ifdef _MSC_VER
	const char* model_name = "E:/GitCode/NN_Test/data/decision_tree.model";
#else
	const char* model_name = "data/decision_tree.model";
#endif
	dt.save_model(model_name);

	return 0;
}

int test_decision_tree_predict()
{
#ifdef _MSC_VER
	const char* model_name = "E:/GitCode/NN_Test/data/decision_tree.model";
#else
	const char* model_name = "data/decision_tree.model";
#endif
	ANN::DecisionTree<float> dt;
	dt.load_model(model_name);
	int max_depth = dt.get_max_depth();
	int min_size = dt.get_min_size();
	fprintf(stdout, "max_depth: %d, min_size: %d\n", max_depth, min_size);

	std::vector<std::vector<float>> test {{-2.5526f,-7.3625f,6.9255f,-0.66811f,1.f},
				       {-4.5531f,-12.5854f,15.4417f,-1.4983f,1.f},
				       {4.0948f,-2.9674f,2.3689f,0.75429f,0.f},
				       {-1.0401f,9.3987f,0.85998f,-5.3336f,0.f},
				       {1.0637f,3.6957f,-4.1594f,-1.9379f,1.f}};
	for (const auto& row : test) {	
		float ret = dt.predict(row);
		fprintf(stdout, "predict result: %.1f, actual value: %.1f\n", ret, row[4]);
	}

	return 0;
}

// =========================== KNN(K-Nearest Neighbor) ======================
int test_knn_classifier_predict()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	const int K{ 3 };

	cv::Mat tmp = cv::imread(image_path + "0_1.jpg", 0);
	const int train_samples_number{ 40 }, predict_samples_number{ 40 };
	const int every_class_number{ 10 };

	cv::Mat train_data(train_samples_number, tmp.rows * tmp.cols, CV_32FC1);
	cv::Mat train_labels(train_samples_number, 1, CV_32FC1);
	float* p = (float*)train_labels.data;
	for (int i = 0; i < 4; ++i) {
		std::for_each(p + i * every_class_number, p + (i + 1)*every_class_number, [i](float& v){v = (float)i; });
	}

	// train data
	for (int i = 0; i < 4; ++i) {
		static const std::vector<std::string> digit{ "0_", "1_", "2_", "3_" };
		static const std::string suffix{ ".jpg" };

		for (int j = 1; j <= every_class_number; ++j) {
			std::string image_name = image_path + digit[i] + std::to_string(j) + suffix;
			cv::Mat image = cv::imread(image_name, 0);
			CHECK(!image.empty() && image.isContinuous());
			image.convertTo(image, CV_32FC1);

			image = image.reshape(0, 1);
			tmp = train_data.rowRange(i * every_class_number + j - 1, i * every_class_number + j);
			image.copyTo(tmp);
		}
	}

	ANN::KNN<float> knn;
	knn.set_k(K);

	std::vector<std::vector<float>> samples(train_samples_number);
	std::vector<float> labels(train_samples_number);
	const int feature_length{ tmp.rows * tmp.cols };

	for (int i = 0; i < train_samples_number; ++i) {
		samples[i].resize(feature_length);
		const float* p1 = train_data.ptr<float>(i);
		float* p2 = samples[i].data();

		memcpy(p2, p1, feature_length * sizeof(float));
	}

	const float* p1 = (const float*)train_labels.data;
	float* p2 = labels.data();
	memcpy(p2, p1, train_samples_number * sizeof(float));

	knn.set_train_samples(samples, labels);

	// predict datta
	cv::Mat predict_data(predict_samples_number, tmp.rows * tmp.cols, CV_32FC1);
	for (int i = 0; i < 4; ++i) {
		static const std::vector<std::string> digit{ "0_", "1_", "2_", "3_" };
		static const std::string suffix{ ".jpg" };

		for (int j = 11; j <= every_class_number + 10; ++j) {
			std::string image_name = image_path + digit[i] + std::to_string(j) + suffix;
			cv::Mat image = cv::imread(image_name, 0);
			CHECK(!image.empty() && image.isContinuous());
			image.convertTo(image, CV_32FC1);

			image = image.reshape(0, 1);
			tmp = predict_data.rowRange(i * every_class_number + j - 10 - 1, i * every_class_number + j - 10);
			image.copyTo(tmp);
		}
	}

	cv::Mat predict_labels(predict_samples_number, 1, CV_32FC1);
	p = (float*)predict_labels.data;
	for (int i = 0; i < 4; ++i) {
		std::for_each(p + i * every_class_number, p + (i + 1)*every_class_number, [i](float& v){v = (float)i; });
	}

	std::vector<float> sample(feature_length);
	int count{ 0 };
	for (int i = 0; i < predict_samples_number; ++i) {
		float value1 = ((float*)predict_labels.data)[i];
		float value2;
		memcpy(sample.data(), predict_data.ptr<float>(i), feature_length * sizeof(float));

		CHECK(knn.predict(sample, value2) == 0);
		fprintf(stdout, "expected value: %f, actual value: %f\n", value1, value2);

		if (int(value1) == int(value2)) ++count;
	}
	fprintf(stdout, "when K = %d, accuracy: %f\n", K, count * 1.f / predict_samples_number);

	return 0;
}

// ================================ logistic regression =====================
int test_logistic_regression_train()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat data, labels;

	for (int i = 1; i < 11; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}
	data.convertTo(data, CV_32F);

	std::unique_ptr<float[]> tmp(new float[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0.f;
		else tmp[i] = 1.f;
	}
	labels = cv::Mat(20, 1, CV_32FC1, tmp.get());

	ANN::LogisticRegression<float> lr;
	const float learning_rate{ 0.00001f };
	const int iterations{ 250 };
	int reg_kinds = lr.REG_DISABLE; //ANN::LogisticRegression<float>::REG_L1;
	int train_method = lr.MINI_BATCH; //ANN::LogisticRegression<float>::BATCH;
	int mini_batch_size = 5;

	int ret = lr.init((float*)data.data, (float*)labels.data, data.rows, data.cols/*,
		reg_kinds, learning_rate, iterations, train_method, mini_batch_size*/);
	if (ret != 0) {
		fprintf(stderr, "logistic regression init fail: %d\n", ret);
		return -1;
	}

#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/logistic_regression.model" };
#else
	const std::string model{ "data/logistic_regression.model" };
#endif
	ret = lr.train(model);
	if (ret != 0) {
		fprintf(stderr, "logistic regression train fail: %d\n", ret);
		return -1;
	}

	return 0;
}

int test_logistic_regression_predict()
{
#ifdef _MSC_VER
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat data, labels, result;

	for (int i = 11; i < 21; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}
	data.convertTo(data, CV_32F);

	std::unique_ptr<int[]> tmp(new int[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0;
		else tmp[i] = 1;
	}
	labels = cv::Mat(20, 1, CV_32SC1, tmp.get());

	CHECK(data.rows == labels.rows);

#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/logistic_regression.model" };
#else
	const std::string model{ "data/logistic_regression.model" };
#endif
	ANN::LogisticRegression<float> lr;
	int ret = lr.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load logistic regression model fail: %d\n", ret);
		return -1;
	}

	for (int i = 0; i < data.rows; ++i) {
		float probability = lr.predict((float*)(data.row(i).data), data.cols);

		fprintf(stdout, "probability: %.6f, ", probability);
		if (probability > 0.5) fprintf(stdout, "predict result: 1, ");
		else fprintf(stdout, "predict result: 0, ");
		fprintf(stdout, "actual result: %d\n", ((int*)(labels.row(i).data))[0]);
	}

	return 0;
}

// ================================ naive bayes classifier =====================
int test_naive_bayes_classifier_train()
{
	std::vector<ANN::sex_info<float>> info;
	info.push_back({ 6.f, 180.f, 12.f, 1 });
	info.push_back({5.92f, 190.f, 11.f, 1});
	info.push_back({5.58f, 170.f, 12.f, 1});
	info.push_back({5.92f, 165.f, 10.f, 1});
	info.push_back({ 5.f, 100.f, 6.f, 0 });
	info.push_back({5.5f, 150.f, 8.f, 0});
	info.push_back({5.42f, 130.f, 7.f, 0});
	info.push_back({5.75f, 150.f, 9.f, 0});

	ANN::NaiveBayesClassifier<float> naive_bayes;
	int ret = naive_bayes.init(info);
	if (ret != 0) {
		fprintf(stderr, "naive bayes classifier init fail: %d\n", ret);
		return -1;
	}

#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/naive_bayes_classifier.model" };
#else
	const std::string model{ "data/naive_bayes_classifier.model" };
#endif
	ret = naive_bayes.train(model);
	if (ret != 0) {
		fprintf(stderr, "naive bayes classifier train fail: %d\n", ret);
		return -1;
	}

	return 0;
}

int test_naive_bayes_classifier_predict()
{
	ANN::sex_info<float> info = { 6.0f, 130.f, 8.f, -1 };

	ANN::NaiveBayesClassifier<float> naive_bayes;
#ifdef _MSC_VER
	const std::string model{ "E:/GitCode/NN_Test/data/naive_bayes_classifier.model" };
#else
	const std::string model{ "data/naive_bayes_classifier.model" };
#endif
	int ret = naive_bayes.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load naive bayes classifier model fail: %d\n", ret);
		return -1;
	}

	ret = naive_bayes.predict(info);
	if (ret == 0) fprintf(stdout, "It is a female\n");
	else fprintf(stdout, "It is a male\n");

	return 0;
}

// ============================ linear regression =============================
int test_linear_regression_train()
{
	std::vector<float> x{6.2f, 9.5f, 10.5f, 7.7f, 8.6f, 6.9f, 7.3f, 2.2f, 5.7f, 2.f,
		2.5f, 4.f, 5.4f, 2.2f, 7.2f, 12.2f, 5.6f, 9.f, 3.6f, 5.f,
		11.3f, 3.4f, 11.9f, 10.5f, 10.7f, 10.8f, 4.8f};
	std::vector<float> y{ 29.f, 44.f, 36.f, 37.f, 53.f, 18.f, 31.f, 14.f, 11.f, 11.f,
		22.f, 16.f, 27.f, 9.f, 29.f, 46.f, 23.f, 39.f, 15.f, 32.f,
		34.f, 17.f, 46.f, 42.f, 43.f, 34.f, 19.f };
	CHECK(x.size() == y.size());

	ANN::LinearRegression<float> lr;

	lr.set_regression_method(ANN::GRADIENT_DESCENT);
	lr.init(x.data(), y.data(), x.size());

	float learning_rate{ 0.001f };
	int iterations{ 1000 };
#ifdef _MSC_VER
	std::string model{ "E:/GitCode/NN_Test/data/linear_regression.model" };
#else
	std::string model{ "data/linear_regression.model" };
#endif
	int ret = lr.train(model, learning_rate, iterations);
	if (ret != 0) {
		fprintf(stderr, "train fail\n");
		return -1;
	}

	std::cout << lr << std::endl; // y = wx + b

	return 0;
}
int test_linear_regression_predict()
{
	ANN::LinearRegression<float> lr;

#ifdef _MSC_VER
	std::string model{ "E:/GitCode/NN_Test/data/linear_regression.model" };
#else
	std::string model{ "data/linear_regression.model" };
#endif
	int ret = lr.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load model fail: %s\n", model.c_str());
		return -1;
	}

	float x = 13.8f;
	float result = lr.predict(x);
	fprintf(stdout, "input value: %f, result value: %f\n", x, result);

	return 0;
}

// =============================== perceptron =================================
int test_perceptron()
{
	// prepare data
	const int len_data = 20;
	const int feature_dimension = 2;
	float data[len_data][feature_dimension] = {
		{ 10.3, 10.7 }, { 20.1, 100.8 }, { 44.9, 8.0 }, { -2.2, 15.3 }, { -33.3, 77.7 },
		{ -10.4, 111.1 }, { 99.3, -2.2 }, { 222.2, -5.5 }, { 10.1, 10.1 }, { 66.6, 30.2 },
		{ 0.1, 0.2 }, { 1.2, 0.03 }, { 0.5, 4.6 }, { -22.3, -11.1 }, { -88.9, -12.3 },
		{ -333.3, -444.4 }, { -111.2, 0.5 }, { -6.6, 2.9 }, { 3.3, -100.2 }, { 5.6, -88.8 } };
	int label_[len_data] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		-1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

	std::vector<ANN::feature> set_feature;
	std::vector<ANN::label> set_label;

	for (int i = 0; i < len_data; i++) {
		ANN::feature feature_single;
		for (int j = 0; j < feature_dimension; j++) {
			feature_single.push_back(data[i][j]);
		}

		set_feature.push_back(feature_single);
		set_label.push_back(label_[i]);

		feature_single.resize(0);
	}

	// train
	int iterates = 1000;
	float learn_rate = 0.5;
	int size_weight = feature_dimension;
	float bias = 2.5;
	ANN::Perceptron perceptron(iterates, learn_rate, size_weight, bias);
	perceptron.getDataset(set_feature, set_label);
	bool flag = perceptron.train();
	if (flag) {
		std::cout << "data set is linearly separable" << std::endl;
	}
	else {
		std::cout << "data set is linearly inseparable" << std::endl;
		return -1;
	}

	// predict
	ANN::feature feature1;
	feature1.push_back(636.6);
	feature1.push_back(881.8);
	std::cout << "the correct result label is 1, " << "the real result label is: " << perceptron.predict(feature1) << std::endl;

	ANN::feature feature2;
	feature2.push_back(-26.32);
	feature2.push_back(-255.95);
	std::cout << "the correct result label is -1, " << "the real result label is: " << perceptron.predict(feature2) << std::endl;

	return 0;
}

// =================================== BP =====================================
int test_BP_train()
{
	ANN::BP bp1;
	bp1.init();
	bp1.train();

	return 0;
}

int test_BP_predict()
{
	ANN::BP bp2;
#ifdef _MSC_VER
	bool flag = bp2.readModelFile("E:/GitCode/NN_Test/data/bp.model");
#else
	bool flag = bp2.readModelFile("data/bp.model");
#endif
	if (!flag) {
		std::cout << "read bp model error" << std::endl;
		return -1;
	}

	int target[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
#ifdef _MSC_VER
	std::string path_images = "E:/GitCode/NN_Test/data/images/digit/handwriting_1/";
#else
	std::string path_images = "data/images/digit/handwriting_1/";
#endif

	int* data_image = new int[width_image_BP * height_image_BP];

	for (int i = 0; i < 10; i++) {
		char ch[15];
		sprintf(ch, "%d", i);
		std::string str;
		str = std::string(ch);
		str += ".jpg";
		str = path_images + str;

		cv::Mat mat = cv::imread(str, 2 | 4);
		if (!mat.data) {
			std::cout << "read image error" << std::endl;
			return -1;
		}

		if (mat.channels() == 3) {
			cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
		}

		if (mat.cols != width_image_BP || mat.rows != height_image_BP) {
			cv::resize(mat, mat, cv::Size(width_image_BP, height_image_BP));
		}

		memset(data_image, 0, sizeof(int) * (width_image_BP * height_image_BP));

		for (int h = 0; h < mat.rows; h++) {
			uchar* p = mat.ptr(h);
			for (int w = 0; w < mat.cols; w++) {
				if (p[w] > 128) {
					data_image[h* mat.cols + w] = 1;
				}
			}
		}

		int ret = bp2.predict(data_image, mat.cols, mat.rows);
		std::cout << "correct result: " << i << ",    actual result: " << ret << std::endl;
	}

	delete[] data_image;

	return 0;
}

// =================================== CNN ====================================
int test_CNN_train()
{
	ANN::CNN cnn1;
	cnn1.init();
	cnn1.train();

	return 0;
}

int test_CNN_predict()
{
	ANN::CNN cnn2;
#ifdef _MSC_VER
	bool flag = cnn2.readModelFile("E:/GitCode/NN_Test/data/cnn.model");
#else
	bool flag = cnn2.readModelFile("data/cnn.model");
#endif
	if (!flag) {
		std::cout << "read cnn model error" << std::endl;
		return -1;
	}

	int width{ 32 }, height{ 32 };
	std::vector<int> target{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
#ifdef _MSC_VER
	std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_2/" };
#else
	std::string image_path{ "data/images/digit/handwriting_2/" };
#endif

	for (auto i : target) {
		std::string str = std::to_string(i);
		str += ".png";
		str = image_path + str;

		cv::Mat src = cv::imread(str, 0);
		if (src.data == nullptr) {
			fprintf(stderr, "read image error: %s\n", str.c_str());
			return -1;
		}

		cv::Mat tmp(src.rows, src.cols, CV_8UC1, cv::Scalar::all(255));
		cv::subtract(tmp, src, tmp);

		cv::resize(tmp, tmp, cv::Size(width, height));

		auto ret = cnn2.predict(tmp.data, width, height);

		fprintf(stdout, "the actual digit is: %d, correct digit is: %d\n", ret, i);
	}

	return 0;
}

