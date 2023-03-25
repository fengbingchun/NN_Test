#include "funset.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <tiny_cnn/tiny_cnn.h>
#include <opencv2/opencv.hpp>

// Blog: http://blog.csdn.net/fengbingchun/article/details/50615176
//       http://blog.csdn.net/fengbingchun/article/details/50573841

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

// rescale output to 0-100
template <typename Activation>
static double rescale(double x)
{
	Activation a;
	return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

static void construct_net(network<mse, adagrad>& nn)
{
	// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
	static const bool tbl[] = {
		O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
		O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
		O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
		X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
		X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
		X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
	};
#undef O
#undef X

	// construct nets
	nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
		<< average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
		<< convolutional_layer<tan_h>(14, 14, 5, 6, 16,
		connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
		<< average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
		<< convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
		<< fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out
}

static void train_lenet(std::string data_dir_path)
{
	// specify loss-function and learning strategy
	network<mse, adagrad> nn;
	construct_net(nn);

	std::cout << "load models..." << std::endl;
	// load MNIST dataset
	std::vector<label_t> train_labels, test_labels;
	std::vector<vec_t> train_images, test_images;

	parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

	std::cout << "start training" << std::endl;
	progress_display disp(train_images.size());
	timer t;
	int minibatch_size = 1; //10;
	int num_epochs = 30;

	nn.optimizer().alpha *= std::sqrt(minibatch_size);

	// create callback
	auto on_enumerate_epoch = [&](){
		std::cout << t.elapsed() << "s elapsed." << std::endl;
		tiny_cnn::result res = nn.test(test_images, test_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&](){
		disp += minibatch_size;
	};

	// training
	nn.train(train_images, train_labels, minibatch_size, num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;
	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);

	// save networks
#ifdef _MSC_VER
	std::ofstream ofs("../../../data/LeNet-weights");
#else
	std::ofstream ofs("data/LeNet-weights");
#endif
	ofs << nn;
}

// convert tiny_cnn::image to cv::Mat and resize
cv::Mat image2mat(image<>& img)
{
	cv::Mat ori(img.height(), img.width(), CV_8U, &img.at(0, 0));
	cv::Mat resized;
	cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
	return resized;
}

static void convert_image(const std::string& imagefilename, double minv, double maxv, int w, int h, vec_t& data)
{
	auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
	if (img.data == nullptr) return; // cannot open, or it's not an image

	cv::Mat_<uint8_t> resized;
	cv::resize(img, resized, cv::Size(w, h));

	// mnist dataset is "white on black", so negate required
	std::transform(resized.begin(), resized.end(), std::back_inserter(data),
		[=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

static void recognize(const std::string& dictionary, const std::string& filename, int target)
{
	network<mse, adagrad> nn;
	construct_net(nn);

	// load nets
	std::ifstream ifs(dictionary.c_str());
	ifs >> nn;

	// convert imagefile to vec_t
	vec_t data;
	convert_image(filename, -1.0, 1.0, 32, 32, data);

	// recognize
	auto res = nn.predict(data);
	std::vector<std::pair<double, int> > scores;

	// sort & print top-3
	for (int i = 0; i < 10; i++)
		scores.emplace_back(rescale<tan_h>(res[i]), i);

	std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

	for (int i = 0; i < 3; i++)
		std::cout << scores[i].second << "," << scores[i].first << std::endl;

	std::cout << "the actual digit is: " << scores[0].second << ", correct digit is: " << target << std::endl;

	// visualize outputs of each layer
	//for (size_t i = 0; i < nn.depth(); i++) {
	//	auto out_img = nn[i]->output_to_image();
	//	cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
	//}
	//// visualize filter shape of first convolutional layer
	//auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
	//cv::imshow("weights:", image2mat(weight));

	//cv::waitKey(0);
}

int test_tiny_cnn_train()
{
#ifdef _MSC_VER
	std::string data_path = "../../../data/database/MNIST";
#else
	std::string data_path = "data/database/MNIST";
#endif
	train_lenet(data_path);

	return 0;
}

int test_tiny_cnn_predict()
{
#ifdef _MSC_VER
	std::string model_path = "../../../data/LeNet-weights";
	std::string image_path = "../../../data/images/digit/handwriting_2/";
#else
	std::string model_path = "data/LeNet-weights";
	std::string image_path = "data/images/digit/handwriting_2/";
#endif
	int target[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	for (int i = 0; i < 10; i++) {
		std::string str = std::to_string(i);
		str += ".png";
		str = image_path + str;

		recognize(model_path, str, target[i]);
	}

	return 0;
}
