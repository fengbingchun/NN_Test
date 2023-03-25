#include "funset.hpp"
#include <string>
#include <algorithm>
#include "tiny_dnn/tiny_dnn.h"

// Blog: https://blog.csdn.net/fengbingchun/article/details/118959997
int test_dnn_batch_normalization()
{
	const std::vector<float> data = { 11.1, -2.2, 23.3, 54.4, 58.5, -16.6,
								-97.7, -28.8, 49.9, -61.3, 52.6, -33.9,
								-2.45, -15.7, 72.4, 9.1, 47.2, 21.7 };
	const int number = 3, channels = 1, height = 1, width = 6;
	const int spatial_size = height * width;

	tiny_dnn::tensor_t in_data(number), out_data(number);
	for (int n = 0; n < number; ++n) {
		in_data[n].resize(spatial_size * channels);
		out_data[n].resize(spatial_size * channels);

		int offset = n * (spatial_size * channels);
		memcpy(in_data[n].data(), data.data() + offset, sizeof(float)*spatial_size*channels);
		std::fill(out_data[n].begin(), out_data[n].end(), 0.);
	}

	std::vector<tiny_dnn::tensor_t*> in(1), out(1);
	in[0] = &in_data;
	out[0] = &out_data;

	tiny_dnn::batch_normalization_layer bn(spatial_size, channels);
	bn.forward_propagation(in, out);

	fprintf(stdout, "tiny_dnn result:\n");
	for (int n = 0; n < number; ++n) {
		for (int s = 0; s < spatial_size * channels; ++s)
			fprintf(stdout, "%f  ", out_data[n][s]);
		fprintf(stdout, "\n");
	}

	return 0;
}


///////////////////////////////////////////////////////////////////////////////
// Blog: http://blog.csdn.net/fengbingchun/article/details/53453931
static void construct_net(tiny_dnn::network<tiny_dnn::sequential>& nn)
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

	// by default will use backend_t::tiny_dnn unless you compiled
	// with -DUSE_AVX=ON and your device supports AVX intrinsics
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

	// construct nets: C: convolution; S: sub-sampling; F: fully connected
	nn << tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(32, 32, 5, 1, 6,  // C1, 1@32x32-in, 6@28x28-out
		tiny_dnn::padding::valid, true, 1, 1, backend_type)
		<< tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
		<< tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(14, 14, 5, 6, 16, // C3, 6@14x14-in, 16@10x10-out
		connection_table(tbl, 6, 16),
		tiny_dnn::padding::valid, true, 1, 1, backend_type)
		<< tiny_dnn::average_pooling_layer<tiny_dnn::activation::tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
		<< tiny_dnn::convolutional_layer<tiny_dnn::activation::tan_h>(5, 5, 5, 16, 120, // C5, 16@5x5-in, 120@1x1-out
		tiny_dnn::padding::valid, true, 1, 1, backend_type)
		<< tiny_dnn::fully_connected_layer<tiny_dnn::activation::tan_h>(120, 10,        // F6, 120-in, 10-out
		true, backend_type);
}

static void train_lenet(const std::string& data_dir_path)
{
	// specify loss-function and learning strategy
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adagrad optimizer;

	construct_net(nn);

	std::cout << "load models..." << std::endl;

	// load MNIST dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	tiny_dnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte", &train_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
	tiny_dnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte", &test_labels);
	tiny_dnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);

	std::cout << "start training" << std::endl;

	tiny_dnn::progress_display disp(static_cast<unsigned long>(train_images.size()));
	tiny_dnn::timer t;
	int minibatch_size = 10;
	int num_epochs = 30;

	optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

	// create callback
	auto on_enumerate_epoch = [&](){
		std::cout << t.elapsed() << "s elapsed." << std::endl;
		tiny_dnn::result res = nn.test(test_images, test_labels);
		std::cout << res.num_success << "/" << res.num_total << std::endl;

		disp.restart(static_cast<unsigned long>(train_images.size()));
		t.restart();
	};

	auto on_enumerate_minibatch = [&](){
		disp += minibatch_size;
	};

	// training
	nn.train<tiny_dnn::mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs, on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);

	// save network model & trained weights
	nn.save(data_dir_path + "/LeNet-model");
}

// rescale output to 0-100
template <typename Activation>
static double rescale(double x)
{
	Activation a;
	return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

static void convert_image(const std::string& imagefilename, double minv, double maxv, int w, int h, tiny_dnn::vec_t& data)
{
	tiny_dnn::image<> img(imagefilename, tiny_dnn::image_type::grayscale);
	tiny_dnn::image<> resized = resize_image(img, w, h);

	// mnist dataset is "white on black", so negate required
	std::transform(resized.begin(), resized.end(), std::back_inserter(data),
		[=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

int test_dnn_mnist_train()
{
#ifdef _MSC_VER
	std::string data_dir_path = "../../../data/database/MNIST";
#else
	std::string data_dir_path = "data/database/MNIST";
#endif
	train_lenet(data_dir_path);

	return 0;
}

int test_dnn_mnist_predict()
{
#ifdef _MSC_VER
	std::string model { "../../../data/database/MNIST/LeNet-model" };
	std::string image_path { "../../../data/images/digit/handwriting_2/"};
#else
	std::string model { "data/database/MNIST/LeNet-model" };
	std::string image_path { "data/images/digit/handwriting_2/"};
#endif
	int target[10] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

	tiny_dnn::network<tiny_dnn::sequential> nn;
	nn.load(model);

	for (int i = 0; i < 10; i++) {
		std::string str = std::to_string(i);
		str += ".png";
		str = image_path + str;

		// convert imagefile to vec_t
		tiny_dnn::vec_t data;
		convert_image(str, -1.0, 1.0, 32, 32, data);

		// recognize
		auto res = nn.predict(data);
		std::vector<std::pair<double, int> > scores;

		// sort & print top-3
		for (int j = 0; j < 10; j++)
			scores.emplace_back(rescale<tiny_dnn::tan_h>(res[j]), j);

		std::sort(scores.begin(), scores.end(), std::greater<std::pair<double, int>>());

		for (int j = 0; j < 3; j++)
			fprintf(stdout, "%d: %f;  ", scores[j].second, scores[j].first);
		fprintf(stderr, "\n");

		// save outputs of each layer
		for (size_t j = 0; j < nn.depth(); j++) {
			auto out_img = nn[j]->output_to_image();
			auto filename = image_path + std::to_string(i) + "_layer_" + std::to_string(j) + ".png";
			out_img.save(filename);
		}

		// save filter shape of first convolutional layer
		auto weight = nn.at<tiny_dnn::convolutional_layer<tiny_dnn::tan_h>>(0).weight_to_image();
		auto filename = image_path + std::to_string(i) + "_weights.png";
		weight.save(filename);

		fprintf(stdout, "the actual digit is: %d, correct digit is: %d \n", scores[0].second, target[i]);
	}

	return 0;
}

template <typename N>
static void construct_net(N& nn)
{
	typedef tiny_dnn::convolutional_layer<tiny_dnn::activation::identity> conv;
	typedef tiny_dnn::max_pooling_layer<tiny_dnn::activation::relu> pool;

	const tiny_dnn::serial_size_t n_fmaps = 32;   ///< number of feature maps for upper layer
	const tiny_dnn::serial_size_t n_fmaps2 = 64;  ///< number of feature maps for lower layer
	const tiny_dnn::serial_size_t n_fc = 64;      ///< number of hidden units in fully-connected layer

	nn << conv(32, 32, 5, 3, n_fmaps, tiny_dnn::padding::same)
		<< pool(32, 32, n_fmaps, 2)
		<< conv(16, 16, 5, n_fmaps, n_fmaps, tiny_dnn::padding::same)
		<< pool(16, 16, n_fmaps, 2)
		<< conv(8, 8, 5, n_fmaps, n_fmaps2, tiny_dnn::padding::same)
		<< pool(8, 8, n_fmaps2, 2)
		<< tiny_dnn::fully_connected_layer<tiny_dnn::activation::identity>(4 * 4 * n_fmaps2, n_fc)
		<< tiny_dnn::fully_connected_layer<tiny_dnn::softmax>(n_fc, 10);
}

int test_dnn_cifar10_train()
{
#ifdef _MSC_VER
	std::string data_dir_path = "../../../data/database/CIFAR/CIFAR-10/";
#else
	std::string data_dir_path = "data/database/CIFAR/CIFAR-10/";
#endif
	double learning_rate = 0.01;

	// specify loss-function and learning strategy
	tiny_dnn::network<tiny_dnn::sequential> nn;
	tiny_dnn::adam optimizer;

	construct_net(nn);

	fprintf(stdout, "learning rate: %f\n", learning_rate);
	fprintf(stdout, "load models...\n");

	// load cifar dataset
	std::vector<tiny_dnn::label_t> train_labels, test_labels;
	std::vector<tiny_dnn::vec_t> train_images, test_images;

	for (int i = 1; i <= 5; i++) {
		parse_cifar10(data_dir_path + "/data_batch_" + std::to_string(i) + ".bin",
			&train_images, &train_labels, -1.0, 1.0, 0, 0);
	}

	parse_cifar10(data_dir_path + "/test_batch.bin",
		&test_images, &test_labels, -1.0, 1.0, 0, 0);

	fprintf(stderr, "test images count: %d, labels count: %d\n", test_images.size(), test_labels.size());
	fprintf(stdout, "start learning\n");

	tiny_dnn::progress_display disp(train_images.size());
	tiny_dnn::timer t;
	const int n_minibatch = 10;     ///< minibatch size
	const int n_train_epochs = 30;  ///< training duration

	optimizer.alpha *= static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);

	// create callback
	auto on_enumerate_epoch = [&]() {
		fprintf(stdout, "%f s elapsed\n", t.elapsed());
		tiny_dnn::result res = nn.test(test_images, test_labels);
		fprintf(stdout, "%d / %d\n", res.num_success, res.num_total);

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&]() {
		disp += n_minibatch;
	};

	// training
	nn.train<tiny_dnn::cross_entropy>(optimizer, train_images, train_labels,
		n_minibatch, n_train_epochs, on_enumerate_minibatch,
		on_enumerate_epoch);

	fprintf(stdout, "end training.\n");

	// test and show results
	nn.test(test_images, test_labels).print_detail(std::cout);

	// save networks
	std::ofstream ofs(data_dir_path + "cifar-weights");
	ofs << nn;

	return 0;
}

int test_dnn_cifar10_predict()
{
	return 0;
}
