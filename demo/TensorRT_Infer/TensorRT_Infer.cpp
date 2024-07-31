#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <memory>
#include <chrono>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include "yolo.hpp"

// Blog: https://blog.csdn.net/fengbingchun/article/details/140826627

#ifdef _MSC_VER

namespace {

constexpr float confidence_threshold{ 0.45f }; // confidence threshold
constexpr float nms_threshold{ 0.50f }; // nms threshold
constexpr char* engine_file{ "../../../data/best.transd.fp32.engine" };
constexpr char* images_dir{ "../../../data/images/predict" };
constexpr char* result_dir{ "../../../data/result" };
constexpr char* classes_file{ "../../../data/images/labels.txt" };

std::vector<std::string> parse_classes_file(const char* name)
{
	std::vector<std::string> classes;

	std::ifstream file(name);
	if (!file.is_open()) {
		std::cerr << "Error: fail to open classes file: " << name << std::endl;
		return classes;
	}

	std::string line;
	while (std::getline(file, line)) {
		auto pos = line.find_first_of(" ");
		classes.emplace_back(line.substr(0, pos));
	}

	file.close();
	return classes;
}

auto get_dir_images(const char* name)
{
	std::map<std::string, std::string> images; // image name, image path + image name

	for (auto const& dir_entry : std::filesystem::directory_iterator(name)) {
		if (dir_entry.is_regular_file())
			images[dir_entry.path().filename().string()] = dir_entry.path().string();
	}

	return images;
}

auto get_random_color(int labels_number)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(100, 255);

	std::vector<cv::Scalar> colors;

	for (auto i = 0; i < labels_number; ++i) {
		colors.emplace_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));
	}

	return colors;
}

} // namespace

int main()
{
    namespace fs = std::filesystem;

    if (!fs::exists(result_dir)) {
        fs::create_directories(result_dir);
    }

    auto classes = parse_classes_file(classes_file);
    if (classes.size() == 0) {
        std::cerr << "Error: fail to parse classes file: " << classes_file << std::endl;
        return -1;
    }

	std::cout << "classes: ";
	for (const auto& val : classes) {
		std::cout << val << " ";
	}
	std::cout << std::endl;

	auto colors = get_random_color(classes.size());

	auto model = yolo::load(engine_file, yolo::Type::V8, confidence_threshold, nms_threshold);

	for (auto i = 0; i < 10; ++i) {
		std::cout << "i: " << i << std::endl;
		for (const auto& [key, val] : get_dir_images(images_dir)) {
			cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
			if (frame.empty()) {
				std::cerr << "Warning: unable to load image: " << val << std::endl;
				continue;
			}

			auto tstart = std::chrono::high_resolution_clock::now();
			auto objs = model->forward(yolo::Image(frame.data, frame.cols, frame.rows));
			auto tend = std::chrono::high_resolution_clock::now();
			std::cout << "elapsed millisenconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms" << std::endl;

			for (const auto& obj : objs) {
				cv::rectangle(frame, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), colors[obj.class_label], 2);

				std::string class_string = classes[obj.class_label] + ' ' + std::to_string(obj.confidence).substr(0, 4);
				cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
				cv::Rect text_box(obj.left, obj.top - 40, text_size.width + 10, text_size.height + 20);

				cv::rectangle(frame, text_box, colors[obj.class_label], cv::FILLED);
				cv::putText(frame, class_string, cv::Point(obj.left + 5, obj.top - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
			}

			std::string path(result_dir);
			path += "/" + key;
			cv::imwrite(path, frame);
		}
	}

	std::cout << "test finish" << std::endl;
    return 0;
}

#endif
