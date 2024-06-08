#include "yolo.hpp"

#ifdef _MSC_VER

#include <iostream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <map>
#include <memory>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <onnxruntime_cxx_api.h>

namespace {

constexpr bool cuda_enabled{ false };
constexpr int input_size[2]{ 640, 640 }; // {height,width}, input shape (1, 3, 640, 640) BCHW and output shape(s): detect:(1,6,8400); segment:(1,38,8400),(1,32,160,160)
constexpr float confidence_threshold{ 0.45 }; // confidence threshold
constexpr float iou_threshold{ 0.50 }; // iou threshold
constexpr float mask_threshold{ 0.50 }; // segment mask threshold

#ifdef _MSC_VER
constexpr char* onnx_file{ "../../../data/best.onnx" };
constexpr char* torchscript_file{ "../../../data/best.torchscript" };
constexpr char* images_dir{ "../../../data/images/predict" };
constexpr char* result_dir{ "../../../data/result" };
constexpr char* classes_file{ "../../../data/images/labels.txt" };
#else
constexpr char* onnx_file{ "data/best.onnx" };
constexpr char* torchscript_file{ "data/best.torchscript" };
constexpr char* images_dir{ "data/images/predict" };
constexpr char* result_dir{ "data/result" };
constexpr char* classes_file{ "data/images/labels.txt" };
#endif

cv::Mat modify_image_size(const cv::Mat& img)
{
	auto max = std::max(img.rows, img.cols);
	cv::Mat ret = cv::Mat::zeros(max, max, CV_8UC3);
	img.copyTo(ret(cv::Rect(0, 0, img.cols, img.rows)));

	return ret;
}

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

void draw_boxes(const std::vector<std::string>& classes, const std::vector<int>& ids, const std::vector<float>& confidences,
	const std::vector<cv::Rect>& boxes, const std::string& name, cv::Mat& frame)
{
	if (ids.size() != confidences.size() || ids.size() != boxes.size() || confidences.size() != boxes.size()) {
		std::cerr << "Error: their lengths are inconsistent: " << ids.size() << ", " << confidences.size() << ", " << boxes.size() << std::endl;
		return;
	}

	std::cout << "image name: " << name << ", number of detections: " << ids.size() << std::endl;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(100, 255);

	for (auto i = 0; i < ids.size(); ++i) {
		auto color = cv::Scalar(dis(gen), dis(gen), dis(gen));
		cv::rectangle(frame, boxes[i], color, 2);

		std::string class_string = classes[ids[i]] + ' ' + std::to_string(confidences[i]).substr(0, 4);
		cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
		cv::Rect text_box(boxes[i].x, boxes[i].y - 40, text_size.width + 10, text_size.height + 20);

		cv::rectangle(frame, text_box, color, cv::FILLED);
		cv::putText(frame, class_string, cv::Point(boxes[i].x + 5, boxes[i].y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
	}

	//cv::imshow("Inference", frame);
	//cv::waitKey(-1);

	std::string path(result_dir);
	path += "/" + name;
	cv::imwrite(path, frame);
}

float letter_box(const cv::Mat& src, cv::Mat& dst, const std::vector<int>& imgsz)
{
	if (src.cols == imgsz[1] && src.rows == imgsz[0]) {
		if (src.data != dst.data);
			dst = src.clone();
		return 1.;
	}

	auto resize_scale = std::min(imgsz[0] * 1. / src.rows, imgsz[1] * 1. / src.cols);
	int new_shape_w = std::round(src.cols * resize_scale);
	int new_shape_h = std::round(src.rows * resize_scale);
	float padw = (imgsz[1] - new_shape_w) / 2.;
	float padh = (imgsz[0] - new_shape_h) / 2.;

	int top = std::round(padh - 0.1);
	int bottom = std::round(padh + 0.1);
	int left = std::round(padw - 0.1);
	int right = std::round(padw + 0.1);

	cv::resize(src, dst, cv::Size(new_shape_w, new_shape_h), 0, 0, cv::INTER_AREA);
	cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114.));

	return resize_scale;
}

torch::Tensor xywh2xyxy(const torch::Tensor& x)
{
	auto y = torch::empty_like(x);
	auto dw = x.index({ "...", 2 }).div(2);
	auto dh = x.index({ "...", 3 }).div(2);
	y.index_put_({ "...", 0 }, x.index({ "...", 0 }) - dw);
	y.index_put_({ "...", 1 }, x.index({ "...", 1 }) - dh);
	y.index_put_({ "...", 2 }, x.index({ "...", 0 }) + dw);
	y.index_put_({ "...", 3 }, x.index({ "...", 1 }) + dh);

	return y;
}

// reference: https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold)
{
	if (bboxes.numel() == 0)
		return torch::empty({ 0 }, bboxes.options().dtype(torch::kLong));

	auto x1_t = bboxes.select(1, 0).contiguous();
	auto y1_t = bboxes.select(1, 1).contiguous();
	auto x2_t = bboxes.select(1, 2).contiguous();
	auto y2_t = bboxes.select(1, 3).contiguous();

	torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

	auto order_t = std::get<1>(scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

	auto ndets = bboxes.size(0);
	torch::Tensor suppressed_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kByte));
	torch::Tensor keep_t = torch::zeros({ ndets }, bboxes.options().dtype(torch::kLong));

	auto suppressed = suppressed_t.data_ptr<uint8_t>();
	auto keep = keep_t.data_ptr<int64_t>();
	auto order = order_t.data_ptr<int64_t>();
	auto x1 = x1_t.data_ptr<float>();
	auto y1 = y1_t.data_ptr<float>();
	auto x2 = x2_t.data_ptr<float>();
	auto y2 = y2_t.data_ptr<float>();
	auto areas = areas_t.data_ptr<float>();

	int64_t num_to_keep = 0;

	for (int64_t _i = 0; _i < ndets; _i++) {
		auto i = order[_i];
		if (suppressed[i] == 1)
			continue;
		keep[num_to_keep++] = i;
		auto ix1 = x1[i];
		auto iy1 = y1[i];
		auto ix2 = x2[i];
		auto iy2 = y2[i];
		auto iarea = areas[i];

		for (int64_t _j = _i + 1; _j < ndets; _j++) {
			auto j = order[_j];
			if (suppressed[j] == 1)
				continue;
			auto xx1 = std::max(ix1, x1[j]);
			auto yy1 = std::max(iy1, y1[j]);
			auto xx2 = std::min(ix2, x2[j]);
			auto yy2 = std::min(iy2, y2[j]);

			auto w = std::max(static_cast<float>(0), xx2 - xx1);
			auto h = std::max(static_cast<float>(0), yy2 - yy1);
			auto inter = w * h;
			auto ovr = inter / (iarea + areas[j] - inter);
			if (ovr > iou_threshold)
				suppressed[j] = 1;
		}
	}

	return keep_t.narrow(0, 0, num_to_keep);
}

torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300)
{
	using torch::indexing::Slice;
	using torch::indexing::None;

	auto bs = prediction.size(0);
	auto nc = prediction.size(1) - 4;
	auto nm = prediction.size(1) - nc - 4;
	auto mi = 4 + nc;
	auto xc = prediction.index({ Slice(), Slice(4, mi) }).amax(1) > conf_thres;

	prediction = prediction.transpose(-1, -2);
	prediction.index_put_({ "...", Slice({None, 4}) }, xywh2xyxy(prediction.index({ "...", Slice(None, 4) })));

	std::vector<torch::Tensor> output;
	for (int i = 0; i < bs; i++) {
		output.push_back(torch::zeros({ 0, 6 + nm }, prediction.device()));
	}

	for (int xi = 0; xi < prediction.size(0); xi++) {
		auto x = prediction[xi];
		x = x.index({ xc[xi] });
		auto x_split = x.split({ 4, nc, nm }, 1);
		auto box = x_split[0], cls = x_split[1], mask = x_split[2];
		auto [conf, j] = cls.max(1, true);
		x = torch::cat({ box, conf, j.toType(torch::kFloat), mask }, 1);
		x = x.index({ conf.view(-1) > conf_thres });
		int n = x.size(0);
		if (!n) { continue; }

		// NMS
		auto c = x.index({ Slice(), Slice{5, 6} }) * 7680;
		auto boxes = x.index({ Slice(), Slice(None, 4) }) + c;
		auto scores = x.index({ Slice(), 4 });
		auto i = nms(boxes, scores, iou_thres);
		i = i.index({ Slice(None, max_det) });
		output[xi] = x.index({ i });
	}

	return torch::stack(output);
}

std::wstring ctow(const char* str)
{
	//std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(std::string); // std::string -> std::wstring
	constexpr size_t len{ 128 };
	wchar_t wch[len];
	swprintf(wch, len, L"%hs", str);

	return std::wstring(wch);
}

float image_preprocess(const cv::Mat& src, cv::Mat& dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);

	float scalex = src.cols * 1.f / input_size[1];
	float scaley = src.rows * 1.f / input_size[0];

	if (scalex > scaley)
		cv::resize(dst, dst, cv::Size(input_size[1], static_cast<int>(src.rows / scalex)));
	else
		cv::resize(dst, dst, cv::Size(static_cast<int>(src.cols / scaley), input_size[0]));

	cv::Mat tmp = cv::Mat::zeros(input_size[0], input_size[1], CV_8UC3);
	dst.copyTo(tmp(cv::Rect(0, 0, dst.cols, dst.rows)));
	dst = tmp;

	return (scalex > scaley) ? scalex : scaley;
}

template<typename T>
void image_to_blob(const cv::Mat& src, T* blob)
{
	for (auto c = 0; c < 3; ++c) {
		for (auto h = 0; h < src.rows; ++h) {
			for (auto w = 0; w < src.cols; ++w) {
				blob[c * src.rows * src.cols + h * src.cols + w] = (src.at<cv::Vec3b>(h, w)[c]) / 255.f;
			}
		}
	}
}

void post_process(const float* data, int rows, int stride, float xfactor, float yfactor, const std::vector<std::string>& classes,
	const std::string& name, cv::Mat& frame)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (auto i = 0; i < rows; ++i) {
		const float* classes_scores = data + 4;

		cv::Mat scores(1, classes.size(), CV_32FC1, (float*)classes_scores);
		cv::Point class_id;
		double max_class_score;

		cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

		if (max_class_score > confidence_threshold) {
			confidences.push_back(max_class_score);
			class_ids.push_back(class_id.x);

			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = int((x - 0.5 * w) * xfactor);
			int top = int((y - 0.5 * h) * yfactor);

			int width = int(w * xfactor);
			int height = int(h * yfactor);

			boxes.push_back(cv::Rect(left, top, width, height));
		}

		data += stride;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold, nms_result);

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> rects;
	for (size_t i = 0; i < nms_result.size(); ++i) {
		ids.emplace_back(class_ids[nms_result[i]]);
		confs.emplace_back(confidences[nms_result[i]]);
		rects.emplace_back(boxes[nms_result[i]]);
	}
	draw_boxes(classes, ids, confs, rects, name, frame);
}

void get_masks(const cv::Mat& features, const cv::Mat& proto, const std::vector<int>& output1_sizes, const cv::Mat& frame, const cv::Rect box, cv::Mat& mk)
{
	const cv::Size shape_src(frame.cols, frame.rows), shape_input(input_size[1], input_size[0]), shape_mask(output1_sizes[3], output1_sizes[2]);

	cv::Mat res = (features * proto).t();
	res = res.reshape(1, { shape_mask.height, shape_mask.width });
	// apply sigmoid to the mask
	cv::exp(-res, res);
	res = 1.0 / (1.0 + res);
	cv::resize(res, res, shape_input);

	float scalex = shape_src.width * 1.0 / shape_input.width;
	float scaley = shape_src.height * 1.0 / shape_input.height;
	cv::Mat tmp;
	if (scalex > scaley)
		cv::resize(res, tmp, cv::Size(shape_src.width, static_cast<int>(shape_input.height * scalex)));
	else
		cv::resize(res, tmp, cv::Size(static_cast<int>(shape_input.width * scaley), shape_src.height));

	cv::Mat dst = tmp(cv::Rect(0, 0, shape_src.width, shape_src.height));
	mk = dst(box) > mask_threshold;
}

void draw_boxes_mask(const std::vector<std::string>& classes, const std::vector<int>& ids, const std::vector<float>& confidences,
	const std::vector<cv::Rect>& boxes, const std::vector<cv::Mat>& masks, const std::string& name, cv::Mat& frame)
{
	std::cout << "image name: " << name << ", number of detections: " << ids.size() << std::endl;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dis(100, 255);
	cv::Mat mk = frame.clone();

	std::vector<cv::Scalar> colors;
	for (auto i = 0; i < classes.size(); ++i)
		colors.emplace_back(cv::Scalar(dis(gen), dis(gen), dis(gen)));

	for (auto i = 0; i < ids.size(); ++i) {
		cv::rectangle(frame, boxes[i], colors[ids[i]], 2);

		std::string class_string = classes[ids[i]] + ' ' + std::to_string(confidences[i]).substr(0, 4);
		cv::Size text_size = cv::getTextSize(class_string, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
		cv::Rect text_box(boxes[i].x, boxes[i].y - 40, text_size.width + 10, text_size.height + 20);

		cv::rectangle(frame, text_box, colors[ids[i]], cv::FILLED);
		cv::putText(frame, class_string, cv::Point(boxes[i].x + 5, boxes[i].y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

		mk(boxes[i]).setTo(colors[ids[i]], masks[i]);
	}

	cv::addWeighted(frame, 0.5, mk, 0.5, 0, frame);

	//cv::imshow("Inference", frame);
	//cv::waitKey(-1);

	std::string path(result_dir);
	cv::imwrite(path + "/" + name, frame);
}

void post_process_mask(const cv::Mat& output0, const cv::Mat& output1, const std::vector<int>& output1_sizes, const std::vector<std::string>& classes, const std::string& name, cv::Mat& frame)
{
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<std::vector<float>> masks;

	float scalex = frame.cols * 1.f / input_size[1]; // note: image_preprocess function
	float scaley = frame.rows * 1.f / input_size[0];
	auto scale = (scalex > scaley) ? scalex : scaley;

	const float* data = (float*)output0.data;
	for (auto i = 0; i < output0.rows; ++i) {
		cv::Mat scores(1, classes.size(), CV_32FC1, (float*)data + 4);
		cv::Point class_id;
		double max_class_score;

		cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

		if (max_class_score > confidence_threshold) {
			confidences.emplace_back(max_class_score);
			class_ids.emplace_back(class_id.x);
			masks.emplace_back(std::vector<float>(data + 4 + classes.size(), data + output0.cols)); // 32

			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = std::max(0, std::min(int((x - 0.5 * w) * scale), frame.cols));
			int top = std::max(0, std::min(int((y - 0.5 * h) * scale), frame.rows));
			int width = std::max(0, std::min(int(w * scale), frame.cols - left));
			int height = std::max(0, std::min(int(h * scale), frame.rows - top));
			boxes.emplace_back(cv::Rect(left, top, width, height));
		}

		data += output0.cols;
	}

	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold, nms_result);

	cv::Mat proto = output1.reshape(0, { output1_sizes[1], output1_sizes[2] * output1_sizes[3] });

	std::vector<int> ids;
	std::vector<float> confs;
	std::vector<cv::Rect> rects;
	std::vector<cv::Mat> mks;
	for (size_t i = 0; i < nms_result.size(); ++i) {
		auto index = nms_result[i];
		ids.emplace_back(class_ids[index]);
		confs.emplace_back(confidences[index]);
		boxes[index] = boxes[index] & cv::Rect(0, 0, frame.cols, frame.rows);

		cv::Mat mk;
		get_masks(cv::Mat(masks[index]).t(), proto, output1_sizes, frame, boxes[index], mk);
		mks.emplace_back(mk);
		rects.emplace_back(boxes[index]);
	}

	draw_boxes_mask(classes, ids, confs, rects, mks, name, frame);
}

} // namespace

/////////////////////////////////////////////////////////////////
// Blog: https://blog.csdn.net/fengbingchun/article/details/139203567
int test_yolov8_detect_opencv()
{
	// reference: ultralytics/examples/YOLOv8-CPP-Inference
	namespace fs = std::filesystem;

	//std::cout << "opencv bulild info: " << cv::getBuildInformation() << std::endl;

	auto net = cv::dnn::readNetFromONNX(onnx_file);
	if (net.empty()) {
		std::cerr << "Error: there are no layers in the network: " << onnx_file << std::endl;
		return -1;
	}

	if (cuda_enabled) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	} else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

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

	for (const auto& [key, val] : get_dir_images(images_dir)) {
		cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
		if (frame.empty()) {
			std::cerr << "Warning: unable to load image: " << val << std::endl;
			continue;
		}

		auto tstart = std::chrono::high_resolution_clock::now();
		cv::Mat bgr = modify_image_size(frame);

		cv::Mat blob;
		cv::dnn::blobFromImage(bgr, blob, 1.0 / 255.0, cv::Size(input_size[1], input_size[0]), cv::Scalar(), true, false);
		net.setInput(blob);

		std::vector<cv::Mat> outputs;
		net.forward(outputs, net.getUnconnectedOutLayersNames());

		int rows = outputs[0].size[1];
		int dimensions = outputs[0].size[2];

		// yolov5 has an output of shape (batchSize, 25200, num classes+4+1) (Num classes + box[x,y,w,h] + confidence[c])
		// yolov8 has an output of shape (batchSize, num classes + 4,  8400) (Num classes + box[x,y,w,h])
		if (dimensions > rows) { // Check if the shape[2] is more than shape[1] (yolov8)
			rows = outputs[0].size[2];
			dimensions = outputs[0].size[1];

			outputs[0] = outputs[0].reshape(1, dimensions);
			cv::transpose(outputs[0], outputs[0]);
		}

		float* data = (float*)outputs[0].data;
		float x_factor = bgr.cols * 1.f / input_size[1];
		float y_factor = bgr.rows * 1.f / input_size[0];

		auto tend = std::chrono::high_resolution_clock::now();
		std::cout << "elapsed millisenconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms" << std::endl;

		post_process(data, rows, dimensions, x_factor, y_factor, classes, key, frame);
	}

	return 0;
}

/////////////////////////////////////////////////////////////////
// Blog: https://blog.csdn.net/fengbingchun/article/details/139217313
int test_yolov8_detect_libtorch()
{
	// reference: ultralytics/examples/YOLOv8-LibTorch-CPP-Inference
	if (auto flag = torch::cuda::is_available(); flag == true)
		std::cout << "cuda is available" << std::endl;
	else
		std::cout << "cuda is not available" << std::endl;

	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

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

	try {
		// load model
		torch::jit::script::Module model;
		if (torch::cuda::is_available() == true)
			model = torch::jit::load(torchscript_file, torch::kCUDA);
		else
			model = torch::jit::load(torchscript_file, torch::kCPU);
		model.eval();
		// note: cpu is normal; gpu is abnormal: the model may not be fully placed on the gpu 
		// model = torch::jit::load(file); model.to(torch::kCUDA) ==> model = torch::jit::load(file, torch::kCUDA)
		// model.to(device, torch::kFloat32);

		for (const auto& [key, val] : get_dir_images(images_dir)) {
			// load image and preprocess
			cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
			if (frame.empty()) {
				std::cerr << "Warning: unable to load image: " << val << std::endl;
				continue;
			}

			cv::Mat bgr;
			letter_box(frame, bgr, {input_size[0], input_size[1]});

			torch::Tensor tensor = torch::from_blob(bgr.data, { bgr.rows, bgr.cols, 3 }, torch::kByte).to(device);
			tensor = tensor.toType(torch::kFloat32).div(255);
			tensor = tensor.permute({ 2, 0, 1 });
			tensor = tensor.unsqueeze(0);
			std::vector<torch::jit::IValue> inputs{ tensor };

			// inference
			torch::Tensor output = model.forward(inputs).toTensor().cpu();

			// NMS
			auto keep = non_max_suppression(output, 0.1f, 0.1f, 300)[0];

			std::vector<int> ids;
			std::vector<float> confidences;
			std::vector<cv::Rect> boxes;
			for (auto i = 0; i < keep.size(0); ++i) {
				int x1 = keep[i][0].item().toFloat();
				int y1 = keep[i][1].item().toFloat();
				int x2 = keep[i][2].item().toFloat();
				int y2 = keep[i][3].item().toFloat();
				boxes.emplace_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));

				confidences.emplace_back(keep[i][4].item().toFloat());
				ids.emplace_back(keep[i][5].item().toInt());
			}

			draw_boxes(classes, ids, confidences, boxes, key, bgr);
		}
	} catch (const c10::Error& e) {
		std::cerr << "Error: " << e.msg() << std::endl;
		return -1;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////
// Blog: https://blog.csdn.net/fengbingchun/article/details/139371680
int test_yolov8_detect_onnxruntime()
{
	// reference: ultralytics/examples/YOLOv8-ONNXRuntime-CPP
	try {
		Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
		Ort::SessionOptions session_option;

		if (cuda_enabled) {
			OrtCUDAProviderOptions cuda_option;
			cuda_option.device_id = 0;
			session_option.AppendExecutionProvider_CUDA(cuda_option);
		}

		session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		session_option.SetIntraOpNumThreads(1);
		session_option.SetLogSeverityLevel(3);

		Ort::Session session(env, ctow(onnx_file).c_str(), session_option);
		Ort::AllocatorWithDefaultOptions allocator;
		std::vector<const char*> input_node_names, output_node_names;
		std::vector<std::string> input_node_names_, output_node_names_;

		for (auto i = 0; i < session.GetInputCount(); ++i) {
			Ort::AllocatedStringPtr input_node_name = session.GetInputNameAllocated(i, allocator);
			input_node_names_.emplace_back(input_node_name.get());
		}

		for (auto i = 0; i < session.GetOutputCount(); ++i) {
			Ort::AllocatedStringPtr output_node_name = session.GetOutputNameAllocated(i, allocator);
			output_node_names_.emplace_back(output_node_name.get());
		}

		for (auto i = 0; i < input_node_names_.size(); ++i)
			input_node_names.emplace_back(input_node_names_[i].c_str());
		for (auto i = 0; i < output_node_names_.size(); ++i)
			output_node_names.emplace_back(output_node_names_[i].c_str());

		Ort::RunOptions options(nullptr);
		std::unique_ptr<float[]> blob(new float[input_size[0] * input_size[1] * 3]);
		std::vector<int64_t> input_node_dims{ 1, 3, input_size[1], input_size[0] };

		auto classes = parse_classes_file(classes_file);
		if (classes.size() == 0) {
			std::cerr << "Error: fail to parse classes file: " << classes_file << std::endl;
			return -1;
		}

		for (const auto& [key, val] : get_dir_images(images_dir)) {
			cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
			if (frame.empty()) {
				std::cerr << "Warning: unable to load image: " << val << std::endl;
				continue;
			}

			auto tstart = std::chrono::high_resolution_clock::now();
			cv::Mat rgb;
			auto resize_scales = image_preprocess(frame, rgb);
			image_to_blob(rgb, blob.get());
			Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
				Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob.get(), 3 * input_size[1] * input_size[0], input_node_dims.data(), input_node_dims.size());
			auto output_tensors = session.Run(options, input_node_names.data(), &input_tensor, input_node_names.size(), output_node_names.data(), output_node_names.size());

			Ort::TypeInfo type_info = output_tensors.front().GetTypeInfo();
			auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
			std::vector<int64_t> output_node_dims = tensor_info.GetShape();
			auto output = output_tensors.front().GetTensorMutableData<float>();
			int stride_num = output_node_dims[1];
			int signal_result_num = output_node_dims[2];
			cv::Mat raw_data = cv::Mat(stride_num, signal_result_num, CV_32F, output);
			raw_data = raw_data.t();
			const float* data = (float*)raw_data.data;

			auto tend = std::chrono::high_resolution_clock::now();
			std::cout << "elapsed millisenconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms" << std::endl;

			post_process(data, signal_result_num, stride_num, resize_scales, resize_scales, classes, key, frame);
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////
// Blog: https://blog.csdn.net/fengbingchun/article/details/139551555
int test_yolov8_segment_onnxruntime()
{
	try {
		Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
		Ort::SessionOptions session_option;

		if (cuda_enabled) {
			OrtCUDAProviderOptions cuda_option;
			cuda_option.device_id = 0;
			session_option.AppendExecutionProvider_CUDA(cuda_option);
		}

		session_option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		session_option.SetIntraOpNumThreads(1);
		session_option.SetLogSeverityLevel(3);

		Ort::Session session(env, ctow(onnx_file).c_str(), session_option);
		Ort::AllocatorWithDefaultOptions allocator;
		std::vector<const char*> input_node_names, output_node_names;
		std::vector<std::string> input_node_names_, output_node_names_;

		for (auto i = 0; i < session.GetInputCount(); ++i) {
			Ort::AllocatedStringPtr input_node_name = session.GetInputNameAllocated(i, allocator);
			input_node_names_.emplace_back(input_node_name.get());
		}

		for (auto i = 0; i < session.GetOutputCount(); ++i) {
			Ort::AllocatedStringPtr output_node_name = session.GetOutputNameAllocated(i, allocator);
			output_node_names_.emplace_back(output_node_name.get());
		}

		for (auto i = 0; i < input_node_names_.size(); ++i)
			input_node_names.emplace_back(input_node_names_[i].c_str());
		for (auto i = 0; i < output_node_names_.size(); ++i)
			output_node_names.emplace_back(output_node_names_[i].c_str());

		std::unique_ptr<float[]> blob(new float[input_size[0] * input_size[1] * 3]);
		std::vector<int64_t> input_node_dims{ 1, 3, input_size[1], input_size[0] };

		auto classes = parse_classes_file(classes_file);
		if (classes.size() == 0) {
			std::cerr << "Error: fail to parse classes file: " << classes_file << std::endl;
			return -1;
		}

		if (!std::filesystem::exists(result_dir)) {
			std::filesystem::create_directories(result_dir);
		}

		for (const auto& [key, val] : get_dir_images(images_dir)) {
			cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
			if (frame.empty()) {
				std::cerr << "Warning: unable to load image: " << val << std::endl;
				continue;
			}

			auto tstart = std::chrono::high_resolution_clock::now();
			cv::Mat rgb;
			image_preprocess(frame, rgb);
			image_to_blob(rgb, blob.get());
			Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
				Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob.get(), 3 * input_size[1] * input_size[0], input_node_dims.data(), input_node_dims.size());
			auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, input_node_names.size(), output_node_names.data(), output_node_names.size());
			if (output_tensors.size() != 2) {
				std::cerr << "Error: output must have 2 layers: " << output_tensors.size() << std::endl;
				return -1;
			}

			// output0
			std::vector<int64_t> output0_node_dims = output_tensors[0].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
			auto output0 = output_tensors[0].GetTensorMutableData<float>();
			cv::Mat data0 = cv::Mat(output0_node_dims[1], output0_node_dims[2], CV_32F, output0);
			data0 = data0.t();

			// output1
			std::vector<int64_t> output1_node_dims = output_tensors[1].GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
			auto output1 = output_tensors[1].GetTensorMutableData<float>();
			std::vector<int> sizes;
			for (auto val : output1_node_dims)
				sizes.emplace_back(val);
			cv::Mat data1 = cv::Mat(sizes, CV_32F, output1);

			auto tend = std::chrono::high_resolution_clock::now();
			std::cout << "elapsed millisenconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms" << std::endl;

			post_process_mask(data0, data1, sizes, classes, key, frame);
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}

/////////////////////////////////////////////////////////////////
// Blog: https://blog.csdn.net/fengbingchun/article/details/139551916
int test_yolov8_segment_opencv()
{
	namespace fs = std::filesystem;

	auto net = cv::dnn::readNetFromONNX(onnx_file);
	if (net.empty()) {
		std::cerr << "Error: there are no layers in the network: " << onnx_file << std::endl;
		return -1;
	}

	if (cuda_enabled) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	} else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

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

	for (const auto& [key, val] : get_dir_images(images_dir)) {
		cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
		if (frame.empty()) {
			std::cerr << "Warning: unable to load image: " << val << std::endl;
			continue;
		}

		auto tstart = std::chrono::high_resolution_clock::now();
		cv::Mat bgr = modify_image_size(frame);

		cv::Mat blob;
		cv::dnn::blobFromImage(bgr, blob, 1.0 / 255.0, cv::Size(input_size[1], input_size[0]), cv::Scalar(), true, false);
		net.setInput(blob);

		std::vector<cv::Mat> outputs;
		net.forward(outputs, net.getUnconnectedOutLayersNames());
		if (outputs.size() != 2) {
			std::cerr << "Error: output must have 2 layers: " << outputs.size() << std::endl;
			return -1;
		}

		// output0
		cv::Mat data0 = cv::Mat(outputs[0].size[1], outputs[0].size[2], CV_32FC1, outputs[0].data).t();

		// output1
		std::vector<int> sizes;
		for (int i = 0; i < 4; ++i)
			sizes.emplace_back(outputs[1].size[i]);
		cv::Mat data1 = cv::Mat(sizes, CV_32F, outputs[1].data);

		auto tend = std::chrono::high_resolution_clock::now();
		std::cout << "elapsed millisenconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms" << std::endl;

		post_process_mask(data0, data1, sizes, classes, key, frame);
	}

	return 0;
}

/////////////////////////////////////////////////////////////////
// Blog: https://blog.csdn.net/fengbingchun/article/details/139552222
int test_yolov8_segment_libtorch()
{
	if (auto flag = torch::cuda::is_available(); flag == true)
		std::cout << "cuda is available" << std::endl;
	else
		std::cout << "cuda is not available" << std::endl;

	torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

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

	if (!std::filesystem::exists(result_dir)) {
		std::filesystem::create_directories(result_dir);
	}

	try {
		torch::jit::script::Module model;
		if (torch::cuda::is_available() == true)
			model = torch::jit::load(torchscript_file, torch::kCUDA);
		else
			model = torch::jit::load(torchscript_file, torch::kCPU);
		model.eval();
		// note: cpu is normal; gpu is abnormal: the model may not be fully placed on the gpu 
		// model = torch::jit::load(file); model.to(torch::kCUDA) ==> model = torch::jit::load(file, torch::kCUDA)
		// model.to(device, torch::kFloat32);

		for (const auto& [key, val] : get_dir_images(images_dir)) {
			cv::Mat frame = cv::imread(val, cv::IMREAD_COLOR);
			if (frame.empty()) {
				std::cerr << "Warning: unable to load image: " << val << std::endl;
				continue;
			}

			auto tstart = std::chrono::high_resolution_clock::now();
			cv::Mat bgr = modify_image_size(frame);
			cv::resize(bgr, bgr, cv::Size(input_size[1], input_size[0]));

			torch::Tensor tensor = torch::from_blob(bgr.data, { bgr.rows, bgr.cols, 3 }, torch::kByte).to(device);
			tensor = tensor.toType(torch::kFloat32).div(255);
			tensor = tensor.permute({ 2, 0, 1 });
			tensor = tensor.unsqueeze(0);
			std::vector<torch::jit::IValue> inputs{ tensor };

			// reference: https://medium.com/@psopen11/complete-guide-to-gpu-accelerated-yolov8-segmentation-in-c-via-libtorch-c-dlls-a0e3e6029d82
			auto output = model.forward(inputs).toTuple()->elements();
			auto output0 = output[0].toTensor().transpose(1, 2).contiguous().to(torch::kCPU);
			auto output1 = output[1].toTensor().to(torch::kCPU);
			if (output0.dim() != 3 || output1.dim() != 4) {
				std::cerr << "Error: unmatch dimensions: " << output0.dim() << "," << output1.dim() << std::endl;
			}

			cv::Mat data0 = cv::Mat(output0.size(1), output0.size(2), CV_32FC1, output0.data_ptr<float>());

			std::vector<int> sizes;
			for (int i = 0; i < 4; ++i)
				sizes.emplace_back(output1.size(i));
			cv::Mat data1 = cv::Mat(sizes, CV_32F, output1.data_ptr<float>());

			auto tend = std::chrono::high_resolution_clock::now();
			std::cout << "elapsed millisenconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms" << std::endl;

			post_process_mask(data0, data1, sizes, classes, key, frame);
		}
	}
	catch (const c10::Error& e) {
		std::cerr << "Error: " << e.msg() << std::endl;
		return -1;
	}

	return 0;
}

#endif // _MSC_VER
