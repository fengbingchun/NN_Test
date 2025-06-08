#include "funset.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// Blog: https://blog.csdn.net/fengbingchun/article/details/138536321

namespace {

bool calculate_image_face_width(cv::CascadeClassifier& face_cascade, const char* image_name, int& P)
{
	cv::Mat bgr = cv::imread(image_name, 1);
	if (!bgr.data) {
		std::cerr << "Error: fail to imread: " << image_name << "\n";
		return false;
	}

	cv::Mat gray;
	cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
	cv::equalizeHist(gray, gray);

	std::vector<cv::Rect> faces;
	face_cascade.detectMultiScale(gray, faces);

	//for (auto i = 0; i < faces.size(); ++i)
	//	cv::rectangle(bgr, faces[i], cv::Scalar(255, 0, 0), 1);
	//cv::imwrite("../../../data/result.jpg", bgr);

	if (faces.size() != 1) {
		std::cerr << "Error: faces size: " << faces.size() << "\n";
		return false;
	}

	P = faces[0].width;

	return true;
}

inline int  calculate_focal_length(int P, int D, int W)
{
	return ((P * D) / W);
}

inline int calculate_distance(int F, int W, int P)
{
	return ((F * W) / P);
}

} // namespace

int test_monocular_ranging_face_triangle_similarity()
{
#ifdef _MSC_VER
	constexpr char file_name[]{ "../../../data/haarcascade_frontalface_alt.xml" };
	constexpr char image_name[]{ "../../../data/images/face/1.jpg" };
#else
	constexpr char file_name[]{ "data/haarcascade_frontalface_alt.xml" };
	constexpr char image_name[]{ "data/images/face/1.jpg" };
#endif

	cv::CascadeClassifier face_cascade;
	if (!face_cascade.load(file_name)) {
		std::cerr << "Error: fail to load file:" << file_name << "\n";
		return -1;
	}

	auto P{ 0 };
	if (!calculate_image_face_width(face_cascade, image_name, P)) {
		std::cerr << "Error: fail to get_image_face_width\n";
		return -1;
	}
	std::cout << "the width of the face in the image: " << P << " pixels\n";

	constexpr int D{ 60 }, W{ 18 }; // cm
	const auto F = calculate_focal_length(P, D, W);
	std::cout << "focal length: " << F << "\n";

	cv::VideoCapture cap(1); // usb camera
	if (!cap.isOpened()) {
		std::cerr << "Error: fail to open capture\n";
		return -1;
	}

	cv::Mat gray;
	constexpr char winn_ame[]{ "Monocular Ranging" };
	cv::namedWindow(winn_ame, 1);
	const std::string text{ "Distance = " };

	for (;;) {
		cv::Mat frame;
		cap >> frame; // get a new frame from camera
		cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(gray, gray);

		std::vector<cv::Rect> faces;
		face_cascade.detectMultiScale(gray, faces);

		for (auto i = 0; i < faces.size(); ++i) {
			cv::rectangle(frame, faces[i], cv::Scalar(255,0,0), 1);

			P = faces[i].width;
			auto D2 = calculate_distance(F, W, P) / 100.; // m

			auto tmp = std::to_string(D2);
			auto pos = tmp.find(".");
			if (pos != std::string::npos)
				tmp = tmp.substr(0, pos+3);

			std::string content = text + tmp + " m";
			cv::putText(frame, content, cv::Point(faces[i].x, faces[i].y - 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
		}

		cv::imshow(winn_ame, frame);
		if (cv::waitKey(30) >= 0)
			break;
	}

	return 0;
}
