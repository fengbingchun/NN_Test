#include "opencv.hpp"
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include "common.hpp"

///////////////////////////////// K-Means ///////////////////////////////
int test_opencv_kmeans()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/79395298
#ifdef _MSC_VER
	const std::string image_path{ "../../../data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat tmp = cv::imread(image_path + "0_1.jpg", 0);
	CHECK(tmp.data != nullptr && tmp.channels() == 1);
	const int samples_number{ 80 }, every_class_number{ 20 }, categories_number{ samples_number / every_class_number};

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

	const int K{ 4 }, attemps{ 100 };
	const cv::TermCriteria term_criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.01);
	cv::Mat labels_, centers_;
	double value = cv::kmeans(samples_data, K, labels_, term_criteria, attemps, cv::KMEANS_RANDOM_CENTERS, centers_);
	fprintf(stdout, "K = %d, attemps = %d, iter count = %d, compactness measure =  %f\n",
		K, attemps, term_criteria.maxCount, value);

	CHECK(labels_.rows == samples_number);
	int* p2 = reinterpret_cast<int*>(labels_.data);
	for (int i = 1; i <= every_class_number; ++i) {
		for (int j = 0; j < categories_number; ++j) {
			fprintf(stdout, "  %d  ", *p2++);
		}
		fprintf(stdout, "\n");
	}

	return 0;
}

////////////////////////////// PCA(Principal Component Analysis) ///////////////////////
int test_opencv_pca()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/79053870
	// reference: opencv-3.3.0/samples/cpp/pca.cpp
#ifdef _MSC_VER
	const std::string image_path{ "../../../data/database/ORL_Faces/" };
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

	cv::Mat data(images.size(), images[0].rows * images[0].cols, CV_32FC1);
	for (int i = 0; i < images.size(); ++i) {
		cv::Mat image_row = images[i].clone().reshape(1, 1);
		cv::Mat row_i = data.row(i);
		image_row.convertTo(row_i, CV_32F);
	}

	cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW, 0.95f);

	std::vector<cv::Mat> result(images.size());
	for (int i = 0; i < images.size(); ++i) {
		// Demonstration of the effect of retainedVariance on the first image
		cv::Mat point = pca.project(data.row(i)); // project into the eigenspace, thus the image becomes a "point"
		cv::Mat reconstruction = pca.backProject(point); // re-create the image from the "point"
		reconstruction = reconstruction.reshape(images[i].channels(), images[i].rows); // reshape from a row vector into image shape
		cv::normalize(reconstruction, reconstruction, 0, 255, cv::NORM_MINMAX, CV_8UC1);
		reconstruction.copyTo(result[i]);
	}
#ifdef _MSC_VER
	save_images(result, "../../../data/pca_result_.jpg", 5);
#else
	save_images(result, "data/pca_result_.jpg", 5);
#endif

	// save file
#ifdef _MSC_VER
	const std::string save_file{ "../../../data/pca.xml" }; // .xml, .yaml, .jsons
#else
	const std::string save_file{ "data/pca.xml" }; // .xml, .yaml, .jsons
#endif
	cv::FileStorage fs(save_file, cv::FileStorage::WRITE);
	pca.write(fs);
	fs.release();

	// read file
	const std::string& read_file = save_file;
	cv::FileStorage fs2(read_file, cv::FileStorage::READ);
	cv::PCA pca2;
	pca2.read(fs2.root());
	fs2.release();

	return 0;
}

///////////////////////////////////// Decision Tree ////////////////////////////////////////
// Blog: http://blog.csdn.net/fengbingchun/article/details/78882055
int test_opencv_decision_tree_train()
{
#ifdef _MSC_VER
	const std::string image_path{ "../../../data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat tmp = cv::imread(image_path + "0_1.jpg", 0);
	CHECK(tmp.data != nullptr);
	const int train_samples_number{ 40 };
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

	cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
	dtree->setMaxCategories(4);
	dtree->setMaxDepth(10);
	dtree->setMinSampleCount(10);
	dtree->setCVFolds(0);
	dtree->setUseSurrogates(false);
	dtree->setUse1SERule(false);
	dtree->setTruncatePrunedTree(false);
	dtree->setRegressionAccuracy(0);
	dtree->setPriors(cv::Mat());

	dtree->train(train_data, cv::ml::ROW_SAMPLE, train_labels);

#ifdef _MSC_VER
	const std::string save_file{ "../../../data/decision_tree_model.xml" }; // .xml, .yaml, .jsons
#else
	const std::string save_file{ "data/decision_tree_model.xml" }; // .xml, .yaml, .jsons
#endif
	dtree->save(save_file);

	return 0;
}

int test_opencv_decision_tree_predict()
{
#ifdef _MSC_VER
	const std::string image_path{ "../../../data/images/digit/handwriting_0_and_1/" };
	const std::string load_file{ "../../../data/decision_tree_model.xml" }; // .xml, .yaml, .jsons
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
	const std::string load_file{ "data/decision_tree_model.xml" }; // .xml, .yaml, .jsons
#endif
	const int predict_samples_number{ 40 };
	const int every_class_number{ 10 };

	cv::Mat tmp = cv::imread(image_path + "0_1.jpg", 0);
	CHECK(tmp.data != nullptr);

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

	cv::Mat result;
	cv::Ptr<cv::ml::DTrees> dtrees = cv::ml::DTrees::load(load_file);

	dtrees->predict(predict_data, result);
	CHECK(result.rows == predict_samples_number);

	cv::Mat predict_labels(predict_samples_number, 1, CV_32FC1);
	float* p = (float*)predict_labels.data;
	for (int i = 0; i < 4; ++i) {
		std::for_each(p + i * every_class_number, p + (i + 1)*every_class_number, [i](float& v){v = (float)i; });
	}

	int count{ 0 };
	for (int i = 0; i < predict_samples_number; ++i) {
		float value1 = ((float*)predict_labels.data)[i];
		float value2 = ((float*)result.data)[i];
		fprintf(stdout, "expected value: %f, actual value: %f\n", value1, value2);

		if (int(value1) == int(value2)) ++count;
	}
	fprintf(stdout, "accuracy: %f\n", count * 1.f / predict_samples_number);

	return 0;
}

/////////////////////////////////////////// K-Nearest Neighbor(KNN) //////////////////////////////////////
int test_opencv_knn_predict()
{
	// Blog: http://blog.csdn.net/fengbingchun/article/details/78485669
	const int K{ 3 };
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
	knn->setDefaultK(K);
	knn->setIsClassifier(true);
	knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);

#ifdef _MSC_VER
	const std::string image_path{"../../../data/images/digit/handwriting_0_and_1/"};
#else
	const std::string image_path{"data/images/digit/handwriting_0_and_1/"};
#endif
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

	knn->train(train_data, cv::ml::ROW_SAMPLE, train_labels);

	// predict datta
	cv::Mat predict_data(predict_samples_number, tmp.rows * tmp.cols, CV_32FC1);
	for (int i = 0; i < 4; ++i) {
		static const std::vector<std::string> digit{ "0_", "1_", "2_", "3_" };
		static const std::string suffix{ ".jpg" };

		for (int j = 11; j <= every_class_number+10; ++j) {
			std::string image_name = image_path + digit[i] + std::to_string(j) + suffix;
			cv::Mat image = cv::imread(image_name, 0);
			CHECK(!image.empty() && image.isContinuous());
			image.convertTo(image, CV_32FC1);

			image = image.reshape(0, 1);
			tmp = predict_data.rowRange(i * every_class_number + j - 10 - 1, i * every_class_number + j - 10);
			image.copyTo(tmp);
		}
	}

	cv::Mat result;
	knn->findNearest(predict_data, K, result);
	CHECK(result.rows == predict_samples_number);

	cv::Mat predict_labels(predict_samples_number, 1, CV_32FC1);
	p = (float*)predict_labels.data;
	for (int i = 0; i < 4; ++i) {
		std::for_each(p + i * every_class_number, p + (i + 1)*every_class_number, [i](float& v){v = (float)i; });
	}

	int count{ 0 };
	for (int i = 0; i < predict_samples_number; ++i) {
		float value1 = ((float*)predict_labels.data)[i];
		float value2 = ((float*)result.data)[i];
		fprintf(stdout, "expected value: %f, actual value: %f\n", value1, value2);

		if (int(value1) == int(value2)) ++count;
	}
	fprintf(stdout, "when K = %d, accuracy: %f\n", K, count * 1.f / predict_samples_number);

	return 0;
}

/////////////////////////////////// Support Vector Machines(SVM) ///////////////////////////
// Blog: http://blog.csdn.net/fengbingchun/article/details/78353140
int test_opencv_svm_train()
{
	// two class classifcation
	const std::vector<int> labels { 1, -1, -1, -1 };
	const std::vector<std::vector<float>> trainingData{ { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
	const int feature_length{ 2 };
	const int samples_count{ (int)trainingData.size()};
	CHECK(labels.size() == trainingData.size());

	std::vector<float> data(samples_count * feature_length, 0.f);
	for (int i = 0; i < samples_count; ++i) {
		for (int j = 0; j < feature_length; ++j) {
			data[i*feature_length + j] = trainingData[i][j];
		}
	}

	cv::Mat trainingDataMat(samples_count, feature_length, CV_32FC1, data.data());
	cv::Mat labelsMat((int)samples_count, 1, CV_32SC1, (int*)labels.data());

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

	CHECK(svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat));

#ifdef _MSC_VER
	const std::string save_file{ "../../../data/svm_model.xml" }; // .xml, .yaml, .jsons
#else
	const std::string save_file{ "data/svm_model.xml" }; // .xml, .yaml, .jsons
#endif
	svm->save(save_file);

	return 0;
}

int test_opencv_svm_predict()
{
#ifdef _MSC_VER
	const std::string model_file { "../../../data/svm_model.xml" };
#else
	const std::string model_file { "data/svm_model.xml" };
#endif
	const std::vector<int> labels{ 1, 1, 1, 1, -1, -1, -1, -1 };
	const std::vector<std::vector<float>> predictData{ { 490.f, 15.f }, { 480.f, 30.f }, { 511.f, 40.f }, { 473.f, 50.f },
		{ 2.f, 490.f }, { 100.f, 200.f }, { 247.f, 223.f }, {510.f, 400.f} };
	const int feature_length{ 2 };
	const int predict_count{ (int)predictData.size() };
	CHECK(labels.size() == predictData.size());

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(model_file);
	for (int i = 0; i < predict_count; ++i) {
		cv::Mat prdictMat = (cv::Mat_<float>(1, 2) << predictData[i][0], predictData[i][1]);
		float response = svm->predict(prdictMat);

		fprintf(stdout, "actual class: %d, target calss: %f\n", labels[i], response);
	}

	return 0;
}

int test_opencv_svm_simple()
{
	// two class classifcation
	// reference: opencv-3.3.0/samples/cpp/tutorial_code/ml/introduction_to_svm/introduction_to_svm.cpp
	const int width{ 512 }, height{ 512 };
	cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

	const int labels[] { 1, -1, -1, -1 };
	const float trainingData[][2] {{ 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };

	cv::Mat trainingDataMat(4, 2, CV_32FC1, (float*)trainingData);
	cv::Mat labelsMat(4, 1, CV_32SC1, (int*)labels);

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));

	svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);

	// Show the decision regions given by the SVM
	cv::Vec3b green(0, 255, 0), blue(255, 0, 0);
	for (int i = 0; i < image.rows; ++i) {
		for (int j = 0; j < image.cols; ++j) {
			cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << j, i);
			float response = svm->predict(sampleMat);

			if (response == 1)
				image.at<cv::Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<cv::Vec3b>(i, j) = blue;
		}
	}

	// Show the training data
	int thickness{ -1 };
	int lineType{ 8 };
	cv::circle(image, cv::Point(501, 10), 5, cv::Scalar(0, 0, 0), thickness, lineType);
	cv::circle(image, cv::Point(255, 10), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	cv::circle(image, cv::Point(501, 255), 5, cv::Scalar(255, 255, 255), thickness, lineType);
	cv::circle(image, cv::Point(10, 501), 5, cv::Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType = 8;
	cv::Mat sv = svm->getUncompressedSupportVectors();

	for (int i = 0; i < sv.rows; ++i) {
		const float* v = sv.ptr<float>(i);
		cv::circle(image, cv::Point((int)v[0], (int)v[1]), 6, cv::Scalar(128, 128, 128), thickness, lineType);
	}

#ifdef _MSC_VER
	cv::imwrite("../../../data/result_svm_simple.png", image);
#else
	cv::imwrite("data/result_svm_simple.png", image);
#endif
	imshow("SVM Simple Example", image);
	cv::waitKey(0);

	return 0;
}

int test_opencv_svm_non_linear()
{
	// two class classifcation
	// reference: opencv-3.3.0/samples/cpp/tutorial_code/ml/non_linear_svms/non_linear_svms.cpp
	const int NTRAINING_SAMPLES{ 100 }; // Number of training samples per class
	const float FRAC_LINEAR_SEP{ 0.9f }; // Fraction of samples which compose the linear separable part

	// Data for visual representation
	const int WIDTH{ 512 }, HEIGHT{ 512 };
	cv::Mat I = cv::Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	// Set up training data randomly
	cv::Mat trainData(2 * NTRAINING_SAMPLES, 2, CV_32FC1);
	cv::Mat labels(2 * NTRAINING_SAMPLES, 1, CV_32SC1);

	cv::RNG rng(100); // Random value generation class

	// Set up the linearly separable part of the training data
	int nLinearSamples = (int)(FRAC_LINEAR_SEP * NTRAINING_SAMPLES);

	// Generate random points for the class 1
	cv::Mat trainClass = trainData.rowRange(0, nLinearSamples);

	// The x coordinate of the points is in [0, 0.4)
	cv::Mat c = trainClass.colRange(0, 1);
	rng.fill(c, cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(0.4 * WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(HEIGHT));

	// Generate random points for the class 2
	trainClass = trainData.rowRange(2 * NTRAINING_SAMPLES - nLinearSamples, 2 * NTRAINING_SAMPLES);
	// The x coordinate of the points is in [0.6, 1]
	c = trainClass.colRange(0, 1);
	rng.fill(c, cv::RNG::UNIFORM, cv::Scalar(0.6*WIDTH), cv::Scalar(WIDTH));
	// The y coordinate of the points is in [0, 1)
	c = trainClass.colRange(1, 2);
	rng.fill(c, cv::RNG::UNIFORM, cv::Scalar(1), cv::Scalar(HEIGHT));

	// Set up the labels for the classes
	labels.rowRange(0, NTRAINING_SAMPLES).setTo(1); // Class 1
	labels.rowRange(NTRAINING_SAMPLES, 2 * NTRAINING_SAMPLES).setTo(2); // Class 2

	// Train the svm
	std::cout << "Starting training process" << std::endl;
	// init, Set up the support vector machines parameters
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setC(0.1);
	svm->setKernel(cv::ml::SVM::LINEAR);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)1e7, 1e-6));

	svm->train(trainData, cv::ml::ROW_SAMPLE, labels);
	std::cout << "Finished training process" << std::endl;

	// Show the decision regions
	cv::Vec3b green(0, 100, 0), blue(100, 0, 0);
	for (int i = 0; i < I.rows; ++i) {
		for (int j = 0; j < I.cols; ++j) {
			cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << i, j);
			float response = svm->predict(sampleMat);

			if (response == 1) I.at<cv::Vec3b>(j, i) = green;
			else if (response == 2) I.at<cv::Vec3b>(j, i) = blue;
		}
	}

	// Show the training data
	int thick = -1;
	int lineType = 8;
	float px, py;
	// Class 1
	for (int i = 0; i < NTRAINING_SAMPLES; ++i) {
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, cv::Point((int)px, (int)py), 3, cv::Scalar(0, 255, 0), thick, lineType);
	}
	// Class 2
	for (int i = NTRAINING_SAMPLES; i <2 * NTRAINING_SAMPLES; ++i) {
		px = trainData.at<float>(i, 0);
		py = trainData.at<float>(i, 1);
		circle(I, cv::Point((int)px, (int)py), 3, cv::Scalar(255, 0, 0), thick, lineType);
	}

	// Show support vectors
	thick = 2;
	lineType = 8;
	cv::Mat sv = svm->getUncompressedSupportVectors();

	for (int i = 0; i < sv.rows; ++i) {
		const float* v = sv.ptr<float>(i);
		circle(I, cv::Point((int)v[0], (int)v[1]), 6, cv::Scalar(128, 128, 128), thick, lineType);
	}

#ifdef _MSC_VER
	imwrite("../../../data/result_svm_non_linear.png", I);
#else
	imwrite("data/result_svm_non_linear.png", I);
#endif
	imshow("SVM for Non-Linear Training Data", I);
	cv::waitKey(0);

	return 0;
}

////////////////////////////////// Logistic Regression ///////////////////////////////
// Blog: http://blog.csdn.net/fengbingchun/article/details/78221693
static void show_image(const cv::Mat& data, int columns, const std::string& name)
{
	cv::Mat big_image;
	for (int i = 0; i < data.rows; ++i) {
		big_image.push_back(data.row(i).reshape(0, columns));
	}

	cv::imshow(name, big_image);
	cv::waitKey(0);
}

static float calculate_accuracy_percent(const cv::Mat& original, const cv::Mat& predicted)
{
	return 100 * (float)cv::countNonZero(original == predicted) / predicted.rows;
}

int test_opencv_logistic_regression_train()
{
#ifdef _MSC_VER
	const std::string image_path{ "../../../data/images/digit/handwriting_0_and_1/" };
#else
	const std::string image_path{ "data/images/digit/handwriting_0_and_1/" };
#endif
	cv::Mat data, labels, result;

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
	//show_image(data, 28, "train data");

	std::unique_ptr<float[]> tmp(new float[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0.f;
		else tmp[i] = 1.f;
	}
	labels = cv::Mat(20, 1, CV_32FC1, tmp.get());

	cv::Ptr<cv::ml::LogisticRegression> lr = cv::ml::LogisticRegression::create();
	lr->setLearningRate(0.00001);
	lr->setIterations(100);
	lr->setRegularization(cv::ml::LogisticRegression::REG_DISABLE);
	lr->setTrainMethod(cv::ml::LogisticRegression::MINI_BATCH);
	lr->setMiniBatchSize(1);

	CHECK(lr->train(data, cv::ml::ROW_SAMPLE, labels));

#ifdef _MSC_VER
	const std::string save_file{ "../../../data/logistic_regression_model.xml" }; // .xml, .yaml, .jsons
#else
	const std::string save_file{ "data/logistic_regression_model.xml" }; // .xml, .yaml, .jsons
#endif
	lr->save(save_file);

	return 0;
}

int test_opencv_logistic_regression_predict()
{
#ifdef _MSC_VER
	const std::string image_path{ "../../../data/images/digit/handwriting_0_and_1/" };
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
	//show_image(data, 28, "test data");

	std::unique_ptr<int[]> tmp(new int[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0;
		else tmp[i] = 1;
	}
	labels = cv::Mat(20, 1, CV_32SC1, tmp.get());

#ifdef _MSC_VER
	const std::string model_file{ "../../../data/logistic_regression_model.xml" };
#else
	const std::string model_file{ "data/logistic_regression_model.xml" };
#endif
	cv::Ptr<cv::ml::LogisticRegression> lr = cv::ml::LogisticRegression::load(model_file);

	lr->predict(data, result);

	fprintf(stdout, "predict result: \n");
	std::cout << "actual: " << labels.t() << std::endl;
	std::cout << "target: " << result.t() << std::endl;
	fprintf(stdout, "accuracy: %.2f%%\n", calculate_accuracy_percent(labels, result));

	return 0;
}
