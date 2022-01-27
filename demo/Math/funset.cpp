#include "funset.hpp"
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>
#include "common.hpp"

int test_loss_function()
{
{ // only one sample
	const int classes_number = 5;
	std::vector<float> sample = { 0.0418, 0.0801, -1.3888, -1.9604, 1.0712 };
	std::vector<float> target = { 0, 0, 1, 0, 0 }; // One-Hot
	std::vector<float> input(classes_number);
	assert(sample.size() == classes_number && target.size() == classes_number && input.size() == classes_number);

	fbc::activation_function_softmax(sample.data(), input.data(), classes_number);
	float output = fbc::loss_function_cross_entropy(target.data(), input.data(), classes_number);
	fprintf(stdout, "output: %.4f\n", output);
}

{ // five samples
	const int classes_number = 5, samples_number = 5;
	std::vector<std::vector<float>> samples = {{0.0418, 0.0801, -1.3888, -1.9604, 1.0712 },
												{0.3519, -0.6115, -0.0325,  0.4484, -0.1736},
												{0.1530,  0.0670, -0.3894, -1.0830, -0.4757},
												{-1.3519, 0.2115, 1.2325,  -1.4484, 0.9736},
												{1.1230,  -0.5670, 1.0894, 1.9890, 0.03567}};
	std::vector<std::vector<float>> targets = {{0, 0, 0, 0, 1},
											   {0, 0, 0, 1, 0},
											   {0, 0, 1, 0, 0},
											   {0, 1, 0, 0, 0},
											   {1, 0, 0, 0, 0}};
	std::vector<std::vector<float>> inputs(samples_number);
	assert(samples[0].size() == classes_number && targets[0].size() == classes_number && inputs.size() == samples_number);

	float output = 0.;
	for (int i = 0; i < samples_number; ++i) {
		inputs[i].resize(classes_number);
		fbc::activation_function_softmax(samples[i].data(), inputs[i].data(), classes_number);
		output += fbc::loss_function_cross_entropy(targets[i].data(), inputs[i].data(), classes_number);
	}
	output /= samples_number;
	fprintf(stdout, "output: %.4f\n", output);
}

	return 0;
}

int test_dropout()
{
	std::random_device rd; std::mt19937 gen(rd());
	int height = 4, width = 8, size = height * width;

	std::unique_ptr<float[]> bottom(new float[size]), top(new float[size]);	
	std::uniform_real_distribution<float> distribution(-10.f, 10.f);
	for (int i = 0; i < size; ++i) {
		bottom[i] = distribution(gen);
	}

	float dropout_ratio = 0.8f;
	if (fbc::dropout(bottom.get(), width, height, top.get(), dropout_ratio) != 0) {
		fprintf(stderr, "Error: fail to dropout\n");
		return -1;
	}

	fprintf(stdout, "bottom data:\n");
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			fprintf(stdout, " %f ", bottom[h * width + w]);
		}
		fprintf(stdout, "\n");
	}
	
	fprintf(stdout, "top data:\n");
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			fprintf(stdout, " %f ", top[h * width + w]);
		}
		fprintf(stdout, "\n");
	}

	return 0;
}

int test_brute_force_string_match()
{
	const std::string str{ "abcdABCD EFaadfk32!@#34flasf dafe" };
	const std::vector<std::string> sub{ "abcde", "ABCD EF", "fe", "!@#", "asf dafe", "afea"};

	for (const auto& val : sub) {
		fbc::brute_force_result result;

		fbc::brute_force(str, val, result);

		fprintf(stdout, "string match result: status: %d, pos: %d\n",
			std::get<0>(result), std::get<1>(result));
	}

	return 0;
}

int test_activation_function()
{
	std::vector<float> src{ 1.1f, -2.2f, 3.3f, 0.4f, -0.5f, -1.6f };
	int length = src.size();
	std::vector<float> dst(length);

	fprintf(stderr, "source vector: \n");
	fbc::print_matrix(src);
	fprintf(stderr, "calculate activation function:\n");

	fprintf(stderr, "type: tanh result: \n");
	fbc::activation_function_tanh(src.data(), dst.data(), length);
	fbc::print_matrix(dst);
	fprintf(stderr, "type: tanh derivative result: \n");
	fbc::activation_function_tanh_derivative(dst.data(), dst.data(), length);
	fbc::print_matrix(dst);

	fprintf(stderr, "type: sigmoid result: \n");
	fbc::activation_function_sigmoid(src.data(), dst.data(), length);
	fbc::print_matrix(dst);
	fprintf(stderr, "type: sigmoid derivative result: \n");
	fbc::activation_function_sigmoid_derivative(src.data(), dst.data(), length);
	fbc::print_matrix(dst);
	fprintf(stderr, "type: sigmoid fast result: \n");
	fbc::activation_function_sigmoid_fast(src.data(), dst.data(), length);
	fbc::print_matrix(dst);

	fprintf(stderr, "type: softplus result: \n");
	fbc::activation_function_softplus(src.data(), dst.data(), length);
	fbc::print_matrix(dst);
	fprintf(stderr, "type: softplus derivative result: \n");
	fbc::activation_function_softplus_derivative(src.data(), dst.data(), length);
	fbc::print_matrix(dst);

	fprintf(stderr, "type: ReLU result: \n");
	fbc::activation_function_ReLU(src.data(), dst.data(), length);
	fbc::print_matrix(dst);
	fprintf(stderr, "type: ReLU derivative result: \n");
	fbc::activation_function_ReLU_derivative(src.data(), dst.data(), length);
	fbc::print_matrix(dst);

	fprintf(stderr, "type: Leaky ReLUs result: \n");
	fbc::activation_function_Leaky_ReLUs(src.data(), dst.data(), length);
	fbc::print_matrix(dst);
	fprintf(stderr, "type: Leaky ReLUs derivative result: \n");
	fbc::activation_function_Leaky_ReLUs_derivative(src.data(), dst.data(), length);
	fbc::print_matrix(dst);

	fprintf(stderr, "type: Leaky ELUs result: \n");
	fbc::activation_function_ELUs(src.data(), dst.data(), length);
	fbc::print_matrix(dst);

	fprintf(stderr, "type: softmax result: \n");
	fbc::activation_function_softmax(src.data(), dst.data(), length);
	fbc::print_matrix(dst);
	fprintf(stderr, "type: softmax derivative result: \n");
	fbc::activation_function_softmax_derivative(src.data(), dst.data(), length);
	fbc::print_matrix(dst);

	return 0;
}

int test_calcCovarMatrix()
{
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
					{ -3.6f, 9.2f, 0.5f, 7.2f },
					{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	fprintf(stderr, "source matrix:\n");
	fbc::print_matrix(vec);

	fprintf(stderr, "\nc++ implement calculate covariance matrix:\n");
	std::vector<std::vector<float>> covar1;
	std::vector<float> mean1;
	if (fbc::calcCovarMatrix(vec, covar1, mean1, false/*true*/) != 0) {
		fprintf(stderr, "C++ implement calcCovarMatrix fail\n");
		return -1;
	}

	fprintf(stderr, "print covariance matrix: \n");
	fbc::print_matrix(covar1);
	fprintf(stderr, "print mean: \n");
	fbc::print_matrix(mean1);

	fprintf(stderr, "\nc++ opencv calculate covariance matrix:\n");
	cv::Mat mat(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			mat.at<float>(y, x) = vec.at(y).at(x);
		}
	}
	//std::cout << mat << std::endl;

	//std::cout << "mat:" << std::endl << mat << std::endl;
	//cv::Mat means(1, mat.cols, mat.type(), cv::Scalar::all(0));
	//for (int i = 0; i < mat.cols; i++)
	//	means.col(i) = (cv::sum(mat.col(i)) / mat.rows).val[0];
	//std::cout << "means:" << std::endl << means << std::endl;
	//cv::Mat tmp = cv::repeat(means, mat.rows, 1);
	//cv::Mat mat2 = mat - tmp;
	//cv::Mat covar = (mat2.t()*mat2) / (mat2.rows /*- 1*/); // （X'*X)/n-1
	//std::cout << "covar:" << std::endl << covar << std::endl;

	cv::Mat covar2, mean2;
	cv::calcCovarMatrix(mat, covar2, mean2, CV_COVAR_NORMAL | CV_COVAR_ROWS/* | CV_COVAR_SCALE*/, CV_32FC1);
	fprintf(stderr, "print covariance matrix: \n");
	fbc::print_matrix(covar2);
	fprintf(stderr, "print mean: \n");
	fbc::print_matrix(mean2);

	return 0;
}

int test_meanStdDev()
{
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
						{ -3.6f, 9.2f, 0.5f, 7.2f },
						{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	fprintf(stderr, "source matrix:\n");
	fbc::print_matrix(vec);

	double mean1 = 0., variance1 = 0., stddev1 = 0.;
	if (fbc::meanStdDev(vec, &mean1, &variance1, &stddev1) != 0) {
		fprintf(stderr, "C++ implement meanStdDev fail\n");
		return -1;
	}
	fprintf(stderr, "\nc++ implement meanStdDev: mean: %f, variance: %f, standard deviation: %f\n",
		mean1, variance1, stddev1);

	cv::Mat mat(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			mat.at<float>(y, x) = vec.at(y).at(x);
		}
	}

	cv::Scalar mean2_, stddev2_;
	cv::meanStdDev(mat, mean2_, stddev2_);
	auto mean2 = mean2_.val[0];
	auto stddev2 = stddev2_.val[0];
	auto variance2 = stddev2 * stddev2;
	fprintf(stderr, "\nopencv implement meanStdDev: mean: %f, variance: %f, standard deviation: %f\n",
		mean2, variance2, stddev2);

	return 0;
}

int test_trace()
{
	std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
					{ -3.6f, 9.2f, 0.5f, 7.2f },
					{ 4.3f, 1.3f, 9.4f, -3.4f } };
	const int rows{ 3 }, cols{ 4 };

	fprintf(stderr, "source matrix:\n");
	fbc::print_matrix(vec);

	float tr = fbc::trace(vec);
	fprintf(stderr, "\nc++ implement trace: %f\n", tr);

	cv::Mat mat(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			mat.at<float>(y, x) = vec.at(y).at(x);
		}
	}

	cv::Scalar scalar = cv::trace(mat);
	fprintf(stderr, "\nopencv implement trace: %f\n", scalar.val[0]);

	return 0;
}

int test_pseudoinverse()
{
	//std::vector<std::vector<float>> vec{ { 0.68f, 0.597f },
	//				{ -0.211f, 0.823f },
	//				{ 0.566f, -0.605f } };
	//const int rows{ 3 }, cols{ 2 };

	std::vector<std::vector<float>> vec{ { 0.68f, 0.597f, -0.211f },
						{ 0.823f, 0.566f, -0.605f } };
	const int rows{ 2 }, cols{ 3 };

	fprintf(stderr, "source matrix:\n");
	fbc::print_matrix(vec);

	fprintf(stderr, "\nc++ implement pseudoinverse:\n");
	std::vector<std::vector<float>> pinv1;
	float  pinvtoler = 1.e-6;
	if (fbc::pinv(vec, pinv1, pinvtoler) != 0) {
		fprintf(stderr, "C++ implement pseudoinverse fail\n");
		return -1;
	}
	fbc::print_matrix(pinv1);

	fprintf(stderr, "\nopencv implement pseudoinverse:\n");
	cv::Mat mat(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			mat.at<float>(y, x) = vec.at(y).at(x);
		}
	}

	cv::Mat pinv2;
	cv::invert(mat, pinv2, cv::DECOMP_SVD);
	fbc::print_matrix(pinv2);

	return 0; 
}

int test_SVD()
{
	//std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
	//				{ -3.6f, 9.2f, 0.5f, 7.2f },
	//				{ 4.3f, 1.3f, 9.4f, -3.4f },
	//				{ 6.4f, 0.1f, -3.7f, 0.9f } };
	//const int rows{ 4 }, cols{ 4 };

	//std::vector<std::vector<float>> vec{ { 1.2f, 2.5f, 5.6f, -2.5f },
	//				{ -3.6f, 9.2f, 0.5f, 7.2f },
	//				{ 4.3f, 1.3f, 9.4f, -3.4f } };
	//const int rows{ 3 }, cols{ 4 };

	std::vector<std::vector<float>> vec{ { 0.68f, 0.597f },
					{ -0.211f, 0.823f },
					{ 0.566f, -0.605f } };
	const int rows{ 3 }, cols{ 2 };

	fprintf(stderr, "source matrix:\n");
	fbc::print_matrix(vec);

	fprintf(stderr, "\nc++ implement singular value decomposition:\n");
	std::vector<std::vector<float>> matD, matU, matVt;
	if (fbc::svd(vec, matD, matU, matVt) != 0) {
		fprintf(stderr, "C++ implement singular value decomposition fail\n");
		return -1;
	}
	fprintf(stderr, "singular values:\n");
	fbc::print_matrix(matD);
	fprintf(stderr, "left singular vectors:\n");
	fbc::print_matrix(matU);
	fprintf(stderr, "transposed matrix of right singular values:\n");
	fbc::print_matrix(matVt);

	fprintf(stderr, "\nopencv singular value decomposition:\n");
	cv::Mat mat(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < cols; ++x) {
			mat.at<float>(y, x) = vec.at(y).at(x);
		}
	}

	/*
		w calculated singular values
		u calculated left singular vectors
		vt transposed matrix of right singular vectors
	*/
	cv::Mat w, u, vt, v;
	cv::SVD::compute(mat, w, u, vt, 4);
	//cv::transpose(vt, v);

	fprintf(stderr, "singular values:\n");
	fbc::print_matrix(w);
	fprintf(stderr, "left singular vectors:\n");
	fbc::print_matrix(u);
	fprintf(stderr, "transposed matrix of right singular values:\n");
	fbc::print_matrix(vt);

	return 0;
}

int test_eigenvalues_eigenvectors()
{
	std::vector<float> vec{ 1.23f, 2.12f, -4.2f,
		2.12f, -5.6f, 8.79f,
		-4.2f, 8.79f, 7.3f };
	const int N{ 3 };

	fprintf(stderr, "source matrix:\n");
	int count{ 0 };
	for (const auto& value : vec) {
		if (count++ % N == 0) fprintf(stderr, "\n");
		fprintf(stderr, "  %f  ", value);
	}
	fprintf(stderr, "\n\n");

	fprintf(stderr, "c++ compute eigenvalues and eigenvectors, sort:\n");
	std::vector<std::vector<float>> eigen_vectors1, mat1;
	std::vector<float> eigen_values1;
	mat1.resize(N);
	for (int i = 0; i < N; ++i) {
		mat1[i].resize(N);
		for (int j = 0; j < N; ++j) {
			mat1[i][j] = vec[i * N + j];
		}
	}

	if (fbc::eigen(mat1, eigen_values1, eigen_vectors1, true) != 0) {
		fprintf(stderr, "campute eigenvalues and eigenvector fail\n");
		return -1;
	}

	fprintf(stderr, "eigenvalues:\n");
	std::vector<std::vector<float>> tmp(N);
	for (int i = 0; i < N; ++i) {
		tmp[i].resize(1);
		tmp[i][0] = eigen_values1[i];
	}
	fbc::print_matrix(tmp);

	fprintf(stderr, "eigenvectors:\n");
	fbc::print_matrix(eigen_vectors1);

	fprintf(stderr, "c++ compute eigenvalues and eigenvectors, no sort:\n");
	if (fbc::eigen(mat1, eigen_values1, eigen_vectors1, false) != 0) {
		fprintf(stderr, "campute eigenvalues and eigenvector fail\n");
		return -1;
	}

	fprintf(stderr, "eigenvalues:\n");
	for (int i = 0; i < N; ++i) {
		tmp[i][0] = eigen_values1[i];
	}
	fbc::print_matrix(tmp);

	fprintf(stderr, "eigenvectors:\n");
	fbc::print_matrix(eigen_vectors1);

	fprintf(stderr, "\nopencv compute eigenvalues and eigenvectors:\n");
	cv::Mat mat2(N, N, CV_32FC1, vec.data());

	cv::Mat eigen_values2, eigen_vectors2;
	bool ret = cv::eigen(mat2, eigen_values2, eigen_vectors2);
	if (!ret) {
		fprintf(stderr, "fail to run cv::eigen\n");
		return -1;
	}

	fprintf(stderr, "eigenvalues:\n");
	fbc::print_matrix(eigen_values2);

	fprintf(stderr, "eigenvectors:\n");
	fbc::print_matrix(eigen_vectors2);

	return 0;
}

int test_norm()
{
	fprintf(stderr, "test norm with C++:\n");
	std::vector<int> norm_types{ 0, 1, 2 }; // 正无穷、L1、L2
	std::vector<std::string> str{ "Inf", "L1", "L2" };

	// 1. vector
	std::vector<float> vec1{ -2, 3, 1 };
	std::vector<std::vector<float>> tmp1(1);
	tmp1[0].resize(vec1.size());
	for (int i = 0; i < vec1.size(); ++i) {
		tmp1[0][i] = vec1[i];
	}

	for (int i = 0; i < str.size(); ++i) {
		double value{ 0.f };
		fbc::norm(tmp1, norm_types[i], &value);

		fprintf(stderr, "vector: %s: %f\n", str[i].c_str(), value);
	}

	// 2. matrix
	std::vector<float> vec2{ -3, 2, 0, 5, 6, 2, 7, 4, 8 };
	const int row_col{ 3 };
	std::vector<std::vector<float>> tmp2(row_col);
	for (int y = 0; y < row_col; ++y) {
		tmp2[y].resize(row_col);
		for (int x = 0; x < row_col; ++x) {
			tmp2[y][x] = vec2[y * row_col + x];
		}
	}

	for (int i = 0; i < str.size(); ++i) {
		double value{ 0.f };
		fbc::norm(tmp2, norm_types[i], &value);

		fprintf(stderr, "matrix: %s: %f\n", str[i].c_str(), value);
	}

	fprintf(stderr, "\ntest norm with opencv:\n");
	norm_types[0] = 1; norm_types[1] = 2; norm_types[2] = 4; // 正无穷、L1、L2
	cv::Mat mat1(1, vec1.size(), CV_32FC1, vec1.data());

	for (int i = 0; i < norm_types.size(); ++i) {
		double value = cv::norm(mat1, norm_types[i]);
		fprintf(stderr, "vector: %s: %f\n", str[i].c_str(), value);
	}

	cv::Mat mat2(row_col, row_col, CV_32FC1, vec2.data());
	for (int i = 0; i < norm_types.size(); ++i) {
		double value = cv::norm(mat2, norm_types[i]);
		fprintf(stderr, "matrix: %s: %f\n", str[i].c_str(), value);
	}

	return 0;
}

int test_inverse_matrix()
{
	std::vector<float> vec{ 5, -2, 2, 7, 1, 0, 0, 3, -3, 1, 5, 0, 3, -1, -9, 4 };
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}

	std::vector<std::vector<float>> arr(N);
	for (int i = 0; i < N; ++i) {
		arr[i].resize(N);

		for (int j = 0; j < N; ++j) {
			arr[i][j] = vec[i * N + j];
		}
	}

	std::vector<std::vector<float>> inv1;
	int ret = fbc::inverse<float>(arr, inv1, N);

	fprintf(stderr, "source matrix: \n");
	fbc::print_matrix<float>(arr);
	fprintf(stderr, "c++ inverse matrix: \n");
	fbc::print_matrix<float>(inv1);

	cv::Mat mat(N, N, CV_32FC1, vec.data());
	cv::Mat inv2 = mat.inv();
	fprintf(stderr, "opencv inverse matrix: \n");
	fbc::print_matrix(inv2);

	return 0;
}

int test_adjoint_matrix()
{
	std::vector<float> vec{5, -2, 2, 7, 1, 0, 0, 3, -3, 1, 5, 0, 3, -1, -9, 4 };
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}

	std::vector<std::vector<float>> arr(N);
	for (int i = 0; i < N; ++i) {
		arr[i].resize(N);

		for (int j = 0; j < N; ++j) {
			arr[i][j] = vec[i * N + j];
		}
	}

	std::vector<std::vector<float>> adj;
	int ret = fbc::adjoint<float>(arr, adj, N);

	fprintf(stderr, "source matrix: \n");
	fbc::print_matrix<float>(arr);
	fprintf(stderr, "adjoint matrx: \n");
	fbc::print_matrix<float>(adj);

	return 0;
}

static double determinant_opencv(const std::vector<float>& vec)
{
	int length = std::sqrt(vec.size());
	cv::Mat mat(length, length, CV_32FC1, const_cast<float*>(vec.data()));

	// In OpenCV, for small matrices(rows=cols<=3),the direct method is used.
	// For larger matrices the function uses LU factorization with partial pivoting.
	return cv::determinant(mat);
}

int test_determinant()
{
	std::vector<float> vec{ 1, 0, 2, -1, 3, 0, 0, 5, 2, 1, 4, -3, 1, 0, 5, 0};
	const int N{ 4 };
	if (vec.size() != (int)pow(N, 2)) {
		fprintf(stderr, "vec must be N^2\n");
		return -1;
	}
	double det1 = determinant_opencv(vec);

	std::vector<std::vector<float>> arr(N);
	for (int i = 0; i < N; ++i) {
		arr[i].resize(N);

		for (int j = 0; j < N; ++j) {
			arr[i][j] = vec[i * N + j];
		}
	}
	double det2 = fbc::determinant<float>(arr, N);

	fprintf(stderr, "det1: %f, det2: %f\n", det1, det2);

	return 0;
}

int test_matrix_transpose()
{
#ifdef _MSC_VER
	const std::vector<std::string> image_name{ "E:/GitCode/NN_Test/data/images/test1.jpg",
		"E:/GitCode/NN_Test/data/images/ret_mat_transpose.jpg"};
#else
	const std::vector<std::string> image_name{ "data/images/test1.jpg",
		"data/images/ret_mat_transpose.jpg"};
#endif
	cv::Mat mat_src = cv::imread(image_name[0]);
	if (!mat_src.data) {
		fprintf(stderr, "read image fail: %s\n", image_name[0].c_str());
		return -1;
	}

	cv::Mat mat_dst(mat_src.cols, mat_src.rows, mat_src.type());

	for (int h = 0; h < mat_dst.rows; ++h) {
		for (int w = 0; w < mat_dst.cols; ++w) {
			const cv::Vec3b& s = mat_src.at<cv::Vec3b>(w, h);
			cv::Vec3b& d = mat_dst.at<cv::Vec3b>(h, w);
			d = s;
		}
	}

	cv::imwrite(image_name[1], mat_dst);

	return 0;
}
