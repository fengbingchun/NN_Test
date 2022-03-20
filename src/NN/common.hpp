#ifndef FBC_NN_COMMON_HPP_
#define FBC_NN_COMMON_HPP_

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#define PI 3.14159265358979323846

#define CHECK(x) { \
	if (x) {} \
	else { fprintf(stderr, "Check Failed: %s, file: %s, line: %d\n", #x, __FILE__, __LINE__); return -1; } \
}

template<typename T>
void generator_real_random_number(T* data, int length, T a = (T)0, T b = (T)1, bool default_random = true);
int compare_file();
int mat_horizontal_concatenate();
int save_images(const std::vector<cv::Mat>& src, const std::string& name, int row_image_count);
template<typename T>
int read_txt_file(const char* name, std::vector<std::vector<T>>& data, const char separator, const int rows, const int cols);


#endif // FBC_NN_COMMON_HPP_
