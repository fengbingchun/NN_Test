#include "batch_normalization.hpp"
#include <string.h>
#include <vector>
#include <cmath>
#include "common.hpp"

namespace ANN {

int BatchNorm::LoadData(const float* data, int length)
{
	CHECK(number_ * channels_ * height_ * width_ == length);

	data_.resize(length);
	memcpy(data_.data(), data, length * sizeof(float));
	return 0;
}

std::unique_ptr<float[]> BatchNorm::Run()
{
	int spatial_size = height_ * width_;
	for (int n = 0; n < number_; ++n) {
		int offset = n * (channels_ * spatial_size);
		for (int c = 0; c < channels_; ++c) {
			const float* p = data_.data() + offset + (c * spatial_size);
			for (int k = 0; k < spatial_size; ++k) {
				mean_[c] += *p++;
			}
		}
	}

	std::transform(mean_.begin(), mean_.end(), mean_.begin(), [=](float_t x) { return x / (number_ * spatial_size); });

	for (int n = 0; n < number_; ++n) {
		int offset = n * (channels_ * spatial_size);
		for (int c = 0; c < channels_; ++c) {
			const float* p = data_.data() + offset + (c * spatial_size);
			for (int k = 0; k < spatial_size; ++k) {
				variance_[c] += std::pow(*p++ - mean_[c], 2.);
			}
		}
	}

	std::transform(variance_.begin(), variance_.end(), variance_.begin(), [=](float_t x) { return x / (std::max(1., number_*spatial_size*1.)); });

	std::vector<float> stddev(channels_);
	for (int c = 0; c < channels_; ++c) {
		stddev[c] = std::sqrt(variance_[c] + epsilon_);
	}

	std::unique_ptr<float[]> output(new float[number_ * channels_ * spatial_size]);
	for (int n = 0; n < number_; ++n) {
		const float* p1 = data_.data() + n * (channels_ * spatial_size);
		float* p2 = output.get() + n * (channels_ * spatial_size);

		for (int c = 0; c < channels_; ++c) {
			for (int k = 0; k < spatial_size; ++k) {
				*p2++ = (*p1++ - mean_[c]) / stddev[c];
			}
		}
	}

	return output;
}

} // namespace ANN