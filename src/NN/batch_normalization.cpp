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
	mean_.resize(channels_ * height_ * width_);
	memset(mean_.data(), 0, mean_.size() * sizeof(float));

	for (int n = 0; n < number_; ++n) {
		const float* p = data_.data() + n * (channels_ * height_ * width_);
		for (int c = 0; c < channels_; ++c) {
			for (int h = 0; h < height_; ++h) {
				for (int w = 0; w < width_; ++w) {
					mean_[c * height_ * width_ + h * width_ + w] += p[c * height_ * width_ + h * width_ + w];
				}
			}
		}
	}

	for (int len = 0; len < channels_ * height_ * width_; ++len) {
		mean_[len] /= number_;
	}

	variance_.resize(channels_ * height_ * width_);
	memset(variance_.data(), 0, variance_.size() * sizeof(float));

	for (int n = 0; n < number_; ++n) {
		const float* p = data_.data() + n * (channels_ * height_ * width_);
		for (int c = 0; c < channels_; ++c) {
			for (int h = 0; h < height_; ++h) {
				for (int w = 0; w < width_; ++w) {
					variance_[c * height_ * width_ + h * width_ + w] += std::pow(p[c * height_ * width_ + h * width_ + w] - mean_[c * height_ * width_ + h * width_ + w], 2.);
				}
			}
		}
	}

	for (int len = 0; len < channels_ * height_ * width_; ++len) {
		variance_[len] /= number_;
	}

	std::unique_ptr<float[]> output(new float[number_ * channels_ * height_ * width_]);
	for (int n = 0; n < number_; ++n) {
		const float* p1 = data_.data() + n * (channels_ * height_ * width_);
		float* p2 = output.get() + n * (channels_ * height_ * width_);

		for (int c = 0; c < channels_; ++c) {
			for (int h = 0; h < height_; ++h) {
				for (int w = 0; w < width_; ++w) {
					p2[c * height_ * width_ + h * width_ + w] = (p1[c * height_ * width_ + h * width_ + w] - mean_[c * height_ * width_ + h * width_ + w]) /
						std::sqrt(variance_[c * height_ * width_ + h * width_ + w] + epsilon_);
				}
			}
		}
	}

	return output;
}

} // namespace ANN