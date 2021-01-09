#include "lrn.hpp"
#include <algorithm>
#include <cmath>

namespace ANN {

template<typename T>
int LRN<T>::run(const T* input, int batch, int channel, int height, int width, T* output) const
{
	if (norm_region_ == NormRegion::ACROSS_CHANNEL)
		return across_channel(input, batch, channel, height, width, output);
	else
		return within_channel(input, batch, channel, height, width, output);
}

template<typename T>
int LRN<T>::across_channel(const T* input, int batch, int channel, int height, int width, T* output) const
{
	int size = channel * height * width;

	for (int p = 0; p < batch; ++p) {
		const T* in = input + size * p;
		T* out = output + size * p;

		// N = channel; n = local_size_; k = bias_
		for (int i = 0; i < channel; ++i) {
			for (int y = 0; y < height; ++y) {
				for (int x = 0; x < width; ++x) {
					T tmp = 0;
					for (int j = std::max(0, static_cast<int>(i - local_size_ / 2)); j <= std::min(channel - 1, static_cast<int>(i + local_size_ / 2)); ++j) {
						tmp += std::pow(in[j * height * width + width * y + x], 2);
					}
					out[i * height * width + width * y + x] = in[i * height * width + width * y + x] / std::pow(bias_ + alpha_ * tmp, beta_);
				}
			}
		}
	}

	return 0;
}

template<typename T>
int LRN<T>::within_channel(const T* input, int batch, int channel, int height, int width, T* output) const
{
	fprintf(stderr, "not implemented\n");
	return -1;
}

template class LRN<float>;

} // namespace ANN