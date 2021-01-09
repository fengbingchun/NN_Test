#ifndef FBC_NN_LRN_HPP_
#define FBC_NN_LRN_HPP_

// Blog: https://blog.csdn.net/fengbingchun/article/details/112393884

namespace ANN {

enum class NormRegion {
		ACROSS_CHANNEL = 0,
		WITHIN_CHANNEL
};

template<typename T = float>
class LRN {
public:
	LRN() = default;
	LRN(unsigned int local_size, T alpha, T beta, T bias, NormRegion norm_region) :
		local_size_(local_size), alpha_(alpha), beta_(beta), bias_(bias), norm_region_(norm_region) {}
	int run(const T* input, int batch, int channel, int height, int width, T* output) const;

private:
	int across_channel(const T* input, int batch, int channel, int height, int width, T* output) const;
	int within_channel(const T* input, int batch, int channel, int height, int width, T* output) const;

	unsigned int local_size_ = 5; // n
	T alpha_ = 1.;
	T beta_ = 0.75;
	T bias_ = 1.; // k
	NormRegion norm_region_ = NormRegion::ACROSS_CHANNEL;
};

} // namespace ANN

#endif // FBC_NN_LRN_HPP_
