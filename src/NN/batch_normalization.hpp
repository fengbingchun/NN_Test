#ifndef FBC_SRC_NN_BATCH_NORM_HPP_
#define FBC_SRC_NN_BATCH_NORM_HPP_

#include <vector>
#include <memory>
#include <algorithm>

// Blog: https://blog.csdn.net/fengbingchun/article/details/118959997 

namespace ANN {

class BatchNorm {
public:
	BatchNorm(int number, int channels, int height, int width) : number_(number), channels_(channels), height_(height), width_(width)
	{
		mean_.resize(channels_);
		std::fill(mean_.begin(), mean_.end(), 0.);
		variance_.resize(channels_);
		std::fill(variance_.begin(), variance_.end(), 0.);
	}

	int LoadData(const float* data, int length);
	std::unique_ptr<float []> Run();

	void SetGamma(float gamma) { gamma_ = gamma; }
	float GetGamma() const { return gamma_; }
	void SetBeta(float beta) { beta_ = beta; }
	float GetBeta() const { return beta_; }
	void SetMean(std::vector<float> mean) { mean_ = mean; }
	std::vector<float> GetMean() const { return mean_; }
	void SetVariance(std::vector<float> variance) { variance_ = variance; }
	std::vector<float> GetVariance() const { return variance_; }
	void SetEpsilon(float epsilon) { epsilon_ = epsilon; }

private:
	int number_; // mini-batch
	int channels_;
	int height_;
	int width_;
	std::vector<float> mean_;
	std::vector<float> variance_;
	float gamma_  = 1.; // 缩放
	float beta_ = 0.; // 平移
	float epsilon_ = 1e-5; // small positive value to avoid zero-division
	std::vector<float> data_;
};

} // namespace ANN

#endif // FBC_SRC_NN_BATCH_NORM_HPP_

