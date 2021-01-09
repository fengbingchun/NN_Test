#ifndef FBC_NN_KNN_HPP_
#define FBC_NN_KNN_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/78537968

#include <memory>
#include <vector>

namespace ANN {

template<typename T>
class KNN {
public:
	KNN() = default;
	void set_k(int k);
	int set_train_samples(const std::vector<std::vector<T>>& samples, const std::vector<T>& labels);
	int predict(const std::vector<T>& sample, T& result) const;

private:
	int k = 3;
	int feature_length = 0;
	int samples_number = 0;
	std::unique_ptr<T[]> samples;
	std::unique_ptr<T[]> labels;
};

} // namespace ANN

#endif // FBC_NN_KNN_HPP_

