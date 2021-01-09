#include "knn.hpp"
#include <limits>
#include <algorithm>
#include <functional>
#include "common.hpp"

namespace ANN {

template<typename T>
void KNN<T>::set_k(int k)
{
	this->k = k;
}

template<typename T>
int KNN<T>::set_train_samples(const std::vector<std::vector<T>>& samples, const std::vector<T>& labels)
{
	CHECK(samples.size() == labels.size());
	this->samples_number = samples.size();
	if (this->k > this->samples_number) this->k = this->samples_number;
	this->feature_length = samples[0].size();

	this->samples.reset(new T[this->feature_length * this->samples_number]);
	this->labels.reset(new T[this->samples_number]);

	T* p = this->samples.get();
	for (int i = 0; i < this->samples_number; ++i) {
		T* q = p + i * this->feature_length;

		for (int j = 0; j < this->feature_length; ++j) {
			q[j] = samples[i][j];
		}

		this->labels.get()[i] = labels[i];
	}
}

template<typename T>
int KNN<T>::predict(const std::vector<T>& sample, T& result) const
{
	if (sample.size() != this->feature_length) {
		fprintf(stderr, "their feature length dismatch: %d, %d", sample.size(), this->feature_length);
		return -1;
	}

	typedef std::pair<T, T> value;
	std::vector<value> info;
	for (int i = 0; i < this->k + 1; ++i) {
		info.push_back(std::make_pair(std::numeric_limits<T>::max(), (T)-1.));
	}

	for (int i = 0; i < this->samples_number; ++i) {
		T s{ 0. };
		const T* p = this->samples.get() + i * this->feature_length;

		for (int j = 0; j < this->feature_length; ++j) {
			s += (p[j] - sample[j]) * (p[j] - sample[j]);
		}

		info[this->k] = std::make_pair(s, this->labels.get()[i]);
		std::stable_sort(info.begin(), info.end(), [](const std::pair<T, T>& p1, const std::pair<T, T>& p2) {
			return p1.first < p2.first; });
	}

	std::vector<T> vec(this->k);
	for (int i = 0; i < this->k; ++i) {
		vec[i] = info[i].second;
	}
	std::sort(vec.begin(), vec.end(), std::greater<T>());
	vec.erase(std::unique(vec.begin(), vec.end()), vec.end());

	std::vector<std::pair<T, int>> ret;
	for (int i = 0; i < vec.size(); ++i) {
		ret.push_back(std::make_pair(vec[i], 0));
	}

	for (int i = 0; i < this->k; ++i) {
		for (int j = 0; j < ret.size(); ++j) {
			if (info[i].second == ret[j].first) {
				++ret[j].second;
				break;
			}
		}
	}

	int max = -1, index = -1;
	for (int i = 0; i < ret.size(); ++i) {
		if (ret[i].second > max) {
			max = ret[i].second;
			index = i;
		}
	}

	result = ret[index].first;

	return 0;
}

template class KNN<float>;
template class KNN<double>;

} // namespace ANN
