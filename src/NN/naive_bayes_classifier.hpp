#ifndef FBC_NN_NAIVEBAYESCLASSIFIER_HPP_
#define FBC_NN_NAIVEBAYESCLASSIFIER_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/78064270

#include <vector>
#include <tuple>
#include <string>

namespace ANN {

template<typename T>
struct sex_info { // height, weight, foot size, sex
	T height;
	T weight;
	T foot_size;
	int sex; // -1: unspecified, 0: female, 1: male
};

template<typename T>
struct MeanVariance { // height/weight/foot_size's mean and variance
	T mean_height;
	T mean_weight;
	T mean_foot_size;
	T variance_height;
	T variance_weight;
	T variance_foot_size;
};

// Gaussian naive Bayes
template<typename T>
class NaiveBayesClassifier {
public:
	NaiveBayesClassifier() = default;
	int init(const std::vector<sex_info<T>>& info);
	int train(const std::string& model);
	int predict(const sex_info<T>& info) const;
	int load_model(const std::string& model) const;

private:
	void calc_mean_variance(const std::vector<T>& data, std::tuple<T, T>& mean_variance) const;
	T calc_attribute_probability(T value, T mean, T variance) const;
	int store_model(const std::string& model) const;

	MeanVariance<T> male_mv, female_mv;
	std::vector<T> male_height, male_weight, male_foot_size;
	std::vector<T> female_height, female_weight, female_foot_size;
	T male_p = (T)0.5;
	T female_p = (T)0.5;
	int male_train_number = 0;
	int female_train_number = 0;
};

} // namespace ANN

#endif // FBC_NN_NAIVEBAYESCLASSIFIER_HPP_
