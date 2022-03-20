#ifndef FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_
#define FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_

/* Blog:
	http://blog.csdn.net/fengbingchun/article/details/79346691
	https://blog.csdn.net/fengbingchun/article/details/123616784
*/

#include <vector>
#include <string>

namespace ANN {

enum class ActivationFunction {
	Sigmoid // logistic sigmoid function
};

enum class LossFunction {
	MSE // Mean Square Error
};

enum class Optimzation {
	BGD, // Batch_Gradient_Descent
	SGD, // Stochastic Gradient Descent
	MBGD // Mini-batch Gradient Descent
};

template<typename T>
class LogisticRegression2 { // two categories
public:
	LogisticRegression2() = default;
	int init(const T* data, const T* labels, int train_num, int feature_length, T learning_rate = 0.00001, int epochs = 300);
	int train(const std::string& model);
	int load_model(const std::string& model);
	T predict(const T* data, int feature_length) const; // y = 1/(1+exp(-(wx+b)))

private:
	int store_model(const std::string& model) const;
	T calculate_z(const std::vector<T>& feature) const;  // z(i)=w^T*x(i)+b
	T calculate_cost_function() const;

	T calculate_activation_function(T value) const;
	T calculate_loss_function() const;
	T calculate_loss_function_derivative() const;
	T calculate_loss_function_derivative(unsigned int index) const;
	void calculate_gradient_descent();

	std::vector<std::vector<T>> x_; // training set
	std::vector<T> y_; // ground truth labels
	std::vector<T> o_; // predict value
	int epochs_ = 100; // epochs
	int m_ = 0; // train samples num
	int feature_length_ = 0; // weights length
	T alpha_ = (T)0.00001; // learning rate
	std::vector<T> w_; // weights
	T b_ = (T)0.; // threshold

	ActivationFunction activation_func_ = ActivationFunction::Sigmoid;
	LossFunction loss_func_ = LossFunction::MSE;
	Optimzation optim_ = Optimzation::BGD;
}; // class LogisticRegression2

} // namespace ANN

#endif // FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_
