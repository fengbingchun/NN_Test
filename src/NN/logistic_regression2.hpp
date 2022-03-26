#ifndef FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_
#define FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_

/* Blog:
	http://blog.csdn.net/fengbingchun/article/details/79346691
	https://blog.csdn.net/fengbingchun/article/details/123616784
*/

#include <vector>
#include <string>
#include <memory>

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

struct Database {
	Database() = default;
	std::vector<std::vector<float>> samples; // training set
	std::vector<int> labels; // ground truth labels
};

class LogisticRegression2 { // two categories
public:
	LogisticRegression2() = default;
	int init(std::unique_ptr<Database> data, int feature_length, float learning_rate = 0.00001, int epochs = 300);
	int train(const std::string& model);
	int load_model(const std::string& model);
	float predict(const float* data, int feature_length) const; // y = 1/(1+exp(-(wx+b)))

private:
	int store_model(const std::string& model) const;
	float calculate_z(const std::vector<float>& feature) const;  // z(i)=w^T*x(i)+b
	float calculate_cost_function() const;

	float calculate_activation_function(float value) const;
	float calculate_loss_function() const;
	float calculate_loss_function_derivative() const;
	float calculate_loss_function_derivative(unsigned int index) const;
	void calculate_gradient_descent();

	std::unique_ptr<Database> data_; // train data(images, labels)
	std::vector<float> o_; // predict value
	int epochs_ = 100; // epochs
	int m_ = 0; // train samples num
	int feature_length_ = 0; // weights length
	float alpha_ = 0.00001; // learning rate
	std::vector<float> w_; // weights
	float b_ = 0.; // threshold
	float error_ = 0.000001;

	ActivationFunction activation_func_ = ActivationFunction::Sigmoid;
	LossFunction loss_func_ = LossFunction::MSE;
	Optimzation optim_ = Optimzation::BGD;
}; // class LogisticRegression2

} // namespace ANN

#endif // FBC_SRC_NN_LOGISTIC_REGRESSION2_HPP_
