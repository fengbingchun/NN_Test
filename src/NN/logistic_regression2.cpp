#include "logistic_regression2.hpp"
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include "common.hpp"

namespace ANN {

template<typename T>
int LogisticRegression2<T>::init(const T* data, const T* labels, int train_num, int feature_length, T learning_rate, int epochs)
{
	if (train_num < 2) {
		fprintf(stderr, "logistic regression train samples num is too little: %d\n", train_num);
		return -1;
	}
	if (learning_rate <= 0) {
		fprintf(stderr, "learning rate must be greater 0: %f\n", learning_rate);
		return -1;
	}
	if (epochs <= 0) {
		fprintf(stderr, "number of epochs cannot be zero or a negative number: %d\n", epochs);
		return -1;
	}

	alpha_ = learning_rate;
	epochs_ = epochs;

	m_ = train_num;
	feature_length_ = feature_length;

	x_.resize(m_);
	y_.resize(m_);
	o_.resize(m_);

	for (int i = 0; i < m_; ++i) {
		const T* p = data + i * feature_length_;
		x_[i].resize(feature_length_);

		for (int j = 0; j < feature_length_; ++j) {
			x_[i][j] = p[j];
		}

		y_[i] = labels[i];
	}

	return 0;
}

template<typename T>
int LogisticRegression2<T>::train(const std::string& model)
{
	CHECK(x_.size() == y_.size());

	w_.resize(feature_length_, (T)0.);
	generator_real_random_number(w_.data(), feature_length_, (T)-0.01f, (T)0.01f, true);
	generator_real_random_number(&b_, 1, (T)-0.01f, (T)0.01f);

	for (int iter = 0; iter < epochs_; ++iter) {
		calculate_gradient_descent();
		fprintf(stdout, "echoch: %d, cost function: %f\n", iter, calculate_cost_function());
	}

	CHECK(store_model(model) == 0);
	return 0;
}

template<typename T>
int LogisticRegression2<T>::load_model(const std::string& model)
{
	std::ifstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length{ 0 };
	file.read((char*)&length, sizeof(length));
	w_.resize(length);
	feature_length_ = length;
	file.read((char*)w_.data(), sizeof(T)*w_.size());
	file.read((char*)&b_, sizeof(T));

	file.close();
	return 0;
}

template<typename T>
T LogisticRegression2<T>::predict(const T* data, int feature_length) const
{
	CHECK(feature_length == feature_length_);

	T value{ (T)0. };
	for (int t = 0; t < feature_length_; ++t) {
		value += data[t] * w_[t];
	}
	value += b_;

	return (calculate_activation_function(value));
}

template<typename T>
int LogisticRegression2<T>::store_model(const std::string& model) const
{
	std::ofstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length = w_.size();
	file.write((char*)&length, sizeof(length));
	file.write((char*)w_.data(), sizeof(T) * w_.size());
	file.write((char*)&b_, sizeof(T));

	file.close();
	return 0;
}

template<typename T>
T LogisticRegression2<T>::calculate_z(const std::vector<T>& feature) const
{
	T z{ 0. };
	for (int i = 0; i < feature_length_; ++i) {
		z += w_[i] * feature[i];
	}
	z += b_;

	return z;
}

template<typename T>
T LogisticRegression2<T>::calculate_cost_function() const
{
	/*// J+=-1/m([y(i)*loga(i)+(1-y(i))*log(1-a(i))])
	// Note: log0 is not defined
	T J{0.};
	for (int i = 0; i < m_; ++i)
		J += -(y_[i] * std::log(o_[i]) + (1 - y_[i]) * std::log(1 - o_[i]) );
	return J/m_;*/

	T J{0.};
	for (int i = 0; i < m_; ++i)
		J += 1./2*std::pow(y_[i] - o_[i], 2);
	return J/m_;
}

template<typename T>
T LogisticRegression2<T>::calculate_activation_function(T value) const
{
	switch (activation_func_) {
		case ActivationFunction::Sigmoid:
		default: // Sigmoid
			return ((T)1 / ((T)1 + std::exp(-value))); // y = 1/(1+exp(-value))
	}
}

template<typename T>
T LogisticRegression2<T>::calculate_loss_function() const
{
	switch (loss_func_) {
		case LossFunction::MSE:
		default: // MSE
			T value = 0.;
			for (int i = 0; i < m_; ++i) {
				value += 1/2.*std::pow(y_[i] - o_[i], 2);
			}
			return value/m_;
	}
}

template<typename T>
T LogisticRegression2<T>::calculate_loss_function_derivative() const
{
	switch (loss_func_) {
		case LossFunction::MSE:
		default: // MSE
			T value = 0.;
			for (int i = 0; i < m_; ++i) {
				value += o_[i] - y_[i];
			}
			return value/m_;
	}
}

template<typename T>
T LogisticRegression2<T>::calculate_loss_function_derivative(unsigned int index) const
{
	switch (loss_func_) {
		case LossFunction::MSE:
		default: // MSE
			return (o_[index] - y_[index]);
	}
}

template<typename T>
void LogisticRegression2<T>::calculate_gradient_descent()
{
	switch (optim_) {
		case Optimzation::BGD:
		default: // BGD
			T db = (T)0.;
			std::vector<T> dw(feature_length_, (T)0.), z(m_, (T)0), dz(m_, (T)0);

			for (int i = 0; i < m_; ++i) {
				z[i] = calculate_z(x_[i]);
				o_[i] = calculate_activation_function(z[i]);
				dz[i] = calculate_loss_function_derivative(i);

				for (int j = 0; j < feature_length_; ++j) {
					dw[j] += x_[i][j] * dz[i]; // dw(i)+=x(i)(j)*dz(i)
				}
				db += dz[i]; // db+=dz(i)
			}

			for (int j = 0; j < feature_length_; ++j) {
				dw[j] /= m_;
				w_[j] -= alpha_ * dw[j];
			}

			b_ -= alpha_*(db/m_);
	}
}

template class LogisticRegression2<float>;
template class LogisticRegression2<double>;

} // namespace ANN

