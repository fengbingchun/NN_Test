#include "logistic_regression2.hpp"
#include <fstream>
#include <algorithm>
#include <random>
#include <cmath>
#include "common.hpp"

namespace ANN {

int LogisticRegression2::init(std::unique_ptr<Database> data, int feature_length, float learning_rate, int epochs)
{
	CHECK(data->samples.size() == data->labels.size());
	m_ = data->samples.size();
	if (m_ < 2) {
		fprintf(stderr, "logistic regression train samples num is too little: %d\n", m_);
		return -1;
	}
	if (learning_rate <= 0) {
		fprintf(stderr, "learning rate must be greater 0: %f\n", learning_rate);
		return -1;
	}
	if (epochs < 1) {
		fprintf(stderr, "number of epochs cannot be zero or a negative number: %d\n", epochs);
		return -1;
	}

	alpha_ = learning_rate;
	epochs_ = epochs;
	feature_length_ = feature_length;
	data_ = std::move(data);
	o_.resize(m_);
	return 0;
}

int LogisticRegression2::train(const std::string& model)
{
	w_.resize(feature_length_, 0.);
	generator_real_random_number(w_.data(), feature_length_, -0.01f, 0.01f, true);
	generator_real_random_number(&b_, 1, -0.01f, 0.01f);
	//fprintf(stdout, "random number: %f, %f, %f, %f, %f, %f\n", w_[0], w_[1], w_[2], w_[3], w_[4], b_);

	if (optim_ == Optimization::BGD) {
		for (int iter = 0; iter < epochs_; ++iter) {
			calculate_gradient_descent();
			auto cost_value = calculate_cost_function();
			fprintf(stdout, "epochs: %d, cost function: %f\n", iter, cost_value);
			if (cost_value < error_) break;
		}
	} else {
		random_shuffle_.resize(data_->samples.size(), 0);
		for (int i = 0; i < data_->samples.size(); ++i)
			random_shuffle_[i] = i;

		float cost_value = 0.;
		for (int iter = 0; iter < epochs_; ++iter) {
			//std::srand(unsigned(std::time(0)));
			//std::random_shuffle(random_shuffle_.begin(), random_shuffle_.end(), generate_random); // 每次执行后random_shuffle_结果不同
			std::default_random_engine generator;
			std::shuffle(random_shuffle_.begin(), random_shuffle_.end(), generator); // 每次执行后random_shuffle_结果相同
			//fprintf(stdout, "random shuffle: %d, %d, %d, %d, %d\n", random_shuffle_[0], random_shuffle_[1], random_shuffle_[2], random_shuffle_[3], random_shuffle_[4]);

			int loop = (m_ + batch_size_ - 1) / batch_size_;
			for (int i = 0; i < loop; ++i) {
				int start = i * batch_size_;
				int end = start + batch_size_ > m_ ? m_ : start + batch_size_;
				calculate_gradient_descent(start, end);

				for (int i = 0; i < m_; ++i)
					o_[i] = calculate_activation_function(calculate_z(data_->samples[i]));

				cost_value = calculate_cost_function();
				//fprintf(stdout, "epochs: %d, loop: %d, cost function: %f\n", iter, i, cost_value);
				if (cost_value < error_) break;
			}
			if (cost_value < error_) break;
		}
	}

	CHECK(store_model(model) == 0);
	return 0;
}

int LogisticRegression2::load_model(const std::string& model)
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
	file.read((char*)w_.data(), sizeof(float)*w_.size());
	file.read((char*)&b_, sizeof(float));

	file.close();
	return 0;
}

float LogisticRegression2::predict(const float* data, int feature_length) const
{
	CHECK(feature_length == feature_length_);

	float value{0.};
	for (int t = 0; t < feature_length_; ++t) {
		value += data[t] * w_[t];
	}
	value += b_;

	return (calculate_activation_function(value));
}

int LogisticRegression2::store_model(const std::string& model) const
{
	std::ofstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length = w_.size();
	file.write((char*)&length, sizeof(length));
	file.write((char*)w_.data(), sizeof(float) * w_.size());
	file.write((char*)&b_, sizeof(float));

	file.close();
	return 0;
}

float LogisticRegression2::calculate_z(const std::vector<float>& feature) const
{
	float z{0.};
	for (int i = 0; i < feature_length_; ++i) {
		z += w_[i] * feature[i];
	}
	z += b_;

	return z;
}

float LogisticRegression2::calculate_cost_function() const
{
	/*// J+=-1/m([y(i)*loga(i)+(1-y(i))*log(1-a(i))])
	// Note: log0 is not defined
	float J{0.};
	for (int i = 0; i < m_; ++i)
		J += -(data_->labels[i] * std::log(o_[i]) + (1 - labels[i]) * std::log(1 - o_[i]) );
	return J/m_;*/

	float J{0.};
	for (int i = 0; i < m_; ++i)
		J += 1./2*std::pow(data_->labels[i] - o_[i], 2);
	return J/m_;
}

float LogisticRegression2::calculate_activation_function(float value) const
{
	switch (activation_func_) {
		case ActivationFunction::Sigmoid:
		default: // Sigmoid
			return (1. / (1. + std::exp(-value))); // y = 1/(1+exp(-value))
	}
}

float LogisticRegression2::calculate_loss_function() const
{
	switch (loss_func_) {
		case LossFunction::MSE:
		default: // MSE
			float value = 0.;
			for (int i = 0; i < m_; ++i) {
				value += 1/2.*std::pow(data_->labels[i] - o_[i], 2);
			}
			return value/m_;
	}
}

float LogisticRegression2::calculate_loss_function_derivative() const
{
	switch (loss_func_) {
		case LossFunction::MSE:
		default: // MSE
			float value = 0.;
			for (int i = 0; i < m_; ++i) {
				value += o_[i] - data_->labels[i];
			}
			return value/m_;
	}
}

float LogisticRegression2::calculate_loss_function_derivative(float predictive_value, float true_value) const
{
	switch (loss_func_) {
		case LossFunction::MSE:
		default: // MSE
			return (predictive_value - true_value);
	}
}

void LogisticRegression2::calculate_gradient_descent(int start, int end)
{
	switch (optim_) {
		case Optimization::Adadelta: {
			int len = end - start;
			std::vector<float> g(feature_length_, 0.), p(feature_length_, 0.);
			std::vector<float> z(len, 0.), dz(len, 0.);
			for (int i = start, x = 0; i < end; ++i, ++x) {
				z[x] = calculate_z(data_->samples[random_shuffle_[i]]);
				dz[x] = calculate_loss_function_derivative(calculate_activation_function(z[x]), data_->labels[random_shuffle_[i]]);

				for (int j = 0; j < feature_length_; ++j) {
					float dw = data_->samples[random_shuffle_[i]][j] * dz[x];
					g[j] = mu_ * g[j] + (1. - mu_) * (dw * dw); // formula 10

					float alpha = (eps_ + std::sqrt(p[j])) / (eps_ + std::sqrt(g[j]));
					float change = alpha * dw;
					p[j] = mu_ * p[j] +  (1. - mu_) * (change * change); // formula 15

					w_[j] = w_[j] - change;
				}

				b_ -= (eps_ * dz[x]);
			}
		}
			break;
		case Optimization::RMSProp: {
			int len = end - start;
			std::vector<float> g(feature_length_, 0.);
			std::vector<float> z(len, 0), dz(len, 0);
			for (int i = start, x = 0; i < end; ++i, ++x) {
				z[x] = calculate_z(data_->samples[random_shuffle_[i]]);
				dz[x] = calculate_loss_function_derivative(calculate_activation_function(z[x]), data_->labels[random_shuffle_[i]]);

				for (int j = 0; j < feature_length_; ++j) {
					float dw = data_->samples[random_shuffle_[i]][j] * dz[x];
					g[j] = mu_ * g[j] + (1. - mu_) * (dw * dw); // formula 18
					w_[j] = w_[j] - alpha_ * dw / (std::sqrt(g[j]) + eps_);
				}

				b_ -= (alpha_ * dz[x]);
			}
		}
			break;
		case Optimization::AdaGrad: {
			int len = end - start;
			std::vector<float> g(feature_length_, 0.);
			std::vector<float> z(len, 0), dz(len, 0);
			for (int i = start, x = 0; i < end; ++i, ++x) {
				z[x] = calculate_z(data_->samples[random_shuffle_[i]]);
				dz[x] = calculate_loss_function_derivative(calculate_activation_function(z[x]), data_->labels[random_shuffle_[i]]);

				for (int j = 0; j < feature_length_; ++j) {
					float dw = data_->samples[random_shuffle_[i]][j] * dz[x];
					g[j] += dw * dw;
					w_[j] = w_[j] - alpha_ * dw / (std::sqrt(g[j]) + eps_);
				}

				b_ -= (alpha_ * dz[x]);
			}
		}
			break;
		case Optimization::SGD_Momentum: {
			int len = end - start;
			std::vector<float> change(feature_length_, 0.);
			std::vector<float> z(len, 0), dz(len, 0);
			for (int i = start, x = 0; i < end; ++i, ++x) {
				z[x] = calculate_z(data_->samples[random_shuffle_[i]]);
				dz[x] = calculate_loss_function_derivative(calculate_activation_function(z[x]), data_->labels[random_shuffle_[i]]);

				for (int j = 0; j < feature_length_; ++j) {
					float new_change = mu_ * change[j] - alpha_ * (data_->samples[random_shuffle_[i]][j] * dz[x]);
					w_[j] += new_change;
					change[j] = new_change;
				}

				b_ -= (alpha_ * dz[x]);
			}
		}
			break;
		case Optimization::SGD:
		case Optimization::MBGD: {
			int len = end - start;
			std::vector<float> z(len, 0), dz(len, 0);
			for (int i = start, x = 0; i < end; ++i, ++x) {
				z[x] = calculate_z(data_->samples[random_shuffle_[i]]);
				dz[x] = calculate_loss_function_derivative(calculate_activation_function(z[x]), data_->labels[random_shuffle_[i]]);

				for (int j = 0; j < feature_length_; ++j) {
					w_[j] = w_[j] - alpha_ * (data_->samples[random_shuffle_[i]][j] * dz[x]);
				}

				b_ -= (alpha_ * dz[x]);
			}
		}
			break;
		case Optimization::BGD:
		default: // BGD
			std::vector<float> z(m_, 0), dz(m_, 0);
			float db = 0.;
			std::vector<float> dw(feature_length_, 0.);
			for (int i = 0; i < m_; ++i) {
				z[i] = calculate_z(data_->samples[i]);
				o_[i] = calculate_activation_function(z[i]);
				dz[i] = calculate_loss_function_derivative(o_[i], data_->labels[i]);

				for (int j = 0; j < feature_length_; ++j) {
					dw[j] += data_->samples[i][j] * dz[i]; // dw(i)+=x(i)(j)*dz(i)
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

} // namespace ANN

