#include "single_hidden_layer.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <memory>
#include <fstream>
#include "common.hpp"

namespace ANN {

template<typename T>
int SingleHiddenLayer<T>::init(const T* data, const T* labels, int train_num, int feature_length,
	int hidden_layer_node_num, T learning_rate, int iterations, int hidden_layer_activation_type, int output_layer_activation_type)
{
	CHECK(train_num > 2 && feature_length > 0 && hidden_layer_node_num > 0 && learning_rate > 0 && iterations > 0);
	CHECK(hidden_layer_activation_type >= 0 && hidden_layer_activation_type < 4);
	CHECK(output_layer_activation_type >= 0 && output_layer_activation_type < 4);

	this->hidden_layer_node_num = hidden_layer_node_num;
	this->alpha = learning_rate;
	this->iterations = iterations;
	this->hidden_layer_activation_type = static_cast<ActivationFunctionType>(hidden_layer_activation_type);
	this->output_layer_activation_type = static_cast<ActivationFunctionType>(output_layer_activation_type);
	this->m = train_num;
	this->feature_length = feature_length;

	this->x.resize(train_num);
	this->y.resize(train_num);

	for (int i = 0; i < train_num; ++i) {
		const T* p = data + i * feature_length;
		this->x[i].resize(feature_length);

		for (int j = 0; j < feature_length; ++j) {
			this->x[i][j] = p[j];
		}

		this->y[i] = labels[i];
	}

	return 0;
}

template<typename T>
void SingleHiddenLayer<T>::init_train_variable()
{
	J = (T)0.;

	dw1.resize(this->hidden_layer_node_num);
	db1.resize(this->hidden_layer_node_num);
	for (int i = 0; i < this->hidden_layer_node_num; ++i) {
		dw1[i].resize(this->feature_length);
		for (int j = 0; j < this->feature_length; ++j) {
			dw1[i][j] = (T)0.;
		}

		db1[i] = (T)0.;
	}

	dw2.resize(this->output_layer_node_num);
	db2.resize(this->output_layer_node_num);
	for (int i = 0; i < this->output_layer_node_num; ++i) {
		dw2[i].resize(this->hidden_layer_node_num);
		for (int j = 0; j < this->hidden_layer_node_num; ++j) {
			dw2[i][j] = (T)0.;
		}

		db2[i] = (T)0.;
	}

	z1.resize(this->m); a1.resize(this->m); da1.resize(this->m); dz1.resize(this->m);
	for (int i = 0; i < this->m; ++i) {
		z1[i].resize(this->hidden_layer_node_num);
		a1[i].resize(this->hidden_layer_node_num);
		dz1[i].resize(this->hidden_layer_node_num);
		da1[i].resize(this->hidden_layer_node_num);

		for (int j = 0; j < this->hidden_layer_node_num; ++j) {
			z1[i][j] = (T)0.;
			a1[i][j] = (T)0.;
			dz1[i][j] = (T)0.;
			da1[i][j] = (T)0.;
		}
	}

	z2.resize(this->m); a2.resize(this->m); da2.resize(this->m); dz2.resize(this->m);
	for (int i = 0; i < this->m; ++i) {
		z2[i].resize(this->output_layer_node_num);
		a2[i].resize(this->output_layer_node_num);
		dz2[i].resize(this->output_layer_node_num);
		da2[i].resize(this->output_layer_node_num);

		for (int j = 0; j < this->output_layer_node_num; ++j) {
			z2[i][j] = (T)0.;
			a2[i][j] = (T)0.;
			dz2[i][j] = (T)0.;
			da2[i][j] = (T)0.;
		}
	}
}

template<typename T>
void SingleHiddenLayer<T>::init_w_and_b()
{
	w1.resize(this->hidden_layer_node_num); // (hidden_layer_node_num, feature_length)
	b1.resize(this->hidden_layer_node_num); // (hidden_layer_node_num, 1)
	w2.resize(this->output_layer_node_num); // (output_layer_node_num, hidden_layer_node_num)
	b2.resize(this->output_layer_node_num); // (output_layer_node_num, 1)

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<T> distribution(-0.01, 0.01);

	for (int i = 0; i < this->hidden_layer_node_num; ++i) {
		w1[i].resize(this->feature_length);
		for (int j = 0; j < this->feature_length; ++j) {
			w1[i][j] = distribution(generator);
		}

		b1[i] = distribution(generator);
	}

	for (int i = 0; i < this->output_layer_node_num; ++i) {
		w2[i].resize(this->hidden_layer_node_num);
		for (int j = 0; j < this->hidden_layer_node_num; ++j) {
			w2[i][j] = distribution(generator);
		}

		b2[i] = distribution(generator);
	}
}

template<typename T>
int SingleHiddenLayer<T>::train(const std::string& model)
{
	CHECK(x.size() == y.size());
	CHECK(output_layer_node_num == 1);

	init_w_and_b();

	for (int iter = 0; iter < this->iterations; ++iter) {
		init_train_variable();

		for (int i = 0; i < this->m; ++i) {
			for (int p = 0; p < this->hidden_layer_node_num; ++p) {
				for (int q = 0; q < this->feature_length; ++q) {
					z1[i][p] += w1[p][q] * x[i][q];
				}

				z1[i][p] += b1[p]; // z[1](i)=w[1]*x(i)+b[1]
				a1[i][p] = calculate_activation_function(z1[i][p], this->hidden_layer_activation_type); // a[1](i)=g[1](z[1](i))
			}

			for (int p = 0; p < this->output_layer_node_num; ++p) {
				for (int q = 0; q < this->hidden_layer_node_num; ++q) {
					z2[i][p] += w2[p][q] * a1[i][q];
				}

				z2[i][p] += b2[p]; // z[2](i)=w[2]*a[1](i)+b[2]
				a2[i][p] = calculate_activation_function(z2[i][p], this->output_layer_activation_type); // a[2](i)=g[2](z[2](i))
			}

			for (int p = 0; p < this->output_layer_node_num; ++p) {
				J += -(y[i] * std::log(a2[i][p]) + (1 - y[i] * std::log(1 - a2[i][p]))); // J+=-[y(i)*loga[2](i)+(1-y(i))*log(1-a[2](i))]
			}

			for (int p = 0; p < this->output_layer_node_num; ++p) {
				da2[i][p] = -(y[i] / a2[i][p]) + ((1. - y[i]) / (1. - a2[i][p])); // da[2](i)=-(y(i)/a[2](i))+((1-y(i))/(1.-a[2](i)))
				dz2[i][p] = da2[i][p] * calcuate_activation_function_derivative(z2[i][p], this->output_layer_activation_type); // dz[2](i)=da[2](i)*g[2]'(z[2](i))
			}

			for (int p = 0; p < this->output_layer_node_num; ++p) {
				for (int q = 0; q < this->hidden_layer_node_num; ++q) {
					dw2[p][q] += dz2[i][p] * a1[i][q]; // dw[2]+=dz[2](i)*(a[1](i)^T)
				}

				db2[p] += dz2[i][p]; // db[2]+=dz[2](i)
			}

			for (int p = 0; p < this->hidden_layer_node_num; ++p) {
				for (int q = 0; q < this->output_layer_node_num; ++q) {
					da1[i][p] = w2[q][p] * dz2[i][q]; // (da[1](i)=w[2](i)^T)*dz[2](i)
					dz1[i][p] = da1[i][p] * calcuate_activation_function_derivative(z1[i][p], this->hidden_layer_activation_type); // dz[1](i)=da[1](i)*(g[1]'(z[1](i)))
				}
			}

			for (int p = 0; p < this->hidden_layer_node_num; ++p) {
				for (int q = 0; q < this->feature_length; ++q) {
					dw1[p][q] += dz1[i][p] * x[i][q]; // dw[1]+=dz[1](i)*(x(i)^T)
				}
				db1[p] += dz1[i][p]; // db[1]+=dz[1](i)
			}
		}

		J /= m;

		for (int p = 0; p < this->output_layer_node_num; ++p) {
			for (int q = 0; q < this->hidden_layer_node_num; ++q) {
				dw2[p][q] = dw2[p][q] / m; // dw[2] /=m
			}

			db2[p] = db2[p] / m; // db[2] /=m
		}

		for (int p = 0; p < this->hidden_layer_node_num; ++p) {
			for (int q = 0; q < this->feature_length; ++q) {
				dw1[p][q] = dw1[p][q] / m; // dw[1] /= m
			}

			db1[p] = db1[p] / m; // db[1] /= m
		}

		for (int p = 0; p < this->output_layer_node_num; ++p) {
			for (int q = 0; q < this->hidden_layer_node_num; ++q) {
				w2[p][q] = w2[p][q] - this->alpha * dw2[p][q]; // w[2]=w[2]-alpha*dw[2]
			}

			b2[p] = b2[p] - this->alpha * db2[p]; // b[2]=b[2]-alpha*db[2]
		}

		for (int p = 0; p < this->hidden_layer_node_num; ++p) {
			for (int q = 0; q < this->feature_length; ++q) {
				w1[p][q] = w1[p][q] - this->alpha * dw1[p][q]; // w[1]=w[1]-alpha*dw[1]
			}

			b1[p] = b1[p] - this->alpha * db1[p]; // b[1]=b[1]-alpha*db[1]
		}
	}

	CHECK(store_model(model) == 0);
}

template<typename T>
int SingleHiddenLayer<T>::load_model(const std::string& model)
{
	std::ifstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	file.read((char*)&this->hidden_layer_node_num, sizeof(int));
	file.read((char*)&this->output_layer_node_num, sizeof(int));
	int type{ -1 };
	file.read((char*)&type, sizeof(int));
	this->hidden_layer_activation_type = static_cast<ActivationFunctionType>(type);
	file.read((char*)&type, sizeof(int));
	this->output_layer_activation_type = static_cast<ActivationFunctionType>(type);
	file.read((char*)&this->feature_length, sizeof(int));

	this->w1.resize(this->hidden_layer_node_num);
	for (int i = 0; i < this->hidden_layer_node_num; ++i) {
		this->w1[i].resize(this->feature_length);
	}
	this->b1.resize(this->hidden_layer_node_num);

	this->w2.resize(this->output_layer_node_num);
	for (int i = 0; i < this->output_layer_node_num; ++i) {
		this->w2[i].resize(this->hidden_layer_node_num);
	}
	this->b2.resize(this->output_layer_node_num);

	int length = w1.size() * w1[0].size();
	std::unique_ptr<T[]> data1(new T[length]);
	T* p = data1.get();
	file.read((char*)p, sizeof(T)* length);
	file.read((char*)this->b1.data(), sizeof(T)* b1.size());

	int count{ 0 };
	for (int i = 0; i < this->w1.size(); ++i) {
		for (int j = 0; j < this->w1[0].size(); ++j) {
			w1[i][j] = p[count++];
		}
	}

	length = w2.size() * w2[0].size();
	std::unique_ptr<T[]> data2(new T[length]);
	p = data2.get();
	file.read((char*)p, sizeof(T)* length);
	file.read((char*)this->b2.data(), sizeof(T)* b2.size());

	count = 0;
	for (int i = 0; i < this->w2.size(); ++i) {
		for (int j = 0; j < this->w2[0].size(); ++j) {
			w2[i][j] = p[count++];
		}
	}

	file.close();

	return 0;
}

template<typename T>
T SingleHiddenLayer<T>::predict(const T* data, int feature_length) const
{
	CHECK(feature_length == this->feature_length);
	CHECK(this->output_layer_node_num == 1);
	CHECK(this->hidden_layer_activation_type >= 0 && this->hidden_layer_activation_type < 4);
	CHECK(this->output_layer_activation_type >= 0 && this->output_layer_activation_type < 4);

	std::vector<T> z1(this->hidden_layer_node_num, (T)0.), a1(this->hidden_layer_node_num, (T)0.),
		z2(this->output_layer_node_num, (T)0.), a2(this->output_layer_node_num, (T)0.);

	for (int p = 0; p < this->hidden_layer_node_num; ++p) {
		for (int q = 0; q < this->feature_length; ++q) {
			z1[p] += w1[p][q] * data[q];
		}

		z1[p] += b1[p];
		a1[p] = calculate_activation_function(z1[p], this->hidden_layer_activation_type);
	}

	for (int p = 0; p < this->output_layer_node_num; ++p) {
		for (int q = 0; q < this->hidden_layer_node_num; ++q) {
			z2[p] += w2[p][q] * a1[q];
		}

		z2[p] += b2[p];
		a2[p] = calculate_activation_function(z2[p], this->output_layer_activation_type);
	}

	return a2[0];
}

template<typename T>
T SingleHiddenLayer<T>::calculate_activation_function(T value, ActivationFunctionType type) const
{
	T result{ 0 };

	switch (type) {
	case Sigmoid:
		result = (T)1. / ((T)1. + std::exp(-value));
		break;
	case TanH:
		result = (T)(std::exp(value) - std::exp(-value)) / (std::exp(value) + std::exp(-value));
		break;
	case ReLU:
		result = std::max((T)0., value);
		break;
	case Leaky_ReLU:
		result = std::max((T)0.01*value, value);
		break;
	default:
		CHECK(0);
		break;
	}

	return result;
}

template<typename T>
T SingleHiddenLayer<T>::calcuate_activation_function_derivative(T value, ActivationFunctionType type) const
{
	T result{ 0 };

	switch (type) {
	case Sigmoid: {
		T tmp = calculate_activation_function(value, Sigmoid);
		result = tmp * (1. - tmp);
	}
		break;
	case TanH: {
		T tmp = calculate_activation_function(value, TanH);
		result = 1 - tmp * tmp;
	}
		break;
	case ReLU:
		result = value < 0. ? 0. : 1.;
		break;
	case Leaky_ReLU:
		result = value < 0. ? 0.01 : 1.;
		break;
	default:
		CHECK(0);
		break;
	}

	return result;
}

template<typename T>
int SingleHiddenLayer<T>::store_model(const std::string& model) const
{
	std::ofstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	file.write((char*)&this->hidden_layer_node_num, sizeof(int));
	file.write((char*)&this->output_layer_node_num, sizeof(int));
	int type = this->hidden_layer_activation_type;
	file.write((char*)&type, sizeof(int));
	type = this->output_layer_activation_type;
	file.write((char*)&type, sizeof(int));
	file.write((char*)&this->feature_length, sizeof(int));

	int length = w1.size() * w1[0].size();
	std::unique_ptr<T[]> data1(new T[length]);
	T* p = data1.get();
	for (int i = 0; i < w1.size(); ++i) {
		for (int j = 0; j < w1[0].size(); ++j) {
			p[i * w1[0].size() + j] = w1[i][j];
		}
	}
	file.write((char*)p, sizeof(T)* length);
	file.write((char*)this->b1.data(), sizeof(T)* this->b1.size());

	length = w2.size() * w2[0].size();
	std::unique_ptr<T[]> data2(new T[length]);
	p = data2.get();
	for (int i = 0; i < w2.size(); ++i) {
		for (int j = 0; j < w2[0].size(); ++j) {
			p[i * w2[0].size() + j] = w2[i][j];
		}
	}
	file.write((char*)p, sizeof(T)* length);
	file.write((char*)this->b2.data(), sizeof(T)* this->b2.size());

	file.close();

	return 0;
}

template class SingleHiddenLayer<float>;
template class SingleHiddenLayer<double>;

} // namespace ANN

