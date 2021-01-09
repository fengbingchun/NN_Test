#include "pca.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include <fstream>
#include "common.hpp"

namespace ANN {

template<typename T>
int PCA<T>::load_data(const std::vector<std::vector<T>>& data, const std::vector<T>& labels)
{
	this->samples_num = data.size();
	this->features_length = data[0].size();
	if (samples_num > features_length) {
		fprintf(stderr, "now only support samples_num <= features_length\n");
		return -1;
	}

	this->data.resize(this->samples_num);
	for (int i = 0; i < this->samples_num; ++i) {
		this->data[i].resize(this->features_length);
		memcpy(this->data[i].data(), data[i].data(), sizeof(T)* this->features_length);
	}

	this->labels.resize(this->samples_num);
	memcpy(this->labels.data(), labels.data(), sizeof(T)*this->samples_num);

	return 0;
}

template<typename T>
int PCA<T>::set_max_components(int max_components)
{
	CHECK(data.size() > 0);
	int count = std::min(features_length, samples_num);
	if (max_components > 0) {
		this->max_components = std::min(count, max_components);
	}
	this->retained_variance = -1.;
}

template<typename T>
int PCA<T>::set_retained_variance(double retained_variance)
{
	CHECK(retained_variance > 0 && retained_variance <= 1);
	this->retained_variance = retained_variance;
	this->max_components = -1;
}

template<typename T>
void PCA<T>::calculate_covariance_matrix(std::vector<std::vector<T>>& covar, bool scale)
{
	const int rows = samples_num;
	const int cols = features_length;
	const int nsamples = rows;
	double scale_ = 1.;
	if (scale) scale_ = 1. / (nsamples /*- 1*/);

	mean.resize(cols, (T)0.);
	for (int w = 0; w < cols; ++w) {
		for (int h = 0; h < rows; ++h) {
			mean[w] += data[h][w];
		}
	}

	for (auto& value : mean) {
		value = 1. / rows * value;
	}

	// int dsize = ata ? src.cols : src.rows; // ata = false;
	int dsize = rows;
	covar.resize(dsize);
	for (int i = 0; i < dsize; ++i) {
		covar[i].resize(dsize, (T)0.);
	}

	Size_ size{ data[0].size(), data.size() };

	T* tdst = covar[0].data();
	int delta_cols = mean.size();
	T delta_buf[4];
	int delta_shift = delta_cols == size.width ? 4 : 0;
	std::unique_ptr<T[]> buf(new T[size.width]);
	T* row_buf = buf.get();

	for (int i = 0; i < size.height; ++i) {
		const T* tsrc1 = data[i].data();
		const T* tdelta1 = mean.data();

		for (int k = 0; k < size.width; ++k) {
			row_buf[k] = tsrc1[k] - tdelta1[k];
		}

		for (int j = i; j < size.height; ++j) {
			double s = 0;
			const T* tsrc2 = data[j].data();
			const T* tdelta2 = mean.data();

			for (int k = 0; k < size.width; ++k) {
				s += (double)row_buf[k] * (tsrc2[k] - tdelta2[k]);
			}

			tdst[j] = (T)(s * scale_);
		}

		if (i < covar.size()-1) {
			tdst = covar[i + 1].data();
		}
	}
}

namespace {

template<typename _Tp>
static inline _Tp hypot_(_Tp a, _Tp b)
{
	a = std::abs(a);
	b = std::abs(b);
	if (a > b) {
		b /= a;
		return a*std::sqrt(1 + b*b);
	}
	if (b > 0) {
		a /= b;
		return b*std::sqrt(1 + a*a);
	}
	return 0;
}

} // namespace

template<typename T>
int PCA<T>::eigen(const std::vector<std::vector<T>>& mat, bool sort_)
{
	using _Tp = T; // typedef T _Tp;
	auto n = mat.size();
	for (const auto& m : mat) {
		if (m.size() != n) {
			fprintf(stderr, "mat must be square and it should be a real symmetric matrix\n");
			return -1;
		}
	}

	eigen_values.resize(n, (T)0.);
	std::vector<T> V(n*n, (T)0.);
	for (int i = 0; i < n; ++i) {
		V[n * i + i] = (_Tp)1;
		eigen_values[i] = mat[i][i];
	}

	const _Tp eps = std::numeric_limits<_Tp>::epsilon();
	int maxIters{ (int)n * (int)n * 30 };
	_Tp mv{ (_Tp)0 };
	std::vector<int> indR(n, 0), indC(n, 0);
	std::vector<_Tp> A;
	for (int i = 0; i < n; ++i) {
		A.insert(A.begin() + i * n, mat[i].begin(), mat[i].end());
	}

	for (int k = 0; k < n; ++k) {
		int m, i;
		if (k < n - 1) {
			for (m = k + 1, mv = std::abs(A[n*k + m]), i = k + 2; i < n; i++) {
				_Tp val = std::abs(A[n*k + i]);
				if (mv < val)
					mv = val, m = i;
			}
			indR[k] = m;
		}
		if (k > 0) {
			for (m = 0, mv = std::abs(A[k]), i = 1; i < k; i++) {
				_Tp val = std::abs(A[n*i + k]);
				if (mv < val)
					mv = val, m = i;
			}
			indC[k] = m;
		}
	}

	if (n > 1) for (int iters = 0; iters < maxIters; iters++) {
		int k, i, m;
		// find index (k,l) of pivot p
		for (k = 0, mv = std::abs(A[indR[0]]), i = 1; i < n - 1; i++) {
			_Tp val = std::abs(A[n*i + indR[i]]);
			if (mv < val)
				mv = val, k = i;
		}
		int l = indR[k];
		for (i = 1; i < n; i++) {
			_Tp val = std::abs(A[n*indC[i] + i]);
			if (mv < val)
				mv = val, k = indC[i], l = i;
		}

		_Tp p = A[n*k + l];
		if (std::abs(p) <= eps)
			break;
		_Tp y = (_Tp)((eigen_values[l] - eigen_values[k])*0.5);
		_Tp t = std::abs(y) + hypot_(p, y);
		_Tp s = hypot_(p, t);
		_Tp c = t / s;
		s = p / s; t = (p / t)*p;
		if (y < 0)
			s = -s, t = -t;
		A[n*k + l] = 0;

		eigen_values[k] -= t;
		eigen_values[l] += t;

		_Tp a0, b0;

#undef rotate
#define rotate(v0, v1) a0 = v0, b0 = v1, v0 = a0*c - b0*s, v1 = a0*s + b0*c

		// rotate rows and columns k and l
		for (i = 0; i < k; i++)
			rotate(A[n*i + k], A[n*i + l]);
		for (i = k + 1; i < l; i++)
			rotate(A[n*k + i], A[n*i + l]);
		for (i = l + 1; i < n; i++)
			rotate(A[n*k + i], A[n*l + i]);

		// rotate eigenvectors
		for (i = 0; i < n; i++)
			rotate(V[n*k + i], V[n*l + i]);

#undef rotate

		for (int j = 0; j < 2; j++) {
			int idx = j == 0 ? k : l;
			if (idx < n - 1) {
				for (m = idx + 1, mv = std::abs(A[n*idx + m]), i = idx + 2; i < n; i++) {
					_Tp val = std::abs(A[n*idx + i]);
					if (mv < val)
						mv = val, m = i;
				}
				indR[idx] = m;
			}
			if (idx > 0) {
				for (m = 0, mv = std::abs(A[idx]), i = 1; i < idx; i++) {
					_Tp val = std::abs(A[n*i + idx]);
					if (mv < val)
						mv = val, m = i;
				}
				indC[idx] = m;
			}
		}
	}

	// sort eigenvalues & eigenvectors
	if (sort_) {
		for (int k = 0; k < n - 1; k++) {
			int m = k;
			for (int i = k + 1; i < n; i++) {
				if (eigen_values[m] < eigen_values[i])
					m = i;
			}
			if (k != m) {
				std::swap(eigen_values[m], eigen_values[k]);
				for (int i = 0; i < n; i++)
					std::swap(V[n*m + i], V[n*k + i]);
			}
		}
	}

	eigen_vectors.resize(n);
	for (int i = 0; i < n; ++i) {
		eigen_vectors[i].resize(n);
		eigen_vectors[i].assign(V.begin() + i * n, V.begin() + i * n + n);
	}

	return 0;
}

template<typename T>
int PCA<T>::gemm(const std::vector<std::vector<T>>& src1, const std::vector<std::vector<T>>& src2, double alpha,
	const std::vector<std::vector<T>>& src3, double beta, std::vector<std::vector<T>>& dst, int flags) const
{
	CHECK(flags == 0); // now only support flags = 0
	CHECK(typeid(T).name() == typeid(double).name() || typeid(T).name() == typeid(float).name()); // T' type can only be float or double
	CHECK(beta == 0. && src3.size() == 0);

	Size_ a_size{ src1[0].size(), src1.size() }, d_size{ src2[0].size(), a_size.height };
	int len{ (int)src2.size() };
	CHECK(a_size.height == len);
	CHECK(d_size.height == dst.size() && d_size.width == dst[0].size());

	for (int y = 0; y < d_size.height; ++y) {
		for (int x = 0; x < d_size.width; ++x) {
			dst[y][x] = 0.;
			for (int t = 0; t < d_size.height; ++t) {
				dst[y][x] += src1[y][t] * src2[t][x];
			}

			dst[y][x] *= alpha;
		}
	}

	return 0;
}
template<typename T>
int PCA<T>::gemm(const std::vector<T>& src1, const std::vector<std::vector<T>>& src2, double alpha,
	const std::vector<T>& src3, double beta, std::vector<T>& dst, int flags) const
{
	CHECK(flags == 0 || flags == 1); // when flags = 1, GEMM_2_T
	CHECK(typeid(T).name() == typeid(double).name() || typeid(T).name() == typeid(float).name()); // T' type can only be float or double

	Size_ a_size{ src1.size(), 1 }, d_size;
	int len = 0;

	switch (flags) {
	case 0:
		d_size = Size_{ static_cast<int>(src2[0].size()), a_size.height };
		len = src2.size();
		CHECK(a_size.width == len);
		break;
	case 1:
		d_size = Size_{ static_cast<int>(src2.size()), a_size.height };
		len = src2[0].size();
		CHECK(a_size.width == len);
		break;
	}

	if (!src3.empty()) {
		CHECK(src3.size() == d_size.width);
	}

	dst.resize(d_size.width);

	const T* src3_ = nullptr;
	std::vector<T> tmp(dst.size(), (T)0.);
	if (src3.empty()) {
		src3_ = tmp.data();
	} else {
		src3_ = src3.data();
	}

	if (src1.size() == src2.size()) {
		for (int i = 0; i < dst.size(); ++i) {
			dst[i] = (T)0.;
			for (int j = 0; j < src2.size(); ++j) {
				dst[i] += src1[j] * src2[j][i];
			}
			dst[i] *= alpha;
			dst[i] += beta * src3_[i];
		}
	} else {
		for (int i = 0; i < dst.size(); ++i) {
			dst[i] = (T)0.;
			for (int j = 0; j < src1.size(); ++j) {
				dst[i] += src1[j] * src2[i][j];
			}
			dst[i] *= alpha;
			dst[i] += beta * src3_[i];
		}
	}

	return 0;
}

template<typename T>
int PCA<T>::normalize(T* dst, int length)
{
	T s = (T)0., a = (T)1.;
	for (int i = 0; i < length; ++i) {
		s += dst[i] * dst[i];
	}

	s = std::sqrt(s);
	s = s > DBL_EPSILON ? a / s : 0.;

	for (int i = 0; i < length; ++i) {
		dst[i] *= s;
	}

	return 0;
}

template<typename T>
int PCA<T>::computeCumulativeEnergy() const
{
	std::vector<T> g(eigen_values.size(), (T)0.);
	for (int ig = 0; ig < eigen_values.size(); ++ig) {
		for (int im = 0; im <= ig; ++im) {
			g[ig] += eigen_values[im];
		}
	}

	int L{ 0 };
	for (L = 0; L < eigen_values.size(); ++L) {
		double energy = g[L] / g[eigen_values.size() - 1];
		if (energy > retained_variance) break;
	}

	L = std::max(2, L);

	return L;
}

template<typename T>
int PCA<T>::train(const std::string& model)
{
	CHECK(retained_variance > 0. || max_components > 0);
	int count = std::min(features_length, samples_num), out_count = count;
	if (max_components > 0) out_count = std::min(count, max_components);
	covar_flags = 0;
	if (features_length <= samples_num) covar_flags = 1;

	std::vector<std::vector<T>> covar(count); // covariance matrix
	calculate_covariance_matrix(covar, true);
	eigen(covar, true);

	std::vector<std::vector<T>> tmp_data(samples_num), evects1(count);
	for (int i = 0; i < samples_num; ++i) {
		tmp_data[i].resize(features_length);
		evects1[i].resize(features_length);

		for (int j = 0; j < features_length; ++j) {
			tmp_data[i][j] = data[i][j] - mean[j];
		}
	}

	gemm(eigen_vectors, tmp_data, 1., std::vector<std::vector<T>>(), 0., evects1, 0);

	eigen_vectors.resize(evects1.size());
	for (int i = 0; i < eigen_vectors.size(); ++i) {
		eigen_vectors[i].resize(evects1[i].size());
		memcpy(eigen_vectors[i].data(), evects1[i].data(), sizeof(T)* evects1[i].size());
	}

	// normalize all eigenvectors
	if (retained_variance > 0) {
		for (int i = 0; i < eigen_vectors.size(); ++i) {
			normalize(eigen_vectors[i].data(), eigen_vectors[i].size());
		}

		// compute the cumulative energy content for each eigenvector
		int L = computeCumulativeEnergy();
		eigen_values.resize(L);
		eigen_vectors.resize(L);
	} else {
		for (int i = 0; i < out_count; ++i) {
			normalize(eigen_vectors[i].data(), eigen_vectors[i].size());
		}

		if (count > out_count) {
			eigen_values.resize(out_count);
			eigen_vectors.resize(out_count);
		}
	}

	save_model(model);

	return 0;
}

template<typename T>
int PCA<T>::subtract(const std::vector<T>& vec1, const std::vector<T>& vec2, std::vector<T>& result) const
{
	CHECK(vec1.size() == vec2.size() && vec1.size() == result.size());

	for (int i = 0; i < vec1.size(); ++i) {
		result[i] = vec1[i] - vec2[i];
	}

	return 0;
}

template<typename T>
int PCA<T>::project(const std::vector<T>& vec, std::vector<T>& result) const
{
	CHECK(!mean.empty() && !eigen_vectors.empty() && mean.size() == vec.size());

	std::vector<T> tmp_data(mean.size());
	subtract(vec, mean, tmp_data);

	gemm(tmp_data, eigen_vectors, 1, std::vector<T>(), 0, result, 1);

	return 0;
}

template<typename T>
int PCA<T>::back_project(const std::vector<T>& vec, std::vector<T>& result) const
{
	CHECK(!mean.empty() && !eigen_vectors.empty() && eigen_vectors.size() == vec.size());
	gemm(vec, eigen_vectors, 1, mean, 1, result, 0);

	return 0;
}

template<typename T>
int PCA<T>::load_model(const std::string& model)
{
	std::ifstream file(model.c_str(), std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int width = 0, height = 0;
	file.read((char*)&width, sizeof(width) * 1);
	file.read((char*)&height, sizeof(height) * 1);
	std::unique_ptr<T[]> data(new T[width * height]);
	file.read((char*)data.get(), sizeof(T)* width * height);
	eigen_vectors.resize(height);
	for (int i = 0; i < height; ++i) {
		eigen_vectors[i].resize(width);
		T* p = data.get() + i * width;
		memcpy(eigen_vectors[i].data(), p, sizeof(T)* width);
	}
	
	file.read((char*)&width, sizeof(width));
	file.read((char*)&height, sizeof(height));
	CHECK(height == 1);
	eigen_values.resize(width);
	file.read((char*)eigen_values.data(), sizeof(T)* width * height);

	file.read((char*)&width, sizeof(width));
	file.read((char*)&height, sizeof(height));
	CHECK(height == 1);
	mean.resize(width);
	file.read((char*)mean.data(), sizeof(T)* width * height);

	file.close();

	return 0;
}

template<typename T>
int PCA<T>::save_model(const std::string& model) const
{
	std::ofstream file(model.c_str(), std::ios::out | std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int width = eigen_vectors[0].size(), height = eigen_vectors.size();
	std::unique_ptr<T[]> data(new T[width * height]);
	for (int i = 0; i < height; ++i) {
		T* p = data.get() + i * width;
		memcpy(p, eigen_vectors[i].data(), sizeof(T) * width);
	}
	file.write((char*)&width, sizeof(width));
	file.write((char*)&height, sizeof(height));
	file.write((char*)data.get(), sizeof(T)* width * height);

	width = eigen_values.size(), height = 1;
	file.write((char*)&width, sizeof(width));
	file.write((char*)&height, sizeof(height));
	file.write((char*)eigen_values.data(), sizeof(T)* width * height);

	width = mean.size(), height = 1;
	file.write((char*)&width, sizeof(width));
	file.write((char*)&height, sizeof(height));
	file.write((char*)mean.data(), sizeof(T)* width * height);

	file.close();
	return 0;
}

template class PCA<float>;
template class PCA<double>;

} // namespace ANN
