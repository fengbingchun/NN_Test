#ifndef FBC_MATH_COMMON_HPP_
#define FBC_MATH_COMMON_HPP_

#include <time.h>
#include <cmath>
#include <vector>
#include <limits>
#include <string>
#include <tuple>
#include <random>
#include <memory>
#include <opencv2/opencv.hpp>

#define EXP 1.0e-5

namespace fbc {

// ========================= Loss Function: cross-entropy =====================
// Blog: https://blog.csdn.net/fengbingchun/article/details/122724181
template<typename _Tp> // y实际值; t预测值; m类别数
_Tp loss_function_cross_entropy(const _Tp* y, const _Tp* t, int m)
{
	_Tp loss = 0.;
	for (auto i = 0; i < m; ++i)
		loss += -y[i] * std::log(t[i]);

	return loss;
}

// ============================ Dropout ================================
// Blog: https://blog.csdn.net/fengbingchun/article/details/89286485
template<class T>
int dropout(const T* bottom, int width, int height,  T* top, float dropout_ratio = 0.5f)
{
	if (dropout_ratio <= 0.f || dropout_ratio >= 1.f) {
		fprintf(stderr, "Error: dropout_ratio's value should be: (0., 1.): %f\n", dropout_ratio);
		return -1;
	}

	std::random_device rd; std::mt19937 gen(rd());
	std::bernoulli_distribution d(1. - dropout_ratio);

	int size = height * width;
	std::unique_ptr<int[]> mask(new int[size]);
	for (int i = 0; i < size; ++i) {
		mask[i] = (int)d(gen);
	}

	float scale = 1. / (1. - dropout_ratio);
	for (int i = 0; i < size; ++i)	{
		top[i] = bottom[i] * mask[i] * scale;
	}

	return 0;
}

// ============================ Brute Force ================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/78496954
typedef std::tuple<int, int> brute_force_result; // <status, pos>

int brute_force(const std::string& str, const std::string& sub, brute_force_result& result)
{
	std::get<0>(result) = -1;
	std::get<1>(result) = -1;

	int length_str = str.length(), length_sub = sub.length();

	if (length_str < length_sub) return 0;

	for (int i = 0; i < length_str - length_sub + 1; ++i) {
		int count{ 0 };

		for (int j = 0; j < length_sub; ++j) {
			const char& c1 = str.at(i + count);
			const char& c2 = sub.at(j);

			if (c1 == c2) ++count;
			else break;
		}

		if (count == length_sub) {
			std::get<0>(result) = 0;
			std::get<1>(result) = i;
		}
	}

	return 0;
}

// ========================= Activation Function: tanh =====================
// Blog: https://blog.csdn.net/fengbingchun/article/details/119202855
template<typename _Tp>
int activation_function_tanh(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		_Tp ep = std::exp(src[i]);
		_Tp em = std::exp(-src[i]);

		dst[i] = (ep - em) / (ep + em);
	}

	return 0;
}

template<typename _Tp>
int activation_function_tanh_derivative(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)1. - src[i] * src[i];
	}

	return 0;
}

// ========================= Activation Function: softmax =====================
// Blog: http://blog.csdn.net/fengbingchun/article/details/75220591
template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
	const _Tp alpha = *std::max_element(src, src + length);
	_Tp denominator{ 0 };

	for (int i = 0; i < length; ++i) {
		dst[i] = std::exp(src[i] - alpha);
		denominator += dst[i];
	}

	for (int i = 0; i < length; ++i) {
		dst[i] /= denominator;
	}

	return 0;
}

template<typename _Tp>
int activation_function_softmax_derivative(const _Tp* src, _Tp* dst, int length)
{
	std::vector<_Tp> y(length, (_Tp)0);
	activation_function_softmax(src, y.data(), length);

	fprintf(stderr, "Error: activation_function_softmax_derivative to do ...\n");
	return -1;
}

// ========================= Activation Function: ELUs ========================
// Blog: http://blog.csdn.net/fengbingchun/article/details/73872828
template<typename _Tp>
int activation_function_ELUs(const _Tp* src, _Tp* dst, int length, _Tp a = 1.)
{
	if (a < 0) {
		fprintf(stderr, "a is a hyper-parameter to be tuned and a>=0 is a constraint\n");
		return -1;
	}

	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] >= (_Tp)0. ? src[i] : (a * (exp(src[i]) - (_Tp)1.));
	}

	return -1;
}

template<typename _Tp>
int activation_function_ELUs_derivative()
{
	fprintf(stderr, "Error: activation_function_ELUs_derivative to do ...\n");
	return -1;
}

// ========================= Activation Function: Leaky_ReLUs =================
// Blog: http://blog.csdn.net/fengbingchun/article/details/73872828
template<typename _Tp>
int activation_function_Leaky_ReLUs(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] > (_Tp)0. ? src[i] : (_Tp)0.01 * src[i];
	}

	return 0;
}

template<typename _Tp>
int activation_function_Leaky_ReLUs_derivative(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] > (_Tp)0. ? (_Tp)1 : (_Tp)0.01;
	}

	return 0;
}

// ========================= Activation Function: ReLU =======================
// Blog: http://blog.csdn.net/fengbingchun/article/details/73872828
template<typename _Tp>
int activation_function_ReLU(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = std::max((_Tp)0., src[i]);
	}

	return 0;
}

template<typename _Tp>
int activation_function_ReLU_derivative(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = src[i] < (_Tp)0 ? (_Tp)0 : (_Tp)1;
	}

	return 0;
}

// ========================= Activation Function: softplus ===================
// Blog: http://blog.csdn.net/fengbingchun/article/details/73872828
template<typename _Tp>
int activation_function_softplus(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = log((_Tp)1. + exp(src[i])); // log1p(exp(src[i]))
	}

	return 0;
}

template<typename _Tp>
int activation_function_softplus_derivative(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(1. / (1. + exp(-src[i])));
	}

	return 0;
}

// ============================ Activation Function: sigmoid ================
// Blog: http://blog.csdn.net/fengbingchun/article/details/73848734
template<typename _Tp>
int activation_function_sigmoid(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(1. / (1. + exp(-src[i])));
	}

	return 0;
}

template<typename _Tp>
int activation_function_sigmoid_derivative(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(exp(-src[i]) / pow((1+exp(-src[i])), 2.f));
	}

	return 0;
}

template<typename _Tp>
int activation_function_sigmoid_fast(const _Tp* src, _Tp* dst, int length)
{
	for (int i = 0; i < length; ++i) {
		dst[i] = (_Tp)(src[i] / (1. + fabs(src[i])));
	}

	return 0;
}

// =============================== 计算协方差矩阵 ============================
// Blog: http://blog.csdn.net/fengbingchun/article/details/73558370
// 按行存储，以行为向量,输入矩阵mat为m行n列，则协方差矩阵covar为n行n列对称矩阵，均值mean为1行n列
template<typename _Tp>
int calcCovarMatrix(const std::vector<std::vector<_Tp>>& mat, std::vector<std::vector<_Tp>>& covar, std::vector<_Tp>& mean, bool scale = false)
{
	const int rows = mat.size();
	const int cols = mat[0].size();
	const int nsamples = rows;
	double scale_ = 1.;
	if (scale) scale_ = 1. / (nsamples /*- 1*/);

	covar.resize(cols);
	for (int i = 0; i < cols; ++i)
		covar[i].resize(cols, (_Tp)0);
	mean.resize(cols, (_Tp)0);

	for (int w = 0; w < cols; ++w) {
		for (int h = 0; h < rows; ++h) {
			mean[w] += mat[h][w];
		}
	}

	for (auto& value : mean) {
		value = 1. / rows * value;
	}

	for (int i = 0; i < cols; ++i) {
		std::vector<_Tp> col_buf(rows, (_Tp)0);
		for (int k = 0; k < rows; ++k)
			col_buf[k] = mat[k][i] - mean[i];

		for (int j = 0; j < cols; ++j) {
			double s0 = 0;
			for (int k = 0; k < rows; ++k) {
				s0 += col_buf[k] * (mat[k][j] - mean[j]);
			}
			covar[i][j] = (_Tp)(s0 * scale_);
		}
	}

	return 0;
}

// =============================== 计算均值、方差、标准差 =====================
// Blog: http://blog.csdn.net/fengbingchun/article/details/73323475
template<typename _Tp>
int meanStdDev(const std::vector<std::vector<_Tp>>& mat, double* mean, double* variance, double* stddev)
{
	int h = mat.size(), w = mat[0].size();
	double sum{ 0. }, sqsum{ 0. };

	for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			double v = static_cast<double>(mat[y][x]);
			sum += v;
			sqsum += v * v;
		}
	}

	double scale = 1. / (h * w);
	*mean = sum * scale;
	*variance = std::max(sqsum*scale - (*mean)*(*mean), 0.);
	*stddev = std::sqrt(*variance);

	return 0;
}

// ================================= 求矩阵的迹 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72895976
template<typename _Tp>
_Tp trace(const std::vector<std::vector<_Tp>>& mat)
{
	_Tp ret{ (_Tp)0 };
	int nm = std::min(mat.size(), mat[0].size());

	for (int i = 0; i < nm; ++i) {
		ret += mat[i][i];
	}

	return ret;
}

// ================================= 矩阵转置 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/71514010
template<typename _Tp>
int transpose(const std::vector<std::vector<_Tp>>& src, std::vector<std::vector<_Tp>>& dst)
{
	int m = src.size();
	int n = src[0].size();

	dst.resize(n);
	for (int i = 0; i < n; ++i) {
		dst[i].resize(m);
	}

	for (int y = 0; y < n; ++y) {
		for (int x = 0; x < m; ++x) {
			dst[y][x] = src[x][y];
		}
	}

	return 0;
}

// ================================= 矩阵奇异值分解 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72853757
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

template<typename _Tp>
static void JacobiSVD(std::vector<std::vector<_Tp>>& At,
	std::vector<std::vector<_Tp>>& _W, std::vector<std::vector<_Tp>>& Vt)
{
	double minval = FLT_MIN;
	_Tp eps = (_Tp)(FLT_EPSILON * 2);
	const int m = At[0].size();
	const int n = _W.size();
	const int n1 = m; // urows
	std::vector<double> W(n, 0.);

	for (int i = 0; i < n; i++) {
		double sd{ 0. };
		for (int k = 0; k < m; k++) {
			_Tp t = At[i][k];
			sd += (double)t*t;
		}
		W[i] = sd;

		for (int k = 0; k < n; k++)
			Vt[i][k] = 0;
		Vt[i][i] = 1;
	}

	int max_iter = std::max(m, 30);
	for (int iter = 0; iter < max_iter; iter++) {
		bool changed = false;
		_Tp c, s;

		for (int i = 0; i < n - 1; i++) {
			for (int j = i + 1; j < n; j++) {
				_Tp *Ai = At[i].data(), *Aj = At[j].data();
				double a = W[i], p = 0, b = W[j];

				for (int k = 0; k < m; k++)
					p += (double)Ai[k] * Aj[k];

				if (std::abs(p) <= eps * std::sqrt((double)a*b))
					continue;

				p *= 2;
				double beta = a - b, gamma = hypot_((double)p, beta);
				if (beta < 0) {
					double delta = (gamma - beta)*0.5;
					s = (_Tp)std::sqrt(delta / gamma);
					c = (_Tp)(p / (gamma*s * 2));
				}
				else {
					c = (_Tp)std::sqrt((gamma + beta) / (gamma * 2));
					s = (_Tp)(p / (gamma*c * 2));
				}

				a = b = 0;
				for (int k = 0; k < m; k++) {
					_Tp t0 = c*Ai[k] + s*Aj[k];
					_Tp t1 = -s*Ai[k] + c*Aj[k];
					Ai[k] = t0; Aj[k] = t1;

					a += (double)t0*t0; b += (double)t1*t1;
				}
				W[i] = a; W[j] = b;

				changed = true;

				_Tp *Vi = Vt[i].data(), *Vj = Vt[j].data();

				for (int k = 0; k < n; k++) {
					_Tp t0 = c*Vi[k] + s*Vj[k];
					_Tp t1 = -s*Vi[k] + c*Vj[k];
					Vi[k] = t0; Vj[k] = t1;
				}
			}
		}

		if (!changed)
			break;
	}

	for (int i = 0; i < n; i++) {
		double sd{ 0. };
		for (int k = 0; k < m; k++) {
			_Tp t = At[i][k];
			sd += (double)t*t;
		}
		W[i] = std::sqrt(sd);
	}

	for (int i = 0; i < n - 1; i++) {
		int j = i;
		for (int k = i + 1; k < n; k++) {
			if (W[j] < W[k])
				j = k;
		}
		if (i != j) {
			std::swap(W[i], W[j]);

			for (int k = 0; k < m; k++)
				std::swap(At[i][k], At[j][k]);

			for (int k = 0; k < n; k++)
				std::swap(Vt[i][k], Vt[j][k]);
		}
	}

	for (int i = 0; i < n; i++)
		_W[i][0] = (_Tp)W[i];

	srand(time(nullptr));

	for (int i = 0; i < n1; i++) {
		double sd = i < n ? W[i] : 0;

		for (int ii = 0; ii < 100 && sd <= minval; ii++) {
			// if we got a zero singular value, then in order to get the corresponding left singular vector
			// we generate a random vector, project it to the previously computed left singular vectors,
			// subtract the projection and normalize the difference.
			const _Tp val0 = (_Tp)(1. / m);
			for (int k = 0; k < m; k++) {
				unsigned int rng = rand() % 4294967295; // 2^32 - 1
				_Tp val = (rng & 256) != 0 ? val0 : -val0;
				At[i][k] = val;
			}
			for (int iter = 0; iter < 2; iter++) {
				for (int j = 0; j < i; j++) {
					sd = 0;
					for (int k = 0; k < m; k++)
						sd += At[i][k] * At[j][k];
					_Tp asum = 0;
					for (int k = 0; k < m; k++) {
						_Tp t = (_Tp)(At[i][k] - sd*At[j][k]);
						At[i][k] = t;
						asum += std::abs(t);
					}
					asum = asum > eps * 100 ? 1 / asum : 0;
					for (int k = 0; k < m; k++)
						At[i][k] *= asum;
				}
			}

			sd = 0;
			for (int k = 0; k < m; k++) {
				_Tp t = At[i][k];
				sd += (double)t*t;
			}
			sd = std::sqrt(sd);
		}

		_Tp s = (_Tp)(sd > minval ? 1 / sd : 0.);
		for (int k = 0; k < m; k++)
			At[i][k] *= s;
	}
}

// matSrc为原始矩阵，支持非方阵，matD存放奇异值，matU存放左奇异向量，matVt存放转置的右奇异向量
template<typename _Tp>
int svd(const std::vector<std::vector<_Tp>>& matSrc,
	std::vector<std::vector<_Tp>>& matD, std::vector<std::vector<_Tp>>& matU, std::vector<std::vector<_Tp>>& matVt)
{
	int m = matSrc.size();
	int n = matSrc[0].size();
	for (const auto& sz : matSrc) {
		if (n != sz.size()) {
			fprintf(stderr, "matrix dimension dismatch\n");
			return -1;
		}
	}

	bool at = false;
	if (m < n) {
		std::swap(m, n);
		at = true;
	}

	matD.resize(n);
	for (int i = 0; i < n; ++i) {
		matD[i].resize(1, (_Tp)0);
	}
	matU.resize(m);
	for (int i = 0; i < m; ++i) {
		matU[i].resize(m, (_Tp)0);
	}
	matVt.resize(n);
	for (int i = 0; i < n; ++i) {
		matVt[i].resize(n, (_Tp)0);
	}
	std::vector<std::vector<_Tp>> tmp_u = matU, tmp_v = matVt;

	std::vector<std::vector<_Tp>> tmp_a, tmp_a_;
	if (!at)
		transpose(matSrc, tmp_a);
	else
		tmp_a = matSrc;

	if (m == n) {
		tmp_a_ = tmp_a;
	}
	else {
		tmp_a_.resize(m);
		for (int i = 0; i < m; ++i) {
			tmp_a_[i].resize(m, (_Tp)0);
		}
		for (int i = 0; i < n; ++i) {
			tmp_a_[i].assign(tmp_a[i].begin(), tmp_a[i].end());
		}
	}
	JacobiSVD(tmp_a_, matD, tmp_v);

	if (!at) {
		transpose(tmp_a_, matU);
		matVt = tmp_v;
	}
	else {
		transpose(tmp_v, matU);
		matVt = tmp_a_;
	}

	return 0;
}

// ================================= 求伪逆矩阵 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72874623

template<typename _Tp> // mat1(m, n) * mat2(n, p) => result(m, p)
static std::vector<std::vector<_Tp>> matrix_mul(const std::vector<std::vector<_Tp>>& mat1, const std::vector<std::vector<_Tp>>& mat2)
{
	std::vector<std::vector<_Tp>> result;
	int m1 = mat1.size(), n1 = mat1[0].size();
	int m2 = mat2.size(), n2 = mat2[0].size();
	if (n1 != m2) {
		fprintf(stderr, "mat dimension dismatch\n");
		return result;
	}

	result.resize(m1);
	for (int i = 0; i < m1; ++i) {
		result[i].resize(n2, (_Tp)0);
	}

	for (int y = 0; y < m1; ++y) {
		for (int x = 0; x < n2; ++x) {
			for (int t = 0; t < n1; ++t) {
				result[y][x] += mat1[y][t] * mat2[t][x];
			}
		}
	}

	return result;
}

template<typename _Tp>
int pinv(const std::vector<std::vector<_Tp>>& src, std::vector<std::vector<_Tp>>& dst, _Tp tolerance)
{
	std::vector<std::vector<_Tp>> D, U, Vt;
	if (svd(src, D, U, Vt) != 0) {
		fprintf(stderr, "singular value decomposition fail\n");
		return -1;
	}

	int m = src.size();
	int n = src[0].size();

	std::vector<std::vector<_Tp>> Drecip, DrecipT, Ut, V;

	transpose(Vt, V);
	transpose(U, Ut);

	if (m < n)
		std::swap(m, n);

	Drecip.resize(n);
	for (int i = 0; i < n; ++i) {
		Drecip[i].resize(m, (_Tp)0);

		if (D[i][0] > tolerance)
			Drecip[i][i] = 1.0f / D[i][0];
	}

	if (src.size() < src[0].size())
		transpose(Drecip, DrecipT);
	else
		DrecipT = Drecip;

	std::vector<std::vector<_Tp>> tmp = matrix_mul(V, DrecipT);
	dst = matrix_mul(tmp, Ut);

	return 0;
}

// =============================== 求方阵的特征值和特征向量 ===============================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72801310
template<typename _Tp>
int eigen(const std::vector<std::vector<_Tp>>& mat, std::vector<_Tp>& eigenvalues, std::vector<std::vector<_Tp>>& eigenvectors, bool sort_ = true)
{
	auto n = mat.size();
	for (const auto& m : mat) {
		if (m.size() != n) {
			fprintf(stderr, "mat must be square and it should be a real symmetric matrix\n");
			return -1;
		}
	}

	eigenvalues.resize(n, (_Tp)0);
	std::vector<_Tp> V(n*n, (_Tp)0);
	for (int i = 0; i < n; ++i) {
		V[n * i + i] = (_Tp)1;
		eigenvalues[i] = mat[i][i];
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
		_Tp y = (_Tp)((eigenvalues[l] - eigenvalues[k])*0.5);
		_Tp t = std::abs(y) + hypot_(p, y);
		_Tp s = hypot_(p, t);
		_Tp c = t / s;
		s = p / s; t = (p / t)*p;
		if (y < 0)
			s = -s, t = -t;
		A[n*k + l] = 0;

		eigenvalues[k] -= t;
		eigenvalues[l] += t;

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
				if (eigenvalues[m] < eigenvalues[i])
					m = i;
			}
			if (k != m) {
				std::swap(eigenvalues[m], eigenvalues[k]);
				for (int i = 0; i < n; i++)
					std::swap(V[n*m + i], V[n*k + i]);
			}
		}
	}

	eigenvectors.resize(n);
	for (int i = 0; i < n; ++i) {
		eigenvectors[i].resize(n);
		eigenvectors[i].assign(V.begin() + i * n, V.begin() + i * n + n);
	}

	return 0;
}

// ================================= 求范数 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72636374
typedef enum Norm_Types_ {
	Norm_INT = 0, // 无穷大
	Norm_L1, // L1
	Norm_L2 // L2
} Norm_Types;

template<typename _Tp>
int norm(const std::vector<std::vector<_Tp>>& mat, int type, double* value)
{
	*value = 0.f;

	switch (type) {
	case Norm_INT: {
				for (int i = 0; i < mat.size(); ++i) {
					for (const auto& t : mat[i]) {
						*value = std::max(*value, (double)(fabs(t)));
					}
				}
	}
		break;
	case Norm_L1: {
				for (int i = 0; i < mat.size(); ++i) {
					for (const auto& t : mat[i]) {
						*value += (double)(fabs(t));
					}
				}
	}
		break;
	case Norm_L2: {
				for (int i = 0; i < mat.size(); ++i) {
					for (const auto& t : mat[i]) {
						*value += t * t;
					}
				}
				*value = std::sqrt(*value);
	}
		break;
	default: {
				fprintf(stderr, "norm type is not supported\n");
				return -1;
	}
	}

	return 0;
}

// ================================= 计算行列式 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72357082
template<typename _Tp>
_Tp determinant(const std::vector<std::vector<_Tp>>& mat, int N)
{
	if (mat.size() != N) {
		fprintf(stderr, "mat must be square matrix\n");
		return -1;
	}
	for (int i = 0; i < mat.size(); ++i) {
		if (mat[i].size() != N) {
			fprintf(stderr, "mat must be square matrix\n");
			return -1;
		}
	}

	_Tp ret{ 0 };

	if (N == 1) return mat[0][0];

	if (N == 2) {
		return (mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]);
	}
	else {
		// first col
		for (int i = 0; i < N; ++i) {
			std::vector<std::vector<_Tp>> m(N - 1);
			std::vector<int> m_rows;
			for (int t = 0; t < N; ++t) {
				if (i != t) m_rows.push_back(t);
			}
			for (int x = 0; x < N - 1; ++x) {
				m[x].resize(N - 1);
				for (int y = 0; y < N - 1; ++y) {
					m[x][y] = mat[m_rows[x]][y + 1];
				}
			}
			int sign = (int)pow(-1, 1 + i + 1);
			ret += mat[i][0] * sign * determinant<_Tp>(m, N - 1);
		}
	}

	return ret;
}

// ================================= 求伴随矩阵 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72518661
template<typename _Tp>
int adjoint(const std::vector<std::vector<_Tp>>& mat, std::vector<std::vector<_Tp>>& adj, int N)
{
	if (mat.size() != N) {
		fprintf(stderr, "mat must be square matrix\n");
		return -1;
	}
	for (int i = 0; i < mat.size(); ++i) {
		if (mat[i].size() != N) {
			fprintf(stderr, "mat must be square matrix\n");
			return -1;
		}
	}

	adj.resize(N);
	for (int i = 0; i < N; ++i) {
		adj[i].resize(N);
	}

	for (int y = 0; y < N; ++y) {
		std::vector<int> m_cols;
		for (int i = 0; i < N; ++i) {
			if (i != y) m_cols.push_back(i);
		}

		for (int x = 0; x < N; ++x) {
			std::vector<int> m_rows;
			for (int i = 0; i < N; ++i) {
				if (i != x) m_rows.push_back(i);
			}

			std::vector<std::vector<_Tp>> m(N - 1);
			for (int i = 0; i < N - 1; ++i) {
				m[i].resize(N - 1);
			}
			for (int j = 0; j < N - 1; ++j) {
				for (int i = 0; i < N - 1; ++i) {
					m[j][i] = mat[m_rows[j]][m_cols[i]];
				}
			}

			int sign = (int)pow(-1, x + y);
			adj[y][x] = sign * determinant<_Tp>(m, N - 1);
		}
	}

	return 0;
}

// ================================= 输出矩阵值 =================================
template<typename _Tp>
void print_matrix(const std::vector<std::vector<_Tp>>& mat)
{
	int rows = mat.size();
	for (int y = 0; y < rows; ++y) {
		for (int x = 0; x < mat[y].size(); ++x) {
			fprintf(stderr, "  %f  ", mat[y][x]);
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

template<typename _Tp>
void print_matrix(const std::vector<_Tp>& vec)
{
	for (int i = 0; i < vec.size(); ++i) {
		fprintf(stderr, "  %f  ", vec[i]);
	}
	fprintf(stderr, "\n");
}

void print_matrix(const cv::Mat& mat)
{
	assert(mat.channels() == 1);

	for (int y = 0; y < mat.rows; ++y) {
		for (int x = 0; x < mat.cols; ++x) {
			if (mat.depth() == CV_8U) {
				unsigned char value = mat.at<uchar>(y, x);
				fprintf(stderr, "  %d  ", value);
			}
			else if (mat.depth() == CV_32F) {
				float value = mat.at<float>(y, x);
				fprintf(stderr, "  %f  ", value);
			}
			else if (mat.depth() == CV_64F) {
				double value = mat.at<double>(y, x);
				fprintf(stderr, "  %f  ", value);
			}
			else {
				fprintf(stderr, "don't support type: %d\n", mat.depth());
				return;
			}
		}
		fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
}

// ================================= 求逆矩阵 =================================
// Blog: http://blog.csdn.net/fengbingchun/article/details/72571800
template<typename _Tp>
int inverse(const std::vector<std::vector<_Tp>>& mat, std::vector<std::vector<_Tp>>& inv, int N)
{
	if (mat.size() != N) {
		fprintf(stderr, "mat must be square matrix\n");
		return -1;
	}
	for (int i = 0; i < mat.size(); ++i) {
		if (mat[i].size() != N) {
			fprintf(stderr, "mat must be square matrix\n");
			return -1;
		}
	}

	_Tp det = determinant(mat, N);
	if (fabs(det) < EXP) {
		fprintf(stderr, "mat's determinant don't equal 0\n");
		return -1;
	}

	inv.resize(N);
	for (int i = 0; i < N; ++i) {
		inv[i].resize(N);
	}

	double coef = 1.f / det;
	std::vector<std::vector<_Tp>> adj;
	if (adjoint(mat, adj, N) != 0) return -1;

	for (int y = 0; y < N; ++y) {
		for (int x = 0; x < N; ++x) {
			inv[y][x] = (_Tp)(coef * adj[y][x]);
		}
	}

	return 0;
}


} // namespace fbc

#endif // FBC_MATH_COMMON_HPP_
