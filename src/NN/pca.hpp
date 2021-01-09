#ifndef FBC_NN_PCA_HPP_
#define FBC_NN_PCA_HPP_

// Blog: http://blog.csdn.net/fengbingchun/article/details/79235028

#include <vector>
#include <string>

namespace ANN {

template<typename T = float>
class PCA {
public:
	PCA() = default;
	int load_data(const std::vector<std::vector<T>>& data, const std::vector<T>& labels);
	int set_max_components(int max_components);
	int set_retained_variance(double retained_variance);
	int load_model(const std::string& model);
	int train(const std::string& model);
	// project into the eigenspace, thus the image becomes a "point"
	int project(const std::vector<T>& vec, std::vector<T>& result) const;
	// re-create the image from the "point"
	int back_project(const std::vector<T>& vec, std::vector<T>& result) const;

private:
	// width,height,eigen_vectors;width,height,eigen_values;width,height,means
	int save_model(const std::string& model) const;
	void calculate_covariance_matrix(std::vector<std::vector<T>>& covar, bool scale = false); // calculate covariance matrix
	int eigen(const std::vector<std::vector<T>>& mat, bool sort_ = true); // calculate eigen vectors and eigen values
	// generalized matrix multiplication: dst = alpha*src1.t()*src2 + beta*src3.t()
	int gemm(const std::vector<std::vector<T>>& src1, const std::vector<std::vector<T>>& src2, double alpha,
		const std::vector<std::vector<T>>& src3, double beta, std::vector<std::vector<T>>& dst, int flags = 0) const;
	int gemm(const std::vector<T>& src1, const std::vector<std::vector<T>>& src2, double alpha,
		const std::vector<T>& src3, double beta, std::vector<T>& dst, int flags = 0) const; // GEMM_2_T: flags = 1
	int normalize(T* dst, int length);
	int computeCumulativeEnergy() const;
	int subtract(const std::vector<T>& vec1, const std::vector<T>& vec2, std::vector<T>& result) const;

	typedef struct Size_ {
		int width;
		int height;
	} Size_;

	std::vector<std::vector<T>> data;
	std::vector<T> labels;
	int samples_num = 0;
	int features_length = 0;
	double retained_variance = -1.; // percentage of variance that PCA should retain
	int max_components = -1; // maximum number of components that PCA should retain
	std::vector<std::vector<T>> eigen_vectors; // eigenvectors of the covariation matrix
	std::vector<T> eigen_values; // eigenvalues of the covariation matrix
	std::vector<T> mean;
	int covar_flags = 0; // when features_length > samples_num, covar_flags is 0, otherwise is 1
};

} // namespace ANN

#endif // FBC_NN_PCA_HPP_
