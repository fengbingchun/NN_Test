#ifndef FBC_MATH_FUNSET_HPP_
#define FBC_MATH_FUNSET_HPP_

int test_loss_function(); // 损失函数
int test_dropout(); // 深度学习中的Dropout,在训练阶段防止过拟合
int test_brute_force_string_match(); // 通过brute force算法求解字符串匹配
int test_activation_function(); // 激活函数
int test_calcCovarMatrix(); // 计算协方差矩阵
int test_meanStdDev(); // 计算矩阵的均值、方差、标准差
int test_trace(); // 求矩阵的迹
int test_pseudoinverse(); // 求伪逆矩阵
int test_SVD(); // 矩阵的奇异值分解
int test_eigenvalues_eigenvectors(); // 雅克比(jacobi)方法求实对称矩阵的特征值和特征向量
int test_norm(); // 求范数
int test_inverse_matrix(); // 求逆矩阵
int test_matrix_transpose(); // 矩阵转置
int test_determinant(); // 计算行列式
int test_adjoint_matrix(); // 计算伴随矩阵

#endif // FBC_MATH_FUNSET_HPP_
