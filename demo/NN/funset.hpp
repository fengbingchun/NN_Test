#ifndef FBC_TEST_NN_FUNSET_HPP_
#define FBC_TEST_NN_FUNSET_HPP_

int test_ollama_model_list();
int test_ollama_chat();
int test_ollama_chat_stream();
int test_ollama_generate();
int test_ollama_generate_stream();
int test_monocular_ranging_face_triangle_similarity();
int test_logistic_regression2_gradient_descent();
int test_batch_normalization(); // Batch Normalization
int test_lrn(); // Local Response Normalization
int test_kmeans();
int test_single_hidden_layer_train(); // two categories
int test_single_hidden_layer_predict();
int test_logistic_regression2_train();
int test_logistic_regression2_predict();
int test_pca();
int test_decision_tree_train(); // two categories
int test_decision_tree_predict();
int test_knn_classifier_predict();
int test_logistic_regression_train();
int test_logistic_regression_predict();
int test_perceptron();
int test_BP_train();
int test_BP_predict();
int test_CNN_train();
int test_CNN_predict();
int test_linear_regression_train();
int test_linear_regression_predict();
int test_naive_bayes_classifier_train();
int test_naive_bayes_classifier_predict();

#endif // FBC_TEST_NN_FUNSET_HPP_
