# NN_Test #
**This project contains some neural network code(Note: depend on opencv/libtorch/onnxruntime):**
- neural network code(C++)
	- Perceptron
	- BP(Back Propagation)
	- CNN(Convolutional Neural Networks)
	- Linear Regression(gradient descent、least squares)
	- Naive Bayes Classifier(sex classification)
	- Logistic Regression(gradient descent, Batch/Mini Batch)
	- KNN(K-Nearest Neighbor, classification)
	- PCA(Principal Components Analysis)
	- single hidden layer(two categories)
	- K-Means
	- Decision Tree(CART)
	- YOLOv8(OpenCV DNN, libtorch, onnxruntime)
- mathematical formula's implement with C++
	- Linear Algebra
		- transpose
		- determinant
		- adjoint matrix
		- inverse matrix
		- norm(vector、matrix)
		- eigenvalues/eigenvectors(real symmetric matrix)
		- SVD(Singular Value Decomposition)
		- pseudoinverse
		- trace
	- Probability
		- mean、variance、standard deviation
		- covariance matrix
	- Activation Function
		- logistic sigmoid function
		- softplus function
		- ReLU(rectified linear unit)
		- Leaky ReLUs(leaky rectified linear units)
		- ELUs(Exponential linear units)
		- softmax function
		- tanh(Hyperbolic tangent)
	- Loss Function
		- MSE(Mean Square Error)
		- Cross-Entropy
	- Optimization
		- BGD(Batch Gradient Descent)
		- SGD(Stochastic Gradient Descent)
		- MBGD(Mini-batch Gradient Descent)
		- SGD with Momentum
		- AdaGrad(Adaptive Gradient)
		- RMSProp(Root Mean Square Propagation)
		- Adadelta(an adaptive learning rate method)
		- Adam(Adaptive Moment Estimation)
		- AdaMax(a variant of Adam based on the infinity norm)
		- NAG(Nesterov Accelerated Gradient)
		- Nadam(Nesterov-accelerated Adaptive Moment Estimation)
	- Layers
		- Dropout
		- LRN(Local Response Normalization)
		- BN(Batch Normalization)
	- common algorithm
		- Brute Force(string match)
		- Triangle similarity(monocular ranging)
- open source library's usage(C++)
	- tiny-cnn [GitHub](https://github.com/nyanp/tiny-cnn)
	- tiny-dnn [GitHub](https://github.com/tiny-dnn/tiny-dnn)
	- libsvm [GitHub](https://github.com/cjlin1/libsvm)
	- OpenCV [GitHub](https://github.com/opencv/opencv)
		- Logistic Regression
		- Support Vector Machines(SVM)
		- KNN(K-Nearest Neighbor)
		- Decision Tree
		- PCA(Principal Components Analysis)
		- k-means
- Python code(support windows and linux platform)
	- Universal Configuration
		- Installation(Anaconda)
			```
			conda create -n NN_Test python=3.9
			conda activate NN_Test
			pip install -r requirements.txt
			git clone https://github.com/fengbingchun/NN_Test
			cd NN_Test/demo/Python
			```
		- Code
			- [KNN(K-Nearest Neighbor)](demo/Python/test_knn_classification.py)
			- [Decision Tree(CART)](demo/Python/test_decision_tree_classification.py)
			- [PCA(Principal Components Analysis)](demo/Python/test_pca.py)
			- k-means
				- [k-means](demo/Python/test_k-means.py)
				- [k-menas opencv](demo/Python/test_k-means_opencv.py)
			- [txt to csv](demo/Python/test_txt_to_csv_convert.py)
	- MediaPipe
		- Installation(Anaconda)
			```
			conda create -n MediaPipe python=3.9
			conda activate MediaPipe
			pip install mediapipe
			pip install requests
			git clone https://github.com/fengbingchun/NN_Test
			cd NN_Test/demo/Python
			```
		- Code
			- [face monocular ranging](demo/Python/test_mediapipe_midas_face_monocular_ranging.py)
	- Ultralytics
		- Installation(Anaconda)
			```
			# cpu
			conda create --name ultralytics-env python=3.8 -y
			conda activate ultralytics-env
			pip install ultralytics

			# cuda
			# install cuda 11.8
			# install cudnn v8.9.7
			conda create --name ultralytics-env-cuda python=3.8 -y
			conda activate ultralytics-env-cuda
			conda install -c pytorch -c nvidia -c conda-forge pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 ultralytics # pytorch 2.2.2

			git clone https://github.com/fengbingchun/NN_Test
			cd NN_Test/demo/Python
			```
		- Code
			- [LabelMe JSON to YOLOv8 TXT: Detect](demo/Python/test_labelme2yolov8_detect.py)
			- [EISeg JSON to YOLOV8 TXT: Segment](demo/Python/test_eiseg2yolov8_segment.py)
			- [YOLOv8 train](demo/Python/test_yolov8_train.py)
			- [YOLOv8 predict](demo/Python/test_yolov8_predict.py)
			- [YOLOv8 segment onnxruntime predict](demo/Python/test_yolov8_segment_onnxruntime_predict.py)
- Database convert to Images or Images conver to Database
	- MNIST
	- CIFAR10/CIFAR100
	- ORL Faces Database

**The project support platform:**
- windows10 64 bits: It can be directly build with VS2022 in windows10 64bits.
- Linux:
	- DatasetToImage support cmake build(file position: prj/linux_cmake_DatasetToImage)
	- Math support cmake build(file position: prj/linux_cmake_Math)
	- tiny-cnn support cmake build(file position: prj/linux_cmake_tiny-cnn)
	- tiny-dnn support cmake build(file position: prj/linux_cmake_tiny-dnn)
	- NN support cmake build(file position: prj/linux_cmake_NN)

**Windows VS Screenshot:**
![](prj/x86_x64_vc12/Screenshot.png)


**Blog:** [fengbingchun](http://blog.csdn.net/fengbingchun/article/category/780354)

