#ifndef FBC_NN_TEST_YOLO_HPP_
#define FBC_NN_TEST_YOLO_HPP_

int test_yolo11_obb_opencv();
int test_yolo11_obb_onnxruntime();
int test_yolov8_obb_openvino();

int test_yolov8_classify_opencv();
int test_yolov8_classify_libtorch();
int test_yolov8_classify_onnxruntime();
int test_yolov8_classify_openvino();

int test_yolov8_detect_opencv();
int test_yolov8_detect_libtorch();
int test_yolov8_detect_onnxruntime();
int test_yolov8_detect_openvino();

int test_yolov8_segment_opencv();
int test_yolov8_segment_libtorch();
int test_yolov8_segment_onnxruntime();
int test_yolov8_segment_openvino();

#endif // FBC_NN_TEST_YOLO_HPP_
