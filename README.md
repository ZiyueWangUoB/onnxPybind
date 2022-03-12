# onnxPybind

## Introduction
Onnxruntime doesn't include a 32bit windows runtime for python, which is problematic as many industry devices are still using 32bit windows. To circumvent this problem, this repository makes use of Pybind to create a dll which 32bit python can use for model inference. 

## Dependencies

* CMake 3.23.0-rc3
* ONNX Runtime 1.10.0
* OpenCV 4.5.5

Haven't tried earlier builds but they should work (apart from maybe onnx).

