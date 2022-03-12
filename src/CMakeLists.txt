#Cmake testing

cmake_minimum_required (VERSION 3.8)
SET(pybind11_DIR E:/work/Clibs/vcpkg/packages/pybind11_x86-windows/share/pybind11)

project (onnxPybind)


find_package(pybind11 CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED )
pybind11_add_module(onnxPybind onnxPybind.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} "E:/work/pybind/build32/packages/Microsoft.ML.OnnxRuntime.1.10.0/build/native/include/")

target_link_libraries(onnxPybind PRIVATE ${OpenCV_DIR}/x86/vc17/lib/opencv_world455d.lib "E:/work/pybind/build32/packages/Microsoft.ML.OnnxRuntime.1.10.0/runtimes/win-x86/native/onnxruntime.lib")