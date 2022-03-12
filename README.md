# onnxPybind

## Introduction
Onnxruntime doesn't include a 32bit windows runtime for python, which is problematic as many industry devices are still using 32bit windows. To circumvent this problem, this repository makes use of Pybind to create a dll which 32bit python can use for model inference. 

## Dependencies

* CMake 3.23.0-rc3
* ONNX Runtime 1.10.0
* OpenCV 4.5.5

Haven't tried earlier builds but they should work (apart from maybe onnx).

## Configuration
Note, you will have to change the cmake to run on your computer. The paths will fail as I installed my packages in weird locations, meaning cmake couldn't find them. Also ORT was installed through Nuget, so the cmake will change depending on that. 

If you wish to not use cmake and just use Visual studio, however, you will need to manually add the PyBind, Onnxruntime, OpenCV libraries to the project. There are many guides on stackoverflow for this. Also remember to link 32-bit python to the project. Because of this, I heavily recommend modifying the cmake file for to your installed library directories, configuring and then building. Otherwise it will be a hassle.

## Usage

After building the DLL, you need to add Onnxruntime and OpenCV dlls to the same directory as the python extension file. 
The model name currently needs to be edited in the C code.
Then, in Python, usage is very simple. 

```
import onnxPybind

imagePath = "myImage.py"

outputs = ortSession.inference(imagePath)
```


## Possible problems

When compiling for Release, you need to change the linker dependency to opencv_world455.dll, not the debug version.

## TODO
- [ ] Model name through python
- [ ] Parsing image directly rather than reading from disk
- [ ] Optimize speeds
