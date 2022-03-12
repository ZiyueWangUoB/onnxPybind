// tryPybind.cpp: Ziyue Wang 2022
//

#include <pybind11/pybind11.h>
#include <iostream>
#include <pybind11/numpy.h>
#include <onnxruntime_cxx_api.h>
#include <pybind11/stl.h>
#include <Python.h>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
using namespace std;

//Minimum example

namespace py = pybind11;


class onnx_c{
#define CV_32F 5
private:
    Ort::Session _session;
    std::vector<const char*> _inputNames;
    std::vector<const char*> _outputNames;
    std::vector<int64_t> _inputDims;
    std::vector<int64_t> _outputDims;

public:

    //Member list, constructors
    onnx_c(Ort::Session,const char* inputName);
    onnx_c() : _session(loadOrt()),
        _inputNames(setInputNames()),
        _outputNames(setOutputNames()),
        _inputDims(setInputDims()),
        _outputDims(setOutputDims())
    {};
    
    Ort::Session loadOrt() {
        string instanceName = "ORTefficient";
        string modelPath = "efficientB3.onnx";

#ifdef _WIN32
        std::string str = modelPath;
        std::wstring wide_string = std::wstring(str.begin(), str.end());
        std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
        std::string model_file = modelPath;
#endif

        Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
            instanceName.c_str());
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(3);
        Ort::Session session = Ort::Session(env, model_file.c_str(), sessionOptions);  // access experimental components via the Experimental namespace};
        return session;
    }

    template <typename T>
    T vectorProduct(const std::vector<T>& v)
    {
        return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
    }

    std::vector<float> inference(string imagePath) {
        cv::Mat imageBGR = cv::imread(imagePath, cv::ImreadModes::IMREAD_COLOR);
        cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
        cv::resize(imageBGR, resizedImageBGR,
            cv::Size(_inputDims.at(2), _inputDims.at(3)),
            cv::InterpolationFlags::INTER_CUBIC);
        cv::cvtColor(resizedImageBGR, resizedImageRGB,
            cv::ColorConversionCodes::COLOR_BGR2RGB);
        resizedImageRGB.convertTo(resizedImage, CV_32F, 1);
        cv::Mat channels[3];
        cv::split(resizedImage, channels);
        cv::merge(channels, 3, resizedImage);
        // HWC to CHW
        cv::dnn::blobFromImage(resizedImage, preprocessedImage);

        size_t inputTensorSize = vectorProduct(_inputDims);
        std::vector<float> inputTensorValues(inputTensorSize);
        inputTensorValues.assign(preprocessedImage.begin<float>(),
            preprocessedImage.end<float>());

        size_t outputTensorSize = vectorProduct(_outputDims);
        std::vector<float> outputTensorValues(outputTensorSize);

        std::vector<Ort::Value> inputTensors;
        std::vector<Ort::Value> outputTensors;

        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        inputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorSize, _inputDims.data(),
            _inputDims.size()));
        outputTensors.push_back(Ort::Value::CreateTensor<float>(
            memoryInfo, outputTensorValues.data(), outputTensorSize,
            _outputDims.data(), _outputDims.size()));

        _session.Run(Ort::RunOptions{ nullptr }, _inputNames.data(),
            inputTensors.data(), 1, _outputNames.data(),
            outputTensors.data(), 1);

        return outputTensorValues;
    }

    std::vector<const char*> setInputNames() {
        Ort::AllocatorWithDefaultOptions allocator;
        const char* inputName = _session.GetInputName(0, allocator);
        std::vector<const char*> inputNames{ inputName };
        return inputNames;
    }

    std::vector<const char*> setOutputNames() {
        Ort::AllocatorWithDefaultOptions allocator;
        const char* outputName = _session.GetOutputName(0, allocator);
        std::vector<const char*> outputNames{ outputName };
        return outputNames;
    }

    std::vector<int64_t> setInputDims() {
        Ort::TypeInfo inputTypeInfo = _session.GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
        //Don't forget to fix batch size!
        inputDims[0] = 1;
        return inputDims;
    }

    std::vector<int64_t> setOutputDims() {
        Ort::TypeInfo outputTypeInfo = _session.GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
        outputDims[0] = 1;
        return outputDims;
    }

    std::vector<const char*> getInputNames() {
        return _inputNames;
    }

    std::vector<const char*> getOutputNames() {
        return _outputNames;
    }

    std::vector<int64_t> getInputDims() {
        return _inputDims;
    }

    std::vector<int64_t> getOutputDims() {
        return _outputDims;
    }

};



PYBIND11_MODULE(onnxPybind, m) {
    m.doc() = "onnx pybind module"; // optional module docstring

    py::class_<onnx_c>(m, "onnx_c")
        .def(py::init<>())
        .def("getInputNames",&onnx_c::getInputNames)
        .def("getOutputNames", &onnx_c::getOutputNames)
        .def("getInputDims", &onnx_c::getInputDims)
        .def("getOutputDims", &onnx_c::getOutputDims)
        .def("inference",&onnx_c::inference);

}

