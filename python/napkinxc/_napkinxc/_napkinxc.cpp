/*
 Copyright (c) 2020-2021 by Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "args.h"
#include "read_data.h"
#include "measure.h"
#include "model.h"
#include "resources.h"
#include "types.h"
#include "version.h"

#include <thread>
#include <future>
#include <chrono>

using namespace std::chrono_literals;
namespace py = pybind11;


enum InputDataType {
    list,
    ndarray,
    csr_matrix
};


template<typename F> void runAsInterruptable(F func){
    std::atomic<bool> done(false);
    std::thread t([&]{
        func();
        done = true;
    });

    while(!done){
        std::this_thread::sleep_for(100ms);
        if (PyErr_CheckSignals() != 0) throw py::error_already_set();
    }

    t.join();
}

template<typename T> std::vector<T> pyListToVector(py::list const &list){
    std::vector<T> vector(list.size());
    for (size_t i = 0; i < list.size(); ++i) vector[i] = py::cast<T>(list[i]);
    return vector;
}

template<typename T> std::vector<T> pyDataToSparseVector(py::object input, InputDataType dataType) {
    std::vector<T> output;
    pyDataToSparseVector(output, input, dataType);
    return output;
}

template<typename T> void pyDataToSparseVector(std::vector<T>& output, py::object input, InputDataType dataType) {
    switch (dataType) {
        case ndarray: {
            py::array_t<double, py::array::c_style | py::array::forcecast> array(input); // TODO: test it with different strides
            for(int i = 0; i < array.size(); ++i)
                output.push_back({i, array.at(i)});
        }
        case list: {
            // Sparse vectors are expected to be list of (id, value) tuples
            py::list list(input);
            for (int i = 0; i < list.size(); ++i) {
                py::tuple pyTuple(list[i]);
                output.push_back({py::cast<int>(pyTuple[0]), py::cast<double>(pyTuple[1])});
            }
        }
        default:
            throw py::value_error("Unsupported data type for pyDataToSparseVector");
    }
}

template<typename T> py::array_t<T> dataToPyArray(T* ptr, std::vector<py::ssize_t> dims){
    std::vector<py::ssize_t> strides(dims);
    for(auto& s : strides) s = sizeof(T);
    return py::array(py::buffer_info(
        (void*)ptr,         // Pointer to data (nullptr -> ask NumPy to allocate!)
        sizeof(T),          // Size of one item
        py::format_descriptor<T>::value, // Buffer format
        dims.size(),        // How many dimensions?
        dims,               // Number of elements for each dimension */
        strides             // Strides for each dimension */
    ));
}


std::tuple<std::vector<std::vector<int>>, py::array_t<int>, py::array_t<int>, py::array_t<double>> loadLibSvmFile(std::string path){
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    Args args;
    args.input = path;
    args.processData = false;
    readData(labels, features, args);

    int rows = features.rows();
    int cells = features.cells();

    // For labels
    std::vector<std::vector<Label>> pyLabels(labels.rows());
    for(auto r = 0; r < labels.rows(); ++r){
        for(auto l = labels[r]; (*l) != -1; ++l) pyLabels[r].push_back(*l);
    }
    labels.clear();

    // For feature CSR-matrix
    int* indptr = new int[rows + 1];
    int* indices = new int[cells];
    double* data = new double[cells];

    int i = 0;
    for(auto r = 0; r < features.rows(); ++r){
        indptr[r] = i;
        if (!std::is_sorted(features[r], features[r] + features.size(r))) std::sort(features[r], features[r] + features.size(r));
        for(auto f = features[r]; (*f).index != -1; ++f){
            indices[i] = (*f).index;
            data[i] = (*f).value;
            ++i;
        }
    }
    indptr[rows] = cells;
    features.clear();

    auto pyIndptr = dataToPyArray(indptr, {rows + 1});
    auto pyIndices = dataToPyArray(indices, {cells});
    auto pyData = dataToPyArray(data, {cells});

    return std::make_tuple(pyLabels, pyIndptr, pyIndices, pyData);
}


class CPPModel {
public:
    CPPModel(){};

    void setArgs(const std::vector<std::string>& arg){
        args.parseArgs(arg);
    }

    void fitOnFile(std::string path){
        runAsInterruptable([&] {
            args.input = path;
            SRMatrix<Label> labels;
            SRMatrix<Feature> features;
            readData(labels, features, args);
            fitHelper(labels, features);
        });
    }

    void fit(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType){
        runAsInterruptable([&] {
            SRMatrix<Label> labels;
            SRMatrix<Feature> features;
            readFeatureMatrix(features, inputFeatures, (InputDataType) featuresDataType);
            readLabelsMatrix(labels, inputLabels, (InputDataType) labelsDataType);
            fitHelper(labels, features);
        });
    }

    void load(){
        args.loadFromFile(joinPath(args.output, "args.bin"));
        if(model == nullptr) model = Model::factory(args);
        if(!model->isLoaded()) model->load(args, args.output);
    }

    void unload(){
        if(model != nullptr && model->isLoaded()) model->unload();
    }

    void setThresholds(std::vector<double> thresholds){
        load();
        model->setThresholds(thresholds);
    }

    void setLabelsWeights(std::vector<double> weights){
        load();
        model->setLabelsWeights(weights);
    }

    std::vector<std::vector<int>> predict(py::object inputFeatures, int featuresDataType, int topK, double threshold){
        auto predWithProba = predictProba(inputFeatures, featuresDataType, topK, threshold);
        return dropProbaHelper(predWithProba);
    }

    std::vector<std::vector<std::pair<int, double>>> predictProba(py::object inputFeatures, int featuresDataType, int topK, double threshold){
        std::vector<std::vector<std::pair<int, double>>> pred;
        runAsInterruptable([&] {
            load();
            SRMatrix<Feature> features;
            readFeatureMatrix(features, inputFeatures, (InputDataType)featuresDataType);
            pred = predictHelper(features, topK, threshold);
        });

        return pred;
    }

    std::vector<double> ofo(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType) {
        std::vector<double> thresholds;
        runAsInterruptable([&] {
            load();
            SRMatrix<Label> labels;
            SRMatrix<Feature> features;
            readFeatureMatrix(features, inputFeatures, (InputDataType) featuresDataType);
            readLabelsMatrix(labels, inputLabels, (InputDataType) labelsDataType);
            args.printArgs("ofo");
            thresholds = model->ofo(features, labels, args);
        });

        return thresholds;
    }

    std::vector<std::vector<int>> predictForFile(std::string path, int topK, double threshold) {
        auto predWithProba = predictProbaForFile(path, topK, threshold);
        return dropProbaHelper(predWithProba);
    }

    std::vector<std::vector<std::pair<int, double>>> predictProbaForFile(std::string path, int topK, double threshold) {
        std::vector<std::vector<std::pair<int, double>>> pred;
        runAsInterruptable([&] {
            load();
            args.input = path;

            SRMatrix<Label> labels;
            SRMatrix<Feature> features;
            readData(labels, features, args);
            pred = predictHelper(features, topK, threshold);
        });

        return pred;
    }

    std::vector<std::pair<std::string, double>> test(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType,
                                                     int topK, double threshold, std::string measuresStr){
        std::vector<std::pair<std::string, double>> results;
        runAsInterruptable([&] {
            load();
            SRMatrix<Label> labels;
            SRMatrix<Feature> features;
            readFeatureMatrix(features, inputFeatures, (InputDataType) featuresDataType);
            readLabelsMatrix(labels, inputLabels, (InputDataType) labelsDataType);
            results = testHelper(labels, features, topK, threshold, measuresStr);
        });

        return results;
    }

    std::vector<std::pair<std::string, double>> testOnFile(std::string path, int topK, double threshold, std::string measuresStr){
        std::vector<std::pair<std::string, double>> results;
        runAsInterruptable([&] {
            load();
            args.input = path;

            SRMatrix<Label> labels;
            SRMatrix<Feature> features;
            readData(labels, features, args);
            results = testHelper(labels, features, topK, threshold, measuresStr);
        });

        return results;
    }

    double callPythonFunction(std::function<double(py::object)> pyFunc, py::object pyArg){
        return pyFunc(pyArg);
    }

    double callPythonObjectMethod(py::object pyObject, py::object pyArg, std::string methodName){
        return pyObject.attr(methodName.c_str())(pyArg).cast<double>();
    }

    bool testDataLoad(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType, std::string path, double eps){
        SRMatrix<Label> labelsFromPython;
        SRMatrix<Feature> featuresFromPython;
        readFeatureMatrix(featuresFromPython, inputFeatures, (InputDataType)featuresDataType);
        readLabelsMatrix(labelsFromPython, inputLabels, (InputDataType)labelsDataType);

        args.input = path;

        SRMatrix<Label> labelsFromFile;
        SRMatrix<Feature> featuresFromFile;
        readData(labelsFromFile, featuresFromFile, args);

        // Labels can be check by comparison
        if (labelsFromPython != labelsFromFile)
            return false;

        // Due to differences in precision features needs manual check
        if(featuresFromFile.cells() != featuresFromPython.cells()
            || featuresFromFile.cols() != featuresFromPython.cols()
            || featuresFromFile.rows() != featuresFromPython.rows())
            return false;

        for(int r = 0; r < featuresFromFile.rows(); ++r){
            for(int c = 0; c < featuresFromFile.size(r); ++c){
                if(featuresFromFile[r][c].index != featuresFromPython[r][c].index)
                    return false;

                if(std::fabs(featuresFromFile[r][c].value - featuresFromPython[r][c].value) > eps)
                    return false;
            }
        }

        return true;
    }

private:
    Args args;
    std::shared_ptr<Model> model;

    // Reads multiple items from a python object and inserts onto a SRMatrix<Label>
    void readLabelsMatrix(SRMatrix<Label>& output, py::object& input, InputDataType dataType) {
        if (dataType == list && py::isinstance<py::list>(input)) {
            py::list rows(input);
            for (size_t r = 0; r < rows.size(); ++r) {
                std::vector<Label> rLabels;

                // Multi-label data
                if (py::isinstance<py::list>(rows[r]) || py::isinstance<py::tuple>(rows[r])) {
                    py::list row(rows[r]);
                    std::vector<double> _rLabels = pyListToVector<double>(row);
                    rLabels = std::vector<Label>(_rLabels.begin(), _rLabels.end());
                }
                else //if (py::isinstance<py::float_>(rows[r]) || py::isinstance<py::int_>(rows[r]))
                    rLabels.push_back(static_cast<Label>(py::cast<double>(rows[r])));

                output.appendRow(rLabels);
            }
        } else
            throw py::value_error("Unsupported data type for Y, should be list of lists or tuples");
    }

    // Reads multiple items from a python object and inserts onto a SRMatrix<Feature>
    //SRMatrix<Feature> readFeatureMatrix(py::object input, InputDataType dataType) {
        //SRMatrix<Feature> output;
    void readFeatureMatrix(SRMatrix<Feature>& output, py::object& input, InputDataType dataType) {
        if (dataType == list && py::isinstance<py::list>(input)) {
            py::list data(input);
            std::vector<Feature> rFeatures;
            for (size_t i = 0; i < data.size(); ++i) {
                rFeatures.clear();
                prepareFeaturesVector(rFeatures, args.bias);

                pyDataToSparseVector<Feature>(rFeatures, data[i], list);

                processFeaturesVector(rFeatures, args.norm, args.hash, args.featuresThreshold);
                output.appendRow(rFeatures);
            }
        } else if (dataType == ndarray) {
            py::array_t<float, py::array::c_style | py::array::forcecast> data(input);
            auto buffer = data.request();
            if (buffer.ndim != 2) throw py::value_error("Data must be a 2d array");

            size_t rows = buffer.shape[0];
            size_t features = buffer.shape[1];

            // Read each row from the sparse matrix, and insert
            std::vector<Feature> rFeatures;
            rFeatures.reserve(features);
            for (size_t r = 0; r < rows; ++r) {
                const float* rData = data.data(r);
                rFeatures.clear();
                prepareFeaturesVector(rFeatures, args.bias);

                for (int f = 0; f < features; ++f)
                    rFeatures.push_back({f, rData[f]});

                processFeaturesVector(rFeatures, args.norm, args.hash, args.featuresThreshold);
                output.appendRow(rFeatures);
            }
        } else if (dataType == csr_matrix) {
            // Check for attributes
            if (!py::hasattr(input, "indptr")) {
                throw py::value_error("Expect csr_matrix CSR matrix");
            }

            // Try to interpret input data as a csr_matrix CSR matrix
            py::array_t<int> indptr(input.attr("indptr"));
            py::array_t<int> indices(input.attr("indices"));
            py::array_t<double> data(input.attr("data"));

            // Read each row from the sparse matrix, and insert
            std::vector<Feature> rFeatures;
            for (int rId = 0; rId < indptr.size() - 1; ++rId) {
                rFeatures.clear();
                prepareFeaturesVector(rFeatures, args.bias);

                for (int i = indptr.at(rId); i < indptr.at(rId + 1); ++i)
                    rFeatures.push_back({indices.at(i), data.at(i)});

                processFeaturesVector(rFeatures, args.norm, args.hash, args.featuresThreshold);
                output.appendRow(rFeatures);
            }
        } else
            throw py::value_error("Unsupported data type");
    }

    inline void fitHelper(SRMatrix<Label>& labels, SRMatrix<Feature>& features){
        args.printArgs("train");
        makeDir(args.output);
        args.saveToFile(joinPath(args.output, "args.bin"));

        // Create and train model (train function also saves model)
        model = Model::factory(args);
        model->train(labels, features, args, args.output);
    }

    inline std::vector<std::vector<std::pair<int, double>>> predictHelper(SRMatrix<Feature>& features, int topK, double threshold){
        args.printArgs("predict");

        // TODO: refactor this
        args.topK = topK;
        args.threshold = threshold;
        auto predictions = model->predictBatch(features, args);

        // This is only safe because it's struct with two fields casted to pair, don't do this with tuples!
        return reinterpret_cast<std::vector<std::vector<std::pair<int, double>>>&>(predictions);
    }

    inline std::vector<std::pair<std::string, double>> testHelper(SRMatrix<Label>& labels, SRMatrix<Feature>& features, int topK, double threshold, std::string measuresStr){
        args.printArgs("test");

        args.topK = topK;
        args.threshold = threshold;
        auto predictions = model->predictBatch(features, args);

        args.measures = measuresStr;
        auto measures = Measure::factory(args, model->outputSize());
        for (auto& m : measures) m->accumulate(labels, predictions);

        std::vector<std::pair<std::string, double>> results;
        for (auto& m : measures)
            results.push_back({m->getName(), m->value()});

        return results;
    }

    inline std::vector<std::vector<int>> dropProbaHelper(std::vector<std::vector<std::pair<int, double>>>& predWithProba){
        std::vector<std::vector<int>> pred;
        pred.reserve(predWithProba.size());
        for(const auto& p : predWithProba){
            pred.push_back(std::vector<int>());
            pred.back().reserve(p.size());
            for(const auto& pi : p) pred.back().push_back(pi.first);
        }
        return pred;
    }
};


PYBIND11_MODULE(_napkinxc, n) {
    n.doc() = "Python bindings for napkinXC C++ core";
    n.attr("__version__") = VERSION;

    n.def("_load_libsvm_file", &loadLibSvmFile);

    py::enum_<InputDataType>(n, "InputDataType")
    .value("list", list)
    .value("ndarray", ndarray)
    .value("csr_matrix", csr_matrix);

    py::class_<CPPModel>(n, "CPPModel")
    .def(py::init<>())
    .def("set_args", &CPPModel::setArgs)
    .def("fit", &CPPModel::fit)
    .def("fit_on_file", &CPPModel::fitOnFile)
    .def("load", &CPPModel::load)
    .def("unload", &CPPModel::unload)
    .def("set_thresholds", &CPPModel::setThresholds)
    .def("set_labels_weights", &CPPModel::setLabelsWeights)
    .def("predict", &CPPModel::predict)
    .def("predict_proba", &CPPModel::predictProba)
    .def("predict_for_file", &CPPModel::predictForFile)
    .def("predict_proba_for_file", &CPPModel::predictProbaForFile)
    .def("ofo", &CPPModel::ofo)
    .def("test", &CPPModel::test)
    .def("test_on_file", &CPPModel::testOnFile)
    //.def("call_python_function", &CPPModel::callPythonFunction)
    //.def("call_python_object_method", &CPPModel::callPythonObjectMethod)
    .def("test_data_load", &CPPModel::testDataLoad);

}
