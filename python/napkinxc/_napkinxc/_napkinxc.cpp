/*
 Copyright (c) 2020 by Marek Wydmuch

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
#include "data_reader.h"
#include "measure.h"
#include "model.h"
#include "resources.h"
#include "types.h"
#include "version.h"

namespace py = pybind11;

enum InputDataType {
    list,
    ndarray,
    csr_matrix
};

class CPPModel {
public:
    CPPModel(){};

    void setArgs(const std::vector<std::string>& arg){
        args.parseArgs(arg);
    }

    void fitOnFile(std::string path){
        args.input = path;
        args.header = false;

        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        std::shared_ptr<DataReader> reader = DataReader::factory(args);
        reader->readData(labels, features, args);
        //reader->saveToFile(joinPath(args.output, "data_reader.bin"));

        fitHelper(labels, features);
    }

    void fit(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType){
        //SRMatrix<Label> labels = readFeatureMatrix(inputFeatures, featuresDataType);
        //SRMatrix<Feature> features = readLabelsMatrix(inputLabels, labelsDataType);

        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        readFeatureMatrix(features, inputFeatures, (InputDataType)featuresDataType);
        readLabelsMatrix(labels, inputLabels, (InputDataType)labelsDataType);

        fitHelper(labels, features);
    }

    std::vector<std::vector<int>> predict(py::object inputFeatures, int featuresDataType, int topK, double threshold){
        SRMatrix<Feature> features;
        readFeatureMatrix(features, inputFeatures, (InputDataType)featuresDataType);
        auto predWithProba = predictHelper(features, topK, threshold);
        return dropProbaHelper(predWithProba);
    }

    std::vector<std::vector<std::pair<int, double>>> predictProba(py::object inputFeatures, int featuresDataType, int topK, double threshold){
        SRMatrix<Feature> features;
        readFeatureMatrix(features, inputFeatures, (InputDataType)featuresDataType);
        return predictHelper(features, topK, threshold);
    }

    std::vector<std::vector<int>> predictForFile(std::string path, int topK, double threshold) {
        args.input = path;
        args.header = false;

        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        std::shared_ptr<DataReader> reader = DataReader::factory(args);
        reader->readData(labels, features, args);

        auto predWithProba = predictHelper(features, topK, threshold);
        return dropProbaHelper(predWithProba);
    }

    std::vector<std::vector<std::pair<int, double>>> predictProbaForFile(std::string path, int topK, double threshold) {
        args.input = path;
        args.header = false;

        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        std::shared_ptr<DataReader> reader = DataReader::factory(args);
        reader->readData(labels, features, args);

        return predictHelper(features, topK, threshold);
    }

    std::vector<std::pair<std::string, double>> test(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType,
                                                     int topK, double threshold, std::string measuresStr){
        auto predictionPairs = predictProba(inputFeatures, featuresDataType, topK, threshold);
        auto predictions = reinterpret_cast<std::vector<std::vector<Prediction>>&>(predictionPairs);

        SRMatrix<Label> labels;
        readLabelsMatrix(labels, inputLabels, (InputDataType)labelsDataType);

        args.measures = measuresStr;
        auto measures = Measure::factory(args, model->outputSize());
        for (auto& m : measures) m->accumulate(labels, predictions);

        std::vector<std::pair<std::string, double>> results;
        for (auto& m : measures)
            results.push_back({m->getName(), m->value()});

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
        args.header = true;

        SRMatrix<Label> labelsFromFile;
        SRMatrix<Feature> featuresFromFile;
        std::shared_ptr<DataReader> reader = DataReader::factory(args);
        reader->readData(labelsFromFile, featuresFromFile, args);

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

    template<typename T> std::vector<T> pyListToVector(py::list const &list){
        std::vector<T> vector(list.size());
        for (size_t i = 0; i < list.size(); ++i) vector[i] = py::cast<T>(list[i]);
        return vector;
    }

    template<typename T> std::vector<T> pyDataToSparseVector(py::object input, InputDataType dataType, int shift = 0) {
        std::vector<T> output;
        pyDataToSparseVector(output, input, dataType, shift);
        return output;
    }

    template<typename T> void pyDataToSparseVector(std::vector<T>& output, py::object input, InputDataType dataType, int shift = 0) {
        switch (dataType) {
            case ndarray: {
                py::array_t<double, py::array::c_style | py::array::forcecast> array(input);
                for(int i = 0; i < array.size(); ++i)
                    output.push_back({i + shift, array.at(i)});
            }
            case list: {
                // Sparse vectors are expected to be list of (id, value) tuples
                py::list list(input);
                for (int i = 0; i < list.size(); ++i) {
                    py::tuple pyTuple(list[i]);
                    output.push_back({py::cast<int>(pyTuple[0]) + shift, py::cast<double>(pyTuple[1])});
                }
            }
            default:
                throw py::value_error("Unsupported data type for pyDataToSparseVector");
        }
    }

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
            throw py::value_error("Unsupported data type");
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
                DataReader::prepareFeaturesVector(rFeatures, args.bias);

                pyDataToSparseVector<Feature>(rFeatures, data[i], list, 2);

                DataReader::processFeaturesVector(rFeatures, args.norm, args.hash, args.featuresThreshold);
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
                DataReader::prepareFeaturesVector(rFeatures, args.bias);

                for (int f = 0; f < features; ++f)
                    rFeatures.push_back({f, rData[f]});

                DataReader::processFeaturesVector(rFeatures, args.norm, args.hash, args.featuresThreshold);
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
                DataReader::prepareFeaturesVector(rFeatures, args.bias);

                for (int i = indptr.at(rId); i < indptr.at(rId + 1); ++i) {
                    rFeatures.push_back({indices.at(i) + 2, data.at(i)});
                }

                DataReader::processFeaturesVector(rFeatures, args.norm, args.hash, args.featuresThreshold);
                output.appendRow(rFeatures);
            }
        } else
            throw py::value_error("Unsupported data type");
    }

    inline void fitHelper(SRMatrix<Label>& labels, SRMatrix<Feature>& features){
        args.printArgs();
        makeDir(args.output);
        args.saveToFile(joinPath(args.output, "args.bin"));

        // Create and train model (train function also saves model)
        model = Model::factory(args);
        model->train(labels, features, args, args.output);
    }

    inline std::vector<std::vector<std::pair<int, double>>> predictHelper(SRMatrix<Feature>& features, int topK, double threshold){
        if(model == nullptr)
            //throw std::runtime_error("Model does not exist!");
            model = Model::factory(args);

        if(!model->isLoaded())
            model->load(args, args.output);

        // TODO: refactor this
        args.topK = topK;
        args.threshold = threshold;
        std::vector<std::vector<Prediction>> predictions = model->predictBatch(features, args);

        // This is only safe because it's struct with two fields casted to pair, don't do this with tuples!
        return reinterpret_cast<std::vector<std::vector<std::pair<int, double>>>&>(predictions);
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
    n.doc() = "Python Bindings for napkinXC C++ core";
    n.attr("__version__") = VERSION;

//    py::Py_Initialize();
//    py::PyEval_InitThreads();
//    py::init_ndarray();

    py::enum_<InputDataType>(n, "InputDataType")
    .value("list", list)
    .value("ndarray", ndarray)
    .value("csr_matrix", csr_matrix);

    py::class_<CPPModel>(n, "CPPModel")
    .def(py::init<>())
    .def("set_args", &CPPModel::setArgs)
    .def("fit", &CPPModel::fit)
    .def("fit_from_file", &CPPModel::fitOnFile)
    .def("predict", &CPPModel::predict)
    .def("predict_for_file", &CPPModel::predictForFile)
    .def("predict_proba", &CPPModel::predictProba)
    .def("predict_proba_for_file", &CPPModel::predictProbaForFile)
    .def("test", &CPPModel::test)
    .def("call_python_function", &CPPModel::callPythonFunction)
    .def("call_python_object_method", &CPPModel::callPythonObjectMethod)
    .def("test_data_load", &CPPModel::testDataLoad);

}
