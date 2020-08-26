/**
 * Copyright (c) 2020 by Marek Wydmuch
 * All rights reserved.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

    void testLoad(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType, std::string path){
        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        readFeatureMatrix(features, inputFeatures, (InputDataType)featuresDataType);
        readLabelsMatrix(labels, inputLabels, (InputDataType)labelsDataType);

        args.input = path;
        args.header = false;

        SRMatrix<Label> labels2;
        SRMatrix<Feature> features2;
        std::shared_ptr<DataReader> reader = DataReader::factory(args);
        reader->readData(labels2, features2, args);

        std::cout << "Labels equal: " << (labels == labels2) << "\n";
        std::cout << "Features equal: " << (features == features2) << "\n";
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
    .def("test_load", &CPPModel::testLoad)
    .def("set_args", &CPPModel::setArgs)
    .def("fit", &CPPModel::fit)
    .def("fit_from_file", &CPPModel::fitOnFile)
    .def("predict", &CPPModel::predict)
    .def("predict_for_file", &CPPModel::predictForFile)
    .def("predict_proba", &CPPModel::predictProba)
    .def("predict_proba_for_file", &CPPModel::predictProbaForFile);

}
