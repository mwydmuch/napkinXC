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
    denseData,
    sparseData,
    stringData,
};

//enum OutputDataType {
//
//};

class CPPModel {
public:
    //ModelWrapper(py::object args){
    CPPModel(){}

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

    std::vector<std::vector<std::pair<int, double>>> predict(py::object inputFeatures, int featuresDataType){
        SRMatrix<Feature> features;
        std::cout << "Reading features...\n";
        readFeatureMatrix(features, inputFeatures, (InputDataType)featuresDataType);

        return predictHelper(features);
    }

    std::vector<std::vector<std::pair<int, double>>> predictForFile(std::string path) {
        args.input = path;
        args.header = false;

        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        std::shared_ptr<DataReader> reader = DataReader::factory(args);
        reader->readData(labels, features, args);

        return predictHelper(features);
    }

private:
    Args args;
    std::shared_ptr<Model> model;

    template<class T> py::list vectorToPyList(const std::vector<T>& vector){
        py::list pyList;
        for (auto& i : vector) pyList.append(py::cast(i));
        return pyList;
    }

    template<typename T> std::vector<T> pyListToVector(py::list const &pyList){
        std::vector<T> vector = std::vector<T>(pyList.size());
        for (size_t i = 0; i < pyList.size(); ++i) vector[i] = py::cast<T>(pyList[i]);
        return vector;
    }

    template<typename T> std::vector<T> pyDataToSparseVector(py::object input, InputDataType dataType) {
        std::vector<T> output;
        switch (dataType) {
            case denseData: {
                py::array_t<double, py::array::c_style | py::array::forcecast> array(input);
                for(int i = 0; i < array.size(); ++i)
                    output.push_back({i, array.at(i)});
                return output;
            }
            case sparseData: {
                // Sparse vectors are expected to be list of (id, value) tuples
                py::list pyList(input);
                for (int i = 0; i < pyList.size(); ++i) {
                    py::tuple pyTuple(pyList[i]);
                    output.push_back({py::cast<int>(pyTuple[0]), py::cast<double>(pyTuple[1])});
                }
                return output;
            }
            default:
                throw std::invalid_argument("Unknown data type for pyDataToSparseVector");
        }
    }

    // Reads multiple items from a python object and inserts onto a SRMatrix<Label>
    void readLabelsMatrix(SRMatrix<Label>& output, py::object& input, InputDataType dataType) {
        if (py::isinstance<py::list>(input)) {
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
            throw std::invalid_argument("Unknown data type");
    }

    // Reads multiple items from a python object and inserts onto a SRMatrix<Feature>
    //SRMatrix<Feature> readFeatureMatrix(py::object input, InputDataType dataType) {
        //SRMatrix<Feature> output;
    void readFeatureMatrix(SRMatrix<Feature>& output, py::object& input, InputDataType dataType) {
        if (py::isinstance<py::list>(input)) {
            py::list items(input);
            for (size_t i = 0; i < items.size(); ++i) {
                std::vector<Feature> rFeatures = pyDataToSparseVector<Feature>(items[i], sparseData);
                output.appendRow(rFeatures);
            }
        } else if (dataType == denseData) {
            // allow numpy arrays to be returned here too
//            py::array_t<float, py::array::c_style | py::array::forcecast> items(input);
//            auto buffer = items.request();
//            if (buffer.ndim != 2) throw std::runtime_error("Data must be a 2d array");
//
//            size_t rows = buffer.shape[0], features = buffer.shape[1];
//            std::vector<float> tempVec(features);
//            for (size_t row = 0; row < rows; ++row) {
//                int id = ids.size() ? ids.at(row) : row;
//                const dist_t* elemVecStart = items.data(row);
//                std::copy(elemVecStart, elemVecStart + features, tempVect.begin());
//                output.push_back(vectSpacePtr->CreateObjFromVect(id, -1, tempVect));
//                //this way it won't always work properly
//                //output->push_back(new Object(id, -1, features * sizeof(dist_t), items.data(row)));
//            }
        } else if (dataType == sparseData) {
            // the attr calls will fail with an attribute error, but this fixes the legacy
            // unittest case
            //if (!py::hasattr(input, "indptr")) {
            //    throw py::value_error("expect CSR matrix here");
            //}

            // Try to interpret input data as a CSR matrix
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
            throw std::invalid_argument("Unknown data type");
    }

    inline void fitHelper(SRMatrix<Label>& labels, SRMatrix<Feature>& features){
        args.printArgs();
        makeDir(args.output);
        args.saveToFile(joinPath(args.output, "args.bin"));

        // Create and train model (train function also saves model)
        model = Model::factory(args);
        model->train(labels, features, args, args.output);
    }

    inline std::vector<std::vector<std::pair<int, double>>> predictHelper(SRMatrix<Feature>& features){
        if(model == nullptr)
            throw std::runtime_error("Model does not exist!");
        if(!model->isLoaded())
            model->load(args, args.output);
        std::vector<std::vector<Prediction>> predictions = model->predictBatch(features, args);

        // This is only safe because it's struct with two fields casted to pair, don't do this with tuples!
        return reinterpret_cast<std::vector<std::vector<std::pair<int, double>>>&>(predictions);
    }
};


PYBIND11_MODULE(_napkinxc, n) {
    n.doc() = "Python Bindings for napkinXC C++ core";
    n.attr("__version__") = VERSION;

//    py::Py_Initialize();
//    py::PyEval_InitThreads();
//    py::init_numpy();

    py::class_<CPPModel>(n, "CPPModel")
    .def(py::init<>())
    .def("test_load", &CPPModel::testLoad)
    .def("set_args", &CPPModel::setArgs)
    .def("fit", &CPPModel::fit)
    .def("fit_from_file", &CPPModel::fitOnFile)
    .def("predict", &CPPModel::predict)
    .def("predict_for_file", &CPPModel::predictForFile);
}
