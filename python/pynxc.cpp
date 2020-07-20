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

class PyModel {
public:
    //ModelWrapper(py::object args){
    PyModel(){
        model = Model::factory(args);
    }

    void save(std::string path){

    }

    void load(std::string path){

    }

    void trainFromFile(std::string path){

    }

    void train(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType){
        //SRMatrix<Label> labels = readFeatureMatrix(inputFeatures, featuresDataType);
        //SRMatrix<Feature> features = readLabelsMatrix(inputLabels, labelsDataType);

        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        readFeatureMatrix(features, inputFeatures, (InputDataType)featuresDataType);
        readLabelsMatrix(labels, inputLabels, (InputDataType)labelsDataType);

        args.printArgs();
        makeDir(args.output);
        args.saveToFile(joinPath(args.output, "args.bin"));

        // Create and train model (train function also saves model)
        model->train(labels, features, args, args.output);
    }

//    py::object predictForFile(std::string path){
//
//    }

//    py::object predict(py::object inputFeatures){
//
//    }

private:
    Args args;
    std::shared_ptr<Model> model;

    template<class T> py::list vectorToPyList(const std::vector<T>& vector){
        py::list pyList;
        for (auto& i : vector) pyList.append(py::cast(i));
        return pyList;
    }

    template<class T> std::vector<T> pyListToVector(py::list const &pyList){
        size_t pyListLength = py::len(pyList);
        std::vector<T> vector = std::vector<T>(pyListLength);
        for (size_t i = 0; i < pyListLength; ++i) vector[i] = py::cast<T>(pyList[i]);
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

    void readLabelsMatrix(SRMatrix<Label>& output, py::object input, InputDataType dataType) {
        if (py::isinstance<py::list>(input)) {
            py::list rows(input);
            for (size_t r = 0; r < rows.size(); ++r) {
                std::vector<Label> rLabels;
                if (py::isinstance<py::list>(rows[r])) {
                    py::list row(rows[r]);
                    rLabels = pyListToVector<Label>(row);
                }
                else
                    rLabels.push_back(py::cast<int>(rows[r]));

                output.appendRow(rLabels);
            }
        } else
            throw std::invalid_argument("Unknown data type");
    }

    // reads multiple items from a python object and inserts onto a similarity::ObjectVector
    // returns the number of elements inserted
    //SRMatrix<Feature> readFeatureMatrix(py::object input, InputDataType dataType) {
        //SRMatrix<Feature> output;
    void readFeatureMatrix(SRMatrix<Feature>& output, py::object input, InputDataType dataType) {
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

            // Try to intrepret input data as a CSR matrix
            py::array_t<int> indptr(input.attr("indptr"));
            py::array_t<int> indices(input.attr("indices"));
            py::array_t<double> data(input.attr("data"));

            // Read each row from the sparse matrix, and insert
            std::vector<Feature> rFeatures;
            for (int rId = 0; rId < indptr.size() - 1; ++rId) {
                rFeatures.clear();

                for (int i = indptr.at(rId); i < indptr.at(rId + 1); ++i) {
                    rFeatures.push_back({indices.at(i), data.at(i)});
                }
                std::sort(rFeatures.begin(), rFeatures.end());
                output.appendRow(rFeatures);
            }
        } else
            throw std::invalid_argument("Unknown data type");
    }
};


PYBIND11_MODULE(pynxc, n) {
    n.doc() = "Python Bindings for napkinXC";
    n.attr("__version__") = py::str(VERSION);

//    py::Py_Initialize();
//    py::PyEval_InitThreads();
//    py::init_numpy();

    py::class_<PyModel>(n, "Model")
    .def(py::init<>())
    .def("train", &PyModel::train);
}
