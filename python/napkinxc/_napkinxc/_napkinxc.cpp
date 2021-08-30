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
#include "basic_types.h"
#include "measure.h"
#include "model.h"
#include "plt.h"
#include "read_data.h"
#include "resources.h"
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

typedef std::tuple<py::array_t<Real>, py::array_t<int>, py::array_t<int>> ScipyCSRMatrixData;

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

template<typename T> std::vector<T> pyListToVector(py::list const &pyList){
    std::vector<T> vector(pyList.size());
    for (size_t i = 0; i < pyList.size(); ++i) vector[i] = py::cast<T>(pyList[i]);
    return vector;
}

template<typename T> void pyDataToSparseVector(std::vector<T>& output, py::object input, InputDataType dataType) {
    if(dataType == ndarray) {
        py::array_t<Real, py::array::c_style | py::array::forcecast> pyArray(input); // TODO: test it with different strides
        output.reserve(output.size() + pyArray.size());
        for(int i = 0; i < pyArray.size(); ++i) output.emplace_back(i, pyArray.at(i));
    }
    else if(dataType == list) {
        py::list pyList(input);
        if(pyList.size() == 0) return;
        output.reserve(output.size() + pyList.size());

        // Sparse vectors are expected to be list of (id (int), value (float)) tuples or list of ids (int).
        if(py::isinstance<py::tuple>(pyList[0])){
            for (int i = 0; i < pyList.size(); ++i) {
                py::tuple pyTuple(pyList[i]);
                output.emplace_back(py::cast<int>(pyTuple[0]), py::cast<Real>(pyTuple[1]));
            }
        } else {
            for (int i = 0; i < pyList.size(); ++i)
                output.emplace_back(py::cast<int>(pyList[i]), 1);
        }
    }
    else throw py::value_error("Unsupported data type for pyDataToSparseVector");
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


ScipyCSRMatrixData SRMatrixToScipyCSRMatrix(SRMatrix& matrix, bool sortIndices){
    int rows = matrix.rows();
    int cells = matrix.cells();

    Real* data = new Real[cells];
    int* indices = new int[cells];
    int* indptr = new int[rows + 1];

    int i = 0;
    for(auto r = 0; r < matrix.rows(); ++r){
        indptr[r] = i;
        if (sortIndices && !std::is_sorted(matrix[r].begin(), matrix[r].end(), IRVPairIndexComp()))
            std::sort(matrix[r].begin(), matrix[r].end(), IRVPairIndexComp());
        for(auto &f : matrix[r]){
            indices[i] = f.index;
            data[i] = f.value;
            ++i;
        }
    }
    indptr[rows] = cells;

    auto pyData = dataToPyArray(data, {cells});
    auto pyIndices = dataToPyArray(indices, {cells});
    auto pyIndptr = dataToPyArray(indptr, {rows + 1});

    return std::make_tuple(pyData, pyIndices, pyIndptr);
}

std::tuple<std::vector<std::vector<int>>, ScipyCSRMatrixData> loadLibSvmFileLabelsList(std::string path, bool sortIndices){
    SRMatrix labels;
    SRMatrix features;

    Args args;
    args.input = path;
    args.processData = false;
    readData(labels, features, args);

    // Labels
    int rows = labels.rows();
    std::vector<std::vector<Label>> pyLabels(labels.rows());
    for(auto r = 0; r < labels.rows(); ++r){
        for(auto &l : labels[r]) pyLabels[r].push_back(l.index);
        if (sortIndices && !std::is_sorted(pyLabels[r].begin(), pyLabels[r].end()))
            std::sort(pyLabels[r].begin(), pyLabels[r].end());
    }

    // Features
    auto pyFeatures = SRMatrixToScipyCSRMatrix(features, sortIndices);

    return std::make_tuple(pyLabels, pyFeatures);
}

std::tuple<ScipyCSRMatrixData, ScipyCSRMatrixData> loadLibSvmFileLabelsCSRMatrix(std::string path, bool sortIndices) {
    SRMatrix labels;
    SRMatrix features;

    Args args;
    args.input = path;
    args.processData = false;
    readData(labels, features, args);

    auto pyLabels = SRMatrixToScipyCSRMatrix(labels, sortIndices);
    auto pyFeatures = SRMatrixToScipyCSRMatrix(features, sortIndices);

    return std::make_tuple(pyLabels, pyFeatures);
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
            SRMatrix labels;
            SRMatrix features;
            readData(labels, features, args);
            fitHelper(labels, features);
        });
    }

    void fit(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType){
        runAsInterruptable([&] {
            SRMatrix labels;
            SRMatrix features;
            readSRMatrix(features, inputFeatures, (InputDataType) featuresDataType, true);
            readSRMatrix(labels, inputLabels, (InputDataType) labelsDataType);
            fitHelper(labels, features);
        });
    }

    void preload(){
        if(model == nullptr){
            args.loadFromFile(joinPath(args.output, "args.bin"));
            model = Model::factory(args);
        }
        if(!model->isPreloaded()) model->preload(args, args.output);
    }

    void load(){
        if(model == nullptr){
            args.loadFromFile(joinPath(args.output, "args.bin"));
            model = Model::factory(args);
        }
        if(!model->isLoaded()) model->load(args, args.output);
    }

    void unload(){
        if(model != nullptr && model->isLoaded()) model->unload();
    }

    void setThresholds(std::vector<Real> thresholds){
        load();
        model->setThresholds(thresholds);
    }

    void setLabelsWeights(std::vector<Real> weights){
        load();
        model->setLabelsWeights(weights);
    }

    std::vector<std::vector<int>> predict(py::object inputFeatures, int featuresDataType, int topK, Real threshold){
        auto predWithProba = predictProba(inputFeatures, featuresDataType, topK, threshold);
        return dropProbaHelper(predWithProba);
    }

    std::vector<std::vector<std::pair<int, Real>>> predictProba(py::object inputFeatures, int featuresDataType, int topK, Real threshold){
        std::vector<std::vector<std::pair<int, Real>>> pred;
        runAsInterruptable([&] {
            load();
            SRMatrix features;
            readSRMatrix(features, inputFeatures, (InputDataType)featuresDataType, true);
            pred = predictHelper(features, topK, threshold);
        });

        return pred;
    }

    std::vector<Real> ofo(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType) {
        std::vector<Real> thresholds;
        runAsInterruptable([&] {
            load();
            SRMatrix labels;
            SRMatrix features;
            readSRMatrix(features, inputFeatures, (InputDataType)featuresDataType, true);
            readSRMatrix(labels, inputLabels, (InputDataType)labelsDataType);
            args.printArgs("ofo");
            thresholds = model->ofo(features, labels, args);
        });

        return thresholds;
    }

    std::vector<std::vector<int>> predictForFile(std::string path, int topK, Real threshold) {
        auto predWithProba = predictProbaForFile(path, topK, threshold);
        return dropProbaHelper(predWithProba);
    }

    std::vector<std::vector<std::pair<int, Real>>> predictProbaForFile(std::string path, int topK, Real threshold) {
        std::vector<std::vector<std::pair<int, Real>>> pred;
        runAsInterruptable([&] {
            load();
            args.input = path;

            SRMatrix labels;
            SRMatrix features;
            readData(labels, features, args);
            pred = predictHelper(features, topK, threshold);
        });

        return pred;
    }

    std::vector<std::pair<std::string, Real>> test(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType,
                                                     int topK, Real threshold, std::string measuresStr){
        std::vector<std::pair<std::string, Real>> results;
        runAsInterruptable([&] {
            load();
            SRMatrix labels;
            SRMatrix features;
            readSRMatrix(features, inputFeatures, (InputDataType)featuresDataType, true);
            readSRMatrix(labels, inputLabels, (InputDataType)labelsDataType);
            results = testHelper(labels, features, topK, threshold, measuresStr);
        });

        return results;
    }

    std::vector<std::pair<std::string, Real>> testOnFile(std::string path, int topK, Real threshold, std::string measuresStr){
        std::vector<std::pair<std::string, Real>> results;
        runAsInterruptable([&] {
            load();
            args.input = path;

            SRMatrix labels;
            SRMatrix features;
            readData(labels, features, args);
            results = testHelper(labels, features, topK, threshold, measuresStr);
        });

        return results;
    }

    void buildTree(py::object inputFeatures, py::object inputLabels, int featuresDataType, int labelsDataType){
        if(args.modelType == plt || args.modelType == hsm) {
            runAsInterruptable([&] {
                if(model == nullptr) model = Model::factory(args);
                auto treeModel = std::dynamic_pointer_cast<PLT>(model);

                SRMatrix labels;
                SRMatrix features;
                readSRMatrix(features, inputFeatures, (InputDataType)featuresDataType, true);
                readSRMatrix(labels, inputLabels, (InputDataType)labelsDataType);

                makeDir(args.output);
                args.saveToFile(joinPath(args.output, "args.bin"));
                treeModel->buildTree(labels, features, args, args.output);
            });
        }
    }

    std::vector<std::vector<std::pair<int, Real>>> getNodesToUpdate(py::object inputLabels, int labelsDataType){
        std::vector<std::vector<std::pair<int, Real>>> nodesToUpdate;
        if(args.modelType == plt || args.modelType == hsm) {
            SRMatrix labels;
            readSRMatrix(labels, inputLabels, (InputDataType)labelsDataType);

            preload();
            auto treeModel = std::dynamic_pointer_cast<PLT>(model);
            nodesToUpdate = treeModel->getNodesToUpdate(labels);
        }
        return nodesToUpdate;
    }

    std::vector<std::vector<std::pair<int, Real>>> getNodesUpdates(py::object inputLabels, int labelsDataType){
        std::vector<std::vector<std::pair<int, Real>>> nodesUpdates;
        if(args.modelType == plt || args.modelType == hsm) {
            SRMatrix labels;
            readSRMatrix(labels, inputLabels, (InputDataType)labelsDataType);

            preload();
            auto treeModel = std::dynamic_pointer_cast<PLT>(model);
            nodesUpdates = treeModel->getNodesUpdates(labels);
        }
        return nodesUpdates;
    }

    std::vector<std::tuple<int, int, int>> getTreeStructure(){
        std::vector<std::tuple<int, int, int>> treeStructure;
        if(args.modelType == plt || args.modelType == hsm) {
            preload();
            auto treeModel = std::dynamic_pointer_cast<PLT>(model);
            treeStructure = treeModel->getTreeStructure();
        }
        return treeStructure;
    }

    void setTreeStructure(std::vector<std::tuple<int, int, int>> treeStructure){
        if(args.modelType == plt || args.modelType == hsm) {
            if(model == nullptr) model = Model::factory(args);
            auto treeModel = std::dynamic_pointer_cast<PLT>(model);
            makeDir(args.output);
            args.saveToFile(joinPath(args.output, "args.bin"));
            treeModel->setTreeStructure(treeStructure, args.output);
        }
    }

//    ScipyCSRMatrixData getWeights() {
//        int cells = 0;
//        int rows = 0;
//
//        Real* data = new Real[cells];
//        int* indices = new int[cells];
//        int* indptr = new int[rows + 1];
//
//        //TODO
//
//        auto pyData = dataToPyArray(data, {cells});
//        auto pyIndices = dataToPyArray(indices, {cells});
//        auto pyIndptr = dataToPyArray(indptr, {rows + 1});
//
//        return std::make_tuple(pyData, pyIndices, pyIndptr);
//    }
//
//    void setWeights(py::object input, InputDataType dataTyp){
//        //TODO
//    }

private:
    Args args;
    std::shared_ptr<Model> model;

    // Reads multiple items from a python object and inserts onto a SRMatrix
    //SRMatrix readSRMatrix(py::object input, InputDataType dataType) {
    //SRMatrix output;
    void readSRMatrix(SRMatrix& output, py::object& input, InputDataType dataType, bool process = false) {
        std::vector<IRVPair> rVec;

        if (dataType == list) {
            py::list pyData(input);
            for (size_t i = 0; i < pyData.size(); ++i) {
                rVec.clear();
                if(process) prepareFeaturesVector(rVec, args.bias);

                pyDataToSparseVector<IRVPair>(rVec, pyData[i], list);

                if(process) processFeaturesVector(rVec, args.norm, args.hash, args.featuresThreshold);
                output.appendRow(rVec);
            }
        } else if (dataType == ndarray) {
            py::array_t<Real, py::array::c_style | py::array::forcecast> pyData(input);
            auto buffer = pyData.request();
            if (buffer.ndim != 2) throw py::value_error("Data must be a 2d array");

            size_t rows = buffer.shape[0];
            size_t rSize = buffer.shape[1];

            // Read each row from the sparse matrix, and insert
            for (size_t r = 0; r < rows; ++r) {
                const Real* rData = pyData.data(r);
                rVec.clear();
                if(process) prepareFeaturesVector(rVec, args.bias);

                for (int f = 0; f < rSize; ++f) {
                    Real v = rData[f];
                    if (v != 0) rVec.emplace_back(f, rData[f]);
                }

                if(process) processFeaturesVector(rVec, args.norm, args.hash, args.featuresThreshold);
                output.appendRow(rVec);
            }
        } else if (dataType == csr_matrix) {
            // Check for attributes
            if (!py::hasattr(input, "indptr")) {
                throw py::value_error("Expect csr_matrix CSR matrix");
            }

            // Try to interpret input data as a csr_matrix CSR matrix
            py::array_t<int> indptr(input.attr("indptr"));
            py::array_t<int> indices(input.attr("indices"));
            py::array_t<Real> data(input.attr("data"));

            // Read each row from the sparse matrix, and insert
            for (int rId = 0; rId < indptr.size() - 1; ++rId) {
                rVec.clear();
                if(process) prepareFeaturesVector(rVec, args.bias);

                for (int i = indptr.at(rId); i < indptr.at(rId + 1); ++i)
                    rVec.emplace_back(indices.at(i), data.at(i));

                if(process) processFeaturesVector(rVec, args.norm, args.hash, args.featuresThreshold);
                output.appendRow(rVec);
            }
        } else
            throw py::value_error("Unsupported data type");
    }

    inline void fitHelper(SRMatrix& labels, SRMatrix& features){
        // Save args to file
        args.printArgs("train");
        makeDir(args.output);
        args.saveToFile(joinPath(args.output, "args.bin"));

        // Create and train model (train function also saves model)
        if(model == nullptr) model = Model::factory(args);
        model->train(labels, features, args, args.output);
    }

    inline std::vector<std::vector<std::pair<int, Real>>> predictHelper(SRMatrix& features, int topK, Real threshold){
        args.printArgs("predict");

        // TODO: refactor this
        args.topK = topK;
        args.threshold = threshold;
        auto predictions = model->predictBatch(features, args);

        // This is only safe because it's struct with two fields casted to pair, don't do this with tuples!
        return reinterpret_cast<std::vector<std::vector<std::pair<int, Real>>>&>(predictions);
    }

    inline std::vector<std::pair<std::string, Real>> testHelper(SRMatrix& labels, SRMatrix& features, int topK, Real threshold, std::string measuresStr){
        args.printArgs("test");

        args.topK = topK;
        args.threshold = threshold;
        auto predictions = model->predictBatch(features, args);

        args.measures = measuresStr;
        auto measures = Measure::factory(args, model->outputSize());
        for (auto& m : measures) m->accumulate(labels, predictions);

        std::vector<std::pair<std::string, Real>> results;
        for (auto& m : measures)
            results.push_back({m->getName(), m->value()});

        return results;
    }

    inline std::vector<std::vector<int>> dropProbaHelper(std::vector<std::vector<std::pair<int, Real>>>& predWithProba){
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

    n.def("_load_libsvm_file_labels_list", &loadLibSvmFileLabelsList);
    n.def("_load_libsvm_file_labels_csr_matrix", &loadLibSvmFileLabelsCSRMatrix);

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
    .def("build_tree", &CPPModel::buildTree)
    .def("get_nodes_to_update", &CPPModel::getNodesToUpdate)
    .def("get_nodes_updates", &CPPModel::getNodesUpdates)
    .def("get_tree_structure", &CPPModel::getTreeStructure)
    .def("set_tree_structure", &CPPModel::setTreeStructure);
//    .def("get_weights", &CPPModel::getWeights)
//    .def("set_weights", &CPPModel::setWeights);
}
