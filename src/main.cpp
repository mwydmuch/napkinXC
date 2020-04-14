/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <iomanip>
#include <iostream>

#include "args.h"
#include "data_reader.h"
#include "measure.h"
#include "model.h"
#include "resources.h"
#include "types.h"

std::vector<double> loadThresholds(std::string infile){
    std::vector<double> thresholds;
    std::ifstream thresholdsIn(infile);
    double t;
    while (thresholdsIn >> t) thresholds.push_back(t);
    return thresholds;
}

void train(Args& args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    args.printArgs();
    makeDir(args.output);
    args.saveToFile(joinPath(args.output, "args.bin"));

    // Create data reader and load train data
    std::shared_ptr<DataReader> reader = DataReader::factory(args);
    reader->readData(labels, features, args);
    reader->saveToFile(joinPath(args.output, "data_reader.bin"));
    std::cout << "Train data statistics:"
              << "\n  Train data points: " << features.rows() << "\n  Uniq features: " << features.cols() - 2
              << "\n  Uniq labels: " << labels.cols()
              << "\n  Labels / data point: " << static_cast<double>(labels.cells()) / labels.rows()
              << "\n  Features / data point: " << static_cast<double>(features.cells()) / features.rows() << "\n";

    auto resAfterData = getResources();

    // Create and train model (train function also saves model)
    std::shared_ptr<Model> model = Model::factory(args);
    model->train(labels, features, args, args.output);
    model->printInfo();

    auto resAfterTraining = getResources();

    // Print resources
    auto realTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            resAfterTraining.timePoint - resAfterData.timePoint)
                                            .count()) /
                    1000;
    auto cpuTime = resAfterTraining.cpuTime - resAfterData.cpuTime;
    std::cout << "Resources during training:"
              << "\n  Train real time (s): " << realTime << "\n  Train CPU time (s): " << cpuTime
              << "\n  Train real time / data point (ms): " << realTime * 1000 / labels.rows()
              << "\n  Train CPU time / data point (ms): " << cpuTime * 1000 / labels.rows()
              << "\n  Peak of real memory during training (MB): " << resAfterTraining.peakRealMem / 1024
              << "\n  Peak of virtual memory during training (MB): " << resAfterTraining.peakVirtualMem / 1024 << "\n";

    std::cerr << "All done!\n";
}

void test(Args& args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs();

    // Create data reader and load test data
    std::shared_ptr<DataReader> reader = DataReader::factory(args);
    reader->loadFromFile(joinPath(args.output, "data_reader.bin"));
    reader->readData(labels, features, args);
    std::cout << "Test data statistics:"
              << "\n  Test data points: " << features.rows()
              << "\n  Labels / data point: " << static_cast<double>(labels.cells()) / labels.rows()
              << "\n  Features / data point: " << static_cast<double>(features.cells()) / features.rows() << "\n";

    auto resAfterData = getResources();

    // Load model and test
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    auto resAfterModel = getResources();

    // Predict for test set
    std::vector<std::vector<Prediction>> predictions;
    if (!args.thresholds.empty()) { // Using thresholds if provided
        std::vector<double> thresholds = loadThresholds(args.thresholds);
        model->setThresholds(thresholds);
        predictions = model->predictBatchWithThresholds(features, args);
    } else
        predictions = model->predictBatch(features, args);

    auto resAfterPrediction = getResources();

    // Create measures and calculate scores
    auto measures = Measure::factory(args, model->outputSize());
    for (auto& m : measures) m->accumulate(labels, predictions);

    // Print results
    std::cout << std::setprecision(5) << "Results:\n";
    for (auto& m : measures) std::cout << "  " << m->getName() << ": " << m->value() << std::endl;
    model->printInfo();

    // Print resources
    auto loadRealTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
            resAfterModel.timePoint - resAfterData.timePoint)
            .count()) / 1000;
    auto realTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            resAfterPrediction.timePoint - resAfterModel.timePoint)
                                            .count()) / 1000;
    auto loadCpuTime = resAfterModel.cpuTime - resAfterData.cpuTime;
    auto cpuTime = resAfterPrediction.cpuTime - resAfterModel.cpuTime;
    std::cout << "Resources during test:"
              << "\n  Loading real time (s): " << loadRealTime
              << "\n  Loading CPU time (s): " << loadCpuTime
              << "\n  Test real time (s): " << realTime << "\n  Test CPU time (s): " << cpuTime
              << "\n  Test real time / data point (ms): " << realTime * 1000 / labels.rows()
              << "\n  Test CPU time / data point (ms): " << cpuTime * 1000 / labels.rows()
              << "\n  Model real memory size (MB): "
              << (resAfterModel.currentRealMem - resAfterData.currentRealMem) / 1024
              << "\n  Model virtual memory size (MB): "
              << (resAfterModel.currentVirtualMem - resAfterData.currentVirtualMem) / 1024
              << "\n  Peak of real memory during testing (MB): " << resAfterPrediction.peakRealMem / 1024
              << "\n  Peak of virtual memory during testing (MB): " << resAfterPrediction.peakVirtualMem / 1024 << "\n";

    std::cerr << "All done!\n";
}

void predict(Args& args) {
    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs();

    // Create data reader
    std::shared_ptr<DataReader> reader = DataReader::factory(args);
    reader->loadFromFile(joinPath(args.output, "data_reader.bin"));

    // Load model
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    std::cout << std::setprecision(5);

    // Predict data from cin and output to cout
    if (args.input == "-") {
        for (std::string line; std::getline(std::cin, line); ) {
            // TODO
        }
    }

    // Read data from file and output prediction to cout
    else {
        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        reader->readData(labels, features, args);

        if (!args.thresholds.empty()) { // Using thresholds if provided
            std::vector<double> thresholds = loadThresholds(args.thresholds);
            model->setThresholds(thresholds);
        }

        if(args.threads > 1) {
            std::vector<std::vector<Prediction>> predictions;
            if (!args.thresholds.empty())
                predictions = model->predictBatchWithThresholds(features, args);
            else
                predictions = model->predictBatch(features, args);

            for (const auto &p : predictions) {
                for (const auto &l : p) std::cout << l.label << ":" << l.value << " ";
                std::cout << std::endl;
            }
        } else { // For 1 thread predict and immediately save to file
            for(int r = 0; r < features.rows(); ++r){
                printProgress(r, features.rows());
                std::vector<Prediction> prediction;

                if (!args.thresholds.empty())
                    model->predictWithThresholds(prediction, features[r], args);
                else
                    model->predict(prediction, features[r], args);

                for (const auto &l : prediction) std::cout << l.label << ":" << l.value << " ";
                std::cout << std::endl;
            }
        }
    }
}

void ofo(Args& args) {
    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs();

    // Create data reader
    std::shared_ptr<DataReader> reader = DataReader::factory(args);
    reader->loadFromFile(joinPath(args.output, "data_reader.bin"));

    // Load model
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    std::cout << std::setprecision(5);

    // Read data from file and output prediction to cout
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    reader->readData(labels, features, args);

    std::vector<double> thresholds = model->ofo(features, labels, args);

    std::ofstream out(args.thresholds);
    for(auto t : thresholds)
        out << t <<std::endl;
    out.close();
}


int main(int argc, char** argv) {
    std::vector<std::string> arg(argv, argv + argc);
    Args args = Args();

    // Parse args
    args.parseArgs(arg);

    if (args.command == "train")
        train(args);
    else if (args.command == "test")
        test(args);
    else if (args.command == "predict")
        predict(args);
    else if (args.command == "ofo")
        ofo(args);

    return 0;
}
