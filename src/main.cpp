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
#include "types.h"
#include "resources.h"


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

    // Create and train model (train function also saves model)
    std::shared_ptr<Model> model = Model::factory(args);
    model->train(labels, features, args, args.output);
    model->printInfo();

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

    auto resAfterData = getResources();

    // Load model and test
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    auto resAfterModel = getResources();

    // Predict for test set
    std::vector<std::vector<Prediction>> predictions = model->predictBatch(features, args);
    model->printInfo();

    auto resAfterPrediction = getResources();

    // Create measures and calculate scores
    auto measures = Measure::factory(args, model->outputSize());
    for (auto& m : measures) m->accumulate(labels, predictions);

    // Print results
    std::cout << std::setprecision(5) << "Results:\n";
    for (auto& m : measures) std::cout << "  " << m->getName() << ": " << m->value() << std::endl;

    // Print resources
    auto realTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(resAfterPrediction.timePoint - resAfterModel.timePoint).count()) / 1000;
    auto cpuTime = resAfterPrediction.cpuTime - resAfterModel.cpuTime;
    std::cout << "Resources:"
                << "\n  Real time (s): " << realTime
                << "\n  CPU time (s): " << cpuTime
                << "\n  Real time / data point (ms): " << realTime * 1000 / labels.rows()
                << "\n  CPU time / data point (ms): " << cpuTime * 1000 / labels.rows()
                << "\n  Model real memory size (MB): " << (resAfterModel.currentRealMem - resAfterData.currentRealMem) / 1024
                << "\n  Model virtual memory size (MB): " << (resAfterModel.currentVirtualMem - resAfterData.currentVirtualMem) / 1024
                << "\n  Peak of real memory usage (MB): " << resAfterPrediction.peakRealMem / 1024
                << "\n  Peak of virtual memory usage (MB): " << resAfterPrediction.peakVirtualMem / 1024
                << "\n";

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
        // TODO
    }

        // Read data from file and output prediction to cout
    else {
        SRMatrix<Label> labels;
        SRMatrix<Feature> features;
        reader->readData(labels, features, args);

        std::vector<Prediction> prediction;
        prediction.reserve(model->outputSize());
        for (int r = 0; r < features.rows(); ++r) {
            model->predict(prediction, features.row(r), args);

            // Print prediction
            std::cout << labels.row(r)[0];
            for (const auto& p : prediction) std::cout << " " << p.label << ":" << p.value;
            std::cout << std::endl;
            prediction.clear();
        }
    }
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

    return 0;
}
