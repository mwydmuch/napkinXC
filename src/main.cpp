/*
 Copyright (c) 2018-2020 by Marek Wydmuch

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

/*
 This is main file for napkinXC command line tool,
 it should contain all commands implementations.
 Only this file should use std:cout.
 */

#include <iomanip>
#include <iostream>

#include "args.h"
#include "data_reader.h"
#include "log.h"
#include "measure.h"
#include "misc.h"
#include "model.h"
#include "resources.h"
#include "types.h"
#include "version.h"


// TODO: refactor this as load/save vector
std::vector<double> loadThresholds(std::string infile){
    std::vector<double> thresholds;
    std::ifstream thresholdsIn(infile);
    double t;
    while (thresholdsIn >> t) thresholds.push_back(t);
    return thresholds;
}

void saveThresholds(std::vector<double>& thresholds, std::string outfile){
    std::ofstream out(outfile);
    for(auto t : thresholds)
        out << t << "\n";
    out.close();
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
    LOG(COUT) << "Train data statistics:"
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
    LOG(COUT) << "Resources during training:"
              << "\n  Train real time (s): " << realTime << "\n  Train CPU time (s): " << cpuTime
              << "\n  Train real time / data point (ms): " << realTime * 1000 / labels.rows()
              << "\n  Train CPU time / data point (ms): " << cpuTime * 1000 / labels.rows()
              << "\n  Peak of real memory during training (MB): " << resAfterTraining.peakRealMem / 1024
              << "\n  Peak of virtual memory during training (MB): " << resAfterTraining.peakVirtualMem / 1024 << "\n";
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
    LOG(COUT) << "Test data statistics:"
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
    LOG(COUT) << std::setprecision(5) << "Results:\n";
    for (auto& m : measures){
        LOG(COUT) << "  " << m->getName() << ": " << m->value();
        //if(m->isMeanMeasure()) LOG(COUT) << " ± " << m->stdDev(); // Print std
        LOG(COUT) << "\n";
    }
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
    LOG(COUT) << "Resources during test:"
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

    LOG(COUT) << std::setprecision(5);

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
                for (const auto &l : p) LOG(COUT) << l.label << ":" << l.value << " ";
                LOG(COUT) << "\n";
            }
        } else { // For 1 thread predict and immediately save to file
            for(int r = 0; r < features.rows(); ++r){
                printProgress(r, features.rows());
                std::vector<Prediction> prediction;

                if (!args.thresholds.empty())
                    model->predictWithThresholds(prediction, features[r], args);
                else
                    model->predict(prediction, features[r], args);

                for (const auto &l : prediction) LOG(COUT) << l.label << ":" << l.value << " ";
                LOG(COUT) << "\n";
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

    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    reader->readData(labels, features, args);

    auto resAfterData = getResources();

    std::vector<double> thresholds = model->ofo(features, labels, args);
    saveThresholds(thresholds, args.thresholds);

    auto resAfterFo = getResources();

    // Print resources
    auto realTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
            resAfterFo.timePoint - resAfterData.timePoint)
            .count()) /
                    1000;
    auto cpuTime = resAfterFo.cpuTime - resAfterData.cpuTime;
    LOG(COUT) << "Resources during F-measure optimization:"
              << "\n  Optimization real time (s): " << realTime
              << "\n  Optimization CPU time (s): " << cpuTime << "\n";
}

void testPredictionTime(Args& args) {
    // Method for testing performance on different batch (test dataset) sizes

    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs();

    // Create data reader
    std::shared_ptr<DataReader> reader = DataReader::factory(args);
    reader->loadFromFile(joinPath(args.output, "data_reader.bin"));

    // Load model
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    reader->readData(labels, features, args);

    // Read batch sizes
    std::vector<int> batchSizes;
    for(const auto& s : split(args.batchSizes))
        batchSizes.push_back(std::stoi(s));

    // Prepare rng for selecting batches
    std::default_random_engine rng(args.seed);
    std::uniform_int_distribution<int> dist(0, features.rows() - 1);

    LOG(COUT) << "Results:";
    for(const auto& batchSize : batchSizes) {
        long double time = 0;
        long double timeSq = 0;
        long double timePerPoint = 0;
        long double timePerPointSq = 0;

        for (int i = 0; i < args.batches; ++i) {
            // Generate batch
            std::vector<Feature*> batch;
            batch.reserve(batchSize);
            for (int j = 0; j < batchSize; ++j)
                batch.push_back(features[dist(rng)]);

            assert(batch.size() == batchSize);

            // Test batch
            double startTime = static_cast<double>(clock()) / CLOCKS_PER_SEC;
            for (const auto& r : batch) {
                std::vector<Prediction> prediction;
                model->predict(prediction, r, args);
            }

            // Accumulate time measurements
            double stopTime = static_cast<double>(clock()) / CLOCKS_PER_SEC;
            double timeDiff = stopTime - startTime;
            time += timeDiff;
            timeSq += timeDiff * timeDiff;

            timeDiff = timeDiff * 1000 / batchSize;
            timePerPoint += timeDiff;
            timePerPointSq += timeDiff * timeDiff;
        }

        long double meanTime = time / args.batches;
        long double meanTimePerPoint = timePerPoint / args.batches;
        LOG(COUT) << "\n  Batch " << batchSize << " test CPU time / batch (s): " << meanTime
                  << "\n  Batch " << batchSize << " test CPU time std (s): " << std::sqrt(timeSq / args.batches - meanTime * meanTime)
                  << "\n  Batch " << batchSize << " test CPU time / data points (ms): " << meanTimePerPoint
                  << "\n  Batch " << batchSize << " test CPU time / data points std (ms): " << std::sqrt(timePerPointSq / args.batches - meanTimePerPoint * meanTimePerPoint);

    }
    LOG(COUT) << "\n";
}

void printHelp() {
    std::cout << R"HELP(Usage: nxc [command] [args ...]

Commands:
    train               Train model on given input data
    test                Test model on given input data
    predict             Predict for given data
    ofo
    version
    help

Args:
    General:
    -i, --input         Input dataset
    -o, --output        Output (model) dir
    -m, --model         Model type (default = plt):
                        Models: ovr, br, hsm, plt, oplt, ubop, ubopHsm, brMips, ubopMips
    --ensemble          Ensemble of models (default = 0)
    -d, --dataFormat    Type of data format (default = libsvm):
                        Supported data formats: libsvm
    -t, --threads       Number of threads used for training and testing (default = 0)
                        Note: -1 to use system #cpus - 1, 0 to use system #cpus
    --memLimit          Amount of memory in GB used for training OVR and BR models (default = 0)
                        Note: 0 to use system memory
    --header            Input contains header (default = 1)
                        Header format for libsvm: #lines #features #labels
    --hash              Size of features space (default = 0)
                        Note: 0 to disable hashing
    --featuresThreshold Prune features belowe given threshold (default = 0.0)
    --seed              Seed

    Base classifiers:
    --optimizer         Use LibLiner or online optimizers (default = libliner)
                        Optimizers: liblinear, sgd, adagrad, fobos
    --bias              Add bias term (default = 1)
    --weightsThreshold  Prune weights below given threshold (default = 0.1)
    --inbalanceLabelsWeighting     Increase the weight of minority labels in base classifiers (default = 0)

    LibLinear:
    -s, --solver        LibLinear solver (default = L2R_LR_DUAL)
                        Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                           L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
                        See: https://github.com/cjlin1/liblinear
    -c, -C, --cost      Inverse of regularization strength. Must be a positive float.
                        Smaller values specify stronger regularization. (default = 10.0)
                        Note: -1 to automatically find best value for each node.
    -e, --eps           Stopping criteria (default = 0.1)
                        See: https://github.com/cjlin1/liblinear

    SGD/AdaGrad:
    -l, --lr, --eta     Step size (learning rate) of SGD/AdaGrad (default = 1.0)
    --epochs            Number of epochs of SGD/AdaGrad (default = 5)
    --adagradEps        AdaGrad epsilon (default = 0.001)

    Tree:
    -a, --arity         Arity of a tree (default = 2)
    --maxLeaves         Maximum number of leaves (labels) in one internal node.
                        Supported by k-means and balanced trees. (default = 100)
    --tree              File with tree structure
    --treeType          Type of a tree to build if file with structure is not provided
                        Tree types: hierarchicalKmeans, huffman, completeInOrder, completeRandom,
                                    balancedInOrder, balancedRandom, onlineComplete, onlineBalanced,
                                    onlineRandom

    K-means tree:
    --kmeansEps         Stopping criteria for K-Means clustering (default = 0.001)
    --kmeansBalanced    Use balanced K-Means clustering (default = 1)

    Prediction:
    --topK              Predict top k elements (default = 5)
    --threshold         Probability threshold (default = 0)
    --setUtility        Type of set-utility function for prediction using ubop, ubopHsm, ubopMips models.
                        Set-utility functions: uP, uF1, uAlpha, uAlphaBeta, uDeltaGamma
                        See: https://arxiv.org/abs/1906.08129

    Set-Utility:
    --alpha
    --beta
    --delta
    --gamma

    Test:
    --measures          Evaluate test using set of measures (default = "p@1,r@1,c@1,p@3,r@3,c@3,p@5,r@5,c@5")
                        Measures: acc (accuracy), p (precision), r (recall), c (coverage),
                                  p@k (precision at k), r@k (recall at k), c@k (coverage at k), s (prediction size)

    )HELP";
}

int main(int argc, char** argv) {
    logLevel = CERR;

    if(argc == 1) {
        std::cout << "No command provided \n";
        printHelp();
        exit(EXIT_FAILURE);
    }

    std::string command(argv[1]);
    std::vector<std::string> arg(argv + 2, argv + argc);
    Args args = Args();

    // Parse args
    try {
        args.parseArgs(arg);
    } catch (std::invalid_argument& e) {
        std::cout << e.what() << "\n";
        printHelp();
        exit(EXIT_FAILURE);
    }

    if (command == "-h" || command == "--help" || command == "help")
        printHelp();
    else if (command == "-v" || command == "--version" || command == "version")
        std::cout << "napkinXC " << VERSION << "\n";
    else if (command == "train")
        train(args);
    else if (command == "test")
        test(args);
    else if (command == "predict")
        predict(args);
    else if (command == "ofo")
        ofo(args);
    else if (command == "testPredictionTime")
        testPredictionTime(args);
    else {
        std::cout << "Unknown command type: " << command << "\n";
        printHelp();
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
