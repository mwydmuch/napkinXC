/*
 Copyright (c) 2018-2021 by Marek Wydmuch

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
#include "read_data.h"
#include "log.h"
#include "measure.h"
#include "misc.h"
#include "model.h"
#include "resources.h"
#include "types.h"
#include "version.h"

std::vector<double> loadVec(std::string infile){
    std::vector<double> vec;
    std::ifstream in(infile);
    double v;
    while (in >> v) vec.push_back(v);
    return vec;
}

void saveVec(std::vector<double>& vec, std::string outfile){
    std::ofstream out(outfile);
    for(auto v : vec)
        out << v << "\n";
    out.close();
}

void loadVecs(std::shared_ptr<Model> model, Args& args){
    std::vector<std::vector<Prediction>> predictions;
    if (!args.thresholds.empty()) { // Using thresholds if provided
        std::vector<double> thresholds = loadVec(args.thresholds);
        model->setThresholds(thresholds);
    }
    if (!args.labelsWeights.empty()) { // Using labelsWeights if provided
        std::vector<double> labelsWeights = loadVec(args.labelsWeights);
        model->setLabelsWeights(labelsWeights);
    }
}

void outputPrediction(std::vector<std::vector<Prediction>>& predictions, std::ofstream& output){
    for (const auto &p : predictions) {
        for (const auto &l : p) output << l.label << ":" << l.value << " ";
        output << "\n";
    }
}

void train(Args& args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    args.printArgs("train");
    makeDir(args.output);
    args.saveToFile(joinPath(args.output, "args.bin"));

    // Create data reader and load train data
    readData(labels, features, args);
    Log(COUT) << "Train data statistics:"
              << "\n  Train data points: " << features.rows()
              << "\n  Uniq features: " << features.cols() - 2
              << "\n  Uniq labels: " << labels.cols()
              << "\n  Labels / data point: " << static_cast<double>(labels.cells()) / labels.rows()
              << "\n  Features / data point: " << static_cast<double>(features.cells()) / features.rows() << "\n";

    auto resAfterData = getResources();

    // Create and train model (train function also saves model)
    std::shared_ptr<Model> model = Model::factory(args);
    loadVecs(model, args);
    model->train(labels, features, args, args.output);
    model->printInfo();

    auto resAfterTraining = getResources();

    // Print resources
    auto realTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            resAfterTraining.timePoint - resAfterData.timePoint)
                                            .count()) / 1000;
    auto cpuTime = resAfterTraining.cpuTime - resAfterData.cpuTime;
    Log(COUT) << "Train resources:"
              << "\n  Train real time (s): " << realTime
              << "\n  Train CPU time (s): " << cpuTime
              << "\n  Train real time / data point (ms): " << realTime * 1000 / labels.rows()
              << "\n  Train CPU time / data point (ms): " << cpuTime * 1000 / labels.rows()
              << "\n  Train peak of real memory (MB): " << resAfterTraining.peakRealMem / 1024
              << "\n  Train peak of virtual memory (MB): " << resAfterTraining.peakVirtualMem / 1024 << "\n";
}

void test(Args& args) {
    SRMatrix<Label> labels;
    SRMatrix<Feature> features;

    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs("test");

    // Load test data
    readData(labels, features, args);
    Log(COUT) << "Test data statistics:"
              << "\n  Test data points: " << features.rows()
              << "\n  Labels / data point: " << static_cast<double>(labels.cells()) / labels.rows()
              << "\n  Features / data point: " << static_cast<double>(features.cells()) / features.rows() << "\n";

    auto resAfterData = getResources();

    // Load model and test
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    auto resAfterModel = getResources();

    // Predict for test set
    loadVecs(model, args);
    std::vector<std::vector<Prediction>> predictions = model->predictBatch(features, args);

    auto resAfterPrediction = getResources();

    // Output predictions
    if(!args.prediction.empty()){
        std::ofstream out(args.prediction);
        outputPrediction(predictions, out);
        out.close();
    }

    // Create measures, calculate and print scores
    if(!args.measures.empty()){
        auto measures = Measure::factory(args, model->outputSize());
        for (auto& m : measures) m->accumulate(labels, predictions);

        Log(COUT) << std::setprecision(5) << "Results:\n";
        for (auto& m : measures){
            Log(COUT) << "  " << m->getName() << ": " << m->value();
            //if(m->isMeanMeasure()) Log(COUT) << " ± " << m->stdDev(); // Print std
            Log(COUT) << "\n";
        }
    }

    // Print additional model statistics
    model->printInfo();

    // Print resources
    auto realTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                            resAfterPrediction.timePoint - resAfterModel.timePoint)
                                            .count()) / 1000;
    auto cpuTime = resAfterPrediction.cpuTime - resAfterModel.cpuTime;
    Log(COUT) << "Test resources:"
              << "\n  Test real time (s): " << realTime
              << "\n  Test CPU time (s): " << cpuTime
              << "\n  Test real time / data point (ms): " << realTime * 1000 / labels.rows()
              << "\n  Test CPU time / data point (ms): " << cpuTime * 1000 / labels.rows()
              << "\n  Model real memory size (MB): "
              << (resAfterModel.currentRealMem - resAfterData.currentRealMem) / 1024
              << "\n  Model virtual memory size (MB): "
              << (resAfterModel.currentVirtualMem - resAfterData.currentVirtualMem) / 1024
              << "\n  Test peak of real memory (MB): " << resAfterPrediction.peakRealMem / 1024
              << "\n  Test peak of virtual memory (MB): " << resAfterPrediction.peakVirtualMem / 1024 << "\n";
}

void predict(Args& args) {
    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs("predict");

    // Load model
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    Log(COUT) << std::setprecision(5);

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
        readData(labels, features, args);

        loadVecs(model, args);

        if(args.threads > 1) {
            std::vector<std::vector<Prediction>> predictions = model->predictBatch(features, args);

            for (const auto &p : predictions) {
                for (const auto &l : p) Log(COUT) << l.label << ":" << l.value << " ";
                Log(COUT) << "\n";
            }
        } else { // For 1 thread predict and immediately output
            for(int r = 0; r < features.rows(); ++r){
                printProgress(r, features.rows());
                std::vector<Prediction> prediction;
                model->predict(prediction, features[r], args);

                for (const auto &l : prediction) Log(COUT) << l.label << ":" << l.value << " ";
                Log(COUT) << "\n";
            }
        }
    }
}

void ofo(Args& args) {
    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs();

    // Load model
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    readData(labels, features, args);

    auto resAfterData = getResources();

    std::vector<double> thresholds = model->ofo(features, labels, args);
    saveVec(thresholds, args.thresholds);

    auto resAfterFo = getResources();

    // Print resources
    auto realTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(
            resAfterFo.timePoint - resAfterData.timePoint)
            .count()) /
                    1000;
    auto cpuTime = resAfterFo.cpuTime - resAfterData.cpuTime;
    Log(COUT) << "Resources during F-measure optimization:"
              << "\n  Optimization real time (s): " << realTime
              << "\n  Optimization CPU time (s): " << cpuTime << "\n";
}

void testPredictionTime(Args& args) {
    // Method for testing performance on different batch (test dataset) sizes

    // Load model args
    args.loadFromFile(joinPath(args.output, "args.bin"));
    args.printArgs();

    // Load model
    std::shared_ptr<Model> model = Model::factory(args);
    model->load(args, args.output);

    SRMatrix<Label> labels;
    SRMatrix<Feature> features;
    readData(labels, features, args);

    // Read batch sizes
    std::vector<int> batchSizes;
    for(const auto& s : split(args.batchSizes))
        batchSizes.push_back(std::stoi(s));

    // Prepare rng for selecting batches
    std::default_random_engine rng(args.seed);
    std::uniform_int_distribution<int> dist(0, features.rows() - 1);

    Log(COUT) << "Results:";
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
        Log(COUT) << "\n  Batch " << batchSize << " test CPU time / batch (s): " << meanTime
                  << "\n  Batch " << batchSize << " test CPU time std (s): " << std::sqrt(timeSq / args.batches - meanTime * meanTime)
                  << "\n  Batch " << batchSize << " test CPU time / data points (ms): " << meanTimePerPoint
                  << "\n  Batch " << batchSize << " test CPU time / data points std (ms): " << std::sqrt(timePerPointSq / args.batches - meanTimePerPoint * meanTimePerPoint);

    }
    Log(COUT) << "\n";
}

void printHelp() {
    std::cout << R"HELP(Usage: nxc [command] [arg...]

Commands:
    train                   Train model on given input data
    test                    Test model on given input data
    predict                 Predict for given data
    ofo                     Use online f-measure optimization
    version                 Print napkinXC version
    help                    Print help

Args:
    General:
    -i, --input             Input dataset, required
    -o, --output            Output (model) dir, required
    -m, --model             Model type (default = plt)
                            Models: ovr, br, hsm, plt, oplt, svbopFull, svbopHf, brMips, svbopMips
    --ensemble              Number of models in ensemble (default = 1)
    -t, --threads           Number of threads to use (default = 0)
                            Note: set to -1 to use a number of available CPUs - 1, 0 to use a number of available CPUs
    --memLimit              Maximum amount of memory (in G) available for training (defualt = 0)
                            Note: set to 0 to set limit to amount of available memory
    --hash                  Size of features space (default = 0)
                            Note: set to 0 to disable hashing
    --featuresThreshold     Prune features below given threshold (default = 0.0)
    --seed                  Seed (default = system time)
    --verbose               Verbose level (default = 2)

    OVR and HSM:
    --pickOneLabelWeighting Allows to use multi-label data by transforming it into multi-class (default = 0)

    Base classifiers:
    --optim, --optimizer    Optimizer used for training binary classifiers (default = liblinear)
                            Optimizers: liblinear, sgd, adagrad
    --bias                  Value of the bias features (default = 1)
    --inbalanceLabelsWeighting      Increase the weight of minority labels in base classifiers (default = 0)
    --weightsThreshold      Threshold value for pruning models weights (default = 0.1)
    --loss                  Loss function to optimize in base classifier (default = log)
                            Losses: log (alias logistic), l2 (alias squaredHinge)

    LIBLINEAR:              (more about LIBLINEAR: https://github.com/cjlin1/liblinear)
    -c, --liblinearC        LIBLINEAR cost co-efficient, inverse of regularization strength, must be a positive float,
                            smaller values specify stronger regularization (default = 10.0)
    --eps, --liblinearEps   LIBLINEAR tolerance of termination criterion (default = 0.1)
    --solver, --liblinearSolver     LIBLINEAR solver (default for log loss = L2R_LR_DUAL, for l2 loss = L2R_L2LOSS_SVC_DUAL)
                                    Overrides default solver set by loss parameter.
                                    Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                                       L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
    --maxIter, --liblinearMaxIter   Maximum number of iterations for LIBLINEAR (default = 100)

    SGD/AdaGrad:
    -l, --lr, --eta         Step size (learning rate) for online optimizers (default = 1.0)
    --epochs                Number of training epochs for online optimizers (default = 1)
    --adagradEps            Defines starting step size for AdaGrad (default = 0.001)

    Tree (PLT and HSM):
    -a, --arity             Arity of tree nodes (default = 2)
    --maxLeaves             Maximum degree of pre-leaf nodes. (default = 100)
    --tree                  File with tree structure
    --treeType              Type of a tree to build if file with structure is not provided
                            tree types: hierarchicalKmeans, huffman, completeKaryInOrder, completeKaryRandom,
                                        balancedInOrder, balancedRandom, onlineComplete

    K-Means tree:
    --kmeansEps             Tolerance of termination criterion of the k-means clustering
                            used in hierarchical k-means tree building procedure (default = 0.001)
    --kmeansBalanced        Use balanced K-Means clustering (default = 1)

    Prediction:
    --topK                  Predict top-k labels (default = 5)
    --threshold             Predict labels with probability above the threshold (default = 0)
    --thresholds            Path to a file with threshold for each label, one threshold per line
    --labelsWeights         Path to a file with weight for each label, one weight per line
    --setUtility            Type of set-utility function for prediction using svbopFull, svbopHf, svbopMips models.
                            Set-utility functions: uP, uF1, uAlfa, uAlfaBeta, uDeltaGamma
                            See: https://arxiv.org/abs/1906.08129

    Set-Utility:
    --alpha
    --beta
    --delta
    --gamma

    Test:
    --measures              Evaluate test using set of measures (default = "p@1,p@3,p@5")
                            Measures: acc (accuracy), p (precision), r (recall), c (coverage), hl (hamming loos)
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
