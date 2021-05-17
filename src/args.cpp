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

#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "args.h"
#include "log.h"
#include "resources.h"
#include "version.h"

Args::Args() {
    parsedArgs = std::vector<std::string>();

    seed = time(nullptr);
    rngSeeder.seed(seed);
    threads = getCpuCount();
    memLimit = getSystemMemory();
    saveGrads = false;
    resume = false;
    loadDense = false;

    // Input/output options
    input = "";
    output = ".";
    prediction = "";
    modelName = "plt";
    modelType = plt;
    hash = 0;
    processData = true;
    bias = 1.0;
    norm = true;
    featuresThreshold = 0.0;

    // Training options
    eps = 0.1;
    cost = 10.0;
    maxIter = 100;
    autoCLin = false;
    autoCLog = false;

    lossType = logistic;
    lossName = "logistic";
    solverType = L2R_LR_DUAL;
    solverName = "L2R_LR_DUAL";
    inbalanceLabelsWeighting = false;
    pickOneLabelWeighting = false;
    optimizerName = "liblinear";
    optimizerType = liblinear;
    weightsThreshold = 0.1;

    // Ensemble options
    ensemble = 0;
    onTheTrotPrediction = false;

    // For online training
    eta = 1.0;
    epochs = 1;
    tmax = -1;
    l2Penalty = 0;
    adagradEps = 0.001;

    // Tree options
    treeStructure = "";
    arity = 2;
    treeType = hierarchicalKmeans;
    treeTypeName = "hierarchicalKmeans";
    maxLeaves = 100;

    // K-Means tree options
    kmeansEps = 0.0001;
    kmeansBalanced = true;
    kmeansWeightedFeatures = false;

    // Online PLT options
    onlineTreeAlpha = 0.5;

    // extremeText options
    dims = 100;

    // MACH options
    machHashes = 10;
    machBuckets = 100;

    // Prediction options
    topK = 5;
    threshold = 0.0;
    thresholds = "";
    ensMissingScores = true;

    // Mips options
    mipsDense = false;
    hnswM = 20;
    hnswEfConstruction = 100;
    hnswEfSearch = 100;

    // Set utility options
    svbopMipsK = 0.05;
    svbopInvIndexK = 1;

    setUtilityType = uP;
    alpha = 0.0;
    beta = 1.0;
    delta = 2.2;
    gamma = 1.2;


    // Measures for test command
    measures = "p@1,p@3,p@5";
    measuresPrecision = 6;

    // Args for OFO command
    ofoType = micro;
    ofoTypeName = "micro";
    ofoTopLabels = 1000;
    ofoA = 10;
    ofoB = 20;

    psA = 0.55;
    psB = 1.5;

    // Args for testPredictionTime command
    batchSizes = "100,1000,10000";
    batches = 10;
}

// Parse args
void Args::parseArgs(const std::vector<std::string>& args, bool keepArgs) {
    Log(CERR_DEBUG) << "Parsing args ...\n";

    if(keepArgs) parsedArgs.insert(parsedArgs.end(), args.begin(), args.end());

    for (int ai = 0; ai < args.size(); ai += 2) {
        Log(CERR_DEBUG) << "  " << args[ai] << " " << args[ai + 1] << "\n";

        if (args[ai][0] != '-')
            throw std::invalid_argument("Provided argument without a dash: " + args[ai]);

        try {
            if (args[ai] == "--verbose")
                logLevel = static_cast<LogLevel>(std::stoi(args.at(ai + 1)));

            else if (args[ai] == "--seed") {
                seed = std::stoi(args.at(ai + 1));
                rngSeeder.seed(seed);
            } else if (args[ai] == "-t" || args[ai] == "--threads") {
                threads = std::stoi(args.at(ai + 1));
                if (threads == 0)
                    threads = getCpuCount();
                else if (threads == -1)
                    threads = getCpuCount() - 1;
            } else if (args[ai] == "--memLimit") {
                memLimit = static_cast<unsigned long long>(std::stof(args.at(ai + 1)) * 1024 * 1024 * 1024);
                if (memLimit == 0) memLimit = getSystemMemory();
            } else if (args[ai] == "--saveGrads")
                saveGrads = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--resume")
                resume = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--loadDense")
                loadDense = std::stoi(args.at(ai + 1)) != 0;

            // Input/output options
            else if (args[ai] == "-i" || args[ai] == "--input")
                input = std::string(args.at(ai + 1));
            else if (args[ai] == "-o" || args[ai] == "--output")
                output = std::string(args.at(ai + 1));
            else if (args[ai] == "--prediction")
                prediction = std::string(args.at(ai + 1));
            else if (args[ai] == "--ensemble")
                ensemble = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--onTheTrotPrediction")
                onTheTrotPrediction = std::stoi(args.at(ai + 1));
            else if (args[ai] == "-m" || args[ai] == "--model") {
                modelName = args.at(ai + 1);
                if (args.at(ai + 1) == "br")
                    modelType = br;
                else if (args.at(ai + 1) == "ovr")
                    modelType = ovr;
                else if (args.at(ai + 1) == "hsm")
                    modelType = hsm;
                else if (args.at(ai + 1) == "plt")
                    modelType = plt;
                else if (args.at(ai + 1) == "svbopFull")
                    modelType = svbopFull;
                else if (args.at(ai + 1) == "svbopFagin")
                    modelType = svbopFagin;
                else if (args.at(ai + 1) == "svbopThreshold")
                    modelType = svbopThreshold;
                else if (args.at(ai + 1) == "svbopInvertedIndex")
                    modelType = svbopInvertedIndex;
                else if (args.at(ai + 1) == "svbopHf")
                    modelType = svbopHf;
                else if (args.at(ai + 1) == "oplt")
                    modelType = oplt;
                else if (args.at(ai + 1) == "xt" || args.at(ai + 1) == "extremeText")
                    modelType = extremeText;
                else if (args.at(ai + 1) == "mach")
                    modelType = mach;
// Mips extension models
#ifdef MIPS_EXT
                else if (args.at(ai + 1) == "brMips")
                    modelType = brMips;
                else if (args.at(ai + 1) == "svbopMips")
                    modelType = svbopMips;
#else
                else if (args.at(ai + 1) == "brMips" || args.at(ai + 1) == "svbopMips")
                    throw std::invalid_argument(args.at(ai + 1) + " model requires MIPS extension");
#endif
                else
                    throw std::invalid_argument("Unknown model type: " + args.at(ai + 1));
            } else if (args[ai] == "--mipsDense")
                mipsDense = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--hnswM")
                hnswM = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--hnswEfConstruction")
                hnswEfConstruction = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--hnswEfSearch")
                hnswEfSearch = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--svbopMipsK")
                svbopMipsK = std::stof(args.at(ai + 1));
            else if (args[ai] == "--svbopInvIndexK")
                svbopInvIndexK = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--setUtility") {
                setUtilityName = args.at(ai + 1);
                if (args.at(ai + 1) == "uP")
                    setUtilityType = uP;
                else if (args.at(ai + 1) == "uR")
                    setUtilityType = uP;
                else if (args.at(ai + 1) == "uF1")
                    setUtilityType = uF1;
                else if (args.at(ai + 1) == "uFBeta")
                    setUtilityType = uFBeta;
                else if (args.at(ai + 1) == "uExp")
                    setUtilityType = uExp;
                else if (args.at(ai + 1) == "uLog")
                    setUtilityType = uLog;
                else if (args.at(ai + 1) == "uDeltaGamma")
                    setUtilityType = uDeltaGamma;
                else if (args.at(ai + 1) == "uAlpha")
                    setUtilityType = uAlpha;
                else if (args.at(ai + 1) == "uAlphaBeta")
                    setUtilityType = uAlphaBeta;
                else
                    throw std::invalid_argument("Unknown set utility type: " + args.at(ai + 1));
            } else if (args[ai] == "--alpha")
                alpha = std::stof(args.at(ai + 1));
            else if (args[ai] == "--beta")
                beta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--delta")
                delta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--gamma")
                gamma = std::stof(args.at(ai + 1));

            else if (args[ai] == "--bias")
                bias = std::stof(args.at(ai + 1));
            else if (args[ai] == "--norm")
                norm = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--hash")
                hash = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--featuresThreshold")
                featuresThreshold = std::stof(args.at(ai + 1));
            else if (args[ai] == "--weightsThreshold")
                weightsThreshold = std::stof(args.at(ai + 1));

            // Training options
            else if (args[ai] == "-e" || args[ai] == "--eps" || args[ai] == "--liblinearEps")
                eps = std::stof(args.at(ai + 1));
            else if (args[ai] == "-c" || args[ai] == "-C" || args[ai] == "--cost" || args[ai] == "--liblinearC")
                cost = std::stof(args.at(ai + 1));
            else if (args[ai] == "--maxIter" || args[ai] == "--liblinearMaxIter")
                maxIter = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--inbalanceLabelsWeighting")
                inbalanceLabelsWeighting = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--pickOneLabelWeighting")
                pickOneLabelWeighting = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--loss") {
                lossName = args.at(ai + 1);
                if (args.at(ai + 1) == "logistic" || args.at(ai + 1) == "log")
                    lossType = logistic;
                else if (args.at(ai + 1) == "squaredHinge" || args.at(ai + 1) == "l2")
                    lossType = squaredHinge;
                else if (args.at(ai + 1) == "pwLogistic" || args.at(ai + 1) == "pwLog")
                    lossType = pwLogistic;
                else if (args.at(ai + 1) == "unLogistic" || args.at(ai + 1) == "unLog")
                    lossType = unLogistic;
                else
                    throw std::invalid_argument("Unknown loss type: " + args.at(ai + 1));
            }
            else if (args[ai] == "--solver" || args[ai] == "--liblinearSolver") {
                solverName = args.at(ai + 1);
                if (args.at(ai + 1) == "L2R_LR_DUAL")
                    solverType = L2R_LR_DUAL;
                else if (args.at(ai + 1) == "L2R_LR")
                    solverType = L2R_LR;
                else if (args.at(ai + 1) == "L1R_LR")
                    solverType = L1R_LR;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC_DUAL")
                    solverType = L2R_L2LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC")
                    solverType = L2R_L2LOSS_SVC;
                else if (args.at(ai + 1) == "L2R_L1LOSS_SVC_DUAL")
                    solverType = L2R_L1LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L1R_L2LOSS_SVC")
                    solverType = L1R_L2LOSS_SVC;
                else
                    throw std::invalid_argument("Unknown solver type: " + args.at(ai + 1));
            } else if (args[ai] == "--optim" || args[ai] == "--optimizer") {
                optimizerName = args.at(ai + 1);
                if (args.at(ai + 1) == "liblinear")
                    optimizerType = liblinear;
                else if (args.at(ai + 1) == "sgd")
                    optimizerType = sgd;
                else if (args.at(ai + 1) == "adagrad")
                    optimizerType = adagrad;
                else
                    throw std::invalid_argument("Unknown optimizer type: " + args.at(ai + 1));
            } else if (args[ai] == "-l" || args[ai] == "--lr" || args[ai] == "--learningRate" || args[ai] == "--eta")
                eta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--epochs")
                epochs = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--tmax")
                tmax = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--adagradEps")
                adagradEps = std::stof(args.at(ai + 1));
            else if (args[ai] == "--l2Penalty")
                l2Penalty = std::stof(args.at(ai + 1));
            else if (args[ai] == "--dims")
                dims = std::stoi(args.at(ai + 1));

            // Tree options
            else if (args[ai] == "-a" || args[ai] == "--arity")
                arity = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--maxLeaves")
                maxLeaves = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--kmeansEps")
                kmeansEps = std::stof(args.at(ai + 1));
            else if (args[ai] == "--kmeansBalanced")
                kmeansBalanced = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--kmeansWeightedFeatures")
                kmeansWeightedFeatures = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--treeStructure") {
                treeStructure = std::string(args.at(ai + 1));
                treeType = custom;
            } else if (args[ai] == "--treeType") {
                treeTypeName = args.at(ai + 1);
                if (args.at(ai + 1) == "completeInOrder")
                    treeType = completeInOrder;
                else if (args.at(ai + 1) == "completeRandom")
                    treeType = completeRandom;
                else if (args.at(ai + 1) == "balancedInOrder")
                    treeType = balancedInOrder;
                else if (args.at(ai + 1) == "balancedRandom")
                    treeType = balancedRandom;
                else if (args.at(ai + 1) == "hierarchicalKmeans")
                    treeType = hierarchicalKmeans;
                else if (args.at(ai + 1) == "huffman")
                    treeType = huffman;
                else if (args.at(ai + 1) == "onlineKaryComplete")
                    treeType = onlineKaryComplete;
                else if (args.at(ai + 1) == "onlineKaryRandom")
                    treeType = onlineKaryRandom;
                else if (args.at(ai + 1) == "onlineRandom")
                    treeType = onlineRandom;
                else if (args.at(ai + 1) == "onlineBestScore")
                    treeType = onlineBestScore;
                else
                    throw std::invalid_argument("Unknown tree type: " + args.at(ai + 1));
            } else if (args[ai] == "--onlineTreeAlpha")
                onlineTreeAlpha = std::stof(args.at(ai + 1));

            // MACH options
            else if (args[ai] == "--machHashes")
                machHashes = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--machBuckets")
                machBuckets = std::stoi(args.at(ai + 1)) != 0;

            // OFO options
            else if (args[ai] == "--ofoType") {
                ofoTypeName = args.at(ai + 1);
                if (args.at(ai + 1) == "micro")
                    ofoType = micro;
                else if (args.at(ai + 1) == "macro")
                    ofoType = macro;
                else if (args.at(ai + 1) == "mixed")
                    ofoType = mixed;
                else
                    throw std::invalid_argument("Unknown ofo type: " + args.at(ai + 1));
            } else if (args[ai] == "--ofoTopLabels")
                ofoTopLabels = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--ofoA")
                ofoA = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--ofoB")
                ofoB = std::stoi(args.at(ai + 1));

            // Prediction/test options
            else if (args[ai] == "--topK")
                topK = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--threshold")
                threshold = std::stof(args.at(ai + 1));
            else if (args[ai] == "--thresholds")
                thresholds = std::string(args.at(ai + 1));
            else if (args[ai] == "--labelsWeights")
                labelsWeights = std::string(args.at(ai + 1));
            else if (args[ai] == "--ensMissingScores")
                ensMissingScores = std::stoi(args.at(ai + 1)) != 0;

            else if (args[ai] == "--batchSizes")
                batchSizes = args.at(ai + 1);
            else if (args[ai] == "--batches")
                batches = std::stoi(args.at(ai + 1));

            else if (args[ai] == "--measures")
                measures = std::string(args.at(ai + 1));
            else if (args[ai] == "--measuresPrecision")
                measuresPrecision = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--autoCLin")
                autoCLin = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--autoCLog")
                autoCLog = std::stoi(args.at(ai + 1)) != 0;
            else
                throw std::invalid_argument("Unknown argument: " + args[ai]);

        } catch (std::out_of_range& e) {
            throw std::invalid_argument(args[ai] + " is missing an argument");
        }
    }

//    if (input.empty() || output.empty())
//        throw std::invalid_argument("Empty input or model path");

    // Change default values for specific cases + parameters warnings
    if (optimizerType == liblinear) {
        if (countArgs(args, {"-s", "--solver", "--liblinearSolver"}) and countArg(args, "--loss"))
            Log(CERR) << "Warning: Default solver for " << lossName << " will be overridden by " << solverName << " solver!\n";
        else{
            if(lossType == logistic){
                solverType = L2R_LR_DUAL;
                solverName = "L2R_LR_DUAL";
            }
            if(lossType == squaredHinge){
                solverType = L2R_L2LOSS_SVC_DUAL;
                solverName = "L2R_L2LOSS_SVC_DUAL";
            }
        }
    }

    if (modelType == oplt && optimizerType == liblinear) {
        if (countArgs(args, {"--optim", "--optimizer"}))
            Log(CERR) << "Online PLT does not support " << optimizerName << " optimizer! Changing to AdaGrad.\n";
        optimizerType = adagrad;
        optimizerName = "adagrad";
    }

    if (modelType == oplt && resume && (treeType != onlineRandom && treeType != onlineBestScore)) {
        if (countArg(args, "--treeType"))
            Log(CERR) << "Warning: Resuming training for Online PLT does not support " << treeTypeName
                      << " tree type! Changing to onlineBestScore.\n";
        treeType = onlineBestScore;
        treeTypeName = "onlineBestScore";
    }

    // If only threshold used set topK to 0, otherwise display warning
    if (threshold > 0) {
        if (countArg(args, "--topK"))
            Log(CERR) << "Warning: Top K and threshold prediction are used at the same time!\n";
        else
            topK = 0;
    }
}

void Args::printArgs(std::string command) {
    Log(CERR) << "napkinXC " << VERSION << " - " << command;
    if (!input.empty())
        Log(CERR) << "\n  Input: " << input << "\n    Bias: " << bias << ", norm: " << norm
                  << ", hash size: " << hash << ", features threshold: " << featuresThreshold;
    Log(CERR) << "\n  Model: " << output << "\n    Type: " << modelName;
    if (ensemble > 1) Log(CERR) << ", ensemble: " << ensemble;

    if (command == "train") {
        Log(CERR) << "\n  Base models optimizer: " << optimizerName;
        if (optimizerType == liblinear)
            Log(CERR) << "\n    Solver: " << solverName << ", eps: " << eps << ", cost: " << cost << ", max iter: " << maxIter;
        else
            Log(CERR) << "\n    Loss: " << lossName << ", eta: " << eta << ", epochs: " << epochs;
        if (optimizerType == adagrad) Log(CERR) << ", AdaGrad eps " << adagradEps;
        Log(CERR) << ", weights threshold: " << weightsThreshold;

        if (modelType == plt || modelType == hsm || modelType == oplt || modelType == svbopHf) {
            if (treeStructure.empty()) {
                Log(CERR) << "\n  Tree type: " << treeTypeName << ", arity: " << arity;
                if (treeType == hierarchicalKmeans)
                    Log(CERR) << ", k-means eps: " << kmeansEps << ", balanced: " << kmeansBalanced
                              << ", weighted features: " << kmeansWeightedFeatures;
                if (treeType == hierarchicalKmeans || treeType == balancedInOrder || treeType == balancedRandom
                    || treeType == onlineBestScore || treeType == onlineRandom)
                    Log(CERR) << ", max leaves: " << maxLeaves;
                if (treeType == onlineBestScore)
                    Log(CERR) << ", alpha: " << onlineTreeAlpha;
            } else {
                Log(CERR) << "\n    Tree: " << treeStructure;
            }
        }
    }

    if(!labelsWeights.empty()) Log(CERR) << "\n  Label weights: " << labelsWeights;

    if (command == "test" || command == "predict") {
        if(thresholds.empty()) Log(CERR) << "\n  Top k: " << topK << ", threshold: " << threshold;
        else Log(CERR) << "\n  Thresholds: " << thresholds;
        if(!labelsWeights.empty()) Log(CERR) << "\n  Labels' weights: " << labelsWeights;
        if (modelType == svbopMips || modelType == brMips) {
            Log(CERR) << "\n  HNSW: M: " << hnswM << ", efConst.: " << hnswEfConstruction << ", efSearch: " << hnswEfSearch;
            if(modelType == svbopMips) Log(CERR) << ", k: " << svbopMipsK;
        }
        if (modelType == svbopFull || modelType == svbopHf || modelType == svbopMips) {
            Log(CERR) << "\n  Set utility: " << setUtilityName;
            if (setUtilityType == uAlpha || setUtilityType == uAlphaBeta) Log(CERR) << ", alpha: " << alpha;
            if (setUtilityType == uAlphaBeta) Log(CERR) << ", beta: " << beta;
            if (setUtilityType == uDeltaGamma) Log(CERR) << ", delta: " << delta << ", gamma: " << gamma;
        }
    }

    if (command == "ofo")
        Log(CERR) << "\n  Epochs: " << epochs << ", initial a: " << ofoA << ", initial b: " << ofoB;

    Log(CERR) << "\n  Threads: " << threads << ", memory limit: " << formatMem(memLimit)
              << "\n  Seed: " << seed << "\n";
}

int Args::countArg(const std::vector<std::string>& args, std::string to_count){
    return std::count(args.begin(), args.end(), to_count);
}

int Args::countArgs(const std::vector<std::string>& args, std::vector<std::string> to_count){
    int count = 0;
    for(const auto &tc: to_count)
        count += std::count(args.begin(), args.end(), tc);
    return count;
}

void Args::save(std::ostream& out) {
    std::string version = VERSION;
    saveVar(out, version);

    saveVar(out, bias);
    saveVar(out, norm);
    saveVar(out, hash);
    saveVar(out, weightsThreshold);

    saveVar(out, modelType);
    saveVar(out, modelName);
    saveVar(out, ensemble);
}

void Args::load(std::istream& in) {
    std::string version;
    loadVar(in, version);
    if(version != VERSION)
        Log(CERR) << "Warning: Model version (" << version << ") does not match napkinXC version (" << VERSION << "), something may not work correctly!\n";

    loadVar(in, bias);
    loadVar(in, norm);
    loadVar(in, hash);
    loadVar(in, weightsThreshold);

    loadVar(in, modelType);
    loadVar(in, modelName);
    loadVar(in, ensemble);

    parseArgs(parsedArgs, false);
}
