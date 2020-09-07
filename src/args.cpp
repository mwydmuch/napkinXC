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
    command = "";
    seed = time(nullptr);
    rngSeeder.seed(seed);

    // Input/output options
    input = "";
    output = ".";
    dataFormatName = "libsvm";
    dataFormatType = libsvm;
    modelName = "plt";
    modelType = plt;
    header = true;
    hash = 0;
    bias = 1.0;
    norm = true;
    featuresThreshold = 0.0;

    // Training options
    threads = getCpuCount();
    memLimit = getSystemMemory();
    eps = 0.1;
    cost = 16.0;
    maxIter = 100;
    autoCLin = false;
    autoCLog = false;

    solverType = L2R_LR_DUAL;
    solverName = "L2R_LR_DUAL";
    lossType = logistic;
    lossName = "logistic";
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
    dims = 100;

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
    ubopMipsK = 0.05;

    setUtilityType = uP;
    alpha = 0.0;
    beta = 1.0;
    delta = 2.2;
    gamma = 1.2;


    // Measures for test command
    measures = "p@1,r@1,c@1,p@3,r@3,c@3,p@5,r@5,c@5";

    // Args for OFO command
    ofoType = micro;
    ofoTypeName = "micro";
    ofoTopLabels = 1000;
    ofoA = 10;
    ofoB = 20;

    // Args for testPredictionTime command
    batchSizes = "100,1000,10000";
    batches = 10;
}

// Parse args
void Args::parseArgs(const std::vector<std::string>& args) {
    LOG(CERR_DEBUG) << "Parsing args...\n";

    for (int ai = 0; ai < args.size(); ai += 2) {
        LOG(CERR_DEBUG) << "  " << args[ai] << " " << args[ai + 1] << "\n";

        if (args[ai][0] != '-')
            throw std::invalid_argument("Provided argument without a dash: " + args[ai]);

        try {
            if (args[ai] == "--verbose")
                logLevel = static_cast<LogLevel>(std::stoi(args.at(ai + 1)));

            else if (args[ai] == "--seed") {
                seed = std::stoi(args.at(ai + 1));
                rngSeeder.seed(seed);
            }

            // Input/output options
            else if (args[ai] == "-i" || args[ai] == "--input")
                input = std::string(args.at(ai + 1));
            else if (args[ai] == "-o" || args[ai] == "--output")
                output = std::string(args.at(ai + 1));
            else if (args[ai] == "-d" || args[ai] == "--dataFormat") {
                dataFormatName = args.at(ai + 1);
                if (args.at(ai + 1) == "libsvm")
                    dataFormatType = libsvm;
                else if (args.at(ai + 1) == "vw" || args.at(ai + 1) == "vowpalwabbit")
                    dataFormatType = vw;
                else
                    throw std::invalid_argument("Unknown date format type: " + args.at(ai + 1));
            } else if (args[ai] == "--ensemble")
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
                else if (args.at(ai + 1) == "ubop")
                    modelType = ubop;
                else if (args.at(ai + 1) == "ubopHsm")
                    modelType = ubopHsm;
                else if (args.at(ai + 1) == "oplt")
                    modelType = oplt;
                else if (args.at(ai + 1) == "extremeText")
                    modelType = extremeText;
// Mips extension models
#ifdef MIPS_EXT
                else if (args.at(ai + 1) == "brMips")
                    modelType = brMips;
                else if (args.at(ai + 1) == "ubopMips")
                    modelType = ubopMips;
#else
                else if (args.at(ai + 1) == "brMips" || args.at(ai + 1) == "ubopMips")
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
            else if (args[ai] == "--ubopMipsK")
                ubopMipsK = std::stof(args.at(ai + 1));
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

            else if (args[ai] == "--header")
                header = std::stoi(args.at(ai + 1)) != 0;
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
            else if (args[ai] == "-t" || args[ai] == "--threads") {
                threads = std::stoi(args.at(ai + 1));
                if (threads == 0)
                    threads = getCpuCount();
                else if (threads == -1)
                    threads = getCpuCount() - 1;
            } else if (args[ai] == "--memLimit") {
                memLimit = static_cast<unsigned long long>(std::stof(args.at(ai + 1)) * 1024 * 1024 * 1024);
                if (memLimit == 0) memLimit = getSystemMemory();
            } else if (args[ai] == "-e" || args[ai] == "--eps" || args[ai] == "--liblinearEps")
                eps = std::stof(args.at(ai + 1));
            else if (args[ai] == "-c" || args[ai] == "-C" || args[ai] == "--cost" || args[ai] == "--liblinearC")
                cost = std::stof(args.at(ai + 1));
            else if (args[ai] == "--maxIter")
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
                else
                    throw std::invalid_argument("Unknown loss type: " + args.at(ai + 1));
            }
            else if (args[ai] == "--solver") {
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
            } else if (args[ai] == "--optimizer") {
                optimizerName = args.at(ai + 1);
                if (args.at(ai + 1) == "liblinear")
                    optimizerType = liblinear;
                else if (args.at(ai + 1) == "sgd")
                    optimizerType = sgd;
                else if (args.at(ai + 1) == "adagrad")
                    optimizerType = adagrad;
                else
                    throw std::invalid_argument("Unknown optimizer type: " + args.at(ai + 1));
            } else if (args[ai] == "-l" || args[ai] == "--lr" || args[ai] == "--eta")
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
            else if (args[ai] == "--ensMissingScores")
                ensMissingScores = std::stoi(args.at(ai + 1)) != 0;

            else if (args[ai] == "--batchSizes")
                batchSizes = args.at(ai + 1);
            else if (args[ai] == "--batches")
                batches = std::stoi(args.at(ai + 1));

            else if (args[ai] == "--measures")
                measures = std::string(args.at(ai + 1));
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
    if (modelType == oplt && optimizerType == liblinear) {
        if (count(args.begin(), args.end(), "optimizer"))
            LOG(CERR) << "Online PLT does not support " << optimizerName << " optimizer! Changing to AdaGrad.\n";
        optimizerType = adagrad;
        optimizerName = "adagrad";
    }

    if (modelType == oplt && (treeType == hierarchicalKmeans || treeType == huffman)) {
        if (count(args.begin(), args.end(), "treeType"))
            LOG(CERR) << "Online PLT does not support " << treeTypeName
                      << " tree type! Changing to complete in order tree.\n";
        treeType = onlineBestScore;
        treeTypeName = "onlineBestScore";
    }

    // If only threshold used set topK to 0, otherwise display warning
    if (threshold > 0) {
        if (count(args.begin(), args.end(), "topK"))
            LOG(CERR) << "Warning: Top K and threshold prediction are used at the same time!\n";
        else
            topK = 0;
    }

    // Warnings about arguments overrides while testing / predicting
}

void Args::printArgs() {
    LOG(CERR) << "napkinXC " << VERSION
              << "\n  Input: " << input << "\n    Data format: " << dataFormatName
              << "\n    Header: " << header << ", bias: " << bias << ", norm: " << norm << ", hash size: " << hash << ", features threshold: " << featuresThreshold
              << "\n  Model: " << output << "\n    Type: " << modelName;

    if (ensemble > 1) LOG(CERR) << ", ensemble: " << ensemble;

    if (command == "train") {
        LOG(CERR) << "\n  Base models optimizer: " << optimizerName;
        if (optimizerType == liblinear)
            LOG(CERR) << "\n    Solver: " << solverName << ", eps: " << eps << ", cost: " << cost << ", max iter: " << maxIter;
        else
            LOG(CERR) << "\n    Loss: " << lossName << ", eta: " << eta << ", epochs: " << epochs;
        if (optimizerType == adagrad) LOG(CERR) << ", AdaGrad eps " << adagradEps;
        LOG(CERR) << ", weights threshold: " << weightsThreshold;

        if (modelType == plt || modelType == hsm || modelType == oplt || modelType == ubopHsm) {
            if (treeStructure.empty()) {
                LOG(CERR) << "\n  Tree type: " << treeTypeName << ", arity: " << arity;
                if (treeType == hierarchicalKmeans)
                    LOG(CERR) << ", k-means eps: " << kmeansEps << ", balanced: " << kmeansBalanced
                              << ", weighted features: " << kmeansWeightedFeatures;
                if (treeType == hierarchicalKmeans || treeType == balancedInOrder || treeType == balancedRandom)
                    LOG(CERR) << ", max leaves: " << maxLeaves;
            } else {
                LOG(CERR) << "\n    Tree: " << treeStructure;
            }
        }
    }

    if (command == "test") {
        if(thresholds.empty()) LOG(CERR) << "\n  Top k: " << topK << ", threshold: " << threshold;
        else LOG(CERR) << "\n  Thresholds: " << thresholds;
        if (modelType == ubopMips || modelType == brMips) {
            LOG(CERR) << "\n  HNSW: M: " << hnswM << ", efConst.: " << hnswEfConstruction << ", efSearch: " << hnswEfSearch;
            if(modelType == ubopMips) LOG(CERR) << ", k: " << ubopMipsK;
        }
        if (modelType == ubop || modelType == ubopHsm || modelType == ubopMips) {
            LOG(CERR) << "\n  Set utility: " << setUtilityName;
            if (setUtilityType == uAlpha || setUtilityType == uAlphaBeta) LOG(CERR) << ", alpha: " << alpha;
            if (setUtilityType == uAlphaBeta) LOG(CERR) << ", beta: " << beta;
            if (setUtilityType == uDeltaGamma) LOG(CERR) << ", delta: " << delta << ", gamma: " << gamma;
        }
    }

    if (command == "ofo")
        LOG(CERR) << "\n  Epochs: " << epochs << ", a: " << ofoA << ", b: " << ofoB;

    LOG(CERR) << "\n  Threads: " << threads << ", memory limit: " << formatMem(memLimit)
              << "\n  Seed: " << seed << "\n";
}

void Args::save(std::ostream& out) {
    out.write((char*)&bias, sizeof(bias));
    out.write((char*)&norm, sizeof(norm));
    out.write((char*)&hash, sizeof(hash));
    out.write((char*)&modelType, sizeof(modelType));
    out.write((char*)&dataFormatType, sizeof(dataFormatType));
    // out.write((char*) &ensemble, sizeof(ensemble));

    saveVar(out, modelName);
    saveVar(out, dataFormatName);
}

void Args::load(std::istream& in) {
    in.read((char*)&bias, sizeof(bias));
    in.read((char*)&norm, sizeof(norm));
    in.read((char*)&hash, sizeof(hash));
    in.read((char*)&modelType, sizeof(modelType));
    in.read((char*)&dataFormatType, sizeof(dataFormatType));
    // in.read((char*) &ensemble, sizeof(ensemble));

    loadVar(in, modelName);
    loadVar(in, dataFormatName);
}
