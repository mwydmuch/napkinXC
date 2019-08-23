/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <iomanip>

#include "args.h"
#include "linear.h"
#include "utils.h"

Args::Args() {
    command = "";
    seed = time(0);

    // Input/output options
    input = "";
    output = "";
    dataFormatName = "libsvm";
    dataFormatType = libsvm;
    modelName = "plt";
    modelType = plt;
    header = true;
    hash = 0;
    bias = true;
    biasValue = 1.0;
    norm = true;
    tfidf = false;
    pruneThreshold = 0.0;

    // Training options
    threads = getCpuCount();
    memLimit = 0;
    mem = 16;
    eps = 0.01;
    cost = 10.0;
    solverType = L2R_LR_DUAL;
    solverName = "L2R_LR_DUAL";
    labelsWeights = false;
    optimizerName = "libliner";
    optimizerType = libliner;
    iter = 50;
    eta = 0.5;
    weightsThreshold = 0.1;
    ensemble = 1;

    // Tree options
    treeStructure = "";
    arity = 2;
    //treeType = completeInOrder;
    //treeTypeName = "completeInOrder";
    treeType = hierarchicalKMeans;
    treeTypeName = "hierarchicalKMeans";
    maxLeaves = 100;

    // Tree sampling
    sampleK = 50;

    // K-Means tree options
    kMeansEps = 0.0001;
    kMeansBalanced = true;
    kMeansWeightedFeatures = false;

    // Prediction options
    topK = 1;
    sparseWeights = true;

    // Set utility options
    setUtilityType = uAlfaBeta;
    alfa = 1.0;
    beta = 1.0;
    epsilon = 0.0;
    delta = 0.0;
    gamma = 0.0;

    // Measures
    measures = "p@1,r@1,c@1,p@3,r@3,c@3,p@5,r@5,c@5";
}

// Parse args
void Args::parseArgs(const std::vector<std::string>& args) {
    command = args[1];

    if(command != "train" && command != "test"){
        std::cerr << "Unknown command type: " << command << std::endl;
        printHelp();
    }

    for (int ai = 2; ai < args.size(); ai += 2) {
        if (args[ai][0] != '-') {
            std::cerr << "Provided argument without a dash! Usage:" << std::endl;
            printHelp();
        }

        try {
            if (args[ai] == "-h" || args[ai] == "--help") {
                std::cerr << "Here is the help! Usage:" << std::endl;
                printHelp();
            }
            else if (args[ai] == "--seed")
                seed = std::stoi(args.at(ai + 1));

            // Input/output options
            else if (args[ai] == "-i" || args[ai] == "--input")
                input = std::string(args.at(ai + 1));
            else if (args[ai] == "-o" || args[ai] == "--output")
                output = std::string(args.at(ai + 1));
            else if (args[ai] == "-d" || args[ai] == "--dataFormat") {
                treeTypeName = args.at(ai + 1);
                if (args.at(ai + 1) == "libsvm") dataFormatType = libsvm;
                else {
                    std::cerr << "Unknown date format type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "--ensemble")
                ensemble = std::stoi(args.at(ai + 1));
            else if (args[ai] == "-m" || args[ai] == "--model") {
                modelName = args.at(ai + 1);
                if (args.at(ai + 1) == "ovr") modelType = ovr;
                else if (args.at(ai + 1) == "br") modelType = br;
                else if (args.at(ai + 1) == "hsm") modelType = hsm;
                else if (args.at(ai + 1) == "hsmEns") modelType = hsmEns;
                else if (args.at(ai + 1) == "plt") modelType = plt;
                else if (args.at(ai + 1) == "pltEns") modelType = pltEns;
                else if (args.at(ai + 1) == "pltNeg") modelType = pltNeg;
                else if (args.at(ai + 1) == "brpltNeg") modelType = brpltNeg;
                else if (args.at(ai + 1) == "ubop") modelType = ubop;
                else if (args.at(ai + 1) == "rbop") modelType = rbop;
                else if (args.at(ai + 1) == "ubopch") modelType = ubopch;
                else {
                    std::cerr << "Unknown model type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "--setUtility") {
                setUtilityName = args.at(ai + 1);
                if (args.at(ai + 1) == "uP") setUtilityType = uP;
                else if (args.at(ai + 1) == "uF1") setUtilityType = uF1;
                else if (args.at(ai + 1) == "uAlfaBeta") setUtilityType = uAlfaBeta;
                else {
                    std::cerr << "Unknown set utility type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "--alfa")
                alfa = std::stof(args.at(ai + 1));
            else if (args[ai] == "--beta")
                beta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--epsilon")
                epsilon = std::stof(args.at(ai + 1));
            else if (args[ai] == "--delta")
                delta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--gamma")
                gamma = std::stof(args.at(ai + 1));

            else if (args[ai] == "--header")
                header = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--bias")
                bias = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--norm")
                norm = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--tfidf")
                tfidf = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--hash")
                hash = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--pruneThreshold")
                pruneThreshold = std::stof(args.at(ai + 1));
            else if (args[ai] == "--weightsThreshold")
                weightsThreshold = std::stof(args.at(ai + 1));
            else if (args[ai] == "--eta")
                eta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--iter")
                iter = std::stoi(args.at(ai + 1));

            // Training options
            else if (args[ai] == "-t" || args[ai] == "--threads"){
                threads = std::stoi(args.at(ai + 1));
                if(threads == 0) threads = getCpuCount();
                else if(threads == -1) threads = getCpuCount() - 1;
            } else if (args[ai] == "-e" || args[ai] == "--eps")
                eps = std::stof(args.at(ai + 1));
            else if (args[ai] == "-c" || args[ai] == "-C" || args[ai] == "--cost")
                cost = std::stof(args.at(ai + 1));
            else if (args[ai] == "--labelsWeights")
                labelsWeights = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--solver") {
                solverName = args.at(ai + 1);
                if (args.at(ai + 1) == "L2R_LR_DUAL") solverType = L2R_LR_DUAL;
                else if (args.at(ai + 1) == "L2R_LR") solverType = L2R_LR;
                else if (args.at(ai + 1) == "L1R_LR") solverType = L1R_LR;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC_DUAL") solverType = L2R_L2LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L2R_L2LOSS_SVC") solverType = L2R_L2LOSS_SVC;
                else if (args.at(ai + 1) == "L2R_L1LOSS_SVC_DUAL") solverType = L2R_L1LOSS_SVC_DUAL;
                else if (args.at(ai + 1) == "L1R_L2LOSS_SVC") solverType = L1R_L2LOSS_SVC;
                else {
                    std::cerr << "Unknown solver type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "--optimizer") {
                optimizerName = args.at(ai + 1);
                if (args.at(ai + 1) == "liblinear") optimizerType = libliner;
                else if (args.at(ai + 1) == "sgd") optimizerType = sgd;
                else{
                    std::cerr << "Unknown optimizer type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "-e" || args[ai] == "--eta")
                eta = std::stof(args.at(ai + 1));
            else if (args[ai] == "--iter")
                iter = std::stoi(args.at(ai + 1));

            // Tree options
            else if (args[ai] == "-a" || args[ai] == "--arity")
                arity = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--maxLeaves")
                maxLeaves = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--kMeansEps")
                kMeansEps = std::stof(args.at(ai + 1));
            else if (args[ai] == "--kMeansBalanced")
                kMeansBalanced = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--kMeansWeightedFeatures")
                kMeansWeightedFeatures = std::stoi(args.at(ai + 1)) != 0;
            else if (args[ai] == "--treeStructure") {
                treeStructure = std::string(args.at(ai + 1));
                treeType = custom;
            }
            else if (args[ai] == "--treeType") {
                treeTypeName = args.at(ai + 1);
                if (args.at(ai + 1) == "completeInOrder") treeType = completeInOrder;
                else if (args.at(ai + 1) == "completeRandom") treeType = completeRandom;
                else if (args.at(ai + 1) == "balancedInOrder") treeType = balancedInOrder;
                else if (args.at(ai + 1) == "balancedRandom") treeType = balancedRandom;
                else if (args.at(ai + 1) == "hierarchicalKMeans") treeType = hierarchicalKMeans;
                else if (args.at(ai + 1) == "huffman") treeType = huffman;
                else {
                    std::cerr << "Unknown tree type: " << args.at(ai + 1) << std::endl;
                    printHelp();
                }
            }
            else if (args[ai] == "--sampleK")
                sampleK = std::stoi(args.at(ai + 1));

            // Prediction options
            else if (args[ai] == "--topK")
                topK = std::stoi(args.at(ai + 1));
            else if (args[ai] == "--sparseWeights")
                sparseWeights = std::stoi(args.at(ai + 1)) != 0;
            else {
                std::cerr << "Unknown argument: " << args[ai] << std::endl;
                printHelp();
            }
        }
        catch (std::out_of_range) {
            std::cerr << args[ai] << " is missing an argument" << std::endl;
            printHelp();
        }
    }

    if (input.empty() || output.empty()) {
        std::cerr << "Empty input or model path." << std::endl;
        printHelp();
    }
}

void Args::printArgs(){
    if (command == "train" || command == "test"){
        std::cerr << "napkinXML - " << command
            << "\n  Input: " << input
            << "\n    Data format: " << dataFormatType
            << "\n    Header: " << header << ", bias: " << bias << ", norm: " << norm << ", prune threshold: " << pruneThreshold
            << "\n  Model: " << output
            << "\n    Type: " << modelName;
        if(command == "train") {
            std::cerr << "\n    Optimizer: " << optimizerName;
            if (optimizerType == libliner)
                std::cerr << "\n    LibLinear: Solver: " << solverName << ", eps: " << eps << ", cost: " << cost
                          << ", weights threshold: " << weightsThreshold;
            else if (optimizerType == sgd)
                std::cerr << "\n    SGD: eta: " << eta << ", iter: " << iter << ", threshold: " << weightsThreshold;
        }
        if(modelType == plt){
            if(treeStructure.empty()) {
                std::cerr << "\n    Tree type: " << treeTypeName << ", arity: " << arity;
                if (treeType == hierarchicalKMeans)
                    std::cerr << ", k-means eps: " << kMeansEps
                              << ", balanced: " << kMeansBalanced << ", weighted features: " << kMeansWeightedFeatures;
                if (treeType == hierarchicalKMeans || treeType == balancedInOrder || treeType == balancedRandom)
                    std::cerr << ", max leaves: " << maxLeaves;
            }
            else {
                std::cerr << "\n    Tree: " << treeStructure;
            }
        }
        if(command == "test" && (modelType == ubop || modelType == rbop || modelType == ubopch)) {
            std::cerr << "\n  Set utility: " << setUtilityName << ", alfa: " << alfa
                      << ", beta: " << beta << ", epsilon: " << epsilon;
        }
        std::cerr << "\n  Threads: " << threads << "\n";
    }
    else if (command == "shrink")
        std::cerr << "napkinXML - " << command
            << "\n  Input model: " << input
            << "\n  Output model: " << output
            << "\n  Threshold: " << weightsThreshold << "\n";
}

void Args::printHelp(){
    std::cerr << R"HELP(Usage: nxml <command> <args>

    Commands:
        train
        test

    Args:
        General:
        -i, --input         Input dataset in LibSvm format
        -o, --output        Output (model) dir
        -m, --model         Model type (default = plt):
                            Models: br, plt
        -d, --dataFormat    Type of data format (default = libsvm):
                            Supported data formats: libsvm
        -t, --threads       Number of threads used for training and testing (default = -1)
                            Note: -1 to use #cpus - 1, 0 to use #cpus
        --header            Input contains header (default = 1)
                            Header format for libsvm: #lines #features #labels
        --seed              Seed

        Base classifiers:
        --optimizer         Use LibLiner or SGD (default = libliner)
                            Optimizers: liblinear, sgd
        --bias              Add bias term (default = 1)
        --labelsWeights     Increase the weight of minority labels in base classifiers (default = 1)

        LibLinear:
        -s, --solver        LibLinear solver (default = L2R_LR_DUAL)
                            Supported solvers: L2R_LR_DUAL, L2R_LR, L1R_LR,
                                               L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, L1R_L2LOSS_SVC
                            See: https://github.com/cjlin1/liblinear
        -c, -C, --cost      Inverse of regularization strength. Must be a positive float.
                            Like in support vector machines, smaller values specify stronger
                            regularization. (default = 10.0)
                            Note: -1 to automatically find best value for each node.
        -e, --eps           Stopping criteria (default = 0.1)
                            See: https://github.com/cjlin1/liblinear

        SGD:
        -e, --eta           Step size of SGD
        --iter              Number of epochs of SGD

        Tree:
        -a, --arity         Arity of a tree (default = 2)
        --maxLeaves         Maximum number of leaves (labels) in one internal node. (default = 100)
        --tree              File with tree structure
        --treeType          Type of a tree to build if file with structure is not provided
                            Tree types: hierarchicalKMeans, huffman, completeInOrder, completeRandom,
                                        balancedInOrder, balancedRandom,

        K-Means tree:
        --kMeansEps         Stopping criteria for K-Means clustering (default = 0.001)
        --kMeansBalanced    Use balanced K-Means clustering (default = 1)

    )HELP";
    exit(EXIT_FAILURE);
}

void Args::save(std::ostream& out){
    //TODO: save names and other parameters that are displayed
    out.write((char*) &bias, sizeof(bias));
    out.write((char*) &norm, sizeof(norm));
    out.write((char*) &hash, sizeof(hash));
    out.write((char*) &modelType, sizeof(modelType));
    out.write((char*) &dataFormatType, sizeof(dataFormatType));
    out.write((char*) &ensemble, sizeof(ensemble));
}

void Args::load(std::istream& in){
    in.read((char*) &bias, sizeof(bias));
    in.read((char*) &norm, sizeof(norm));
    in.read((char*) &hash, sizeof(hash));
    in.read((char*) &modelType, sizeof(modelType));
    in.read((char*) &dataFormatType, sizeof(dataFormatType));
    in.read((char*) &ensemble, sizeof(ensemble));
}
