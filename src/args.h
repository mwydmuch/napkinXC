/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>

#include "types.h"
#include "utils.h"

// All tree tree types
enum ModelType {
    ovr,
    br,
    hsm,
    plt,
    ubop,
    rbop,
    hsmubop
};

enum TreeType {
    hierarchicalKMeans,
    huffman,
    completeInOrder,
    completeRandom,
    balancedInOrder,
    balancedRandom,
    custom
};

enum OptimizerType { libliner, sgd };

enum DataFormatType { libsvm };

enum SetBasedUType {
    uP,
    uF1,
    uAlfa,
    uAlfaBeta,
    uDeltaGamma
};

class Args: public FileHelper{
public:
    Args();

    std::string command;
    int seed;

    // Input/output options
    std::string input;
    std::string output;
    DataFormatType dataFormatType;
    ModelType modelType;
    bool header;
    bool bias;
    double biasValue;
    double C;
    bool norm;
    int hash;
    int maxFeatures;

    // Training options
    int threads;
    int solverType;
    OptimizerType optimizerType;
    double eps;
    double cost;
    double threshold;
    bool labelsWeights;
    int iter;
    double eta;

    // Tree options
    int arity;
    TreeType treeType;
    std::string treeStructure;
    int maxLeaves;

    // K-Means tree options
    double kMeansEps;
    bool kMeansBalanced;
    bool kMeansWeightedFeatures;

    // Prediction options
    int topK;
    bool sparseWeights;

    void parseArgs(const std::vector<std::string>& args);
    void printArgs();
    void printHelp();

    void save(std::ostream& out) override;
    void load(std::istream& in) override;

    // Set based
    SetBasedUType setBasedUType;

private:
    std::string solverName;
    std::string treeTypeName;
    std::string optimizerName;
    std::string modelName;
    std::string dataFormatName;
};
