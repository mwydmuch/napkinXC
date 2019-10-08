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
    ubopch,
    ubopmips,
    hsmEns,
    pltEns,
    pltNeg,
    brpltNeg,
    oplt
};

enum TreeType {
    hierarchicalKMeans,
    huffman,
    completeInOrder,
    completeRandom,
    balancedInOrder,
    balancedRandom,
    online,
    custom
};

enum OptimizerType { libliner, sgd };

enum DataFormatType { libsvm };

enum SetUtilityType {
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
    bool norm;
    bool tfidf;
    int hash;
    double pruneThreshold;

    // Training options
    int threads;
    size_t memLimit; // TODO: Implement this for OVA models
    int solverType;
    OptimizerType optimizerType;
    double eps;
    double cost;
    double weightsThreshold;
    bool labelsWeights;
    int iter;
    double eta;
    double mem;
    int ensemble;

    // For online training
    int epochs;

    // Tree models

    // Tree options
    int arity;
    TreeType treeType;
    std::string treeStructure;
    int maxLeaves;

    // Tree sampling
    int sampleK;

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

    // Set utility options
    SetUtilityType setUtilityType;
    double alfa;
    double beta;
    double epsilon;
    double delta;
    double gamma;

    // Measures
    std::string measures;

private:
    std::string solverName;
    std::string treeTypeName;
    std::string optimizerName;
    std::string modelName;
    std::string dataFormatName;
    std::string setUtilityName;
};
