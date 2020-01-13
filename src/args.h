/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <random>
#include <string>

#include "misc.h"
#include "types.h"

// All tree tree types
enum ModelType {
    ovr,
    br,
    hsm,
    plt,
    ubop,
    rbop,
    ubopHsm,
    oplt,
    // Mips extension models
    brMips,
    ubopMips,
};

enum TreeType {
    hierarchicalKMeans,
    huffman,
    completeInOrder,
    completeRandom,
    balancedInOrder,
    balancedRandom,
    onlineBalanced,
    onlineComplete,
    onlineRandom,
    onlineBottomUp,
    custom,

    onlineBestScore,
    onlineKMeans
};

enum OptimizerType { liblinear, sgd, adagrad, fobos };

enum DataFormatType { libsvm, vw };

enum SetUtilityType { uP, uF1, uAlfa, uAlfaBeta, uDeltaGamma };

class Args : public FileHelper {
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
    int hash;
    double featuresThreshold;

    // Threading and memory options
    int threads;
    unsigned long long memLimit; // TODO: Implement this for some models

    // Training options
    int solverType;
    OptimizerType optimizerType;
    double eps;
    double cost;
    double weightsThreshold;
    int ensemble;
    bool onTheTrotPrediction;
    bool inbalanceLabelsWeighting;
    bool hsmPickOneLabelWeighting;

    // For online training
    double eta;
    int epochs;
    double fobosPenalty;
    int tmax;
    double adagradEps;

    // Tree models

    // Tree options
    TreeType treeType;
    std::string treeStructure;
    int arity;
    int maxLeaves;
    int maxDepth;
    bool newOnline;

    // K-Means tree options
    double kMeansEps;
    bool kMeansBalanced;
    bool kMeansWeightedFeatures;
    int kMeansHash;

    // Prediction options
    int topK;
    double threshold;
    std::string thresholds;

    inline int getSeed() { return rngSeeder(); };
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
    std::default_random_engine rngSeeder;

    std::string solverName;
    std::string treeTypeName;
    std::string optimizerName;
    std::string modelName;
    std::string dataFormatName;
    std::string setUtilityName;
};
