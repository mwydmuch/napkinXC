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
    extremeText,
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
    custom
};

enum OptimizerType { liblinear, sgd, adagrad, fobos };

enum DataFormatType { libsvm, vw };

enum SetUtilityType { uP, uF1, uAlpha, uAlphaBeta, uDeltaGamma };

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
    int maxIter;
    double weightsThreshold;
    int ensemble;
    bool onTheTrotPrediction;
    bool inbalanceLabelsWeighting;
    bool pickOneLabelWeighting;

    // For online training
    double eta;
    int epochs;
    double l2Penalty;
    double fobosPenalty;
    int tmax;
    double adagradEps;

    // extremeText
    size_t dims;

    // Tree models

    // Tree options
    TreeType treeType;
    std::string treeStructure;
    int arity;
    int maxLeaves;

    // K-Means tree options
    double kMeansEps;
    bool kMeansBalanced;
    bool kMeansWeightedFeatures;

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

    // Mips options
    int mipsEfConstruction;
    int mipsEfSearch;

    // Set utility options
    double ubopMipsK;
    double ubopMipsSample;

    SetUtilityType setUtilityType;
    double alpha;
    double beta;
    double epsilon;
    double delta;
    double gamma;

    // OFO options
    double ofoA = 10;
    double ofoB = 20;

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
