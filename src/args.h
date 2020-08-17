/**
 * Copyright (c) 2018-2020 by Marek Wydmuch
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
    onlineKAryComplete,
    onlineKAryRandom,
    onlineRandom,
    onlineBestScore,
    custom // custom tree has to be the last one
};

enum OptimizerType { liblinear, sgd, adagrad };

enum DataFormatType { libsvm, vw };

enum SetUtilityType {
    uP,
    uR,
    uF1,
    uFBeta,
    uExp,
    uLog,
    uDeltaGamma,
    uAlpha,
    uAlphaBeta
};

enum LossType {
    squeredHinge,
    logistic
};

enum OFOType {
    micro,
    macro,
    mixed
};

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
    double bias;
    bool norm;
    int hash;
    double featuresThreshold;

    // Threading and memory options
    int threads;
    unsigned long long memLimit; // TODO: Implement this for some models

    // Training options
    int solverType;
    OptimizerType optimizerType;
    LossType lossType;
    double eps;
    double cost;
    int maxIter;
    double weightsThreshold;
    int ensemble;
    bool onTheTrotPrediction;
    bool inbalanceLabelsWeighting;
    bool pickOneLabelWeighting;
    bool autoCLin;
    bool autoCLog;

    // For online training
    double eta;
    int epochs;
    double l2Penalty;
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

    // Online tree options
    double onlineTreeAlpha;

    // Prediction options
    int topK;
    double threshold;
    std::string thresholds;
    bool ensMissingScores;

    inline int getSeed() { return rngSeeder(); };
    void parseArgs(const std::vector<std::string>& args);
    void printArgs();

    void save(std::ostream& out) override;
    void load(std::istream& in) override;

    // Mips options
    bool mipsDense;
    int hnswM;
    int hnswEfConstruction;
    int hnswEfSearch;

    // Set utility options
    double ubopMipsK;

    SetUtilityType setUtilityType;
    double alpha;
    double beta;
    double delta;
    double gamma;


    // Measures for test command
    std::string measures;

    // Args for OFO command
    OFOType ofoType;
    std::string ofoTypeName;
    double ofoTopLabels;
    double ofoA;
    double ofoB;

    // Args for testPredictionTime command
    std::string batchSizes;
    int batches;

private:
    std::default_random_engine rngSeeder;

    std::string solverName;
    std::string lossName;
    std::string treeTypeName;
    std::string optimizerName;
    std::string modelName;
    std::string dataFormatName;
    std::string setUtilityName;
};
