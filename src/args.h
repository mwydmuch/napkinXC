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

#pragma once

#include <random>
#include <string>

#include "misc.h"

// All tree tree types
enum ModelType {
    ovr,
    br,
    hsm,
    plt,
    svbopFull,
    svbopHf,
    oplt,
    extremeText,
    mach,
    brMips, // MIPS extension model
    svbopMips, // MIPS extension model
    svbopThreshold,
    svbopFagin,
    svbopInvertedIndex
};

enum TreeType {
    hierarchicalKmeans,
    huffman,
    completeInOrder,
    completeRandom,
    balancedInOrder,
    balancedRandom,
    onlineKaryComplete,
    onlineKaryRandom,
    onlineRandom,
    onlineBestScore,
    custom // custom tree has to be the last one
};

enum OptimizerType { liblinear, sgd, adagrad };

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
    logistic,
    squaredHinge,
    unLogistic,
    pwLogistic,
    asymteric,
};

enum OFOType {
    micro,
    macro,
    mixed
};

class Args : public FileHelper {
public:
    Args();

    inline int getSeed() { return rngSeeder(); };
    void parseArgs(const std::vector<std::string>& args, bool keepArgs = true);
    void printArgs(std::string command = "");
    int countArg(const std::vector<std::string>& args, std::string to_count);
    int countArgs(const std::vector<std::string>& args, std::vector<std::string> to_count);
    void save(std::ostream& out) override;
    void load(std::istream& in) override;

    // Threading, memory and seed options
    int seed;
    int threads;
    unsigned long long memLimit; // TODO: Implement this for some models
    bool saveGrads;
    bool resume;
    bool loadDense;

    // Input/output options
    std::string input;
    std::string output;
    std::string prediction;
    ModelType modelType;
    bool processData;
    double bias;
    bool norm;
    int hash;
    double featuresThreshold;

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

    // Tree models

    // Tree options
    TreeType treeType;
    std::string treeStructure;
    int arity;
    int maxLeaves;

    // K-Means tree options
    double kmeansEps;
    bool kmeansBalanced;
    bool kmeansWeightedFeatures;

    // Online tree options
    double onlineTreeAlpha;

    // extremeText options
    size_t dims;

    // MACH options
    int machHashes;
    int machBuckets;

    // Prediction options
    int topK;
    double threshold;
    std::string thresholds;
    std::string labelsWeights;
    bool ensMissingScores;

    // Mips options
    bool mipsDense;
    int hnswM;
    int hnswEfConstruction;
    int hnswEfSearch;

    // Set utility options
    double svbopMipsK;
    int svbopInvIndexK;

    SetUtilityType setUtilityType;
    double alpha;
    double beta;
    double delta;
    double gamma;

    // Measures for test command
    std::string measures;
    int measuresPrecision;

    // Args for OFO command
    OFOType ofoType;
    double ofoTopLabels;
    double ofoA;
    double ofoB;

    double psA;
    double psB;

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
    std::string setUtilityName;
    std::string ofoTypeName;

    std::vector<std::string> parsedArgs;
};
