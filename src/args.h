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

#include "basic_types.h"
#include "enums.h"
#include "save_load.h"

class Args : public FileHelper {
public:
    Args();

    inline int getSeed() { return rngSeeder(); };
    void parseArgs(const std::vector<std::string>& args, bool keepArgs = true);
    void printArgs(std::string command = "");
    int countArg(const std::vector<std::string>& args, std::string to_count);
    int countArgs(const std::vector<std::string>& args, std::vector<std::string> to_count);
    void save(std::ofstream& out) override;
    void load(std::ifstream& in) override;

    // Threading, memory and seed options
    int seed;
    int threads;
    unsigned long long memLimit; // TODO: Implement this for some models
    bool saveGrads;
    bool resume;
    RepresentationType loadAs;

    // Input/output options
    std::string input;
    std::string output;
    std::string prediction;
    ModelType modelType;
    bool processData;
    Real bias;
    bool norm;
    int hash;
    Real featuresThreshold;

    // Training options
    int solverType;
    OptimizerType optimizerType;
    LossType lossType;
    Real eps;
    Real cost;
    int maxIter;
    Real weightsThreshold;
    bool inbalanceLabelsWeighting;
    bool pickOneLabelWeighting;
    bool autoCLin;
    bool autoCLog;
    bool reportLoss;

    // Ensemble options
    int ensemble;
    bool ensOnTheTrot;
    bool ensMissingScores;

    // For online training
    Real eta;
    int epochs;
    Real l2Penalty;
    int tmax;
    Real adagradEps;

    // Tree models

    // Tree options
    TreeType treeType;
    std::string treeStructure;
    int arity;
    int maxLeaves;
    int flattenTree;

    // K-Means tree options
    Real kmeansEps;
    bool kmeansBalanced;
    bool kmeansWeightedFeatures;

    // Online tree options
    Real onlineTreeAlpha;

    // extremeText options
    size_t dims;

    // MACH options
    int machHashes;
    int machBuckets;

    // Prediction options
    int topK;
    Real threshold;
    std::string thresholds;
    std::string labelsWeights;
    std::string labelsBiases;
    TreeSearchType treeSearchType;
    int beamSearchWidth;
    bool beamSearchUnpack;
    int batchRows;
    int startRow;
    int endRow;
    int predictionPrecision;
    bool covWeights;
    
    // Measures for test command
    std::string metrics;
    int metricsPrecision;

    // Args for OFO command
    OFOType ofoType;
    Real ofoTopLabels;
    Real ofoA;
    Real ofoB;

    Real psA;
    double psB;

    // Args for testPredictionTime command
    std::string tptBatchSizes;
    int tptBatches;

private:
    std::default_random_engine rngSeeder;

    std::string solverName;
    std::string lossName;
    std::string treeTypeName;
    std::string optimizerName;
    std::string modelName;
    std::string setUtilityName;
    std::string ofoTypeName;
    std::string treeSearchName;
    std::string representationName;

    std::vector<std::string> parsedArgs;
};
