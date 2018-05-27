/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include "types.h"

// All tree tree types
enum TreeType {
    completeInOrder,
    completeRandom,
    hierarchicalKMeans,
    kMeansWithProjection,
    topDown,
    balancedInOrder,
    balancedRandom,
    huffman,
    leaveFreqBehind,
    kMeansHuffman
};

enum OptimizerType { libliner, sgd };

class Args{
public:
    Args();

    std::string command;
    int seed;

    // Input/output options
    std::string input;
    std::string model;
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
    int optimizerType;
    std::string optimizerName;
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
    int projectDim;

    // K-Means tree options
    double kMeansEps;
    bool kMeansBalanced;

    // Prediction options
    int topK;
    bool sparseWeights;

    // KNN options
    int kNN;
    int kNNMaxFreq;

    void parseArgs(const std::vector<std::string>& args);
    void readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features);
    void readLine(std::string& line, std::vector<Label>& lLabels, std::vector<Feature>& lFeatures);
    void printArgs();
    void printHelp();

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile);
    void load(std::istream& in);

private:
    int hLabels;
    int hFeatures;

    std::string solverName;
    std::string treeTypeName;
};
