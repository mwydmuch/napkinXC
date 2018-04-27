/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include "types.h"

enum TreeType { completeInOrder, completeRandom, kMeans };

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
    bool norm;
    int hash;
    double threshold;

    // Training options
    int threads;
    int solverType;
    double eps;
    double cost;
    bool labelsWeights;

    // Tree options
    int arity;
    TreeType treeType;
    std::string tree;
    int maxLeaves;

    // Prediction options
    int topK;
    bool sparseWeights;

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
