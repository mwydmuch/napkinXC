/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include "types.h"

enum TreeType { completeInOrder, completeRandom, complete, given, topDown, balancedInOrder, balancedRandom };

class Args{
public:
    Args();

    std::string command;

    // Input/output options
    std::string input;
    std::string model;
    bool header;
    bool bias;
    bool norm;
    int hash;
    double threshold;
    bool sparseWeights;

    // Training options
    int threads;
    int solverType;
    double eps;

    // Tree options
    int arity;
    TreeType treeType;
    std::string tree;

    // Prediction options
    int topK;

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
