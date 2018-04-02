/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include "types.h"

enum TreeType { completeInOrder, completeRandom, complete, given };

class Args{
public:
    Args();

    // Input/output options
    std::string input;
    std::string model;
    bool header;
    int hash;

    // Training options
    int threads;
    int solverType;
    double eps;
    bool bias;

    // Tree options
    int arity;
    TreeType treeType;
    std::string tree;

    // Prediction options
    int topK;

    void parseArgs(const std::vector<std::string>& args);
    void readData(SRMatrix<Label>& labels, SRMatrix<Feature>& features);
    void printHelp();

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile);
    void load(std::istream& in);

private:
    int hLabels;
    int hFeatures;
};
