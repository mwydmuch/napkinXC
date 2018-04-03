/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include "args.h"
#include "types.h"

class Base {
public:
    Base();
    ~Base();

    void train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args);
    double predict(Feature* features);

    void save(std::string outfile, Args& args);
    void save(std::ostream& out, Args& args);
    void load(std::string infile, Args& args);
    void load(std::istream& in, Args& args);

private:
    bool sparse;
    int wSize;
    int classCount;
    int firstClass;

    double* W;
    std::unordered_map<int, double>* sparseW;
};
