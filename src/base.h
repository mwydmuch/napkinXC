/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "args.h"
#include "types.h"

class Base {
public:
    Base();
    ~Base();

    void train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args);
    double predict(Feature* features);

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile, bool asSparse = false);
    void load(std::istream& in, bool asSparse = false);

private:
    bool sparse;
    int wSize;
    int classCount;
    int firstClass;

    std::vector<Feature> sparseW;
    std::vector<double> W;

    // LibLinear's stuff
    model *M;

    // For testing/debuging
    bool useLinearPredict;
};
