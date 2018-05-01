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
    double predictValue(Feature* features);
    double predictLoss(Feature* features);
    double predictProbability(Feature* features);

    int inline denseSize(){ return wSize * sizeof(double); }
    int inline mapSize(){ return nonZeroW * (sizeof(void*) + sizeof(int) + sizeof(double)); }
    int inline sparseSize(){ return nonZeroW * (sizeof(int) + sizeof(double)); }

    void toMap(); // from W to mapW
    void toDense(); // from sparseW or mapW to W
    void toSparse(); // from W to sparseW
    void threshold(double threshold);

    void save(std::string outfile, Args& args);
    void save(std::ostream& out, Args& args);
    void load(std::string infile, Args& args);
    void load(std::istream& in, Args& args);

    void printWeights();

private:
    bool sparse;
    bool hingeLoss;

    int wSize;
    int nonZeroW;
    int classCount;
    int firstClass;

    double* W;
    std::unordered_map<int, double>* mapW;
    Feature* sparseW;

};
