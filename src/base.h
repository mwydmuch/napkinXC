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
    double predictValue(double* features);

    template<typename U>
    double predictLoss(U* features);
    template<typename U>
    double predictProbability(U* features);

    inline size_t denseSize(){ return wSize * sizeof(double); }
    inline size_t mapSize(){ return nonZeroW * (sizeof(void*) + sizeof(int) + sizeof(double)); }
    inline size_t sparseSize(){ return nonZeroW * (sizeof(int) + sizeof(double)); }
    size_t size();

    void toMap(); // From dense weights (W) to sparse weights in hashmap (mapW)
    void toDense(); // From sparse weights (sparseW or mapW) to dense weights (W)
    void toSparse(); // From dense (W) to sparse weights (sparseW)
    void threshold(double threshold);

    void save(std::string outfile, Args& args);
    void save(std::ostream& out, Args& args);
    void load(std::string infile, Args& args);
    void load(std::istream& in, Args& args);

    void printWeights();

private:
    bool hingeLoss;

    int wSize;
    int nonZeroW;
    int classCount;
    int firstClass;

    double* W;
    std::unordered_map<int, double>* mapW;
    Feature* sparseW;

};

template<typename U>
double Base::predictLoss(U* features){
    if(classCount < 2) return -static_cast<double>(firstClass);
    double val = predictValue(features);
    if(hingeLoss) val = std::pow(fmax(0, 1 - val), 2); // Hinge squared loss
    else val = log(1 + exp(-val)); // Log loss
    return val;
}

template<typename U>
double Base::predictProbability(U* features){
    if(classCount < 2) return static_cast<double>(firstClass);
    double val = predictValue(features);
    if(hingeLoss) val = 1.0 / (1.0 + exp(-2 * val)); // Probability for squared Hinge loss solver
    else val = 1.0 / (1.0 + exp(-val)); // Probability
    return val;
}
