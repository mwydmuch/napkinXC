/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

#include "args.h"
#include "types.h"

class Base {
public:
    Base();
    ~Base();

    void train(int n, std::vector<double>& labels, std::vector<Feature*>& features, Args &args);
    double predictValue(Feature* features);
    double predictValue(double* features);

    template<typename T>
    double predictLoss(T* features);
    template<typename T>
    double predictProbability(T* features);

    inline size_t denseSize(){ return wSize * sizeof(double); }
    inline size_t mapSize(){ return nonZeroW * (sizeof(void*) + sizeof(int) + sizeof(double)); }
    inline size_t sparseSize(){ return nonZeroW * (sizeof(int) + sizeof(double)); }
    size_t size();

    void toMap(); // From dense weights (W) to sparse weights in hashmap (mapW)
    void toDense(); // From sparse weights (sparseW or mapW) to dense weights (W)
    void toSparse(); // From dense (W) to sparse weights (sparseW)
    void threshold(double threshold);

    void save(std::ostream& out);
    void load(std::istream& in);

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

template<typename T>
double Base::predictLoss(T* features){
    if(classCount < 2) return -static_cast<double>(firstClass);
    double val = predictValue(features);
    if(hingeLoss) val = std::pow(std::fmax(0, 1 - val), 2); // Hinge squared loss
    else val = log(1 + std::exp(-val)); // Log loss
    return val;
}

template<typename T>
double Base::predictProbability(T* features){
    if(classCount < 2) return static_cast<double>(firstClass);
    double val = predictValue(features);
    if(hingeLoss) val = 1.0 / (1.0 + std::exp(-2 * val)); // Probability for squared Hinge loss solver
    else val = 1.0 / (1.0 + std::exp(-val)); // Probability
    return val;
}


// Base utils

Base* trainBase(int n, std::vector<double>& baseLabels, std::vector<Feature*>& baseFeatures, Args& args);

void trainBases(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                std::vector<std::vector<Feature*>>& baseFeatures, Args& args);

void trainBasesWithSameFeatures(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                                std::vector<Feature*>& baseFeatures, Args& args);

void trainBasesWithSameFeatures(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                                std::vector<Feature*>& baseFeatures, Args& args);

std::vector<Base*> loadBases(std::string infile);
