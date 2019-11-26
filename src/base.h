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
#include "robin_hood.h"

typedef double Weight;
typedef std::pair<int, Weight> SparseW;
#define UnorderedMap robin_hood::unordered_map
//#define UnorderedMap std::unordered_map

class Base {
public:
    Base();
    ~Base();

    void update(double label, Feature* features, Args &args);
    void unsafeUpdate(double label, Feature* features, Args &args);
    void train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args);
    void trainLiblinear(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, int positiveLabel, Args &args);
    void trainOnline(std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args);

    double predictValue(Feature* features);
    double predictValue(double* features);

    template<typename T>
    double predictLoss(T* features);
    template<typename T>
    double predictProbability(T* features);

    inline size_t denseSize(){ return wSize * sizeof(Weight); }
    inline size_t mapSize(){ return nonZeroW * (sizeof(void*) + sizeof(int) + sizeof(Weight)); }
    inline size_t sparseSize(){ return nonZeroW * (sizeof(int) + sizeof(double)); }
    size_t size();
    inline size_t featureSpaceSize() { return wSize; }

    inline Weight* getW(){ return W; }
    inline UnorderedMap<int, Weight>* getMapW(){ return mapW; }
    inline Feature* getSparseW(){ return sparseW; }

    inline int getFirstClass() { return firstClass; }

    void toMap(); // From dense weights (W) to sparse weights in hashmap (mapW)
    void toDense(); // From sparse weights (sparseW or mapW) to dense weights (W)
    void toSparse(); // From dense (W) to sparse weights (sparseW)
    void pruneWeights(double threshold);
    void setupOnlineTraining(Args &args);
    void finalizeOnlineTraining();
    void multiplyWeights(double a);
    void invertWeights();

    void save(std::ostream& out);
    void load(std::istream& in);

    Base* copy();
    Base* copyInverted();

    void printWeights();

private:
    std::mutex updateMtx;
    bool hingeLoss;

    int wSize;
    int nonZeroW;
    int classCount;
    int firstClass;
    int t;
    float pi;

    // Weights
    Weight* W;
    Weight* G;
    UnorderedMap<int, Weight>* mapW;
    UnorderedMap<int, Weight>* mapG;
    Feature* sparseW;

    template<typename T>
    void updateSGD(T& W, Feature* features, double grad, double eta);

    template<typename T>
    void updateAdaGrad(T& W, T& G, Feature* features, double grad, double eta, double eps);

    template<typename T>
    void updateFobos(T& W, Feature* features, double grad, double eta, double penalty);

    //void forEachW(const std::function <void (Weight&)>& f);
    //void forEachIW(const std::function <void (const int&, Weight&)>& f);
};

template<typename T>
void Base::updateSGD(T& W, Feature* features, double grad, double eta){
    double lr = eta * sqrt(1.0 / t);
    Feature *f = features;
    while (f->index != -1) {
        W[f->index - 1] -= lr * grad * f->value;
        ++f;
    }
    if (f != features && (f - 1)->index > wSize) wSize = (f - 1)->index;
}

template<typename T>
void Base::updateAdaGrad(T& W, T& G, Feature* features, double grad, double eta, double eps){
    Feature *f = features;
    while (f->index != -1) {
        G[f->index - 1] += f->value * f->value * grad * grad;
        double lr = eta * std::sqrt(1.0 / (eps + G[f->index - 1]));
        W[f->index - 1] -= lr * (grad * f->value);
        ++f;
    }
    if (f != features && (f - 1)->index > wSize) wSize = (f - 1)->index;
}

template<typename T>
void Base::updateFobos(T& W, Feature* features, double grad, double eta, double penalty){
    double lr = eta / (1 + eta * t * penalty);
    pi *= (1 + penalty * lr);

    Feature *f = features;
    while (f->index != -1) {
        W[f->index - 1] -= pi * lr * grad * f->value;
        ++f;
    }
    if (f != features && (f - 1)->index > wSize) wSize = (f - 1)->index;
}

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

void trainBases(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                std::vector<std::vector<Feature*>>& baseFeatures, Args& args);

void trainBasesWithSameFeatures(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                                std::vector<Feature*>& baseFeatures, Args& args);

void trainBasesWithSameFeatures(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                                std::vector<Feature*>& baseFeatures, Args& args);

std::vector<Base*> loadBases(std::string infile);
