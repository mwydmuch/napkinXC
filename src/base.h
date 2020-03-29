/**
 * Copyright (c) 2018 by Marek Wydmuch
 * Copyright (c) 2019 by Marek Wydmuch, Kalina Kobus
 * All rights reserved.
 */

#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "types.h"


class Base {
public:
    Base();
    ~Base();

    void update(double label, Feature* features, Args& args);
    void unsafeUpdate(double label, Feature* features, Args& args);
    void train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
               std::vector<double>* instancesWeights, Args& args);
    void trainLiblinear(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
                        std::vector<double>* instancesWeights, int positiveLabel, Args& args);
    void trainOnline(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args& args);

    // For online training
    void setupOnlineTraining(Args& args, int n = 0, bool startWithDenseW = false);
    void finalizeOnlineTraining(Args& args);

    double predictValue(Feature* features);
    double predictProbability(Feature* features);

    inline Weight* getW() { return W; }
    inline UnorderedMap<int, Weight>* getMapW() { return mapW; }
    inline SparseWeight* getSparseW() { return sparseW; }

    inline size_t denseSize() { return wSize * sizeof(Weight); }
    inline size_t mapSize() { return nonZeroW * (sizeof(void*) + sizeof(int) + sizeof(Weight)); }
    inline size_t sparseSize() { return nonZeroW * (sizeof(int) + sizeof(double)); }
    size_t size();
    inline int getFirstClass() { return firstClass; }

    void clear();
    void toMap();    // From dense weights (W) to sparse weights in hashmap (mapW)
    void toDense();  // From sparse weights (sparseW or mapW) to dense weights (W)
    void toSparse(); // From dense (W) to sparse weights (sparseW)
    void pruneWeights(double threshold);
    void invertWeights();

    void save(std::ostream& out);
    void load(std::istream& in);

    Base* copy();
    Base* copyInverted();

    // Used for debug
    void printWeights();

private:
    std::mutex updateMtx;
    bool hingeLoss;

    int wSize;
    int nonZeroW;
    int classCount;
    int firstClass;
    int firstClassCount;
    int t;
    double pi; // For FOBOS

    // Weights
    Weight* W;
    Weight* G;
    UnorderedMap<int, Weight>* mapW;
    UnorderedMap<int, Weight>* mapG;
    SparseWeight* sparseW;

    template <typename T> void updateSGD(T& W, Feature* features, double grad, double eta);

    template <typename T> void updateAdaGrad(T& W, T& G, Feature* features, double grad, double eta, double eps);

    template <typename T> void updateFobos(T& W, Feature* features, double grad, double eta, double penalty);

    void forEachW(const std::function<void(Weight&)>& f);
    void forEachIW(const std::function<void(const int&, Weight&)>& f);
};

template <typename T> void Base::updateSGD(T& W, Feature* features, double grad, double eta) {
    double lr = eta * sqrt(1.0 / t);
    Feature* f = features;
    while (f->index != -1) {
        W[f->index] -= lr * grad * f->value;
        ++f;
    }
}

template <typename T> void Base::updateAdaGrad(T& W, T& G, Feature* features, double grad, double eta, double eps) {
    Feature* f = features;
    while (f->index != -1) {
        G[f->index] += f->value * f->value * grad * grad;
        double lr = eta * std::sqrt(1.0 / (eps + G[f->index]));
        W[f->index] -= lr * (grad * f->value);
        ++f;
    }
}

template <typename T> void Base::updateFobos(T& W, Feature* features, double grad, double eta, double penalty) {
    double lr = eta / (1 + eta * t * penalty);
    pi *= (1 + penalty * lr);

    Feature* f = features;
    while (f->index != -1) {
        W[f->index] -= pi * lr * grad * f->value;
        ++f;
    }
}
