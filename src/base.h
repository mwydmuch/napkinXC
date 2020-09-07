/*
 Copyright (c) 2018 by Marek Wydmuch
 Copyright (c) 2019-2020 by Marek Wydmuch, Kalina Jasinska-Kobus

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#pragma once

#include <cmath>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>

#include "args.h"
#include "types.h"


class Base {
public:
    Base();
    ~Base();

    void update(double label, Feature* features, Args& args);
    void unsafeUpdate(double label, Feature* features, Args& args);
    void train(int n, int r, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
               std::vector<double>* instancesWeights, Args& args);
    void trainLiblinear(int n, int r, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
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

    inline int getWSize() { return wSize; }
    inline int getNonZeroW() { return nonZeroW; }
    inline size_t denseSize() { return wSize * sizeof(Weight); }
    inline size_t mapSize() { return nonZeroW * (sizeof(int) + sizeof(int) + sizeof(Weight)); }
    inline size_t sparseSize() { return nonZeroW * (sizeof(int) + sizeof(Weight)); }
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

    bool isDummy() { return (classCount < 2); }

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

    // Weights
    Weight* W;
    Weight* G;
    UnorderedMap<int, Weight>* mapW;
    UnorderedMap<int, Weight>* mapG;
    SparseWeight* sparseW;

    template <typename T> static void updateSGD(T& W, T& G, Feature* features, double grad, int t, Args& args);
    template <typename T> static void updateAdaGrad(T& W, T& G, Feature* features, double grad, int t, Args& args);

    static double logisticGrad(double label, double pred){
        return (1.0 / (1.0 + std::exp(-pred))) - label;
    }

    static double squaredHingeGrad(double label, double pred){
        double _label = 2 * label - 1;
        double v = _label * pred;
        // return v > 1 ? 0.0 : -_label; // hinge grad
        if(v > 1.0)
            return 0.0;
        else
            return -2 * std::max(1.0 - v, 0.0) * _label;
    }

    void forEachW(const std::function<void(Weight&)>& f);
    void forEachIW(const std::function<void(const int&, Weight&)>& f);
};

template <typename T> void Base::updateSGD(T& W, T& G, Feature* features, double grad, int t, Args& args) {
    double eta = args.eta;
    double lr = eta * sqrt(1.0 / t);
    Feature* f = features;
    while (f->index != -1) {
        W[f->index] -= lr * grad * f->value;
        ++f;
    }
}

template <typename T> void Base::updateAdaGrad(T& W, T& G, Feature* features, double grad, int t, Args& args) {
    double eta = args.eta;
    double eps = args.adagradEps;
    Feature* f = features;
    while (f->index != -1) {
        G[f->index] += f->value * f->value * grad * grad;
        double lr = eta * std::sqrt(1.0 / (eps + G[f->index]));
        W[f->index] -= lr * (grad * f->value);
        ++f;
        // TODO: add correct regularization
        //double reg = l2 * W[f->index];
        //W[f->index] -= lr * (grad * f->value + reg);
    }
}
