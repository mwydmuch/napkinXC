/*
 Copyright (c) 2018-2021 by Marek Wydmuch, Kalina Jasinska-Kobus

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


//TODO: Refactor base class

struct ProblemData {
    std::vector<double>& binLabels;
    std::vector<Feature*>& binFeatures;
    std::vector<double>& instancesWeights;
    int n; // features space size

    int labelsCount;
    int* labels;
    double* labelsWeights;
    double invPs; // inverse propensity
    int r; // number of all examples

    ProblemData(std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, int n, std::vector<double>& instancesWeights):
                binLabels(binLabels), binFeatures(binFeatures), n(n), instancesWeights(instancesWeights) {
        labelsCount = 0;
        labels = NULL;
        labelsWeights = NULL;
        invPs = 1.0;
        r = 0;
    }
};


class Base {
public:
    Base();
    Base(Args& args);
    ~Base();

    void update(double label, Feature* features, Args& args);
    void unsafeUpdate(double label, Feature* features, Args& args);
    void train(ProblemData& problemData, Args& args);
    void trainLiblinear(ProblemData& problemData, Args& args);
    void trainOnline(ProblemData& problemData, Args& args);

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

    size_t size();
    inline int getFirstClass() { return firstClass; }

    void clear();
    void clearW();
    void toMap();    // From dense weights (W) to sparse weights in hashmap (mapW)
    void toDense();  // From sparse weights (sparseW or mapW) to dense weights (W)
    void toSparse(); // From dense (W) to sparse weights (sparseW)
    void pruneWeights(double threshold);
    void invertWeights();
    void setFirstClass(int first);

    void save(std::ostream& out, bool saveGrads=false);
    void load(std::istream& in, bool loadGrads=false, bool loadDense=false);

    Base* copy();
    Base* copyInverted();

    bool isDummy() { return (classCount < 2); }
    void setDummy() { clear(); }

    // Used for debug
    void printWeights();

private:
    std::mutex updateMtx;
    bool hingeLoss;

    int wSize;
    int nonZeroW;
    int nonZeroG;
    int classCount;
    int firstClass;
    int firstClassCount;
    int t;

    // Weights //TODO: Change this to one type of Vector object
    Weight* W;
    Weight* G;
    UnorderedMap<int, Weight>* mapW;
    UnorderedMap<int, Weight>* mapG;
    SparseWeight* sparseW;

    template <typename T> static void updateSGD(T& W, T& G, Feature* features, double grad, int t, Args& args);
    template <typename T> static void updateAdaGrad(T& W, T& G, Feature* features, double grad, int t, Args& args);

    static double logisticLoss(double label, double pred, double w){
        double prob = (1.0 / (1.0 + std::exp(-pred)));
        return -label * std::log(prob) - (1 - label) * std::log(1 - prob);
    }

    static double logisticGrad(double label, double pred, double w){
        return (1.0 / (1.0 + std::exp(-pred))) - label;
    }

    static double squaredHingeGrad(double label, double pred, double w){
        double _label = 2 * label - 1;
        double v = _label * pred;
        if(v > 1.0)
            return 0.0;
        else
            return -2 * std::max(1.0 - v, 0.0) * _label;
    }

    static double unbiasedLogisticGrad(double label, double pred, double w){
        return 1 / (1 + std::exp(-pred)) - label * w;
    }

    static double unbiasedLogisticLoss(double label, double pred, double w){
        double prob = (1.0 / (1.0 + std::exp(-pred)));
        return -label * w * std::log(prob) - (1 - label * w) * std::log(1 - prob);
    }

    static double pwLogisticGrad(double label, double pred, double w){
        return -(2 * (label * w - label * 0.5) / (1.0 + std::exp(-pred))) - label + 1;
    }

    static double pwLogisticLoss(double label, double pred, double w){
        double prob = (1.0 / (1.0 + std::exp(-pred)));
        return -(2 * w - 1) * label * std::log(prob) - (1 - label) * std::log(1 - prob);
    }

    void saveVecHeader(std::ostream& out, bool sparse, size_t size, size_t nonZero);
    void saveVec(std::ostream& out, Weight* V, size_t size, size_t nonZero);
    void saveVec(std::ostream& out, SparseWeight* V, size_t size, size_t nonZero);
    void saveVec(std::ostream& out, UnorderedMap<int, Weight>* mapV, size_t size, size_t nonZero);

    Weight* loadAsDense(std::istream& in);
    UnorderedMap<int, Weight>* loadAsMap(std::istream& in);
    void skipLoadVec(std::istream& in);

    // TODO: Improve
    void forEachW(const std::function<void(Weight&)>& f);
    void forEachIW(const std::function<void(const int&, Weight&)>& f);
    void forEachG(const std::function<void(Weight&)>& f);
    void forEachIG(const std::function<void(const int&, Weight&)>& f);
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
