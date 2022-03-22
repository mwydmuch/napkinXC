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
#include <mutex>

#include "args.h"
#include "vector.h"


struct ProblemData {
    std::vector<Real>& binLabels;
    std::vector<Feature*>& binFeatures;
    std::vector<Real>& instancesWeights;
    int n; // features space size

    int labelsCount;
    int* labels;
    Real* labelsWeights;
    Real invPs; // inverse propensity
    int r; // number of all examples
    Real loss;

    ProblemData(std::vector<Real>& binLabels, std::vector<Feature*>& binFeatures, int n, std::vector<Real>& instancesWeights):
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

    void update(Real label, Feature* feature, Args& args);
    void unsafeUpdate(Real label, Feature* feature, Args& args);
    void train(ProblemData& problemData, Args& args);
    void trainLiblinear(ProblemData& problemData, Args& args);
    void trainOnline(ProblemData& problemData, Args& args);

    // For online training
    void setupOnlineTraining(Args& args, int n = 0, bool startWithDenseW = false);
    void finalizeOnlineTraining(Args& args);

    Real predictValue(SparseVector& features);
    Real predictProbability(SparseVector& features);

    inline AbstractVector* getW() { return W; };
    inline AbstractVector* getG() { return G; };

    inline void setW(AbstractVector* vec) { W = vec; };
    inline void setG(AbstractVector* vec) { G = vec; };

    unsigned long long mem();
    inline int getFirstClass() { return firstClass; }
    void clear();

    void to(RepresentationType type); // Change representation type of base classifier
    RepresentationType getType();
    void pruneWeights(Real threshold);
    void setFirstClass(int first);
    void setLoss(LossType);

    void save(std::ofstream& out, bool saveGrads=false);
    void load(std::ifstream& in, bool loadGrads=false, RepresentationType loadAs=map);

    Base* copy();
    Base* copyInverted();

    bool isDummy() { return (classCount < 2); }
    void setDummy() { clear(); }

private:
    std::mutex updateMtx;
    LossType lossType;
    Real (*lossFunc)(Real, Real, Real);
    Real (*gradFunc)(Real, Real, Real);

    int classCount;
    int firstClass;
    int firstClassCount;
    int t;

    // Weights (parameters)
    AbstractVector* W;
    AbstractVector* G;

    AbstractVector* vecTo(AbstractVector*, RepresentationType type);
};
