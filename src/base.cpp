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

#include <fstream>
#include <iostream>
#include <random>

#include "base.h"
#include "linear.h"
#include "log.h"
#include "misc.h"
#include "online_optimization.h"
#include "threads.h"


//TODO: Refactor base class

Base::Base() {
    lossType = logistic;
    classCount = 0;
    firstClass = 0;
    firstClassCount = 0;
    t = 0;

    W = nullptr;
    G = nullptr;
}

Base::Base(Args& args): Base(){
    if(args.optimizerType != liblinear)
        setupOnlineTraining(args);
}

Base::~Base() { clear(); }

void Base::update(Real label, Feature* features, Args& args) {
    std::lock_guard<std::mutex> lock(updateMtx);

    unsafeUpdate(label, features, args);
}

void Base::unsafeUpdate(Real label, Feature* features, Args& args) {
    if (args.tmax != -1 && args.tmax < t) return;

    ++t;
    if (label == firstClass) ++firstClassCount;

    Real pred = W->dot(features);
    Real grad = gradFunc(label, pred, 0); // Online version doesn't support weights  right now

    if (args.optimizerType == sgd)
        updateSGD(*W, *G, features, grad, t, args);
    else if (args.optimizerType == adagrad)
        updateAdaGrad(*W, *G, features, grad, t, args);
    else throw std::invalid_argument("Unknown optimizer type");

    // Check if we should change sparse W to dense W
    /*
    if (mapW != nullptr && wSize != 0) {
        nonZeroW = mapW->size();
        if (mapSize() > denseSize()) toDense();
    }
     */
}

void Base::trainLiblinear(ProblemData& problemData, Args& args) {
    Real cost = args.cost;
    if (args.autoCLog)
        cost *= 1.0 + log(static_cast<Real>(problemData.r) / problemData.binFeatures.size());
    if (args.autoCLin)
        cost *= static_cast<Real>(problemData.r) / problemData.binFeatures.size();



    problem P = {/*.l =*/ static_cast<int>(problemData.binLabels.size()),
                 /*.n =*/ problemData.n,
                 /*.y =*/ problemData.binLabels.data(),
                 /*.x =*/ reinterpret_cast<feature_node**>(problemData.binFeatures.data()),
                 /*.bias =*/ -1,
                 /*.W =*/ problemData.instancesWeights.data()};

    parameter C = {/*.solver_type =*/ args.solverType,
                   /*.eps =*/ args.eps,
                   /*.C =*/ cost,
                   /*.nr_weight =*/ problemData.labelsCount,
                   /*.weight_label =*/ problemData.labels,
                   /*.weight =*/ problemData.labelsWeights,
                   /*.p =*/ 0,
                   /*.init_sol =*/ NULL,
                   /*.max_iter =*/ args.maxIter};

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    model* M = train_liblinear(&P, &C);

    assert(M->nr_class <= 2);
    assert(M->nr_feature == problemData.n);

    // Set base's attributes
    firstClass = M->label[0];
    classCount = M->nr_class;

    // Copy weights
    W = new Vector(problemData.n + 1);
    for (int i = 0; i < problemData.n; ++i) W->insertD(i + 1, M->w[i]); // Shift by 1

    if(args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC ||
        args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC)
        lossType = squaredHinge;

    // Delete LibLinear model
    free_model_content(M);
    free(M);
}

void Base::trainOnline(ProblemData& problemData, Args& args) {
    classCount = 2;
    firstClass = 1;
    t = 0;

    Vector* newW = new Vector(problemData.n);
    Vector* newG = nullptr;

    // Set update function
    void (*updateFunc)(Vector&, Vector&, Feature*, Real, int, Args&);
    if(args.optimizerType == sgd) {
        updateFunc = &updateSGD;
    }
    else if (args.optimizerType == adagrad){
        updateFunc = &updateAdaGrad;
        newG = new Vector(problemData.n);
    }
    else
        throw std::invalid_argument("Unknown online update function type");

    const int examples = problemData.binFeatures.size();
    for (int e = 0; e < args.epochs; ++e)
        for (int r = 0; r < examples; ++r) {
            Real label = problemData.binLabels[r];
            Feature* features = problemData.binFeatures[r];

            if (args.tmax != -1 && args.tmax < t) break;

            ++t;
            if (problemData.binLabels[r] == firstClass) ++firstClassCount;

            Real pred = newW->dot(features);
            Real grad = gradFunc(label, pred, problemData.invPs) * problemData.instancesWeights[r];
            if (!std::isinf(grad) && !std::isnan(grad))
                updateFunc(*newW, *newG, features, grad, t, args);
        }

    W = newW;
    G = newG;
}

void Base::train(ProblemData& problemData, Args& args) {
    // Delete previous weights
    delete W;
    delete G;

    // Set loss function
    setLoss(args.lossType);

    if (problemData.binLabels.empty()) {
        firstClass = 0;
        classCount = 0;
        return;
    }

    assert(problemData.binLabels.size() == problemData.binFeatures.size());
    assert(problemData.instancesWeights.size() >= problemData.binLabels.size());

    int positiveLabels = std::count(problemData.binLabels.begin(), problemData.binLabels.end(), 1.0);
    if (positiveLabels == 0 || positiveLabels == problemData.binLabels.size()) {
        firstClass = static_cast<int>(problemData.binLabels[0]);
        classCount = 1;
        return;
    }

    // Apply some weighting for very unbalanced data
    if (args.inbalanceLabelsWeighting) {
        problemData.labelsCount = 2;
        problemData.labels = new int[2];
        problemData.labels[0] = 0;
        problemData.labels[1] = 1;
        problemData.labelsWeights = new Real[2];

        int negativeLabels = static_cast<int>(problemData.binLabels.size()) - positiveLabels;
        if (negativeLabels > positiveLabels) {
            problemData.labelsWeights[0] = 1.0;
            problemData.labelsWeights[1] = 1.0 + log(static_cast<Real>(negativeLabels) / positiveLabels);
        } else {
            problemData.labelsWeights[0] = 1.0 + log(static_cast<Real>(positiveLabels) / negativeLabels);
            problemData.labelsWeights[1] = 1.0;
        }
    }

    if (args.optimizerType == liblinear) trainLiblinear(problemData, args);
    else trainOnline(problemData, args);

    // Calculate final train loss
    if(args.reportLoss) {
        Real meanLoss = 0;
        const int examples = problemData.binFeatures.size();
        for (int r = 0; r < examples; ++r) {
            Real pred =  W->dot(problemData.binFeatures[r]);
            if (firstClass == 0) pred *= -1;
            const Real loss = lossFunc(problemData.binLabels[r], pred, problemData.invPs);
            if (!std::isinf(loss) && !std::isnan(loss)) meanLoss += loss;
        }
        meanLoss /= examples;
        problemData.loss = meanLoss;
    }

    // Apply threshold and calculate number of non-zero weights
    pruneWeights(args.weightsThreshold);
    if(W->sparseMem() < W->denseMem()){
        auto newW = new SparseVector(*W);
        delete W;
        W = newW;
    }

    delete[] problemData.labels;
    delete[] problemData.labelsWeights;
}

void Base::setupOnlineTraining(Args& args, int n, bool startWithDenseW) {
    // Set loss
    setLoss(args.lossType);

    // Init weights
    if (n != 0 && startWithDenseW) {
        W = new Vector(n);
        if (args.optimizerType == adagrad) G = new Vector(n);
    } else {
        W = new MapVector();
        if (args.optimizerType == adagrad) G = new MapVector();
    }

    classCount = 2;
    firstClass = 1;
    t = 0;
}

void Base::finalizeOnlineTraining(Args& args) {
    // Because aux bases needs previous weights
    /*
    if (firstClassCount == t || firstClassCount == 0) {
        classCount = 1;
        if (firstClassCount == 0) firstClass = 1 - firstClass;
    }
    */
    pruneWeights(args.weightsThreshold);
}

Real Base::predictValue(SparseVector& features) {
    if (classCount < 2 || !W) return static_cast<Real>((1 - 2 * firstClass) * -10);
    Real val = W->dot(features);
    if (firstClass == 0) val *= -1;

    return val;
}

Real Base::predictProbability(SparseVector& features) {
    Real val = predictValue(features);
    if (lossType == squaredHinge)
        //val = 1.0 / (1.0 + std::exp(-2 * val)); // Probability for squared Hinge loss solver
        val = std::exp(-std::pow(std::max(0.0, 1.0 - val), 2));
    else
        val = 1.0 / (1.0 + std::exp(-val)); // Probability
    return val;
}

void Base::clear() {
    classCount = 0;
    firstClass = 0;
    firstClassCount = 0;
    t = 0;
    delete W;
    W = nullptr;
    delete G;
    G = nullptr;
}

void Base::pruneWeights(Real threshold) {
    if(W != nullptr) {
        Real bias = W->at(1); // Do not prune bias feature
        W->prune(threshold);
        W->insertD(1, bias);
    }
}

void Base::save(std::ofstream& out, bool saveGrads) {
    saveVar(out, classCount);
    saveVar(out, firstClass);
    saveVar(out, lossType);

    if (classCount > 1) {
        // Save main weights vector size to estimate optimal representation
        size_t s = W->size();
        size_t n0 = W->nonZero();
        saveVar(out, s);
        saveVar(out, n0);

        W->save(out);
        bool grads = (saveGrads && G != nullptr);
        saveVar(out, grads);
        if (grads) G->save(out);
    }
}

void Base::load(std::ifstream& in, bool loadGrads, RepresentationType loadAs) {
    clear();
    loadVar(in, classCount);
    loadVar(in, firstClass);
    loadVar(in, lossType);
    setLoss(lossType);

    if (classCount > 1) {
        size_t s;
        size_t n0;
        loadVar(in, s);
        loadVar(in, n0);

        // Decide on optimal representation in case of map
        size_t denseSize = Vector::estimateMem(s, n0);
        size_t mapSize = MapVector::estimateMem(s, n0);
        size_t sparseSize = SparseVector::estimateMem(s, n0);
        bool loadMap = loadGrads || (mapSize < denseSize || s == 0);
        bool loadSparse = (sparseSize < denseSize || s == 0);

        if(loadAs == map && loadMap) W = new MapVector();
        else if(loadAs == sparse && loadSparse) W = new SparseVector();
        else W = new Vector();
        W->load(in);

        bool grads;
        loadVar(in, grads);
        if(grads) {
            if(loadGrads){
                if(loadAs == map && loadMap) G = new MapVector();
                else if(loadAs == sparse && loadSparse) G = new SparseVector();
                else G = new Vector();
                G->load(in);
            }
            else AbstractVector::skipLoad(in);
        }
//        Log(CERR) << "  Load base: classCount: " << classCount << ", firstClass: "
//                  << firstClass << ", non-zero weights: " << n0 << "/" << s << "\n";
    }
}

void Base::setLoss(LossType lossType){
    this->lossType = lossType;
    if (lossType == logistic) {
        lossFunc = &logisticLoss;
        gradFunc = &logisticGrad;
    }
    else if (lossType == squaredHinge) {
        gradFunc = &squaredHingeGrad;
    }
    else if (lossType == unLogistic) {
        lossFunc = &unbiasedLogisticLoss;
        gradFunc = &unbiasedLogisticGrad;
    }
    else if (lossType == pwLogistic) {
        lossFunc = &pwLogisticLoss;
        gradFunc = &pwLogisticGrad;
    }
    else
        throw std::invalid_argument("Unknown loss function type");
}

void Base::setFirstClass(int first){
    if(firstClass != first){
        if(W != nullptr) W->invert();
        if(G != nullptr) G->invert();
        firstClass = first;
        firstClassCount = t - firstClassCount;
    }
}

Base* Base::copy() {
    Base* c = new Base();
    if (W != nullptr) c->W = W->copy();
    if (G != nullptr) c->G = G->copy();

    c->firstClass = firstClass;
    c->classCount = classCount;
    c->lossType = lossType;
    c->lossFunc = lossFunc;
    c->gradFunc = gradFunc;
    c->t = t;
    c->firstClassCount = firstClassCount;

    return c;
}

Base* Base::copyInverted() {
    Base* c = copy();
    if(c->W != nullptr) c->W->invert();
    // For AdaGrad, G accumulates squares of features and gradients it needs to be always positive
    //if(c->G != nullptr) c->G->invert();
    return c;
}

void Base::to(RepresentationType type) {
    auto newW = vecTo(W, type);
    if(newW != nullptr){
        delete W;
        W = newW;
    }
    auto newG = vecTo(G, type);
    if(newG != nullptr){
        delete G;
        G = newG;
    }
}

RepresentationType Base::getType() {
    if(W != nullptr) return W->type();
    else return dense;
}

unsigned long long Base::mem(){
    unsigned long long totalMem = sizeof(Base);
    if(W != nullptr) totalMem += W->mem();
    if(G != nullptr) totalMem += G->mem();
    return totalMem;
}

AbstractVector* Base::vecTo(AbstractVector* vec, RepresentationType type){
    if(vec == nullptr || vec->type() == type) return vec;
    AbstractVector* newVec;
    if(type == dense) newVec = new Vector(*vec);
    else if(type == map) newVec = new MapVector(*vec);
    else if(type == sparse) newVec = new SparseVector(*vec);
    else throw std::invalid_argument("Unknown representation type");
    return newVec;
}
