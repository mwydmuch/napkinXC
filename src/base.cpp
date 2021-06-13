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
#include "threads.h"


//TODO: Refactor base class

Base::Base() {
    hingeLoss = false;

    wSize = 0;
    nonZeroW = 0;
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

void Base::update(double label, Feature* features, Args& args) {
    std::lock_guard<std::mutex> lock(updateMtx);

    unsafeUpdate(label, features, args);
}

void Base::unsafeUpdate(double label, Feature* features, Args& args) {
    if (args.tmax != -1 && args.tmax < t) return;

    ++t;
    if (label == firstClass) ++firstClassCount;

    double pred = predictValue(features);
    double grad;
    if(args.lossType == logistic)
        grad = logisticGrad(label, pred, 0);
    else
        grad = squaredHingeGrad(label, pred, 0);

    if (args.optimizerType == sgd)
        updateSGD(W, G, features, grad, t, args);
    else if (args.optimizerType == adagrad)
        updateAdaGrad(W, G, features, grad, t, args);
    else throw std::invalid_argument("Unknown optimizer type");

    // Check if we should change sparse W to dense W
//    if (mapW != nullptr && wSize != 0) {
//        nonZeroW = mapW->size();
//        if (mapSize() > denseSize()) toDense();
//    }
}

void Base::trainLiblinear(ProblemData& problemData, Args& args) {
    double cost = args.cost;
    if (args.autoCLog)
        cost *= 1.0 + log(static_cast<double>(problemData.r) / problemData.binFeatures.size());
    if (args.autoCLin)
        cost *= static_cast<double>(problemData.r) / problemData.binFeatures.size();

    problem P = {.l = static_cast<int>(problemData.binLabels.size()),
                 .n = problemData.n,
                 .y = problemData.binLabels.data(),
                 .x = problemData.binFeatures.data(),
                 .bias = -1,
                 .W = problemData.instancesWeights.data()};

    parameter C = {.solver_type = args.solverType,
                   .eps = args.eps,
                   .C = cost,
                   .nr_weight = problemData.labelsCount,
                   .weight_label = problemData.labels,
                   .weight = problemData.labelsWeights,
                   .p = 0,
                   .init_sol = NULL,
                   .max_iter = args.maxIter};

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    model* M = train_liblinear(&P, &C);

    assert(M->nr_class <= 2);
    assert(M->nr_feature == problemData.n);

    // Set base's attributes
    wSize = problemData.n + 1;
    firstClass = M->label[0];
    classCount = M->nr_class;

    // Copy weights
    W = new Vector<Weight>((size_t)wSize);
    for (int i = 0; i < problemData.n; ++i) W->insertD(i + 1, M->w[i]); // Shift by -1

    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC ||
                args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    free_model_content(M);
    free(M);
}

void Base::trainOnline(ProblemData& problemData, Args& args) {
    setupOnlineTraining(args, problemData.n, true);

    // Set loss function
    double (*lossFunc)(double, double, double);
    double (*gradFunc)(double, double, double);
    if (args.lossType == logistic) {
        lossFunc = &logisticLoss;
        gradFunc = &logisticGrad;
    }
    else if (args.lossType == squaredHinge)
        gradFunc = &squaredHingeGrad;
    else if (args.lossType == pwLogistic) {
        lossFunc = &pwLogisticLoss;
        gradFunc = &pwLogisticGrad;
    }
    else
        throw std::invalid_argument("Unknown loss function type");

    // Set update function
    void (*updateFunc)(AbstractVector<Weight>*, AbstractVector<Weight>*, Feature*, double, int, Args&);
    if(args.optimizerType == sgd)
        updateFunc = &updateSGD;
    else if (args.optimizerType == adagrad)
        updateFunc = &updateAdaGrad;
    else
        throw std::invalid_argument("Unknown online update function type");

    const int examples = problemData.binFeatures.size();
    double loss = 0;
    for (int e = 0; e < args.epochs; ++e)
        for (int r = 0; r < examples; ++r) {
            double label = problemData.binLabels[r];
            Feature* features = problemData.binFeatures[r];

            if (args.tmax != -1 && args.tmax < t) break;

            ++t;
            if (problemData.binLabels[r] == firstClass) ++firstClassCount;

            double pred = W->dot(features);
            double grad = gradFunc(label, pred, problemData.invPs) * problemData.instancesWeights[r];
            updateFunc(W, G, features, grad, t, args);

            // Report loss
//            loss += lossFunc(label, pred, problemData.invPs);
//            int iter = e * examples + r;
//            if(iter % 10000 == 9999)
//                Log(CERR) << "  Iter: " << iter << "/" << args.epochs * examples << ", loss: " << loss / iter << "\n";
        }

    finalizeOnlineTraining(args);
}

void Base::train(ProblemData& problemData, Args& args) {

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
        problemData.labelsWeights = new double[2];

        int negativeLabels = static_cast<int>(problemData.binLabels.size()) - positiveLabels;
        if (negativeLabels > positiveLabels) {
            problemData.labelsWeights[0] = 1.0;
            problemData.labelsWeights[1] = 1.0 + log(static_cast<double>(negativeLabels) / positiveLabels);
        } else {
            problemData.labelsWeights[0] = 1.0 + log(static_cast<double>(positiveLabels) / negativeLabels);
            problemData.labelsWeights[1] = 1.0;
        }
    }

    if (args.optimizerType == liblinear) trainLiblinear(problemData, args);
    else trainOnline(problemData, args);

    // Apply threshold and calculate number of non-zero weights
    pruneWeights(args.weightsThreshold);
    //if(W->sparseMem() < W->denseMem()) W = new SparseVector<Weight>(*W);

    delete[] problemData.labels;
    delete[] problemData.labelsWeights;
}

void Base::setupOnlineTraining(Args& args, int n, bool startWithDenseW) {
    wSize = n;
    if (wSize != 0 && startWithDenseW) {
        W = new Vector<Weight>(wSize);
        if (args.optimizerType == adagrad) G = new Vector<Weight>(wSize);
    } else {
        W = new MapVector<Weight>(wSize);
        if (args.optimizerType == adagrad) G = new MapVector<Weight>(wSize);
    }

    classCount = 2;
    firstClass = 1;
    t = 0;
}

void Base::finalizeOnlineTraining(Args& args) {
    // Because aux bases needs previous weights, TODO: Change this later
    /*
    if (firstClassCount == t || firstClassCount == 0) {
        classCount = 1;
        if (firstClassCount == 0) firstClass = 1 - firstClass;
    }
    */
    nonZeroW = W->nonZero();
    nonZeroG = nonZeroW;
    pruneWeights(args.weightsThreshold);
}

double Base::predictValue(Feature* features) {
    if (classCount < 2) return static_cast<double>((1 - 2 * firstClass) * -10);
    double val = W->dot(features);
    if (firstClass == 0) val *= -1;

    return val;
}

double Base::predictProbability(Feature* features) {
    double val = predictValue(features);
    if (hingeLoss)
        //val = 1.0 / (1.0 + std::exp(-2 * val)); // Probability for squared Hinge loss solver
        val = std::exp(-std::pow(std::max(0.0, 1.0 - val), 2));
    else
        val = 1.0 / (1.0 + std::exp(-val)); // Probability
    return val;
}

void Base::clear() {
    hingeLoss = false;

    wSize = 0;
    nonZeroW = 0;
    classCount = 0;
    firstClass = 0;
    firstClassCount = 0;
    t = 0;

    clearW();
    clearG();
}

void Base::clearW() {
    delete W;
    W = nullptr;
}

void Base::clearG() {
    delete G;
    G = nullptr;
}

void Base::pruneWeights(double threshold) {
    nonZeroW = 0;

    W->forEachID([&](const int& i, Weight& w) {
        if (i == 1 || (w != 0 && fabs(w) >= threshold)) ++nonZeroW; // Do not prune bias feature
        else w = 0;
    });
}

void Base::save(std::ostream& out, bool saveGrads) {
    out.write((char*)&classCount, sizeof(classCount));
    out.write((char*)&firstClass, sizeof(firstClass));

    if (classCount > 1) {
        // Decide on optimal file coding

        out.write((char*)&hingeLoss, sizeof(hingeLoss));
        //out.write((char*)&wSize, sizeof(wSize));
        //out.write((char*)&nonZeroW, sizeof(nonZeroW));

        if(W != nullptr) W->save(out);
        bool grads = (saveGrads && G != nullptr);
        saveVar(out, grads);
        if(grads) G->save(out);
    }
}

void Base::load(std::istream& in, bool loadGrads, RepresentationType loadAs) {
    in.read((char*)&classCount, sizeof(classCount));
    in.read((char*)&firstClass, sizeof(firstClass));

    if (classCount > 1) {
        in.read((char*)&hingeLoss, sizeof(hingeLoss));
        //in.read((char*)&wSize, sizeof(wSize));
        //in.read((char*)&nonZeroW, sizeof(nonZeroW));

        //TODO: Improve this
        bool loadSparse = true;
        if(loadSparse && loadAs == map){
            W = new MapVector<Weight>();
            G = new MapVector<Weight>();
        }
        else if(loadSparse && loadAs == sparse){
            W = new SparseVector<Weight>();
            G = new SparseVector<Weight>();
        }
        else{
            W = new Vector<Weight>();
            G = new Vector<Weight>();
        }
        W->load(in);

        bool grads;
        loadVar(in, grads);
        if(grads) {
            if(loadGrads) G->load(in);
            else{
                G->skipLoad(in);
                delete G;
            }
        }

//        Log(CERR) << "  Load base: classCount: " << classCount << ", firstClass: "
//                  << firstClass << ", weights: " << nonZeroW << "/" << wSize << "\n";
    }
}

size_t Base::size() {
    size_t size = sizeof(Base);
    if (W) size += W->size();
    if (G) size += G->size();
    return size;
}

void Base::printWeights() {
    W->forEachID([&](const int& i, Weight& w) { Log(CERR) << i << ":" << w << " "; });
    Log(CERR) << "\n";
}

void Base::setFirstClass(int first){
    if(firstClass != first){
        W->invert();
        if(G != nullptr) G->invert();
        firstClass = first;
    }
}

Base* Base::copy() {
    Base* copy = new Base();
    if (W) copy->W = W->copy();
    if (G) copy->G = G->copy();

    copy->firstClass = firstClass;
    copy->classCount = classCount;
    copy->wSize = wSize;
    copy->nonZeroW = nonZeroW;

    return copy;
}

Base* Base::copyInverted() {
    Base* c = copy();
    c->W->invert();
    if(c->G != nullptr) c->G->invert();
    return c;
}

void Base::to(RepresentationType type) {
    if(W != nullptr){
        AbstractVector<Weight>* newW;
        if(type == dense) newW = new Vector<Weight>(*W);
        else if(type == map) newW = new MapVector<Weight>(*W);
        else if(type == sparse) newW = new SparseVector<Weight>(*W);
        delete W;
        W = newW;
    }
    if(G != nullptr){
        AbstractVector<Weight>* newG;
        if(type == dense) newG = new Vector<Weight>(*G);
        else if(type == map) newG = new MapVector<Weight>(*G);
        else if(type == sparse) newG = new SparseVector<Weight>(*G);
        delete G;
        G = newG;
    }
}

void Base::toMap() {
    to(map);
}

void Base::toDense() {
    to(dense);
}

void Base::toSparse() {
    to(sparse);
}

void Base::updateSGD(AbstractVector<Weight>* W, AbstractVector<Weight>* G, Feature* features, double grad, int t, Args& args) {
    double eta = args.eta;
    double lr = eta * sqrt(1.0 / t);
    Feature* f = features;
    while (f->index != -1) {
        (*W)[f->index] -= lr * grad * f->value;
        ++f;
    }
}

void Base::updateAdaGrad(AbstractVector<Weight>* W, AbstractVector<Weight>* G, Feature* features, double grad, int t, Args& args) {
    double eta = args.eta;
    double eps = args.adagradEps;
    Feature* f = features;
    while (f->index != -1) {
        (*G)[f->index] += f->value * f->value * grad * grad;
        double lr = eta * std::sqrt(1.0 / (eps + (*G)[f->index]));
        (*W)[f->index] -= lr * (grad * f->value);
        ++f;
        // TODO: add correct regularization
        //double reg = l2 * W[f->index];
        //W[f->index] -= lr * (grad * f->value + reg);
    }
}
