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

#include <fstream>
#include <iostream>
#include <random>

#include "base.h"
#include "linear.h"
#include "log.h"
#include "misc.h"
#include "threads.h"


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
    mapW = nullptr;
    mapG = nullptr;
    sparseW = nullptr;
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
        grad = logisticGrad(label, pred);
    else
        grad = squaredHingeGrad(label, pred);

    if (args.optimizerType == sgd) {
        if (mapW != nullptr)
            updateSGD((*mapW), (*mapG), features, grad, t, args);
        else if (W != nullptr)
            updateSGD(W, G, features, grad, t, args);
    } else if (args.optimizerType == adagrad) {
        if (mapW != nullptr)
            updateAdaGrad((*mapW), (*mapG), features, grad, t, args);
        else if (W != nullptr)
            updateAdaGrad(W, G, features, grad, t, args);
    }

    // Check if we should change sparse W to dense W
    if (mapW != nullptr && wSize != 0) {
        nonZeroW = mapW->size();
        if (mapSize() > denseSize()) toDense();
    }
}

void Base::trainLiblinear(int n, int r, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
                          std::vector<double>* instancesWeights, int positiveLabels, Args& args) {

    int labelsCount = 0;
    int* labels = NULL;
    double* labelsWeights = NULL;
    int negativeLabels = static_cast<int>(binLabels.size()) - positiveLabels;

    // Apply some weighting for very unbalanced data
    if (args.inbalanceLabelsWeighting) {
        labelsCount = 2;
        labels = new int[2];
        labels[0] = 0;
        labels[1] = 1;

        labelsWeights = new double[2];
        if (negativeLabels > positiveLabels) {
            labelsWeights[0] = 1.0;
            labelsWeights[1] = 1.0 + log(static_cast<double>(negativeLabels) / positiveLabels);
        } else {
            labelsWeights[0] = 1.0 + log(static_cast<double>(positiveLabels) / negativeLabels);
            labelsWeights[1] = 1.0;
        }
    }

    double cost = args.cost;
    if(args.autoCLog)
        cost *= 1.0 + log(static_cast<double>(r) / binFeatures.size());
    if (args.autoCLin)
        cost *= static_cast<double>(r) / binFeatures.size();

    auto y = binLabels.data();
    auto x = binFeatures.data();
    int l = static_cast<int>(binLabels.size());

    bool deleteInstanceWeights = false;
    if (instancesWeights == nullptr) {
        instancesWeights = new std::vector<double>(l);
        std::fill(instancesWeights->begin(), instancesWeights->end(), 1.0);
        deleteInstanceWeights = true;
    }

    problem P = {.l = l,
                 .n = n,
                 .y = y,
                 .x = x,
                 .bias = -1,
                 .W = instancesWeights->data()};

    parameter C = {.solver_type = args.solverType,
                   .eps = args.eps,
                   .C = cost,
                   .nr_weight = labelsCount,
                   .weight_label = labels,
                   .weight = labelsWeights,
                   .p = 0,
                   .init_sol = NULL,
                   .max_iter = args.maxIter};

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    model* M = train_liblinear(&P, &C);

    assert(M->nr_class <= 2);
    assert(M->nr_feature == n);

    // Set base's attributes
    wSize = n + 1;
    firstClass = M->label[0];
    classCount = M->nr_class;

    // Copy weights
    W = new Weight[wSize];
    W[0] = 0;
    for (int i = 0; i < n; ++i) W[i + 1] = M->w[i];

    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC ||
                args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    free_model_content(M);
    free(M);
    delete[] labels;
    delete[] labelsWeights;
    if(deleteInstanceWeights) delete instancesWeights;
}

void Base::trainOnline(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args& args) {
    setupOnlineTraining(args, n, true);

    // Set loss function
    double (*gradFunc)(double, double);
    if (args.lossType == logistic)
        gradFunc = &logisticGrad;
    else if (args.lossType == squaredHinge)
        gradFunc = &squaredHingeGrad;
    else
        throw std::invalid_argument("Unknown loss function type");

    // Set update function
    void (*updateFunc)(Weight*&, Weight*&, Feature*, double, int, Args&);
    if(args.optimizerType == sgd)
        updateFunc = &updateSGD<Weight*>;
    else if (args.optimizerType == adagrad)
        updateFunc = &updateAdaGrad<Weight*>;
    else
        throw std::invalid_argument("Unknown online update function type");

    const int examples = binFeatures.size();
    for (int e = 0; e < args.epochs; ++e)
        for (int r = 0; r < examples; ++r) {

            double label = binLabels[r];
            Feature* features = binFeatures[r];

            if (args.tmax != -1 && args.tmax < t) break;

            ++t;
            if (binLabels[r] == firstClass) ++firstClassCount;

            //double pred = predictValue(binFeatures[r]);
            double pred = dotVectors(features, W, wSize);
            double grad = gradFunc(label, pred);
            updateFunc(W, G, features, grad, t, args);
        }

    finalizeOnlineTraining(args);
}

void Base::train(int n, int r, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
                 std::vector<double>* instancesWeights, Args& args) {

    if(instancesWeights != nullptr && args.optimizerType != liblinear)
        throw std::invalid_argument("train: optimizer type does not support training with weights");

    if (binLabels.empty()) {
        firstClass = 0;
        classCount = 0;
        return;
    }

    int positiveLabels = std::count(binLabels.begin(), binLabels.end(), 1.0);
    if (positiveLabels == 0 || positiveLabels == binLabels.size()) {
        firstClass = static_cast<int>(binLabels[0]);
        classCount = 1;
        return;
    }

    //assert(binLabels.size() == binFeatures.size());
    if (instancesWeights != nullptr) assert(instancesWeights->size() == binLabels.size());

    if (args.optimizerType == liblinear)
        trainLiblinear(n, r, binLabels, binFeatures, instancesWeights, positiveLabels, args);
    else
        trainOnline(n, binLabels, binFeatures, args);

    // Apply threshold and calculate number of non-zero weights
    pruneWeights(args.weightsThreshold);
    if (sparseSize() < denseSize()) toSparse();
}

void Base::setupOnlineTraining(Args& args, int n, bool startWithDenseW) {
    wSize = n;
    if (wSize != 0 && startWithDenseW) {
        W = new Weight[wSize];
        std::memset(W, 0, wSize * sizeof(Weight));
        if (args.optimizerType == adagrad) {
            G = new Weight[wSize];
            std::memset(G, 0, wSize * sizeof(Weight));
        }
    } else {
        mapW = new UnorderedMap<int, Weight>();
        if (args.optimizerType == adagrad) mapG = new UnorderedMap<int, Weight>();
    }

    classCount = 2;
    firstClass = 1;
}

void Base::finalizeOnlineTraining(Args& args) {
    if (firstClassCount == t || firstClassCount == 0) {
        classCount = 1;
        if (firstClassCount == 0) firstClass = 1 - firstClass;
    }

    pruneWeights(args.weightsThreshold);
}

double Base::predictValue(Feature* features) {
    if (classCount < 2) return static_cast<double>(firstClass * 10);
    double val = 0;

    if (mapW) { // Sparse features dot sparse weights in hash map
        Feature* f = features;
        while (f->index != -1) {
            auto w = mapW->find(f->index);
            if (w != mapW->end()) val += w->second * f->value;
            ++f;
        }
    } else if (W)
        val = dotVectors(features, W, wSize); // Sparse features dot dense weights
    else
        throw std::runtime_error("Prediction using sparse features and sparse weights is not supported!");

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

void Base::forEachW(const std::function<void(Weight&)>& func) {
    if (W != nullptr)
        for (int i = 0; i < wSize; ++i) func(W[i]);
    else if (mapW != nullptr)
        for (auto& w : *mapW) func(w.second);
    else if (sparseW != nullptr)
        for (int i = 0; i < nonZeroW; ++i) func(sparseW[i].second);
}

void Base::forEachIW(const std::function<void(const int&, Weight&)>& func) {
    if (W != nullptr)
        for (int i = 0; i < wSize; ++i) func(i, W[i]);
    else if (mapW != nullptr)
        for (auto& w : *mapW) func(w.first, w.second);
    else if (sparseW != nullptr)
        for (int i = 0; i < nonZeroW; ++i) func(sparseW[i].first, sparseW[i].second);
}

void Base::clear() {
    delete[] W;
    W = nullptr;
    delete[] G;
    G = nullptr;

    delete mapW;
    mapW = nullptr;
    delete mapG;
    mapG = nullptr;

    delete[] sparseW;
    sparseW = nullptr;
}

void Base::toMap() {
    if (mapW == nullptr) {
        mapW = new UnorderedMap<int, Weight>();

        assert(W != nullptr);
        for (int i = 0; i < wSize; ++i)
            if (W[i] != 0) mapW->insert({i, W[i]});
        delete[] W;
        W = nullptr;
    }

    if (mapG == nullptr && G != nullptr) {
        mapG = new UnorderedMap<int, Weight>();

        for (int i = 0; i < wSize; ++i)
            if (G[i] != 0) mapG->insert({i, W[i]});
        delete[] G;
        G = nullptr;
    }
}

void Base::toDense() {
    if (W == nullptr) {
        W = new Weight[wSize];
        std::memset(W, 0, wSize * sizeof(Weight));
        assert(mapW != nullptr);
        for (const auto& w : *mapW) W[w.first] = w.second;
        delete mapW;
        mapW = nullptr;
    }

    if (G == nullptr && mapG != nullptr) {
        G = new Weight[wSize];
        std::memset(G, 0, wSize * sizeof(Weight));

        for (const auto& w : *mapG) G[w.first] = w.second;
        delete mapG;
        mapG = nullptr;
    }
}

void Base::toSparse() {
    if (sparseW == nullptr) {
        auto tmpSparseW = new SparseWeight[nonZeroW];
        auto sW = tmpSparseW;

        forEachIW([&](const int& i, Weight& w) {
            if (w != 0) {
                sW->first = i;
                sW->second = w;
                ++sW;
            }
        });

        clear();
        sparseW = tmpSparseW;
    }
}

void Base::pruneWeights(double threshold) {
    nonZeroW = 0;

    forEachW([&](Weight& w) {
        if (w != 0 && fabs(w) >= threshold)
            ++nonZeroW;
        else
            w = 0;
    });
}

void Base::save(std::ostream& out) {
    out.write((char*)&classCount, sizeof(classCount));
    out.write((char*)&firstClass, sizeof(firstClass));

    if (classCount > 1) {
        // Decide on optimal file coding
        bool saveSparse = sparseSize() < denseSize() || W == nullptr;

        out.write((char*)&hingeLoss, sizeof(hingeLoss));
        out.write((char*)&wSize, sizeof(wSize));
        out.write((char*)&nonZeroW, sizeof(nonZeroW));
        out.write((char*)&saveSparse, sizeof(saveSparse));

        if (saveSparse) {
            forEachIW([&](const int& i, Weight& w) {
                if (w != 0) {
                    out.write((char*)&i, sizeof(i));
                    out.write((char*)&w, sizeof(w));
                }
            });
        } else
            out.write((char*)W, wSize * sizeof(Weight));
    }

    // LOG(CERR) << "  Saved base: sparse: " << saveSparse << ", classCount: " << classCount << ", firstClass: "
    //    << firstClass << ", weights: " << nonZeroCount << "/" << wSize << ", size: " << size()/1024 << "/" <<
    //    denseSize()/1024 << "K\n";
}

void Base::load(std::istream& in) {
    in.read((char*)&classCount, sizeof(classCount));
    in.read((char*)&firstClass, sizeof(firstClass));

    if (classCount > 1) {
        bool loadSparse;

        in.read((char*)&hingeLoss, sizeof(hingeLoss));
        in.read((char*)&wSize, sizeof(wSize));
        in.read((char*)&nonZeroW, sizeof(nonZeroW));
        in.read((char*)&loadSparse, sizeof(loadSparse));

        if (loadSparse) {
            bool loadAsMap = mapSize() < denseSize() && wSize > 50000;

            if(loadAsMap){
                mapW = new UnorderedMap<int, Weight>();
                mapW->reserve(nonZeroW);
            }
            else{
                W = new Weight[wSize];
                std::memset(W, 0, wSize * sizeof(Weight));
            }

            int index;
            Weight w;
            for (int i = 0; i < nonZeroW; ++i) {
                in.read((char*)&index, sizeof(index));
                in.read((char*)&w, sizeof(Weight));

                if (sparseW != nullptr) sparseW[i] = {index, w};
                if (mapW != nullptr) mapW->insert({index, w});
                if (W != nullptr) W[index] = w;
            }
        } else {
            W = new Weight[wSize];
            std::memset(W, 0, wSize * sizeof(Weight));
            in.read((char*)W, wSize * sizeof(Weight));
        }
    }
}

size_t Base::size() {
    size_t size = sizeof(Base);
    if (W) size += denseSize();
    if (mapW) size += mapSize();
    if (sparseW) size += sparseSize();
    return size;
}

void Base::printWeights() {
    forEachIW([&](const int& i, Weight& w) { LOG(CERR) << i << ":" << w << " "; });
    LOG(CERR) << "\n";
}

void Base::invertWeights() {
    forEachW([&](Weight& w) { w *= -1; });
}

Base* Base::copy() {
    Base* copy = new Base();
    if (W) {
        copy->W = new Weight[wSize];
        std::memcmp(copy->W, W, wSize * sizeof(Weight));
    }
    if (G) {
        copy->G = new Weight[wSize];
        std::memcmp(copy->G, G, wSize * sizeof(Weight));
    }

    if (mapW) copy->mapW = new UnorderedMap<int, Weight>(mapW->begin(), mapW->end());
    if (mapG) copy->mapG = new UnorderedMap<int, Weight>(mapG->begin(), mapG->end());

    if (sparseW) {
        copy->sparseW = new SparseWeight[nonZeroW];
        std::memcmp(copy->sparseW, sparseW, (nonZeroW) * sizeof(SparseWeight));
    }

    copy->firstClass = firstClass;
    copy->classCount = classCount;
    copy->wSize = wSize;
    copy->nonZeroW = nonZeroW;

    return copy;
}

Base* Base::copyInverted() {
    Base* c = copy();
    c->invertWeights();
    return c;
}
