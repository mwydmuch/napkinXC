/**
 * Copyright (c) 2018 by Marek Wydmuch
 * Copyright (c) 2019 by Marek Wydmuch, Kalina Kobus
 * All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <random>

#include "base.h"
#include "linear.h"
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
    pi = 1.0;
}

Base::Base(Args& args) {
    Base();

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
    double grad = (1.0 / (1.0 + std::exp(-pred))) - label;

    if (args.optimizerType == sgd) {
        if (mapW != nullptr)
            updateSGD((*mapW), features, grad, args.eta);
        else if (W != nullptr)
            updateSGD(W, features, grad, args.eta);
    } else if (args.optimizerType == adagrad) {
        if (mapW != nullptr)
            updateAdaGrad((*mapW), (*mapG), features, grad, args.eta, args.adagradEps);
        else if (W != nullptr)
            updateAdaGrad(W, G, features, grad, args.eta, args.adagradEps);
    } else if (args.optimizerType == fobos) {
        if (mapW != nullptr)
            updateFobos((*mapW), features, grad, args.eta, args.fobosPenalty);
        else if (W != nullptr)
            updateFobos(W, features, grad, args.eta, args.fobosPenalty);
    }

    // Check if we should change sparse W to dense W
    if (mapW != nullptr && wSize != 0) {
        nonZeroW = mapW->size();
        if (mapSize() > denseSize()) toDense();
    }
}

void Base::trainLiblinear(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
                          std::vector<double>* instancesWeights, int positiveLabels, Args& args) {

    model* M = nullptr;
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

    auto y = binLabels.data();
    auto x = binFeatures.data();
    int l = static_cast<int>(binLabels.size());

    if (instancesWeights == nullptr) {
        instancesWeights = new std::vector<double>(l);
        std::fill(instancesWeights->begin(), instancesWeights->end(), 1.0);
    }

    problem P = {.l = l, .n = n, .y = y, .x = x, .bias = (args.bias > 0 ? 1.0 : -1.0), .W = instancesWeights->data()};

    parameter C = {.solver_type = args.solverType,
                   .eps = args.eps,
                   .C = args.cost,
                   .nr_weight = labelsCount,
                   .weight_label = labels,
                   .weight = labelsWeights,
                   .p = 0,
                   .init_sol = NULL};

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    // Optimize C for small datasets
    if (args.cost < 0 && binLabels.size() > 100 && binLabels.size() < 1000) {
        double bestC = -1;
        double bestP = -1;
        double bestScore = -1;
        find_parameters(&P, &C, 1, 4.0, -1, &bestC, &bestP, &bestScore);
        C.C = bestC;
    } else if (args.cost < 0)
        C.C = 8.0;

    M = train_liblinear(&P, &C);

    assert(M->nr_class <= 2);
    assert(M->nr_feature + (args.bias > 0 ? 1 : 0) == n);

    // Set base's attributes
    wSize = n;
    firstClass = M->label[0];
    classCount = M->nr_class;

    // Copy weights
    W = new Weight[n];
    for (int i = 0; i < n; ++i) W[i + 1] = M->w[i];
    delete[] M->w;

    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC ||
                args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    delete[] M->label;
    delete[] labels;
    delete[] labelsWeights;
    delete M;
}

void Base::trainOnline(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args& args) {
    setupOnlineTraining(args, n);

    const int examples = binFeatures.size() * args.epochs;
    for (int i = 0; i < examples; ++i) {
        int r = i % binFeatures.size();
        unsafeUpdate(binLabels[r], binFeatures[r], args);
    }

    finalizeOnlineTraining(args);
}

void Base::train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures,
                 std::vector<double>* instancesWeights, Args& args) {

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

    assert(binLabels.size() == binFeatures.size());
    if (instancesWeights != nullptr) assert(instancesWeights->size() == binLabels.size());

    if (args.optimizerType == liblinear)
        trainLiblinear(n, binLabels, binFeatures, instancesWeights, positiveLabels, args);
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
    pi = 1.0;
    t = 0;
}

void Base::finalizeOnlineTraining(Args& args) {
    if (pi != 1) forEachW([&](Weight& w) { w /= pi; });

    if (firstClassCount == t || firstClassCount == 0) {
        classCount = 1;
        if (firstClassCount == 0) firstClass = 1 - firstClass;
    }

    pruneWeights(args.weightsThreshold);
}

double Base::predictValue(Feature* features) {
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
    val /= pi; // Fobos

    return val;
}

double Base::predictProbability(Feature* features) {
    if (classCount < 2) return static_cast<double>(firstClass);
    double val = predictValue(features);
    if (hingeLoss)
        val = 1.0 / (1.0 + std::exp(-2 * val)); // Probability for squared Hinge loss solver
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
        // bool saveSparse = sparseSize() < denseSize();
        bool saveSparse = true;

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

    // std::cerr << "  Saved base: sparse: " << saveSparse << ", classCount: " << classCount << ", firstClass: "
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
            mapW = new UnorderedMap<int, Weight>();

            int index;
            Weight w;

            for (int i = 0; i < nonZeroW; ++i) {
                in.read((char*)&index, sizeof(index));
                in.read((char*)&w, sizeof(Weight));
                if (sparseW != nullptr) sparseW[i] = {index, w};
                if (mapW != nullptr) mapW->insert({index, w});
            }
        } else {
            W = new Weight[wSize];
            std::memset(W, 0, wSize * sizeof(Weight));
            in.read((char*)W, wSize * sizeof(Weight));
        }
    }
}

size_t Base::size() {
    if (W) return denseSize();
    if (mapW) return mapSize();
    if (sparseW) return sparseSize();
    return 0;
}

void Base::printWeights() {
    forEachIW([&](const int& i, Weight& w) { std::cerr << i << ":" << w << " "; });
    std::cerr << std::endl;
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
