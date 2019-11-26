/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <fstream>
#include <iostream>
#include <random>

#include "base.h"
#include "linear.h"
#include "online_training.h"
#include "threads.h"
#include "utils.h"


Base::Base(bool onlineTraning){
    hingeLoss = false;

    wSize = 0;
    nonZeroW = 0;
    classCount = 0;
    firstClass = 0;
    t = 0;

    W = nullptr;
    G = nullptr;
    mapW = nullptr;
    mapG = nullptr;
    sparseW = nullptr;
    pi = 1.0;
}

Base::~Base(){
    delete[] W;
    delete[] G;
    delete mapW;
    delete mapG;
    delete[] sparseW;
}

void Base::update(double label, Feature* features, Args &args) {
    std::lock_guard<std::mutex> lock(updateMtx);

    unsafeUpdate(label, features, args);
}

void Base::unsafeUpdate(double label, Feature* features, Args &args) {
    if (args.tmax != -1 && args.tmax < t)
        return;

    t++;
    double pred = predictValue(features);
    double grad = (1.0 / (1.0 + std::exp(-pred))) - label;

    if (args.optimizerType == sgd) {
        double lr = args.eta * sqrt(1.0 / t);
        // Regularization is probably incorrect due to sparse features.
        double reg;
        Feature *f = features;

        while (f->index != -1) {
            if (mapW != nullptr) {
                reg = args.penalty * (*mapW)[f->index - 1];
                (*mapW)[f->index - 1] -= lr * (grad * f->value + reg);
//                (*mapW)[f->index - 1] -= lr * grad * f->value;
                if (f->index > wSize) wSize = f->index;

            } else if (W != nullptr) {
                reg = args.penalty * W[f->index - 1];
                W[f->index - 1] -= lr * (grad * f->value + reg);
//                W[f->index - 1] -= lr * grad * f->value;
            }

            ++f;
        }
    } else if (args.optimizerType == fobos) {
        // Implementation based on http://proceedings.mlr.press/v48/jasinska16-supp.pdf
        double lr = args.eta/(1 + args.eta*t*args.penalty);
        Feature *f = features;
        pi *= (1 + args.penalty*lr);

        while (f->index != -1) {
            if (mapW != nullptr) {
                (*mapW)[f->index - 1] -= pi * lr * grad * f->value;
                if (f->index > wSize) wSize = f->index;

            } else if (W != nullptr) {
                W[f->index - 1] -= pi* lr * grad * f->value;
            }
            ++f;
        }
    } else if (args.optimizerType == adagrad) {
        double lr;
        double reg;
        Feature *f = features;
        // Adagrad with sgd dropout like regularization
        while (f->index != -1) {
            if (mapW != nullptr && mapG != nullptr) {
                (*mapG)[f->index - 1] += f->value * f->value *  grad * grad;
                lr = args.eta * sqrt(1.0 / (args.adagrad_eps + (*mapG)[f->index - 1]));
                reg = args.penalty * (*mapW)[f->index - 1];
                (*mapW)[f->index - 1] -= lr * (grad * f->value + reg);
                if (f->index > wSize) wSize = f->index;
            } else if (W != nullptr && G != nullptr) {
                G[f->index - 1] += f->value * f->value *  grad * grad;
                lr = args.eta * sqrt(1.0 / (args.adagrad_eps + G[f->index - 1]));
                reg = args.penalty * W[f->index - 1];
                W[f->index - 1] -= lr * (grad * f->value + reg);
            }

            ++f;
        }
    }

    if (mapW != nullptr) {
        nonZeroW = mapW->size();
        if (mapSize() > denseSize()) toDense();
    }
}

void Base::trainLiblinear(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, int positiveLabels, Args &args){

    model *M = nullptr;
    int labelsCount = 0;
    int* labels = NULL;
    double* labelsWeights = NULL;
    int negativeLabels = static_cast<int>(binLabels.size()) - positiveLabels;

    // Apply some weighting for very unbalanced data
    if(args.labelsWeights){
        labelsCount = 2;
        labels = new int[2];
        labels[0] = 0;
        labels[1] = 1;

        labelsWeights = new double[2];
        if(negativeLabels > positiveLabels){
            labelsWeights[0] = 1.0;
            labelsWeights[1] = 1.0 + log(static_cast<double>(negativeLabels) / positiveLabels);
        } else{
            labelsWeights[0] = 1.0 + log(static_cast<double>(positiveLabels) / negativeLabels);
            labelsWeights[1] = 1.0;
        }
    }

    auto y = binLabels.data();
    auto x = binFeatures.data();
    int l = static_cast<int>(binLabels.size());

    problem P = {
            .l = l,
            .n = n,
            .y = y,
            .x = x,
            .bias = (args.bias > 0 ? 1.0 : -1.0)
    };

    parameter C = {
            .solver_type = args.solverType,
            .eps = args.eps,
            .C = args.cost,
            .nr_weight = labelsCount,
            .weight_label = labels,
            .weight = labelsWeights,
            .p = 0,
            .init_sol = NULL
    };

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    // Optimize C for small datasets
    if(args.cost < 0 && binLabels.size() > 100 && binLabels.size() < 1000){
        double bestC = -1;
        double bestP = -1;
        double bestScore = -1;
        find_parameters(&P, &C, 1, 4.0, -1, &bestC, &bestP, &bestScore);
        C.C = bestC;
    } else if(args.cost < 0) C.C = 8.0;

    M = train_linear(&P, &C);

    assert(M->nr_class <= 2);
    assert(M->nr_feature + (args.bias > 0 ? 1 : 0) == n);

    // Set base's attributes
    wSize = n;
    firstClass = M->label[0];
    classCount = M->nr_class;

    // Copy weights
    W = new Weight[n];
    for(int i = 0; i < n; ++i)
        W[i] = M->w[i];
    delete[] M->w;

    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC
                || args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    delete[] M->label;
    delete[] labels;
    delete[] labelsWeights;
    delete M;
}

void Base::trainOnline(std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args){
    setupOnlineTraining(args);

    const int examples = binFeatures.size() * args.epochs;
    for(int i = 0; i < examples; ++i){
        int r = i % binFeatures.size();
        unsafeUpdate(binLabels[r], binFeatures[r], args);
    }

    finalizeOnlineTraining();
}

void Base::train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args){

    if(binLabels.empty()){
        firstClass = 0;
        classCount = 0;
        return;
    }

    int positiveLabels = std::count(binLabels.begin(), binLabels.end(), 1.0);
    if(positiveLabels == 0 || positiveLabels == binLabels.size()){
        firstClass = static_cast<int>(binLabels[0]);
        classCount = 1;
        return;
    }

    assert(binLabels.size() == binFeatures.size());

    if (args.optimizerType == libliner) trainLiblinear(n, binLabels, binFeatures, positiveLabels, args);
    else trainOnline(binLabels, binFeatures, args);

    // Apply threshold and calculate number of non-zero weights
    pruneWeights(args.weightsThreshold);
    if(sparseSize() < denseSize()) toSparse();
}

double Base::predictValue(double* features){
    double val = 0;

    if(sparseW) val = dotVectors(sparseW, features); // Dense features dot sparse weights
    else throw std::runtime_error("Prediction using dense features and dense weights is not supported!");

    if(firstClass == 0) val *= -1;
    return val;
}

double Base::predictValue(Feature* features){
    double val = 0;

    if(mapW){ // Sparse features dot sparse weights in hash map
        Feature* f = features;
        while(f->index != -1) {
            auto w = mapW->find(f->index - 1);
            if(w != mapW->end()) val += w->second * f->value;
            ++f;
        }
    } else if (W) val = dotVectors(features, W); // Sparse features dot dense weights
    else throw std::runtime_error("Prediction using sparse features and sparse weights is not supported!");

    if(firstClass == 0) val *= -1;

    val /= pi; // FOBOS

    return val;
}

void Base::toMap(){
    if(mapW == nullptr){
        mapW = new UnorderedMap<int, Weight>();

        assert(W != nullptr);
        for (int i = 0; i < wSize; ++i)
            if (W[i] != 0) mapW->insert({i, W[i]});
        delete[] W;
        W = nullptr;
    }

    if(mapG == nullptr && G != nullptr){
        mapG = new UnorderedMap<int, Weight>();

        for(int i = 0; i < wSize; ++i)
            if(G[i] != 0) mapG->insert({i, W[i]});
        delete[] G;
        G = nullptr;
    }
}

void Base::toDense(){
    if(W == nullptr){
        W = new Weight[wSize];
        std::memset(W, 0, wSize * sizeof(Weight));

        assert(mapW != nullptr);
        for(const auto& w : *mapW) W[w.first] = w.second;
        delete mapW;
        mapW = nullptr;
    }

    if(G == nullptr && mapG != nullptr){
        G = new Weight[wSize];
        std::memset(G, 0, wSize * sizeof(Weight));

        for(const auto& w : *mapG) G[w.first] = w.second;
        delete mapG;
        mapG = nullptr;
    }
}

void Base::toSparse(){
    if(sparseW == nullptr){
        sparseW = new Feature[nonZeroW + 1];
        sparseW[nonZeroW].index = -1;

        if(W != nullptr) {
            Feature *f = sparseW;
            for (int i = 0; i < wSize; ++i) {
                if (W[i] != 0) {
                    f->index = i;
                    f->value = W[i];
                    ++f;
                }
            }

            delete[] W;
            W = nullptr;
            delete[] G;
            G = nullptr;
        } else if(mapW != nullptr) {
            Feature *f = sparseW;
            for(const auto& w : *mapW){
                if (w.second != 0) {
                    f->index = w.first;
                    f->value = w.second;
                    ++f;
                }
            }

            delete mapW;
            mapW = nullptr;
            delete mapG;
            mapG = nullptr;
        }
    }
}

void Base::setupOnlineTraining(Args &args){
    mapW = new UnorderedMap<int, Weight>();
    if(args.optimizerType == adagrad)
        mapG = new UnorderedMap<int, Weight>();
    classCount = 2;
    firstClass = 1;
}

void Base::finalizeOnlineTraining(){
    if(W) {
        for (int i = 0; i < wSize; ++i) {
            W[i] /= pi;
        }
    } else if(mapW){
        for (auto &w : *mapW) {
            w.second /= pi;
        }
    } else if(sparseW) {
        Feature *f = sparseW;
        while (f->index != -1) {
            f->value /= pi;
            ++f;
        }
    }
}

void Base::pruneWeights(double threshold){
    nonZeroW = 0;

    if(W) {
        for (int i = 0; i < wSize; ++i) {
            if (W[i] != 0 && fabs(W[i]) >= threshold) ++nonZeroW;
            else W[i] = 0;
        }
    } else if(mapW){
        for (auto &w : *mapW) {
            if (w.second != 0 && fabs(w.second) >= threshold) ++nonZeroW;
            else w.second = 0;
        }
    } else if(sparseW) {
        Feature *f = sparseW;
        while (f->index != -1) {
            if (f->value != 0 && fabs(f->value) >= threshold) ++nonZeroW;
            else f->value = 0;
            ++f;
        }
    }
}

void Base::save(std::ostream& out){
    out.write((char*) &classCount, sizeof(classCount));
    out.write((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        // Decide on optimal file coding
        //bool saveSparse = sparseSize() < denseSize();
        bool saveSparse = true;

        out.write((char*) &hingeLoss, sizeof(hingeLoss));
        out.write((char*) &wSize, sizeof(wSize));
        out.write((char*) &nonZeroW, sizeof(nonZeroW));
        out.write((char*) &saveSparse, sizeof(saveSparse));

        if(saveSparse){
            if(sparseW) {
                Feature *f = sparseW;
                while (f->index != -1) {
                    if(f->value != 0) {
                        out.write((char *) &f->index, sizeof(f->index));
                        out.write((char *) &f->value, sizeof(f->value));
                    }
                    ++f;
                }
            } else if(mapW) {
                for(const auto &f : *mapW){
                    if(f.second != 0) {
                        out.write((char *) &f.first, sizeof(f.first));
                        out.write((char *) &f.second, sizeof(f.second));
                    }
                }
            } else {
                for(int i = 0; i < wSize; ++i){
                    if(W[i] != 0){
                        out.write((char*) &i, sizeof(i));
                        out.write((char*) &W[i], sizeof(Weight));
                    }
                }
            }
        } else out.write((char*) W, wSize * sizeof(Weight));
    }

    //std::cerr << "  Saved base: sparse: " << saveSparse << ", classCount: " << classCount << ", firstClass: "
    //    << firstClass << ", weights: " << nonZeroCount << "/" << wSize << ", size: " << size()/1024 << "/" << denseSize()/1024 << "K\n";
}

void Base::load(std::istream& in) {
    in.read((char*) &classCount, sizeof(classCount));
    in.read((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        bool loadSparse;

        in.read((char*) &hingeLoss, sizeof(hingeLoss));
        in.read((char*) &wSize, sizeof(wSize));
        in.read((char*) &nonZeroW, sizeof(nonZeroW));
        in.read((char*) &loadSparse, sizeof(loadSparse));

        if(loadSparse){
            mapW = new UnorderedMap<int, Weight>();

            int index;
            Weight w;

            for (int i = 0; i < nonZeroW; ++i) {
                in.read((char*) &index, sizeof(index));
                in.read((char*) &w, sizeof(Weight));
                if (sparseW != nullptr){
                    sparseW[i].index = index;
                    sparseW[i].value = w;
                }
                if (mapW != nullptr) mapW->insert({index, w});
            }
        } else {
            W = new Weight[wSize];
            std::memset(W, 0, wSize * sizeof(Weight));
            in.read((char*) W, wSize * sizeof(Weight));
        }
    }

    //std::cerr << "  Loaded base: sparse: " << sparse << ", classCount: " << classCount << ", firstClass: " << firstClass << ", weights: "
    //    << nonZeroW << "/" << wSize << ", size: " << size()/1024 << "/" << denseSize()/1024 << "K\n";
}

size_t Base::size(){
    if(W) return denseSize();
    if(mapW) return mapSize();
    if(sparseW) return sparseSize();
    return 0;
}

void Base::printWeights(){
    if (W != nullptr)
        for(int i = 0; i < wSize; ++i) std::cerr << W[i] <<" ";
    else if (mapW != nullptr)
        for (const auto& f : *mapW)
            std::cerr << f.first << ":" << f.second << " ";
    else if (sparseW != nullptr) {
        Feature* f = sparseW;
        while(f->index != -1 && f->index < wSize) {
            std::cerr << f->index << ":" << f->value << " ";
            ++f;
        }
    } else std::cerr << "No weights";
    std::cerr << "\n";
}

void Base::multiplyWeights(double a){
    if (W != nullptr)
        for(int i = 0; i < wSize; ++i) W[i] *= a;
    else if (mapW != nullptr)
        for (auto& f : *mapW) f.second *= a;
    else if (sparseW != nullptr) {
        Feature* f = sparseW;
        while(f->index != -1 && f->index < wSize) {
            f->value *= a;
            ++f;
        }
    }
}

void Base::invertWeights(){
    multiplyWeights(-1);
}

Base* Base::copy(){
    Base* copy = new Base();
    if(W){
        copy->W = new Weight[wSize];
        std::memcmp(copy->W, W, wSize * sizeof(Weight));
    }
    if(G){
        copy->G = new Weight[wSize];
        std::memcmp(copy->G, G, wSize * sizeof(Weight));
    }

    if(mapW) copy->mapW = new UnorderedMap<int, Weight>(mapW->begin(), mapW->end());
    if(mapG) copy->mapG = new UnorderedMap<int, Weight>(mapG->begin(), mapG->end());

    if(sparseW) {
        copy->sparseW = new Feature[nonZeroW + 1];
        std::memcmp(copy->sparseW , sparseW, (nonZeroW + 1) * sizeof(Feature));
    }

    copy->firstClass = firstClass;
    copy->classCount = classCount;
    copy->wSize = wSize;
    copy->nonZeroW = nonZeroW;

    return copy;
}

Base* Base::copyInverted(){
    Base* c = copy();
    c->invertWeights();
    return c;
}

// Base utils
Base* trainBase(int n, std::vector<double>& baseLabels, std::vector<Feature*>& baseFeatures, Args& args){
    Base* base = new Base();
    base->train(n, baseLabels, baseFeatures, args);
    return base;
}

// TODO: Move this to model.cpp

void trainBases(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                std::vector<std::vector<Feature*>>& baseFeatures, Args& args){

    std::ofstream out(outfile);
    int size = baseLabels.size();
    out.write((char*) &size, sizeof(size));
    trainBases(out, n, baseLabels, baseFeatures, args);
    out.close();
}

void trainBases(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                std::vector<std::vector<Feature*>>& baseFeatures, Args& args){

    std::cerr << "Starting training base estimators in " << args.threads << " threads ...\n";

    assert(baseLabels.size() == baseFeatures.size());
    int size = baseLabels.size(); // This "batch" size

    // Run learning in parallel
    ThreadPool tPool(args.threads);
    std::vector<std::future<Base*>> results;

    for(int i = 0; i < size; ++i)
        results.emplace_back(tPool.enqueue(trainBase, n, baseLabels[i], baseFeatures[i], args));

    // Saving in the main thread
    for(int i = 0; i < results.size(); ++i) {
        printProgress(i, results.size());
        Base* base = results[i].get();
        base->save(out);
        delete base;
    }
}

void trainBasesWithSameFeatures(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                                std::vector<Feature*>& baseFeatures, Args& args){
    std::ofstream out(outfile);
    int size = baseLabels.size();
    out.write((char*) &size, sizeof(size));
    trainBasesWithSameFeatures(out, n, baseLabels, baseFeatures, args);
    out.close();
}

void trainBasesWithSameFeatures(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                                std::vector<Feature*>& baseFeatures, Args& args){

    std::cerr << "Starting training base estimators in " << args.threads << " threads ...\n";
    int size = baseLabels.size(); // This "batch" size

    // Run learning in parallel
    ThreadPool tPool(args.threads);
    std::vector<std::future<Base *>> results;

    for (int i = 0; i < size; ++i)
        results.emplace_back(tPool.enqueue(trainBase, n, baseLabels[i], baseFeatures, args));

    // Saving in the main thread
    for (int i = 0; i < results.size(); ++i) {
        printProgress(i, results.size());
        Base *base = results[i].get();
        base->save(out);
        delete base;
    }
}

std::vector<Base*> loadBases(std::string infile){
    std::cerr << "Loading base estimators ...\n";
    
    std::vector<Base*> bases;

    std::ifstream in(infile);
    int size;
    in.read((char*) &size, sizeof(size));
    bases.reserve(size);
    for(int i = 0; i < size; ++i) {
        printProgress(i, size);
        bases.emplace_back(new Base());
        bases.back()->load(in);
    }
    in.close();
    
    return bases;
}
