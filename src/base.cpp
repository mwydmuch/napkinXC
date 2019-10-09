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


Base::Base(){
    hingeLoss = false;

    wSize = 0;
    nonZeroW = 0;
    classCount = 0;
    firstClass = 0;
    t = 0;

    W = nullptr;
    mapW = nullptr;
    sparseW = nullptr;
}

Base::~Base(){
    delete[] W;
    delete mapW;
    delete[] sparseW;
}

void Base::update(double label, Feature* features, Args &args){
    if(mapW == nullptr)
        mapW = new std::unordered_map<int, double>();

    double pred = predictValue(features);
    double a = args.eta * sqrt(1.0 / (double)(++t));
    double grad = a * label / (1.0 + exp(label * pred));

    Feature* f = features;
    while(f->index != -1) {
        (*mapW)[f->index - 1] += grad * f->value;
        ++f;
    }
}

void Base::train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args){

    if(binLabels.empty()){
        firstClass = 0;
        classCount = 0;
        return;
    }

    int positiveLabel = 0;
    for(auto l : binLabels)
        if(l == 1.0) ++positiveLabel;

    if(positiveLabel == 0 || positiveLabel == binLabels.size()){
        firstClass = static_cast<int>(binLabels[0]);
        classCount = 1;
        return;
    }

    int labelsCount = 0;
    int* labels = NULL;
    double* labelsWeights = NULL;
    int negativeLabel = static_cast<int>(binLabels.size()) - positiveLabel;

    if(args.labelsWeights){
        labelsCount = 2;
        labels = new int[2];
        labels[0] = 0;
        labels[1] = 1;

        labelsWeights = new double[2];
        if(negativeLabel > positiveLabel){
            labelsWeights[0] = 1.0;
            labelsWeights[1] = 1.0 + log(static_cast<double>(negativeLabel) / positiveLabel);
        } else{
            labelsWeights[0] = 1.0 + log(static_cast<double>(positiveLabel) / negativeLabel);
            labelsWeights[1] = 1.0;
        }
    }

    auto y = binLabels.data();
    auto x = binFeatures.data();
    int l = static_cast<int>(binLabels.size());

    assert(binLabels.size() == binFeatures.size());
    problem P = {
        .l = l,
        .n = n,
        .y = y,
        .x = x,
        .bias = (args.bias > 0 ? 1.0 : -1.0)
    };

    model *M = nullptr;
    if (args.optimizerType == libliner) {
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

        if(args.cost < 0 && binLabels.size() > 100 && binLabels.size() < 1000){
            double bestC = -1;
            double bestP = -1;
            double bestScore = -1;
            find_parameters(&P, &C, 1, 4.0, -1, &bestC, &bestP, &bestScore);
            C.C = bestC;
        } else if(args.cost < 0) C.C = 8.0;

        M = train_linear(&P, &C);

    } else if (args.optimizerType == sgd) {
        online_parameter OC = {
                .iter = args.iter,
                .eta = args.eta,
                .nr_weight = labelsCount,
                .weight_label = labels,
                .weight = labelsWeights,
                .p = 0,
                .init_sol = NULL
        };

        M = train_online(&P, &OC);
    }

    assert(M->nr_class <= 2);
    assert(M->nr_feature + (args.bias > 0 ? 1 : 0) == n);

    // Set base's attributes
    wSize = n;
    firstClass = M->label[0];
    classCount = M->nr_class;
    W = M->w;
    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC
                || args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    delete[] M->label;
    delete[] labels;
    delete[] labelsWeights;
    delete M;

    // Apply threshold and calculate number of non-zero weights
    threshold(args.weightsThreshold);
    if(sparseSize() < denseSize()) toSparse();
}

double Base::predictValue(double* features){
    double val = 0;

    if(sparseW) val = dotVectors(sparseW, features); // Dense features dot sparse weights
    else throw "Prediction using dense features require sparse weights!\n";

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
    else throw "Prediction using sparse features and sparse weights is not supported!\n";

    if(firstClass == 0) val *= -1;
    return val;
}

void Base::toMap(){
    if(mapW == nullptr){
        mapW = new std::unordered_map<int, double>();

        if(W != nullptr){
            for(int i = 0; i < wSize; ++i)
                if(W[i] != 0) mapW->insert({i, W[i]});
            delete[] W;
            W = nullptr;
        } else if(sparseW != nullptr){
            Feature* f = sparseW;
            while(f->index != -1){
                mapW->insert({f->index, f->value});
                ++f;
            }
            delete[] sparseW;
            sparseW = nullptr;
        }
    }
}

void Base::toDense(){
    if(W == nullptr){
        W = new double[wSize];
        std::memset(W, 0, wSize * sizeof(double));

        if(mapW != nullptr){
            for(const auto& w : *mapW) W[w.first] = w.second;
            delete mapW;
            mapW = nullptr;
        } else if(sparseW != nullptr){
            Feature* f = sparseW;
            while(f->index != -1){
                W[f->index] = f->value;
                ++f;
            }
            delete[] sparseW;
            sparseW = nullptr;
        }
    }
}

void Base::toSparse(){
    if(sparseW == nullptr){
        assert(W != nullptr);

        sparseW = new Feature[nonZeroW + 1];
        sparseW[nonZeroW].index = -1;
        Feature* f = sparseW;
        for(int i = 0; i < wSize; ++i){
            if(W[i] != 0){
                f->index = i;
                f->value = W[i];
                ++f;
            }
        }

        delete[] W;
        W = nullptr;
    }
}

void Base::threshold(double threshold){
    assert(W != nullptr);
    nonZeroW = 0;
    for (int i = 0; i < wSize; ++i){
        if(W[i] != 0 && fabs(W[i]) >= threshold) ++nonZeroW;
        else W[i] = 0;
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
                    out.write((char *) &f->index, sizeof(f->index));
                    out.write((char *) &f->value, sizeof(f->value));
                    ++f;
                }
            } else if(mapW) {
                for(const auto &f : *mapW){
                    out.write((char *) &f.first, sizeof(f.first));
                    out.write((char *) &f.second, sizeof(f.second));
                }
            } else {
                for(int i = 0; i < wSize; ++i){
                    if(W[i] != 0){
                        out.write((char*) &i, sizeof(i));
                        out.write((char*) &W[i], sizeof(double));
                    }
                }
            }
        } else out.write((char*) W, wSize * sizeof(double));
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
            mapW = new std::unordered_map<int, double>();
            //sparseW = new Feature[nonZeroW + 1];
            //sparseW[nonZeroW].index = -1;
            int index;
            double w;

            for (int i = 0; i < nonZeroW; ++i) {
                in.read((char*) &index, sizeof(index));
                in.read((char*) &w, sizeof(w));
                if (sparseW != nullptr){
                    sparseW[i].index = index;
                    sparseW[i].value = w;
                }
                if (mapW != nullptr) mapW->insert({index, w});
            }
        } else {
            W = new double[wSize];
            std::memset(W, 0, wSize * sizeof(double));
            in.read((char*) W, wSize * sizeof(double));
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

float* Base::toDenseFloat(){
    auto* fW = new float[wSize];
    std::memset(fW, 0, wSize * sizeof(float));
    if(W != nullptr) {
        for(int i = 0; i < wSize; ++i) fW[i] = W[i];
    } else if(mapW != nullptr){
        for(const auto& w : *mapW) fW[w.first] = w.second;
    } else if(sparseW != nullptr){
        Feature* f = sparseW;
        while(f->index != -1){
            fW[f->index] = f->value;
            ++f;
        }
        delete[] sparseW;
        sparseW = nullptr;
    }
    unitNorm(fW, wSize);
    return fW;
}


// Base utils

Base* trainBase(int n, std::vector<double>& baseLabels, std::vector<Feature*>& baseFeatures, Args& args){
    Base* base = new Base();
    //printVector(baseLabels);
    //printVector(baseFeatures);
    base->train(n, baseLabels, baseFeatures, args);
    return base;
}

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
    if(args.threads > 1){
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
    } else {
        // Run training in the main thread
        for(int i = 0; i < size; ++i){
            printProgress(i, size);
            Base base;
            base.train(n, baseLabels[i], baseFeatures[i], args);
            base.save(out);
        }
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
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(int i = 0; i < size; ++i)
            results.emplace_back(tPool.enqueue(trainBase, n, baseLabels[i], baseFeatures, args));

        // Saving in the main thread
        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            Base* base = results[i].get();
            base->save(out);
            delete base;
        }
    } else {
        // Run training in the main thread
        for(int i = 0; i < size; ++i){
            printProgress(i, size);
            Base base;
            base.train(n, baseLabels[i], baseFeatures, args);
            base.save(out);
        }
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
