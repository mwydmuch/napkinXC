/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <cmath>
#include <fstream>
#include <iostream>

#include "base.h"
#include "linear.h"


Base::Base(){
    sparse = false;
    hingeLoss = false;

    wSize = 0;
    nonZeroW = 0;
    classCount = 0;
    firstClass = 0;

    W = nullptr;
    mapW = nullptr;
    sparseW = nullptr;
}

Base::~Base(){
    if(W != nullptr) delete[] W;
    if(mapW != nullptr) delete mapW;
    if(sparseW != nullptr) delete[] sparseW;
}

void Base::train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args){

    if(binLabels.size() == 0){
        firstClass = 0;
        classCount = 0;
        return;
    }

    int positiveLabel = 0;
    for(auto l : binLabels)
        if(l == 1.0) ++positiveLabel;

    if(positiveLabel == 0 || positiveLabel == binLabels.size()){
        firstClass = binLabels[0];
        classCount = 1;
        return;
    }

    int labelsCount = 0;
    int* labels = NULL;
    double* labelsWeights = NULL;

    if(args.labelsWeights){
        labelsCount = 2;
        labels = new int[2];
        labels[0] = 0;
        labels[1] = 1;

        labelsWeights = new double[2];
        if(binLabels.size() - positiveLabel > positiveLabel){
            labelsWeights[0] = 1.0;
            labelsWeights[1] = 1.0 + log(static_cast<double>(binLabels.size() - positiveLabel) / positiveLabel);
        }
        else{
            labelsWeights[0] = 1.0 + log(static_cast<double>(positiveLabel) / (binLabels.size() - positiveLabel));
            labelsWeights[1] = 1.0;
        }
    }

    assert(binLabels.size() == binFeatures.size());
    problem P = {
        .l = static_cast<int>(binLabels.size()),
        .n = n,
        .y = binLabels.data(),
        .x = binFeatures.data(),
        .bias = (args.bias > 0 ? 1.0 : -1.0)
    };

    parameter C = {
        .solver_type = args.solverType,
        .eps = args.eps,
        .C = args.C,
        //.nr_weight = 0,
        //.weight_label = NULL,
        //.weight = NULL,
        .nr_weight = labelsCount,
        .weight_label = labels,
        .weight = labelsWeights,
        .p = 0,
        .init_sol = NULL
    };

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    model* M = train_linear(&P, &C);
    assert(M->nr_class <= 2);
    assert(M->nr_feature + (args.bias > 0 ? 1 : 0) == n);

    // Set base attributes
    sparse = false;
    wSize = n;
    firstClass = M->label[0];
    classCount = M->nr_class;
    W = M->w;
    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC
        || args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    delete[] M->label;
    if(labels != NULL) delete[] labels;
    if(labelsWeights != NULL) delete[] labelsWeights;
    delete M;

    // Apply threshold and calculate number of zero weights
    threshold(args.threshold);
    if(sparseSize() < denseSize()) toSparse();
}

double Base::predictValue(Feature* features){
    double val = 0;
    Feature* f = features;

    if(sparse){
        while(f->index != -1) {
            auto w = mapW->find(f->index - 1);
            if(w != mapW->end()) val += w->second * f->value;
            ++f;
        }
    } else {
        while(f->index != -1) {
            val += W[f->index - 1] * f->value;
            ++f;
        }
    }

    if(firstClass == 0) val *= -1;
    return val;
}

double Base::predictLoss(Feature* features){
    if(classCount < 2) return -static_cast<double>(firstClass);
    double val = predictValue(features);

    if(hingeLoss) val = std::pow(fmax(0, 1 - val), 2); // Hinge squared loss
    else val = log(1 + exp(-val)); // Log loss
    return val;
}

double Base::predictProbability(Feature* features){
    if(classCount < 2) return static_cast<double>(firstClass);
    double val = predictValue(features);
    val = 1.0 / (1.0 + exp(-val)); // Probability
    return val;
}

void Base::toMap(){
    if(mapW == nullptr){
        assert(W != nullptr);

        mapW = new std::unordered_map<int, double>();
        for(int i = 0; i < wSize; ++i)
            if(W[i] != 0) mapW->insert({i, W[i]});
        delete[] W;
        W = nullptr;
        sparse = true;
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
            while(f->index != -1) W[f->index] = f->value;
            delete[] sparseW;
            sparseW = nullptr;
        }

        sparse = false;
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
        sparse = true;
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

void Base::save(std::string outfile){
    std::ofstream out(outfile);
    save(out);
    out.close();
}

void Base::save(std::ostream& out){
    out.write((char*) &classCount, sizeof(classCount));
    out.write((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        // Decide on optimal file coding
        bool saveSparse = sparseSize() < denseSize();

        out.write((char*) &hingeLoss, sizeof(hingeLoss));
        out.write((char*) &wSize, sizeof(wSize));
        out.write((char*) &nonZeroW, sizeof(nonZeroW));
        out.write((char*) &saveSparse, sizeof(saveSparse));

        if(saveSparse){
            if(sparse){
                Feature* f = sparseW;
                while(f->index != -1) {
                    out.write((char*) &f->index, sizeof(f->index));
                    out.write((char*) &f->value, sizeof(f->value));
                    ++f;
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
    //    << firstClass << ", weights: " << nonZeroCount << "/" << wSize << ", size: " << sparseSize/1024 << "/" << denseSize/1024 << "K\n";
}

void Base::load(std::string infile, bool sparseWeights){
    std::ifstream in(infile);
    load(in, sparseWeights);
    in.close();
}

void Base::load(std::istream& in, bool sparseWeights) {
    in.read((char*) &classCount, sizeof(classCount));
    in.read((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        bool loadSparse;

        in.read((char*) &hingeLoss, sizeof(hingeLoss));
        in.read((char*) &wSize, sizeof(wSize));
        in.read((char*) &nonZeroW, sizeof(nonZeroW));
        in.read((char*) &loadSparse, sizeof(loadSparse));

        // Decide on weights coding
        sparse = sparseWeights && mapSize() < denseSize();

        if(sparse) mapW = new std::unordered_map<int, double>();
        else {
            W = new double[wSize];
            std::memset(W, 0, wSize * sizeof(double));
        }

        if(loadSparse){
            int index;
            double w;

            for (int i = 0; i < nonZeroW; ++i) {
                in.read((char*) &index, sizeof(index));
                in.read((char*) &w, sizeof(w));
                if (mapW != nullptr) mapW->insert({index, w});
                else W[index] = w;
            }
        } else {
            if (sparse) {
                double w;
                for (int i = 0; i < wSize; ++i) {
                    in.read((char*) &w, sizeof(w));
                    if (w != 0) mapW->insert({i, w});
                }
            } else in.read((char*) W, wSize * sizeof(double));
        }
    }
    //std::cerr << "  Loaded base: sparse: " << sparse << ", classCount: " << classCount << ", firstClass: " << firstClass << ", weights: "
    //    << nonZeroW << "/" << wSize << ", size: " << nonZeroW * (2 * sizeof(int) + sizeof(double))/1024 << "/" << wSize * sizeof(double)/1024 << "K\n";
}
