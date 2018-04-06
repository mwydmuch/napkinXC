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
    wSize = 0;
    classCount = 0;
    firstClass = 0;

    W = nullptr;
    sparseW = nullptr;
}

Base::~Base(){
    if(W != nullptr) delete[] W;
    if(sparseW != nullptr) delete sparseW;
}

void Base::train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args){

    assert(binLabels.size() == binFeatures.size());
    problem P = {
        .l = static_cast<int>(binLabels.size()),
        .n = n,
        .y = binLabels.data(),
        .x = binFeatures.data(),
        .bias = (args.bias ? 1.0 : 0.0)
    };

    parameter C = {
        .solver_type = args.solverType,
        .eps = args.eps,
        .C = 1,
        .nr_weight = 0,
        .weight_label = NULL,
        .weight = NULL,
        .p = 0.1,
        .init_sol = NULL
    };

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    model* M = train_linear(&P, &C);
    assert(M->nr_class <= 2);
    assert(M->nr_feature + 1 == P.n);

    wSize = M->nr_feature + M->bias;
    firstClass = M->label[0];
    classCount = M->nr_class;
    W = M->w;

    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC
        || args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    delete[] M->label;
    delete M;
}

double Base::predictValue(Feature* features){
    double val = 0;
    Feature* f = features;

    if(sparse){
        assert(sparseW != nullptr);
        while(f->index != -1) {
            auto w = sparseW->find(f->index - 1);
            if(w != sparseW->end()) val += w->second * f->value;
            ++f;
        }
    }
    else {
        assert(W != nullptr);
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

void Base::toSparse(){
    if(!sparse){
        sparseW = new std::unordered_map<int, double>();
        for(int i = 0; i < wSize; ++i)
            if(W[i] != 0) sparseW->insert({i, W[i]});
        delete[] W;
        sparse = true;
    }
}

void Base::toDense(){
    if(sparse){
        W = new double[wSize];
        std::memset(W, 0, wSize * sizeof(double));
        for(auto w : *sparseW)
            W[w.first] = w.second;
        delete sparseW;
        sparse = false;
    }
}

void Base::save(std::string outfile, Args& args){
    std::ofstream out(outfile);
    save(out, args);
    out.close();
}

void Base::save(std::ostream& out, Args& args){
    out.write((char*) &classCount, sizeof(classCount));
    out.write((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        assert(W != nullptr);

        // Decide on optimal file coding
        int denseSize = wSize * sizeof(double), nonZeroCount = 0;
        for (int i = 0; i < wSize; ++i)
            if(W[i] != 0 && fabs(W[i]) >= args.threshold) ++nonZeroCount;

        int sparseSize = nonZeroCount * (sizeof(int) + sizeof(double));
        bool saveSparse = sparseSize < denseSize;

        out.write((char*) &hingeLoss, sizeof(hingeLoss));
        out.write((char*) &wSize, sizeof(wSize));
        out.write((char*) &nonZeroCount, sizeof(nonZeroCount));
        out.write((char*) &saveSparse, sizeof(saveSparse));

        if(saveSparse){
            for(int i = 0; i < wSize; ++i){
                if(W[i] != 0 && fabs(W[i]) >= args.threshold){
                    out.write((char*) &i, sizeof(i));
                    out.write((char*) &W[i], sizeof(double));
                }
            }
        } else out.write((char*) W, wSize * sizeof(double));

        //std::cerr << "  Saved base: sparse: " << saveSparse << ", classCount: " << classCount << ", firstClass: "
        //    << firstClass << ", weights: " << nonZeroCount << "/" << wSize << ", size: " << sparseSize/1024 << "/" << denseSize/1024 << "K\n";
    }
    //else std::cerr << "  Saved base: classCount: " << classCount << ", firstClass: " << firstClass << "\n";
}

void Base::load(std::string infile, Args& args){
    std::ifstream in(infile);
    load(in, args);
    in.close();
}

void Base::load(std::istream& in, Args& args) {
    in.read((char*) &classCount, sizeof(classCount));
    in.read((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        bool loadSparse;
        int nonZeroCount;

        in.read((char*) &hingeLoss, sizeof(hingeLoss));
        in.read((char*) &wSize, sizeof(wSize));
        in.read((char*) &nonZeroCount, sizeof(nonZeroCount));
        in.read((char*) &loadSparse, sizeof(loadSparse));

        // Decide on weights coding
        if(args.sparseWeights){
            int denseSize = wSize * sizeof(double);
            // Unordered map stores elements inside buckets in the list structure, memory used for buckets is omitted here
            int sparseSize = nonZeroCount * (2 * sizeof(int) + sizeof(double));
            sparse = sparseSize < denseSize;
        }
        else sparse = false;

        if(!sparse){
            W = new double[wSize];
            std::memset(W, 0, wSize * sizeof(double));
        } else sparseW = new std::unordered_map<int, double>();

        if(loadSparse){
            int index;
            double w;

            for (int i = 0; i < nonZeroCount; ++i) {
                in.read((char*) &index, sizeof(index));
                in.read((char*) &w, sizeof(w));
                if (sparse) sparseW->insert({index, w});
                else W[index] = w;
            }
        } else {
            if (sparse) {
                double w;
                for (int i = 0; i < wSize; ++i) {
                    in.read((char*) &w, sizeof(w));
                    if (w != 0) sparseW->insert({i, w});
                }
            } else in.read((char*) W, wSize * sizeof(double));
        }

        //std::cerr << "  Loaded base: sparse: " << sparse << ", classCount: " << classCount << ", firstClass: " << firstClass << ", weights: "
        //    << nonZeroCount << "/" << wSize << ", size: " << nonZeroCount * (2 * sizeof(int) + sizeof(double))/1024 << "/" << wSize * sizeof(double)/1024 << "K\n";
    }
    //else std::cerr << "  Loaded base: classCount: " << classCount << ", firstClass: " << firstClass << "\n";
}
