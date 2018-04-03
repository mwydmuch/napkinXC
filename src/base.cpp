/**
 * Copyright (c) 2018 by Marek Wydmuch
 * All rights reserved.
 */

#include <cmath>

#include "base.h"
#include "linear.h"

#include <iostream>
Base::Base(){
    sparse = false;
    wSize = 0;
    classCount = 0;
    firstClass = 0;
    M = nullptr;

    useLinearPredict = false;
}

Base::~Base(){
    if(M) {
        free_model_content(M);
        delete M;
    }
}

void Base::train(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args &args){

    assert(binLabels.size() == binFeatures.size());
    problem P = {
        .l = static_cast<int>(binLabels.size()),
        .n = n - 1, // -1, because features are +1
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
        .p = 0,
        .init_sol = NULL
    };

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    M = train_linear(&P, &C);
    assert(M->nr_class <= 2);
    assert(M->nr_feature + 1 == P.n);

    wSize = M->nr_feature + M->bias;
    firstClass = M->label[0];
    classCount = M->nr_class;
}

double Base::predict(Feature* features){
    //std::cerr << "Predicting base ...\n";

    if(useLinearPredict){
        double p[2];
        predict_probability(M, features, reinterpret_cast<double*>(p));
        if(M->label[0] == 0) return 1.0 - p[0];
        else return p[0];
    }

    if(classCount == 1)
        return static_cast<double>(firstClass);

    double p = 0;
    Feature* f = features;

    if(sparse){
        //Feature* w = sparseW.data();
        while(f->index != -1) {
            //while(w->index < f->index - 1) ++w;
            //if(w->index == f->index - 1) p += w->value * f->value;
            auto w = sparseW.find(f->index - 1);
            if(w != sparseW.end()) p += w->second * f->value;
            ++f;
        }
    }
    else {
        while(f->index != -1) {
            p += W[f->index - 1] * f->value;
            ++f;
        }
    }

    if(firstClass == 1) return 1.0 / (1.0 + exp(-p));
    else return 1.0 - (1.0 / (1.0 + exp(-p)));
}

void Base::save(std::string outfile){
    if(useLinearPredict){
        assert(M != nullptr);
        save_model(outfile.c_str(), M);
        return;
    }

    std::ofstream out(outfile);
    save(out);
    out.close();
}

void Base::save(std::ostream& out){
    out.write((char*) &classCount, sizeof(classCount));
    out.write((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        assert(M != nullptr);

        // Decide on optimal file coding
        int denseSize = wSize * sizeof(double), nonZeroCount = 0;
        for (int i = 0; i < wSize; ++i)
            if(M->w[i] != 0) ++nonZeroCount;

        int sparseSize = nonZeroCount * (sizeof(int) + sizeof(double));
        bool saveSparse = sparseSize < denseSize;

        out.write((char*) &wSize, sizeof(wSize));
        out.write((char*) &nonZeroCount, sizeof(nonZeroCount));
        out.write((char*) &saveSparse, sizeof(saveSparse));

        if(saveSparse){
            for(int i = 0; i < wSize; ++i){
                if(M->w[i] != 0){
                    out.write((char*) &i, sizeof(i));
                    out.write((char*) &M->w[i], sizeof(double));
                }
            }
        } else out.write((char*) M->w, wSize * sizeof(double));
    }

    //std::cerr << "Saved base: classCount: " << classCount << ", firstClass: " << firstClass << ", wSize: " << wSize << "\n";
}

void Base::load(std::string infile, CodingType coding){
    if(useLinearPredict){
        M = load_model(infile.c_str());
        assert(M->nr_class <= 2);
        return;
    }

    std::ifstream in(infile);
    load(in, coding);
    in.close();
}

void Base::load(std::istream& in, CodingType coding) {

    in.read((char*) &classCount, sizeof(classCount));
    in.read((char*) &firstClass, sizeof(firstClass));

    if(classCount > 1) {
        bool loadSparse;
        int nonZeroCount;

        in.read((char*) &wSize, sizeof(wSize));
        in.read((char*) &nonZeroCount, sizeof(nonZeroCount));
        in.read((char*) &loadSparse, sizeof(loadSparse));

        // Decide on weights coding
        if(coding == spaceOptimal){
            int denseSize = wSize * sizeof(double);
            // Unordered map stores elements inside buckets in the list structure, memory used for buckets omitted
            int sparseSize = nonZeroCount * (2 * sizeof(int) + sizeof(double));
            sparse = sparseSize < denseSize;
        }
        else if(coding == dense) sparse = false;
        else sparse = true;

        W.clear();
        if(!sparse) W = std::vector<double>(wSize);
        sparseW = std::unordered_map<int, double>();

        if(loadSparse){
            int index;
            double w;

            for (int i = 0; i < nonZeroCount; ++i) {
                in.read((char*) &index, sizeof(index));
                in.read((char*) &w, sizeof(w));
                if (sparse)
                    //sparseW.push_back({index, w});
                    sparseW.insert({index, w});
                else W[index] = w;
            }
        } else {
            if (sparse) {
                double w;
                for (int i = 0; i < wSize; ++i) {
                    in.read((char*) &w, sizeof(w));
                    //if (w != 0) sparseW.push_back({i, w});
                    if (w != 0) sparseW.insert({i, w});
                }
            } else in.read((char*) W.data(), wSize * sizeof(double));
        }
    }

    // std::cerr << "Loaded base: sparse: " << sparse << ", classCount: " << classCount << ", firstClass: " << firstClass << ", wSize: " << wSize << "\n";
}
