/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cassert>
#include <algorithm>
#include <vector>
#include <list>
#include <cmath>
#include <climits>

#include "ovr.h"
#include "threads.h"


OVR::OVR(){}

OVR::~OVR() {
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

void OVR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(lCols);

    std::cerr << "Assigning labels ...\n";

    // Gather examples for each node
    for(int i = 0; i < binLabels.size(); ++i)
        binLabels[i].reserve(rows);

    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);

        int rSize = labels.size(r);
        auto rLabels = labels.row(r);

        for(int i = 0; i < binLabels.size(); ++i)
            binLabels[i].push_back(0.0);

        if(rSize == 1)
            binLabels[rLabels[0]].back() = 1.0;
        else {
            if (rSize > 1)
                throw "OVR is multi-class classifier, encountered example with more then 1 label! Use BR instead.";
            else if (rSize < 1){
                std::cerr << "Example without label, skipping ...\n";
                continue;
            }
        }

    }

    std::cerr << "Starting training in " << args.threads << " threads ...\n";

    std::ofstream weightsOut(joinPath(args.output, "ovr_weights.bin"));
    weightsOut.write((char*) &lCols, sizeof(lCols));
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(int i = 0; i < binLabels.size(); ++i)
            results.emplace_back(tPool.enqueue(baseTrain, features.cols(), std::ref(binLabels[i]),
                                               std::ref(features.allRows()), std::ref(args)));

        // Saving in the main thread
        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            Base* base = results[i].get();
            base->save(weightsOut);
            delete base;
        }
    } else {
        for(int i = 0; i < binLabels.size(); ++i){
            printProgress(i, binLabels.size());
            Base base;
            base.train(features.cols(), binLabels[i], features.allRows(), args);
            base.save(weightsOut);
        }
    }
    weightsOut.close();
}

void OVR::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    double sum = 0;
    for(int i = 0; i < bases.size(); ++i) {
        double value = bases[i]->predictProbability(features);
        sum += value;
        prediction.push_back({i, value});
    }

    for(auto& p : prediction)
        p.value /= sum;

    sort(prediction.rbegin(), prediction.rend());
    if(args.topK > 0) prediction.resize(args.topK);
}

void OVR::load(std::string infile){
    std::cerr << "Loading OVR model ...\n";

    std::cerr << "Loading base classifiers ...\n";
    std::ifstream weightsIn(joinPath(infile, "ovr_weights.bin"));

    int size;
    weightsIn.read((char*) &size, sizeof(size));
    for(int i = 0; i < size; ++i) {
        printProgress(i, size);
        bases.emplace_back(new Base());
        bases.back()->load(weightsIn);
    }
    weightsIn.close();
}

