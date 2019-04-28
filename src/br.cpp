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

#include "br.h"
#include "threads.h"


BR::BR(){}

BR::~BR() {
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

Base* binaryTrainThread(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args& args){
    Base* base = new Base();
    base->train(n, binLabels, binFeatures, args);
    return base;
}

void BR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    std::cerr << "Training tree ...\n";

    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(lCols);
    //std::vector<std::vector<Feature*>> binFeatures(tree->t);

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

        if (rSize > 0) {
            for (int i = 0; i < rSize; ++i) {
                binLabels[rLabels[i]].back() = 1.0;
            }
        }
    }

    std::cerr << "Starting training in " << args.threads << " threads ...\n";

    std::ofstream weightsOut(joinPath(args.output, "br_weights.bin"));
    weightsOut.write((char*) &lCols, sizeof(lCols));
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(int i = 0; i < binLabels.size(); ++i)
            results.emplace_back(tPool.enqueue(binaryTrainThread, features.cols(), std::ref(binLabels[i]),
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

    std::cerr << "All done\n";
}


void BR::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    for(int i = 0; i < bases.size(); ++i)
        prediction.push_back({i, bases[i]->predictProbability(features)});

    sort(prediction.rbegin(), prediction.rend());
    prediction.resize(args.topK);
}

void BR::load(std::string infile){
    std::cerr << "Loading BR model ...\n";

    std::cerr << "Loading base classifiers ...\n";
    std::ifstream weightsIn(joinPath(infile, "br_weights.bin"));

    int size;
    weightsIn.read((char*) &size, sizeof(size));
    for(int i = 0; i < size; ++i) {
        printProgress(i, size);
        bases.emplace_back(new Base());
        bases.back()->load(weightsIn);
    }
    weightsIn.close();
}

