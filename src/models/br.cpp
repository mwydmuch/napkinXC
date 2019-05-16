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

void BR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(lCols);

    std::cerr << "Assigning labels for base estimators ...\n";

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

    trainBasesWithSameFeatures(joinPath(args.output, "br_weights.bin"), features.cols(), binLabels, features.allRows(), args);
}


void BR::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    for(int i = 0; i < bases.size(); ++i)
        prediction.push_back({i, bases[i]->predictProbability(features)});

    sort(prediction.rbegin(), prediction.rend());
    if(args.topK > 0) prediction.resize(args.topK);
}

void BR::load(std::string infile){
    std::cerr << "Loading BR model ...\n";
    bases = loadBases(joinPath(infile, "br_weights.bin"));
}

