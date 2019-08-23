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

#include "br_plt_neg.h"


BRPLTNeg::BRPLTNeg(){
    plt = nullptr;
}

BRPLTNeg::~BRPLTNeg() {
    delete plt;
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

void BRPLTNeg::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output){
    std::cerr << "Training PLT Slice model ...\n";

    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    std::string pltDir = joinPath(output, "plt");
    makeDir(pltDir);
    plt = new PLT();
    plt->train(labels, features, args, pltDir);
    plt->load(args, pltDir);

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(plt->outputSize());
    std::vector<std::vector<Feature*>> binFeatures(plt->outputSize());

    std::cerr << "Assigning labels for base estimators ...\n";

    // Gather examples for each node
    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);

        std::unordered_set<int> lPositive;

        int rSize = labels.size(r);
        auto rLabels = labels.row(r);

        // Add true labels
        if (rSize > 0){
            for (int i = 0; i < rSize; ++i) {
                lPositive.insert(rLabels[i]);
                binLabels[rLabels[i]].push_back(1.0);
                binFeatures[rLabels[i]].push_back(features.row(r));
            }
        }

        // Predict additional labels
        std::vector<Prediction> pltPrediction;
        plt->predictTopK(pltPrediction, features.row(r), args.sampleK);
        for (const auto& p : pltPrediction){
            if(!lPositive.count(p.label)) {
                binLabels[p.label].push_back(0.0);
                binFeatures[p.label].push_back(features.row(r));
            }
        }
    }

    trainBases(joinPath(output, "weights.bin"), features.cols(), binLabels, binFeatures, args);
}

void BRPLTNeg::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    auto saveTopK = args.topK;
    plt->predictTopK(prediction, features, args.sampleK);
    for (auto& p : prediction){
        p.value = bases[p.label]->predictProbability(features);
    }
    args.topK = saveTopK;

    sort(prediction.rbegin(), prediction.rend());
    if(args.topK > 0) prediction.resize(args.topK);
}

double BRPLTNeg::predictForLabel(Label label, Feature* features, Args &args){
    return bases[label]->predictProbability(features);
}

void BRPLTNeg::load(Args &args, std::string infile){
    std::cerr << "Loading PLT Slice model ...\n";

    plt = new PLT();
    plt->load(args, joinPath(infile, "plt"));

    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();
}

