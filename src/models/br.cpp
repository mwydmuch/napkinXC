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

void BR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output){
    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    std::ofstream out(joinPath(output, "weights.bin"));
    int size = lCols;
    out.write((char*) &size, sizeof(size));

    int parts = 1;
    int range = lCols / parts + 1;

    std::vector<std::vector<double>> binLabels(range);
    for(int i = 0; i < binLabels.size(); ++i)
        binLabels[i].reserve(rows);

    for(int p = 0; p < parts; ++p){

        if(parts > 1)
            std::cerr << "Assigning labels for base estimators (" << p + 1 << "/" << parts << ") ...\n";
        else
            std::cerr << "Assigning labels for base estimators ...\n";

        int rStart = p * range;
        int rStop = (p + 1) * range;

        for(int r = 0; r < rows; ++r){
            printProgress(r, rows);

            int rSize = labels.size(r);
            auto rLabels = labels.row(r);

            //checkRow(rLabels, features.row(r));

            for(int i = 0; i < binLabels.size(); ++i)
                binLabels[i].push_back(0.0);

            for (int i = 0; i < rSize; ++i)
                if(rSize == 1 && rLabels[0] >= rStart && rLabels[0] < rStop)
                    binLabels[rLabels[0] - rStart].back() = 1.0;
        }

        trainBasesWithSameFeatures(out, features.cols(), binLabels, features.allRows(), args);

        for(int i = 0; i < binLabels.size(); ++i)
            binLabels[i].clear();
    }

    out.close();
}

void BR::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    for(int i = 0; i < bases.size(); ++i)
        prediction.push_back({i, bases[i]->predictProbability(features)});

    if(args.topK > 0){
        sort(prediction.rbegin(), prediction.rend());
        prediction.resize(args.topK);
    }
}

double BR::predictForLabel(Label label, Feature* features, Args &args){
    return bases[label]->predictProbability(features);
}

void BR::load(Args &args, std::string infile){
    std::cerr << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();
}

