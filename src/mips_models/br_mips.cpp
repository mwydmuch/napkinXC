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

#include "set_utility.h"
#include "ubop_mips.h"


BRMIPS::BRMIPS(){}

void BRMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){

    std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, args.topK);
    while(!mipsPrediction.empty()) {
        auto p = mipsPrediction.top();
        mipsPrediction.pop();
        prediction.push_back(p);
    }
}

void BRMIPS::load(Args &args, std::string infile){
    std::cerr << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();

    std::cerr << "Adding points to MIPSIndex ...\n";
    size_t dim = bases[0]->size();
    mipsIndex = new MIPSIndex(dim, args);
    for(int i = 0; i < m; ++i){
        printProgress(i, m);
        bases[i]->toMap();
        mipsIndex->addPoint(bases[i]->getMapW(), i, bases[i]->getFirstClass() ? 1 : -1);
        //bases[i]->toDense();
        //mipsIndex->addPoint(bases[i]->getW(), bases[i]->size(), i);
    }

    mipsIndex->createIndex(args);
}
