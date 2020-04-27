/**
 * Copyright (c) 2019-2020 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <list>
#include <vector>

#include "set_utility.h"
#include "ubop_mips.h"


BRMIPS::BRMIPS() {
    type = brMips;
    name = "BR MIPS";
}

void BRMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {

    std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, args.topK);
    while (!mipsPrediction.empty()) {
        auto p = mipsPrediction.top();
        mipsPrediction.pop();
        prediction.push_back(p);
    }
}

void BRMIPS::load(Args& args, std::string infile) {
    std::cerr << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();

    size_t dim = 0;
    bool sparse = false;
    for (int i = 0; i < m; ++i) {
        if(bases[i]->getWSize() > dim)
            dim = bases[i]->getWSize();
        if(bases[i]->getMapW() != nullptr){
            sparse = true;
            break;
        }
    }

    mipsIndex = new MIPSIndex(dim, !args.mipsDense, args);
    std::cerr << "Adding " << m << " points with " << dim << " dims to MIPSIndex ...\n";
    for (int i = 0; i < m; ++i) {
        printProgress(i, m);
        if(!bases[i]->isDummy()) {
            if (!bases[i]->getFirstClass()) bases[i]->invertWeights();
            if(bases[i]->getMapW() != nullptr) mipsIndex->addPoint(bases[i]->getMapW(), i);
            else mipsIndex->addPoint(bases[i]->getW(), dim, i);
        }
    }

    mipsIndex->createIndex(args);
}
