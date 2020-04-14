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

    std::cerr << "Adding points to MIPSIndex ...\n";
    size_t dim = bases[0]->getWSize();
    mipsIndex = new MIPSIndex(dim, args);
    for (int i = 0; i < m; ++i) {
        printProgress(i, m);
        if(!bases[i]->isDummy()) {
            bases[i]->toMap();
            if (!bases[i]->getFirstClass()) bases[i]->invertWeights();
            mipsIndex->addPoint(bases[i]->getMapW(), i);

            //bases[i]->toDense();
            //mipsIndex->addPoint(bases[i]->getW(), dim, i);
        }
    }

    mipsIndex->createIndex(args);
}
