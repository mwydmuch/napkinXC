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


UBOPMIPS::UBOPMIPS() {
    type = ubopMips;
    name = "UBOP MIPS";
}

void UBOPMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {

    int k;
    if(args.ubopMipsK < 1) k = std::ceil(static_cast<double>(bases.size()) * args.ubopMipsK);
    else k = args.ubopMipsK;

    std::vector<Prediction> allPredictions;
    UnorderedSet<int> seenLabels;
    std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, k);
    while (!mipsPrediction.empty()) {
        auto p = mipsPrediction.top();
        mipsPrediction.pop();
        p.value = exp(p.value);
        allPredictions.push_back(p);
        seenLabels.insert(p.label);
    }

    // BOP part
    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());
    double P = 0, bestU = 0;
    for (int i = 0; i < allPredictions.size(); ++i) {
        auto& p = allPredictions[i];
        P += p.value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU <= U) {
            prediction.push_back(p);
            bestU = U;
        } else
            break;

        if (i + 1 == allPredictions.size()) {
            k *= 2;
            std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, k);
            while (!mipsPrediction.empty()) {
                auto p = mipsPrediction.top();
                mipsPrediction.pop();

                if (!seenLabels.count(p.label)){
                    p.value = exp(p.value);
                    allPredictions.push_back(p);
                    seenLabels.insert(p.label);
                }
            }
        }
    }
}
