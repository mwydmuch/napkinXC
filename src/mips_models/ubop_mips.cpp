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


UBOPMIPS::UBOPMIPS(){
    type = ubopMips;
    name = "UBOP MIPS";
}

void UBOPMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){

    int k;
    if (args.setUtilityType == uP) k = 2;
    else k = std::ceil(static_cast<double>(bases.size()) / 10);

    std::vector<Prediction> allPredictions;
    std::unordered_set<int> seenLabels;
    double sum = 0;

    std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, k);
    while(!mipsPrediction.empty()) {
        auto p = mipsPrediction.top();
        p.value = exp(p.value);
        mipsPrediction.pop();
        allPredictions.push_back(p);
        sum += p.value;
        seenLabels.insert(p.label);
    }

    double tmpSum = 0;
    int sample = (bases.size() - k) / 10;
    std::default_random_engine rng(args.seed);
    std::uniform_int_distribution<int> labelsRandomizer(0, bases.size());
    for(int i = 0; i < sample; ++i) {
        int r = labelsRandomizer(rng);
        double value = exp(bases[r]->predictValue(features));
        tmpSum += value;

        if(!seenLabels.count(r))
            seenLabels.insert(r);
        else --i;
    }

    sum += tmpSum * static_cast<double>(bases.size() - k) / sample;

    // Normalize
    for(auto& p : allPredictions)
        p.value /= sum;

    // BOP part
    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());
    double P = 0, bestU = 0;
    for(int i = 0; i < allPredictions.size(); ++i){
        auto& p = allPredictions[i];
        P += p.value;
        double U = u->g(prediction.size() + 1) * P;
        if(bestU <= U) {
            prediction.push_back(p);
            bestU = U;
        } else break;

        if(i + 1 == allPredictions.size()){
            k *= 2;
            std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, k);
            int j = 0;
            while(!mipsPrediction.empty()) {
                auto p = mipsPrediction.top();
                mipsPrediction.pop();
                if(j > i){
                    p.value = exp(p.value) / sum;
                    allPredictions.push_back(p);
                }
            }
        }
    }

    //for(auto& p : allPredictions)
    //    std::cerr << p.label << " " << p.value << "\n";
    //std::cerr << "pred size: " << prediction.size() << " P: " << P << " best U: " << bestU << " sum " <<  sum << "\n";
    //exit(0);
}
