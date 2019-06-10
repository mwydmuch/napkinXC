/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "ubop_ch.h"
#include "set_utility.h"


UBOPCH::UBOPCH(){}

void UBOPCH::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    std::priority_queue<TreeNodeValue> nQueue;

    double value = bases[tree->root->index]->predictProbability(features);
    assert(value == 1);
    nQueue.push({tree->root, value});
    ++rCount;

    std::shared_ptr<SetUtility> u = setUtilityFactory(args, static_cast<Model*>(this));

    double P = 0, bestU = 0;
    while (!nQueue.empty()){
        predictNext(nQueue, prediction, features);

        P += prediction.back().value;
        double U = u->g(prediction.size()) * P;

        if(bestU < U)
            bestU = U;
        else {
            P -= prediction.back().value;
            prediction.pop_back();
            break;
        }
    }
}