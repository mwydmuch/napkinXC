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

#include "ubop.h"
#include "set_utility.h"


UBOP::UBOP(){}

void UBOP::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    std::vector<Prediction> allPredictions;

    double sum = 0;
    for(int i = 0; i < bases.size(); ++i) {
        double value = bases[i]->predictProbability(features);
        sum += value;
        allPredictions.push_back({i, value});
    }

    for(auto& p : allPredictions)
        p.value /= sum;

    sort(allPredictions.rbegin(), allPredictions.rend());

    std::shared_ptr<SetUtility> u = setUtilityFactory(args, static_cast<Model*>(this));

    double P = 0, bestU = 0;
    for(const auto& p : allPredictions){
        prediction.push_back(p);

        P += p.value;
        double U = u->g(prediction.size()) * P;
        if(bestU <= U)
            bestU = U;
        else {
            P -= p.value;
            prediction.pop_back();
            //if(u->checkstop(prediction.size(), bases.size())) break;
            break;
        }
    }
}
