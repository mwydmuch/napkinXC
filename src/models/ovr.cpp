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

#include "ovr.h"
#include "threads.h"


OVR::OVR(){
    type = ovr;
    name = "OVR";
}

void OVR::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    double sum = 0;

    for(int i = 0; i < bases.size(); ++i) {
        //double value = bases[i]->predictProbability(features); // Normalization
        double value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
        prediction.push_back({i, value});
    }

    for(auto& p : prediction)
        p.value /= sum;

    sort(prediction.rbegin(), prediction.rend());
    resizePrediction(prediction, args);
}

double OVR::predictForLabel(Label label, Feature* features, Args &args){
    double sum = 0;
    for(int i = 0; i < bases.size(); ++i) {
        //double value = bases[i]->predictProbability(features); // Standard normalization
        double value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
    }

    return exp(bases[label]->predictValue(features)) / sum;
}
