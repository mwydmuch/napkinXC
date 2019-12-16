/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <list>
#include <vector>

#include "ovr.h"
#include "threads.h"


OVR::OVR() {
    type = ovr;
    name = "OVR";
}

std::vector<Prediction> OVR::predictForAllLabels(Feature* features, Args& args) {
    std::vector<Prediction> prediction;
    double sum = 0;

    for (int i = 0; i < bases.size(); ++i) {
        // double value = bases[i]->predictProbability(features); // Normalization
        double value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
        prediction.push_back({i, value});
    }

    for (auto& p : prediction) p.value /= sum;
    return prediction;
}

double OVR::predictForLabel(Label label, Feature* features, Args& args) {
    double sum = 0;
    for (int i = 0; i < bases.size(); ++i) {
        // double value = bases[i]->predictProbability(features); // Standard normalization
        double value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
    }

    return exp(bases[label]->predictValue(features)) / sum;
}
