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


void OVR::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    double sum = 0;

    for(int i = 0; i < bases.size(); ++i) {
        //double value = bases[i]->predictProbability(features); // Standard normalization
        double value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
        prediction.push_back({i, value});
    }

    for(auto& p : prediction)
        p.value /= sum;

    sort(prediction.rbegin(), prediction.rend());
    if(args.topK > 0){
        prediction.resize(args.topK);

    }
}

double OVR::predictForLabel(Label label, Feature* features, Args &args){
    double sum = 0;
    for(int i = 0; i < bases.size(); ++i) {
        double value = bases[i]->predictProbability(features);
        sum += value;
    }

    return bases[label]->predictProbability(features) / sum;
}

void OVR::printInfo(){
    std::cerr << "OVR additional stats:"
              << "\n  Mean # estimators per data point: " << bases.size()
              << "\n";
}

void OVR::checkRow(Label* labels, Feature* feature){
    int l = -1;
    while(labels[++l] > -1);
    if (l > 1) {
        //std::cerr << "Encountered example with more then 1 label! OVR is multi-class classifier, use BR instead!";
        //throw "OVR is multi-class classifier, encountered example with more then 1 label! Use BR instead.";
    }
    else if (l < 1){
        std::cerr << "Example without label, skipping ...\n";
    }
}


