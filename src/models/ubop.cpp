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

    args.topK = -1;
    OVR::predict(allPredictions, features, args);

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());

    double P = 0, bestU = 0;
    for(const auto& p : allPredictions){
        P += p.value;
        double U = u->g(prediction.size() + 1) * P;
        if(bestU <= U) {
            prediction.push_back(p);
            bestU = U;
        } else break;
    }

    //std::cerr << "pred size: " << prediction.size() << " P: " << P << " best U: " << bestU << " sum " <<  sum << "\n";
    //exit(0);
}
