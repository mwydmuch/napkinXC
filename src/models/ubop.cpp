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

#include "set_utility.h"
#include "ubop.h"


UBOP::UBOP() {
    type = ubop;
    name = "UBOP";
}

void UBOP::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    std::vector<Prediction> allPredictions;
    allPredictions = OVR::predictForAllLabels(features, args);
    sort(allPredictions.rbegin(), allPredictions.rend());

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());

    double P = 0, bestU = 0;
    for (const auto& p : allPredictions) {
        P += p.value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU <= U) {
            prediction.push_back(p);
            bestU = U;
        } else
            break;
    }

    //std::cerr << "  UBOP: pred. size: " << prediction.size() << " P: " << P << " best U: " << bestU << "\n";
}
