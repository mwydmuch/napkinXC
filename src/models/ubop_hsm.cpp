/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "ubop_hsm.h"
#include "set_utility.h"


UBOPHSM::UBOPHSM() {
    type = ubopHsm;
    name = "UBOP HSM";
}

void UBOPHSM::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    TopKQueue<TreeNodeValue> nQueue(0);

    double value = bases[tree->root->index]->predictProbability(features);
    assert(value == 1);
    nQueue.push({tree->root, value});
    ++dataPointCount;

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());

    double P = 0, bestU = 0;
    while (!nQueue.empty()) {
        auto p = predictNextLabel(nQueue, features, 0.0);
        P += p.value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU < U) {
            prediction.push_back(p);
            bestU = U;
        } else
            break;
    }
}