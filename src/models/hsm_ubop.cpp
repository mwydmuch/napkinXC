/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "hsm_ubop.h"
#include "set_value.h"


HSMUBOP::HSMUBOP(){}

void HSMUBOP::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    std::priority_queue<TreeNodeValue> nQueue;

    double value = bases[tree->root->index]->predictProbability(features);
    assert(value == 1);
    nQueue.push({tree->root, value});


    std::shared_ptr<SetBasedU> u = setBasedUFactory(args);

    double P = 0, bestU = 0;
    while (!nQueue.empty()){
        predictNext(nQueue, prediction, features);

        P += prediction.back().value;
        double U = u->g(prediction.size(), tree->k) * P;

        //std::cerr << P << " " << U << " " << prediction.size() << "\n";

        if(bestU < U)
            bestU = U;
        else {
            P -= prediction.back().value;
            prediction.pop_back();
            if(u->checkstop(prediction.size(), tree->k)) break;
        }
    }
}