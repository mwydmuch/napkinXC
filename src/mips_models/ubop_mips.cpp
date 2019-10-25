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


UBOPMIPS::UBOPMIPS(){}

void UBOPMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){

    bool sampling = false;

    int k = 5;
    std::vector<Prediction> allPredictions;
    std::unordered_set<int> seenLabels;
    double sum = 0;

    std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, k);
    while(!mipsPrediction.empty()) {
        auto p = mipsPrediction.top();
        mipsPrediction.pop();
        allPredictions.push_back(p);
        sum += p.value;
        seenLabels.insert(p.label);
    }

    if(sampling){
        // Variant with additional labels sampling
        double tmpSum = 0;
        int sample = (bases.size() - k) / 10;
        std::default_random_engine rng(args.seed);
        std::uniform_int_distribution<int> labelsRandomizer(0, bases.size());
        for(int i = 0; i < sample; ++i) {
            int r = labelsRandomizer(rng);
            double prob = bases[r]->predictProbability(features);
            tmpSum += prob;

            if(!seenLabels.count(r)){
                allPredictions.push_back({r, prob});
                seenLabels.insert(r);
            }
        }

        sum += tmpSum * static_cast<double>(bases.size() - k) / sample;

        // Sort
        sort(allPredictions.rbegin(), allPredictions.rend());
    }

    // Normalize
    for(auto& p : allPredictions)
        p.value /= sum;

    // BOP part
    std::shared_ptr<SetUtility> u = setUtilityFactory(args, static_cast<Model*>(this));
    double P = 0, bestU = 0;
    for(const auto& p : allPredictions){

        P += p.value;
        double U = u->g(prediction.size()) * P;
        if(bestU <= U) {
            prediction.push_back(p);
            bestU = U;
        } else break;
    }

    //for(auto& p : allPredictions)
    //    std::cerr << p.label << " " << p.value << "\n";
    //std::cerr << "pred size: " << prediction.size() << " P: " << P << " best U: " << bestU << " sum " <<  sum << "\n";
    //exit(0);
}

void UBOPMIPS::load(Args &args, std::string infile){
    std::cerr << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();

    std::cerr << "Adding points to MIPSIndex ...\n";
    size_t dim = bases[0]->size();
    mipsIndex = new MIPSIndex(dim, args);
    for(int i = 0; i < m; ++i){
        printProgress(i, m);
        bases[i]->toMap();
        mipsIndex->addPoint(bases[i]->getMapW(), i, bases[i]->getFirstClass() ? 1 : -1);
        //bases[i]->toDense();
        //mipsIndex->addPoint(bases[i]->getW(), bases[i]->size(), i);
    }

    mipsIndex->createIndex(args);
}
