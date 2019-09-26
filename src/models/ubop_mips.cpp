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
#include "models/ubop_mips.h"


UBOPMIPS::UBOPMIPS(){}

void UBOPMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    float *denseFeatures = new float[dim];
    std::memset(denseFeatures, 0, dim * sizeof(float));

    Feature* f = features;
    while(f->index != -1){
        denseFeatures[f->index] = -f->value;
        ++f;
    }
    unitNorm(denseFeatures, dim);

    double sum = 0;
    std::unordered_set<int> seenLabels;
    std::vector<Prediction> allPredictions;

    int sample = bases.size() / 10;
    std::default_random_engine rng(args.seed);
    std::uniform_int_distribution<int> labelsRandomizer(0, bases.size());
    for(int i = 0; i < sample; ++i) {
        int r = labelsRandomizer(rng);
        double prob = bases[r]->predictProbability(features);
        sum += prob;

        if(!seenLabels.count(r)){
            allPredictions.push_back({r, prob});
            seenLabels.insert(r);
        }
    }
    sum *= static_cast<double>(bases.size()) / sample;

    std::priority_queue<std::pair<float, hnswlib::labeltype>> topPrediction = hnswIndex->mips(denseFeatures, 16);
    while(!topPrediction.empty()) {
        auto p = topPrediction.top();
        topPrediction.pop();

        double prob = bases[p.second]->predictProbability(features);

        //std::cerr << p.second << " " << p.first << "\n";

        if(!seenLabels.count(p.second)){
            allPredictions.push_back({static_cast<int>(p.second), prob});
            seenLabels.insert(p.second);
        }
    }

//    for(int i = 0; i < bases.size(); ++i) {
//        double value = bases[i]->predictProbability(features);
//        if(!seenLabels.count(i)){
//            allPredictions.push_back({i, value});
//            seenLabels.insert(i);
//        }
//    }

    for(auto& p : allPredictions)
        p.value /= sum;

    sort(allPredictions.rbegin(), allPredictions.rend());

//    for(auto& p : allPredictions)
//        std::cerr << p.label << " " << p.value << "\n";

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
            break;
        }
    }

    //std::cerr << "pred size: " << prediction.size() << " P: " << P << " best U: " << bestU << " sum " <<  sum << "\n";
    //exit(0);
}

void UBOPMIPS::load(Args &args, std::string infile){
    std::cerr << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();
    dim = bases[0]->featureSpaceSize();

    std::cerr << "Building MIPSIndex ...\n";
    mipsIndex = new MIPSIndex(dim, m);
    for(int i = 0; i < m; ++i){
        printProgress(i, m);
        float* fW = bases[i]->toDenseFloat();
        mipsIndex->addPoint(fW, i);
        delete[] fW;
    }
}
