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
#include <array>
#include <mutex>

#include "br_plt_neg.h"
#include "threads.h"

BRPLTNeg::BRPLTNeg(){
    plt = nullptr;
}

BRPLTNeg::~BRPLTNeg() {
    delete plt;
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

void BRPLTNeg::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output){
    std::cerr << "Training BR PLT model ...\n";

    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    std::string pltDir = joinPath(output, "plt");
    makeDir(pltDir);
    plt = new PLT();
    plt->train(labels, features, args, pltDir);
    plt->load(args, pltDir);

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(plt->outputSize());
    std::vector<std::vector<Feature*>> binFeatures(plt->outputSize());

    // Gather examples for each node
    if(args.threads > 1){
        std::cerr << "Assigning labels for base estimators in " << args.threads << " threads ...\n";

        std::array<std::mutex, LABELS_MUTEXES> mutexes;
        ThreadSet tSet;
        for (int t = 0; t < args.threads; ++t)
            tSet.add(assignDataPointsThread, std::ref(binLabels), std::ref(binFeatures), std::ref(labels), std::ref(features),
                     std::ref(args), plt, t, args.threads, std::ref(mutexes));
        tSet.joinAll();
    } else {
        std::cerr << "Assigning labels for base estimators ...\n";

        for(int r = 0; r < rows; ++r){
            printProgress(r, rows);

            std::unordered_set<int> lPositive;

            int rSize = labels.size(r);
            auto rLabels = labels.row(r);

            // Add true labels
            if (rSize > 0){
                for (int i = 0; i < rSize; ++i) {
                    lPositive.insert(rLabels[i]);
                    binLabels[rLabels[i]].push_back(1.0);
                    binFeatures[rLabels[i]].push_back(features.row(r));
                }
            }

            // Sample labels using PLT
            std::vector<Prediction> pltPrediction;
            plt->predictTopK(pltPrediction, features.row(r), args.sampleK);
            for (const auto& p : pltPrediction){
                if(!lPositive.count(p.label)) {
                    binLabels[p.label].push_back(0.0);
                    binFeatures[p.label].push_back(features.row(r));
                }
            }
        }
    }

    trainBases(joinPath(output, "weights.bin"), features.cols(), binLabels, binFeatures, args);
}

void assignDataPointsThread(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                            const SRMatrix<Label>& labels, const SRMatrix<Feature>& features, Args &args, PLT* plt,
                            int threadId, int threads, std::array<std::mutex, LABELS_MUTEXES>& mutexes){

    int rows = features.rows();
    int part = (rows / threads) + 1;
    int partStart = threadId * part;
    int partEnd = std::min((threadId + 1) * part, rows);

    for (int r = partStart; r < partEnd; ++r) {
        if(threadId == 0) printProgress(r, partEnd);

        std::unordered_set<int> lPositive;

        int rSize = labels.size(r);
        auto rLabels = labels.row(r);

        // Add true labels
        if (rSize > 0){
            for (int i = 0; i < rSize; ++i) {
                lPositive.insert(rLabels[i]);

                std::mutex &m = mutexes[rLabels[i] % mutexes.size()];
                m.lock();

                binLabels[rLabels[i]].push_back(1.0);
                binFeatures[rLabels[i]].push_back(features.row(r));

                m.unlock();
            }
        }

        // Sample labels using PLT
        std::vector<Prediction> pltPrediction;
        //plt->predictTopK(pltPrediction, features.row(r), args.sampleK);
        plt->predictTopKBeam(pltPrediction, features.row(r), args.sampleK);
        for (const auto& p : pltPrediction){
            if(!lPositive.count(p.label)) {
                std::mutex &m = mutexes[p.label % mutexes.size()];
                m.lock();

                binLabels[p.label].push_back(0.0);
                binFeatures[p.label].push_back(features.row(r));

                m.unlock();
            }
        }

        /*
        std::default_random_engine rng(args.seed);
        std::uniform_int_distribution<int> labelsRandomizer(0, plt->outputSize());
        // Sample randomly additional labels
        for(int i = 0; i < args.sampleK; ++i){
            int rl = labelsRandomizer(rng);
            std::mutex &m = mutexes[rl % mutexes.size()];
            m.lock();

            binLabels[rl].push_back(0.0);
            binFeatures[rl].push_back(features.row(r));

            m.unlock();
        }
         */
    }
}

void BRPLTNeg::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    plt->predictTopK(prediction, features, args.sampleK);
    for (auto& p : prediction)
        p.value = bases[p.label]->predictProbability(features);

    sort(prediction.rbegin(), prediction.rend());
    if(args.topK > 0) prediction.resize(args.topK);
}

double BRPLTNeg::predictForLabel(Label label, Feature* features, Args &args){
    return bases[label]->predictProbability(features);
}

void BRPLTNeg::load(Args &args, std::string infile){
    std::cerr << "Loading PLT Slice model ...\n";

    plt = new PLT();
    plt->load(args, joinPath(infile, "plt"));

    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();
}

