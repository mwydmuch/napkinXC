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

#include "bloom.h"
#include "misc.h"
#include "threads.h"


Bloom::Bloom() {

}

Bloom::~Bloom() {
    for (auto b : bases) delete b;
}

void Bloom::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {
    int hashCount = args.hashes;
    bucketCount = args.buckets;

    Log(CERR) << "  Number of hashes: " << hashCount << ", number of buckets per hash: " << bucketCount << "\n";

    m = labels.cols();
    std::ofstream out(joinPath(output, "hashes.bin"));
    out.write((char*)&m, sizeof(m));
    out.write((char*)&bucketCount, sizeof(bucketCount));
    out.write((char*)&hashCount, sizeof(hashCount));

    // Generate hashes and save them to file
    long seed = args.getSeed();
    std::default_random_engine rng(seed);
    std::uniform_int_distribution dist(1, bucketCount);
    for(int i = 0; i < hashCount; ++i){
        unsigned int a = getFirstBiggerPrime(dist(rng));
        unsigned int b = getFirstBiggerPrime(dist(rng));
        unsigned int p = getFirstBiggerPrime(bucketCount + dist(rng));

        out.write((char*)&a, sizeof(a));
        out.write((char*)&b, sizeof(b));
        out.write((char*)&p, sizeof(p));

        hashes.emplace_back(a, b, p);
    }

    out.close();

    int size = bucketCount;

    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    std::vector<std::vector<double>> binLabels(size);
    for (int i = 0; i < binLabels.size(); ++i) binLabels[i].reserve(rows);

    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        int rSize = labels.size(r);
        auto rLabels = labels.row(r);

        for (int i = 0; i < binLabels.size(); ++i) binLabels[i].push_back(0.0);

        for (int i = 0; i < rSize; ++i) {
            for (int j = 0; j < hashes.size(); ++j)
                binLabels[baseForLabel(rLabels[i], j)].back() = 1.0;
        }
    }

    trainBasesWithSameFeatures(joinPath(output, "weights.bin"), features.cols(), binLabels, features.allRows(), nullptr,
                               args);
}

void Bloom::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    // Brute force prediction

    prediction.reserve(m);
    for (int i = 0; i < m; ++i)
        prediction.emplace_back(i, 0.0);

    for (int i = 0; i < bases.size(); ++i) {
        double value = bases[i]->predictProbability(features);
        for (const auto &l : baseToLabels[i])
            prediction[l].value += value;
    }

    std::nth_element(prediction.begin(), prediction.begin() + args.topK, prediction.end(), std::greater<Prediction>());
    prediction.resize(args.topK);
    prediction.shrink_to_fit();
    std::sort(prediction.begin(), prediction.end(), std::greater<Prediction>());

    /*
    std::vector<double> predictionScore(m, 0);
    std::vector<int> predictionCount(m, 0);

    std::vector<std::pair<double, int>> basesPred;
    basesPred.reserve(bases.size());
    for (int i = 0; i < bases.size(); ++i)
        basesPred.emplace_back(bases[i]->predictProbability(features), i);
    std::sort(basesPred.rbegin(), basesPred.rend());


    for (int i = 0; i < bases.size(); ++i){
        //std::cout << basesPred[i].first << " " << basesPred[i].second << "\n";
        for (const auto &l : baseToLabels[i]) {
            predictionScore[l] += basesPred[i].first;
            ++predictionCount[l];
            std::cout << l << " " << predictionScore[l] << " " << predictionCount[l] << "/" << hashes.size() << ", ";
            if(predictionCount[l] >= hashes.size())
                prediction.push_back({l, predictionScore[l]});
        }
        //std::cout << "\n" << prediction.size() << "\n";
        if(prediction.size() >= args.topK){
            std::sort(prediction.rbegin(), prediction.rend());
            prediction.resize(args.topK);
            break;
        }
    }
    */

    //exit(1);
}

double Bloom::predictForLabel(Label label, Feature* features, Args& args) {
    double prob = 1;
    for (int i = 0; i < hashes.size(); ++i)
        prob *= bases[baseForLabel(label, i)]->predictProbability(features);
    return prob;
}

void Bloom::load(Args& args, std::string infile) {
    Log(CERR) << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));

    Log(CERR) << "Loading hashes ...\n";
    std::ifstream in(joinPath(infile, "hashes.bin"));
    int hashCount;
    unsigned int a, b, p;
    in.read((char*)&m, sizeof(m));
    in.read((char*)&bucketCount, sizeof(bucketCount));
    in.read((char*)&hashCount, sizeof(hashCount));
    for(int i = 0; i < hashCount; ++i){
        in.read((char*)&a, sizeof(a));
        in.read((char*)&b, sizeof(b));
        in.read((char*)&p, sizeof(p));
        hashes.emplace_back(a, b, p);
    }
    in.close();

    // This is needed for fast brute force prediction
    baseToLabels.resize(bases.size());
    for(int i = 0; i < m; ++i)
        for (int j = 0; j < hashes.size(); ++j)
            baseToLabels[baseForLabel(i, j)].push_back(i);

}
