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

#include "mach.h"
#include "threads.h"


MACH::MACH() {

}

MACH::~MACH() {
    for (auto b : bases) delete b;
}

bool MACH::isPrime(int number){
    if(number % 2 == 0) return false;
    double numberSqrt = std::sqrt(static_cast<double>(number));
    for(int i = 3; i <= numberSqrt; i += 2)
        if(number % i == 0) return false;
    return true;
}

int MACH::getFirstBiggerPrime(int number){
    while(!isPrime(number)) ++number;
    return number;
}

void MACH::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {
    int hashCount = args.machHashes;
    bucketCount = args.machBuckets;

    Log(CERR) << "  Number of hashes: " << hashCount << ", number of buckets per hash: " << bucketCount << "\n";

    long seed = args.getSeed();
    std::default_random_engine rng(seed);

    m = labels.cols();

    // Generate hashes and save them to file
    std::ofstream out(joinPath(output, "graph.bin"));
    out.write((char*)&m, sizeof(m));
    out.write((char*)&bucketCount, sizeof(bucketCount));
    out.write((char*)&hashCount, sizeof(hashCount));
    
    for(int i = 0; i < hashCount; ++i){
        int a = getFirstBiggerPrime(rng() % m);
        int b = getFirstBiggerPrime(bucketCount + rng() % m);

        out.write((char*)&a, sizeof(a));
        out.write((char*)&b, sizeof(b));

        hashes.emplace_back(a, b);
    }

    out.close();

    int size = hashes.size() * bucketCount;

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

    // Train bases
    std::vector<ProblemData> binProblemData;
    std::vector<double> binWeights(features.rows(), 1);
    for(int i = 0; i < size; ++i) binProblemData.emplace_back(binLabels[i], features.allRows(), features.cols(), binWeights);
    trainBases(joinPath(output, "weights.bin"), binProblemData, args);
}

void MACH::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    // Brute force prediction
    prediction.reserve(m);
    for (int i = 0; i < m; ++i){
        prediction.push_back({i, 1.0});
    }

    for (int i = 0; i < bases.size(); ++i) {
        double value = bases[i]->predictProbability(features);
        for (const auto &l : baseToLabels[i]) {
            prediction[l].value += value;
        }
    }

    sort(prediction.rbegin(), prediction.rend());
    prediction.resize(args.topK);
    prediction.shrink_to_fit();

    // TODO: Faster prediction
    /*
    std::priority_queue<Prediction> nQueue;
    std::vector<double> basePredictions(bases.size());
    for (int i = 0; i < bases.size(); ++i)
        basePredictions[i] = bases[i]->predictProbability(features);

    //...
     */

}

double MACH::predictForLabel(Label label, Feature* features, Args& args) {
    double prob = 1;
    for (int i = 0; i < hashes.size(); ++i)
        prob *= bases[baseForLabel(label, i)]->predictProbability(features);
    return prob;
}

void MACH::load(Args& args, std::string infile) {
    Log(CERR) << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));

    Log(CERR) << "Loading hashes ...\n";
    std::ifstream in(joinPath(infile, "hashes.bin"));
    int hashCount, a, b;
    in.read((char*)&m, sizeof(m));
    in.read((char*)&bucketCount, sizeof(bucketCount));
    in.read((char*)&hashCount, sizeof(hashCount));
    for(int i = 0; i < hashCount; ++i){
        in.read((char*)&a, sizeof(a));
        in.read((char*)&b, sizeof(b));
        hashes.emplace_back(a, b);
    }
    in.close();

    // This is needed for fast brute force prediction
    baseToLabels.resize(bases.size());
    for(int i = 0; i < m; ++i)
        for (int j = 0; j < hashes.size(); ++j)
            baseToLabels[baseForLabel(i, j)].push_back(i);

}
