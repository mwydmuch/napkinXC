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

#include "br.h"
#include "threads.h"


BR::BR() {
    type = br;
    name = "BR";
}

BR::~BR() {
    for (auto b : bases) delete b;
}

void BR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {
    // Check data
    int rows = features.rows();
    int lCols = labels.cols();
    assert(rows == labels.rows());

    std::ofstream out(joinPath(output, "weights.bin"));
    int size = lCols;
    out.write((char*)&size, sizeof(size));

    int parts = calculateNumberOfParts(labels, features, args);
    int range = lCols / parts + 1;

    assert(lCols < range * parts);
    std::vector<std::vector<double>> binLabels(range);
    for (int i = 0; i < binLabels.size(); ++i) binLabels[i].reserve(rows);

    for (int p = 0; p < parts; ++p) {
        if (parts > 1)
            std::cerr << "Assigning labels for base estimators (" << p + 1 << "/" << parts << ") ...\n";
        else
            std::cerr << "Assigning labels for base estimators ...\n";

        int rStart = p * range;
        int rStop = (p + 1) * range;

        for (int r = 0; r < rows; ++r) {
            printProgress(r, rows);

            int rSize = labels.size(r);
            auto rLabels = labels.row(r);

            for (auto &l : binLabels) l.push_back(0.0);
            for (int i = 0; i < rSize; ++i)
                if (rLabels[i] >= rStart && rLabels[i] < rStop) binLabels[rLabels[i] - rStart].back() = 1.0;
        }

        trainBasesWithSameFeatures(out, features.cols(), binLabels, features.allRows(), nullptr, args);
        for (auto& l : binLabels) l.clear();
    }

    out.close();
}

void BR::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    prediction = predictForAllLabels(features, args);

    sort(prediction.rbegin(), prediction.rend());
    if (args.threshold > 0) {
        int i = 0;
        while (prediction[i++].value > args.threshold)
            ;
        prediction.resize(i - 1);
    }
    if (args.topK > 0) prediction.resize(args.topK);
}

std::vector<Prediction> BR::predictForAllLabels(Feature* features, Args& args) {
    std::vector<Prediction> prediction;
    prediction.reserve(bases.size());
    for (int i = 0; i < bases.size(); ++i) prediction.push_back({i, bases[i]->predictProbability(features)});
    return prediction;
}

void BR::predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, std::vector<float>& thresholds,
                               Args& args) {
    std::vector<Prediction> tmpPrediction = predictForAllLabels(features, args);
    for (auto& p : tmpPrediction) {
        if (p.value >= thresholds[p.label]) prediction.push_back(p);
    }
}

double BR::predictForLabel(Label label, Feature* features, Args& args) {
    return bases[label]->predictProbability(features);
}

void BR::load(Args& args, std::string infile) {
    std::cerr << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();
}

void BR::printInfo() {
    std::cerr << name << " additional stats:"
              << "\n  Mean # estimators per data point: " << bases.size() << "\n";
}

size_t BR::calculateNumberOfParts(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    int rows = features.rows();
    int lCols = labels.cols();

    // Calculate required memory
    // Size of required data
    unsigned long long dataMem = labels.mem() + features.mem();
    unsigned long long tmpDataMem = lCols * (rows * sizeof(double) + sizeof(void*));
    unsigned long long baseMem = args.threads * 3 * features.cols() * sizeof(double);
    unsigned long long reqMem = tmpDataMem + dataMem + baseMem;
    std::cerr << "Required memory to train: " << formatMem(reqMem) << ", available memory: " << formatMem(args.memLimit) << std::endl;

    size_t parts = tmpDataMem / (args.memLimit - dataMem - baseMem) + 1;
    return parts;
}
