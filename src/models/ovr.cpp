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

#include "ovr.h"
#include "threads.h"


OVR::OVR() {
    type = ovr;
    name = "OVR";
}

void OVR::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {
    // Check data
    int rows = features.rows();
    int bRows = rows;
    int lCols = labels.cols();
    assert(rows == labels.rows());

    std::vector<double>* binWeights = nullptr;
    if (args.pickOneLabelWeighting) {
        bRows = labels.cells();
        binWeights = new std::vector<double>();
        binWeights->reserve(bRows);
    }

    std::vector<Feature*> binFeatures;
    binFeatures.reserve(bRows);

    for (int r = 0; r < rows; ++r) {
        int rSize = labels.size(r);

        if (rSize != 1 && !args.pickOneLabelWeighting) {
            std::cerr << "Row " << r << ": encountered example with " << rSize
                      << " labels! OVR is multi-class classifier, use BR or --pickOneLabelWeighting option instead!\n";
            continue;
        }

        for (int i = 0; i < rSize; ++i)
            binFeatures.push_back(features[r]);
        if(args.pickOneLabelWeighting)
            for (int i = 0; i < rSize; ++i)
                binWeights->push_back(1.0 / rSize);
    }

    std::ofstream out(joinPath(output, "weights.bin"));
    int size = lCols;
    out.write((char*)&size, sizeof(size));

    // TODO: Calculate required memory
    unsigned long long reqMem = lCols * (rows * sizeof(double) + sizeof(void*)) + labels.mem() + features.mem();

    int parts = calculateNumberOfParts(labels, features, args);
    int range = lCols / parts + 1;

    assert(lCols < range * parts);
    std::vector<std::vector<double>> binLabels(range);
    for (int i = 0; i < binLabels.size(); ++i) binLabels[i].reserve(bRows);

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

            if(rSize != 1 && !args.pickOneLabelWeighting) continue;

            for (int i = 0; i < rSize; ++i){
                for (auto &l : binLabels) l.push_back(0.0);
                if (rLabels[i] >= rStart && rLabels[i] < rStop) binLabels[rLabels[i] - rStart].back() = 1.0;
            }
        }

        assert(binLabels[0].size() == binWeights->size());

        trainBasesWithSameFeatures(out, features.cols(), binLabels, binFeatures, binWeights, args);
        for (auto& l : binLabels) l.clear();
    }

    out.close();
    delete binWeights;
}

std::vector<Prediction> OVR::predictForAllLabels(Feature* features, Args& args) {
    std::vector<Prediction> prediction;
    prediction.reserve(bases.size());
    double sum = 0;

    for (int i = 0; i < bases.size(); ++i) {
        double value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
        prediction.push_back({i, value});
    }

    for (auto& p : prediction) p.value /= sum;
    return prediction;
}

double OVR::predictForLabel(Label label, Feature* features, Args& args) {
    double sum = 0;
    for (int i = 0; i < bases.size(); ++i) {
        double value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
    }

    return exp(bases[label]->predictValue(features)) / sum;
}
