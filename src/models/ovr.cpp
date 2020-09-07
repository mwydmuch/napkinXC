/*
 Copyright (c) 2019 by Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
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
            LOG(CERR) << "Encountered example with " << rSize
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

    int parts = calculateNumberOfParts(labels, features, args);
    int range = lCols / parts + 1;

    assert(lCols < range * parts);
    std::vector<std::vector<double>> binLabels(range);
    for (int i = 0; i < binLabels.size(); ++i) binLabels[i].reserve(bRows);

    for (int p = 0; p < parts; ++p) {
        if (parts > 1)
            LOG(CERR) << "Assigning labels for base estimators (" << p + 1 << "/" << parts << ") ...\n";
        else
            LOG(CERR) << "Assigning labels for base estimators ...\n";

        int rStart = p * range;
        int rStop = (p + 1) * range;

        for (int r = 0; r < rows - 1; ++r) {
            printProgress(r, rows);

            int rSize = labels.size(r);
            auto rLabels = labels.row(r);

            if(rSize != 1 && !args.pickOneLabelWeighting) continue;

            for (int i = 0; i < rSize; ++i){
                for (auto &l : binLabels) l.push_back(0.0);
                if (rLabels[i] >= rStart && rLabels[i] < rStop) binLabels[rLabels[i] - rStart].back() = 1.0;
            }
        }

        if(args.pickOneLabelWeighting)
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
