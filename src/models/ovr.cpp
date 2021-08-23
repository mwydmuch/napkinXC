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

void OVR::assignDataPoints(std::vector<std::vector<Real>>& binLabels, std::vector<Feature*>& binFeatures, std::vector<Real>& binWeights,
                          SRMatrix& labels, SRMatrix& features, int rStart, int rStop, Args& args){
    int rows = labels.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        int rSize = labels.size(r);
        auto& rLabels = labels[r];

        if (rSize != 1 && !args.pickOneLabelWeighting)
            throw std::invalid_argument("Encountered example with " + std::to_string(rSize) + " labels! OVR is multi-class classifier, use BR or --pickOneLabelWeighting option instead!");

        for (int i = 0; i < rSize; ++i){
            binFeatures.push_back(features[r].data());
            binWeights.push_back(1.0 / rSize);
            for (auto &bl: binLabels) bl.push_back(0);
            if (rLabels[i] >= rStart && rLabels[i] < rStop) binLabels[rLabels[i] - rStart].back() = 1;
        }
    }
}

std::vector<Prediction> OVR::predictForAllLabels(SparseVector& features, Args& args) {
    std::vector<Prediction> prediction;
    prediction.reserve(bases.size());
    Real sum = 0;

    for (int i = 0; i < bases.size(); ++i) {
        Real value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
        prediction.emplace_back(i, value);
    }

    for (auto& p : prediction) p.value /= sum;
    return prediction;
}

Real OVR::predictForLabel(Label label, SparseVector& features, Args& args) {
    Real sum = 0;
    for (int i = 0; i < bases.size(); ++i) {
        Real value = exp(bases[i]->predictValue(features)); // Softmax normalization
        sum += value;
    }

    return exp(bases[label]->predictValue(features)) / sum;
}
