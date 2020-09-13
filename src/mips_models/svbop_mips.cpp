/*
 Copyright (c) 2019-2020 by Marek Wydmuch

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

#include "set_utility.h"
#include "svbop_mips.h"


SVBOPMIPS::SVBOPMIPS() {
    type = svbopMips;
    name = "SVBOP-MIPS";
}

void SVBOPMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {

    int k;
    if(args.svbopMipsK < 1) k = std::ceil(static_cast<double>(bases.size()) * args.svbopMipsK);
    else k = args.svbopMipsK;

    std::vector<Prediction> allPredictions;
    UnorderedSet<int> seenLabels;
    std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, k);
    while (!mipsPrediction.empty()) {
        auto p = mipsPrediction.top();
        mipsPrediction.pop();
        p.value = exp(p.value);
        allPredictions.push_back(p);
        seenLabels.insert(p.label);
    }

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());
    double P = 0, bestU = 0;
    for (int i = 0; i < allPredictions.size(); ++i) {
        auto& p = allPredictions[i];
        P += p.value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU <= U) {
            prediction.push_back(p);
            bestU = U;
        } else
            break;

        if (i + 1 == allPredictions.size()) {
            k *= 2;
            std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, k);
            while (!mipsPrediction.empty()) {
                auto p = mipsPrediction.top();
                mipsPrediction.pop();

                if (!seenLabels.count(p.label)){
                    p.value = exp(p.value);
                    allPredictions.push_back(p);
                    seenLabels.insert(p.label);
                }
            }
        }
    }
}
