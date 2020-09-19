/*
 Copyright (c) 2020 by Marek Wydmuch

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
#include "svbop_threshold.h"


SVBOPThreshold::SVBOPThreshold() {
    type = svbopThreshold;
    name = "SVBOP-Threshold";

    productCount = 0;
    dataPointCount = 0;
    correctTop = 0;
}


void SVBOPThreshold::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    int dim = R.size();

    UnorderedSet<int> predictedSet;
    predictedSet.reserve(m);
    std::vector<Prediction> predicted;
    predicted.reserve(m);

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());
    double P = 0, bestU = 0;

    for(int k = 0; k < m; ++k) {
        double lowerBound = -99999;
        double upperBound = 99999;

        int i = 0;
        while (lowerBound < upperBound) {
            //LOG(CERR) << "  lowerBound: " << lowerBound << ", upperBound: " << upperBound << ", i: " << i << "\n";

            upperBound = 0;
            for(Feature *f = features; f->index != -1; ++f) {
                //LOG(CERR) << "    f->index: " << f->index << ", f->value: " << f->value << ", R[f->index].size(): " << R[f->index].size() << "\n";

                if(f->index >= R.size() || R[f->index].size() <= i)
                    continue;

                WeightIndex r;
                if(f->value > 0) r = R[f->index][i];
                else r = R[f->index][R.size() - 1 - i];

                if(r.value == 0) continue;

                //LOG(CERR) << "    f->value: " << f->value << ", r.index: " << r.index << ", r.value: " << r.value << "\n";

                upperBound += f->value * r.value;
                if (!predictedSet.count(r.index)) {
                    double score = bases[r.index]->predictValue(features);
                    predictedSet.insert(r.index);
                    predicted.push_back({r.index, score});
                    std::make_heap(predicted.begin(), predicted.end());
                }
            }

            //LOG(CERR) << "    predicted.size(): " << predicted.size() << "\n";
            lowerBound = predicted.front().value;
            ++i;

            //LOG(CERR) << "  lowerBound: " << lowerBound << ", upperBound: " << upperBound << ", i: " << i << "\n";
        }

        P += exp(predicted.front().value);
        double U = u->g(prediction.size() + 1) * P;
        if (bestU <= U) {
            prediction.push_back(predicted.front());
            bestU = U;

            std::pop_heap(predicted.begin(), predicted.end());
            predicted.pop_back();
        } else
            break;
    }

    productCount += predictedSet.size();
    ++dataPointCount;

//    LOG(COUT) << "  SVBOP-Threshold: pred. size: " << prediction.size() << " P: " << P << " best U: " << bestU << "\n";
//    printVector(prediction);
//    int x;
//    std::cin >> x;
}


void SVBOPThreshold::load(Args& args, std::string infile) {
    LOG(CERR) << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();

    size_t dim = 0;
    for (int i = 0; i < m; ++i) {
        if (bases[i]->getWSize() > dim)
            dim = bases[i]->getWSize();
    }

    LOG(CERR) << "Building inverted index for " << dim << " features ...\n";
    R = std::vector<std::vector<WeightIndex>>(dim);

    for (int i = 0; i < m; ++i) {
        printProgress(i, m);
        if (!bases[i]->isDummy()) {
            bases[i]->setFirstClass(1);
            if (bases[i]->getMapW() != nullptr) {
                for (auto f: (*bases[i]->getMapW()))
                    R[f.first].push_back({i, f.second});
            } else {
                for (int f = 0; f < bases[i]->getWSize(); ++f)
                    if (bases[i]->getW()[f] != 0)
                        R[f].push_back({i, bases[i]->getW()[f]});
            }
        }
    }

    for (int i = 0; i < dim; ++i) {
        R[i].push_back({-1, 0});
        std::sort(R[i].rbegin(), R[i].rend());
    }
}

void SVBOPThreshold::printInfo() {
    LOG(COUT) << name << " additional stats:"
              << "\n  Correct top: " << static_cast<double>(correctTop) / dataPointCount
              << "\n  Mean # estimators per data point: " << static_cast<double>(productCount) / dataPointCount << "\n";
}