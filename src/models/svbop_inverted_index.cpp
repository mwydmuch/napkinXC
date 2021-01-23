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
#include "svbop_inverted_index.h"


SVBOPInvertedIndex::SVBOPInvertedIndex() {
    type = svbopInvertedIndex;
    name = "SVBOP-Inverted Index";

    productCount = 0;
    dataPointCount = 0;
    correctTop = 0;
}

void SVBOPInvertedIndex::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    int dim = R.size();

    UnorderedSet<int> predictedSet;
    predictedSet.reserve(m);
    std::vector<Prediction> predicted;
    predicted.reserve(m);

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());
    double P = 0, bestU = 0;

    int i = 0;
    for(int k = 0; k < m; ++k) {

        for(int j = 0; j < args.svbopInvIndexK; ++j){
            for(Feature *f = features; f->index != -1; ++f) {
                //Log(CERR) << "    f->index: " << f->index << ", f->value: " << f->value << ", R[f->index].size(): " << R[f->index].size() << "\n";

                if(f->index >= R.size() || R[f->index].size() <= i)
                    continue;

                WeightIndex r;
                if(f->value > 0) r = R[f->index][i];
                else r = R[f->index][R.size() - 1 - i];

                if(r.value == 0) continue;

                //Log(CERR) << "    f->value: " << f->value << ", r.index: " << r.index << ", r.value: " << r.value << "\n";

                if (!predictedSet.count(r.index)) {
                    double score = bases[r.index]->predictValue(features);
                    predictedSet.insert(r.index);
                    predicted.push_back({r.index, score});
                    std::make_heap(predicted.begin(), predicted.end());
                }
            }

            ++i;
            //Log(CERR) << "    predicted.size(): " << predicted.size() << "\n";
        }

        double value = exp(predicted.front().value);
        P += value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU <= U) {
            prediction.push_back({predicted.front().label, value});
            bestU = U;

            std::pop_heap(predicted.begin(), predicted.end());
            predicted.pop_back();
        } else
            break;
    }

    productCount += predictedSet.size();
    ++dataPointCount;
}

void SVBOPInvertedIndex::load(Args& args, std::string infile) {
    Log(CERR) << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();

    size_t dim = 0;
    for (int i = 0; i < m; ++i) {
        if (bases[i]->getWSize() > dim)
            dim = bases[i]->getWSize();
    }

    Log(CERR) << "Building inverted index for " << dim << " features ...\n";
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

    loaded = true;
}

void SVBOPInvertedIndex::printInfo() {
    Log(COUT) << name << " additional stats:"
              << "\n  Correct top: " << static_cast<double>(correctTop) / dataPointCount
              << "\n  Mean # estimators per data point: " << static_cast<double>(productCount) / dataPointCount << "\n";
}

SVBOPFagin::SVBOPFagin() {
    type = svbopFagin;
    name = "SVBOP-Fagin";
}

void SVBOPFagin::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    int dim = R.size();

    UnorderedSet<int> predictedSet;
    predictedSet.reserve(m);
    std::vector<Prediction> predicted;
    predicted.reserve(m);
    std::vector<int> inCount(m);

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());
    double P = 0, bestU = 0;

    int fCount = 0;
    for(Feature *f = features; f->index != -1; ++f) ++fCount;

    //Log(CERR) << "  fCount: " << fCount << "\n";

    int inAllCount = 0;
    int i = 0;
    for(int k = 1; k <= m; ++k) {
        while(inAllCount < k) {
            for(Feature *f = features; f->index != -1; ++f) {
                //Log(CERR) << "    f->index: " << f->index << ", f->value: " << f->value << ", R[f->index].size(): " << R[f->index].size() << "\n";

                if(f->value == 0)
                    continue;

                if(f->index >= R.size() || R[f->index].empty()){
                    f->value = 0;
                    --fCount;
                    continue;
                }

                if(i >= R[f->index].size())
                    continue;

                WeightIndex r;
                if(f->value > 0) r = R[f->index][i];
                else r = R[f->index][R.size() - 1 - i];

                if(r.value == 0){
                    for(int j = 0; j < R[f->index].size(); --j)
                        if(R[f->index][j].index >= 0) --inCount[R[f->index][j].index];
                    for(int j = 0; j < m; ++j){
                        ++inCount[j];
                        if(inCount[j] >= fCount) ++inAllCount;
                    }
                    continue;
                }

                ++inCount[r.index];
                if(inCount[r.index] >= fCount) ++inAllCount;

                if (!predictedSet.count(r.index)) {
                    double score = bases[r.index]->predictValue(features);
                    predictedSet.insert(r.index);
                    predicted.push_back({r.index, score});
                    std::make_heap(predicted.begin(), predicted.end());
                }
            }

            //Log(CERR) << "    inAllCount: " << inAllCount << ", fCount: " << fCount << "\n";
            //Log(CERR) << "    predicted.size(): " << predicted.size() << ", i:" << i << "\n";
            ++i;
        }

        double value = exp(predicted.front().value);
        P += value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU <= U) {
            prediction.push_back({predicted.front().label, value});
            bestU = U;

            std::pop_heap(predicted.begin(), predicted.end());
            predicted.pop_back();
        } else
            break;
    }

    productCount += predictedSet.size();
    ++dataPointCount;
}

SVBOPThreshold::SVBOPThreshold() {
    type = svbopThreshold;
    name = "SVBOP-Threshold";
}

void SVBOPThreshold::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    int dim = R.size();

    UnorderedSet<int> predictedSet;
    predictedSet.reserve(m);
    std::vector<Prediction> predicted;
    predicted.reserve(m);

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());
    double P = 0, bestU = 0;

    int i = 0;
    for(int k = 0; k < m; ++k) {
        double lowerBound = -99999;
        double upperBound = 99999;

        while (lowerBound < upperBound) {
            //Log(CERR) << "  lowerBound: " << lowerBound << ", upperBound: " << upperBound << ", i: " << i << "\n";

            upperBound = 0;
            for(Feature *f = features; f->index != -1; ++f) {
                //Log(CERR) << "    f->index: " << f->index << ", f->value: " << f->value << ", R[f->index].size(): " << R[f->index].size() << "\n";

                if(f->index >= R.size() || R[f->index].size() <= i)
                    continue;

                WeightIndex r;
                if(f->value > 0) r = R[f->index][i];
                else r = R[f->index][R.size() - 1 - i];

                if(r.value == 0) continue;

                //Log(CERR) << "    f->value: " << f->value << ", r.index: " << r.index << ", r.value: " << r.value << "\n";

                upperBound += f->value * r.value;
                if (!predictedSet.count(r.index)) {
                    double score = bases[r.index]->predictValue(features);
                    predictedSet.insert(r.index);
                    predicted.push_back({r.index, score});
                    std::make_heap(predicted.begin(), predicted.end());
                }
            }

            //Log(CERR) << "    predicted.size(): " << predicted.size() << "\n";
            lowerBound = predicted.front().value;
            ++i;

            //Log(CERR) << "  lowerBound: " << lowerBound << ", upperBound: " << upperBound << ", i: " << i << "\n";
        }

        double value = exp(predicted.front().value);
        P += value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU <= U) {
            prediction.push_back({predicted.front().label, value});
            bestU = U;

            std::pop_heap(predicted.begin(), predicted.end());
            predicted.pop_back();
        } else
            break;
    }

    productCount += predictedSet.size();
    ++dataPointCount;
}


