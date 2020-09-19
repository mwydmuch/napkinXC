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
#include "svbop_fagin.h"


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

    //LOG(CERR) << "  fCount: " << fCount << "\n";

    int inAllCount = 0;
    for(int k = 1; k <= m; ++k) {

        int i = 0;
        while(inAllCount < k) {
            for(Feature *f = features; f->index != -1; ++f) {
                //LOG(CERR) << "    f->index: " << f->index << ", f->value: " << f->value << ", R[f->index].size(): " << R[f->index].size() << "\n";

                if(f->value == 0)
                    continue;

                if(f->index >= R.size() || R[f->index].empty()){
                    f->value = 0;
                    --fCount;
                    continue;
                }

                if(R[f->index].size() <= i)
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

            //LOG(CERR) << "    inAllCount: " << inAllCount << ", fCount: " << fCount << "\n";
            //LOG(CERR) << "    predicted.size(): " << predicted.size() << ", i:" << i << "\n";
            ++i;
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

//    LOG(COUT) << "  SVBOP-Fagin: pred. size: " << prediction.size() << " P: " << P << " best U: " << bestU << "\n";
//    printVector(prediction);
//    int x;
//    std::cin >> x;
}
