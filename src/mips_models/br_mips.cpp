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


BRMIPS::BRMIPS() {
    type = brMips;
    name = "BR-MIPS";
}

void BRMIPS::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {

    std::priority_queue<Prediction> mipsPrediction = mipsIndex->predict(features, args.topK);
    while (!mipsPrediction.empty()) {
        auto p = mipsPrediction.top();
        mipsPrediction.pop();
        prediction.push_back(p);
    }
}

void BRMIPS::load(Args& args, std::string infile) {
    Log(CERR) << "Loading weights ...\n";
    bases = loadBases(joinPath(infile, "weights.bin"));
    m = bases.size();

    size_t dim = 0;
    bool sparse = false;
    for (int i = 0; i < m; ++i) {
        if(bases[i]->getWSize() > dim)
            dim = bases[i]->getWSize();
        if(bases[i]->getMapW() != nullptr){
            sparse = true;
            break;
        }
    }

    mipsIndex = new MIPSIndex(dim, !args.mipsDense, args);
    Log(CERR) << "Adding " << m << " points with " << dim << " dims to MIPSIndex ...\n";
    for (int i = 0; i < m; ++i) {
        printProgress(i, m);
        if(!bases[i]->isDummy()) {
            if (!bases[i]->getFirstClass()) bases[i]->invertWeights();
            if(bases[i]->getMapW() != nullptr) mipsIndex->addPoint(bases[i]->getMapW(), i);
            else mipsIndex->addPoint(bases[i]->getW(), dim, i);
        }
    }

    mipsIndex->createIndex(args);
    loaded = true;
}
