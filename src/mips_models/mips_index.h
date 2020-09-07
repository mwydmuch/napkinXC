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

#pragma once

#include "index.h"
#include "init.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "params.h"
#include "rangequery.h"
#include "space.h"
#include "space/space_sparse_vector.h"
#include "space/space_vector.h"
#include "spacefactory.h"

#include "args.h"
#include "base.h"
#include "model.h"

#include <iostream>
#include <vector>

#define LOG_OPTION 2
#define DATA_T float

using namespace similarity;

class MIPSIndex {
public:
    MIPSIndex(int dim, bool sparse, Args& args);
    ~MIPSIndex();

    void addPoint(Weight* pointData, int size, int label);
    void addPoint(UnorderedMap<int, Weight>* pointData, int label);
    void createIndex(Args& args);

    void setEfSearch(int ef);
    std::priority_queue<Prediction> predict(Feature* data, int k);

    inline size_t getSize() { return data.size(); }

protected:
    bool sparse;
    int dim;
    std::string methodType;
    std::string spaceType;
    Space<DATA_T>* space;
    Index<DATA_T>* index;
    ObjectVector data;

    int efSearch;
};
