/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
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
    MIPSIndex(size_t dim, Args& args);
    ~MIPSIndex();

    void addPoint(double* pointData, int size, int label);
    void addPoint(UnorderedMap<int, Weight>* pointData, int label);
    void createIndex(Args& args);
    inline size_t getSize() { return data.size(); }
    std::priority_queue<Prediction> predict(Feature* data, size_t k);

protected:
    size_t dim;
    std::string methodType;
    std::string spaceType;
    Space<DATA_T>* space;
    Index<DATA_T>* index;
    ObjectVector data;
};
