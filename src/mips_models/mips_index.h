/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "space.h"
#include "init.h"
#include "index.h"
#include "params.h"
#include "rangequery.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "space.h"
#include "space/space_vector.h"
#include "space/space_sparse_vector.h"
#include "spacefactory.h"

#include "args.h"
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
    void addPoint(std::unordered_map<int, double>* pointData, int label, DATA_T multi);
    void createIndex(Args& args);
    inline size_t getSize(){ return data.size(); }
    std::priority_queue<Prediction> predict(Feature* data, size_t k);

protected:
    size_t dim;
    std::string methodType;
    std::string spaceType;
    Space<DATA_T>* space;
    Index<DATA_T>* index;
    ObjectVector data;
};
