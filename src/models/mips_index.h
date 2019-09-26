/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include <iostream>


class MIPSIndex {
public:
    MIPSIndex(const size_t dim, const size_t max_elements): dim(dim), max_elements(max_elements) {
        //l2space = new hnswlib::L2Space(dim);
        l2space = new hnswlib::InnerProductSpace(dim);
        hnsw = new hnswlib::HierarchicalNSW<float>(l2space, max_elements, 32, 300);
        hnsw->setEf(48);
    }

    ~MIPSIndex() {
        delete l2space;
        delete hnsw;
    }

    void addPoint(float* data, hnswlib::labeltype label){
        hnsw->addPoint((void *)(data), label);
    }

    size_t getSize(){
        return hnsw->cur_element_count;
    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> mips(float* data, int k) {
        return hnsw->searchKnn((void *)(data), k);
    }


protected:
    hnswlib::HierarchicalNSW<float> *hnsw;
    hnswlib::SpaceInterface<float> *l2space;

    const size_t dim;
    const size_t max_elements;
};
