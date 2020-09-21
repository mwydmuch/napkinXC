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

#include "mips_index.h"

using namespace similarity;

MIPSIndex::MIPSIndex(int dim, bool sparse, Args& args) : dim(dim), sparse(sparse) {
    int seed = 0;

    // Init library, specify a log file
    if (LOG_OPTION == 1) initLibrary(seed, LIB_LOGFILE, "logfile.txt");
    // No logging
    if (LOG_OPTION == 2) initLibrary(seed, LIB_LOGNONE, NULL);
    // Use STDERR
    if (LOG_OPTION == 3) initLibrary(seed, LIB_LOGSTDERR, NULL);

    methodType = "hnsw";
    AnyParams empty;
    if(sparse) spaceType = "negdotprod_sparse_fast";
    else spaceType = "negdotprod";

    space = SpaceFactoryRegistry<DATA_T>::Instance().CreateSpace(spaceType, empty);
}

MIPSIndex::~MIPSIndex() {
    for (auto d : data) delete d;
    data.clear();
    delete space;
    delete index;
}

void MIPSIndex::addPoint(Weight* pointData, int size, int label) {
    assert(dim == size);
    if(sparse){
        std::vector<SparseVectElem < DATA_T>> input;
        for (int i = 0; i < size; ++i)
            if(pointData[i] != 0) input.push_back(SparseVectElem<DATA_T>(i, pointData[i]));
        std::sort(input.begin(), input.end());

        auto sparseSpace = reinterpret_cast<const SpaceSparseVector<DATA_T> *>(space);
        data.push_back(sparseSpace->CreateObjFromVect(label, -1, input));
    } else {
        std::vector<DATA_T> input(dim, 0);
        for (int i = 0; i < size; ++i) input[i] = pointData[i];
        auto denseSpace = reinterpret_cast<const VectorSpace<DATA_T> *>(space);
        data.push_back(denseSpace->CreateObjFromVect(label, -1, input));
    }
}

void MIPSIndex::addPoint(UnorderedMap<int, Weight>* pointData, int label) {
    if(sparse) {
        std::vector<SparseVectElem < DATA_T>> input;
        for (const auto &d : *pointData) input.push_back(SparseVectElem<DATA_T>(d.first, d.second));
        std::sort(input.begin(), input.end());

        auto sparseSpace = reinterpret_cast<const SpaceSparseVector<DATA_T> *>(space);
        data.push_back(sparseSpace->CreateObjFromVect(label, -1, input));
    } else {
        std::vector<DATA_T> input(dim, 0);
        for (const auto &d : *pointData)
            if(d.first < dim) input[d.first] = d.second;

        auto denseSpace = reinterpret_cast<const VectorSpace<DATA_T>*>(space);
        data.push_back(denseSpace->CreateObjFromVect(label, -1, input));
    }
}

void MIPSIndex::createIndex(Args& args) {
    Log(CERR) << "Creating MIPS index in " << args.threads << " threads ...\n";

    AnyParams indexParams({
        "post=2",
        "delaunay_type=2",
        "M=" + std::to_string(args.hnswM),
        "efConstruction=" + std::to_string(args.hnswEfConstruction),
        "indexThreadQty=" + std::to_string(args.threads),
    });

    index = MethodFactoryRegistry<DATA_T>::Instance().CreateMethod(true, methodType, spaceType, *space, data);
    index->CreateIndex(indexParams);

    setEfSearch(args.hnswEfSearch);
}

void MIPSIndex::setEfSearch(int ef){
    AnyParams QueryTimeParams({"efSearch=" + std::to_string(ef),});

    // Setting query-time parameters
    index->SetQueryTimeParams(QueryTimeParams);
    efSearch = ef;
}

std::priority_queue<Prediction> MIPSIndex::predict(Feature* data, int k) {
    if(efSearch < k) setEfSearch(k);

    std::priority_queue<Prediction> result;

    Object* query;
    if(sparse) {
        // Sparse query
        std::vector<SparseVectElem < DATA_T>> input;

        Feature *f = data;
        while (f->index != -1) {
            input.push_back(SparseVectElem<DATA_T>(f->index, f->value));
            ++f;
        }
        //std::sort(output.begin(), output.end());

        auto sparse = reinterpret_cast<const SpaceSparseVector<DATA_T>*>(space);
        query = sparse->CreateObjFromVect(0, -1, input);
    } else {
        // Dense query
        std::vector<DATA_T> input(dim, 0);

        Feature *f = data;
        while (f->index != -1) {
            input[f->index] = f->value;
            ++f;
        }

        auto dense = reinterpret_cast<VectorSpace<DATA_T>*>(space);
        query = dense->CreateObjFromVect(0, -1, input);
    }

    KNNQuery<DATA_T> knn(*space, query, k);
    index->Search(&knn, -1);
    auto knnResult = knn.Result()->Clone();
    while (!knnResult->Empty()) {
        result.push({knnResult->TopObject()->id(), -knnResult->TopDistance()});
        knnResult->Pop();
    }

    return result;
}
