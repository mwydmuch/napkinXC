/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "mips_index.h"

using namespace similarity;

MIPSIndex::MIPSIndex(int dim, Args& args) : dim(dim) {
    int seed = 0;

    // Init library, specify a log file
    if (LOG_OPTION == 1) initLibrary(seed, LIB_LOGFILE, "logfile.txt");
    // No logging
    if (LOG_OPTION == 2) initLibrary(seed, LIB_LOGNONE, NULL);
    // Use STDERR
    if (LOG_OPTION == 3) initLibrary(seed, LIB_LOGSTDERR, NULL);

    spaceType = "negdotprod_sparse_fast";
    methodType = "hnsw";
    AnyParams empty;
    space = SpaceFactoryRegistry<DATA_T>::Instance().CreateSpace(spaceType, empty);
}

MIPSIndex::~MIPSIndex() {
    for (auto d : data) delete d;
    data.clear();
    delete space;
    delete index;
}

void MIPSIndex::addPoint(double* pointData, int size, int label) {
    assert(dim == size);
    std::vector<DATA_T> output(pointData, pointData + size);
    auto denseSpace = reinterpret_cast<VectorSpace<DATA_T>*>(space);
    data.push_back(denseSpace->CreateObjFromVect(label, -1, output));
}

void MIPSIndex::addPoint(UnorderedMap<int, Weight>* pointData, int label) {
    std::vector<SparseVectElem<DATA_T>> output;

    for (const auto& d : *pointData) output.push_back(SparseVectElem<DATA_T>(d.first, d.second));

    std::sort(output.begin(), output.end());
    auto sparseSpace = reinterpret_cast<const SpaceSparseVector<DATA_T>*>(space);
    data.push_back(sparseSpace->CreateObjFromVect(label, -1, output));
}

void MIPSIndex::createIndex(Args& args) {
    std::cerr << "Creating MIPS index in " << args.threads << " threads ...\n";

    AnyParams indexParams({
        "post=2",
        "indexThreadQty=" + std::to_string(args.threads),
    });

    index = MethodFactoryRegistry<DATA_T>::Instance().CreateMethod(true, methodType, spaceType, *space, data);
    index->CreateIndex(indexParams);

    /*
    AnyParams QueryTimeParams(
        {
            //"efSearch=50",
        }
    );

    // Setting query-time parameters
    index->SetQueryTimeParams(QueryTimeParams);
    */
}

std::priority_queue<Prediction> MIPSIndex::predict(Feature* data, int k) {
    // std::cerr << "Quering index\n";

    std::priority_queue<Prediction> result;

    // Sparse query
    std::vector<SparseVectElem<DATA_T>> output;

    Feature* f = data;
    while (f->index != -1) {
        output.push_back(SparseVectElem<DATA_T>(f->index - 1, f->value));
        ++f;
    }
    std::sort(output.begin(), output.end());
    auto sparse = reinterpret_cast<const SpaceSparseVector<DATA_T>*>(space);
    auto query = sparse->CreateObjFromVect(0, -1, output);

    // Dense query
    /*
    std::vector<DATA_T> output;
    output.resize(dim);
    std::fill(output.begin(), output.end(), 0);

    Feature *f = data;
    while (f->index != -1) {
        output[f->index - 1] = f->value;
        ++f;
    }

    auto denseSpace = reinterpret_cast<VectorSpace<DATA_T>*>(space);
    auto query = denseSpace->CreateObjFromVect(0, -1, output);
     */

    KNNQuery<DATA_T> knn(*space, query, k);
    index->Search(&knn, -1);
    auto knnResult = knn.Result()->Clone();
    while (!knnResult->Empty()) {
        result.push({knnResult->TopObject()->id(), -knnResult->TopDistance()});
        // result.push({knnResult->TopObject()->id(), 1.0 / (1.0 + std::exp(knnResult->TopDistance()))});
        knnResult->Pop();
    }

    return result;
}
