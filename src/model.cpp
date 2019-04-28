/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <string>
#include <fstream>
#include <iomanip>

#include "model.h"
#include "br.h"
#include "plt.h"
#include "threads.h"

std::shared_ptr<Model> modelFactory(Args &args){
    std::shared_ptr<Model> model = nullptr;
    switch (args.modelType) {
        case ModelType::br :
            model = std::static_pointer_cast<Model>(std::make_shared<BR>());
            break;
        case ModelType::plt :
            model = std::static_pointer_cast<Model>(std::make_shared<PLT>());
            break;

    }

    return model;
}

Model::Model() { }

Model::~Model() { }

std::mutex testMutex;
int pointTestThread(Model* model, Label* labels, Feature* features, Args& args,
                    std::vector<int>& correctAt, std::vector<std::unordered_set<int>>& coveredAt){

    std::vector<Prediction> prediction;
    model->predict(prediction, features, args);

    testMutex.lock();
    for (int i = 0; i < args.topK; ++i){
        int l = -1;
        while(labels[++l] > -1)
            if (prediction[i].label == labels[l]){
                ++correctAt[i];
                coveredAt[i].insert(prediction[i].label);
                break;
            }
    }
    testMutex.unlock();

    return 0;
}

int batchTestThread(int threadId, Model* model, SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args,
                    const int startRow, const int stopRow,
                    std::vector<int>& correctAt, std::vector<std::unordered_set<int>>& coveredAt){

    //std::cerr << "  Thread " << threadId << " predicting rows from " << startRow << " to " << stopRow << "\n";

    std::vector<int> localCorrectAt (args.topK);
    std::vector<std::unordered_set<int>> localCoveredAt(args.topK);

    //std::vector<double> denseFeatures(features.cols());

    for(int r = startRow; r < stopRow; ++r){
        //setVector(features.row(r), denseFeatures, -1);

        std::vector<Prediction> prediction;
        //tree->predict(prediction, denseFeatures.data(), bases, kNNs, args);
        model->predict(prediction, features.row(r), args);

        for (int i = 0; i < args.topK; ++i)
            for (int j = 0; j < labels.size(r); ++j)
                if (prediction[i].label == labels.row(r)[j]){
                    ++localCorrectAt[i];
                    localCoveredAt[i].insert(prediction[i].label);
                    break;
                }

        //setVectorToZeros(features.row(r), denseFeatures, -1);
        if(!threadId) printProgress(r - startRow, stopRow - startRow);
    }

    testMutex.lock();
    for (int i = 0; i < args.topK; ++i) {
        correctAt[i] += localCorrectAt[i];
        for(const auto& l : localCoveredAt[i]) coveredAt[i].insert(l);
    }
    testMutex.unlock();

    return 0;
}

void Model::test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    SRMatrix<Label> trainLabels;
    SRMatrix<Feature> trainFeatures;

    std::cerr << "Starting testing in " << args.threads << " threads ...\n";

    std::vector<int> correctAt(args.topK);
    std::vector<std::unordered_set<int>> coveredAt(args.topK);
    int rows = features.rows();
    assert(rows == labels.rows());

    if(args.threads > 1){
        // Run prediction in parallel

        // Thread pool
        // Implements method with examples in sparse representation
        /*
        ThreadPool tPool(args.threads);
        std::vector<std::future<int>> results;

        for(int r = 0; r < rows; ++r)
            results.emplace_back(tPool.enqueue(pointTestThread, this, labels.row(r), features.row(r), std::ref(bases),
                                               std::ref(kNNs), std::ref(args), std::ref(correctAt), std::ref(coveredAt)));

        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            results[i].get();
        }
         */

        // Batches
        // Implements method with example in dense representation
        ThreadSet tSet;
        int tRows = ceil(static_cast<double>(rows)/args.threads);
        for(int t = 0; t < args.threads; ++t)
            tSet.add(batchTestThread, t, this, std::ref(labels), std::ref(features), std::ref(args),
                     t * tRows, std::min((t + 1) * tRows, labels.rows()), std::ref(correctAt), std::ref(coveredAt));
        tSet.joinAll();

    } else {
        std::vector<Prediction> prediction;

        //std::vector<double> denseFeatures(features.cols());

        for(int r = 0; r < rows; ++r){
            //setVector(features.row(r), denseFeatures, -1);

            prediction.clear();
            //predict(prediction, denseFeatures.data(), bases, kNNs, args);
            predict(prediction, features.row(r), args);
            for (int i = 0; i < args.topK; ++i)
                for (int j = 0; j < labels.sizes()[r]; ++j)
                    if (prediction[i].label == labels.data()[r][j]){
                        ++correctAt[i];
                        coveredAt[i].insert(prediction[i].label);
                        break;
                    }

            //setVectorToZeros(features.row(r), denseFeatures, -1);
            printProgress(r, rows);
        }
    }

    double precisionAt = 0, coverageAt;
    std::cerr << std::setprecision(5);
    for (int i = 0; i < args.topK; ++i) {
        int k = i + 1;
        if(i > 0) for(const auto& l : coveredAt[i - 1]) coveredAt[i].insert(l);
        precisionAt += correctAt[i];
        coverageAt = coveredAt[i].size();
        std::cerr << "P@" << k << ": " << precisionAt / (rows * k)
                  << ", R@" << k << ": " << precisionAt / labels.cells()
                  << ", C@" << k << ": " << coverageAt / labels.cols() << "\n";
    }

    std::cerr << "All done\n";
}
