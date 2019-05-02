/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <string>
#include <fstream>
#include <iomanip>

#include "model.h"
#include "ovr.h"
#include "br.h"
#include "hsm.h"
#include "plt.h"
#include "threads.h"

#include "set_value.h"
#include "hsm_ubop.h"
#include "ubop.h"


std::shared_ptr<Model> modelFactory(Args &args){
    std::shared_ptr<Model> model = nullptr;
    switch (args.modelType) {
        case ovr :
            model = std::static_pointer_cast<Model>(std::make_shared<OVR>());
            break;
        case br :
            model = std::static_pointer_cast<Model>(std::make_shared<BR>());
            break;
        case hsm :
            model = std::static_pointer_cast<Model>(std::make_shared<HSM>());
            break;
        case plt :
            model = std::static_pointer_cast<Model>(std::make_shared<PLT>());
            break;
        case ubop :
            model = std::static_pointer_cast<Model>(std::make_shared<UBOP>());
            break;
        case hsmubop :
            model = std::static_pointer_cast<Model>(std::make_shared<HSMUBOP>());
            break;
        default:
            throw "Unknown model type!";
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

    for(int r = startRow; r < stopRow; ++r){

        std::vector<Prediction> prediction;
        model->predict(prediction, features.row(r), args);

        for (int i = 0; i < args.topK; ++i)
            for (int j = 0; j < labels.size(r); ++j)
                if (prediction[i].label == labels.row(r)[j]){
                    ++localCorrectAt[i];
                    localCoveredAt[i].insert(prediction[i].label);
                    break;
                }

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

int batchTestThread2(int threadId, Model* model, SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args,
                    const int startRow, const int stopRow,
                    double& gacc, double& gu_alfa, double& gu_alfa_beta, double& gu_delta_gamma, double& gu_P, double& gu_F1, double& gpSize){

    //std::cerr << "  Thread " << threadId << " predicting rows from " << startRow << " to " << stopRow << "\n";

    // Set based measures
    double lacc = 0, lu_alfa = 0, lu_alfa_beta = 0, lu_delta_gamma = 0, lu_P = 0, lu_F1 = 0, lpSize = 0;

    for(int r = startRow; r < stopRow; ++r){

        std::vector<Prediction> prediction;
        model->predict(prediction, features.row(r), args);

        lacc += acc(labels.row(r)[0], prediction);
        //lu_alfa += u_alfa(labels.row(r)[0], prediction, 0.5);
        lu_alfa_beta += u_alfa_beta(labels.row(r)[0], prediction, 1.0, 1.0, labels.cols());
        //lu_delta_gamma += u_delta_gamma(labels.row(r)[0], prediction, 1, 1);
        lu_P += u_P(labels.row(r)[0], prediction);
        lu_F1 += u_F1(labels.row(r)[0], prediction);
        lpSize += prediction.size();

        if(!threadId) printProgress(r - startRow, stopRow - startRow);
    }

    testMutex.lock();

    gacc += lacc;
    gu_alfa += lu_alfa;
    gu_alfa_beta += lu_alfa_beta,
    gu_delta_gamma += lu_delta_gamma;
    gu_P += lu_P;
    gu_F1 += lu_F1;
    gpSize += lpSize;

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

    // TODO: Rewrite test part
    double gacc = 0, gu_alfa = 0, gu_alfa_beta = 0, gu_delta_gamma = 0, gu_P = 0, gu_F1 = 0, gpSize = 0;

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
//        ThreadSet tSet;
//        int tRows = ceil(static_cast<double>(rows)/args.threads);
//        for(int t = 0; t < args.threads; ++t)
//            tSet.add(batchTestThread, t, this, std::ref(labels), std::ref(features), std::ref(args),
//                     t * tRows, std::min((t + 1) * tRows, labels.rows()), std::ref(correctAt), std::ref(coveredAt));
//        tSet.joinAll();

        ThreadSet tSet;
        int tRows = ceil(static_cast<double>(rows)/args.threads);
        for(int t = 0; t < args.threads; ++t)
            tSet.add(batchTestThread2, t, this, std::ref(labels), std::ref(features), std::ref(args),
                     t * tRows, std::min((t + 1) * tRows, labels.rows()),
                     std::ref(gacc), std::ref(gu_alfa), std::ref(gu_alfa_beta), std::ref(gu_delta_gamma), std::ref(gu_P),
                     std::ref(gu_F1), std::ref(gpSize));
        tSet.joinAll();

    } else {
        std::vector<Prediction> prediction;
        for(int r = 0; r < rows; ++r){
            prediction.clear();
            predict(prediction, features.row(r), args);
            for (int i = 0; i < args.topK; ++i)
                for (int j = 0; j < labels.sizes()[r]; ++j)
                    if (prediction[i].label == labels.data()[r][j]){
                        ++correctAt[i];
                        coveredAt[i].insert(prediction[i].label);
                        break;
                    }

            printProgress(r, rows);
        }
    }

    /*
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
     */

    std::cerr << std::setprecision(5);
    std::cerr << "Acc: " << gacc / rows << "\n"
              << "uP: " << gu_P / rows<< "\n"
              << "uF1: " << gu_F1 / rows<< "\n"
              << "uAlfa: " << gu_alfa / rows<< "\n"
              << "uAlfaBeta: " << gu_alfa_beta / rows<< "\n"
              << "uDeltaGamma: " << gu_delta_gamma / rows<< "\n"
              << "Avg. P Size: " << gpSize / rows<< "\n";

}
