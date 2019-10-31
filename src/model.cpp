/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <string>
#include <fstream>
#include <iomanip>

#include "model.h"
#include "threads.h"
#include "measure.h"
#include "ensemble.h"

#include "version.h"
#include "br.h"
#include "ovr.h"
#include "hsm.h"
#include "plt_neg.h"
#include "br_plt_neg.h"
#include "ubop.h"
#include "rbop.h"
#include "ubop_hsm.h"
#include "online_plt.h"
#include "batch_plt.h"

// Mips extension models
#ifdef MIPS_EXT
#include "br_mips.h"
#include "ubop_mips.h"
#endif

std::shared_ptr<Model> modelFactory(Args &args){
    std::shared_ptr<Model> model = nullptr;

    if(args.ensemble > 1){
        switch (args.modelType) {
            case hsm :
                model = std::static_pointer_cast<Model>(std::make_shared<Ensemble<HSM>>());
                break;
            case plt :
                model = std::static_pointer_cast<Model>(std::make_shared<Ensemble<BatchPLT>>());
                break;
            default:
                std::cerr << "Ensemble is not supported for this model type!\n";
                exit(EXIT_FAILURE);
        }
    } else {
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
                model = std::static_pointer_cast<Model>(std::make_shared<BatchPLT>());
                break;
            case pltNeg :
                model = std::static_pointer_cast<Model>(std::make_shared<PLTNeg>());
                break;
            case brpltNeg :
                model = std::static_pointer_cast<Model>(std::make_shared<BRPLTNeg>());
                break;
            case ubop :
                model = std::static_pointer_cast<Model>(std::make_shared<UBOP>());
                break;
            case rbop :
                model = std::static_pointer_cast<Model>(std::make_shared<RBOP>());
                break;
            case ubopHsm :
                model = std::static_pointer_cast<Model>(std::make_shared<UBOPHSM>());
                break;
            case oplt :
                model = std::static_pointer_cast<Model>(std::make_shared<OnlinePLT>());
                break;
                // Mips extension models
            #ifdef MIPS_EXT
            case brMips :
                model = std::static_pointer_cast<Model>(std::make_shared<BRMIPS>());
                break;
            case ubopMips :
                model = std::static_pointer_cast<Model>(std::make_shared<UBOPMIPS>());
                break;
            #endif
            default:
                throw std::invalid_argument("modelFactory: Unknown model type!");
        }
    }

    return model;
}


Model::Model() { }

Model::~Model() { }

std::mutex modelTestMutex;
/*
int pointTestThread(Model* model, Label* labels, Feature* features, Args& args,
        std::vector<std::shared_ptr<Measure>>& measures){

    // Predict
    std::vector<Prediction> prediction;
    model->predict(prediction, features, args);

    // Calculate measures
    modelTestMutex.lock();
    for(auto& m : measures)
        m->accumulate(labels, prediction);
    modelTestMutex.unlock();

    return 0;
}
*/

int batchTestThread(int threadId, Model* model, SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args,
                    const int startRow, const int stopRow, std::vector<std::shared_ptr<Measure>>& measures){

    //std::cerr << "  Thread " << threadId << " predicting rows from " << startRow << " to " << stopRow << "\n";

    // Predict
    const int batchSize = stopRow - startRow;
    std::vector<std::vector<Prediction>> predictions(batchSize);
    for(int r = startRow; r < stopRow; ++r){
        int i = r - startRow;
        model->predict(predictions[i], features.row(r), args);
        if(!threadId) printProgress(i, batchSize);
    }

    // Calculate measures
    modelTestMutex.lock();
    for(auto& m : measures)
        for(int r = startRow; r < stopRow; ++r) {
            int i = r - startRow;
            m->accumulate(labels.row(r), predictions[i]);
        }
    modelTestMutex.unlock();

    return 0;
}

void Model::test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    SRMatrix<Label> trainLabels;
    SRMatrix<Feature> trainFeatures;

    std::cerr << "Starting testing in " << args.threads << " threads ...\n";

    auto measures = measuresFactory(args, this);

    int rows = features.rows();
    assert(rows == labels.rows());

    if(args.threads > 1){
        // Run prediction in parallel

        // Thread pool
        /*
        ThreadPool tPool(args.threads);
        std::vector<std::future<int>> results;

        for(int r = 0; r < rows; ++r)
            results.emplace_back(tPool.enqueue(pointTestThread, this, labels.row(r), features.row(r),
                                               std::ref(args), std::ref(measures)));

        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            results[i].get();
        }
         */

        // Thread sets
        // This one is more efficient
        ThreadSet tSet;
        int tRows = ceil(static_cast<double>(rows)/args.threads);
        for(int t = 0; t < args.threads; ++t)
            tSet.add(batchTestThread, t, this, std::ref(labels), std::ref(features), std::ref(args),
                     t * tRows, std::min((t + 1) * tRows, labels.rows()), std::ref(measures));
        tSet.joinAll();

    } else
        batchTestThread(0, this, labels, features, args, 0, labels.rows(), measures);


    std::cerr << std::setprecision(5) << "Results:\n";
    for(auto& m : measures)
        std::cerr << "  " << m->getName() << ": " << m->value() << std::endl;
}

void Model::checkRow(Label* labels, Feature* feature){ }
