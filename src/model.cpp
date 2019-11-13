/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <string>
#include <fstream>
#include <iomanip>
#include <mutex>

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

    if(args.ensemble > 0){
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

void batchTestThread(int threadId, Model* model, std::vector<std::vector<Prediction>>& predictions,
        SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow){
    const int batchSize = stopRow - startRow;
    for(int r = startRow; r < stopRow; ++r){
        int i = r - startRow;
        model->predict(predictions[r], features[r], args);
        if(!threadId) printProgress(i, batchSize);
    }
}

void Model::test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) {
    std::cerr << "Starting testing in " << args.threads << " threads ...\n";

    // Predict for test set
    std::vector<std::vector<Prediction>> predictions = predictBatch(features, args);

    // Create measures and calculate scores
    auto measures = measuresFactory(args, this);
    for (auto& m : measures) m->accumulate(predictions, labels);

    // Print results
    std::cerr << std::setprecision(5) << "Results:\n";
    for (auto& m : measures)
        std::cerr << "  " << m->getName() << ": " << m->value() << std::endl;
}

std::vector<std::vector<Prediction>> Model::predictBatch(SRMatrix<Feature>& features, Args& args){
    int rows = features.rows();
    std::vector<std::vector<Prediction>> predictions(rows);

    if(args.threads > 1){
        // Run prediction in parallel using thread set
        ThreadSet tSet;
        int tRows = ceil(static_cast<double>(rows)/args.threads);
        for(int t = 0; t < args.threads; ++t)
            tSet.add(batchTestThread, t, this, std::ref(predictions), std::ref(features), std::ref(args),
                     t * tRows, std::min((t + 1) * tRows, rows));
        tSet.joinAll();

    } else
        batchTestThread(0, this, predictions, features, args, 0, rows);

    return predictions;
}

void Model::checkRow(Label* labels, Feature* feature){ }
