/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

#include "ensemble.h"
#include "measure.h"
#include "model.h"
#include "threads.h"

#include "br.h"
#include "hsm.h"
#include "online_plt.h"
#include "ovr.h"
#include "plt.h"
#include "rbop.h"
#include "ubop.h"
#include "ubop_hsm.h"
#include "version.h"

// Mips extension models
#ifdef MIPS_EXT
#include "br_mips.h"
#include "ubop_mips.h"
#endif

std::shared_ptr<Model> Model::factory(Args& args) {
    std::shared_ptr<Model> model = nullptr;

    if (args.ensemble > 0) {
        switch (args.modelType) {
        case hsm: model = std::static_pointer_cast<Model>(std::make_shared<Ensemble<HSM>>()); break;
        case plt: model = std::static_pointer_cast<Model>(std::make_shared<Ensemble<BatchPLT>>()); break;
        default: throw std::invalid_argument("Ensemble is not supported for this model type!");
        }
    } else {
        switch (args.modelType) {
        case ovr: model = std::static_pointer_cast<Model>(std::make_shared<OVR>()); break;
        case br: model = std::static_pointer_cast<Model>(std::make_shared<BR>()); break;
        case hsm: model = std::static_pointer_cast<Model>(std::make_shared<HSM>()); break;
        case plt: model = std::static_pointer_cast<Model>(std::make_shared<BatchPLT>()); break;
        case ubop: model = std::static_pointer_cast<Model>(std::make_shared<UBOP>()); break;
        case rbop: model = std::static_pointer_cast<Model>(std::make_shared<RBOP>()); break;
        case ubopHsm: model = std::static_pointer_cast<Model>(std::make_shared<UBOPHSM>()); break;
        case oplt:
            model = std::static_pointer_cast<Model>(std::make_shared<OnlinePLT>());
            break;
            // Mips extension models
#ifdef MIPS_EXT
        case brMips: model = std::static_pointer_cast<Model>(std::make_shared<BRMIPS>()); break;
        case ubopMips: model = std::static_pointer_cast<Model>(std::make_shared<UBOPMIPS>()); break;
#endif
        default: throw std::invalid_argument("Unknown model type!");
        }
    }

    return model;
}

Model::Model() {}

Model::~Model() {}

void Model::predictWithThresholds(std::vector<Prediction>& prediction, Feature* features,
                                  std::vector<float>& thresholds, Args& args) {
    std::vector<Prediction> tmpPrediction;
    predict(tmpPrediction, features, args);
    for (auto& p : tmpPrediction)
        if (p.value >= thresholds[p.label]) prediction.push_back(p);
}

void Model::predictBatchThread(int threadId, Model* model, std::vector<std::vector<Prediction>>& predictions,
                               SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow) {
    const int batchSize = stopRow - startRow;
    for (int r = startRow; r < stopRow; ++r) {
        int i = r - startRow;
        model->predict(predictions[r], features[r], args);
        if (!threadId) printProgress(i, batchSize);
    }
}

std::vector<std::vector<Prediction>> Model::predictBatch(SRMatrix<Feature>& features, Args& args) {
    std::cerr << "Starting prediction in " << args.threads << " threads ...\n";

    int rows = features.rows();
    std::vector<std::vector<Prediction>> predictions(rows);

    // Run prediction in parallel using thread set
    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(rows) / args.threads);
    for (int t = 0; t < args.threads; ++t)
        tSet.add(predictBatchThread, t, this, std::ref(predictions), std::ref(features), std::ref(args), t * tRows,
                 std::min((t + 1) * tRows, rows));
    tSet.joinAll();

    return predictions;
}

void Model::predictBatchWithThresholdsThread(int threadId, Model* model,
                                             std::vector<std::vector<Prediction>>& predictions,
                                             SRMatrix<Feature>& features, std::vector<float>& thresholds, Args& args,
                                             const int startRow, const int stopRow) {
    const int batchSize = stopRow - startRow;
    for (int r = startRow; r < stopRow; ++r) {
        int i = r - startRow;
        model->predictWithThresholds(predictions[r], features[r], thresholds, args);
        if (!threadId) printProgress(i, batchSize);
    }
}

std::vector<std::vector<Prediction>> Model::predictBatchWithThresholds(SRMatrix<Feature>& features,
                                                                       std::vector<float>& thresholds, Args& args) {
    std::cerr << "Starting prediction in " << args.threads << " threads ...\n";

    int rows = features.rows();
    std::vector<std::vector<Prediction>> predictions(rows);

    // Run prediction in parallel using thread set
    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(rows) / args.threads);
    for (int t = 0; t < args.threads; ++t)
        tSet.add(predictBatchWithThresholdsThread, t, this, std::ref(predictions), std::ref(features),
                 std::ref(thresholds), std::ref(args), t * tRows, std::min((t + 1) * tRows, rows));
    tSet.joinAll();

    return predictions;
}

std::vector<float> Model::ofo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args) {

    // Initialize thresholds with zeros
    std::vector<float> thresholds(m, 0);
    // Variables required for OFO
    std::vector<float> as(m, 0);
    std::vector<float> bs(m, 0);

    std::cerr << "Optimizing thresholds with OFO using 1 thread.\n";
    int rows = features.rows();
    int log_every = rows/100;
    int l;

    for(int epoch = 0; epoch < args.epochs; epoch++){
        for (int r = 0; r < rows; ++r) {
            // Predict with current thresholds
            std::vector<Prediction> prediction; // In every iteration a clear vector
            this->predictWithThresholds(prediction, features[r], thresholds, args);
            if (r % log_every == 0) printProgress(r, rows);

            // Update a and b counters
            for (const auto& p : prediction) {
                // b[j] =  sum_{i = 1}^{t} \hat y_j + ..
                bs[p.label]++;

                // a[j] = sum_{i = 1}^{t} y_j \hat y_j
                l = -1;
                while (labels[r][++l] > -1)
                    if (p.label == labels[r][l]) {
                        as[p.label]++;
                        break;
                    }
            }
            // b[j] =  .. + sum_{i = 1}^{t} y_j
            l = -1;
            while (labels[r][++l] > -1){
                bs[labels[r][l]]++;
            }

            // Update thresholds, only those that may have changed due to update of as or bs,
            // For simplicity I compute some of them twice because it does not really matter

            for (const auto& p : prediction) {
                thresholds[p.label] = as[p.label] / bs[p.label];
            }
            l = -1;
            while (labels[r][++l] > -1){
                thresholds[labels[r][l]] = as[labels[r][l]] / bs[labels[r][l]];
            }
        }
    }
    return thresholds;
}

Base* Model::trainBase(int n, std::vector<double>& baseLabels, std::vector<Feature*>& baseFeatures,
                       std::vector<double>* instancesWeights, Args& args) {
    Base* base = new Base();
    base->train(n, baseLabels, baseFeatures, instancesWeights, args);
    return base;
}

void Model::saveResults(std::ofstream& out, std::vector<std::future<Base*>>& results) {
    for (int i = 0; i < results.size(); ++i) {
        printProgress(i, results.size());
        Base* base = results[i].get();
        base->save(out);
        delete base;
    }
}

void Model::trainBases(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                       std::vector<std::vector<Feature*>>& baseFeatures,
                       std::vector<std::vector<double>*>* instancesWeights, Args& args) {

    std::ofstream out(outfile);
    int size = baseLabels.size();
    out.write((char*)&size, sizeof(size));
    trainBases(out, n, baseLabels, baseFeatures, instancesWeights, args);
    out.close();
}

void Model::trainBases(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                       std::vector<std::vector<Feature*>>& baseFeatures,
                       std::vector<std::vector<double>*>* instancesWeights, Args& args) {

    assert(baseLabels.size() == baseFeatures.size());
    if (instancesWeights != nullptr) assert(baseLabels.size() == instancesWeights->size());

    int size = baseLabels.size(); // This "batch" size
    std::cerr << "Starting training " << size << " base estimators in " << args.threads << " threads ...\n";

    // Run learning in parallel
    ThreadPool tPool(args.threads);
    std::vector<std::future<Base*>> results;

    for (int i = 0; i < size; ++i)
        results.emplace_back(tPool.enqueue(trainBase, n, baseLabels[i], baseFeatures[i],
                                           (instancesWeights != nullptr) ? (*instancesWeights)[i] : nullptr, args));

    // Saving in the main thread
    saveResults(out, results);
}

void Model::trainBasesWithSameFeatures(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                                       std::vector<Feature*>& baseFeatures,
                                       std::vector<std::vector<double>*>* instancesWeights, Args& args) {
    std::ofstream out(outfile);
    int size = baseLabels.size();
    out.write((char*)&size, sizeof(size));
    trainBasesWithSameFeatures(out, n, baseLabels, baseFeatures, instancesWeights, args);
    out.close();
}

void Model::trainBasesWithSameFeatures(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                                       std::vector<Feature*>& baseFeatures,
                                       std::vector<std::vector<double>*>* instancesWeights, Args& args) {

    int size = baseLabels.size(); // This "batch" size
    std::cerr << "Starting training " << size << " base estimators in " << args.threads << " threads ...\n";

    // Run learning in parallel
    ThreadPool tPool(args.threads);
    std::vector<std::future<Base*>> results;

    for (int i = 0; i < size; ++i)
        results.emplace_back(tPool.enqueue(trainBase, n, baseLabels[i], baseFeatures,
                                           (instancesWeights != nullptr) ? (*instancesWeights)[i] : nullptr, args));

    // Saving in the main thread
    saveResults(out, results);
}

std::vector<Base*> Model::loadBases(std::string infile) {
    std::cerr << "Loading base estimators ...\n";

    std::vector<Base*> bases;

    std::ifstream in(infile);
    int size;
    in.read((char*)&size, sizeof(size));
    bases.reserve(size);
    for (int i = 0; i < size; ++i) {
        printProgress(i, size);
        bases.emplace_back(new Base());
        bases.back()->load(in);
    }
    in.close();

    return bases;
}
