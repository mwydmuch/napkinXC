/**
 * Copyright (c) 2019 by Marek Wydmuch
 * Copyright (c) 2020 by Marek Wydmuch, Kalina Kobus
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
#include "ubop.h"
#include "ubop_hsm.h"
#include "version.h"
#include "extreme_text.h"

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
        case ubopHsm: model = std::static_pointer_cast<Model>(std::make_shared<UBOPHSM>()); break;
        case oplt: model = std::static_pointer_cast<Model>(std::make_shared<OnlinePLT>()); break;
        case extremeText: model = std::static_pointer_cast<Model>(std::make_shared<ExtremeText>()); break;
#ifdef MIPS_EXT
        // Mips extension models
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

void Model::predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, Args& args) {
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

void Model::setThresholds(std::vector<double> th){
    thresholds = th;
}

void Model::updateThresholds(UnorderedMap<int, double> thToUpdate){
    for(auto& th : thToUpdate)
        thresholds[th.first] = th.second;
}

void Model::predictBatchWithThresholdsThread(int threadId, Model* model, std::vector<std::vector<Prediction>>& predictions,
                                             SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow) {
    const int batchSize = stopRow - startRow;
    for (int r = startRow; r < stopRow; ++r) {
        int i = r - startRow;
        model->predictWithThresholds(predictions[r], features[r], args);
        if (!threadId) printProgress(i, batchSize);
    }
}

std::vector<std::vector<Prediction>> Model::predictBatchWithThresholds(SRMatrix<Feature>& features, Args& args) {
    std::cerr << "Starting prediction in " << args.threads << " threads ...\n";

    int rows = features.rows();
    std::vector<std::vector<Prediction>> predictions(rows);

    // Run prediction in parallel using thread set
    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(rows) / args.threads);
    for (int t = 0; t < args.threads; ++t)
        tSet.add(predictBatchWithThresholdsThread, t, this, std::ref(predictions), std::ref(features),
                 std::ref(args), t * tRows, std::min((t + 1) * tRows, rows));
    tSet.joinAll();

    return predictions;
}

void Model::ofoThread(int threadId, Model* model, std::vector<double>& as, std::vector<double>& bs,
                      SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args, const int startRow, const int stopRow) {

    const int rowsRange = stopRow - startRow;
    const int examples = rowsRange * args.epochs;

    for (int i = 0; i < examples; ++i) {
        if (!threadId) printProgress(i, examples);
        int r = startRow + i % rowsRange;

        // Predict with current thresholds
        std::vector<Prediction> prediction;
        model->predictWithThresholds(prediction, features[r], args);

        // Update a and b counters
        for (const auto& p : prediction) {
            // b[j] =  sum_{i = 1}^{t} \hat y_j + ..
            bs[p.label]++;

            // a[j] = sum_{i = 1}^{t} y_j \hat y_j
            int l = -1;
            while (labels[r][++l] > -1)
                if (p.label == labels[r][l]) {
                    as[p.label]++;
                    break;
                }
        }

        // b[j] =  .. + sum_{i = 1}^{t} y_j
        int l = -1;
        while (labels[r][++l] > -1)
            bs[labels[r][l]]++;

        // Update thresholds, only those that may have changed due to update of as or bs,
        // For simplicity I compute some of them twice because it does not really matter
        UnorderedMap<int, double> thresholdsToUpdate;
        for (const auto& p : prediction)
            thresholdsToUpdate[p.label] = as[p.label] / bs[p.label];
        l = -1;
        while (labels[r][++l] > -1)
            thresholdsToUpdate[labels[r][l]] = as[labels[r][l]] / bs[labels[r][l]];

        model->updateThresholds(thresholdsToUpdate);
    }
}

std::vector<double> Model::ofo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args) {

    double thresholdEps = 0.00001;

    // Variables required for OFO
    std::vector<double> as(m, 0);
    std::vector<double> bs(m, 0);
    thresholds = std::vector<double>(m, 1.0);

    std::vector<Feature> storedThresholds;

    // OFO Bootstrap
    if(args.ofoBootstrap) {
        std::cerr << "Bootstrapping OFO ...\n";
        const int rows = features.rows();
        for (int r = 0; r < rows; ++r) {
            printProgress(r, rows);

            int l = -1;
            while (labels[r][++l] > -1) {
                as[labels[r][l]] += predictForLabel(labels[r][l], features[r], args);
                ++bs[labels[r][l]];
            }
        }

        for (int i = 0; i < m; ++i) {
            if(bs[i] > args.ofoBootstrapMin) {
                as[i] = as[i] / bs[i] * args.ofoBootstrapScale;
                bs[i] = args.ofoBootstrapScale;
                thresholds[i] = as[i] / bs[i];
            } else {
                if(args.ofoBootstrapMin == 1 && bs[i] == 1) storedThresholds.push_back({i, as[i] - thresholdEps});
                else storedThresholds.push_back({i, 0.5});
            }
        }

    } else {
        as = std::vector<double>(m, args.ofoA);
        bs = std::vector<double>(m, args.ofoB);
        thresholds = std::vector<double>(m, args.ofoA / args.ofoB);
    }

    std::cerr << "Optimizing thresholds with OFO for " << args.epochs << " epochs using " << args.threads << " threads ...\n";

    // Set initial thresholds
    setThresholds(thresholds);

    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(features.rows()) / args.threads);
    for (int t = 0; t < args.threads; ++t)
        tSet.add(ofoThread, t, this, std::ref(as), std::ref(bs), std::ref(features), std::ref(labels), std::ref(args),
                 t * tRows, std::min((t + 1) * tRows, features.rows()));
    tSet.joinAll();

    // Apply stored thresholds
    for(auto& st : storedThresholds)
        thresholds[st.index] = st.value;

    return thresholds;
}

void Model::macroFSearchThread(Model* model, std::vector<std::vector<int>>& buckets, std::vector<std::vector<double>>& trueP,
                        SRMatrix<Feature>& features, Args& args, int threadId, int threads){

    const int rows = features.rows();
    for (int r = threadId; r < rows; r += threads){
        if (!threadId) printProgress(r, rows);

        std::vector<Prediction> prediction;
        model->predictWithThresholds(prediction, features[r], args);

        for (const auto& p : prediction) {
            for(int b = 0; b < buckets.size(); ++b){
                if(p.value < trueP[p.label][b + 1]){
                    ++buckets[p.label][b];
                    break;
                }
            }
        }
    }
}

std::vector<double> Model::macroFSearch(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args){
    std::cerr << "Optimizing thresholds using " << args.threads << " thread ...\n";

    double thresholdEps = 0.00001;

    std::cerr << "  Getting predictions for true labels ...\n";
    std::vector<std::vector<double>> trueP(m);
    const int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        int l = -1;
        while (labels[r][++l] > -1)
            trueP[labels[r][l]].push_back(predictForLabel(labels[r][l], features[r], args));
    }

    std::vector<Feature> storedThresholds;
    thresholds = std::vector<double>(m, 0);
    std::vector<std::vector<int>> buckets(m);
    for(int i = 0; i < m; ++i){
        if(trueP[i].size() > args.ofoBootstrapMin) {
            std::sort(trueP[i].begin(), trueP[i].end());
            trueP[i].push_back(1.0);
            buckets[i].resize(trueP[i].size() - 1, 0);
            thresholds[i] = trueP[i][0];
        } else {
            if(args.ofoBootstrapMin == 1 && trueP[i].size() == 1)
                storedThresholds.push_back({i, trueP[i][0] - thresholdEps});
            else storedThresholds.push_back({i, 0.5});
        }
    }

    std::cerr << "  Predicting in " << args.threads << " threads ...\n";
    setThresholds(thresholds);
    ThreadSet tSet;
    for (int t = 0; t < args.threads; ++t)
        tSet.add(macroFSearchThread, this, std::ref(buckets), std::ref(trueP), std::ref(features), std::ref(args), t, args.threads);
    tSet.joinAll();

    std::cerr << "  Calculating optimal thresholds ...\n";
    for(int i = 0; i < m; ++i){
        printProgress(i, rows);
        double bestF1 = 0;
        double bestThr = 0;
        for(int j = 0; j < trueP[i].size() - 1; ++j){
            double tp = trueP[i].size() - j;
            double f = std::accumulate(buckets[i].begin() + j, buckets[i].end(), 0) - tp;
            double F1 = 2 * tp / (2 * tp + f);
            if(F1 > bestF1){
                bestF1 = F1;
                bestThr = trueP[i][j] - thresholdEps;
            } else break;
        }
        thresholds[i] = bestThr;
    }

    for(auto& st : storedThresholds)
        thresholds[st.index] = st.value;

    return thresholds;
}

Base* Model::trainBase(int n, std::vector<double>& baseLabels, std::vector<Feature*>& baseFeatures,
                       std::vector<double>* instancesWeights, Args& args) {
    Base* base = new Base();
    base->train(n, baseLabels, baseFeatures, instancesWeights, args);
    return base;
}

void Model::trainBatchThread(int n, std::vector<std::promise<Base *>>& results, std::vector<std::vector<double>>& baseLabels,
                             std::vector<std::vector<Feature*>>& baseFeatures,
                             std::vector<std::vector<double>*>* instancesWeights, Args& args, int threadId, int threads) {

    size_t size = baseLabels.size();
    for (int i = threadId; i < size; i += threads)
        results[i].set_value(trainBase(n, baseLabels[i], baseFeatures[i],
                                   (instancesWeights != nullptr) ? (*instancesWeights)[i] : nullptr, args));
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

    size_t size = baseLabels.size(); // This "batch" size
    std::cerr << "Starting training " << size << " base estimators in " << args.threads << " threads ...\n";
    std::cerr << "  Required memory: " << formatMem(args.threads * args.threads * n * sizeof(double)) << std::endl;

    // Run learning in parallel
    if(args.threads > 1) {
        // Thread set solution
        ThreadSet tSet;
        std::vector<std::promise<Base *>> resultsPromise(size);
        std::vector<std::future<Base *>> results(size);
        for(int i = 0; i < size; ++i) results[i] = resultsPromise[i].get_future();
        for (int t = 0; t < args.threads; ++t)
            tSet.add(trainBatchThread, n, std::ref(resultsPromise), std::ref(baseLabels), std::ref(baseFeatures), instancesWeights, args, t, args.threads);

        // Thread pool solution is slower
        /*
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base *>> results;
        results.reserve(size);
        for (int i = 0; i < size; ++i)
            results.emplace_back(tPool.enqueue(trainBase, n, std::ref(baseLabels[i]), std::ref(baseFeatures[i]),
                                               (instancesWeights != nullptr) ? (*instancesWeights)[i] : nullptr, args));
        */

        // Saving in the main thread
        saveResults(out, results);
        tSet.joinAll();
    } else {
        for (int i = 0; i < size; ++i){
            Base* base = new Base();
            base->train(n, baseLabels[i], baseFeatures[i], (instancesWeights != nullptr) ? (*instancesWeights)[i] : nullptr, args);
            base->save(out);
            delete base;
        }
    }
}

void Model::trainBatchWithSameFeaturesThread(int n, std::vector<std::promise<Base *>>& results,
                                             std::vector<std::vector<double>>& baseLabels,
                                             std::vector<Feature*>& baseFeatures,
                                             std::vector<double>* instancesWeights, Args& args, int threadId, int threads){
    size_t size = baseLabels.size();
    for(int i = threadId; i < size; i += threads)
        results[i].set_value(trainBase(n, baseLabels[i], baseFeatures, instancesWeights, args));
}

void Model::trainBasesWithSameFeatures(std::string outfile, int n, std::vector<std::vector<double>>& baseLabels,
                                       std::vector<Feature*>& baseFeatures,
                                       std::vector<double>* instancesWeights, Args& args) {
    std::ofstream out(outfile);
    int size = baseLabels.size();
    out.write((char*)&size, sizeof(size));
    trainBasesWithSameFeatures(out, n, baseLabels, baseFeatures, instancesWeights, args);
    out.close();
}

void Model::trainBasesWithSameFeatures(std::ofstream& out, int n, std::vector<std::vector<double>>& baseLabels,
                                       std::vector<Feature*>& baseFeatures,
                                       std::vector<double>* instancesWeights, Args& args) {

    int size = baseLabels.size(); // This "batch" size
    std::cerr << "Starting training " << size << " base estimators in " << args.threads << " threads ...\n";

    // Run learning in parallel
    if(args.threads > 1) {
        // Thread set solution
        ThreadSet tSet;
        std::vector<std::promise<Base *>> resultsPromise(size);
        std::vector<std::future<Base *>> results(size);
        for(int i = 0; i < size; ++i) results[i] = resultsPromise[i].get_future();
        for (int t = 0; t < args.threads; ++t)
            tSet.add(trainBatchWithSameFeaturesThread, n, std::ref(resultsPromise), std::ref(baseLabels), std::ref(baseFeatures), instancesWeights, args, t, args.threads);

        // Thread pool solution is slower
        /*
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base *>> results;
        results.reserve(size);

        for (int i = 0; i < size; ++i)
            results.emplace_back(tPool.enqueue(trainBase, n, std::ref(baseLabels[i]), std::ref(baseFeatures), instancesWeights, args));
        */

        // Saving in the main thread
        saveResults(out, results);
        tSet.joinAll();
    } else {
        for (int i = 0; i < size; ++i){
            Base* base = new Base();
            base->train(n, baseLabels[i], baseFeatures, instancesWeights, args);
            base->save(out);
            delete base;
        }
    }
}

std::vector<Base*> Model::loadBases(std::string infile) {
    std::cerr << "Loading base estimators ...\n";

    double nonZeroSum = 0;
    unsigned long long memSize = 0;
    int sparse = 0;

    std::vector<Base*> bases;
    std::ifstream in(infile);
    int size;
    in.read((char*)&size, sizeof(size));
    bases.reserve(size);
    for (int i = 0; i < size; ++i) {
        printProgress(i, size);
        auto b = new Base();
        b->load(in);
        nonZeroSum += b->getNonZeroW();
        memSize += b->size();
        if(b->getMapW() != nullptr) ++sparse;
        bases.push_back(b);
    }
    in.close();

    std::cerr << "  Loaded bases: " << size
              << "\n  Bases size: " << formatMem(memSize) << "\n  Non zero weights / bases: " << nonZeroSum / size
              << "\n  Dense classifiers: " << size - sparse << "\n  Sparse classifiers: " << sparse << std::endl;

    return bases;
}
