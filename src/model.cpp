/*
 Copyright (c) 2019-2021 by Marek Wydmuch, Kalina Jasinska-Kobus

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

#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

#include "ensemble.h"
#include "log.h"
#include "measure.h"
#include "model.h"
#include "threads.h"

#include "br.h"
#include "hsm.h"
#include "online_plt.h"
#include "ovr.h"
#include "plt.h"
#include "svbop_full.h"
#include "svbop_inverted_index.h"
#include "svbop_hf.h"
#include "version.h"
#include "extreme_text.h"

// Mips extension models
#ifdef MIPS_EXT
#include "br_mips.h"
#include "svbop_mips.h"
#endif

std::shared_ptr<Model> Model::factory(Args& args) {
    std::shared_ptr<Model> model = nullptr;

    if (args.ensemble > 1) {
        switch (args.modelType) {
        case hsm: model = std::static_pointer_cast<Model>(std::make_shared<Ensemble<HSM>>()); break;
        case plt: model = std::static_pointer_cast<Model>(std::make_shared<Ensemble<BatchPLT>>()); break;
        default: throw std::invalid_argument("Ensemble is not supported for this model type");
        }
    } else {
        switch (args.modelType) {
        case ovr: model = std::static_pointer_cast<Model>(std::make_shared<OVR>()); break;
        case br: model = std::static_pointer_cast<Model>(std::make_shared<BR>()); break;
        case hsm: model = std::static_pointer_cast<Model>(std::make_shared<HSM>()); break;
        case plt: model = std::static_pointer_cast<Model>(std::make_shared<BatchPLT>()); break;
        case svbopFull: model = std::static_pointer_cast<Model>(std::make_shared<SVBOPFull>()); break;
        case svbopInvertedIndex: model = std::static_pointer_cast<Model>(std::make_shared<SVBOPInvertedIndex>()); break;
        case svbopFagin: model = std::static_pointer_cast<Model>(std::make_shared<SVBOPFagin>()); break;
        case svbopThreshold: model = std::static_pointer_cast<Model>(std::make_shared<SVBOPThreshold>()); break;
        case svbopHf: model = std::static_pointer_cast<Model>(std::make_shared<SVBOPHF>()); break;
        case oplt: model = std::static_pointer_cast<Model>(std::make_shared<OnlinePLT>()); break;
        case extremeText: model = std::static_pointer_cast<Model>(std::make_shared<ExtremeText>()); break;
#ifdef MIPS_EXT
        // Mips extension models
        case brMips: model = std::static_pointer_cast<Model>(std::make_shared<BRMIPS>()); break;
        case svbopMips: model = std::static_pointer_cast<Model>(std::make_shared<SVBOPMIPS>()); break;
#endif
        default: throw std::invalid_argument("Unknown model type");
        }
    }

    return model;
}

Model::Model():loaded(false), m(0) {}

Model::~Model() {
    unload();
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
    Log(CERR) << "Starting prediction in " << args.threads << " threads ...\n";

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
//    if(th.size() != m)
//        throw std::invalid_argument("Size of thresholds vector dose not match number of model outputs");
    thresholds = th;
}

void Model::updateThresholds(UnorderedMap<int, double> thToUpdate){
    for(auto& th : thToUpdate)
        thresholds[th.first] = th.second;
}

void Model::setLabelsWeights(std::vector<double> lw){
//    if(lw.size() != m)
//        throw std::invalid_argument("Size of labels' weights vector dose not match number of model outputs");
    labelsWeights = lw;
}

double Model::microOfo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args){
    double a = args.ofoA;
    double b = args.ofoB;

    Log(CERR) << "Optimizing Micro F measure for " << args.epochs << " epochs using " << args.threads << " threads ...\n";

    const int examples = features.rows() * args.epochs;
    for (int i = 0; i < examples; ++i) {
        printProgress(i, examples);
        int r = i % features.rows();

        // Predict with current thresholds
        std::vector<Prediction> prediction;
        args.threshold = a / b;
        predict(prediction, features[r], args);

        // Update a and b counters
        for (const auto &p : prediction) {
            for (int l = -1; labels[r][l] > -1; ++l)
                if (p.label == labels[r][l]) {
                    a++;
                    break;
                }
        }

        b += prediction.size() + labels.size(r);
    }

    return a / b;
}

std::vector<double> Model::macroOfo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args){
    // Variables required for OFO
    std::vector<double> as(m, args.ofoA);
    std::vector<double> bs(m, args.ofoB);
    thresholds = std::vector<double>(m, args.ofoA / args.ofoB);

    Log(CERR) << "Optimizing Macro F measure for " << args.epochs << " epochs using " << args.threads
              << " threads ...\n";

    // Set initial thresholds
    setThresholds(thresholds);

    ThreadSet tSet;
    int tRows = ceil(static_cast<double>(features.rows()) / args.threads);
    for (int t = 0; t < args.threads; ++t)
        tSet.add(macroOfoThread, t, this, std::ref(as), std::ref(bs), std::ref(features), std::ref(labels),
                 std::ref(args),
                 t * tRows, std::min((t + 1) * tRows, features.rows()));
    tSet.joinAll();

    return thresholds;
}

void Model::macroOfoThread(int threadId, Model* model, std::vector<double>& as, std::vector<double>& bs,
                      SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args, const int startRow, const int stopRow) {

    const int rowsRange = stopRow - startRow;
    const int examples = rowsRange * args.epochs;

    for (int i = 0; i < examples; ++i) {
        if (!threadId) printProgress(i, examples);
        int r = startRow + i % rowsRange;

        // Predict with current thresholds
        std::vector<Prediction> prediction;
        model->predict(prediction, features[r], args);

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
            if(labels[r][l] < bs.size()) bs[labels[r][l]]++;

        // Update thresholds, only those that may have changed due to update of as or bs,
        // For simplicity I compute some of them twice because it does not really matter
        UnorderedMap<int, double> thresholdsToUpdate;
        for (const auto& p : prediction)
            thresholdsToUpdate[p.label] = as[p.label] / bs[p.label];
        l = -1;
        while (labels[r][++l] > -1)
            if(labels[r][l] < bs.size()) thresholdsToUpdate[labels[r][l]] = as[labels[r][l]] / bs[labels[r][l]];

        model->updateThresholds(thresholdsToUpdate);
    }
}

std::vector<double> Model::ofo(SRMatrix<Feature>& features, SRMatrix<Label>& labels, Args& args) {

    args.topK = 0;
    args.threshold = 0;
    thresholds.clear();
    labelsWeights.clear();

    if(args.ofoType == macro)
        thresholds = macroOfo(features, labels, args);
    else if(args.ofoType == OFOType::micro)
        thresholds = std::vector<double>(m, microOfo(features, labels, args));
    else {
        std::vector<double> macroThr = macroOfo(features, labels, args);
        args.epochs = 1;
        double microThr = microOfo(features, labels, args);

        Log(CERR) << "Mixing thresholds for top " << args.ofoTopLabels << " labels ...\n";
        std::vector<Prediction> priors = computeLabelsPriors(labels);
        std::sort(priors.rbegin(), priors.rend());

        thresholds = std::vector<double>(m, microThr);
        for(int i = 0; i < args.ofoTopLabels; ++i)
            thresholds[priors[i].label] = macroThr[priors[i].label];
    }

    return thresholds;
}

Base* Model::trainBase(ProblemData& problemsData, Args& args) {
    Base* base = new Base();
    base->train(problemsData, args);
    return base;
}

void Model::trainBatchThread(std::vector<std::promise<Base *>>& results, std::vector<ProblemData>& problemsData, Args& args, int threadId, int threads) {
    size_t size = problemsData.size();
    for (int i = threadId; i < size; i += threads)
        results[i].set_value(trainBase(problemsData[i], args));
}

void Model::saveResults(std::ofstream& out, std::vector<std::future<Base*>>& results, bool saveGrads) {
    for (int i = 0; i < results.size(); ++i) {
        printProgress(i, results.size());
        Base* base = results[i].get();
        base->save(out, saveGrads);
        delete base;
    }
}

void Model::trainBases(std::string outfile, std::vector<ProblemData>& problemsData, Args& args) {
    std::ofstream out(outfile);
    int size = problemsData.size();
    out.write((char*)&size, sizeof(size));
    trainBases(out, problemsData, args);
    out.close();
}

void Model::trainBases(std::ofstream& out, std::vector<ProblemData>& problemsData, Args& args) {

    size_t size = problemsData.size(); // This "batch" size
    Log(CERR) << "Starting training " << size << " base estimators in " << args.threads << " threads ...\n";
    //Log(CERR) << "  Required memory: " << formatMem(args.threads * args.threads * n * sizeof(double)) << "\n";

    // Run learning in parallel
    if(args.threads > 1) {
        // Thread set solution
        ThreadSet tSet;
        std::vector<std::promise<Base *>> resultsPromise(size);
        std::vector<std::future<Base *>> results(size);
        for(int i = 0; i < size; ++i) results[i] = resultsPromise[i].get_future();
        for (int t = 0; t < args.threads; ++t)
            tSet.add(trainBatchThread, std::ref(resultsPromise), std::ref(problemsData), args, t, args.threads);

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
        saveResults(out, results, args.saveGrads);
        tSet.joinAll();
    } else {
        for (int i = 0; i < size; ++i){
            Base* base = new Base();
            base->train(problemsData[i], args);
            base->save(out, args.saveGrads);
            delete base;
        }
    }
}

std::vector<Base*> Model::loadBases(std::string infile, bool resume, bool loadDense) {
    Log(CERR) << "Loading base estimators ...\n";

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
        b->load(in, resume, loadDense);

        nonZeroSum += b->getNonZeroW();
        memSize += b->size();
        if(b->getMapW() != nullptr) ++sparse;
        bases.push_back(b);
    }
    in.close();

    Log(CERR) << "  Loaded bases: " << size
              << "\n  Bases size: " << formatMem(memSize) << "\n  Non zero weights / bases: " << nonZeroSum / size
              << "\n  Dense classifiers: " << size - sparse << "\n  Sparse classifiers: " << sparse << "\n";

    return bases;
}
