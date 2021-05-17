/*
 Copyright (c) 2018-2021 by Marek Wydmuch, Kalina Jasinska-Kobus

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
#include <iostream>
#include <random>

#include "base.h"
#include "linear.h"
#include "log.h"
#include "misc.h"
#include "threads.h"


//TODO: Refactor base class

Base::Base() {
    hingeLoss = false;

    wSize = 0;
    nonZeroW = 0;
    classCount = 0;
    firstClass = 0;
    firstClassCount = 0;
    t = 0;

    W = nullptr;
    G = nullptr;
    mapW = nullptr;
    mapG = nullptr;
    sparseW = nullptr;
}

Base::Base(Args& args): Base(){
    if(args.optimizerType != liblinear)
        setupOnlineTraining(args);
}

Base::~Base() { clear(); }

void Base::update(double label, Feature* features, Args& args) {
    std::lock_guard<std::mutex> lock(updateMtx);

    unsafeUpdate(label, features, args);
}

void Base::unsafeUpdate(double label, Feature* features, Args& args) {
    if (args.tmax != -1 && args.tmax < t) return;

    ++t;
    if (label == firstClass) ++firstClassCount;

    double pred = predictValue(features);
    double grad;
    if(args.lossType == logistic)
        grad = logisticGrad(label, pred, 0);
    else
        grad = squaredHingeGrad(label, pred, 0);

    if (args.optimizerType == sgd) {
        if (mapW != nullptr)
            updateSGD((*mapW), (*mapG), features, grad, t, args);
        else if (W != nullptr)
            updateSGD(W, G, features, grad, t, args);
    } else if (args.optimizerType == adagrad) {
        if (mapW != nullptr)
            updateAdaGrad((*mapW), (*mapG), features, grad, t, args);
        else if (W != nullptr)
            updateAdaGrad(W, G, features, grad, t, args);
    }
    else throw std::invalid_argument("Unknown optimizer type");

    // Check if we should change sparse W to dense W
//    if (mapW != nullptr && wSize != 0) {
//        nonZeroW = mapW->size();
//        if (mapSize() > denseSize()) toDense();
//    }
}

void Base::trainLiblinear(ProblemData& problemData, Args& args) {
    double cost = args.cost;
    if (args.autoCLog)
        cost *= 1.0 + log(static_cast<double>(problemData.r) / problemData.binFeatures.size());
    if (args.autoCLin)
        cost *= static_cast<double>(problemData.r) / problemData.binFeatures.size();

    problem P = {.l = static_cast<int>(problemData.binLabels.size()),
                 .n = problemData.n,
                 .y = problemData.binLabels.data(),
                 .x = problemData.binFeatures.data(),
                 .bias = -1,
                 .W = problemData.instancesWeights.data()};

    parameter C = {.solver_type = args.solverType,
                   .eps = args.eps,
                   .C = cost,
                   .nr_weight = problemData.labelsCount,
                   .weight_label = problemData.labels,
                   .weight = problemData.labelsWeights,
                   .p = 0,
                   .init_sol = NULL,
                   .max_iter = args.maxIter};

    auto output = check_parameter(&P, &C);
    assert(output == NULL);

    model* M = train_liblinear(&P, &C);

    assert(M->nr_class <= 2);
    assert(M->nr_feature == problemData.n);

    // Set base's attributes
    wSize = problemData.n + 1;
    firstClass = M->label[0];
    classCount = M->nr_class;

    // Copy weights
    W = new Weight[wSize];
    W[0] = 0;
    for (int i = 0; i < problemData.n; ++i) W[i + 1] = M->w[i]; // Shift by -1

    hingeLoss = args.solverType == L2R_L2LOSS_SVC_DUAL || args.solverType == L2R_L2LOSS_SVC ||
                args.solverType == L2R_L1LOSS_SVC_DUAL || args.solverType == L1R_L2LOSS_SVC;

    // Delete LibLinear model
    free_model_content(M);
    free(M);
}

void Base::trainOnline(ProblemData& problemData, Args& args) {
    setupOnlineTraining(args, problemData.n, true);

    // Set loss function
    double (*lossFunc)(double, double, double);
    double (*gradFunc)(double, double, double);
    if (args.lossType == logistic) {
        lossFunc = &logisticLoss;
        gradFunc = &logisticGrad;
    }
    else if (args.lossType == squaredHinge) {
        gradFunc = &squaredHingeGrad;
        hingeLoss = true;
    }
    else if (args.lossType == unLogistic) {
        lossFunc = &unbiasedLogisticLoss;
        gradFunc = &unbiasedLogisticGrad;
    }
    else if (args.lossType == pwLogistic) {
        lossFunc = &pwLogisticLoss;
        gradFunc = &pwLogisticGrad;
        //lossFunc = &asLoss;
        //gradFunc = &asGrad;
    }
    else
        throw std::invalid_argument("Unknown loss function type");

    // Set update function
    void (*updateFunc)(Weight*&, Weight*&, Feature*, double, int, Args&);
    if(args.optimizerType == sgd)
        updateFunc = &updateSGD<Weight*>;
    else if (args.optimizerType == adagrad)
        updateFunc = &updateAdaGrad<Weight*>;
    else
        throw std::invalid_argument("Unknown online update function type");

    const int examples = problemData.binFeatures.size();
    double loss = 0;
    for (int e = 0; e < args.epochs; ++e)
        for (int r = 0; r < examples; ++r) {
            double label = problemData.binLabels[r];
            Feature* features = problemData.binFeatures[r];

            if (args.tmax != -1 && args.tmax < t) break;

            ++t;
            if (problemData.binLabels[r] == firstClass) ++firstClassCount;

            double pred = dotVectors(features, W, wSize);
            //if (pred > 10 || pred < -10) continue;
            double grad = gradFunc(label, pred, problemData.invPs) * problemData.instancesWeights[r];
            if (!std::isinf(grad) && !std::isnan(grad)) updateFunc(W, G, features, grad, t, args);

            // Report loss
//            loss += lossFunc(label, pred, problemData.invPs);
//            int iter = e * examples + r;
//            if(iter % 1000 == 999)
//                Log(CERR) << "  Iter: " << iter << "/" << args.epochs * examples << ", loss: " << loss / iter << "\n";
        }

    finalizeOnlineTraining(args);
}

void Base::train(ProblemData& problemData, Args& args) {

    if (problemData.binLabels.empty()) {
        firstClass = 0;
        classCount = 0;
        return;
    }

    assert(problemData.binLabels.size() == problemData.binFeatures.size());
    assert(problemData.instancesWeights.size() >= problemData.binLabels.size());

    int positiveLabels = std::count(problemData.binLabels.begin(), problemData.binLabels.end(), 1.0);
    if (positiveLabels == 0 || positiveLabels == problemData.binLabels.size()) {
        firstClass = static_cast<int>(problemData.binLabels[0]);
        classCount = 1;
        return;
    }

    // Apply some weighting for very unbalanced data
    if (args.inbalanceLabelsWeighting) {
        problemData.labelsCount = 2;
        problemData.labels = new int[2];
        problemData.labels[0] = 0;
        problemData.labels[1] = 1;
        problemData.labelsWeights = new double[2];

        int negativeLabels = static_cast<int>(problemData.binLabels.size()) - positiveLabels;
        if (negativeLabels > positiveLabels) {
            problemData.labelsWeights[0] = 1.0;
            problemData.labelsWeights[1] = 1.0 + log(static_cast<double>(negativeLabels) / positiveLabels);
        } else {
            problemData.labelsWeights[0] = 1.0 + log(static_cast<double>(positiveLabels) / negativeLabels);
            problemData.labelsWeights[1] = 1.0;
        }
    }

    if (args.optimizerType == liblinear) trainLiblinear(problemData, args);
    else trainOnline(problemData, args);

    // TODO?: Calculate final training loss

    // Apply threshold and calculate number of non-zero weights
    pruneWeights(args.weightsThreshold);
    if (sparseSize(nonZeroW) < denseSize(wSize)) toSparse();

    delete[] problemData.labels;
    delete[] problemData.labelsWeights;
}

void Base::setupOnlineTraining(Args& args, int n, bool startWithDenseW) {
    wSize = n;
    if (wSize != 0 && startWithDenseW) {
        W = new Weight[wSize];
        std::memset(W, 0, wSize * sizeof(Weight));
        if (args.optimizerType == adagrad) {
            G = new Weight[wSize];
            std::memset(G, 0, wSize * sizeof(Weight));
        }
    } else {
        mapW = new UnorderedMap<int, Weight>();
        if (args.optimizerType == adagrad) mapG = new UnorderedMap<int, Weight>();
    }

    classCount = 2;
    firstClass = 1;
    t = 0;
}

void Base::finalizeOnlineTraining(Args& args) {
    // Because aux bases needs previous weights, TODO: Change this later
    /*
    if (firstClassCount == t || firstClassCount == 0) {
        classCount = 1;
        if (firstClassCount == 0) firstClass = 1 - firstClass;
    }
    */
    if(mapW != nullptr)
        nonZeroW = mapW->size();
    else
        nonZeroW = wSize;
    nonZeroG = nonZeroW;
    pruneWeights(args.weightsThreshold);
}

double Base::predictValue(Feature* features) {
    if (classCount < 2) return static_cast<double>((1 - 2 * firstClass) * -10);
    double val = 0;

    if (mapW) { // Sparse features dot sparse weights in hash map
        Feature* f = features;
        while (f->index != -1) {
            auto w = mapW->find(f->index);
            if (w != mapW->end()) val += w->second * f->value;
            ++f;
        }
    } else if (W)
        val = dotVectors(features, W, wSize); // Sparse features dot dense weights
    else
        throw std::runtime_error("Prediction using sparse features and sparse weights is not supported!");

    if (firstClass == 0) val *= -1;

    return val;
}

double Base::predictProbability(Feature* features) {
    double val = predictValue(features);
    if (hingeLoss)
        //val = 1.0 / (1.0 + std::exp(-2 * val)); // Probability for squared Hinge loss solver
        val = std::exp(-std::pow(std::max(0.0, 1.0 - val), 2));
    else
        val = 1.0 / (1.0 + std::exp(-val)); // Probability
    return val;
}

void Base::forEachW(const std::function<void(Weight&)>& func) {
    if (W != nullptr)
        for (int i = 0; i < wSize; ++i) func(W[i]);
    else if (mapW != nullptr)
        for (auto& w : *mapW) func(w.second);
    else if (sparseW != nullptr)
        for (int i = 0; i < nonZeroW; ++i) func(sparseW[i].second);
}

void Base::forEachG(const std::function<void(Weight&)>& func) {
    if (G != nullptr)
        for (int i = 0; i < wSize; ++i) func(G[i]);
    else if (mapG != nullptr)
        for (auto& w : *mapG) func(w.second);
//    else if (sparseG != nullptr)
//        for (int i = 0; i < nonZeroW; ++i) func(sparseG[i].second);
}

void Base::forEachIW(const std::function<void(const int&, Weight&)>& func) {
    if (W != nullptr)
        for (int i = 0; i < wSize; ++i) func(i, W[i]);
    else if (mapW != nullptr)
        for (auto& w : *mapW) func(w.first, w.second);
    else if (sparseW != nullptr)
        for (int i = 0; i < nonZeroW; ++i) func(sparseW[i].first, sparseW[i].second);
}

void Base::forEachIG(const std::function<void(const int&, Weight&)>& func) {
    if (G != nullptr)
        for (int i = 0; i < wSize; ++i) func(i, G[i]);
    else if (mapG != nullptr)
        for (auto& w : *mapG) func(w.first, w.second);
//    else if (sparseG != nullptr)
//        for (int i = 0; i < nonZeroW; ++i) func(sparseG[i].first, sparseW[i].second);
}


void Base::clear() {
    hingeLoss = false;

    wSize = 0;
    nonZeroW = 0;
    classCount = 0;
    firstClass = 0;
    firstClassCount = 0;
    t = 0;

    clearW();
}

void Base::clearW() {
    delete[] W;
    W = nullptr;
    delete[] G;
    G = nullptr;

    delete mapW;
    mapW = nullptr;
    delete mapG;
    mapG = nullptr;

    delete[] sparseW;
    sparseW = nullptr;
}

void Base::toMap() {
    if (mapW == nullptr) {
        mapW = new UnorderedMap<int, Weight>();

        assert(W != nullptr);
        for (int i = 0; i < wSize; ++i)
            if (W[i] != 0) mapW->insert({i, W[i]});
        delete[] W;
        W = nullptr;
    }

    if (mapG == nullptr && G != nullptr) {
        mapG = new UnorderedMap<int, Weight>();

        for (int i = 0; i < wSize; ++i)
            if (G[i] != 0) mapG->insert({i, W[i]});
        delete[] G;
        G = nullptr;
    }
}

void Base::toDense() {
    if (W == nullptr) {
        W = new Weight[wSize];
        std::memset(W, 0, wSize * sizeof(Weight));
        assert(mapW != nullptr);
        for (const auto& w : *mapW) W[w.first] = w.second;
        delete mapW;
        mapW = nullptr;
    }

    if (G == nullptr && mapG != nullptr) {
        G = new Weight[wSize];
        std::memset(G, 0, wSize * sizeof(Weight));

        for (const auto& w : *mapG) G[w.first] = w.second;
        delete mapG;
        mapG = nullptr;
    }
}

void Base::toSparse() {
    if (sparseW == nullptr) {
        auto tmpSparseW = new SparseWeight[nonZeroW];
        auto sW = tmpSparseW;

        forEachIW([&](const int& i, Weight& w) {
            if (w != 0) {
                sW->first = i;
                sW->second = w;
                ++sW;
            }
        });

        clearW();
        sparseW = tmpSparseW;
    }
}

void Base::pruneWeights(double threshold) {
    nonZeroW = 0;

    forEachIW([&](const int& i, Weight& w) {
        if (i == 1 || (w != 0 && fabs(w) >= threshold)) ++nonZeroW; // Do not prune bias feature
        else w = 0;
    });
}

void Base::save(std::ostream& out, bool saveGrads) {
    out.write((char*)&classCount, sizeof(classCount));
    out.write((char*)&firstClass, sizeof(firstClass));

    if (classCount > 1) {
        // Decide on optimal file coding

        out.write((char*)&hingeLoss, sizeof(hingeLoss));
        out.write((char*)&wSize, sizeof(wSize));
        out.write((char*)&nonZeroW, sizeof(nonZeroW));

        if(W != nullptr) saveVec(out, W, wSize, nonZeroW);
        else if(mapW != nullptr) saveVec(out, mapW, wSize, nonZeroW);
        else if(sparseW != nullptr) saveVec(out, sparseW, wSize, nonZeroW);

        bool grads = (saveGrads && (G != nullptr || mapG != nullptr));
        saveVar(out, grads);
        if(grads){
            if (G != nullptr) saveVec(out, G, wSize, nonZeroG);
            else if (mapG != nullptr) saveVec(out, mapG, wSize, nonZeroG);
        }

//        Log(CERR) << "  Save base: classCount: " << classCount << ", firstClass: "
//                  << firstClass << ", weights: " << nonZeroW << "/" << wSize << "\n";
    }
}

void Base::load(std::istream& in, bool loadGrads, bool loadDense) {
    in.read((char*)&classCount, sizeof(classCount));
    in.read((char*)&firstClass, sizeof(firstClass));

    if (classCount > 1) {
        in.read((char*)&hingeLoss, sizeof(hingeLoss));
        in.read((char*)&wSize, sizeof(wSize));
        in.read((char*)&nonZeroW, sizeof(nonZeroW));

        //TODO: Improve this
        bool loadSparse = (!loadDense && (wSize == 0 || (mapSize(nonZeroW) < denseSize(wSize) && wSize > 50000)));
        if(loadSparse) mapW = loadAsMap(in);
        else W = loadAsDense(in);

        bool grads;
        loadVar(in, grads);
        if(grads) {
            if(loadGrads) {
                if (loadSparse) mapG = loadAsMap(in);
                else G = loadAsDense(in);
            }
            else skipLoadVec(in);
        }

//        Log(CERR) << "  Load base: classCount: " << classCount << ", firstClass: "
//                  << firstClass << ", weights: " << nonZeroW << "/" << wSize << "\n";
    }
}

size_t Base::size() {
    size_t size = sizeof(Base);
    if (W) size += denseSize(wSize);
    if (mapW) size += mapSize(mapW->size());
    if (sparseW) size += sparseSize(nonZeroW);
    if (G) size += denseSize(wSize);
    if (mapG) size += mapSize(mapG->size());
    return size;
}

void Base::printWeights() {
    forEachIW([&](const int& i, Weight& w) { Log(CERR) << i << ":" << w << " "; });
    Log(CERR) << "\n";
}

void Base::invertWeights() {
    forEachW([&](Weight& w) { w *= -1; });
}

void Base::setFirstClass(int first){
    if(firstClass != first){
        invertWeights();
        firstClass = first;
    }
}

Base* Base::copy() {
    Base* copy = new Base();
    if (W) {
        copy->W = new Weight[wSize];
        std::memcmp(copy->W, W, wSize * sizeof(Weight));
    }
    if (G) {
        copy->G = new Weight[wSize];
        std::memcmp(copy->G, G, wSize * sizeof(Weight));
    }

    if (mapW) copy->mapW = new UnorderedMap<int, Weight>(mapW->begin(), mapW->end());
    if (mapG) copy->mapG = new UnorderedMap<int, Weight>(mapG->begin(), mapG->end());

    if (sparseW) {
        copy->sparseW = new SparseWeight[nonZeroW];
        std::memcmp(copy->sparseW, sparseW, (nonZeroW) * sizeof(SparseWeight));
    }

    copy->firstClass = firstClass;
    copy->classCount = classCount;
    copy->wSize = wSize;
    copy->nonZeroW = nonZeroW;

    return copy;
}

Base* Base::copyInverted() {
    Base* c = copy();
    c->invertWeights();
    return c;
}

void Base::saveVecHeader(std::ostream& out, bool sparse, size_t size, size_t nonZero) {
    saveVar(out, sparse);
    saveVar(out, size);
    saveVar(out, nonZero);
}

void Base::saveVec(std::ostream& out, Weight* V, size_t size, size_t nonZero){
    bool sparse = sparseSize(nonZero) < denseSize(size) || size == 0;
    saveVecHeader(out, sparse, size, nonZero);

    if(sparse){
        int saved = 0;
        for (int i = 0; i < size; ++i){
            if (V[i] != 0) {
                saveVar(out, i);
                saveVar(out, V[i]);
                ++saved;
            }
        }
        assert(saved == nonZero);
    } else out.write((char*)V, size * sizeof(Weight));
}

void Base::saveVec(std::ostream& out, SparseWeight* V, size_t size, size_t nonZero){
    saveVecHeader(out, true, size, nonZero);
    for (int i = 0; i < nonZero; ++i) saveVar(out, V[i]);
}

void Base::saveVec(std::ostream& out, UnorderedMap<int, Weight>* mapV, size_t size, size_t nonZero){
    saveVecHeader(out, true, size, mapV->size());
    for(const auto& w : (*mapV)) saveVar(out, w);
}

Weight* Base::loadAsDense(std::istream& in){
    bool sparse;
    loadVar(in, sparse);

    size_t size;
    loadVar(in, size);

    size_t nonZero;
    loadVar(in, nonZero);
    //std::cerr << "Dense: " << sparse <<  " " << size << " " << nonZero << "\n";

    Weight *V = new Weight[size];
    if(sparse) {
        std::memset(V, 0, size * sizeof(Weight));

        int index;
        Weight value;

        for(int i = 0; i < nonZero; ++i){
            loadVar(in, index);
            loadVar(in, value);
            V[index] = value;
        }
    } else in.read((char *) V, size * sizeof(Weight));

    return V;
}

UnorderedMap<int, Weight>* Base::loadAsMap(std::istream& in){
    bool sparse;
    loadVar(in, sparse);

    size_t size;
    loadVar(in, size);

    size_t nonZero;
    loadVar(in, nonZero);
    //std::cerr << "Map: " << sparse <<  " " << size << " " << nonZero << "\n";

    auto mapV = new UnorderedMap<int, Weight>();
    mapV->reserve(nonZero);

    if(sparse) {
        int index;
        Weight value;

        for (int i = 0; i < nonZero; ++i) {
            loadVar(in, index);
            loadVar(in, value);
            mapV->insert({index, value});
        }
    } else {
        Weight value;
        for (int i = 0; i < size; ++i) {
            loadVar(in, value);
            if(value != 0) mapV->insert({i, value});
        }
    }

    return mapV;
}

void Base::skipLoadVec(std::istream& in){
    bool sparse;
    loadVar(in, sparse);

    size_t size;
    loadVar(in, size);

    size_t nonZero;
    loadVar(in, nonZero);

    if(sparse) in.seekg(nonZero * (sizeof(int) + sizeof(Weight)), std::ios::cur);
    else in.seekg(size * sizeof(Weight), std::ios::cur);
}

