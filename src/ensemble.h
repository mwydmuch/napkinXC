/*
 Copyright (c) 2019-2021 by Marek Wydmuch

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

#pragma once

#include <unordered_map>
#include <unordered_set>

#include "log.h"
#include "model.h"

struct EnsemblePrediction {
    int label;
    Real value;
    std::vector<int> members;

    bool operator<(const EnsemblePrediction& r) const { return value < r.value; }
};


template <typename T> class Ensemble : public Model {
public:
    Ensemble();
    ~Ensemble() override;

    void train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) override;
    Real predictForLabel(Label label, SparseVector& features, Args& args) override;
    std::vector<std::vector<Prediction>> predictBatch(SRMatrix& features, Args& args) override;

    void setThresholds(std::vector<Real> th) override;
    void setLabelsWeights(std::vector<Real> lw) override;
    void setLabelsBiases(std::vector<Real> lb) override;

    void load(Args& args, std::string infile) override;
    void unload() override;

    void printInfo() override;

protected:
    std::vector<T*> members;
    T* loadMember(Args& args, const std::string& infile, int memberNo);
    void accumulatePrediction(UnorderedMap<int, EnsemblePrediction>& ensemblePredictions,
                              std::vector<Prediction>& prediction, int memberNo);

    void accumulatePrediction(UnorderedMap<int, Prediction>& ensemblePredictions,
                              std::vector<Prediction>& prediction);
};


template <typename T> Ensemble<T>::Ensemble() {}

template <typename T> Ensemble<T>::~Ensemble() {
    for (auto& m : members) delete m;
}

template <typename T>
void Ensemble<T>::train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) {
    Log(CERR) << "Training ensemble of " << args.ensemble << " models ...\n";
    Log::updateGlobalIndent(2);
    for (int i = 0; i < args.ensemble; ++i) {
        Log(CERR) << "Training ensemble " << i << " ...\n";
        Log::updateGlobalIndent(2);
        std::string memberDir = joinPath(output, "member_" + std::to_string(i));
        makeDir(memberDir);
        T* member = new T();
        member->train(labels, features, args, memberDir);
        delete member;
        Log::updateGlobalIndent(-2);
    }
    Log::updateGlobalIndent(-2);
}

template <typename T>
void Ensemble<T>::accumulatePrediction(UnorderedMap<int, EnsemblePrediction>& ensemblePredictions,
                                       std::vector<Prediction>& prediction, int memberNo) {
    for (auto& mP : prediction) {
        auto ensP = ensemblePredictions.find(mP.label);
        if (ensP != ensemblePredictions.end()) {
            ensP->second.value += mP.value;
            ensP->second.members.push_back(memberNo);
        } else
            ensemblePredictions[mP.label] = {mP.label, mP.value, {memberNo}};
    }
}

template <typename T>
void Ensemble<T>::accumulatePrediction(UnorderedMap<int, Prediction>& ensemblePredictions,
                                       std::vector<Prediction>& prediction) {
    for (auto& mP : prediction) {
        auto ensP = ensemblePredictions.find(mP.label);
        if (ensP != ensemblePredictions.end()) ensP->second.value += mP.value;
        else ensemblePredictions[mP.label] = {mP.label, mP.value};
    }
}

template <typename T> void Ensemble<T>::predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) {

    UnorderedMap<int, EnsemblePrediction> ensemblePredictions;
    for (size_t i = 0; i < members.size(); ++i) {
        prediction.clear();
        members[i]->predict(prediction, features, args);
        accumulatePrediction(ensemblePredictions, prediction, i);
    }

    prediction.clear();
    for (auto& p : ensemblePredictions) {
        if(args.ensMissingScores) {
            for (size_t i = 0; i < members.size(); ++i) {
                if (!std::count(p.second.members.begin(), p.second.members.end(), i))
                    p.second.value += members[i]->predictForLabel(p.second.label, features, args);
            }
        }
        prediction.push_back({p.second.label, p.second.value / members.size()});
    }

    sort(prediction.rbegin(), prediction.rend());
    if (args.topK > 0) prediction.resize(args.topK);
}

template <typename T> Real Ensemble<T>::predictForLabel(Label label, SparseVector& features, Args& args) {
    Real value = 0;
    for (auto& m : members) value += m->predictForLabel(label, features, args);
    return value / members.size();
}

template <typename T>
std::vector<std::vector<Prediction>> Ensemble<T>::predictBatch(SRMatrix& features, Args& args) {
    int rows = features.rows();
    std::vector<UnorderedMap<int, Prediction>> simpleEnsemblePredictions;
    std::vector<UnorderedMap<int, EnsemblePrediction>> allEnsemblePredictions;

    if(args.ensMissingScores) allEnsemblePredictions.resize(rows);
    else simpleEnsemblePredictions.resize(rows);

    T* tmpMember;

    // Get top predictions for members
    for (int i = 0; i < args.ensemble; ++i) {
        if(!members[i]->isLoaded()) loadMember(args, args.output, i);

        std::vector<std::vector<Prediction>> memberPredictions = members[i]->predictBatch(features, args);
        if(args.ensMissingScores)
            for (int j = 0; j < rows; ++j) accumulatePrediction(allEnsemblePredictions[j], memberPredictions[j], i);
        else
            for (int j = 0; j < rows; ++j) accumulatePrediction(simpleEnsemblePredictions[j], memberPredictions[j]);
        
        if (args.ensOnTheTrot) members[i]->unload();
    }

    std::vector<std::vector<Prediction>> predictions(rows);

    // Predict missing predictions for specific labels
    if(args.ensMissingScores) {
        for (int i = 0; i < args.ensemble; ++i) {
            if(!members[i]->isLoaded()) loadMember(args, args.output, i);

            for (int j = 0; j < rows; ++j) {
                printProgress(j, rows);
                for (auto &p : allEnsemblePredictions[j]) {
                    if (!std::count(p.second.members.begin(), p.second.members.end(), i))
                        p.second.value += members[i]->predictForLabel(p.second.label, features[j], args);
                }
            }

            if (args.ensOnTheTrot) members[i]->unload();
        }

        for (int i = 0; i < rows; ++i) {
            predictions[i].reserve(allEnsemblePredictions[i].size());
            for (auto& p : allEnsemblePredictions[i])
                predictions[i].emplace_back(p.second.label, p.second.value / args.ensemble);
        }
    } else {
        for (int i = 0; i < rows; ++i) {
            predictions[i].reserve(simpleEnsemblePredictions[i].size());
            for (auto& p : simpleEnsemblePredictions[i])
                predictions[i].emplace_back(p.second.label, p.second.value / args.ensemble);
        }
    }

    // Create final predictions
    for (int i = 0; i < rows; ++i) {
        sort(predictions[i].rbegin(), predictions[i].rend());
        if (args.topK > 0) predictions[i].resize(args.topK);
    }

    return predictions;
}

template <typename T> T* Ensemble<T>::loadMember(Args& args, const std::string& infile, int memberNo) {
    Log(CERR) << "Loading ensemble member " << memberNo << " ...\n";
    Log::updateGlobalIndent(2);
    assert(memberNo < args.ensemble || memberNo < members.size());

    auto member = members[memberNo];
    member->load(args, joinPath(infile, "member_" + std::to_string(memberNo)));

    if (!thresholds.empty()) member->setThresholds(thresholds);
    if(!labelsWeights.empty()) member->setLabelsWeights(labelsWeights);
    if(!labelsBiases.empty()) member->setLabelsBiases(labelsBiases);

    Log::updateGlobalIndent(-2);
    return member;
}

template <typename T> void Ensemble<T>::load(Args& args, std::string infile) {
    if (!args.ensOnTheTrot) Log(CERR) << "Loading ensemble of " << args.ensemble << " models ...\n";
    Log::updateGlobalIndent(2);
    for (int i = 0; i < args.ensemble; ++i){
        members.push_back(new T());
        if(i == 0 || !args.ensOnTheTrot) loadMember(args, infile, i);
    }
    m = members[0]->outputSize();
    Log::updateGlobalIndent(-2);
}

template <typename T> void Ensemble<T>::unload() {
    for (auto& m : members) m->unload();
    Model::unload();
}

template <typename T> void Ensemble<T>::printInfo() {
    Log(CERR) << "Ensemble of " << members.size() << " info:\n";
    Log::updateGlobalIndent(2);
    for (int i = 0; i < members.size(); ++i) {
        Log(CERR) << "Member " << i << " info:\n";
        members[i]->printInfo();
    }
    Log::updateGlobalIndent(-2);
}

template <typename T> void Ensemble<T>::setThresholds(std::vector<Real> th){
    Model::setThresholds(th);
    for(auto& m : members){
        if(m->isPreloaded()) m->setThresholds(th);
    }
}

template <typename T> void Ensemble<T>::setLabelsWeights(std::vector<Real> lw){
    Model::setLabelsWeights(lw);
    for (auto &m : members){
        if(m->isPreloaded()) m->setLabelsWeights(lw);
    }
}

template <typename T> void Ensemble<T>::setLabelsBiases(std::vector<Real> lb){
    Model::setLabelsBiases(lb);
    for (auto &m : members){
        if(m->isPreloaded()) m->setLabelsBiases(lb);
    }
}