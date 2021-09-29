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

    void setLabelsWeights(std::vector<Real> lw) override;

    void load(Args& args, std::string infile) override;

    void printInfo() override;

protected:
    std::vector<T*> members;
    T* loadMember(Args& args, const std::string& infile, int memberNo);
    void accumulatePrediction(std::unordered_map<int, EnsemblePrediction>& ensemblePredictions,
                              std::vector<Prediction>& prediction, int memberNo);
};


template <typename T> Ensemble<T>::Ensemble() {}

template <typename T> Ensemble<T>::~Ensemble() {
    for (auto& m : members) delete m;
}

template <typename T>
void Ensemble<T>::train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) {
    Log(CERR) << "Training ensemble of " << args.ensemble << " models ...\n";

    for (int i = 0; i < args.ensemble; ++i) {
        std::string memberDir = joinPath(output, "member_" + std::to_string(i));
        makeDir(memberDir);
        T* member = new T();
        member->train(labels, features, args, memberDir);
        delete member;
    }
}

template <typename T>
void Ensemble<T>::accumulatePrediction(std::unordered_map<int, EnsemblePrediction>& ensemblePredictions,
                                       std::vector<Prediction>& prediction, int memberNo) {

    for (auto& mP : prediction) {
        auto ensP = ensemblePredictions.find(mP.label);
        if (ensP != ensemblePredictions.end()) {
            ensP->second.value += mP.value;
            ensP->second.members.push_back(memberNo);
        } else
            ensemblePredictions.insert({mP.label, {mP.label, mP.value, {memberNo}}});
    }
}

template <typename T> void Ensemble<T>::predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) {

    std::unordered_map<int, EnsemblePrediction> ensemblePredictions;
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
    std::vector<std::unordered_map<int, EnsemblePrediction>> ensemblePredictions(rows);

    T* tmpMember;

    // Get top predictions for members
    for (int i = 0; i < args.ensemble; ++i) {
        if (args.onTheTrotPrediction) tmpMember = loadMember(args, args.output, i);
        else tmpMember = members[i];

        std::vector<std::vector<Prediction>> memberPredictions = tmpMember->predictBatch(features, args);
        for (int j = 0; j < rows; ++j) accumulatePrediction(ensemblePredictions[j], memberPredictions[j], i);

        if (args.onTheTrotPrediction) delete tmpMember;
    }

    // Predict missing predictions for specific labels
    if(args.ensMissingScores) {
        for (int i = 0; i < args.ensemble; ++i) {
            if (args.onTheTrotPrediction) tmpMember = loadMember(args, args.output, i);
            else tmpMember = members[i];

            for (int j = 0; j < rows; ++j) {
                printProgress(j, rows);
                for (auto &p : ensemblePredictions[j]) {
                    if (!std::count(p.second.members.begin(), p.second.members.end(), i))
                        p.second.value += tmpMember->predictForLabel(p.second.label, features[j], args);
                }
            }

            if (args.onTheTrotPrediction) delete tmpMember;
        }
    }

    // Create final predictions
    std::vector<std::vector<Prediction>> predictions(rows);
    for (int i = 0; i < rows; ++i) {
        for (auto& p : ensemblePredictions[i])
            predictions[i].push_back({p.second.label, p.second.value / args.ensemble});

        sort(predictions[i].rbegin(), predictions[i].rend());
        if (args.topK > 0) predictions[i].resize(args.topK);
    }

    return predictions;
}

template <typename T> T* Ensemble<T>::loadMember(Args& args, const std::string& infile, int memberNo) {
    Log(CERR) << "  Loading ensemble member number " << memberNo << " ...\n";
    assert(memberNo < args.ensemble);
    T* member = new T();
    member->load(args, joinPath(infile, "member_" + std::to_string(memberNo)));

    if(!labelsWeights.empty())
        member->setLabelsWeights(labelsWeights);

    return member;
}

template <typename T> void Ensemble<T>::load(Args& args, std::string infile) {
    if (!args.onTheTrotPrediction) {
        Log(CERR) << "Loading ensemble of " << args.ensemble << " models ...\n";
        for (int i = 0; i < args.ensemble; ++i) members.push_back(loadMember(args, infile, i));
        m = members[0]->outputSize();
    } else {
        T* member = loadMember(args, infile, 0);
        m = member->outputSize();
        delete member;
    }
}

template <typename T> void Ensemble<T>::printInfo() {}

template <typename T> void Ensemble<T>::setLabelsWeights(std::vector<Real> lw){
    Model::setLabelsWeights(lw);
    if (members.size())
        for (size_t i = 0; i < members.size(); ++i)
            members[i]->setLabelsWeights(lw);
}