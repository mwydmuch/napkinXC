/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <unordered_map>

#include "model.h"

struct EnsemblePrediction{
    int label;
    double value;
    std::vector<size_t> members;

    bool operator<(const EnsemblePrediction &r) const { return value < r.value; }
};


template <typename T>
class Ensemble: public Model{
public:
    Ensemble();
    ~Ensemble() override;

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
    double predictForLabel(Label label, Feature* features, Args &args) override;

    void load(Args &args, std::string infile) override;

    void printInfo() override;

protected:
    std::vector<T*> members;
};


template <typename T>
Ensemble<T>::Ensemble(){ }

template <typename T>
Ensemble<T>::~Ensemble(){
    for(size_t i = 0; i < members.size(); ++i)
        delete members[i];
}

template <typename T>
void Ensemble<T>::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output){
    std::cerr << "Training ensemble of " << args.ensemble << " models ...\n";

    for(int i = 0; i < args.ensemble; ++i){
        std::string memberDir = joinPath(output, "member_" + std::to_string(i));
        makeDir(memberDir);
        T* member = new T();
        member->train(labels, features, args, memberDir);
        delete member;
    }
}

template <typename T>
void Ensemble<T>::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){

    std::unordered_map<int, EnsemblePrediction> ensemblePredictions;
    for(size_t i = 0; i < members.size(); ++i){
        members[i]->predict(prediction, features, args);

        for(auto &mP : prediction) {
            auto ensP = ensemblePredictions.find(mP.label);
            if (ensP != ensemblePredictions.end()) {
                ensP->second.value += mP.value;
                ensP->second.members.push_back(i);
            } else
                ensemblePredictions.insert({mP.label, {mP.label, mP.value, {i}}});
        }
    }

    prediction.clear();
    for(auto &p : ensemblePredictions){
        double value = p.second.value;
        for(size_t i = 0; i < members.size(); ++i){
            if(!std::count(p.second.members.begin(), p.second.members.end(), i))
                value += members[i]->predictForLabel(p.second.label, features, args);
        }
        prediction.push_back({p.second.label, value / members.size()});
    }

    sort(prediction.rbegin(), prediction.rend());
    if(args.topK > 0) prediction.resize(args.topK);
}

template <typename T>
double Ensemble<T>::predictForLabel(Label label, Feature* features, Args &args){
    double value = 0;
    for(auto &m : members)
        value += m->predictForLabel(label, features, args);
    return value / members.size();
}

template <typename T>
void Ensemble<T>::load(Args &args, std::string infile){
    std::cerr << "Loading ensemble of " << args.ensemble << " models ...\n";

    for(int i = 0; i < args.ensemble; ++i){
        T* member = new T();
        member->load(args, joinPath(infile, "member_" + std::to_string(i)));
        members.push_back(member);
    }
}

template <typename T>
void Ensemble<T>::printInfo(){

}