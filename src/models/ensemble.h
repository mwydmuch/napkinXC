/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "model.h"

template <typename T>
class Ensemble: public Model{
public:
    Ensemble();
    ~Ensemble() override;

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
    double predict(Label label, Feature* features, Args &args) override;

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
    for(auto &m : members){
        m->predict(prediction, features, args);
    }
}

template <typename T>
double Ensemble<T>::predict(Label label, Feature* features, Args &args){
    return 1.0;
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