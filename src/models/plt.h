/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "base.h"
#include "model.h"
#include "tree.h"


// This is virutal class for all PLT based models: HSM, Batch PLT, Online PLT
class PLT: virtual public Model{
public:
    PLT();
    ~PLT() override;

    //void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, Feature* features, Args &args) override;
    double predictForLabel(Label label, Feature* features, Args &args) override;

    void predictTopK(std::vector<Prediction>& prediction, Feature* features, int k);

    void load(Args &args, std::string infile) override;

    void printInfo() override;

    Tree* tree;
protected:
    std::vector<Base*> bases;
    
    virtual void assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                                  SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    virtual void getNodesToUpdate(std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                                  int* rLabels, int rSize);
    void addFeatures(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                     std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                     Feature* features);
    virtual Prediction predictNext(std::priority_queue<TreeNodeValue>& nQueue, Feature* features);

    // Additional statistics
    int nCount; // Number of visited nodes (updated/evaluated classifiers) during training or prediction
    int rCount; // Data points count
    std::vector<double> lossPerLevel;
};
