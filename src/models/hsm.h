/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <tuple>
#include <algorithm>
#include <random>

#include "batch_plt.h"
#include "tree.h"


class HSM: public BatchPLT{  // HSM is multi-class version of PLT
public:
    HSM();

    double predictForLabel(Label label, Feature* features, Args &args) override;

    void printInfo() override;

protected:
    //void assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
    //                              SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args) override;
    void getNodesToUpdate(std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                          int* rLabels, int rSize) override;
    void predictNext(std::priority_queue<TreeNodeValue>& nQueue, std::vector<Prediction>& prediction, Feature* features) override;

    int eCount; // Number of updated/evaluated classifiers during training or prediction
    int pLen; // Len of the path
};
