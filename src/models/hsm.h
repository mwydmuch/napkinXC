/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "plt.h"
#include "tree.h"


class HSM : public BatchPLT { // HSM is multi-class version of PLT
public:
    HSM();

    double predictForLabel(Label label, Feature* features, Args& args) override;
    void printInfo() override;

protected:
    void assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                          std::vector<std::vector<double>*>* binWeights, SRMatrix<Label>& labels,
                          SRMatrix<Feature>& features, Args& args) override;
    void getNodesToUpdate(std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                          const int rLabel);
    Prediction predictNextLabel(std::priority_queue<TreeNodeValue>& nQueue, Feature* features,
                                double threshold) override;

    int eCount; // Number of updated/evaluated classifiers during training or prediction
    int pLen;   // Len of the path
};
