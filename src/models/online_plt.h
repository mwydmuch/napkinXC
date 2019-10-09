/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "plt.h"
#include "online_model.h"


class OnlinePLT: public OnlineModel, public PLT {
public:
    void init(int labelCount, Args &args) override;
    void update(Label* labels, size_t labelsSize, Feature* features, size_t featuresSize, Args &args) override;
    void save(Args &args, std::string output) override;

protected:
    bool onlineTree;
    std::vector<Base*> tmpBases;

    void getNodesToUpdate(std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                          int* rLabels, int rSize) override;

};

