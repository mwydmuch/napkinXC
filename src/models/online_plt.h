/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "online_model.h"
#include "plt.h"


class OnlinePLT : public OnlineModel, public PLT {
public:
    OnlinePLT();
    ~OnlinePLT() override;

    void init(int labelCount, Args& args) override;
    void update(const int row, Label* labels, size_t labelsSize, Feature* features, size_t featuresSize,
                Args& args) override;
    void save(Args& args, std::string output) override;

protected:
    bool onlineTree;
    std::vector<Base*> tmpBases;
    std::mutex treeMtx;

    void expandTopDown(Label newLabel, Feature* features, Args& args);
    void expandBottomUp(Label newLabel, Args& args);
};
