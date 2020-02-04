/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "online_model.h"
#include "plt.h"

#include <mutex>
#include <shared_mutex>


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
    std::shared_timed_mutex treeMtx;

    std::vector<float> norms;
    std::vector<UnorderedMap<int, float>> centroids;
    std::mutex centroidsMtx;

    void expandTree(Label newLabel, Feature* features, Args& args);
    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1, Base* base = nullptr, Base* tmpBase = nullptr);
    void expandTree(const std::vector<Label>& newLabels, Feature* features, Args& args);
};
