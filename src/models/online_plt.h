/*
 Copyright (c) 2019-2020 by Marek Wydmuch

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

#include "online_model.h"
#include "plt.h"

#include <mutex>
#include <shared_mutex>


class OnlinePLT : public OnlineModel, public PLT {
public:
    OnlinePLT();
    ~OnlinePLT() override;

    void init(Args& args) override;
    void init(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args) override;
    void update(const int row, Label* labels, size_t labelsSize, Feature* features, size_t featuresSize,
                Args& args) override;

    void save(Args& args, std::string output) override;
    void load(Args& args, std::string infile) override;

protected:
    bool onlineTree;

    std::vector<Base*> auxBases; // Aux classifiers
    std::shared_timed_mutex treeMtx;

    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1, Base* base = nullptr, Base* auxBase = nullptr);
    void expandTree(const std::vector<Label>& newLabels, Feature* features, Args& args);
};
