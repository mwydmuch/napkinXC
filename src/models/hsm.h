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

#include <algorithm>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "label_tree.h"
#include "plt.h"


class HSM : public BatchPLT { // HSM is multi-class version of PLT
public:
    HSM();

    Real predictForLabel(Label label, SparseVector& features, Args& args) override;
    void printInfo() override;

protected:
    void assignDataPoints(std::vector<std::vector<Real>>& binLabels,
                          std::vector<std::vector<Feature*>>& binFeatures,
                          std::vector<std::vector<Real>>& binWeights,
                          SRMatrix& labels, SRMatrix& features, Args& args) override;
    void getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative, int rLabel);
    Prediction predictNextLabel(
        std::function<bool(TreeNode*, Real)>& ifAddToQueue, std::function<Real(TreeNode*, Real)>& calculateValue,
        TopKQueue<TreeNodeValue>& nQueue, SparseVector& features) override;

    int pathLength;   // Length of the path
};
