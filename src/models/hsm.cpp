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

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <list>
#include <unordered_set>
#include <vector>

#include "hsm.h"
#include "threads.h"


HSM::HSM() {
    pathLength = 0;
    name = "HSM";
    type = hsm;
}

void HSM::assignDataPoints(std::vector<std::vector<Real>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           std::vector<std::vector<Real>>& binWeights, SRMatrix& labels,
                           SRMatrix& features, Args& args) {
    Log(CERR) << "Assigning data points to nodes ...\n";

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    // Gather examples for each node
    int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        SparseVector& rLabels = labels[r];
        int rSize = rLabels.nonZero();

        // Check row
        if (!args.pickOneLabelWeighting && rSize != 1)
            throw std::invalid_argument("Encountered example with " + std::to_string(rSize) + " labels. HSM is multi-class classifier, use PLT or --pickOneLabelWeighting option instead.");

        for (auto &l : labels[r]){
            getNodesToUpdate(nPositive, nNegative, l.index);
            addNodesLabelsAndFeatures(binLabels, binFeatures, nPositive, nNegative, features[r]);
            if (args.pickOneLabelWeighting) {
                Real w = 1.0 / rSize;
                for (const auto& n : nPositive) binWeights[n->index].push_back(w);
                for (const auto& n : nNegative) binWeights[n->index].push_back(w);
            }

            nodeUpdateCount += nPositive.size() + nNegative.size();
        }
        ++dataPointCount;
    }
}

void HSM::getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative, int label) {

    std::vector<TreeNode*> path;

    auto ni = tree->leaves.find(label);
    if (ni == tree->leaves.end())
        throw std::invalid_argument("Encountered example with " + std::to_string(label) + " that does not exists in the tree.");
    TreeNode* n = ni->second;
    path.push_back(n);
    while (n->parent) {
        n = n->parent;
        path.push_back(n);
    }

    assert(path.size());
    assert(path.back() == tree->root);

    for (int i = path.size() - 1; i >= 0; --i) {
        TreeNode *n = path[i], *p = n->parent;
        if (p == nullptr || p->children.size() == 1) {
            nPositive.insert(n);
        } else if (p->children.size() == 2) { // Binary node requires just 1 probability estimator
            TreeNode *c0 = n->parent->children[0];
            if (c0 == n) nPositive.insert(c0);
            else nNegative.insert(c0);
        } else if (p->children.size() > 2) { // Node with arity > 2 requires OVR estimator
            for (const auto& c : p->children) {
                if (c == n) nPositive.insert(c);
                else nNegative.insert(c);
            }
        }
    }

    pathLength += path.size();
}

Prediction HSM::predictNextLabel(
    std::function<bool(TreeNode*, Real)>& ifAddToQueue, std::function<Real(TreeNode*, Real)>& calculateValue,
    TopKQueue<TreeNodeValue>& nQueue, SparseVector& features) {

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if (!nVal.node->children.empty()) {
            if (nVal.node->children.size() == 2) {
                Real value = bases[nVal.node->children[0]->index]->predictProbability(features);
                addToQueue(ifAddToQueue, calculateValue, nQueue, nVal.node->children[0], nVal.value * value);
                addToQueue(ifAddToQueue, calculateValue, nQueue, nVal.node->children[1], nVal.value * (1.0 - value));
                ++nodeEvaluationCount;
            } else {
                Real sum = 0;
                std::vector<Real> values;
                values.reserve(nVal.node->children.size());
                for (const auto& child : nVal.node->children) {
                    values.emplace_back(std::exp(bases[child->index]->predictValue(features))); // Softmax normalization
                    sum += values.back();
                }

                for (int i = 0; i < nVal.node->children.size(); ++i)
                    addToQueue(ifAddToQueue, calculateValue, nQueue, nVal.node->children[i], nVal.value * values[i] / sum);

                nodeEvaluationCount += nVal.node->children.size();
            }
        }
        if (nVal.node->label >= 0) return {nVal.node->label, nVal.value};
    }

    return {-1, 0};
}

Real HSM::predictForLabel(Label label, SparseVector& features, Args& args) {
    Real value = 0;
    TreeNode* n = tree->leaves[label];
    while (n->parent) {
        if (n->parent->children.size() == 2) {
            if (n == n->parent->children[0])
                value *= bases[n->children[0]->index]->predictProbability(features);
            else
                value *= 1.0 - bases[n->children[0]->index]->predictProbability(features);
            ++nodeEvaluationCount;
        } else {
            Real sum = 0;
            Real tmpValue = 0;
            for (const auto& child : n->parent->children) {
                if (child == n) {
                    tmpValue = std::exp(bases[child->index]->predictValue(features)); // Softmax normalization
                    sum += tmpValue;
                } else
                    sum += std::exp(bases[child->index]->predictValue(features));
            }
            value *= tmpValue / sum;
            nodeEvaluationCount += n->parent->children.size();
        }
        n = n->parent;
    }

    return value;
}

void HSM::printInfo() {
    PLT::printInfo();
    if(pathLength > 0)
        Log(COUT) << "  Path length: " << static_cast<Real>(pathLength) / dataPointCount << "\n";
}
