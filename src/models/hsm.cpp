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

void HSM::assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           std::vector<std::vector<double>>& binWeights, SRMatrix<Label>& labels,
                           SRMatrix<Feature>& features, Args& args) {
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

        auto rSize = labels.size(r);
        auto rLabels = labels[r];

        // Check row
        if (!args.pickOneLabelWeighting && rSize != 1) {
            Log(CERR) << "Encountered example with " << rSize
                      << " labels HSM is multi-class classifier, use PLT instead\n";
            continue;
        }

        for (int i = 0; i < rSize; ++i) {
            getNodesToUpdate(nPositive, nNegative, rLabels[i]);
            addNodesLabelsAndFeatures(binLabels, binFeatures, nPositive, nNegative, features[r]);
            if (args.pickOneLabelWeighting) {
                double w = 1.0 / rSize;
                for (const auto& n : nPositive) binWeights[n->index].push_back(w);
                for (const auto& n : nNegative) binWeights[n->index].push_back(w);
            }

            nodeUpdateCount += nPositive.size() + nNegative.size();
        }
        ++dataPointCount;
    }
}

void HSM::getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                           const int rLabel) {

    std::vector<TreeNode*> path;

    auto ni = tree->leaves.find(rLabel);
    if (ni == tree->leaves.end()) {
        Log(CERR) << "Encountered example with label " << rLabel << " that does not exists in the tree\n";
        return;
    }
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
    std::function<bool(TreeNode*, double)>& ifAddToQueue, std::function<double(TreeNode*, double)>& calculateValue,
    TopKQueue<TreeNodeValue>& nQueue, Feature* features) {

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if (!nVal.node->children.empty()) {
            if (nVal.node->children.size() == 2) {
                double value = bases[nVal.node->children[0]->index]->predictProbability(features);
                addToQueue(ifAddToQueue, calculateValue, nQueue, nVal.node->children[0], nVal.value * value);
                addToQueue(ifAddToQueue, calculateValue, nQueue, nVal.node->children[1], nVal.value * (1.0 - value));
                ++nodeEvaluationCount;
            } else {
                double sum = 0;
                std::vector<double> values;
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

double HSM::predictForLabel(Label label, Feature* features, Args& args) {
    double value = 0;
    TreeNode* n = tree->leaves[label];
    while (n->parent) {
        if (n->parent->children.size() == 2) {
            if (n == n->parent->children[0])
                value *= bases[n->children[0]->index]->predictProbability(features);
            else
                value *= 1.0 - bases[n->children[0]->index]->predictProbability(features);
            ++nodeEvaluationCount;
        } else {
            double sum = 0;
            double tmpValue = 0;
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
        Log(COUT) << "  Path length: " << static_cast<double>(pathLength) / dataPointCount << "\n";
}
