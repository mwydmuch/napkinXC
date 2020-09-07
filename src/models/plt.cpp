/*
 Copyright (c) 2018-2020 by Marek Wydmuch, Kalina Jasinska-Kobus, Robert Istvan Busa-Fekete

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
#include <vector>

#include "plt.h"


PLT::PLT() {
    tree = nullptr;
    treeSize = 0;
    treeDepth = 0;
    nodeEvaluationCount = 0;
    nodeUpdateCount = 0;
    dataPointCount = 0;
    type = plt;
    name = "PLT";
}

PLT::~PLT() {
    delete tree;
    for (auto b : bases) delete b;
}

void PLT::assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           std::vector<std::vector<double>*>* binWeights, SRMatrix<Label>& labels,
                           SRMatrix<Feature>& features, Args& args) {

    LOG(CERR) << "Assigning data points to nodes ...\n";

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    // Gather examples for each node
    int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r], labels.size(r));
        addNodesLabelsAndFeatures(binLabels, binFeatures, nPositive, nNegative, features[r]);

        nodeUpdateCount += nPositive.size() + nNegative.size();
        ++dataPointCount;
    }

    unsigned long long usedMem = nodeUpdateCount * (sizeof(double) + sizeof(Feature*)) + binLabels.size() * (sizeof(binLabels) + sizeof(binFeatures));
    LOG(CERR) << "  Temporary data size: " << formatMem(usedMem) << "\n";
}

std::vector<std::vector<std::pair<int, int>>> PLT::assignDataPoints(SRMatrix<Label>& labels, SRMatrix<Feature>& features){
    std::vector<std::vector<std::pair<int, int>>> nodesDataPoints;

    // Positive and negative nodes
    UnorderedSet<TreeNode*> nPositive;
    UnorderedSet<TreeNode*> nNegative;

    // Gather examples for each node
    int rows = features.rows();
    for (int r = 0; r < rows; ++r) {
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels[r], labels.size(r));
        addNodesDataPoints(nodesDataPoints, r, nPositive, nNegative);
    }

    return nodesDataPoints;
}

void PLT::getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                           const int* rLabels, const int rSize) {
    for (int i = 0; i < rSize; ++i) {
        auto ni = tree->leaves.find(rLabels[i]);
        if (ni == tree->leaves.end()) {
            LOG(CERR) << "Encountered example with label " << rLabels[i] << " that does not exists in the tree\n";
            continue;
        }
        TreeNode* n = ni->second;
        nPositive.insert(n);
        while (n->parent) {
            n = n->parent;
            nPositive.insert(n);
        }
    }

    if (!nPositive.count(tree->root)) {
        nNegative.insert(tree->root);
        return;
    }

    for(auto& n : nPositive) {
        for (const auto &child : n->children) {
            if (!nPositive.count(child))
                nNegative.insert(child);
        }
    }
}

void PLT::addNodesLabelsAndFeatures(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                      UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                      Feature* features) {
    for (const auto& n : nPositive) {
        binLabels[n->index].push_back(1.0);
        binFeatures[n->index].push_back(features);
    }

    for (const auto& n : nNegative) {
        binLabels[n->index].push_back(0.0);
        binFeatures[n->index].push_back(features);
    }
}

void PLT::addNodesDataPoints(std::vector<std::vector<std::pair<int, int>>>& nodesDataPoints, int row,
                             UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative) {
    for (const auto& n : nPositive)
        nodesDataPoints[n->index].push_back({row, 1.0});

    for (const auto& n : nNegative)
        nodesDataPoints[n->index].push_back({row, 1.0});
}

void PLT::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    TopKQueue<TreeNodeValue> nQueue(args.topK);
    //TopKQueue<TreeNodeValue> nQueue(0);

    nQueue.push({tree->root, predictForNode(tree->root, features)});
    ++nodeEvaluationCount;
    ++dataPointCount;

    Prediction p = predictNextLabel(nQueue, features, args.threshold);
    while ((prediction.size() < args.topK || args.topK == 0) && p.label != -1) {
        prediction.push_back(p);
        p = predictNextLabel(nQueue, features, args.threshold);
    }
}

Prediction PLT::predictNextLabel(TopKQueue<TreeNodeValue>& nQueue, Feature* features, double threshold) {
    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if (!nVal.node->children.empty()) {
            for (const auto& child : nVal.node->children)
                addToQueue(nQueue, child, nVal.value * predictForNode(child, features), threshold);
            nodeEvaluationCount += nVal.node->children.size();
        }
        if (nVal.node->label >= 0) return {nVal.node->label, nVal.value};
    }

    return {-1, 0};
}

void PLT::setThresholds(std::vector<double> th){
    thresholds = th;

    //LOG(CERR) << "Setting thresholds for PLT ...\n";
    for(auto& n : tree->nodes) {
        n->th = 1;
        for (auto &l : n->labels) {
            if (thresholds[l] < n->th) {
                n->th = thresholds[l];
                n->thLabel = l;
            }
        }
        //LOG(CERR) << "  Node " << n->index << ", labels: " << n->labels.size() << ", min: " << n->th << std::endl;
    }

    tree->root->th = 0;
    tree->root->thLabel = 0;
}

void PLT::updateThresholds(UnorderedMap<int, double> thToUpdate){
    for(auto& th : thToUpdate)
        thresholds[th.first] = th.second;

    for(auto& th : thToUpdate){
        TreeNode* n = tree->leaves[th.first];
        while(n != tree->root){
            if(th.second < n->th){
                n->th = th.second;
                n->thLabel = th.first;
            } else if (th.first == n->thLabel && th.second > n->th){
                for(auto& l : n->labels) {
                    if(thresholds[l] < n->th){
                        n->th = thresholds[l];
                        n->thLabel = l;
                    }
                }
            }
            n = n->parent;
        }
    }
}

void PLT::predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    TopKQueue<TreeNodeValue> nQueue;

    nQueue.push({tree->root, predictForNode(tree->root, features)});
    ++nodeEvaluationCount;
    ++dataPointCount;

    Prediction p = predictNextLabelWithThresholds(nQueue, features);
    while (p.label != -1) {
        prediction.push_back(p);
        p = predictNextLabelWithThresholds(nQueue, features);
    }
}

Prediction PLT::predictNextLabelWithThresholds(TopKQueue<TreeNodeValue>& nQueue, Feature* features) {
    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if (!nVal.node->children.empty()) {
            for (const auto& child : nVal.node->children)
                addToQueueThresholds(nQueue, child, nVal.value * predictForNode(child, features));
            nodeEvaluationCount += nVal.node->children.size();
        }
        if (nVal.node->label >= 0) return {nVal.node->label, nVal.value};
    }

    return {-1, 0};
}

double PLT::predictForLabel(Label label, Feature* features, Args& args) {
    auto fn = tree->leaves.find(label);
    if(fn == tree->leaves.end()) return 0;
    TreeNode* n = fn->second;
    double value = bases[n->index]->predictProbability(features);
    while (n->parent) {
        n = n->parent;
        value *= predictForNode(n, features);
        ++nodeEvaluationCount;
    }
    return value;
}

void PLT::load(Args& args, std::string infile) {
    LOG(CERR) << "Loading " << name << " model ...\n";

    tree = new Tree();
    tree->loadFromFile(joinPath(infile, "tree.bin"));
    bases = loadBases(joinPath(infile, "weights.bin"));
    assert(bases.size() == tree->nodes.size());
    m = tree->getNumberOfLeaves();

    if(!args.thresholds.empty())
        tree->populateNodeLabels();
}

void PLT::printInfo() {
    LOG(COUT) << name << " additional stats:"
              << "\n  Tree size: " << (tree != nullptr ? tree->nodes.size() : treeSize)
              << "\n  Tree depth: " << (tree != nullptr ? tree->getTreeDepth() : treeDepth) << "\n";
    if(nodeUpdateCount > 0)
        LOG(COUT) << "  Updated estimators / data point: " << static_cast<double>(nodeUpdateCount) / dataPointCount << "\n";
    if(nodeEvaluationCount > 0)
        LOG(COUT) << "  Evaluated estimators / data point: " << static_cast<double>(nodeEvaluationCount) / dataPointCount << "\n";
}

void BatchPLT::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {

    // Create tree
    if (!tree) {
        tree = new Tree();
        tree->buildTreeStructure(labels, features, args);
    }
    m = tree->getNumberOfLeaves();

    LOG(CERR) << "Training tree ...\n";

    // Check data
    assert(features.rows() == labels.rows());
    //assert(tree->k >= labels.cols());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(tree->t);
    std::vector<std::vector<Feature*>> binFeatures(tree->t);
    std::vector<std::vector<double>*>* binWeights = nullptr;

    if (type == hsm && args.pickOneLabelWeighting) {
        binWeights = new std::vector<std::vector<double>*>(tree->t);
        for (auto& p : *binWeights) p = new std::vector<double>();
    }

    assignDataPoints(binLabels, binFeatures, binWeights, labels, features, args);

    // Save tree and free it, it is no longer needed
    tree->saveToFile(joinPath(output, "tree.bin"));
    tree->saveTreeStructure(joinPath(output, "tree"));
    treeSize = tree->nodes.size();
    treeDepth = tree->getTreeDepth();
    delete tree;
    tree = nullptr;

    trainBases(joinPath(output, "weights.bin"), features.cols(), binLabels, binFeatures, binWeights, args);

    if (type == hsm && args.pickOneLabelWeighting) {
        for (auto& w : *binWeights) delete w;
        delete binWeights;
    }
}
