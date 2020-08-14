/**
 * Copyright (c) 2018-2020 by Marek Wydmuch, Kalina Jasinska-Kobus, Robert Istvan Busa-Fekete
 * All rights reserved.
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

    std::cerr << "Assigning data points to nodes ...\n";

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
    std::cerr << "  Temporary data size: " << formatMem(usedMem) << std::endl;
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
            std::cerr << "Encountered example with label " << rLabels[i] << " that does not exists in the tree\n";
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
    if (args.beamSearch)
        beamSearch(prediction, features, args);
    else
        ucSearch(prediction, features, args);
}

void PLT::ucSearch(std::vector<Prediction>& prediction, Feature* features, Args& args) {
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

void PLT::beamSearch(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    TopKQueue<TreeNodeValue> nQueue(args.beam);
    std::priority_queue<TreeNodeValue> predictedQueue;

    nQueue.push({tree->root, predictForNode(tree->root, features)});
    ++nodeEvaluationCount;
    ++dataPointCount;


    while(!nQueue.empty()){
        TopKQueue<TreeNodeValue> nQueueNext(args.beam);
        while(!nQueue.empty()){
            TreeNodeValue nVal = nQueue.top();
            nQueue.pop();

            for (const auto& child : nVal.node->children) {
                if (!child->children.empty()) {
                    nQueueNext.push({child, nVal.value * predictForNode(child, features)}, true);
                }
                else{
                    predictedQueue.push({child, nVal.value * predictForNode(child, features)});
                }
            }
            nodeEvaluationCount += nVal.node->children.size();
        }
        nQueue = nQueueNext;
    }
    while((prediction.size() < args.topK || args.topK == 0) && !predictedQueue.empty()){
        TreeNodeValue nVal = predictedQueue.top();
        predictedQueue.pop();
        prediction.push_back({nVal.node->label, nVal.value});
    }
}


void PLT::setThresholds(std::vector<double> th){
    thresholds = th;

    //std::cerr << "Setting thresholds for PLT ...\n";
    for(auto& n : tree->nodes) {
        n->th = 1;
        for (auto &l : n->labels) {
            if (thresholds[l] < n->th) {
                n->th = thresholds[l];
                n->thLabel = l;
            }
        }
        //std::cerr << "  Node " << n->index << ", labels: " << n->labels.size() << ", min: " << n->th << std::endl;
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
    std::cerr << "Loading " << name << " model ...\n";

    tree = new Tree();
    tree->loadFromFile(joinPath(infile, "tree.bin"));
    bases = loadBases(joinPath(infile, "weights.bin"));
    assert(bases.size() == tree->nodes.size());
    m = tree->getNumberOfLeaves();

    if(!args.thresholds.empty())
        tree->populateNodeLabels();
}

void PLT::printInfo() {
    std::cout << name << " additional stats:"
              << "\n  Tree size: " << (tree != nullptr ? tree->nodes.size() : treeSize)
              << "\n  Tree depth: " << (tree != nullptr ? tree->getTreeDepth() : treeDepth) << "\n";
    if(nodeUpdateCount > 0)
        std::cout << "  Updated estimators / data point: " << static_cast<double>(nodeUpdateCount) / dataPointCount << "\n";
    if(nodeEvaluationCount > 0)
        std::cout << "  Evaluated estimators / data point: " << static_cast<double>(nodeEvaluationCount) / dataPointCount << "\n";
}

void BatchPLT::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) {

    // Create tree
    if (!tree) {
        tree = new Tree();
        tree->buildTreeStructure(labels, features, args);
    }
    m = tree->getNumberOfLeaves();

    std::cerr << "Training tree ...\n";

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
