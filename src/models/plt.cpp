/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasinska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cassert>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <list>
#include <cmath>
#include <climits>

#include "plt.h"


PLT::PLT(){
    tree = nullptr;
    nCount = 0;
    rCount = 0;
    name = "PLT";
}

PLT::~PLT(){
    delete tree;
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

void PLT::assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args){

    std::cerr << "Assigning data points to nodes ...\n";

    // Positive and negative nodes
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    // Gather examples for each node
    int rows = features.rows();
    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        getNodesToUpdate(nPositive, nNegative, labels.row(r), labels.size(r));
        addFeatures(binLabels, binFeatures, nPositive, nNegative, features.row(r));

        nCount += nPositive.size() + nNegative.size();
        ++rCount;
    }
}

void PLT::getNodesToUpdate(std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                           int* rLabels, int rSize){
    if (rSize > 0){
        for (int i = 0; i < rSize; ++i) {
            TreeNode *n = tree->leaves[rLabels[i]];
            nPositive.insert(n);
            while (n->parent) {
                n = n->parent;
                nPositive.insert(n);
            }
        }

        std::queue<TreeNode*> nQueue; // Nodes queue
        nQueue.push(tree->root); // Push root

        while(!nQueue.empty()) {
            TreeNode* n = nQueue.front(); // Current node
            nQueue.pop();

            for(const auto& child : n->children) {
                if (nPositive.count(child)) nQueue.push(child);
                else nNegative.insert(child);
            }
        }
    } else nNegative.insert(tree->root);
}

void PLT::addFeatures(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                      std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                      Feature* features){
    for (const auto& n : nPositive){
        binLabels[n->index].push_back(1.0);
        binFeatures[n->index].push_back(features);
    }

    for (const auto& n : nNegative){
        binLabels[n->index].push_back(0.0);
        binFeatures[n->index].push_back(features);
    }
}

void PLT::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    predictTopK(prediction, features, args.topK);
}

void PLT::predictTopK(std::vector<Prediction>& prediction, Feature* features, int k){
    std::priority_queue<TreeNodeValue> nQueue;

    // Note: loss prediction gets worse results for tree with higher arity then 2
    double val = bases[tree->root->index]->predictProbability(features);
    //double val = -bases[tree->root->index]->predictLoss(features);
    nQueue.push({tree->root, val});
    ++nCount;
    ++rCount;

    while (prediction.size() < k && !nQueue.empty()) predictNext(nQueue, prediction, features);
}

void PLT::predictNext(std::priority_queue<TreeNodeValue>& nQueue, std::vector<Prediction>& prediction, Feature* features) {

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top(); // Current node
        nQueue.pop();

        //std::cerr << "HEAP -> " << nVal.node->index << " " << nVal.value << "\n";

        if(nVal.node->children.size()){
            for(const auto& child : nVal.node->children){
                double value = nVal.value * bases[child->index]->predictProbability(features); // When using probability
                //double value = nVal.value - bases[child->index]->predictLoss(features); // When using loss
                nQueue.push({child, value});
            }
            nCount += nVal.node->children.size();
        }
        if(nVal.node->label >= 0){
            prediction.push_back({nVal.node->label, nVal.value}); // When using probability
            break;
        }
    }
}

void PLT::predictTopKBeam(std::vector<Prediction>& prediction, Feature* features, int k){
    std::priority_queue<TreeNodeValue> nQueue;
    double val = bases[tree->root->index]->predictProbability(features);
    nQueue.push({tree->root, val});

    ++rCount;

    /*
    while (prediction.size() < k && !nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if(nVal.node->children.size()){
            for(const auto& child : nVal.node->children){
                if(child->label >= 0) prediction.push_back({child->label, nVal.value}); // When using probability
                else {
                    double value = nVal.value * bases[child->index]->predictProbability(features); // When using probability
                    //double value = nVal.value - bases[child->index]->predictLoss(features); // When using loss
                    nQueue.push({child, value});
                    ++nCount;
                }
            }
        }
    }
     */

    while (prediction.size() < k && !nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if(nVal.node->children.size()){
            for(const auto& child : nVal.node->children){
                if(child->label >= 0) prediction.push_back({child->label, nVal.value}); // When using probability
                else {
                    double value = nVal.value * bases[child->index]->predictProbability(features); // When using probability
                    //double value = nVal.value - bases[child->index]->predictLoss(features); // When using loss
                    nQueue.push({child, value});
                    ++nCount;
                }
            }
        }
    }

    sort(prediction.rbegin(), prediction.rend());
    prediction.resize(k);
}

double PLT::predictForLabel(Label label, Feature* features, Args &args){
    double value = 0;
    TreeNode *n = tree->leaves[label];
    value *= bases[n->index]->predictProbability(features);
    while (n->parent){
        n = n->parent;
        value *= bases[n->index]->predictProbability(features);
    }
    return value;
}

void PLT::load(Args &args, std::string infile){
    std::cerr << "Loading " << name << " model ...\n";

    tree = new Tree();
    tree->loadFromFile(joinPath(infile, "tree.bin"));
    bases = loadBases(joinPath(infile, "weights.bin"));
    assert(bases.size() == tree->nodes.size());
    m = tree->numberOfLeaves();
}

void PLT::printInfo(){
    std::cerr << "PLT additional stats:"
              << "\n  Mean # nodes per data point: " << static_cast<double>(nCount) / rCount
              << "\n";
}
