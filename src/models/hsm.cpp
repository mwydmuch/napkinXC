/**
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

#include "hsm.h"
#include "threads.h"


HSM::HSM() {
    eCount = 0;
    pLen = 0;
    name = "HSM";
}

// Old assign data points code
/*
void HSM::assignDataPoints(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                           SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args){

    std::cerr << "Assigning data points to nodes ...\n";

    // Nodes on path
    std::vector<TreeNode*> path;

    // Gather examples for each node
    int rows = features.rows();
    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);

        path.clear();

        int rSize = labels.size(r);
        auto rLabels = labels.row(r);

        if (rSize == 1){
            TreeNode *n = tree->leaves[rLabels[0]];
            path.push_back(n);
            while (n->parent){
                n = n->parent;
                path.push_back(n);
            }
        }
        else {
            if (rSize > 1) {
                //std::cerr << "Encountered example with more then 1 label! HSM is multi-class classifier, use BR instead!";
                continue;
                //throw "OVR is multi-class classifier, encountered example with more then 1 label! Use BR instead.";
            }
            else if (rSize < 1){
                std::cerr << "Example without label, skipping ...\n";
                continue;
            }
        }

        assert(path.size());
        assert(path.back() == tree->root);

        for(int i = path.size() - 1; i >= 0; --i){
            TreeNode *n = path[i], *p = n->parent;
            if(p == nullptr || p->children.size() == 1){
                binLabels[n->index].push_back(1.0);
                binFeatures[n->index].push_back(features.row(r));
                eCount += 1;
            }
            else if(p->children.size() == 2){ // Binary node requires just 1 probability estimator
                TreeNode *c0 = n->parent->children[0], *c1 = n->parent->children[1];
                if(c0 == n)
                    binLabels[c0->index].push_back(1.0);
                else
                    binLabels[c0->index].push_back(0.0);
                binFeatures[c0->index].push_back(features.row(r));
                binLabels[c1->index].push_back(0.0); // Second one will end up as a dummy estimator
                binFeatures[c1->index].push_back(features.row(r));
                eCount += 1;
            }
            else if(p->children.size() > 2){ // Node with arity > 2 requires OVR estimator
                for(const auto& c : p->children){
                    binLabels[c->index].push_back(0.0);
                    binFeatures[c->index].push_back(features.row(r));
                }
                binLabels[n->index].back() = 1.0;
                eCount += p->children.size();
            }
        }

        pLen += path.size();
        ++rCount;
    }
}
*/

void HSM::getNodesToUpdate(std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                           int* rLabels, int rSize){
    
    std::vector<TreeNode*> path;
    if (rSize == 1){
        TreeNode *n = tree->leaves[rLabels[0]];
        path.push_back(n);
        while (n->parent){
            n = n->parent;
            path.push_back(n);
        }
    }
    else {
        if (rSize > 1) {
            //std::cerr << "Encountered example with more then 1 label! HSM is multi-class classifier, use BR instead!";
            return;
            //throw "OVR is multi-class classifier, encountered example with more then 1 label! Use BR instead.";
        }
        else if (rSize < 1){
            std::cerr << "Example without label, skipping ...\n";
            return;
        }
    }

    assert(path.size());
    assert(path.back() == tree->root);

    for(int i = path.size() - 1; i >= 0; --i){
        TreeNode *n = path[i], *p = n->parent;
        if(p == nullptr || p->children.size() == 1){
            nPositive.insert(n);
            eCount += 1;
        }
        else if(p->children.size() == 2){ // Binary node requires just 1 probability estimator
            TreeNode *c0 = n->parent->children[0], *c1 = n->parent->children[1];
            if(c0 == n)
                nPositive.insert(c0);
            else
                nNegative.insert(c0);
            nNegative.insert(c1);
            eCount += 1;
        }
        else if(p->children.size() > 2){ // Node with arity > 2 requires OVR estimator
            for(const auto& c : p->children){
                if(c == n)
                    nPositive.insert(c);
                else
                    nNegative.insert(c);
            }
            eCount += p->children.size();
        }
    }

    pLen += path.size();
}

Prediction HSM::predictNext(std::priority_queue<TreeNodeValue>& nQueue, Feature* features){

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if(nVal.node->children.size()){
            if(nVal.node->children.size() == 2) {
                double value = bases[nVal.node->children[0]->index]->predictProbability(features);
                nQueue.push({nVal.node->children[0], nVal.value * value});
                nQueue.push({nVal.node->children[1], nVal.value * (1.0 - value)});
                ++eCount;
            }
            else {
                double sum = 0;
                std::vector<double> values;
                values.reserve(nVal.node->children.size());
                for (const auto &child : nVal.node->children) {
                    //values.emplace_back(bases[child->index]->predictProbability(features)); // Normalization
                    values.emplace_back(std::exp(bases[child->index]->predictValue(features))); // Softmax normalization
                    sum += values.back();
                }

                for(int i = 0; i < nVal.node->children.size(); ++i)
                    nQueue.push({nVal.node->children[i], nVal.value * values[i] / sum});

                eCount += nVal.node->children.size();
            }
        }
        if(nVal.node->label >= 0)
            return {nVal.node->label, nVal.value};
    }
}

double HSM::predictForLabel(Label label, Feature* features, Args &args){
    double value = 0;
    TreeNode *n = tree->leaves[label];
    while (n->parent){
        if(n->parent->children.size() == 2) {
            if(n == n->parent->children[0])
                value *= bases[n->children[0]->index]->predictProbability(features);
            else
                value *= 1.0 - bases[n->children[0]->index]->predictProbability(features);
        }
        else {
            double sum = 0;
            double tmpValue = 0;
            for (const auto &child : n->parent->children) {
                if(child == n) {
                    tmpValue = bases[child->index]->predictProbability(features);
                    sum += tmpValue;
                } else
                    sum += bases[child->index]->predictProbability(features);
            }
            value *= tmpValue / sum;
        }
        n = n->parent;
    }

    return value;
}

void HSM::printInfo(){
    std::cerr << "HSM additional stats:"
              << "\n  Mean path len: " << static_cast<double>(pLen) / rCount
              << "\n  Mean # estimators per data point: " << static_cast<double>(eCount) / rCount
              << "\n";
}

