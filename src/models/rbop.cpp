/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cassert>
#include <algorithm>
#include <vector>
#include <list>
#include <cmath>
#include <climits>

#include "rbop.h"
#include "set_utility.h"


RBOP::RBOP(){
    type = rbop;
    name = "RBOP";
}

void RBOP::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());

    std::priority_queue<TreeNodeValue> nQueue;
    std::priority_queue<TreeNodeValue> kQueue;

    double value = bases[tree->root->index]->predictProbability(features);
    assert(value == 1);
    nQueue.push({tree->root, value});
    ++rCount;

    TreeNode* bestN = tree->root;
    double bestU = u->g(tree->leaves.size()) * value;
    double bestP = value;

    // Q part
    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if(nVal.node->label >= 0){
            kQueue = std::priority_queue<TreeNodeValue>();
            //prediction.push_back({nVal.node->label, nVal.value});
            break;
        }

        int childrenAdded = 0;
        if(nVal.node->children.size()){
            if(nVal.node->children.size() == 2) {
                double value = bases[nVal.node->children[0]->index]->predictProbability(features);
                double P = nVal.value * value;
                if(P >= args.epsilon) {
                    TreeNode* c = nVal.node->children[0];
                    nQueue.push({c, P});
                    ++childrenAdded;

                    double U = u->g(tree->getNumberOfLeaves(c)) * P;
                    if(bestU < U) {
                        bestU = U;
                        bestN = c;
                        bestP = P;
                    }
                }

                P = nVal.value * (1.0 - value);
                if(P >= args.epsilon) {
                    TreeNode* c = nVal.node->children[1];
                    nQueue.push({c, P});
                    ++childrenAdded;

                    double U = u->g(tree->getNumberOfLeaves(c)) * P;
                    if(bestU < U) {
                        bestU = U;
                        bestN = c;
                        bestP = P;
                    }
                }
                ++eCount;
            }
            else {
                double sum = 0;
                std::vector<double> values;
                for (const auto &child : nVal.node->children) {
                    values.emplace_back(bases[child->index]->predictProbability(features));
                    sum += values.back();
                }

                for(int i = 0; i < nVal.node->children.size(); ++i) {
                    double P = nVal.value * values[i] / sum;
                    if(P >= args.epsilon) {
                        TreeNode* c = nVal.node->children[i];
                        nQueue.push({c, P});
                        ++childrenAdded;

                        double U = u->g(tree->getNumberOfLeaves(c)) * P;
                        if(bestU < U) {
                            bestU = U;
                            bestN = c;
                            bestP = P;
                        }
                    }
                }

                eCount += nVal.node->children.size();
            }

            if(!childrenAdded)
                kQueue.push({nVal.node, nVal.value});
        }
    }

    // K part
    while (!kQueue.empty()) {
        TreeNodeValue nVal = kQueue.top();
        kQueue.pop();
        double tmpBestP = 0;
        TreeNode* tmpBestN = nVal.node;

        if(nVal.node->children.size()){
            if(nVal.node->children.size() == 2) {
                double value = bases[nVal.node->children[0]->index]->predictProbability(features);
                double P = nVal.value * value;
                if(P > tmpBestP) {
                    tmpBestP = P;
                    tmpBestN = nVal.node->children[0];
                }

                P = nVal.value * (1.0 - value);
                if(P > bestP) {
                    tmpBestP = P;
                    tmpBestN = nVal.node->children[1];
                }
                ++eCount;
            }
            else {
                double sum = 0;
                std::vector<double> values;
                for (const auto &child : nVal.node->children) {
                    values.emplace_back(bases[child->index]->predictProbability(features));
                    sum += values.back();
                }

                for(int i = 0; i < nVal.node->children.size(); ++i) {
                    double P = nVal.value * values[i] / sum;
                    if(P > tmpBestP) {
                        tmpBestP = P;
                        tmpBestN = nVal.node->children[i];
                    }
                }

                eCount += nVal.node->children.size();
            }
        }

        double U = u->g(tree->getNumberOfLeaves(tmpBestN)) * tmpBestP;
        if(bestU < U) {
            bestU = U;
            bestN = tmpBestN;
            bestP = tmpBestP;
        }
    }

    // Generate prediction
    std::queue<TreeNode*> predQueue;
    predQueue.push(bestN);

    while(!predQueue.empty()){
        TreeNode* n = predQueue.front();
        predQueue.pop();

        if(n->label >= 0) prediction.push_back({n->label, bestP});
        for(auto c : n->children) predQueue.push(c);
    }

    //TODO divide values?
}
