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
#include "threads.h"

PLT::PLT(){
    tree = nullptr;
    nCount = 0;
    rCount = 0;
}

PLT::~PLT(){
    delete tree;
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

void PLT::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output){
    std::cerr << "Building tree ...\n";

    tree = new Tree();
    tree->buildTreeStructure(labels, features, args);

    std::cerr << "Training tree ...\n";

    // Check data
    int rows = features.rows();
    assert(rows == labels.rows());
    assert(tree->k >= labels.cols());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(tree->t);
    std::vector<std::vector<Feature*>> binFeatures(tree->t);

    // Positive and negative nodes
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    std::cerr << "Assigning data points to nodes ...\n";

    // Gather examples for each node
    for(int r = 0; r < rows; ++r){
        printProgress(r, rows);

        nPositive.clear();
        nNegative.clear();

        int rSize = labels.size(r);
        auto rLabels = labels.row(r);

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

        for (const auto& n : nPositive){
            binLabels[n->index].push_back(1.0);
            binFeatures[n->index].push_back(features.row(r));
        }

        for (const auto& n : nNegative){
            binLabels[n->index].push_back(0.0);
            binFeatures[n->index].push_back(features.row(r));
        }

        nCount += nPositive.size() + nNegative.size();
        ++rCount;
    }

    trainBases(joinPath(output, "weights.bin"), features.cols(), binLabels, binFeatures, args);

    // Save tree
    tree->saveToFile(joinPath(output, "tree.bin"));
}

void PLT::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    std::priority_queue<TreeNodeValue> nQueue;

    // Note: loss prediction gets worse results for tree with higher arity then 2
    double val = bases[tree->root->index]->predictProbability(features);
    //double val = -bases[tree->root->index]->predictLoss(features);
    nQueue.push({tree->root, val});
    ++nCount;
    ++rCount;

    while (prediction.size() < args.topK && !nQueue.empty()) predictNext(nQueue, prediction, features);
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

double PLT::predict(Label label, Feature* features, Args &args){
    return 1.0;
}

void PLT::load(Args &args, std::string infile){
    std::cerr << "Loading PLT model ...\n";

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
