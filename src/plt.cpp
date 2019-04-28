/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasinska, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#include <cassert>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <list>
#include <chrono>
#include <random>
#include <cmath>
#include <climits>

#include "plt.h"
#include "threads.h"

PLT::PLT(){
    tree = nullptr;
}

PLT::~PLT() {
    delete tree;
}


void PLT::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    tree = new Tree();

    // Build tree structure
    tree->buildTreeStructure(labels, features, args);

    // Train the tree structure
    trainTreeStructure(labels, features, args);
}

Base* nodeTrainThread(int n, std::vector<double>& binLabels, std::vector<Feature*>& binFeatures, Args& args){
    Base* base = new Base();
    base->train(n, binLabels, binFeatures, args);
    return base;
}

void PLT::trainTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    std::cerr << "Training tree ...\n";

    // For stats
    long nCount = 0, yCount = 0;

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

    std::cerr << "Assigning points to nodes ...\n";

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
        yCount += rSize;
    }

    std::cerr << "Starting training in " << args.threads << " threads ...\n";

    std::ofstream weightsOut(joinPath(args.output, "plt_weights.bin"));
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(const auto& n : tree->nodes)
            results.emplace_back(tPool.enqueue(nodeTrainThread, features.cols(), std::ref(binLabels[n->index]),
                                               std::ref(binFeatures[n->index]), std::ref(args)));

        // Saving in the main thread
        for(int i = 0; i < results.size(); ++i) {
            printProgress(i, results.size());
            Base* base = results[i].get();
            base->save(weightsOut);
            delete base;
        }
    } else {
        for(int i = 0; i < tree->nodes.size(); ++i){
            printProgress(i, tree->nodes.size());
            Base base;
            base.train(features.cols(), binLabels[tree->nodes[i]->index], binFeatures[tree->nodes[i]->index], args);
            base.save(weightsOut);
        }
    }
    weightsOut.close();

    std::cerr << "  Points count: " << rows
              << "\n  Nodes per point: " << static_cast<double>(nCount) / rows
              << "\n  Labels per point: " << static_cast<double>(yCount) / rows
              << "\n";

    // Save tree
    tree->saveToFile(joinPath(args.output, "plt_tree.bin"));

    std::cerr << "All done\n";
}

void PLT::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    std::priority_queue<TreeNodeValue> nQueue;

    // Note: loss prediction gets worse results for tree with higher arity then 2
    double val = bases[tree->root->index]->predictProbability(features);
    //double val = -bases[tree->root->index]->predictLoss(features);
    nQueue.push({tree->root, val});

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top(); // Current node
        nQueue.pop();

        //std::cerr << "HEAP -> " << nVal.node->index << " " << nVal.value << "\n";

        if(nVal.node->label >= 0){
            prediction.push_back({nVal.node->label, nVal.value}); // When using probability
            //prediction.push_back({nVal.node, exp(nVal.value)}); // When using loss
            if (prediction.size() >= args.topK)
                break;
        } else {
            for(const auto& child : nVal.node->children){
                val = nVal.value * bases[child->index]->predictProbability(features); // When using probability
                //val = nVal.value - bases[child->index]->predictLoss(features); // When using loss
                nQueue.push({child, val});
            }
        }
    }
}

void PLT::load(std::string infile){
    std::cerr << "Loading PLT model ...\n";

    tree = new Tree();
    tree->loadFromFile(joinPath(infile, "plt_tree.bin"));

    std::cerr << "Loading base classifiers ...\n";
    std::ifstream weightsIn(joinPath(infile, "plt_weights.bin"));
    for(int i = 0; i < tree->t; ++i) {
        printProgress(i, tree->t);
        bases.emplace_back(new Base());
        bases.back()->load(weightsIn);
    }
    weightsIn.close();
}

