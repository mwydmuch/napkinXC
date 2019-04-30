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


HSM::HSM(){
    tree = nullptr;
}

HSM::~HSM() {
    delete tree;
    for(size_t i = 0; i < bases.size(); ++i)
        delete bases[i];
}

void HSM::train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args){
    std::cerr << "Building tree ...\n";

    tree = new Tree();
    tree->buildTreeStructure(labels, features, args);

    std::cerr << "Training tree ...\n";

    // For stats
    long pLen = 0, eCount = 0;

    // Check data
    int rows = features.rows();
    assert(rows == labels.rows());
    assert(tree->k >= labels.cols());

    // Examples selected for each node
    std::vector<std::vector<double>> binLabels(tree->t);
    std::vector<std::vector<Feature*>> binFeatures(tree->t);

    // Nodes on path
    std::vector<TreeNode*> path;

    std::cerr << "Assigning data points to nodes ...\n";

    // Gather examples for each node
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
            if (rSize > 1)
                throw "HSM is multi-class classifier, encountered example with more then 1 label! Use PLT instead.";
            else if (rSize < 1){
                std::cerr << "Example without label, skipping ...\n";
                continue;
            }
        }

        assert(path.size());
        assert(path.back() == tree.root);

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
    }

    std::cerr << "Starting training in " << args.threads << " threads ...\n";

    std::ofstream weightsOut(joinPath(args.output, "hsm_weights.bin"));
    if(args.threads > 1){
        // Run learning in parallel
        ThreadPool tPool(args.threads);
        std::vector<std::future<Base*>> results;

        for(const auto& n : tree->nodes)
            results.emplace_back(tPool.enqueue(baseTrain, features.cols(), std::ref(binLabels[n->index]),
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

    std::cerr << "  Data points count: " << rows
              << "\n  Avg. path len: " << static_cast<double>(pLen) / rows
              << "\n  Estimators updates per data point: " << static_cast<double>(eCount) / rows
              << "\n";

    // Save tree
    tree->saveToFile(joinPath(args.output, "hsm_tree.bin"));
}

void HSM::predict(std::vector<Prediction>& prediction, Feature* features, Args &args){
    std::priority_queue<TreeNodeValue> nQueue;

    double val = bases[tree->root->index]->predictProbability(features);
    assert(val == 1);
    nQueue.push({tree->root, val});

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top();
        nQueue.pop();

        if(nVal.node->label >= 0){
            prediction.push_back({nVal.node->label, nVal.value});
            if (prediction.size() >= args.topK)
                break;
        }
        if(nVal.node->children.size()){
            if(nVal.node->children.size() == 2) {
                double value = bases[nVal.node->children[0]->index]->predictProbability(features);
                nQueue.push({nVal.node->children[0], nVal.value * value});
                nQueue.push({nVal.node->children[1], nVal.value * (1.0 - value)});
            }
            else {
                double sum = 0;
                std::vector<double> values;
                for (const auto &child : nVal.node->children) {
                    values.emplace_back(bases[child->index]->predictProbability(features));
                    sum += values.back();
                }

                for(int i = 0; i < nVal.node->children.size(); ++i)
                    nQueue.push({nVal.node->children[i], nVal.value * values[i] / sum});
            }
        }
    }
}

void HSM::load(std::string infile){
    std::cerr << "Loading HSM model ...\n";

    tree = new Tree();
    tree->loadFromFile(joinPath(infile, "hsm_tree.bin"));

    std::cerr << "Loading base classifiers ...\n";
    std::ifstream weightsIn(joinPath(infile, "hsm_weights.bin"));
    for(int i = 0; i < tree->t; ++i) {
        printProgress(i, tree->t);
        bases.emplace_back(new Base());
        bases.back()->load(weightsIn);
    }
    weightsIn.close();
}

