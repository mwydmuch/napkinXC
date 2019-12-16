/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "online_plt.h"

OnlinePLT::OnlinePLT(){
    type = oplt;
    name = "Online PLT";
}

OnlinePLT::~OnlinePLT(){
    for(auto b: tmpBases) delete b;
}

void OnlinePLT::init(int labelCount, Args &args) {
    tree = new Tree();
    tree->buildTreeStructure(labelCount, args);

    if (!tree->isOnline()) {
        bases.resize(tree->t);
        for (auto &b : bases) {
            b = new Base();
            b->setupOnlineTraining(args);
        }
    }
}

void OnlinePLT::update(const int row, Label* labels, size_t labelsSize, Feature* features, size_t featuresSize, Args &args){
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;
    std::unordered_map<TreeNode*, double> weights;

    if(tree->isOnline()) { // Check if example contains a new label
        std::lock_guard<std::mutex> lock(treeMtx);
        for (int i = 0; i < labelsSize; ++i) {
            if(!tree->leaves.count(labels[i]))
                tree->expandTree(labels[i], bases, tmpBases, args); // Expand tree in case of the new label
        }
    }

    getNodesToUpdate(nPositive, nNegative, labels, labelsSize);

    // Update positive base estimators
    for (const auto& n : nPositive)
        bases[n->index]->update(1.0, features, args);

    // Update negative
    for (const auto& n : nNegative)
        bases[n->index]->update(0.0, features, args);

    if(tree->isOnline()){ // Update temporary nodes
        for (const auto& n : nPositive)
            tmpBases[n->index]->update(0.0, features, args);
    }
}

void OnlinePLT::save(Args &args, std::string output){

    // Save base classifiers
    std::ofstream out(joinPath(output, "weights.bin"));
    int size = bases.size();
    out.write((char*) &size, sizeof(size));
    for(int i = 0; i < bases.size(); ++i) {
        bases[i]->finalizeOnlineTraining(args);
        bases[i]->save(out);
    }
    out.close();

    // Save tree
    tree->saveToFile(joinPath(output, "tree.bin"));

    // Save tree structure
    tree->saveTreeStructure(joinPath(output, "tree.txt"));
}
