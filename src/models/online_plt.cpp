/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "online_plt.h"

void OnlinePLT::init(int labelCount, Args &args){
    tree = new Tree();
    tree->buildTreeStructure(labelCount, args);
    bases.resize(tree->t);
    for(auto &b : bases) b = new Base();
    onlineTree = false;

    // TODO: add online tree building
}

void OnlinePLT::update(Label* labels, size_t labelsSize, Feature* features, size_t featuresSize, Args &args){
    std::unordered_set<TreeNode*> nPositive;
    std::unordered_set<TreeNode*> nNegative;

    getNodesToUpdate(nPositive, nNegative, labels, labelsSize);

    // Update positive base estimators
    for (const auto& n : nPositive)
        bases[n->index]->update(1.0, features, args);

    // Update negative
    for (const auto& n : nNegative)
        bases[n->index]->update(-1.0, features, args);
}

void OnlinePLT::getNodesToUpdate(std::unordered_set<TreeNode*>& nPositive, std::unordered_set<TreeNode*>& nNegative,
                                 int* rLabels, int rSize){
    if(onlineTree){
        // TODO: add online tree building here
    } else
        PLT::getNodesToUpdate(nPositive, nNegative, rLabels, rSize);
}

void OnlinePLT::save(Args &args, std::string output){

    // Save base classifiers
    std::ofstream out(joinPath(output, "weights.bin"));
    int size = bases.size();
    out.write((char*) &size, sizeof(size));
    for(int i = 0; i < bases.size(); ++i)
        bases[i]->save(out);
    out.close();

    // Save tree
    tree->saveToFile(joinPath(output, "tree.bin"));

    // Save tree structure
    tree->saveTreeStructure(joinPath(output, "tree.txt"));
}