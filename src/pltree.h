/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>

#include "args.h"
#include "types.h"
#include "base.h"

struct TreeNode{
    int index; // index of the base predictor
    int label; // -1 means it is internal node

    TreeNode* parent; // pointer to the parent node
    std::vector<TreeNode*> children; // pointers to the children nodes
};

struct TreeNodeProb{
    TreeNode* node;
    double p; // node's probability

    bool operator<(const TreeNodeProb &r) const { return p < r.p; }
};

class PLTree{
public:
    PLTree();
    ~PLTree();

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void predict(std::vector<TreeNodeProb>& prediction, Feature* features, std::vector<Base*>& bases, int k);
    void test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, std::vector<Base*>& bases, Args& args);

    int nodes();
    int labels();

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile);
    void load(std::istream& in);

private:
    int k; // number of labels, should be equal to treeLeaves.size()
    int t; // number of tree nodes, should be equal to tree.size()

    TreeNode *treeRoot;
    std::vector<TreeNode*> tree; // pointers to tree nodes
    std::unordered_map<int, TreeNode*> treeLeaves; // leaves map

    void buildTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void buildCompleteTree(int labelCount, int arity, bool randomizeTree = false);
    void loadTreeStructure(std::string file);
};
