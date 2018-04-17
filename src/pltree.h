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
    int index; // Index of the base predictor
    int label; // -1 means it is internal node

    TreeNode* parent; // Pointer to the parent node
    std::vector<TreeNode*> children; // Pointers to the children nodes
};

struct TreeNodeValue{
    TreeNode* node;
    double val; // Node's value/probability

    bool operator<(const TreeNodeValue &r) const { return val < r.val; }
};

struct Assignation{
    int index;
    int value;
};

struct TreeNodePartition{
    TreeNode* node;
    std::vector<Assignation> *partition;
};

class PLTree{
public:
    PLTree();
    ~PLTree();

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void predict(std::vector<TreeNodeValue>& prediction, Feature* features, std::vector<Base*>& bases, int k);
    void test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, std::vector<Base*>& bases, Args& args);

    inline int nodes() { return t; }
    inline int labels() { return k; }

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile);
    void load(std::istream& in);

private:
    int k; // Number of labels, should be equal to treeLeaves.size()
    int t; // Number of tree nodes, should be equal to tree.size()

    TreeNode *treeRoot;
    std::vector<TreeNode*> tree; // Pointers to tree nodes
    std::unordered_map<int, TreeNode*> treeLeaves; // Leaves map

    void buildTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void buildTreeTopDown(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void buildCompleteTree(int labelCount, int arity, bool randomizeTree = false);
    void loadTreeStructure(std::string file);

    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1);
};
