/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <tuple>
#include <algorithm>

#include "args.h"
#include "types.h"
#include "base.h"
#include "models/kmeans.h"
#include "utils.h"

// Tree node
struct TreeNode{
    int index; // Index of the base classifier
    int label; // -1 means it is internal node

    TreeNode* parent; // Pointer to the parent node
    std::vector<TreeNode*> children; // Pointers to the children nodes
    int nextToExpand;
};

// For K-Means based trees
struct TreeNodePartition{
    TreeNode* node;
    std::vector<Assignation>* partition;
};

// For Huffman based trees
struct TreeNodeFrequency{
    TreeNode* node;
    int frequency;

    // Sorted from the lowest to the highest frequency
    bool operator<(const TreeNodeFrequency &r) const { return frequency > r.frequency; }
};

// For prediction in tree based models
struct TreeNodeValue{
    TreeNode* node;
    double value; // Node's value/probability/loss

    bool operator<(const TreeNodeValue &r) const { return value < r.value; }
};

class Tree: public FileHelper{
public:
    Tree();
    ~Tree();

    // Build tree structure of given type
    void buildTreeStructure(int labelCount, Args &args);
    void buildTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);

    // Custom tree structure
    void loadTreeStructure(std::string file);
    void saveTreeStructure(std::string file);

    void save(std::ostream& out) override;
    void load(std::istream& in) override;

    int k; // Number of labels, should be equal to leaves.size()
    int t; // Number of tree nodes, should be equal to nodes.size()

    TreeNode *root; // Pointer to root node
    std::vector<TreeNode*> nodes; // Pointers to tree nodes
    std::unordered_map<int, TreeNode*> leaves; // Leaves map;

    // Online tree
    inline bool isOnline(){ return online; }
    void expandTree(Label newLabel, std::vector<Base*>& bases, std::vector<Base*>& tmpBases, Args& args);

    int numberOfLeaves(TreeNode *rootNode = nullptr);

private:
    bool online;
    int nextToExpand;

    // Hierarchical K-Means
    void buildKMeansTree(SRMatrix<Feature>& labelsFeatures, Args &args);

    // Huffman tree
    void buildHuffmanTree(SRMatrix<Label>& labels, Args &args);

    // Just random complete and balance tree
    void buildCompleteTree(int labelCount, bool randomizeOrder, Args &args);
    void buildBalancedTree(int labelCount, bool randomizeOrder, Args &args);

    // Tree utils
    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1);
    inline void setParent(TreeNode *n, TreeNode *parent){
        n->parent = parent;
        if(parent != nullptr) parent->children.push_back(n);
    }
    void setLabel(TreeNode *n, int label);
    int getDepth(TreeNode *n);
    void moveSubtree(TreeNode* n, TreeNode* m);
    TreeNode* getNextNodeToExpand(Args &args);
    void printTree(TreeNode *rootNode = nullptr);
};
