/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "base.h"
#include "misc.h"
#include "models/kmeans.h"
#include "types.h"

// Tree node
struct TreeNode {
    int index; // Index of the base classifier
    int label; // -1 means it is internal node

    TreeNode* parent;                // Pointer to the parent node
    std::vector<TreeNode*> children; // Pointers to the children nodes

    // Node helpers
    int depth;
    int subtreeLeaves;
};

// For prediction in tree based models / Huffman trees building
struct TreeNodeValue {
    TreeNode* node;
    double value; // Node's value/probability/loss

    bool operator<(const TreeNodeValue& r) const { return value < r.value; }
};

// For K-Means based trees
struct TreeNodePartition {
    TreeNode* node;
    std::vector<Assignation>* partition;
};

class Tree : public FileHelper {
public:
    Tree();
    ~Tree();

    // Build tree structure of given type
    void buildTreeStructure(int labelCount, Args& args);
    void buildTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args);

    // Custom tree structure
    void loadTreeStructure(std::string file);
    void saveTreeStructure(std::string file);

    void save(std::ostream& out) override;
    void load(std::istream& in) override;

    int k; // Number of labels, should be equal to leaves.size()
    int t; // Number of tree nodes, should be equal to nodes.size()

    TreeNode* root;                         // Pointer to root node
    std::vector<TreeNode*> nodes;           // Pointers to tree nodes
    UnorderedMap<int, TreeNode*> leaves;    // Leaves map;

    // Tree utils
    // TODO: Clean it up
    int getNumberOfLeaves(TreeNode* rootNode = nullptr);
    int getTreeDepth(TreeNode* rootNode = nullptr);

    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1);
    void setParent(TreeNode* n, TreeNode* parent);
    void setLabel(TreeNode* n, int label);

    void calculateNodesDepth();
    double meanLeafDepth();
    int minLeafDepth();
    int maxLeafDepth();

    void moveSubtree(TreeNode* oldParent, TreeNode* newParent);
    int distanceBetweenNodes(TreeNode* n1, TreeNode* n2);
    void printTree(TreeNode* rootNode = nullptr);

    int nextToExpand;
    TreeNode* nextSubtree;

private:
    // Hierarchical K-Means
    static TreeNodePartition buildKMeansTreeThread(TreeNodePartition nPart, SRMatrix<Feature>& labelsFeatures,
                                                   Args& args, int seed);
    void buildKMeansTree(SRMatrix<Feature>& labelsFeatures, Args& args);

    // Huffman tree
    void buildHuffmanTree(SRMatrix<Label>& labels, Args& args);

    // Just random complete and balance tree
    void buildCompleteTree(int labelCount, bool randomizeOrder, Args& args);
    void buildBalancedTree(int labelCount, bool randomizeOrder, Args& args);

    // Build tree in online way (simulate online tree building)
    void buildOnlineTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args);
    TreeNode* getNodeToExpand(Args& args);

};
