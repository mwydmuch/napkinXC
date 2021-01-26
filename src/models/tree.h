/*
 Copyright (c) 2018-2020 by Marek Wydmuch, Kalina Jasinska-Kobus, Robert Istvan Busa-Fekete

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
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

    int subtreeLeaves;
};

// For prediction in tree based models / Huffman trees building
struct TreeNodeValue {
    TreeNodeValue(TreeNode* node, double value): node(node), prob(value), value(value) {};
    TreeNodeValue(TreeNode* node, double prob, double value): node(node), prob(prob), value(value) {};

    TreeNode* node;
    double prob; // Node's probability
    double value; // Node's probability/value/loss, used for tree search

    bool operator<(const TreeNodeValue& r) const { return value < r.value; }
    bool operator>(const TreeNodeValue& r) const { return value > r.value; }
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

    // Hierarchical K-Means
    void buildKmeansTree(SRMatrix<Feature>& labelsFeatures, Args& args);

    // Huffman tree
    void buildHuffmanTree(SRMatrix<Label>& labels, Args& args);

    // Just random complete and balance tree
    void buildCompleteTree(int labelCount, bool randomizeOrder, Args& args);
    void buildBalancedTree(int labelCount, bool randomizeOrder, Args& args);

    // Simulate simple online tree building
    void buildOnlineTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args);

    // Custom tree structure
    void loadTreeStructure(std::string file);
    void saveTreeStructure(std::string file);

    void save(std::ostream& out) override;
    void load(std::istream& in) override;

    int k; // Number of labels, should be equal to leaves.size()
    int t; // Number of tree nodes, should be equal to nodes.size()

    TreeNode* root;                      // Pointer to root node
    std::vector<TreeNode*> nodes;        // Pointers to tree nodes
    UnorderedMap<int, TreeNode*> leaves; // Leaves map;

    // Tree utils
    void printTree(TreeNode* rootNode = nullptr);
    int getNumberOfLeaves(TreeNode* rootNode = nullptr);
    int getTreeDepth(TreeNode* rootNode = nullptr);
    int getNodeDepth(TreeNode* n);
    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1);
    inline void setParent(TreeNode* n, TreeNode* parent) {
        n->parent = parent;
        if (parent != nullptr) parent->children.push_back(n);
    }
    void setLabel(TreeNode* n, int label);
    void moveSubtree(TreeNode* oldParent, TreeNode* newParent);
    void populateNodeLabels();
    int distanceBetweenNodes(TreeNode* n1, TreeNode* n2);
    void squashTree();

private:
    static TreeNodePartition buildKmeansTreeThread(TreeNodePartition nPart, SRMatrix<Feature>& labelsFeatures,
                                                   Args& args, int seed);

};
