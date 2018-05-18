/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * All rights reserved.
 */

#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <random>

#include "args.h"
#include "types.h"
#include "base.h"
#include "kmeans.h"
#include "utils.h"

class KNN;

struct TreeNode{
    int index; // Index of the base classifier
    int label; // -1 means it is internal node

    TreeNode* parent; // Pointer to the parent node
    std::vector<TreeNode*> children; // Pointers to the children nodes

    bool kNNNode; // Node uses K-NN classifier
};

struct TreeNodeValue{
    TreeNode* node;
    double value; // Node's value/probability/loss

    bool operator<(const TreeNodeValue &r) const { return value < r.value; }
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

    bool operator<(const TreeNodeFrequency &r) const { return frequency < r.frequency; }
};

struct NodeJob{
    int parent;
    std::vector<int> labels;
    std::vector<int> instances;
};

struct JobResult{
    Base *base;
    int parent;
    std::vector<int> instances;
    std::vector<int> labels;
};

class PLTree{
public:
    PLTree();
    ~PLTree();

    void buildTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void predict(std::vector<TreeNodeValue>& prediction, Feature* features, std::vector<Base*>& bases, std::vector<KNN*>& kNNs, Args &args);
    void test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args);

    inline int nodes() { return t; }
    inline int labels() { return k; }

    void save(std::string outfile);
    void save(std::ostream& out);
    void load(std::string infile);
    void load(std::istream& in);

private:
    std::default_random_engine rng;

    int k; // Number of labels, should be equal to treeLeaves.size()
    int t; // Number of tree nodes, should be equal to tree.size()

    TreeNode *treeRoot;
    std::vector<TreeNode*> tree; // Pointers to tree nodes
    std::unordered_map<int, TreeNode*> treeLeaves; // Leaves map

    // Training
    void trainTreeStructure(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);

    // Tree building methods

    // Top down, definitions in pltree_topdown.cpp
    // TODO: clean this a little bit
    void addModelToTree(Base *model, int parent, std::vector<int> &labels, std::vector<int> &instances,
                        std::ofstream &out, Args &args, std::vector<NodeJob> &nextLevelJobs);
    void addRootToTree(Base *model, int parent, std::vector<int> &labels, std::vector<int> &instances,
                               std::ofstream &out, Args &args, std::vector<NodeJob> &nextLevelJobs);
    void trainTopDown(SRMatrix<Label> &labels, SRMatrix<Feature> &features, Args &args);
    struct JobResult trainRoot(SRMatrix<Label> &labels, SRMatrix<Feature> &features, Args &args);
    TreeNode* buildBalancedTreeRec(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end );

    // TODO: do we need this?
    /*
    std::vector<struct JobResult> processJob(int index, std::vector<int>& jobInstances, std::vector<int>& jobLabels,
                                         std::ofstream& out,SRMatrix<Label>& labels, SRMatrix<Feature>& features,
                                         Args& args);
    void buildTreeTopDown(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void cut(SRMatrix<Label>& labels, SRMatrix<Feature>& features, std::vector<int>& active, std::vector<int>& left, std::vector<int>& right, Args &args);
    */

    // Random projection
    void generateRandomProjection(std::vector<std::vector<double>>& data, int projectDim, int dim);
    void projectLabelsRepresentation(SRMatrix<Feature>& labelsFeatures, std::vector<std::vector<double>>& randomMatrix,
                                    std::vector<std::vector<int>>& labelToIndices, SRMatrix<Feature>& features, Args &args);
    void balancedKMeansWithRandomProjection(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);

    // Hierarchical K-Means
    void buildKMeansTree(SRMatrix<Feature>& labelsFeatures, Args &args);

    // Some experimental tree structures
    void buildLeaveFreqBehindTree(SRMatrix<Feature>& labelsFeatures, std::vector<Frequency>& labelsFreq, Args& args);

    void buildKMeansHuffmanTree(SRMatrix<Feature>& labelsFeatures, std::vector<Frequency>& labelsFreq, SRMatrix<Label>& labels, Args& args);

    // Just random complete and balance tree
    void buildCompleteTree(int labelCount, bool randomizeOrder, Args &args);
    void buildBalancedTree(int labelCount, bool randomizeOrder, Args &args);

    // Huffman tree
    void buildHuffmanTree(SRMatrix<Label>& labels, Args &args);

    // Custom tree structure
    void loadTreeStructure(std::string file);
    void saveTreeStructure(std::string file);

    // Tree building utils
    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1);
    void printTree(TreeNode *root = nullptr);
};
