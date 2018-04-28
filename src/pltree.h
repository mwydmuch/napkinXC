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

struct TreeNode{
    int index; // Index of the base predictor
    int label; // -1 means it is internal node

    TreeNode* parent; // Pointer to the parent node
    std::vector<TreeNode*> children; // Pointers to the children nodes
};

struct TreeNodeValue{
    TreeNode* node;
    double value; // Node's value/probability

    bool operator<(const TreeNodeValue &r) const { return value < r.value; }
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

struct LabelsAssignation{
    int index;
    int value;
};

struct LabelsDistances{
    int index;
    std::vector<Feature> values;

    bool operator<(const LabelsDistances &r) const { return values[0].value < r.values[0].value; }
};

struct TreeNodePartition{
    TreeNode* node;
    std::vector<LabelsAssignation> *partition;
};

class FreqTuple{
public:
    int64_t f;
    TreeNode* node;
public:
    FreqTuple(int64_t f_, TreeNode* node_){
        f=f_; node=node_;
    }
    int64_t getFrequency() const { return f;}
};

struct DereferenceCompareNode : public std::binary_function<FreqTuple*, FreqTuple*, bool>{
    bool operator()(const FreqTuple* lhs, const FreqTuple* rhs) const {
        return lhs->getFrequency() > rhs->getFrequency();
    }
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
    std::default_random_engine rng;

    int k; // Number of labels, should be equal to treeLeaves.size()
    int t; // Number of tree nodes, should be equal to tree.size()

    TreeNode *treeRoot;
    std::vector<TreeNode*> tree; // Pointers to tree nodes
    std::unordered_map<int, TreeNode*> treeLeaves; // Leaves map


    // Tree building methods
    void addModelToTree(Base *model, int parent, std::vector<int> &labels, std::vector<int> &instances,
                        std::ofstream &out, Args &args, std::vector<NodeJob> &nextLevelJobs);
    void addRootToTree(Base *model, int parent, std::vector<int> &labels, std::vector<int> &instances,
                               std::ofstream &out, Args &args, std::vector<NodeJob> &nextLevelJobs);
    void trainTopDown(SRMatrix<Label> &labels, SRMatrix<Feature> &features, Args &args);
    void trainFixed(SRMatrix<Label> &labels, SRMatrix<Feature> &features, Args &args);
//    std::vector<struct JobResult> processJob(int index, std::vector<int>& jobInstances, std::vector<int>& jobLabels,
//                                             std::ofstream& out,SRMatrix<Label>& labels, SRMatrix<Feature>& features,
//                                             Args& args);
    struct JobResult trainRoot(SRMatrix<Label> &labels, SRMatrix<Feature> &features, Args &args);

    void buildHuffmanPLTree(SRMatrix<Label>& labels, Args &args);
//    void buildTreeTopDown(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
//    void cut(SRMatrix<Label>& labels, SRMatrix<Feature>& features, std::vector<int>& active, std::vector<int>& left, std::vector<int>& right, Args &args);

    void balancedKMeans(std::vector<LabelsAssignation>* partition, SRMatrix<Feature>& labelsFeatures, Args &args);
    void balancedKMeansWithRandomProjection(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void buildKMeansTree(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args &args);
    void buildCompleteTree(int labelCount, int arity, bool randomizeTree = false);
    void buildBalancedTree(int labelCount, int arity, bool randomizeTree);
    TreeNode* buildBalancedTreeRec(std::vector<int>::const_iterator begin, std::vector<int>::const_iterator end );
    void loadTreeStructure(std::string file);

    TreeNode* createTreeNode(TreeNode* parent = nullptr, int label = -1);

    void printTree(TreeNode *n);
    void printTree();

    void getRandomProjection(std::vector<std::vector<double>> &data, int projectDim, int dim);

    void computeLabelRepresentation(SRMatrix<Feature> &labelsFeatures, std::vector<std::vector<double>> &randomMatrix,
                                    std::vector<LabelsAssignation> *partition, std::vector<std::vector<int>>& labelToIndices,
                                    SRMatrix<Feature>& features, Args &args);
};
