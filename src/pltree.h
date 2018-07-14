/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
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
#include <random>

#include "args.h"
#include "types.h"
#include "base.h"
#include "kmeans.h"
#include "knn.h"
#include "utils.h"

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

    template<typename T>
    void predict(std::vector<TreeNodeValue>& prediction, T* features, std::vector<Base*>& bases, std::vector<KNN*>& kNNs, Args &args);

    void test(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args);
    void predict(SRMatrix<Feature>& features, Args& args);

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

template<typename T>
void PLTree::predict(std::vector<TreeNodeValue>& prediction, T* features, std::vector<Base*>& bases, std::vector<KNN*>& kNNs, Args &args){
    std::priority_queue<TreeNodeValue> nQueue;

    // Note: loss prediction gets worse results for tree with higher arity then 2
    double val = bases[treeRoot->index]->predictProbability(features);
    //double val = -bases[treeRoot->index]->predictLoss(features);
    nQueue.push({treeRoot, val});

    while (!nQueue.empty()) {
        TreeNodeValue nVal = nQueue.top(); // Current node
        nQueue.pop();

        //std::cerr << "HEAP -> " << nVal.node->index << " " << nVal.value << "\n";

        if(nVal.node->label >= 0){
            prediction.push_back({nVal.node, nVal.value}); // When using probability
            //prediction.push_back({nVal.node, exp(nVal.value)}); // When using loss
            if (prediction.size() >= args.topK)
                break;
        } else {
            if(nVal.node->kNNNode && args.kNN){ // KNN supports only probabilities
                TreeNode* n = nVal.node;
                std::vector<Feature> result;
                kNNs[nVal.node->index]->predict(features, args.kNN, result);
                for(const auto& r : result){
                    val = nVal.value * r.value;
                    nQueue.push({tree[r.index], val});
                }
            }
            for(const auto& child : nVal.node->children){
                val = nVal.value * bases[child->index]->predictProbability(features); // When using probability
                //val = nVal.value - bases[child->index]->predictLoss(features); // When using loss
                nQueue.push({child, val});
            }
        }
    }
}
