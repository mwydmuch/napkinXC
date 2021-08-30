/*
 Copyright (c) 2018-2021 by Marek Wydmuch, Kalina Jasinska-Kobus, Robert Istvan Busa-Fekete

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

#include "base.h"
#include "label_tree.h"
#include "model.h"

// Additional node information for prediction with thresholds
struct TreeNodeThrExt {
    Real th;
    int label;
};

// Additional node information for prediction with weights
struct TreeNodeWeightsExt {
    Real weight;
    int label;
};

// This is virtual class for all PLT based models: HSM, Batch PLT, Online PLT
class PLT : virtual public Model {
public:
    PLT();

    void predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) override;
    Real predictForLabel(Label label, SparseVector& features, Args& args) override;
    std::vector<std::vector<Prediction>> predictBatch(SRMatrix& features, Args& args) override;
    std::vector<std::vector<Prediction>> predictWithBeamSearch(SRMatrix& features, Args& args);

    void setThresholds(std::vector<Real> th) override;
    void updateThresholds(UnorderedMap<int, Real> thToUpdate) override;
    void setLabelsWeights(std::vector<Real> lw) override;

    void load(Args& args, std::string infile) override;
    void unload() override;

    void printInfo() override;

    void setTree(LabelTree*t) { tree = t; };
    LabelTree* getTree() { return tree; };
    bool isTreeLoaded() { return (tree != nullptr); };
    void preload(Args& args, std::string infile) override;

    // Helpers for Python PLT Framework
    void buildTree(SRMatrix& labels, SRMatrix& features, Args& args, std::string output);
    std::vector<std::vector<std::pair<int, Real>>> getNodesToUpdate(const SRMatrix& labels);
    std::vector<std::vector<std::pair<int, Real>>> getNodesUpdates(const SRMatrix& labels);

    void setTreeStructure(std::vector<std::tuple<int, int, int>> treeStructure, std::string output);
    std::vector<std::tuple<int, int, int>> getTreeStructure();

protected:
    LabelTree* tree;
    std::vector<Base*> bases;

    std::vector<std::vector<int>> nodesLabels;
    std::vector<TreeNodeThrExt> nodesThr; // For prediction with thresholds
    std::vector<TreeNodeWeightsExt> nodesWeights; // For prediction with labels weights

    void calculateNodesLabels();
    void setNodeThreshold(TreeNode* n);
    void setNodeWeight(TreeNode* n);

    virtual void assignDataPoints(std::vector<std::vector<Real>>& binLabels,
                                  std::vector<std::vector<Feature*>>& binFeatures,
                                  std::vector<std::vector<Real>>& binWeights,
                                  SRMatrix& labels, SRMatrix& features, Args& args);

    void getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative, const SparseVector& labels);
    static void addNodesLabelsAndFeatures(std::vector<std::vector<Real>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                                          UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative, SparseVector& features);

    // Helper methods for prediction
    virtual Prediction predictNextLabel(std::function<bool(TreeNode*, Real)>& ifAddToQueue, std::function<Real(TreeNode*, Real)>& calculateValue,
                                        TopKQueue<TreeNodeValue>& nQueue, SparseVector& features);

    virtual inline Real predictForNode(TreeNode* node, SparseVector& features){
        return bases[node->index]->predictProbability(features);
    }

    inline void addToQueue(std::function<bool(TreeNode*, Real)>& ifAddToQueue, std::function<Real(TreeNode*, Real)>& calculateValue,
                           TopKQueue<TreeNodeValue>& nQueue, TreeNode* node, Real prob){
        Real value = calculateValue(node, prob);
        if (ifAddToQueue(node, prob)) nQueue.push({node, prob, value}, node->label > -1);

    }

    // Additional statistics
    int nodeEvaluationCount; // Number of visited nodes during training prediction (updated/evaluated classifiers)
    int nodeUpdateCount; // Number of visited nodes during training or prediction
    int dataPointCount; // Data points count
};

class BatchPLT : public PLT {
public:
    void train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) override;
};
