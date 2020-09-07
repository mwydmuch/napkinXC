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

#include "base.h"
#include "model.h"
#include "tree.h"


// This is virtual class for all PLT based models: HSM, Batch PLT, Online PLT
class PLT : virtual public Model {
public:
    PLT();
    ~PLT() override;

    void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) override;
    double predictForLabel(Label label, Feature* features, Args& args) override;

    void setThresholds(std::vector<double> th) override;
    void updateThresholds(UnorderedMap<int, double> thToUpdate) override;
    void predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, Args& args) override;

    void load(Args& args, std::string infile) override;

    void printInfo() override;

    // For Python PLT Framework
    std::vector<std::vector<std::pair<int, int>>> assignDataPoints(SRMatrix<Label>& labels, SRMatrix<Feature>& features);

protected:
    Tree* tree;
    std::vector<Base*> bases;

    virtual void assignDataPoints(std::vector<std::vector<double>>& binLabels,
                                  std::vector<std::vector<Feature*>>& binFeatures,
                                  std::vector<std::vector<double>*>* binWeights, SRMatrix<Label>& labels,
                                  SRMatrix<Feature>& features, Args& args);
    void getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                          const int* rLabels, const int rSize);

    static void addNodesLabelsAndFeatures(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                                   UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative, Feature* features);
    static void addNodesDataPoints(std::vector<std::vector<std::pair<int, int>>>& nodesDataPoints, int row,
                                   UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative);

    // Helper methods for prediction
    virtual Prediction predictNextLabel(TopKQueue<TreeNodeValue>& nQueue, Feature* features, double threshold);
    virtual Prediction predictNextLabelWithThresholds(TopKQueue<TreeNodeValue>& nQueue, Feature* features);

    virtual inline double predictForNode(TreeNode* node, Feature* features){
        return bases[node->index]->predictProbability(features);
    }

    inline static void addToQueue(TopKQueue<TreeNodeValue>& nQueue, TreeNode* node, double value,
                                  double threshold) {
        if (value >= threshold) nQueue.push({node, value}, node->label > -1);
    }

    inline static void addToQueueThresholds(TopKQueue<TreeNodeValue>& nQueue, TreeNode* node, double value) {
        if (value >= node->th) nQueue.push({node, value}, node->label > -1);
    }

    // Additional statistics
    int treeSize;
    int treeDepth;
    int nodeEvaluationCount; // Number of visited nodes during training prediction (updated/evaluated classifiers)
    int nodeUpdateCount; // Number of visited nodes during training or prediction
    int dataPointCount; // Data points count
};

class BatchPLT : public PLT {
public:
    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) override;
};
