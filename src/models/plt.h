/**
 * Copyright (c) 2018 by Marek Wydmuch, Kalina Jasińska, Robert Istvan Busa-Fekete
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
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
    void predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, std::vector<float>& thresholds,
                               Args& args) override;
    double predictForLabel(Label label, Feature* features, Args& args) override;

    void load(Args& args, std::string infile) override;

    void printInfo() override;

protected:
    Tree* tree;
    std::vector<Base*> bases;

    virtual void assignDataPoints(std::vector<std::vector<double>>& binLabels,
                                  std::vector<std::vector<Feature*>>& binFeatures,
                                  std::vector<std::vector<double>*>* binWeights, SRMatrix<Label>& labels,
                                  SRMatrix<Feature>& features, Args& args);
    void getNodesToUpdate(UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                          const int* rLabels, const int rSize);
    void addFeatures(std::vector<std::vector<double>>& binLabels, std::vector<std::vector<Feature*>>& binFeatures,
                     UnorderedSet<TreeNode*>& nPositive, UnorderedSet<TreeNode*>& nNegative,
                     Feature* features);

    // Helper methods for prediction
    virtual Prediction predictNextLabel(TopKQueue<TreeNodeValue>& nQueue, Feature* features,
                                        double threshold);
    virtual Prediction predictNextLabelWithThresholds(TopKQueue<TreeNodeValue>& nQueue, Feature* features,
                                                      std::vector<float>& thresholds);

    virtual inline double predictForNode(TreeNode* node, Feature* features){
        return bases[node->index]->predictProbability(features);
    }

    inline static void addToQueue(TopKQueue<TreeNodeValue>& nQueue, TreeNode* node, double value,
                                  double threshold) {
        if (value >= threshold) nQueue.push({node, value});
    }

    inline static void addToQueue(TopKQueue<TreeNodeValue>& nQueue, TreeNode* node, double value,
                                  std::vector<float>& thresholds) {
        float minThreshold = 1.0;
        for (const auto& l : node->labels) minThreshold = std::min(minThreshold, thresholds[l]);
        if (value >= minThreshold) nQueue.push({node, value});
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
