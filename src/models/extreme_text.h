/**
 * Copyright (c) 2020 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "plt.h"

typedef double XTWeight; 

class ExtremeText : public PLT {
public:
    ExtremeText();

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) override;

    void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) override;
    void predictWithThresholds(std::vector<Prediction>& prediction, Feature* features, std::vector<float>& thresholds,
                               Args& args) override;
    double predictForLabel(Label label, Feature* features, Args& args) override;

    void load(Args& args, std::string infile) override;

protected:
    Matrix<XTWeight> inputW;  // Input vectors (word vectors)
    Matrix<XTWeight> outputW; // Tree node vectors
    size_t dims;

    double update(double lr, Feature* features, Label* labels, int rSize, Args& args);
    double updateNode(TreeNode* node, double label, Vector<XTWeight>& hidden, Vector<XTWeight>& gradient, double lr, double l2);

    Feature* computeHidden(Feature* features);

    inline double predictForNode(TreeNode* node, Feature* features) override {
        return 1.0 / (1.0 + std::exp(-dotVectors(features, outputW[node->index])));
    };

    static void trainThread(int threadId, ExtremeText* model, SRMatrix<Label>& labels,
                                  SRMatrix<Feature>& features, Args& args, const int startRow, const int stopRow);

    static void printProgress(int state, int max, double lr, double loss) {
        if (max > 100 && state % (max / 100) == 0)
            std::cerr << "  Progress: " << state / (max / 100) << "%, lr: " << lr << ", loss: " << loss << "\r";
    }

};