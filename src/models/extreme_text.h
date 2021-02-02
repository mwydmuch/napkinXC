/*
 Copyright (c) 2020 by Marek Wydmuch

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

#include "plt.h"

typedef double XTWeight; 

class ExtremeText : public PLT {
public:
    ExtremeText();

    void train(SRMatrix<Label>& labels, SRMatrix<Feature>& features, Args& args, std::string output) override;

    void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) override;
    double predictForLabel(Label label, Feature* features, Args& args) override;

    void load(Args& args, std::string infile) override;

protected:
    Matrix<XTWeight> inputW;  // Input vectors (word vectors)
    Matrix<XTWeight> outputW; // Tree node vectors
    int dims;

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
            Log(CERR) << "  Progress: " << state / (max / 100) << "%, lr: " << lr << ", loss: " << loss << "\r";
    }

    inline double log(double x) {
        return std::log(x + 1e-5);
    }

    inline double sigmoid(double x) const {
        if (x < -8) return 0.0;
        else if (x > 8) return 1.0;
        else return 1.0 / (1.0 + std::exp(-x));
    }
};