/*
 Copyright (c) 2019-2020 by Marek Wydmuch

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

#include "svbop_hf.h"
#include "set_utility.h"


SVBOPHF::SVBOPHF() {
    type = svbopHf;
    name = "SVBOP-HF";
}

void SVBOPHF::predict(std::vector<Prediction>& prediction, Feature* features, Args& args) {
    TopKQueue<TreeNodeValue> nQueue;

    double value = bases[tree->root->index]->predictProbability(features);
    assert(value == 1);
    nQueue.push({tree->root, value, value});
    ++dataPointCount;

    std::shared_ptr<SetUtility> u = SetUtility::factory(args, outputSize());

    // Set functions
    std::function<bool(TreeNode*, double)> ifAddToQueue = [&] (TreeNode* node, double prob) {
        return true;
    };

    std::function<double(TreeNode*, double)> calculateValue = [&] (TreeNode* node, double prob) {
        return prob;
    };

    double P = 0, bestU = 0;
    while (!nQueue.empty()) {
        auto p = predictNextLabel(ifAddToQueue, calculateValue, nQueue, features);
        P += p.value;
        double U = u->g(prediction.size() + 1) * P;
        if (bestU < U) {
            prediction.push_back(p);
            bestU = U;
        } else
            break;
    }
}