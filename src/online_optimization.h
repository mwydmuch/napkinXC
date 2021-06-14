/*
 Copyright (c) 2018-2021 by Marek Wydmuch

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

#include<cmath>


double logisticLoss(double label, double pred, double w){
    double prob = (1.0 / (1.0 + std::exp(-pred)));
    return -label * std::log(prob) - (1 - label) * std::log(1 - prob);
}

double logisticGrad(double label, double pred, double w){
    return (1.0 / (1.0 + std::exp(-pred))) - label;
}

double hingeGrad(double label, double pred, double w){
    double _label = 2 * label - 1;
    double v = _label * pred;
    if(v > 1.0) return 0.0;
    else return -_label;
}

double squaredHingeGrad(double label, double pred, double w){
    double _label = 2 * label - 1;
    double v = _label * pred;
    if(v > 1.0) return 0.0;
    else return -2 * std::max(1.0 - v, 0.0) * _label;
}

double pwLogisticGrad(double label, double pred, double w){
    return (1.0 / (1.0 + std::exp(-pred))) - w * label;
}

double pwLogisticLoss(double label, double pred, double w){
    double prob = (1.0 / (1.0 + std::exp(-pred)));
    return -label * w * log(prob) + (label - 1/w) * w * log(1 - prob);
}

template <typename T>
void updateSGD(T& W, T& G, Feature* features, double grad, int t, Args& args){
    double eta = args.eta;
    double lr = eta * sqrt(1.0 / t);
    Feature* f = features;
    while (f->index != -1) {
        W[f->index] -= lr * grad * f->value;
        ++f;
    }
}

template <typename T>
void updateAdaGrad(T& W, T& G, Feature* features, double grad, int t, Args& args){
    double eta = args.eta;
    double eps = args.adagradEps;
    Feature* f = features;
    while (f->index != -1) {
        G[f->index] += f->value * f->value * grad * grad;
        double lr = eta * std::sqrt(1.0 / (eps + G[f->index]));
        W[f->index] -= lr * (grad * f->value);
        ++f;
        // TODO: add correct regularization
        //double reg = l2 * W[f->index];
        //W[f->index] -= lr * (grad * f->value + reg);
    }
}
