/*
 Copyright (c) 2019-2021 by Marek Wydmuch

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


class BR : public Model {
public:
    BR();
    ~BR() override { unload(); }

    void train(SRMatrix& labels, SRMatrix& features, Args& args, std::string output) override;
    void predict(std::vector<Prediction>& prediction, SparseVector& features, Args& args) override;
    Real predictForLabel(Label label, SparseVector& features, Args& args) override;

    void load(Args& args, std::string infile) override;
    void unload() override;

    void printInfo() override;

protected:
    std::vector<Base*> bases;
    virtual void assignDataPoints(std::vector<std::vector<Real>>& binLabels,
                                  std::vector<Feature*>& binFeatures,
                                  std::vector<Real>& binWeights,
                                  SRMatrix& labels, SRMatrix& features, int rStart, int rStop, Args& args);
    virtual std::vector<Prediction> predictForAllLabels(SparseVector& features, Args& args);
    static size_t calculateNumberOfParts(SRMatrix& labels, SRMatrix& features, Args& args);
};
