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

#pragma once

#include "models/ovr.h"


struct WeightIndex {
    int index;
    double value;

    bool operator<(const WeightIndex& r) const { return value < r.value; }

    friend std::ostream& operator<<(std::ostream& os, const WeightIndex& p) {
        os << p.index << ":" << p.value;
        return os;
    }
};

class SVBOPInvertedIndex : public OVR {
public:
    SVBOPInvertedIndex();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) override;
    void load(Args& args, std::string infile) override;

    void printInfo() override;

protected:
    std::vector<std::vector<WeightIndex>> R;

    int productCount;
    int dataPointCount; // Data points count
    int correctTop;
};

class SVBOPFagin : public SVBOPInvertedIndex {
public:
    SVBOPFagin();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) override;
};

class SVBOPThreshold : public SVBOPInvertedIndex {
public:
    SVBOPThreshold();

    void predict(std::vector<Prediction>& prediction, Feature* features, Args& args) override;
};