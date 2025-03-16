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

#include <unordered_set>

#include "args.h"
#include "model.h"


class Metric {
public:
    static std::vector<std::shared_ptr<Metric>> factory(Args& args, int outputSize);

    Metric();

    virtual void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) = 0;
    void accumulate(SRMatrix& labels, std::vector<std::vector<Prediction>>& predictions);
    virtual double value();

    inline bool isMeanMetric(){ return meanMetric; };
    inline double mean(){ return value(); };
    double stdDev();

    inline std::string getName() { return name; }

protected:
    void addValue(double value);

    bool meanMetric;
    std::string name;
    double sum;
    double sumSq;
    int count;
};

class MetricAtK : public Metric {
public:
    explicit MetricAtK(int k);

protected:
    int k;
};

class TruePositivesAtK : public MetricAtK {
public:
    explicit TruePositivesAtK(int k);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    static double calculate(SparseVector& labels, const std::vector<Prediction>& prediction, int k);
};

class TruePositives : public Metric {
public:
    TruePositives();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    static double calculate(SparseVector& labels, const std::vector<Prediction>& prediction);
};

class FalsePositives : public Metric {
public:
    FalsePositives();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    static double calculate(SparseVector& labels, const std::vector<Prediction>& prediction);
};

class FalseNegatives : public Metric {
public:
    FalseNegatives();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    static double calculate(SparseVector& labels, const std::vector<Prediction>& prediction);
};

class Recall : public Metric {
public:
    Recall();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class RecallAtK : public MetricAtK {
public:
    explicit RecallAtK(int k);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class Precision : public Metric {
public:
    Precision();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class PrecisionAtK : public MetricAtK {
public:
    explicit PrecisionAtK(int k);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class DCGAtK : public MetricAtK {
public:
    explicit DCGAtK(int k);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    static double calculate(SparseVector& labels, const std::vector<Prediction>& prediction, int k);
};

class NDCGAtK : public MetricAtK{
public:
    explicit NDCGAtK(int k);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class Coverage : public Metric {
public:
    explicit Coverage(int outputSize);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    UnorderedSet<int> seen;
    int m;
};

class CoverageAtK : public MetricAtK {
public:
    CoverageAtK(int outputSize, int k);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    UnorderedSet<int> seen;
    int m;
};

class Accuracy : public Metric {
public:
    Accuracy();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class PredictionSize : public Metric {
public:
    PredictionSize();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class HammingLoss : public Metric {
public:
    HammingLoss();

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
};

class SampleF1 : public Metric {
public:
    SampleF1();

    void accumulate(SparseVector& labels, const std::vector<Prediction> &prediction) override;
};

class MicroF1 : public Metric {
public:
    MicroF1();

    void accumulate(SparseVector& labels, const std::vector<Prediction> &prediction) override;
};

class MacroF1 : public Metric {
public:
    explicit MacroF1(int outputSize);

    void accumulate(SparseVector& labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    std::vector<double> labelsTP;
    std::vector<double> labelsFP;
    std::vector<double> labelsFN;
    int m;
    int zeroDivisionDenominator;
};
