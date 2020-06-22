/**
 * Copyright (c) 2019-2020 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include <unordered_set>

#include "args.h"
#include "model.h"


class Measure {
public:
    static std::vector<std::shared_ptr<Measure>> factory(Args& args, int outputSize);

    Measure();

    virtual void accumulate(Label* labels, const std::vector<Prediction>& prediction) = 0;
    void accumulate(SRMatrix<Label>& labels, std::vector<std::vector<Prediction>>& predictions);
    virtual double value();

    inline bool isMeanMeasure(){ return meanMeasure; };
    inline double mean(){ return value(); };
    double stdDev();

    inline std::string getName() { return name; }

protected:
    void addValue(double value);

    bool meanMeasure;
    std::string name;
    double sum;
    double sumSq;
    int count;
};

class MeasureAtK : public Measure {
public:
    explicit MeasureAtK(int k);

protected:
    int k;
};

class TruePositivesAtK : public MeasureAtK {
public:
    explicit TruePositivesAtK(int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    static double calculate(Label* labels, const std::vector<Prediction>& prediction, int k);
};

class TruePositives : public Measure {
public:
    TruePositives();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    static double calculate(Label* labels, const std::vector<Prediction>& prediction);
};

class FalsePositives : public Measure {
public:
    FalsePositives();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    static double calculate(Label* labels, const std::vector<Prediction>& prediction);
};

class FalseNegatives : public Measure {
public:
    FalseNegatives();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    static double calculate(Label* labels, const std::vector<Prediction>& prediction);
};

class Recall : public Measure {
public:
    Recall();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class RecallAtK : public MeasureAtK {
public:
    explicit RecallAtK(int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class Precision : public Measure {
public:
    Precision();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class PrecisionAtK : public MeasureAtK {
public:
    explicit PrecisionAtK(int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class DCGAtK : public MeasureAtK {
public:
    explicit DCGAtK(int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    static double calculate(Label* labels, const std::vector<Prediction>& prediction, int k);
};

class NDCGAtK : public MeasureAtK{
public:
    explicit NDCGAtK(int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class Coverage : public Measure {
public:
    explicit Coverage(int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    UnorderedSet<int> seen;
    int m;
};

class CoverageAtK : public MeasureAtK {
public:
    CoverageAtK(int outputSize, int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    UnorderedSet<int> seen;
    int m;
};

class Accuracy : public Measure {
public:
    Accuracy();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class PredictionSize : public Measure {
public:
    PredictionSize();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class HammingLoss : public Measure {
public:
    HammingLoss();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class SampleF1 : public Measure {
public:
    SampleF1();

    void accumulate(Label *labels, const std::vector<Prediction> &prediction) override;
};

class MicroF1 : public Measure {
public:
    MicroF1();

    void accumulate(Label *labels, const std::vector<Prediction> &prediction) override;
};

class MacroF1 : public Measure {
public:
    explicit MacroF1(int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    std::vector<double> labelsTP;
    std::vector<double> labelsFP;
    std::vector<double> labelsFN;
    int m;
    int zeroDivisionDenominator;
};
