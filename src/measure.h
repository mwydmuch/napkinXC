/**
 * Copyright (c) 2019 by Marek Wydmuch
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

    inline std::string getName() { return name; }

protected:
    std::string name;
    double sum;
    int count;
};

class MeasureAtK : public Measure {
public:
    MeasureAtK(int k);

protected:
    int k;
};

class TruePositivesAtK : public MeasureAtK {
public:
    TruePositivesAtK(int k);

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

class Recall : public Measure { // (Macro)
public:
    Recall();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class RecallAtK : public MeasureAtK {
public:
    RecallAtK(int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class Precision : public Measure { // (Macro)
public:
    Precision();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class PrecisionAtK : public MeasureAtK {
public:
    PrecisionAtK(int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};



class Coverage : public Measure {
public:
    Coverage(int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    std::unordered_set<int> seen;
    int m;
};

class CoverageAtK : public MeasureAtK {
public:
    CoverageAtK(int outputSize, int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    std::unordered_set<int> seen;
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

class F1 : public Measure {
public:
    F1();

    void accumulate(Label *labels, const std::vector<Prediction> &prediction) override;
};

class MicroF : public Measure {
public:
    MicroF();

    void accumulate(Label *labels, const std::vector<Prediction> &prediction) override;
};

class MacroF : public Measure {
public:
    MacroF(int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    std::vector<double> labelsTP;
    std::vector<double> labelsFP;
    std::vector<double> labelsFN;
    int m;
};


/*
class MicroRecall : public Measure {
public:
    MicroRecall();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class MicroPrecision : public Measure {
public:
    MicroPrecision();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};
 */