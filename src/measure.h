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

    Measure(Args& args, int outputSize);

    virtual void accumulate(Label* labels, const std::vector<Prediction>& prediction) = 0;
    void accumulate(SRMatrix<Label>& labels, std::vector<std::vector<Prediction>>& predictions);
    virtual double value();

    inline std::string getName(){ return name; }

protected:
    std::string name;
    double sum;
    int count;
};

class MeasureAtK: public Measure {
public:
    MeasureAtK(Args& args, int outputSize, int k);

protected:
    int k;
};

class Recall: public Measure {
public:
    Recall(Args& args, int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class RecallAtK: public MeasureAtK {
public:
    RecallAtK(Args& args, int outputSize, int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class Precision: public Measure {
public:
    Precision(Args& args, int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class PrecisionAtK: public MeasureAtK {
public:
    PrecisionAtK(Args& args, int outputSize, int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class Coverage: public Measure {
public:
    Coverage(Args& args, int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    std::unordered_set<int> seen;
    int m;
};

class CoverageAtK: public MeasureAtK {
public:
    CoverageAtK(Args& args, int outputSize, int k);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
    double value() override;

protected:
    std::unordered_set<int> seen;
    int m;
};

class Accuracy: public Measure {
public:
    Accuracy(Args& args, int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};

class PredictionSize: public Measure{
public:
    PredictionSize(Args& args, int outputSize);

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;
};
