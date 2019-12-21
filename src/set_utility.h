/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "args.h"
#include "measure.h"


class SetUtility : public Measure {
public:
    static std::shared_ptr<SetUtility> factory(Args& args, int outputSize);

    SetUtility();

    void accumulate(Label* labels, const std::vector<Prediction>& prediction) override;

    virtual double u(double c, const std::vector<Prediction>& prediction) = 0;
    virtual double g(int pSize) = 0;
};

class PrecisionUtility : public SetUtility {
public:
    PrecisionUtility();

    double u(double c, const std::vector<Prediction>& prediction) override;
    double g(int pSize) override;
};

class F1Utility : public SetUtility {
public:
    F1Utility();

    double u(double c, const std::vector<Prediction>& prediction) override;
    double g(int pSize) override;
};

class UtilityAlfa : public SetUtility {
public:
    UtilityAlfa(double alfa, int outputSize);

    double u(double c, const std::vector<Prediction>& prediction) override;
    double g(int pSize) override;

protected:
    double alfa;
    int m;
};

class UtilityAlfaBeta : public SetUtility {
public:
    UtilityAlfaBeta(double alfa, double beta, int outputSize);

    double u(double c, const std::vector<Prediction>& prediction) override;
    double g(int pSize) override;

protected:
    double alfa;
    double beta;
    int m;
};

class UtilityDeltaGamma : public SetUtility {
public:
    UtilityDeltaGamma(double delta, double gamma);

    double u(double c, const std::vector<Prediction>& prediction) override;
    double g(int pSize) override;

protected:
    double delta;
    double gamma;
};
