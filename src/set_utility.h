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

    virtual double u(double c, const std::vector<Prediction>& prediction);
    virtual double g(int pSize) = 0;
};

class PrecisionUtility : public SetUtility {
public:
    PrecisionUtility();
    double g(int pSize) override;
};

class RecallUtility : public SetUtility {
public:
    RecallUtility();
    double g(int pSize) override;
};

class FBetaUtility : public SetUtility {
public:
    FBetaUtility(double beta);
    double g(int pSize) override;

protected:
    double beta;
};

class ExpUtility : public SetUtility {
public:
    ExpUtility(double gamma);
    double g(int pSize) override;

protected:
    double gamma;
};

class LogUtility : public SetUtility {
public:
    LogUtility();
    double g(int pSize) override;
};

class UtilityDeltaGamma : public SetUtility {
public:
    UtilityDeltaGamma(double delta, double gamma);
    double g(int pSize) override;

protected:
    double delta;
    double gamma;
};

class UtilityAlphaBeta : public SetUtility {
public:
    UtilityAlphaBeta(double alpha, double beta, int outputSize);
    double g(int pSize) override;

protected:
    double alpha;
    double beta;
    int m;
};
