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
