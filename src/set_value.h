/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "args.h"
#include "model.h"


double acc(double label, const std::vector<Prediction>& prediction);

double recall(double label, const std::vector<Prediction>& prediction);

double u_delta_gamma(double label, const std::vector<Prediction>& prediction, double delta, double gamma);

double g_delta_gamma(int pSize, double delta, double gamma);

class SetBasedU {
public:
    SetBasedU();
    SetBasedU(Args& args);

    virtual double u(double c, const std::vector<Prediction>& prediction, int k) = 0;
    virtual double g(int pSize, int k) = 0;

    bool checkstop(int pSize, int k);
    inline std::string getName(){ return name; }

protected:
    std::string name;
};

class U_P: public SetBasedU{
public:
    U_P(Args& args);

    double u(double c, const std::vector<Prediction>& prediction, int k) override;
    double g(int pSize, int k) override;
};

class U_F1: public SetBasedU{
public:
    U_F1(Args& args);

    double u(double c, const std::vector<Prediction>& prediction, int k) override;
    double g(int pSize, int k) override;
};

class U_Alfa: public SetBasedU{
public:
    U_Alfa(Args& args);

    double u(double c, const std::vector<Prediction>& prediction, int k) override;
    double g(int pSize, int k) override;

protected:
    double alfa;
};

class U_AlfaBeta: public SetBasedU{
public:
    U_AlfaBeta(Args& args);

    double u(double c, const std::vector<Prediction>& prediction, int k) override;
    double g(int pSize, int k) override;

protected:
    double alfa;
    double beta;
};

std::shared_ptr<SetBasedU> setBasedUFactory(Args& args);
