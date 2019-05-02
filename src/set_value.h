/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#pragma once

#include "args.h"
#include "model.h"


double acc(double label, const std::vector<Prediction>& prediction);

double u_P(double label, const std::vector<Prediction>& prediction);

double g_P(double pSize);

double u_F1(double label, const std::vector<Prediction>& prediction);

double g_F1(int pSize);

double u_alfa(double label, const std::vector<Prediction>& prediction, double alfa);

double g_alfa(int pSize, double alfa);

double u_delta_gamma(double label, const std::vector<Prediction>& prediction, double delta, double gamma);

double g_delta_gamma(int pSize, double delta, double gamma);

double u_alfa_beta(double label, const std::vector<Prediction>& prediction, double alfa, double beta, int K);

double g_alfa_beta(int pSize, double alfa, double beta, int K);

class SetBasedU {
public:
    virtual double u(double c, const std::vector<Prediction>& prediction, int k) = 0;
    virtual double g(int pSize, int k) = 0;

    bool checkstop(int pSize, int k);
};

class U_P: public SetBasedU{
    double u(double c, const std::vector<Prediction>& prediction, int k) override;
    double g(int pSize, int k) override;
};

class U_F1: public SetBasedU{
    double u(double c, const std::vector<Prediction>& prediction, int k) override;
    double g(int pSize, int k) override;
};

class U_AlfaBeta: public SetBasedU{
    double u(double c, const std::vector<Prediction>& prediction, int k) override;
    double g(int pSize, int k) override;
};


std::shared_ptr<SetBasedU> setBasedUFactory(Args& args);


