/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cmath>

#include "set_value.h"


double acc(double label, const std::vector<Prediction>& prediction){
    return label == prediction[0].label ? 1.0 : 0.0;
}

double u_P(double label, const std::vector<Prediction>& prediction){
    for(const auto& p : prediction)
        if(p.label == label)
            return 1.0 / prediction.size();
    return 0.0;
}

double g_P(double pSize){
    return 1.0 / pSize;
}

double u_F1(double label, const std::vector<Prediction>& prediction){
    for(const auto& p : prediction)
        if(p.label == label)
            return 2.0 / (1 + prediction.size());
    return 0.0;
}

double g_F1(int pSize){
    return 2.0 / (1.0 + pSize);
}

double u_alfa(double label, const std::vector<Prediction>& prediction, double alfa){
    for(const auto& p : prediction)
        if(p.label == label) {
            if (prediction.size() == 1) return 1.0;
            else return 1 - alfa;
        }
    return 0.0;
}

double g_alfa(int pSize, double alfa){
    if(pSize == 1) return 1.0;
    else return 1.0 - alfa;
}

double u_delta_gamma(double label, const std::vector<Prediction>& prediction, double delta, double gamma){
    for(const auto& p : prediction)
        if(p.label == label)
            return delta / prediction.size() - gamma / (prediction.size() * prediction.size());
    return 0.0;
}

double g_delta_gamma(double pSize, double delta, double gamma) {
    return delta / pSize - gamma / (pSize * pSize);
}

double u_alfa_beta(double label, const std::vector<Prediction>& prediction, double alfa, double beta, int K){
    for(const auto& p : prediction)
        if(p.label == label)
            return 1.0 - alfa * pow(static_cast<double>(prediction.size() - 1) / (K - 1), beta);
    return 0.0;
}

double g_alfa_beta(int pSize, double alfa, double beta, int K){
    return 1.0 - alfa * pow(static_cast<double>(pSize - 1) / (K - 1), beta);
}


double U_P::u(double c, const std::vector<Prediction>& prediction, int k){
    return u_P(c, prediction);
}

double U_P::g(int pSize, int k){
    return g_P(pSize);
}

double U_F1::u(double c, const std::vector<Prediction>& prediction, int k){
    return u_F1(c, prediction);
}

double U_F1::g(int pSize, int k){
    return g_F1(pSize);
}

double U_AlfaBeta::u(double c, const std::vector<Prediction>& prediction, int k) {
    return u_alfa_beta(c, prediction, 1.0, 1.0, k);
}

double U_AlfaBeta::g(int pSize, int k){
    return g_alfa_beta(pSize, 1.0, 1.0, k);
}


std::shared_ptr<SetBasedU> setBasedUFactory(Args& args){
    std::shared_ptr<SetBasedU> u = nullptr;
    switch (args.setBasedUType) {
        case uP :
            u = std::static_pointer_cast<SetBasedU>(std::make_shared<U_P>());
            break;
        case uF1 :
            u = std::static_pointer_cast<SetBasedU>(std::make_shared<U_F1>());
            break;
        case uAlfaBeta :
            u = std::static_pointer_cast<SetBasedU>(std::make_shared<U_AlfaBeta>());
            break;
        default:
            throw "Unknown set based utiltiy type!";
    }

    return u;
}

bool SetBasedU::checkstop(int pSize, int k){
    double l = g(pSize, k) / (g(pSize, k) - g(pSize + 1, k));
    double r = g(pSize + 2, k) / (g(pSize + 1, k) - g(pSize + 2, k));

    //std::cerr << g(pSize, k) << " " << g(pSize + 1, k) << " " << g(pSize + 2, k) << "\n";

    //std::cerr << l << " " << r << "\n";

    return l >= r;
}
