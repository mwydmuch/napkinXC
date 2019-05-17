/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cmath>

#include "set_value.h"


double acc(double label, const std::vector<Prediction>& prediction){
    return label == prediction[0].label ? 1.0 : 0.0;
}

double recall(double label, const std::vector<Prediction>& prediction){
    for(const auto& p : prediction)
        if(p.label == label)
            return 1.0;
    return 0.0;
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

SetBasedU::SetBasedU(){
    name = "virtual U";
}

SetBasedU::SetBasedU(Args& args){
    name = "virtual U";
}

U_P::U_P(Args& args){
    name = "uP";
}

double U_P::u(double c, const std::vector<Prediction>& prediction, int k){
    for(const auto& p : prediction)
        if(p.label == c)
            return 1.0 / prediction.size();
    return 0.0;
}

double U_P::g(int pSize, int k){
    return 1.0 / pSize;
}

U_F1::U_F1(Args& args){
    name = "uF1";
}

double U_F1::u(double c, const std::vector<Prediction>& prediction, int k){
    for(const auto& p : prediction)
        if(p.label == c)
            return 2.0 / (1 + prediction.size());
    return 0.0;
}

double U_F1::g(int pSize, int k){
    return 2.0 / (1.0 + pSize);
}

U_Alfa::U_Alfa(Args& args){
    alfa = args.alfa;
    name = "uAlfa(" + std::to_string(alfa) + ")";
}

double U_Alfa::u(double c, const std::vector<Prediction>& prediction, int k) {
    for(const auto& p : prediction)
        if(p.label == c) {
            if (prediction.size() == 1) return 1.0;
            else return 1 - alfa;
        }
    return 0.0;
}

double U_Alfa::g(int pSize, int k){
    if(pSize == 1) return 1.0;
    else return 1.0 - alfa;
}

U_AlfaBeta::U_AlfaBeta(Args& args){
    alfa = args.alfa;
    beta = args.beta;
    name = "uAlfaBeta(" + std::to_string(alfa) + ", " + std::to_string(beta) + ")";
}

double U_AlfaBeta::u(double c, const std::vector<Prediction>& prediction, int k) {
    for(const auto& p : prediction)
        if(p.label == c)
            return 1.0 - alfa * pow(static_cast<double>(prediction.size() - 1) / (k - 1), beta);
    return 0.0;
}

double U_AlfaBeta::g(int pSize, int k){
    return 1.0 - alfa * pow(static_cast<double>(pSize - 1) / (k - 1), beta);
}

std::shared_ptr<SetBasedU> setBasedUFactory(Args& args){
    std::shared_ptr<SetBasedU> u = nullptr;
    switch (args.setBasedUType) {
        case uP :
            u = std::static_pointer_cast<SetBasedU>(std::make_shared<U_P>(args));
            break;
        case uF1 :
            u = std::static_pointer_cast<SetBasedU>(std::make_shared<U_F1>(args));
            break;
        case uAlfaBeta :
            u = std::static_pointer_cast<SetBasedU>(std::make_shared<U_AlfaBeta>(args));
            break;
        default:
            throw "Unknown set based utility type!";
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
