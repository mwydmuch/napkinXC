/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cmath>

#include "set_utility.h"


std::shared_ptr<SetUtility> SetUtility::factory(Args& args, int outputSize) {
    std::shared_ptr<SetUtility> u = nullptr;
    switch (args.setUtilityType) {
    case uP: u = std::static_pointer_cast<SetUtility>(std::make_shared<PrecisionUtility>(args, outputSize)); break;
    case uF1: u = std::static_pointer_cast<SetUtility>(std::make_shared<F1Utility>(args, outputSize)); break;
    case uAlfa: u = std::static_pointer_cast<SetUtility>(std::make_shared<UtilityAlfa>(args, outputSize)); break;
    case uAlfaBeta:
        u = std::static_pointer_cast<SetUtility>(std::make_shared<UtilityAlfaBeta>(args, outputSize));
        break;
    case uDeltaGamma:
        u = std::static_pointer_cast<SetUtility>(std::make_shared<UtilityDeltaGamma>(args, outputSize));
        break;
    default: throw std::invalid_argument("Unknown set based utility type!");
    }

    return u;
}

SetUtility::SetUtility(Args& args, int outputSize) : Measure(args, outputSize) {}

void SetUtility::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += u(labels[0], prediction);
    ++count;
}

PrecisionUtility::PrecisionUtility(Args& args, int outputSize) : SetUtility(args, outputSize) { name = "U_P"; }

double PrecisionUtility::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) return 1.0 / prediction.size();
    return 0.0;
}

double PrecisionUtility::g(int pSize) { return 1.0 / pSize; }

F1Utility::F1Utility(Args& args, int outputSize) : SetUtility(args, outputSize) { name = "U_F1"; }

double F1Utility::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) return 2.0 / (1 + prediction.size());
    return 0.0;
}

double F1Utility::g(int pSize) { return 2.0 / (1.0 + pSize); }

UtilityAlfa::UtilityAlfa(Args& args, int outputSize) : SetUtility(args, outputSize) {
    alfa = args.alfa;
    m = outputSize;
    name = "U_alfa(" + std::to_string(alfa) + ")";
}

double UtilityAlfa::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) {
            if (prediction.size() == 1)
                return 1.0;
            else
                return 1 - alfa;
        }
    return 0.0;
}

double UtilityAlfa::g(int pSize) {
    if (pSize == 1)
        return 1.0;
    else
        return 1.0 - alfa;
}

UtilityAlfaBeta::UtilityAlfaBeta(Args& args, int outputSize) : SetUtility(args, outputSize) {
    alfa = args.alfa;
    beta = args.beta;
    m = outputSize;
    if (alfa <= 0) alfa = static_cast<double>(m - 1) / m;
    if (beta <= 0) beta = std::log2(static_cast<double>(m) / (2 * (m - 1))) / std::log2(1.0 / (m - 1));
    name = "U_alfa_beta(" + std::to_string(alfa) + ", " + std::to_string(beta) + ")";
}

double UtilityAlfaBeta::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) return 1.0 - alfa * pow(static_cast<double>(prediction.size() - 1) / (m - 1), beta);
    return 0.0;
}

double UtilityAlfaBeta::g(int pSize) { return 1.0 - alfa * pow(static_cast<double>(pSize - 1) / (m - 1), beta); }

UtilityDeltaGamma::UtilityDeltaGamma(Args& args, int outputSize) : SetUtility(args, outputSize) {
    delta = args.delta;
    gamma = args.gamma;
    name = "U_delta_gamma(" + std::to_string(delta) + ", " + std::to_string(gamma) + ")";
}

double UtilityDeltaGamma::u(double c, const std::vector<Prediction>& prediction) { return 0.0; }

double UtilityDeltaGamma::g(int pSize) { return 1.0; }
