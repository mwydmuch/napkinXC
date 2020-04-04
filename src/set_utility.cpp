/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include <cmath>

#include "set_utility.h"


std::shared_ptr<SetUtility> SetUtility::factory(Args& args, int outputSize) {
    std::shared_ptr<SetUtility> u = nullptr;
    switch (args.setUtilityType) {
    case uP: u = std::static_pointer_cast<SetUtility>(std::make_shared<PrecisionUtility>()); break;
    case uF1: u = std::static_pointer_cast<SetUtility>(std::make_shared<F1Utility>()); break;
    case uAlpha: u = std::static_pointer_cast<SetUtility>(std::make_shared<UtilityAlpha>(args.alpha, outputSize)); break;
    case uAlphaBeta:
        u = std::static_pointer_cast<SetUtility>(std::make_shared<UtilityAlphaBeta>(args.alpha, args.beta, outputSize));
        break;
    case uDeltaGamma:
        u = std::static_pointer_cast<SetUtility>(std::make_shared<UtilityDeltaGamma>(args.delta, args.gamma));
        break;
    default: throw std::invalid_argument("Unknown set based utility type!");
    }

    return u;
}

SetUtility::SetUtility() {}

void SetUtility::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += u(labels[0], prediction);
    ++count;
}

PrecisionUtility::PrecisionUtility() { name = "U_P"; }

double PrecisionUtility::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) return 1.0 / prediction.size();
    return 0.0;
}

double PrecisionUtility::g(int pSize) { return 1.0 / pSize; }

F1Utility::F1Utility() { name = "U_F1"; }

double F1Utility::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) return 2.0 / (1 + prediction.size());
    return 0.0;
}

double F1Utility::g(int pSize) { return 2.0 / (1.0 + pSize); }

UtilityAlpha::UtilityAlpha(double alpha, int outputSize) : alpha(alpha), m(outputSize) {
    name = "U_alpha(" + std::to_string(alpha) + ")";
}

double UtilityAlpha::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) {
            if (prediction.size() == 1)
                return 1.0;
            else
                return 1 - alpha;
        }
    return 0.0;
}

double UtilityAlpha::g(int pSize) {
    if (pSize == 1)
        return 1.0;
    else
        return 1.0 - alpha;
}

UtilityAlphaBeta::UtilityAlphaBeta(double alpha, double beta, int outputSize) : alpha(alpha), beta(beta), m(outputSize) {
    if (alpha <= 0) this->alpha = static_cast<double>(m - 1) / m;
    name = "U_alpha_beta(" + std::to_string(this->alpha) + ", " + std::to_string(beta) + ")";
}

double UtilityAlphaBeta::u(double c, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction)
        if (p.label == c) return 1.0 - alpha * pow(static_cast<double>(prediction.size() - 1) / (m - 1), beta);
    return 0.0;
}

double UtilityAlphaBeta::g(int pSize) { return 1.0 - alpha * pow(static_cast<double>(pSize - 1) / (m - 1), beta); }

UtilityDeltaGamma::UtilityDeltaGamma(double delta, double gamma) : delta(delta), gamma(gamma) {
    name = "U_delta_gamma(" + std::to_string(delta) + ", " + std::to_string(gamma) + ")";
}

double UtilityDeltaGamma::u(double c, const std::vector<Prediction>& prediction) { return 0.0; }

double UtilityDeltaGamma::g(int pSize) { return 1.0; }
