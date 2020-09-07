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

#include <cmath>

#include "set_utility.h"


std::shared_ptr<SetUtility> SetUtility::factory(Args& args, int outputSize) {
    std::shared_ptr<SetUtility> u = nullptr;
    switch (args.setUtilityType) {
    case uP: u = std::static_pointer_cast<SetUtility>(std::make_shared<PrecisionUtility>()); break;
    case uR: u = std::static_pointer_cast<SetUtility>(std::make_shared<RecallUtility>()); break;
    case uF1: u = std::static_pointer_cast<SetUtility>(std::make_shared<FBetaUtility>(1)); break;
    case uFBeta: u = std::static_pointer_cast<SetUtility>(std::make_shared<FBetaUtility>(args.beta)); break;
    case uExp: u = std::static_pointer_cast<SetUtility>(std::make_shared<FBetaUtility>(args.beta)); break;
    case uLog: u = std::static_pointer_cast<SetUtility>(std::make_shared<FBetaUtility>(args.beta)); break;
    case uAlpha: u = std::static_pointer_cast<SetUtility>(std::make_shared<UtilityAlphaBeta>(args.alpha, 0, outputSize)); break;
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
    addValue(u(labels[0], prediction));
}

double SetUtility::u(double c, const std::vector<Prediction>& prediction){
    for (const auto& p : prediction)
        if (p.label == c) return g(prediction.size());
    return 0.0;
}

PrecisionUtility::PrecisionUtility() { name = "Precision utility"; }

double PrecisionUtility::g(int pSize) { return 1.0 / pSize; }

RecallUtility::RecallUtility() { name = "Recall utility"; }

double RecallUtility::g(int pSize) { return 1.0; }

FBetaUtility::FBetaUtility(double beta) : beta(beta) {
    if(std::round(beta) == beta)
        name = "F" + std::to_string(static_cast<int>(beta)) + " utility";
    else
        name = "F beta utility (" + std::to_string(beta) + ")";
}

double FBetaUtility::g(int pSize) { return (1.0 + beta * beta) / (pSize + beta * beta); }

ExpUtility::ExpUtility(double gamma) : gamma(gamma) {
    name = "Exp. utility (" + std::to_string(gamma) +")";
}

double ExpUtility::g(int pSize) { return std::log(1 + 1/pSize); }

LogUtility::LogUtility() { name = "Log. utility"; }

double LogUtility::g(int pSize) { return std::log(1 + 1/pSize); }

UtilityAlphaBeta::UtilityAlphaBeta(double alpha, double beta, int outputSize) : alpha(alpha), beta(beta), m(outputSize) {
    if (alpha <= 0) this->alpha = static_cast<double>(m - 1) / m;
    if (beta <= 0) this->beta = std::log(static_cast<double>(m) / 2) / std::log(1.0 / (m - 1)) + 1;
    name = "Alpha beta utility (" + std::to_string(this->alpha) + ", " + std::to_string(beta) + ")";
}

double UtilityAlphaBeta::g(int pSize) { return 1.0 - alpha * pow(static_cast<double>(pSize - 1) / (m - 1), beta); }

UtilityDeltaGamma::UtilityDeltaGamma(double delta, double gamma) : delta(delta), gamma(gamma) {
    name = "Delta gamma utility (" + std::to_string(delta) + ", " + std::to_string(gamma) + ")";
}

double UtilityDeltaGamma::g(int pSize) { return delta / pSize - gamma / (pSize * pSize); }
