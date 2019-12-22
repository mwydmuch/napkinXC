/**
 * Copyright (c) 2019 by Marek Wydmuch
 * All rights reserved.
 */

#include "model.h"
#include "threads.h"
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>

#include "measure.h"
#include "set_utility.h"


std::vector<std::shared_ptr<Measure>> Measure::factory(Args& args, int outputSize) {
    std::vector<std::shared_ptr<Measure>> measures;

    std::vector<std::string> measuresNames = split(toLower(args.measures), ',');
    for (const auto& m : measuresNames) {
        // TODO: Add wrong values handling
        std::vector<std::string> mAt = split(m, '@');
        if (mAt.size() > 1) {
            int k = std::stoi(mAt[1]);
            if (mAt[0] == "p")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<PrecisionAtK>(k)));
            else if (mAt[0] == "r")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<RecallAtK>(k)));
            else if (mAt[0] == "c")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<CoverageAtK>(outputSize, k)));
            else if (mAt[0] == "tp")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<TruePositivesAtK>(k)));
            else
                throw std::invalid_argument("Unknown measure type!");
        } else {
            if (m == "p")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Precision>()));
            else if (m == "r")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Recall>()));
            else if (m == "c")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Coverage>(outputSize)));
            else if (m == "acc")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Accuracy>()));
            else if (m == "s")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<PredictionSize>()));
            else if (m == "hl")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<HammingLoss>()));
            else if (m == "tp")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<TruePositives>()));
            else if (m == "fp")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<FalsePositives>()));
            else if (m == "fn")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<FalseNegatives>()));
            else if (m == "uP")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<PrecisionUtility>()));
            else if (m == "uF1")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<F1Utility>()));
            else if (m == "uAlfa")
                measures.push_back(
                    std::static_pointer_cast<Measure>(std::make_shared<UtilityAlfa>(args.alfa, outputSize)));
            else if (m == "uAlfaBeta")
                measures.push_back(std::static_pointer_cast<Measure>(
                    std::make_shared<UtilityAlfaBeta>(args.alfa, args.beta, outputSize)));
            else if (m == "uDeltaGamma")
                measures.push_back(
                    std::static_pointer_cast<Measure>(std::make_shared<UtilityDeltaGamma>(args.delta, args.gamma)));
            else
                throw std::invalid_argument("Unknown measure type!");
        }
    }

    return measures;
}

Measure::Measure() {
    sum = 0;
    count = 0;
}

void Measure::accumulate(SRMatrix<Label>& labels, std::vector<std::vector<Prediction>>& predictions) {
    assert(predictions.size() == labels.rows());
    for (int i = 0; i < labels.rows(); ++i) accumulate(labels[i], predictions[i]);
}

double Measure::value() { return sum / count; }

MeasureAtK::MeasureAtK(int k) : k(k) {
    if (k < 1) throw std::invalid_argument("K cannot be lower then 1!");
}

TruePositivesAtK::TruePositivesAtK(int k) : MeasureAtK(k) { name = "TP@"; }

void TruePositivesAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += calculate(labels, prediction, k);
    ++count;
}

double TruePositivesAtK::calculate(Label* labels, const std::vector<Prediction>& prediction, int k) {
    double tp = 0;
    for (int i = 0; i < std::min(k, static_cast<int>(prediction.size())); ++i) {
        int l = -1;
        while (labels[++l] > -1)
            if (prediction[i].label == labels[l]) {
                ++tp;
                break;
            }
    }

    return tp;
}

TruePositives::TruePositives() { name = "TP"; }

void TruePositives::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += TruePositives::calculate(labels, prediction);
    ++count;
}

double TruePositives::calculate(Label* labels, const std::vector<Prediction>& prediction) {
    return TruePositivesAtK::calculate(labels, prediction, prediction.size());
}

FalsePositives::FalsePositives() { name = "FP"; }

void FalsePositives::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += FalsePositives::calculate(labels, prediction);
    ++count;
}

double FalsePositives::calculate(Label* labels, const std::vector<Prediction>& prediction) {
    double fp = 0;

    for (const auto& p : prediction) {
        int l = -1;
        bool found = false;
        while (labels[++l] > -1) {
            if (p.label == labels[l]) {
                found = true;
                break;
            }
        }
        if (!found) ++fp;
    }

    return fp;
}

FalseNegatives::FalseNegatives() { name = "FN"; }

void FalseNegatives::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += FalseNegatives::calculate(labels, prediction);
    ++count;
}

double FalseNegatives::calculate(Label* labels, const std::vector<Prediction>& prediction) {
    double fn = 0;

    int l = -1;
    while (labels[++l] > -1) {
        bool found = false;
        for (const auto& p : prediction) {
            if (p.label == labels[l]) {
                found = true;
                break;
            }
        }
        if (!found) ++fn;
    }
    return fn;
}


Recall::Recall() { name = "Recall"; }

void Recall::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    int l = -1;
    while (labels[++l] > -1)
        ;
    if (l > 0) {
        sum += tp / l;
        ++count;
    }
}

RecallAtK::RecallAtK(int k) : MeasureAtK(k) { name = "R@" + std::to_string(k); }

void RecallAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositivesAtK::calculate(labels, prediction, k);
    int l = -1;
    while (labels[++l] > -1)
        ;
    if (l > 0) {
        sum += tp / l;
        ++count;
    }
}

Precision::Precision() { name = "Precision"; }

void Precision::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    if (!prediction.empty()) {
        sum += tp / prediction.size();
        ++count;
    }
}

PrecisionAtK::PrecisionAtK(int k) : MeasureAtK(k) { name = "P@" + std::to_string(k); }

void PrecisionAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += TruePositivesAtK::calculate(labels, prediction, k) / k;
    ++count;
}

Coverage::Coverage(int outputSize) : m(outputSize) { name = "Coverage"; }

void Coverage::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    int l;
    for (const auto& p : prediction) {
        l = -1;
        while (labels[++l] > -1)
            if (p.label == labels[l]) {
                seen.insert(p.label);
                break;
            }
    }
}

double Coverage::value() { return static_cast<double>(seen.size()) / m; }

CoverageAtK::CoverageAtK(int outputSize, int k) : MeasureAtK(k), m(outputSize) { name = "C@" + std::to_string(k); }

void CoverageAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    for (int i = 0; i < std::min(k, static_cast<int>(prediction.size())); ++i) {
        int l = -1;
        while (labels[++l] > -1)
            if (prediction[i].label == labels[l]) {
                seen.insert(prediction[i].label);
                break;
            }
    }
}

double CoverageAtK::value() { return static_cast<double>(seen.size()) / m; }

Accuracy::Accuracy() { name = "Acc"; }

void Accuracy::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    if (!prediction.empty() && labels[0] == prediction[0].label) ++sum;
    ++count;
}

PredictionSize::PredictionSize() { name = "Prediction size"; }

void PredictionSize::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += prediction.size();
    ++count;
}

HammingLoss::HammingLoss() { name = "Hamming loss"; }

void HammingLoss::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    sum += FalsePositives::calculate(labels, prediction) + FalseNegatives::calculate(labels, prediction);
    ++count;
}
