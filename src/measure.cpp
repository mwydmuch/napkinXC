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
            else if (m == "f1")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<F1>()));
            else if (m == "microf")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<MicroF>()));
            else if (m == "macrof")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<MacroF>(outputSize)));
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
            else if (m == "u")
                measures.push_back(std::static_pointer_cast<Measure>(SetUtility::factory(args, outputSize)));
            else
                throw std::invalid_argument("Unknown measure type!");
        }
    }

    return measures;
}

Measure::Measure() {
    sum = 0;
    sumSq = 0;
    count = 0;
}

void Measure::accumulate(SRMatrix<Label>& labels, std::vector<std::vector<Prediction>>& predictions) {
    assert(predictions.size() == labels.rows());
    for (int i = 0; i < labels.rows(); ++i) accumulate(labels[i], predictions[i]);
}

double Measure::value() { return sum / count; }

double Measure::stdDev() {
    double m = mean();
    return sumSq / count - m * m;
}

void Measure::addValue(double value){
    sum += value;
    sumSq += value * value;
    ++count;
}

MeasureAtK::MeasureAtK(int k) : k(k) {
    if (k < 1) throw std::invalid_argument("K cannot be lower then 1!");
}

TruePositivesAtK::TruePositivesAtK(int k) : MeasureAtK(k) {
    name = "TP@";
    meanMeasure = true;
}

void TruePositivesAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(calculate(labels, prediction, k));
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

TruePositives::TruePositives() {
    name = "TP";
    meanMeasure = true;
}

void TruePositives::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(TruePositives::calculate(labels, prediction));
}

double TruePositives::calculate(Label* labels, const std::vector<Prediction>& prediction) {
    return TruePositivesAtK::calculate(labels, prediction, prediction.size());
}

FalsePositives::FalsePositives() {
    name = "FP";
    meanMeasure = true;
}

void FalsePositives::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(FalsePositives::calculate(labels, prediction));
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

FalseNegatives::FalseNegatives() {
    name = "FN";
    meanMeasure = true;
}

void FalseNegatives::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(FalseNegatives::calculate(labels, prediction));
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


Recall::Recall() {
    name = "Recall";
    meanMeasure = true;
}

void Recall::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    int l = -1;
    while (labels[++l] > -1)
        ;
    if (l > 0)
        addValue(tp / l);
}

RecallAtK::RecallAtK(int k) : MeasureAtK(k) {
    name = "R@" + std::to_string(k);
    meanMeasure = true;
}

void RecallAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositivesAtK::calculate(labels, prediction, k);
    int l = -1;
    while (labels[++l] > -1)
        ;
    if (l > 0)
        addValue(tp / l);
}

Precision::Precision() {
    name = "Precision";
    meanMeasure = true;
}

void Precision::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    if (!prediction.empty()) addValue(tp / prediction.size());
}

PrecisionAtK::PrecisionAtK(int k) : MeasureAtK(k) {
    name = "P@" + std::to_string(k);
    meanMeasure = true;
}

void PrecisionAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(TruePositivesAtK::calculate(labels, prediction, k) / k);
}

Coverage::Coverage(int outputSize) : m(outputSize) {
    name = "Coverage";
    meanMeasure = false;
}

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

CoverageAtK::CoverageAtK(int outputSize, int k) : MeasureAtK(k), m(outputSize) {
    name = "C@" + std::to_string(k);
    meanMeasure = false;
}

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

Accuracy::Accuracy() {
    name = "Acc";
    meanMeasure = true;
}

void Accuracy::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    if (!prediction.empty() && labels[0] == prediction[0].label) addValue(1);
    else addValue(0);
}

PredictionSize::PredictionSize() {
    name = "Prediction size";
    meanMeasure = true;
}

void PredictionSize::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(prediction.size());
}

HammingLoss::HammingLoss() {
    name = "Hamming loss";
    meanMeasure = true;
}

void HammingLoss::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(FalsePositives::calculate(labels, prediction) + FalseNegatives::calculate(labels, prediction));
}

F1::F1() {
    name = "F1";
    meanMeasure = true;
}

void F1::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    int l = -1;
    while (labels[++l] > -1)
        ;

    if (!prediction.empty() && l > 0) {
        double p = tp / prediction.size();
        double r = tp / l;
        if(p > 0 && r > 0) addValue(2 * p * r / (p + r));
    }
}

MicroF::MicroF() {
    name = "MicroF";
    meanMeasure = false;
}

void MicroF::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    sum += 2 * tp;
    count += 2 * tp + FalsePositives::calculate(labels, prediction) + FalseNegatives::calculate(labels, prediction);
}

MacroF::MacroF(int outputSize) : m(outputSize) {
    name = "MacroF";
    meanMeasure = false;
    labelsTP.resize(m, 0);
    labelsFP.resize(m, 0);
    labelsFN.resize(m, 0);
}

void MacroF::accumulate(Label* labels, const std::vector<Prediction>& prediction){

    for (const auto& p : prediction) {
        int l = -1;
        bool found = false;
        while (labels[++l] > -1) {
            if (p.label == labels[l]) {
                found = true;
                ++labelsTP[p.label];
                break;
            }
        }
        if (!found) ++labelsFP[p.label];
    }

    int l = -1;
    while (labels[++l] > -1) {
        bool found = false;
        for (const auto& p : prediction) {
            if (p.label == labels[l]) {
                found = true;
                break;
            }
        }
        if (!found) ++labelsFN[labels[l]];
    }
}

double MacroF::value(){
    // TODO jaka jest wartosc miary f jesli nie ma zadnych pozytywow potrzebnych? 0 czy 1?
    double sum = 0;
    for(int i = 0; i < m; ++i){
        double denominator = 2 * labelsTP[i] + labelsFP[i] + labelsFN[i];
        denominator = (denominator > 0) ? 2 * labelsTP[i] / denominator : 1.0;
        //sum += (denominator > 0) ? 2 * labelsTP[i] / denominator : 1.0;
        sum += denominator;
        //std::cout << "Label: " << i << ", F1: " << denominator << "\n";
    }
    return sum / m;
}
