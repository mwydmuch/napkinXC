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
            if (mAt[0] == "p" || mAt[0] == "precision")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<PrecisionAtK>(k)));
            else if (mAt[0] == "r" || mAt[0] == "recall")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<RecallAtK>(k)));
            else if (mAt[0] == "dcg")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<DCGAtK>(k)));
            else if (mAt[0] == "ndcg")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<NDCGAtK>(k)));
            else if (mAt[0] == "c" || mAt[0] == "coverage")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<CoverageAtK>(outputSize, k)));
            else if (mAt[0] == "tp")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<TruePositivesAtK>(k)));
            else
                throw std::invalid_argument("Unknown measure type: " + mAt[0]);
        } else {
            if (m == "p" || m =="precision")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Precision>()));
            else if (m == "r" || m =="recall")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Recall>()));
            else if (m == "samplef1")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<SampleF1>()));
            else if (m == "microf1")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<MicroF1>()));
            else if (m == "macrof1")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<MacroF1>(outputSize)));
            else if (m == "c" || m == "coverage")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Coverage>(outputSize)));
            else if (m == "acc" || m == "accuracy")
                measures.push_back(std::static_pointer_cast<Measure>(std::make_shared<Accuracy>()));
            else if (m == "s" || m == "size")
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
                throw std::invalid_argument("Unknown measure type: " + m + "!");
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

DCGAtK::DCGAtK(int k) : MeasureAtK(k) {
    name = "DCG@" + std::to_string(k);
    meanMeasure = true;
}

void DCGAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    addValue(calculate(labels, prediction, k));
}

double DCGAtK::calculate(Label* labels, const std::vector<Prediction>& prediction, int k){
    double score = 0;
    for (int i = 0; i < std::min(k, static_cast<int>(prediction.size())); ++i) {
        int l = -1;
        while (labels[++l] > -1)
            if (prediction[i].label == labels[l]) {
                score += 1.0 / std::log2(i + 2);
                break;
            }
    }

    return score;
}


NDCGAtK::NDCGAtK(int k) : MeasureAtK(k) {
    name = "nDCG@" + std::to_string(k);
    meanMeasure = true;
}

void NDCGAtK::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double nDenominator = 0;

    int l = 0;
    while (labels[l] > -1 && l < k) {
        nDenominator += 1.0 / std::log2(l + 2);
        ++l;
    }

    if (l > 0) addValue(DCGAtK::calculate(labels, prediction, k) / nDenominator);
    else addValue(0);

}

Coverage::Coverage(int outputSize) : m(outputSize) {
    name = "Coverage";
    meanMeasure = false;
}

void Coverage::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    for (const auto& p : prediction) {
        int l = -1;
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

SampleF1::SampleF1() {
    name = "Sample-F1";
    meanMeasure = true;
}

void SampleF1::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
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

MicroF1::MicroF1() {
    name = "Micro-F1";
    meanMeasure = false;
}

void MicroF1::accumulate(Label* labels, const std::vector<Prediction>& prediction) {
    double tp = TruePositives::calculate(labels, prediction);
    sum += 2 * tp;
    count += 2 * tp + FalsePositives::calculate(labels, prediction) + FalseNegatives::calculate(labels, prediction);
}

MacroF1::MacroF1(int outputSize) : m(outputSize), zeroDivisionDenominator(1) {
    name = "Macro-F1";
    meanMeasure = false;
    labelsTP.resize(m, 0);
    labelsFP.resize(m, 0);
    labelsFN.resize(m, 0);
}

void MacroF1::accumulate(Label* labels, const std::vector<Prediction>& prediction){

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

double MacroF1::value(){
    double sum = 0;
    for(int i = 0; i < m; ++i){
        double denominator = 2 * labelsTP[i] + labelsFP[i] + labelsFN[i];
        denominator = (denominator > 0) ? 2 * labelsTP[i] / denominator : zeroDivisionDenominator;
        sum += denominator;
    }
    return sum / m;
}
